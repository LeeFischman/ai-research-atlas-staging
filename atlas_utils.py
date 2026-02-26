# atlas_utils.py
# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities for the AI Research Atlas pipeline.
#
# Imported by update_map_v2.py.  update_map.py is deliberately left unchanged
# so it continues to work as a standalone fallback / reference implementation.
#
# Contents
# --------
#   §1  Text helpers          scrub_model_words, _strip_urls
#   §2  Docs cleanup          clear_docs_contents
#   §3  Prominence scoring    load_author_cache, save_author_cache,
#                             fetch_author_hindices, calculate_prominence
#   §3b Semantic Scholar      load_ss_cache, save_ss_cache,
#                             fetch_semantic_scholar_data, _arxiv_id_base
#   §4  Rolling database      load_existing_db, merge_papers
#   §5  arXiv fetch           fetch_arxiv  (exponential back-off)
#   §6  Embedding / UMAP      embed_and_project, _embed_only,
#                             compute_hybrid_distances, embed_and_project_hybrid
#   §7  Atlas build + deploy  build_and_deploy_atlas
#   §8  HTML panels           build_panel_html
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH         = "database.parquet"
STOP_WORDS_PATH = "stop_words.csv"
RETENTION_DAYS  = 14      # papers older than this are pruned each run
ARXIV_MAX       = 400    # max papers fetched per arXiv query

# arXiv retry policy
BASE_WAIT   = 15
MAX_WAIT    = 480
MAX_RETRIES = 7


# ══════════════════════════════════════════════════════════════════════════════
# §1  TEXT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def scrub_model_words(text: str) -> str:
    """Remove inflections of 'model' to reduce TF-IDF noise."""
    pattern = re.compile(r'model(?:s|ing|ed|er|ers)?\b', re.IGNORECASE)
    return " ".join(pattern.sub("", text).split())


def _strip_urls(text: str) -> str:
    """Remove URLs and arXiv-style citation keys from abstract text.

    Targets:
      - http/https URLs (e.g. https://github.com/user/repo)
      - Bare github.com / huggingface.co / arxiv.org references
      - Citation-key tokens containing digits (e.g. cho2026_tokenizer,
        agentlab_main, youtu_cy06ljee1jq) — identifiers, not natural language
    """
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b(?:github|huggingface|arxiv)\.(?:com|org|io)\S*', '',
                  text, flags=re.IGNORECASE)
    # Remove citation-key tokens: word containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    return " ".join(text.split())


# ══════════════════════════════════════════════════════════════════════════════
# §2  DOCS CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def clear_docs_contents(target_dir: str) -> None:
    """Delete every file/folder inside target_dir (but keep the dir itself)."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"  Skipped {file_path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# §3  PROMINENCE SCORING  (h-index based)
#
# Formula
# ───────
#   h_score = log1p(max_h)    * W_MAX
#           + log1p(median_h) * W_MEDIAN
#
#   max_h    : highest h-index across all authors
#   median_h : median h-index across all authors
#   log1p    : compresses the long right tail of h-index distributions
#              (log1p(x) = log(1+x), so h=0→0, h=9≈2.3, h=29≈3.4, h=79≈4.4)
#
# Four tiers (h_score thresholds are first-guess constants — tune after seeing
# real distributions in run_state.json):
#   h_score ≥ TIER_ELITE     → "Elite"
#   h_score ≥ TIER_ENHANCED  → "Enhanced"
#   h_score ≥ TIER_EMERGING  → "Emerging"
#   h_score <  TIER_EMERGING → "Unverified"
#
# Author h-index cache
#   File  : author_cache.json  (alongside database.parquet)
#   Key   : normalised author name (stripped, lowercased)
#   Value : {"hindex": int, "fetched_at": ISO timestamp}
#   TTL   : AUTHOR_CACHE_TTL_DAYS (30 days — h-index changes slowly)
#   Scope : ALL authors per paper
# ══════════════════════════════════════════════════════════════════════════════

AUTHOR_CACHE_PATH     = "author_cache.json"
AUTHOR_CACHE_TTL_DAYS = 30

# ── Formula weights ───────────────────────────────────────────────────────────
# Both start at 1.0. max_h naturally exceeds median_h so no artificial
# multiplier is needed. Tune after inspecting real tier distributions.
W_MAX    = 1.0
W_MEDIAN = 1.0

# ── Tier thresholds ───────────────────────────────────────────────────────────
# Representative h_score values with W_MAX=W_MEDIAN=1.0:
#   Unknown solo author          max=0,  median=0  → 0.0
#   Grad students only           max=2,  median=1  → 1.8
#   One mid-career researcher    max=10, median=3  → 3.8
#   Solid team                   max=20, median=10 → 5.4
#   Strong PI + good team        max=40, median=15 → 6.5
#   Elite collaboration          max=60, median=25 → 7.6
TIER_ELITE    = 7.0
TIER_ENHANCED = 5.0
TIER_EMERGING = 3.0
# below TIER_EMERGING            → "Unverified"


def categorize_authors(n: int) -> str:
    if n <= 3:  return "1-3 Authors"
    if n <= 7:  return "4-7 Authors"
    return "8+ Authors"


def load_author_cache() -> dict:
    """Load the author h-index cache from disk; return empty dict if missing."""
    if os.path.exists(AUTHOR_CACHE_PATH):
        with open(AUTHOR_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_author_cache(cache: dict) -> None:
    """Persist the author h-index cache to disk."""
    with open(AUTHOR_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_author_hindices(author_names: list, cache: dict) -> list:
    """Return a list of h-indices for ALL authors, using OpenAlex API.

    Checks the cache first; fetches from OpenAlex for missing/stale entries.
    Stale = fetched_at older than AUTHOR_CACHE_TTL_DAYS.
    Sleeps 0.12 s between uncached requests to stay under OpenAlex's 10 req/s.

    Parameters
    ----------
    author_names : list of str — raw author names from arXiv
    cache        : the shared in-memory cache dict (mutated in-place)

    Returns
    -------
    list of int — one h-index per author (0 for failed/unknown lookups),
                  in the same order as author_names
    """
    import urllib.parse
    import urllib.request as _req

    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=AUTHOR_CACHE_TTL_DAYS)
    ).isoformat()

    hindices = []

    for name in author_names:
        name = name.strip()
        if not name:
            hindices.append(0)
            continue
        key = name.lower()

        # Cache hit?
        entry = cache.get(key)
        if entry and entry.get("fetched_at", "") > cutoff_ts:
            hindices.append(entry.get("hindex", 0))
            continue

        # Fetch from OpenAlex
        try:
            q   = urllib.parse.quote(name)
            url = f"https://api.openalex.org/authors?search={q}&per_page=5"
            req = _req.Request(
                url,
                headers={"User-Agent":
                    "ai-research-atlas/2.0 "
                    "(https://github.com/LeeFischman/ai-research-atlas; "
                    "mailto:lee.fischman@gmail.com)"},
            )
            with _req.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            results = data.get("results", [])
            hindex  = 0
            if results:
                hindex = results[0].get("summary_stats", {}).get("h_index", 0) or 0

            cache[key] = {
                "hindex":     hindex,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            hindices.append(hindex)
            time.sleep(0.12)   # ~8 req/s

        except Exception as e:
            print(f"  OpenAlex lookup failed for '{name}': {e}")
            hindices.append(0)

    return hindices


def _safe_hindices(row) -> list:
    """Extract author_hindices from a row as a clean list of ints.

    Handles three cases:
      - author_hindices column present and is a list  → use it directly
      - only max_author_hindex present (old DB rows)  → treat as [max_author_hindex]
      - neither present / NaN                         → return [0]
    """
    val = row.get("author_hindices", None)
    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
        return [int(h) if (h is not None and not (isinstance(h, float) and np.isnan(h)))
                else 0 for h in val]

    # Backward compat: old rows have max_author_hindex (int) but no list
    old = row.get("max_author_hindex", None)
    if old is not None and not (isinstance(old, float) and np.isnan(old)):
        return [int(old)]

    return [0]


def calculate_prominence(row) -> str:
    """Compute prominence tier from author h-indices.

    Reads row["author_hindices"] (list of ints, one per author).
    Falls back to row["max_author_hindex"] for rows written by older pipeline.

    Formula
    -------
    h_score = log1p(max_h) * W_MAX + log1p(median_h) * W_MEDIAN

    Returns one of: "Elite" / "Enhanced" / "Emerging" / "Unverified"
    """
    import math
    hindices = _safe_hindices(row)

    max_h    = max(hindices) if hindices else 0
    median_h = float(np.median(hindices)) if hindices else 0.0

    h_score = math.log1p(max_h) * W_MAX + math.log1p(median_h) * W_MEDIAN

    if h_score >= TIER_ELITE:
        return "Elite"
    elif h_score >= TIER_ENHANCED:
        return "Enhanced"
    elif h_score >= TIER_EMERGING:
        return "Emerging"
    else:
        return "Unverified"


# Alias so callers using the old name continue to work during transition
calculate_reputation = calculate_prominence


# ══════════════════════════════════════════════════════════════════════════════
# §3b  SEMANTIC SCHOLAR ENRICHMENT
#
# For each arXiv paper, fetches from the Semantic Scholar API:
#   citation_count             — total citations received
#   influential_citation_count — citations that "heavily build on" this paper
#                                (S2's own classifier; per-citation signal)
#   citation_velocity          — approx citations in the last 3 months
#   tldr                       — S2's one-sentence AI-generated summary
#
# Implementation notes
# ────────────────────
# • Uses the batch endpoint (POST /graph/v1/paper/batch) — up to 500 IDs per
#   request, so even 400 papers is a single HTTP call.
# • arXiv IDs from the parquet (e.g. "2501.12345v1") are stripped of their
#   version suffix before being sent as "arXiv:2501.12345".
# • Cache key  : base arXiv ID (version-stripped, e.g. "2501.12345")
# • Cache value: {citation_count, influential_citation_count, citation_velocity,
#                 tldr, fetched_at}
# • Cache TTL  : SS_CACHE_TTL_DAYS (7 days — citations move faster than h-index)
# • Papers not yet indexed in S2 (brand-new, or outside its corpus) are stored
#   with all-zero counts and empty TLDR; they are re-fetched after the TTL expires.
# • Optional env var SEMANTIC_SCHOLAR_API_KEY unlocks higher rate limits.
#   Without it the free tier allows ~1 req/s (plenty for a daily batch job).
# • "Highly influential" in S2 is a per-citation signal, not a per-paper boolean.
#   The paper-level equivalent is influential_citation_count, stored here.
# ══════════════════════════════════════════════════════════════════════════════

SS_CACHE_PATH     = "ss_cache.json"
SS_CACHE_TTL_DAYS = 7     # re-fetch after 7 days so citation counts stay fresh

_SS_BATCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=citationCount,influentialCitationCount,citationVelocity,tldr"
)


def load_ss_cache() -> dict:
    """Load the Semantic Scholar cache from disk; return empty dict if missing."""
    if os.path.exists(SS_CACHE_PATH):
        with open(SS_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_ss_cache(cache: dict) -> None:
    """Persist the Semantic Scholar cache to disk."""
    with open(SS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _arxiv_id_base(arxiv_id: str) -> str:
    """Strip version suffix from an arXiv entry ID.

    '2501.12345v1' → '2501.12345'
    '2501.12345'   → '2501.12345'
    """
    return re.sub(r'v\d+$', '', arxiv_id.strip())


def fetch_semantic_scholar_data(arxiv_ids: list, cache: dict) -> dict:
    """Fetch Semantic Scholar metadata for a list of arXiv IDs.

    Checks the cache first; sends a batch request for uncached / stale IDs.
    Mutates `cache` in-place with new entries (caller is responsible for saving).

    Parameters
    ----------
    arxiv_ids : list of str — raw arXiv entry IDs (may include version suffix)
    cache     : the shared in-memory cache dict (mutated in-place)

    Returns
    -------
    dict mapping base_arxiv_id → {
        "citation_count":             int,
        "influential_citation_count": int,
        "citation_velocity":          int,   # approx citations in last 3 months
        "tldr":                       str,   # AI-generated one-liner, or ""
        "fetched_at":                 str,   # ISO timestamp
    }
    All values default to 0 / "" for papers not found in S2.
    """
    import urllib.request as _req

    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=SS_CACHE_TTL_DAYS)
    ).isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()

    results   = {}
    to_fetch  = []

    for arxiv_id in arxiv_ids:
        base  = _arxiv_id_base(arxiv_id)
        entry = cache.get(base)
        if entry and entry.get("fetched_at", "") > cutoff_ts:
            results[base] = entry
        else:
            to_fetch.append(base)

    if not to_fetch:
        print(f"  All {len(arxiv_ids)} papers found in S2 cache.")
        return results

    # De-duplicate while preserving order
    seen      = set()
    to_fetch  = [b for b in to_fetch if not (b in seen or seen.add(b))]
    cached_n  = len(arxiv_ids) - len(to_fetch)
    print(f"  Fetching {len(to_fetch)} papers from Semantic Scholar "
          f"({cached_n} cache hits)...")

    ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    base_headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "ai-research-atlas/2.0 "
            "(https://github.com/LeeFischman/ai-research-atlas; "
            "mailto:lee.fischman@gmail.com)"
        ),
    }
    if ss_api_key:
        base_headers["x-api-key"] = ss_api_key
    # Rate limit: ~1 req/s free tier, ~3 req/s with key
    inter_chunk_sleep = 0.4 if ss_api_key else 1.2

    BATCH_SIZE = 500
    for chunk_i, chunk_start in enumerate(range(0, len(to_fetch), BATCH_SIZE)):
        chunk  = to_fetch[chunk_start:chunk_start + BATCH_SIZE]
        ss_ids = [f"arXiv:{b}" for b in chunk]

        payload = json.dumps({"ids": ss_ids}).encode("utf-8")
        req     = _req.Request(
            _SS_BATCH_URL, data=payload, headers=base_headers, method="POST"
        )

        try:
            with _req.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            found_n = null_n = 0
            for base, paper in zip(chunk, data):
                if paper is None:
                    # Not yet indexed in S2 (brand-new paper, or outside corpus)
                    entry = {
                        "citation_count":             0,
                        "influential_citation_count": 0,
                        "citation_velocity":          0,
                        "tldr":                       "",
                        "fetched_at":                 now_iso,
                    }
                    null_n += 1
                else:
                    tldr_text = ""
                    if isinstance(paper.get("tldr"), dict):
                        tldr_text = paper["tldr"].get("text", "") or ""
                    entry = {
                        "citation_count":             int(paper.get("citationCount")            or 0),
                        "influential_citation_count": int(paper.get("influentialCitationCount") or 0),
                        "citation_velocity":          int(paper.get("citationVelocity")         or 0),
                        "tldr":                       tldr_text,
                        "fetched_at":                 now_iso,
                    }
                    found_n += 1
                cache[base]   = entry
                results[base] = entry

            print(f"  S2 chunk {chunk_i + 1}: "
                  f"{found_n} found, {null_n} not indexed.")

        except Exception as e:
            print(f"  S2 batch fetch failed (chunk {chunk_i + 1}): {e}")
            # Store default zeros for the failed chunk so pipeline continues
            for base in chunk:
                if base not in results:
                    entry = {
                        "citation_count":             0,
                        "influential_citation_count": 0,
                        "citation_velocity":          0,
                        "tldr":                       "",
                        "fetched_at":                 now_iso,
                    }
                    cache[base]   = entry
                    results[base] = entry

        if chunk_start + BATCH_SIZE < len(to_fetch):
            time.sleep(inter_chunk_sleep)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# §4  ROLLING DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_db(db_path: str = DB_PATH, bypass_pruning: bool = False) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    df = pd.read_parquet(db_path)
    if bypass_pruning:
        print(f"  Loaded {len(df)} papers (retention pruning skipped — offline mode).")
        return df
    if "date_added" not in df.columns:
        print("  Existing DB has no date_added column — starting fresh.")
        return pd.DataFrame()
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    df["date_added"] = pd.to_datetime(df["date_added"], utc=True)
    before = len(df)
    df = df[df["date_added"] >= cutoff].reset_index(drop=True)
    pruned = before - len(df)
    if pruned:
        print(f"  Pruned {pruned} papers older than {RETENTION_DAYS} days.")
    return df


def merge_papers(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new papers into existing; duplicate arXiv IDs: new overwrites old."""
    if existing.empty:
        return new
    kept = existing[~existing["id"].isin(new["id"])]
    overwritten = len(existing) - len(kept)
    if overwritten:
        print(f"  Overwrote {overwritten} updated paper(s).")
    return pd.concat([kept, new], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# §5  arXiv FETCH WITH EXPONENTIAL BACK-OFF
# ══════════════════════════════════════════════════════════════════════════════

def fetch_arxiv(client, search) -> list:
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return list(client.results(search))
        except Exception as e:
            last_exc = e
            err_str  = str(e).lower()
            is_429   = (
                "429" in err_str
                or "too many requests" in err_str
                or (isinstance(e, urllib.error.HTTPError) and e.code == 429)
            )
            wait       = min(BASE_WAIT * (2 ** attempt), MAX_WAIT)
            total_wait = wait + random.uniform(0, wait * 0.25)
            label      = "Rate limited (429)" if is_429 else f"Error: {e}"
            print(f"  {label} — attempt {attempt + 1}/{MAX_RETRIES}. "
                  f"Retrying in {total_wait:.0f}s...")
            time.sleep(total_wait)
    raise RuntimeError(
        f"arXiv fetch failed after {MAX_RETRIES} attempts. Last: {last_exc}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# §6  EMBEDDING / UMAP
# ══════════════════════════════════════════════════════════════════════════════

def embed_and_project(df: pd.DataFrame, model_name: str = "specter2") -> pd.DataFrame:
    """Embed new papers and (re-)project ALL papers to 50D + 2D with UMAP.

    Model name controls which column set is written:
      "specter2" → embedding, embedding_50d, projection_x, projection_y
      "sbert"    → embedding_sbert, embedding_sbert_50d,
                   projection_sbert_x, projection_sbert_y

    All column sets coexist in the parquet — switching is a one-line env change.
    """
    from sentence_transformers import SentenceTransformer
    import umap as umap_lib

    if model_name == "specter2":
        hf_model_id  = "allenai/specter2_base"
        col_embed    = "embedding"
        col_embed_50d = "embedding_50d"
        col_proj_x   = "projection_x"
        col_proj_y   = "projection_y"
        text_col     = "text"
    elif model_name == "sbert":
        hf_model_id  = "sentence-transformers/all-mpnet-base-v2"
        col_embed    = "embedding_sbert"
        col_embed_50d = "embedding_sbert_50d"
        col_proj_x   = "projection_sbert_x"
        col_proj_y   = "projection_sbert_y"
        text_col     = "abstract"
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

    print(f"  Loading embedding model: {hf_model_id}")
    model = SentenceTransformer(hf_model_id)

    # Incremental embedding — only new papers get encoded
    if col_embed in df.columns:
        needs_embed = df[col_embed].isna()
    else:
        df[col_embed] = None
        needs_embed = pd.Series([True] * len(df))

    n_new = needs_embed.sum()
    if n_new:
        print(f"  Embedding {n_new} new paper(s) with {model_name.upper()}...")
        idx    = df.index[needs_embed].tolist()
        texts  = df.loc[idx, text_col].tolist()
        vecs   = model.encode(texts, show_progress_bar=True, batch_size=16,
                              convert_to_numpy=True)
        for i, pos in enumerate(idx):
            df.at[pos, col_embed] = vecs[i].tolist()
    else:
        print(f"  All papers already embedded with {model_name.upper()} — skipping.")

    all_vectors = np.array(df[col_embed].tolist(), dtype=np.float32)
    n = len(all_vectors)
    print(f"  Projecting {n} papers with UMAP (two-stage, {model_name.upper()})...")

    # Stage 1: 768D → 50D (for clustering)
    reducer_50d = umap_lib.UMAP(n_components=50, metric="cosine",
                                random_state=42, n_neighbors=15)
    coords_50d = reducer_50d.fit_transform(all_vectors)
    df[col_embed_50d] = [row.tolist() for row in coords_50d]

    # Stage 2: 768D → 2D (for display / direction vectors)
    reducer_2d = umap_lib.UMAP(n_components=2, metric="cosine",
                               random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer_2d.fit_transform(all_vectors)
    df[col_proj_x] = coords_2d[:, 0].astype(float)
    df[col_proj_y] = coords_2d[:, 1].astype(float)
    return df


def _embed_only(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Embed papers missing vectors without running UMAP.

    Used by hybrid mode to populate raw vector columns before distance blending.
    """
    from sentence_transformers import SentenceTransformer

    if model_name == "specter2":
        hf_model_id = "allenai/specter2_base"
        col_embed   = "embedding"
        text_col    = "text"
    elif model_name == "sbert":
        hf_model_id = "sentence-transformers/all-mpnet-base-v2"
        col_embed   = "embedding_sbert"
        text_col    = "abstract"
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

    if col_embed in df.columns:
        needs_embed = df[col_embed].isna()
    else:
        df[col_embed] = None
        needs_embed = pd.Series([True] * len(df))

    n_new = needs_embed.sum()
    if n_new:
        print(f"  Embedding {n_new} new paper(s) with {model_name.upper()}...")
        model  = SentenceTransformer(hf_model_id)
        idx    = df.index[needs_embed].tolist()
        texts  = df.loc[idx, text_col].tolist()
        vecs   = model.encode(texts, show_progress_bar=True, batch_size=16,
                              convert_to_numpy=True)
        for i, pos in enumerate(idx):
            df.at[pos, col_embed] = vecs[i].tolist()
    else:
        print(f"  All papers already embedded with {model_name.upper()} — skipping.")
    return df


def compute_hybrid_distances(
    df: pd.DataFrame,
    w_specter2: float = 0.75,
    w_sbert:    float = 0.0,
    w_tfidf:    float = 0.25,
) -> np.ndarray:
    """Build a normalised hybrid distance matrix from up to three sources.

    Each source is independently normalised to [0, 1] before blending.
    Weights are also normalised so the output is always in [0, 1].

    Returns: symmetric (n × n) float32 distance matrix.
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize

    n = len(df)
    total_w = w_specter2 + w_sbert + w_tfidf
    if total_w == 0:
        raise ValueError("At least one hybrid weight must be > 0.")
    w_specter2 /= total_w
    w_sbert    /= total_w
    w_tfidf    /= total_w

    dist = np.zeros((n, n), dtype=np.float32)

    def _norm(d: np.ndarray) -> np.ndarray:
        mx = d.max()
        return (d / mx).astype(np.float32) if mx > 0 else d.astype(np.float32)

    if w_specter2 > 0:
        if "embedding" not in df.columns or df["embedding"].isna().any():
            raise RuntimeError("SPECTER2 vectors missing.")
        vecs = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
        dist += w_specter2 * _norm(cosine_distances(vecs).astype(np.float32))
        print(f"  Hybrid: added SPECTER2 (w={w_specter2:.2f}).")

    if w_sbert > 0:
        if "embedding_sbert" not in df.columns or df["embedding_sbert"].isna().any():
            raise RuntimeError("SBERT vectors missing.")
        vecs = sk_normalize(np.array(df["embedding_sbert"].tolist(), dtype=np.float32))
        dist += w_sbert * _norm(cosine_distances(vecs).astype(np.float32))
        print(f"  Hybrid: added SBERT (w={w_sbert:.2f}).")

    if w_tfidf > 0:
        texts  = df["abstract"].apply(_strip_urls).tolist()
        tfidf  = TfidfVectorizer(max_features=20_000, sublinear_tf=True,
                                  min_df=2, ngram_range=(1, 2))
        tfidf_m = tfidf.fit_transform(texts)
        dist += w_tfidf * _norm(cosine_distances(tfidf_m).astype(np.float32))
        print(f"  Hybrid: added TF-IDF (w={w_tfidf:.2f}).")

    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    print(f"  Hybrid dist: shape={dist.shape}, "
          f"min={dist.min():.4f}, max={dist.max():.4f}, mean={dist.mean():.4f}.")
    return dist


def embed_and_project_hybrid(
    df: pd.DataFrame,
    w_specter2: float = 0.75,
    w_sbert:    float = 0.0,
    w_tfidf:    float = 0.25,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ensure both model vectors exist, build hybrid distance matrix, project to 2D.

    Writes projection_hybrid_x/y to df.
    Returns (df, dist_matrix) so callers can reuse the distance matrix.
    """
    import umap as umap_lib

    if w_specter2 > 0:
        df = _embed_only(df, "specter2")
    if w_sbert > 0:
        df = _embed_only(df, "sbert")

    dist = compute_hybrid_distances(df, w_specter2, w_sbert, w_tfidf)
    n    = len(df)
    print(f"  Projecting {n} papers with UMAP on hybrid distance matrix...")
    reducer = umap_lib.UMAP(
        n_components=2, metric="precomputed", random_state=42,
        n_neighbors=min(15, n - 1), min_dist=0.1,
    )
    coords_2d = reducer.fit_transform(dist)
    df["projection_hybrid_x"] = coords_2d[:, 0].astype(float)
    df["projection_hybrid_y"] = coords_2d[:, 1].astype(float)
    print("  Hybrid projection → projection_hybrid_x / projection_hybrid_y.")
    return df, dist


# ══════════════════════════════════════════════════════════════════════════════
# §7  ATLAS BUILD + DEPLOY
# ══════════════════════════════════════════════════════════════════════════════

def build_and_deploy_atlas(
    db_path:     str,
    proj_x_col:  str,
    proj_y_col:  str,
    labels_path: str,
    run_date:    str,
    docs_dir:    str = "docs",
) -> None:
    """Run the embedding-atlas CLI, unzip, patch config, inject panel HTML.

    Parameters
    ----------
    db_path     : path to the paper parquet (must contain proj_x_col / proj_y_col)
    proj_x_col  : name of the x-coordinate column in db_path
    proj_y_col  : name of the y-coordinate column in db_path
    labels_path : path to the labels parquet (columns: x, y, text)
    run_date    : human-readable date string injected into the panel footer
    docs_dir    : output directory for the built site (default "docs")
    """
    clear_docs_contents(docs_dir)
    print(f"  Building atlas (x={proj_x_col}, y={proj_y_col})...")

    atlas_cmd = [
        "embedding-atlas", db_path,
        "--x",      proj_x_col,
        "--y",      proj_y_col,
        "--labels", labels_path,
        "--export-application", "site.zip",
    ]
    subprocess.run(atlas_cmd, check=True)
    os.system(f"unzip -o site.zip -d {docs_dir}/ && touch {docs_dir}/.nojekyll")

    # ── Config override ────────────────────────────────────────────────────
    config_path = os.path.join(docs_dir, "data", "metadata.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            conf = json.load(f)

        conf["name_column"]  = "title"
        conf["label_column"] = "title"
        conf["color_by"]     = "Prominence"
        conf.setdefault("column_mappings", {}).update({
            "title":        "title",
            "abstract":     "abstract",
            "Prominence":   "Prominence",
            "author_count": "author_count",
            "author_tier":  "author_tier",
            "url":          "url",
        })

        with open(config_path, "w") as f:
            json.dump(conf, f, indent=4)
        print("  Config updated.")
    else:
        print(f"  {config_path} not found — skipping config override.")

    # ── Panel HTML injection ───────────────────────────────────────────────
    # font_html (GA + fonts) → <head>; panel_html (CSS/JS/DOM) → before </body>
    index_file = os.path.join(docs_dir, "index.html")
    if os.path.exists(index_file):
        font_html, panel_html = build_panel_html(run_date)
        with open(index_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if "</head>" in content:
            content = content.replace("</head>", font_html + "\n</head>")
        else:
            content = font_html + "\n" + content
        if "</body>" in content:
            content = content.replace("</body>", panel_html + "\n</body>")
        else:
            content += panel_html
        with open(index_file, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        print("  Info panel injected into index.html.")
    else:
        print(f"  {index_file} not found — skipping panel injection.")

    print("  Atlas build complete.")


# ══════════════════════════════════════════════════════════════════════════════
# §8  HTML PANELS
# ══════════════════════════════════════════════════════════════════════════════

def build_panel_html(run_date: str) -> tuple[str, str]:
    """Return (font_html, panel_html).

    font_html  — GA4 snippet + font <link> tags, injected into <head>.
    panel_html — CSS, JS, and panel DOM, injected before </body>.
    """
    font_html = (
        '<!-- Google tag (gtag.js) -->' +
        '<script async src="https://www.googletagmanager.com/gtag/js?id=G-6LKWKT8838"></script>' +
        '<script>window.dataLayer=window.dataLayer||[];' +
        'function gtag(){dataLayer.push(arguments);}gtag("js",new Date());gtag("config","G-6LKWKT8838");</script>' +
        '<link rel="preconnect" href="https://fonts.googleapis.com">' +
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>' +
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )
    panel_html = (
        """
<style>
  :root {
    --arm-font: 'Inter', system-ui, -apple-system, sans-serif;
    --arm-accent: #60a5fa;
    --arm-accent-dim: rgba(96,165,250,0.12);
    --arm-border: rgba(255,255,255,0.08);
    --arm-muted: #94a3b8;
    --arm-bg: rgba(15,23,42,0.82);
    --arm-panel-w: 300px;
  }

  /* ── Top title bar ───────────────────────────────────────────────── */
  #arm-title-bar {
    position:fixed; top:0; left:50%; transform:translateX(-50%);
    z-index:1000000;
    background:rgba(15,23,42,0.82); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    border:1px solid var(--arm-border); border-top:none;
    border-radius:0 0 10px 10px;
    padding:6px 22px;
    font-family:var(--arm-font); font-size:15px; font-weight:700;
    color:#f1f5f9; letter-spacing:-0.01em; line-height:1.3;
    white-space:nowrap; user-select:none;
    box-shadow:0 3px 20px rgba(0,0,0,0.4);
  }
  #arm-title-bar span { color:var(--arm-accent); }

  /* ── Tab strip ───────────────────────────────────────────────────── */
  #arm-tab-strip {
    position:fixed; left:0; top:50%; transform:translateY(-50%);
    z-index:1000000; display:flex; flex-direction:column; gap:4px;
  }
  .arm-tab {
    display:flex; align-items:center; justify-content:center;
    writing-mode:vertical-rl; text-orientation:mixed;
    background:rgba(15,23,42,0.90); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    color:var(--arm-accent); border:1px solid var(--arm-border); border-left:none;
    padding:18px 9px; cursor:pointer;
    font-family:var(--arm-font); font-weight:600; font-size:12px;
    letter-spacing:0.12em; text-transform:uppercase;
    box-shadow:3px 0 20px rgba(0,0,0,0.4);
    transition:background 0.2s,color 0.2s,box-shadow 0.2s; user-select:none;
  }
  #arm-shortcuts-tab { border-radius:0 10px 0 0; }
  #arm-about-tab     { border-radius:0 0 10px 0; }
  .arm-tab:hover     { background:rgba(30,41,59,0.95); color:#93c5fd; box-shadow:4px 0 24px rgba(96,165,250,0.2); }
  .arm-tab.arm-active { background:rgba(30,41,59,0.98); color:#93c5fd; }

  /* ── Shared panel chrome ─────────────────────────────────────────── */
  .arm-panel {
    position:fixed; left:0; top:50%;
    transform:translateY(-50%) translateX(-110%);
    z-index:999999; width:var(--arm-panel-w); max-height:88vh;
    display:flex; flex-direction:column;
    background:var(--arm-bg); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    border:1px solid var(--arm-border); border-left:none; border-radius:0 16px 16px 0;
    box-shadow:6px 0 40px rgba(0,0,0,0.6),inset 1px 0 0 rgba(255,255,255,0.04);
    font-family:var(--arm-font); font-size:13px; color:#e2e8f0; line-height:1.65;
    transition:transform 0.32s cubic-bezier(0.4,0,0.2,1); overflow:hidden;
  }
  .arm-panel.arm-open { transform:translateY(-50%) translateX(0); }
  .arm-body { overflow-y:auto; overflow-x:hidden; padding:22px 20px 16px; flex:1; scrollbar-width:thin; scrollbar-color:#334155 transparent; }
  .arm-body::-webkit-scrollbar { width:4px; }
  .arm-body::-webkit-scrollbar-thumb { background:#334155; border-radius:4px; }

  /* ── Shared typography ───────────────────────────────────────────── */
  .arm-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px; }
  .arm-title  { font-size:15px; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em; line-height:1.3; margin:0; }
  .arm-title span { color:var(--arm-accent); }
  .arm-close  { background:none; border:none; color:var(--arm-muted); cursor:pointer; font-size:17px; line-height:1; padding:2px 4px; border-radius:4px; transition:color 0.15s,background 0.15s; flex-shrink:0; margin-left:8px; }
  .arm-close:hover { color:#f1f5f9; background:rgba(255,255,255,0.07); }
  .arm-byline { font-size:12px; color:var(--arm-muted); margin-bottom:16px; }
  .arm-byline a { color:var(--arm-accent); text-decoration:none; font-weight:500; }
  .arm-byline a:hover { color:#93c5fd; text-decoration:underline; }
  .arm-divider { border:none; border-top:1px solid var(--arm-border); margin:14px 0; }
  .arm-section { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#475569; margin:0 0 8px; }
  .arm-p { color:#94a3b8; margin:0 0 10px; font-size:12.5px; }
  .arm-p a { color:var(--arm-accent); text-decoration:none; }
  .arm-p a:hover { color:#93c5fd; text-decoration:underline; }
  .arm-book { display:flex; align-items:center; gap:10px; background:rgba(255,255,255,0.03); border:1px solid var(--arm-border); border-radius:8px; padding:10px 12px; text-decoration:none; transition:background 0.2s,border-color 0.2s; margin-bottom:8px; }
  .arm-book:hover { background:rgba(96,165,250,0.07); border-color:rgba(96,165,250,0.3); }
  .arm-book-icon { font-size:22px; flex-shrink:0; }
  .arm-book-text { display:flex; flex-direction:column; }
  .arm-book-title { font-size:12px; font-weight:600; color:#e2e8f0; line-height:1.3; margin-bottom:2px; }
  .arm-book-sub { font-size:11px; color:var(--arm-accent); }
  .arm-footer { padding:10px 20px 14px; border-top:1px solid var(--arm-border); display:flex; align-items:center; gap:7px; flex-shrink:0; }
  .arm-status-dot { width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 6px rgba(34,197,94,0.7); flex-shrink:0; animation:arm-pulse 2.5s ease-in-out infinite; }
  @keyframes arm-pulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }
  .arm-status-text { font-size:11px; color:#475569; font-family:var(--arm-font); }
  .arm-status-text strong { color:#64748b; font-weight:500; }

  /* ── Shortcut tiles ──────────────────────────────────────────────── */
  .arm-sc-intro { font-size:12.5px; color:#94a3b8; margin:0 0 14px; line-height:1.5; }
  .arm-tile {
    display:flex; align-items:center; gap:11px;
    background:rgba(255,255,255,0.03); border:1px solid var(--arm-border);
    border-radius:8px; padding:10px 12px; text-decoration:none;
    transition:background 0.2s,border-color 0.2s; margin-bottom:7px; cursor:pointer;
  }
  .arm-tile:hover { background:rgba(96,165,250,0.07); border-color:rgba(96,165,250,0.3); }
  .arm-tile:hover .arm-tile-label { color:#e2e8f0; }
  .arm-tile.arm-tile-active { background:rgba(96,165,250,0.12); border-color:rgba(96,165,250,0.4); }
  .arm-tile.arm-tile-active .arm-tile-label { color:#93c5fd; }
  .arm-tile-icon { flex-shrink:0; width:18px; height:18px; color:#64748b; display:flex; align-items:center; justify-content:center; }
  .arm-tile-text { display:flex; flex-direction:column; gap:1px; }
  .arm-tile-label { font-size:12px; font-weight:600; color:#cbd5e1; line-height:1.3; transition:color 0.15s; }
  .arm-tile-sub   { font-size:11px; color:#64748b; line-height:1.3; }

  /* ── Custom point popup ──────────────────────────────────────────── */
  #arm-cp {
    position:fixed; z-index:999998; width:320px;
    background:var(--arm-bg); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    border:1px solid var(--arm-border); border-radius:12px;
    box-shadow:0 8px 32px rgba(0,0,0,0.6);
    font-family:var(--arm-font); color:#e2e8f0;
    display:none;
  }
  .arm-cp-body { padding:16px; display:flex; flex-direction:column; gap:10px; }
  .arm-cp-title-row { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; }
  .arm-cp-title { font-size:13px; font-weight:700; color:#f1f5f9; line-height:1.4; text-decoration:none; flex:1; }
  .arm-cp-title:hover { color:var(--arm-accent); text-decoration:underline; }
  .arm-cp-badge { flex-shrink:0; font-size:10px; font-weight:600; border-radius:4px; padding:2px 6px; white-space:nowrap; margin-top:2px; border:1px solid; }
  .arm-cp-badge-elite      { color:#fbbf24; background:rgba(251,191,36,0.12); border-color:rgba(251,191,36,0.35); }
  .arm-cp-badge-enhanced   { color:#60a5fa; background:rgba(96,165,250,0.12); border-color:rgba(96,165,250,0.35); }
  .arm-cp-badge-emerging   { color:#34d399; background:rgba(52,211,153,0.10); border-color:rgba(52,211,153,0.30); }
  .arm-cp-badge-unverified { color:#64748b; background:rgba(100,116,139,0.10); border-color:rgba(100,116,139,0.25); }
  .arm-cp-topic { font-size:11px; color:var(--arm-accent); font-weight:500; }
  .arm-cp-abstract { font-size:11.5px; color:#94a3b8; line-height:1.6; max-height:9.6em; overflow-y:auto; scrollbar-width:thin; scrollbar-color:#334155 transparent; }
  .arm-cp-abstract::-webkit-scrollbar { width:4px; }
  .arm-cp-abstract::-webkit-scrollbar-thumb { background:#334155; border-radius:4px; }
  .arm-cp-meta { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .arm-cp-meta span { font-size:11px; color:#64748b; }
  .arm-cp-badge { font-size:10px; font-weight:600; letter-spacing:0.04em; padding:2px 7px; border-radius:99px; border:1px solid; }
  .arm-cp-badge-elite      { color:#f59e0b; background:rgba(245,158,11,0.12); border-color:rgba(245,158,11,0.35); }
  .arm-cp-badge-enhanced   { color:#60a5fa; background:rgba(96,165,250,0.12); border-color:rgba(96,165,250,0.35); }
  .arm-cp-badge-emerging   { color:#4ade80; background:rgba(74,222,128,0.10); border-color:rgba(74,222,128,0.30); }
  .arm-cp-badge-unverified { color:#475569; background:rgba(71,85,105,0.15);  border-color:rgba(71,85,105,0.35); }
  .arm-cp-keywords { font-size:10.5px; color:#475569; line-height:1.5; }
  .arm-cp-footer { padding:10px 16px; border-top:1px solid var(--arm-border); display:flex; justify-content:flex-end; }
  .arm-cp-btn { font-size:11px; font-weight:600; color:var(--arm-accent); background:var(--arm-accent-dim); border:1px solid rgba(96,165,250,0.25); border-radius:6px; padding:5px 12px; text-decoration:none; transition:background 0.15s; font-family:var(--arm-font); cursor:pointer; }
  .arm-cp-btn:hover { background:rgba(96,165,250,0.2); }
</style>

<script>
  function armToggle(panelId, tabId, hideTabId) {
    var panel = document.getElementById(panelId);
    var isOpen = panel.classList.contains('arm-open');
    document.querySelectorAll('.arm-panel').forEach(function(p) { p.classList.remove('arm-open'); });
    document.querySelectorAll('.arm-tab').forEach(function(t) { t.classList.remove('arm-active'); t.style.display = ''; });
    if (!isOpen) {
      panel.classList.add('arm-open');
      document.getElementById(tabId).classList.add('arm-active');
      if (hideTabId) document.getElementById(hideTabId).style.display = 'none';
    }
  }
  function armClose(panelId, tabId) {
    document.getElementById(panelId).classList.remove('arm-open');
    document.getElementById(tabId).classList.remove('arm-active');
    document.querySelectorAll('.arm-tab').forEach(function(t) { t.style.display = ''; });
  }

  // Atlas color-by: find the select whose options include Prominence,
  // set value via native setter (Svelte needs this), fire change event.
  function armSetColor(columnName, tileEl) {
    var colorSelect = Array.from(document.querySelectorAll('select')).find(function(sel) {
      return Array.from(sel.options).some(function(o) { return o.text.includes('Prominence'); });
    });
    if (colorSelect) {
      var nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype, 'value'
      ).set;
      nativeSetter.call(colorSelect, JSON.stringify(columnName));
      colorSelect.dispatchEvent(new Event('change', { bubbles: true }));
    }
    armClose('arm-shortcuts-panel', 'arm-shortcuts-tab');
    document.querySelectorAll('.arm-tile').forEach(function(t) { t.classList.remove('arm-tile-active'); });
    if (tileEl) { tileEl.classList.add('arm-tile-active'); }
  }
</script>

<!-- ── Custom point popup ──────────────────────────────────────────── -->
<div id="arm-cp">
  <div class="arm-cp-body">
    <a class="arm-cp-title" id="arm-cp-title" href="#" target="_blank" rel="noopener"></a>
    <div class="arm-cp-abstract" id="arm-cp-abstract"></div>
    <div class="arm-cp-meta"   id="arm-cp-meta"></div>
  </div>
  <div class="arm-cp-footer">
    <a class="arm-cp-btn" id="arm-cp-btn" href="#" target="_blank" rel="noopener">Open on arXiv →</a>
  </div>
</div>

<script>
(function() {
  var cp = document.getElementById('arm-cp');

  function extractData(popup) {
    var data = {};
    popup.querySelectorAll('[class*="px-2"][class*="flex"][class*="items-center"]').forEach(function(row) {
      var kids = Array.from(row.children).filter(function(c) { return c.textContent.trim(); });
      if (kids.length >= 2) {
        data[kids[0].textContent.trim()] = kids[1].textContent.trim();
      }
    });
    return data;
  }

  function showCP(data, atlasPopup) {
    var title      = data['title']      || '';
    var url        = data['url']        || '#';
    var abs        = data['abstract']   || '';
    var count      = data['author_count'] || '';
    var date       = (data['date_added'] || '').substring(0, 10);
    var prominence = data['Prominence'] || '';

    document.getElementById('arm-cp-title').textContent = title;
    document.getElementById('arm-cp-title').href = url;
    document.getElementById('arm-cp-btn').href = url;

    document.getElementById('arm-cp-abstract').textContent = abs;

    var metaEl = document.getElementById('arm-cp-meta');
    metaEl.innerHTML = '';
    if (count) { var s1 = document.createElement('span'); s1.textContent = '\\u{1F465}\\uFE0E ' + count + ' authors'; metaEl.appendChild(s1); }
    if (date)  { var s2 = document.createElement('span'); s2.textContent = '\\u{1F4C5}\\uFE0E ' + date; metaEl.appendChild(s2); }
    if (prominence) {
      var pill = document.createElement('span');
      pill.textContent = prominence;
      var tierClass = {
        'Elite':      'arm-cp-badge-elite',
        'Enhanced':   'arm-cp-badge-enhanced',
        'Emerging':   'arm-cp-badge-emerging',
        'Unverified': 'arm-cp-badge-unverified'
      }[prominence] || 'arm-cp-badge-unverified';
      pill.className = 'arm-cp-badge ' + tierClass;
      metaEl.appendChild(pill);
    }

    // Smart positioning: right of Atlas popup, flip left if near viewport edge
    var rect = atlasPopup.getBoundingClientRect();
    var left = rect.right + 12;
    if (left + 330 > window.innerWidth) { left = rect.left - 330 - 12; }
    if (left < 8) { left = 8; }
    var top = Math.max(8, rect.top);

    cp.style.left = left + 'px';
    cp.style.top  = top  + 'px';
    cp.style.display = 'block';

    // Clamp vertically after rendering so we know the height
    requestAnimationFrame(function() {
      var maxTop = window.innerHeight - cp.offsetHeight - 8;
      if (parseFloat(cp.style.top) > maxTop) { cp.style.top = Math.max(8, maxTop) + 'px'; }
    });
  }

  function hideCP() { cp.style.display = 'none'; }

  function findAtlasPopup(node) {
    if (node.nodeType !== 1) return null;
    if (node.classList && node.classList.contains('border-slate-300') && node.classList.contains('shadow-md')) return node;
    return node.querySelector ? node.querySelector('.border-slate-300.shadow-md') : null;
  }

  function waitForRoot() {
    var root = document.querySelector('.embedding-atlas-root');
    if (!root) { setTimeout(waitForRoot, 300); return; }

    // Hide Atlas's own popup immediately whenever it appears
    new MutationObserver(function(mutations) {
      mutations.forEach(function(m) {
        m.addedNodes.forEach(function(node) {
          var popup = findAtlasPopup(node);
          if (popup) { popup.style.display = 'none'; }
        });
      });
    }).observe(root, { childList: true, subtree: true });

    // On every click inside Atlas, wait for its popup to settle then read it
    root.addEventListener('click', function() {
      // Poll for up to 600ms for Atlas popup to appear/update
      var attempts = 0;
      var interval = setInterval(function() {
        attempts++;
        var popup = root.querySelector('.border-slate-300.shadow-md');
        if (popup) {
          popup.style.display = 'none';
          showCP(extractData(popup), popup);
          clearInterval(interval);
        } else if (attempts > 12) {
          // No popup appeared — user clicked empty space, hide ours
          clearInterval(interval);
          hideCP();
        }
      }, 50);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForRoot);
  } else {
    waitForRoot();
  }
})();
</script>

<!-- ── Top title bar ──────────────────────────────────────────────── -->
<div id="arm-title-bar">The <span>AI Research</span> Atlas</div>

<!-- ── Tab strip ──────────────────────────────────────────────────── -->
<div id="arm-tab-strip">
  <button id="arm-shortcuts-tab" class="arm-tab"
    onclick="armToggle('arm-shortcuts-panel','arm-shortcuts-tab','arm-about-tab')"
    aria-label="Open Shortcuts panel">Shortcuts</button>
  <button id="arm-about-tab" class="arm-tab"
    onclick="armToggle('arm-about-panel','arm-about-tab','arm-shortcuts-tab')"
    aria-label="Open About panel">About</button>
</div>

<!-- ═══════════════════════════════════════════════════════════
     SHORTCUTS PANEL
═══════════════════════════════════════════════════════════ -->
<div id="arm-shortcuts-panel" class="arm-panel" role="complementary" aria-label="Shortcuts">
  <div class="arm-body">
    <div class="arm-header">
      <p class="arm-title"><span>Shortcuts</span></p>
      <button class="arm-close" onclick="armClose('arm-shortcuts-panel','arm-shortcuts-tab')" aria-label="Close">&#x2715;</button>
    </div>
    <p class="arm-sc-intro">Click a tile to see points colored by</p>

    <a class="arm-tile" href="javascript:void(0)" onclick="armSetColor('Prominence', this)">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="9,2 11.1,6.6 16.2,7.3 12.6,10.7 13.5,15.8 9,13.4 4.5,15.8 5.4,10.7 1.8,7.3 6.9,6.6"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author prominence</span><span class="arm-tile-sub">Elite · Enhanced · Emerging · Unverified</span></span>
    </a>

    <a class="arm-tile" href="javascript:void(0)" onclick="armSetColor('author_count', this)">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="6.5" cy="6" r="2.5"/><path d="M1 16c0-3.3 2.5-5 5.5-5"/><circle cx="13" cy="6" r="2.5"/><path d="M17 16c0-3.3-2.5-5-5.5-5"/><path d="M10 16c0-2.8-1.8-4.5-4-4.5"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author count</span><span class="arm-tile-sub">More authors tends to be better</span></span>
    </a>

    <a class="arm-tile" href="javascript:void(0)" onclick="armSetColor('author_tier', this)">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5L9 4l5 4.5"/><path d="M5.5 8v5.5h7V8"/><path d="M7 13.5v-3h4v3"/><path d="M3 8.5h12"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author seniority</span><span class="arm-tile-sub">Highlights established researchers</span></span>
    </a>

  </div>
  <div class="arm-footer">
    <div class="arm-status-dot"></div>
    <span class="arm-status-text">Last updated <strong>""" + run_date + """ UTC</strong></span>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════
     ABOUT PANEL
═══════════════════════════════════════════════════════════ -->
<div id="arm-about-panel" class="arm-panel" role="complementary" aria-label="About this atlas">
  <div class="arm-body">
    <div class="arm-header">
      <p class="arm-title">The <span>AI Research</span> Atlas</p>
      <button class="arm-close" onclick="armClose('arm-about-panel','arm-about-tab')" aria-label="Close panel">&#x2715;</button>
    </div>
    <div class="arm-byline">By <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" rel="noopener">Lee Fischman</a></div>

    <p class="arm-p">A live semantic map of recent AI research from arXiv (cs.AI), rebuilt every day across a rolling """ + str(RETENTION_DAYS) + """-day window. Each point is a paper. Groups are determined by Claude Haiku based on shared research methodology and problem formulation.</p>
    <p class="arm-p"><a href="https://github.com/LeeFischman/AI-research-atlas" target="_blank" rel="noopener">More...</a></p>

    <hr class="arm-divider">

    <p class="arm-section">How to use</p>
    <p class="arm-p">Click any point to read its abstract and open the PDF on arXiv. Use the search bar to find papers by keyword or phrase. Drag to pan; scroll or pinch to zoom.</p>

    <hr class="arm-divider">

    <p class="arm-section">Books by the author</p>
    <a class="arm-book" href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" rel="noopener">
      <span class="arm-book-icon">&#x1F4D8;</span>
      <span class="arm-book-text"><span class="arm-book-title">Building Deep Learning Products</span><span class="arm-book-sub">Available on Amazon &#x2192;</span></span>
    </a>
    <p class="arm-p" style="margin-top:4px;"><a href="https://donate.stripe.com/6oU5kD9sE17y87J3gI1ck00" target="_blank" rel="noopener">Buy me a bagel &#x1F96F;</a></p>

    <hr class="arm-divider">

    <p class="arm-section">Powered by</p>
    <p class="arm-p"><a href="https://apple.github.io/embedding-atlas/" target="_blank" rel="noopener">Apple Embedding Atlas</a>, <a href="https://openalex.org" target="_blank" rel="noopener">OpenAlex</a>, and <a href="https://allenai.org/blog/specter2-adapting-scientific-document-embeddings-to-multiple-fields-and-task-formats-c95686c06567" target="_blank" rel="noopener">SPECTER2</a></p>
  </div>
  <div class="arm-footer">
    <div class="arm-status-dot"></div>
    <span class="arm-status-text">Last updated <strong>""" + run_date + """ UTC</strong></span>
  </div>
</div>
"""
    )
    return font_html, panel_html
