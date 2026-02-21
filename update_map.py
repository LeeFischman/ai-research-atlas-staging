import arxiv
import pandas as pd
import numpy as np
import subprocess
import os
import re
import json
import time
import shutil
import random
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DB_PATH         = "database.parquet"
STOP_WORDS_PATH = "stop_words.csv"
RUN_STATE_PATH        = "run_state.json"        # tracks last fetch date to avoid duplicate arXiv calls
AUTHOR_STATS_CACHE   = "author_stats_cache.json"  # persists OpenAlex author stats between runs
RETENTION_DAYS  = 14      # papers older than this are pruned each run
ARXIV_MAX       = 1500    # max papers fetched per arXiv query

# Embedding mode is controlled by the EMBEDDING_MODE env var.
# Set automatically by the workflow_dispatch input in the YAML.
# "full"        — CLI handles SPECTER2 + UMAP internally on every run.
# "incremental" — Python embeds only NEW papers; UMAP runs over all stored vectors.
# Embedding mode controls how papers are embedded and projected each run.
# "full"        — CLI handles SPECTER2 + UMAP internally. Slower but always
#                 produces a globally coherent layout. --text feeds both
#                 embeddings and TF-IDF labels so label_text column is unused.
# "incremental" — Python embeds only NEW papers; UMAP re-projects all stored
#                 vectors. Faster. --text only feeds TF-IDF so label_text
#                 (title-only) is used for sharper cluster labels.
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "incremental").strip().lower()

print(f"▶  Embedding mode : {EMBEDDING_MODE.upper()}")


# ──────────────────────────────────────────────
# 1. TEXT SCRUBBER
# ──────────────────────────────────────────────
def scrub_model_words(text: str) -> str:
    pattern = re.compile(r'model(?:s|ing|ed|er|ers)?\b', re.IGNORECASE)
    return " ".join(pattern.sub("", text).split())


# ──────────────────────────────────────────────
# 2. DOCS CLEANUP
# ──────────────────────────────────────────────
def clear_docs_contents(target_dir: str) -> None:
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


# ──────────────────────────────────────────────
# 3. RUN STATE
#    Persists the last successful arXiv fetch date so re-running the
#    workflow on the same calendar day (e.g. after a partial failure)
#    skips the fetch and reuses the papers already in the rolling DB.
# ──────────────────────────────────────────────
def load_run_state() -> dict:
    if os.path.exists(RUN_STATE_PATH):
        try:
            with open(RUN_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_run_state(state: dict) -> None:
    with open(RUN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ──────────────────────────────────────────────
# 4. REPUTATION SCORING
# ──────────────────────────────────────────────
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([

    # ── US Universities ──────────────────────────────────────────────────
    "MIT", "Stanford", "CMU", "Carnegie Mellon",
    "UC Berkeley", "Berkeley", "Caltech",
    "Harvard", "Princeton", "Yale", "Columbia",
    "University of Washington", "UW Seattle",
    "University of Michigan", "University of Illinois",
    "UIUC", "UT Austin", "NYU", "Cornell",
    "Georgia Tech", "UCLA", "UCSD", "USC",
    "University of Pennsylvania", "UPenn",
    "Johns Hopkins", "Duke", "Brown", "Dartmouth",
    "University of Wisconsin", "Purdue", "Ohio State",
    "University of Maryland", "University of Massachusetts",

    # ── US National Labs & Research Institutes ───────────────────────────
    "Allen Institute", "AI2", "Allenai",
    "IBM Research", "Microsoft Research", "MSR",
    "Google Research", "Google Brain", "Google DeepMind",
    "NVIDIA Research", "Intel Labs", "Salesforce Research",
    "Adobe Research", "Baidu Research", "Amazon Science",
    "Apple ML Research",

    # ── US AI Labs ───────────────────────────────────────────────────────
    "OpenAI", "Anthropic", "DeepMind", "Google DeepMind",
    "FAIR", "Meta AI", "xAI", "Mistral",
    "Cohere", "Stability AI", "Inflection AI",
    "Character AI", "Runway", "Hugging Face",

    # ── UK ───────────────────────────────────────────────────────────────
    "Oxford", "University of Oxford",
    "Cambridge", "University of Cambridge",
    "Imperial College", "UCL", "University College London",
    "Edinburgh", "University of Edinburgh",
    "King's College London", "University of Manchester",
    "University of Bristol", "University of Warwick",
    "Alan Turing Institute",

    # ── Canada ───────────────────────────────────────────────────────────
    "University of Toronto", "UofT", "Vector Institute",
    "McGill", "McGill University",
    "Mila", "Montreal Institute",
    "University of Montreal",
    "University of British Columbia", "UBC",
    "University of Alberta",

    # ── France ───────────────────────────────────────────────────────────
    "INRIA", "ENS", "CNRS", "Sorbonne",
    "PSL University", "Paris-Saclay",

    # ── Germany ──────────────────────────────────────────────────────────
    "Max Planck", "MPI",
    "TU Berlin", "TU Munich", "Technical University of Munich",
    "DFKI", "Helmholtz",
    "University of Tuebingen",

    # ── Switzerland ──────────────────────────────────────────────────────
    "ETH Zurich", "EPFL", "University of Zurich",

    # ── Netherlands ──────────────────────────────────────────────────────
    "University of Amsterdam", "CWI", "Delft",

    # ── Scandinavia ──────────────────────────────────────────────────────
    "KTH", "Chalmers", "University of Copenhagen", "DTU", "NTNU",

    # ── Israel ───────────────────────────────────────────────────────────
    "Technion", "Hebrew University", "Weizmann Institute",
    "Tel Aviv University", "Bar-Ilan University",

    # ── China ────────────────────────────────────────────────────────────
    "Tsinghua", "Tsinghua University",
    "Peking University", "PKU",
    "Shanghai AI Lab",
    "Zhejiang University",
    "USTC", "Fudan University", "Renmin University",
    "Chinese Academy of Sciences", "CAS",
    "BAAI", "Beijing Academy of Artificial Intelligence",
    "Alibaba DAMO", "Tencent AI Lab", "ByteDance", "SenseTime",

    # ── Japan ────────────────────────────────────────────────────────────
    "University of Tokyo", "Kyoto University",
    "RIKEN", "RIKEN AIP",
    "Osaka University", "Tohoku University",
    "Tokyo Institute of Technology", "Tokyo Tech",
    "Preferred Networks",

    # ── South Korea ──────────────────────────────────────────────────────
    "KAIST", "Seoul National University", "SNU",
    "POSTECH", "Yonsei University",
    "Samsung Research", "Naver", "LG AI Research", "Kakao",

    # ── Singapore ────────────────────────────────────────────────────────
    "NUS", "National University of Singapore",
    "NTU", "Nanyang Technological University",
    "AI Singapore", "ASTAR",

    # ── Australia ────────────────────────────────────────────────────────
    "University of Sydney", "University of Melbourne",
    "Australian National University", "ANU",
    "University of Queensland", "CSIRO",

    # ── India ────────────────────────────────────────────────────────────
    "IIT", "IIT Bombay", "IIT Delhi", "IIT Madras",
    "IISc", "Indian Institute of Science",
    "Microsoft Research India",

    # ── Middle East ──────────────────────────────────────────────────────
    "KAUST", "King Abdullah University",
    "Mohamed bin Zayed University", "MBZUAI",

    # ── Latin America ────────────────────────────────────────────────────
    "University of Sao Paulo", "USP", "PUC-Rio",

]) + r")\b", re.IGNORECASE)


def categorize_authors(n: int) -> str:
    if n <= 3:  return "1-3 Authors"
    if n <= 7:  return "4-7 Authors"
    return "8+ Authors"

def author_reputation_score(n: int) -> int:
    """
    Larger teams signal institutional backing or collaborative projects,
    which correlates loosely with higher-quality, better-resourced work.
    Solo and very small teams get no boost; large consortia get the most.
    """
    if n >= 8:  return 3   # large consortium / industrial lab
    if n >= 4:  return 1   # mid-sized collaboration
    return 0               # 1-3 authors — no author-count bonus

def calculate_reputation(row) -> str:
    score = 0
    full_text = f"{row['title']} {row['abstract']}".lower()

    # Institution match — strongest signal.
    # Use structured OpenAlex institution names when available (more accurate);
    # fall back to regex text-search in title/abstract for unindexed papers.
    inst_names = row.get("openalex_institution_names") if hasattr(row, "get") else None
    if isinstance(inst_names, list) and inst_names:
        inst_text = " | ".join(inst_names)
        if INSTITUTION_PATTERN.search(inst_text):
            score += 3
    else:
        if INSTITUTION_PATTERN.search(full_text):
            score += 3

    # Author seniority — at least one established researcher on the paper
    seniority = row.get("author_seniority") if hasattr(row, "get") else None
    if seniority == "Established":
        score += 2

    # Public codebase — indicates reproducibility commitment
    if any(k in full_text for k in ["github.com", "huggingface.co"]):
        score += 2

    # Author count — larger teams tend to reflect institutional resources
    score += author_reputation_score(row["author_count"])

    return "Reputation Enhanced" if score >= 3 else "Reputation Std"


# ──────────────────────────────────────────────
# 5. ROLLING DATABASE
# ──────────────────────────────────────────────
def load_existing_db() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    df = pd.read_parquet(DB_PATH)
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
    """Merge new papers; duplicate arXiv IDs: new version overwrites old."""
    if existing.empty:
        return new
    kept = existing[~existing["id"].isin(new["id"])]
    overwritten = len(existing) - len(kept)
    if overwritten:
        print(f"  Overwrote {overwritten} updated paper(s).")
    return pd.concat([kept, new], ignore_index=True)


# ──────────────────────────────────────────────
# 6. ARXIV FETCH WITH EXPONENTIAL BACKOFF
# ──────────────────────────────────────────────
BASE_WAIT   = 15
MAX_WAIT    = 480
MAX_RETRIES = 7

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
    raise RuntimeError(f"arXiv fetch failed after {MAX_RETRIES} attempts. Last: {last_exc}")


# ──────────────────────────────────────────────
# 5.5 OPENALEX INSTITUTION LOOKUP
#
# Replaces regex text-search with structured institution data.
# Batch lookups via the OpenAlex REST API using arXiv IDs.
# No API key required; mailto param places us in the polite pool
# (~10 req/sec, effectively unlimited for our volume).
#
# Response fields used in Step 1:
#   authorships[].institutions[].display_name  → matched against INSTITUTION_PATTERN
#   authorships[].institutions[].country_code  → stored for Step 3
#   authorships[].institutions[].type          → stored for Step 4
#
# Response fields used in Step 2:
#   authorships[].author.id                    → OpenAlex author ID for seniority lookup
#
# Schema additions to database.parquet:
#   openalex_institution_names    list[str]  — deduplicated institution names
#   openalex_institution_countries list[str] — ISO-2 country codes (parallel to names)
#   openalex_institution_types    list[str]  — education/company/government/nonprofit
#   openalex_author_ids           list[str]  — OpenAlex author IDs for this paper
#   author_seniority              str        — "Established" / "Emerging" / "Unknown"
#   openalex_fetched              bool       — True once we have queried OpenAlex
# ──────────────────────────────────────────────
OPENALEX_EMAIL  = "lee@leefischman.com"   # polite-pool identifier
OPENALEX_SLEEP  = 0.12                     # seconds between requests (~8 req/sec)

def fetch_openalex_data(arxiv_ids: list) -> dict:
    """
    Look up institution data for a list of arXiv IDs using pyalex.

    pyalex handles all URL construction and endpoint resolution internally,
    avoiding the URL-encoding issues encountered with raw urllib calls.

    Returns a dict containing ONLY the IDs that got a successful response.
    IDs not present in the returned dict stay openalex_fetched=False and
    are retried on the next daily run (covers the ~1-2 day indexing lag).
    """
    if not arxiv_ids:
        return {}

    try:
        import pyalex
        pyalex.config.email = OPENALEX_EMAIL
    except ImportError:
        print("  pyalex not installed — skipping OpenAlex lookup. "
              "Add 'pyalex' to requirements.txt.")
        return {}

    result: dict[str, dict] = {}
    indexed = 0
    not_found = 0
    errors = 0

    print(f"  Querying OpenAlex for {len(arxiv_ids)} arXiv IDs "
          f"(pyalex, ~{len(arxiv_ids) * OPENALEX_SLEEP:.0f}s)...")

    for i, aid in enumerate(arxiv_ids):
        clean_id = re.sub(r"v\d+$", "", aid)

        try:
            # Use the arXiv DOI as the OpenAlex entity ID.
            # Format: doi:10.48550/arxiv.{id} — confirmed working via browser test.
            # This avoids all URL-encoding issues: the DOI contains no slashes
            # in the path, so pyalex passes it through cleanly.
            work = pyalex.Works()[f"doi:10.48550/arxiv.{clean_id}"]

            names, countries, types = [], [], []
            author_ids = []
            for authorship in work.get("authorships", []):
                author_id = authorship.get("author", {}).get("id", "").strip()
                if author_id and author_id not in author_ids and len(author_ids) < 8:
                    author_ids.append(author_id)
                for inst in authorship.get("institutions", []):
                    name    = inst.get("display_name", "").strip()
                    country = inst.get("country_code", "").strip()
                    itype   = inst.get("type", "").strip()
                    if name and name not in names:
                        names.append(name)
                        countries.append(country)
                        types.append(itype)

            # ── Step 5: topic taxonomy and keywords ──────────────────────
            # primary_topic gives the single best taxonomy match.
            # Only trust it when score ≥ 0.85; below that label as Unknown.
            TOPIC_SCORE_THRESHOLD = 0.0
            primary = work.get("primary_topic") or {}
            topic_score = primary.get("score", 0) or 0
            if topic_score >= TOPIC_SCORE_THRESHOLD:
                openalex_topic    = primary.get("display_name", "Unknown") or "Unknown"
                subfield_obj      = primary.get("subfield") or {}
                openalex_subfield = subfield_obj.get("display_name", "Unknown") or "Unknown"
            else:
                openalex_topic    = "Unknown"
                openalex_subfield = "Unknown"

            # Keywords: free-text phrases, filtered to score ≥ 0.5 to drop noise.
            KEYWORD_SCORE_THRESHOLD = 0.5
            openalex_keywords = [
                kw["display_name"]
                for kw in (work.get("keywords") or [])
                if (kw.get("score") or 0) >= KEYWORD_SCORE_THRESHOLD
                and kw.get("display_name")
            ]

            result[aid] = {
                "institution_names":     names,
                "institution_countries": countries,
                "institution_types":     types,
                "author_ids":            author_ids,
                "openalex_topic":        openalex_topic,
                "openalex_subfield":     openalex_subfield,
                "openalex_keywords":     openalex_keywords,
            }
            if names:
                indexed += 1

        except Exception as exc:
            err_str = str(exc).lower()
            if "404" in err_str or "not found" in err_str or "no work" in err_str:
                not_found += 1
            else:
                errors += 1
                print(f"  OpenAlex error for {aid} ({type(exc).__name__}: {exc}) — will retry next run.")

        time.sleep(OPENALEX_SLEEP)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(arxiv_ids)} queried "
                  f"({indexed} with institutions so far)")

    responded = len(result)
    print(f"  OpenAlex: {indexed} with institutions | "
          f"{responded - indexed} found but no institution data | "
          f"{not_found} not yet indexed (will retry) | "
          f"{errors} errors.")
    return result


def apply_openalex_columns(df: pd.DataFrame, openalex: dict) -> pd.DataFrame:
    """
    Write OpenAlex lookup results into the DataFrame.

    Only rows whose arXiv ID is present in `openalex` are updated —
    the dict only contains IDs that got a 200 response, so:
      - openalex_fetched=True  → we got a response (may or may not have institutions)
      - openalex_fetched=False → 404 or error; will be retried on the next run
    """
    df = df.copy()

    for col in ("openalex_institution_names", "openalex_institution_countries",
                "openalex_institution_types", "openalex_author_ids",
                "openalex_keywords"):
        if col not in df.columns:
            df[col] = None

    for col in ("openalex_topic", "openalex_subfield"):
        if col not in df.columns:
            df[col] = "Unknown"

    if "openalex_fetched" not in df.columns:
        df["openalex_fetched"] = False

    for idx, row in df.iterrows():
        aid = row["id"]
        if aid not in openalex:
            continue                   # 404 / error — leave openalex_fetched=False
        data = openalex[aid]
        df.at[idx, "openalex_institution_names"]     = data["institution_names"]
        df.at[idx, "openalex_institution_countries"] = data["institution_countries"]
        df.at[idx, "openalex_institution_types"]     = data["institution_types"]
        df.at[idx, "openalex_author_ids"]            = data.get("author_ids", [])
        df.at[idx, "openalex_topic"]                 = data.get("openalex_topic", "Unknown")
        df.at[idx, "openalex_subfield"]              = data.get("openalex_subfield", "Unknown")
        df.at[idx, "openalex_keywords"]              = data.get("openalex_keywords", [])
        df.at[idx, "openalex_fetched"]               = True

    return df



# ──────────────────────────────────────────────
# 6a. AUTHOR SENIORITY (Step 2)
#
# Fetches career citation counts for each unique author across the corpus.
# Authors are deduplicated so shared authors are fetched only once per run.
# Results are cached in-memory; nothing extra is written to parquet beyond
# the per-paper author_seniority column.
#
# Seniority tiers (based on the highest-ranked author on the paper):
#   Established  — cited_by_count ≥ 1000  OR  works_count ≥ 50
#   Emerging     — cited_by_count ≥ 100   OR  works_count ≥ 10
#   Unknown      — no OpenAlex author data available
#
# Reputation impact: +2 points for at least one Established author.
# ──────────────────────────────────────────────

# Seniority thresholds
ESTABLISHED_CITATIONS = 1000
ESTABLISHED_WORKS     = 50
EMERGING_CITATIONS    = 100
EMERGING_WORKS        = 10


def fetch_author_stats(author_ids: list) -> dict:
    """
    Fetch cited_by_count and works_count for a list of OpenAlex author IDs.

    Deduplicates before fetching so shared authors (common in cs.AI) are
    only queried once. Returns a dict mapping author_id → stats dict.
    IDs that fail silently return no entry (seniority falls back to Unknown).
    """
    if not author_ids:
        return {}

    try:
        import pyalex
        pyalex.config.email = OPENALEX_EMAIL
    except ImportError:
        return {}

    unique_ids = list(dict.fromkeys(author_ids))   # deduplicate, preserve order
    stats: dict[str, dict] = {}
    errors = 0

    print(f"  Fetching author stats for {len(unique_ids)} unique authors "
          f"(~{len(unique_ids) * OPENALEX_SLEEP:.0f}s)...")

    for i, author_id in enumerate(unique_ids):
        try:
            author = pyalex.Authors()[author_id]
            stats[author_id] = {
                "cited_by_count": author.get("cited_by_count", 0) or 0,
                "works_count":    author.get("works_count", 0)    or 0,
            }
        except Exception as exc:
            err_str = str(exc).lower()
            if "404" not in err_str and "not found" not in err_str:
                errors += 1
        time.sleep(OPENALEX_SLEEP)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(unique_ids)} authors fetched")

    print(f"  Author stats: {len(stats)} retrieved | "
          f"{len(unique_ids) - len(stats) - errors} not found | {errors} errors.")
    return stats


def compute_author_seniority(author_ids: list, stats: dict) -> str:
    """
    Return the seniority tier of the most senior author on a paper.
    Tiers: Established > Emerging > Unknown.
    """
    if not author_ids or not stats:
        return "Unknown"

    best = "Unknown"
    for aid in author_ids:
        s = stats.get(aid)
        if not s:
            continue
        if (s["cited_by_count"] >= ESTABLISHED_CITATIONS or
                s["works_count"] >= ESTABLISHED_WORKS):
            return "Established"   # can't do better — short-circuit
        if (s["cited_by_count"] >= EMERGING_CITATIONS or
                s["works_count"] >= EMERGING_WORKS):
            best = "Emerging"

    return best


def load_author_stats_cache() -> dict:
    """Load persisted author stats from disk. Returns empty dict if not found."""
    if os.path.exists(AUTHOR_STATS_CACHE):
        try:
            with open(AUTHOR_STATS_CACHE) as f:
                cache = json.load(f)
                print(f"  Author stats cache: loaded {len(cache)} entries from {AUTHOR_STATS_CACHE}.")
                return cache
        except Exception:
            pass
    return {}


def save_author_stats_cache(cache: dict) -> None:
    """Persist author stats to disk so they survive between runs."""
    with open(AUTHOR_STATS_CACHE, "w") as f:
        json.dump(cache, f)


def apply_author_seniority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect all author IDs from papers that have openalex_author_ids,
    fetch stats only for authors NOT already in the on-disk cache,
    then write author_seniority for every row.

    The cache (author_stats_cache.json) persists between runs so authors
    seen in previous runs are never re-queried — same pattern as run_state.json
    for arXiv fetches.
    """
    if "openalex_author_ids" not in df.columns:
        df["author_seniority"] = "Unknown"
        return df

    # Gather all unique author IDs across the corpus
    all_author_ids = []
    for ids in df["openalex_author_ids"]:
        if isinstance(ids, list):
            all_author_ids.extend(ids)

    if not all_author_ids:
        df["author_seniority"] = "Unknown"
        return df

    unique_ids = list(dict.fromkeys(all_author_ids))

    # Load cache and only fetch IDs we haven't seen before
    cache = load_author_stats_cache()
    uncached = [aid for aid in unique_ids if aid not in cache]

    if uncached:
        print(f"  Author stats: {len(unique_ids) - len(uncached)} cached, "
              f"{len(uncached)} new to fetch.")
        new_stats = fetch_author_stats(uncached)
        cache.update(new_stats)
        save_author_stats_cache(cache)
    else:
        print(f"  Author stats: all {len(unique_ids)} authors in cache — skipping API calls.")

    # Compute per-paper seniority using the full cache
    df = df.copy()
    if "author_seniority" not in df.columns:
        df["author_seniority"] = "Unknown"

    for idx, row in df.iterrows():
        aids = row.get("openalex_author_ids")
        if isinstance(aids, list) and aids:
            df.at[idx, "author_seniority"] = compute_author_seniority(aids, cache)

    est  = (df["author_seniority"] == "Established").sum()
    emg  = (df["author_seniority"] == "Emerging").sum()
    unk  = (df["author_seniority"] == "Unknown").sum()
    print(f"  Author seniority: {est} Established | {emg} Emerging | {unk} Unknown.")
    return df



# ──────────────────────────────────────────────
# 7. COUNTRY & INSTITUTION TYPE DIMENSIONS (Steps 3 & 4)
#
# Both fields are already stored in the parquet from the Step 1 fetch:
#   openalex_institution_countries  list[str]  — ISO-2 codes, parallel to names
#   openalex_institution_types      list[str]  — education/company/government/nonprofit
#
# We derive single scalar columns for Embedding Atlas color-by:
#   institution_country  — most common country among the paper's institutions,
#                          with a simple majority rule; "Unknown" if no data.
#   institution_type     — most common type; "Unknown" if no data.
#
# "Most common" rather than "first" avoids over-weighting one author's
# affiliation on large multi-institution papers.
# ──────────────────────────────────────────────

def most_common(values: list) -> str:
    """Return the most frequent non-empty value in a list, or 'Unknown'."""
    counts: dict[str, int] = {}
    for v in values:
        if v and isinstance(v, str):
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else "Unknown"


def apply_geo_and_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive scalar institution_country and institution_type columns from
    the list columns written by apply_openalex_columns.
    Rows without OpenAlex data get 'Unknown' for both.
    """
    df = df.copy()

    # Ensure source list columns exist (graceful on pre-Step-1 parquet)
    for col in ("openalex_institution_countries", "openalex_institution_types"):
        if col not in df.columns:
            df[col] = None

    df["institution_country"] = df["openalex_institution_countries"].apply(
        lambda v: most_common(v) if isinstance(v, list) else "Unknown"
    )
    df["institution_type"] = df["openalex_institution_types"].apply(
        lambda v: most_common(v) if isinstance(v, list) else "Unknown"
    )

    # Summary for the run log
    countries = df["institution_country"].value_counts()
    top_countries = ", ".join(
        f"{c} ({n})" for c, n in countries.head(5).items() if c != "Unknown"
    )
    types = df["institution_type"].value_counts()
    type_summary = ", ".join(
        f"{t} ({n})" for t, n in types.items() if t != "Unknown"
    )
    unknown_c = (df["institution_country"] == "Unknown").sum()
    unknown_t = (df["institution_type"]    == "Unknown").sum()
    print(f"  Institution countries: top 5 — {top_countries or 'none yet'} "
          f"| {unknown_c} Unknown.")
    print(f"  Institution types: {type_summary or 'none yet'} "
          f"| {unknown_t} Unknown.")
    return df


# ──────────────────────────────────────────────
# 8. INCREMENTAL EMBEDDING
#    Embeds only papers missing vectors, then re-projects
#    ALL papers with UMAP for a globally coherent layout.
# ──────────────────────────────────────────────
def embed_and_project(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    import umap as umap_lib

    model = SentenceTransformer(EMBEDDING_MODEL_ID)

    if "embedding" in df.columns:
        needs_embed = df["embedding"].isna()
    else:
        df["embedding"] = None
        needs_embed = pd.Series([True] * len(df))

    n_new = needs_embed.sum()
    if n_new:
        print(f"  Embedding {n_new} new paper(s) with SPECTER2...")
        idx     = df.index[needs_embed].tolist()
        texts   = df.loc[idx, "text"].tolist()
        vectors = model.encode(texts, show_progress_bar=True, batch_size=16,
                               convert_to_numpy=True)
        # Assign row-by-row using integer positions to avoid pandas mixed-type
        # issues when setting list values into a column containing None.
        for i, pos in enumerate(idx):
            df.at[pos, "embedding"] = vectors[i].tolist()
    else:
        print("  All papers already embedded — skipping SPECTER2.")

    all_vectors = np.array(df["embedding"].tolist(), dtype=np.float32)
    n = len(all_vectors)
    print(f"  Projecting {n} papers with UMAP (two-stage)...")

    # ── Stage 1: 768D → 50D (for clustering) ────────────────────────────
    # Reducing to 50D preserves far more structure than going straight to 2D.
    # HDBSCAN then clusters in this richer space using cosine metric, which
    # is appropriate for embedding vectors where direction > magnitude.
    # The 50D coords are stored in the parquet so they don't need recomputing
    # on every run — only new papers trigger a full re-projection.
    reducer_50d = umap_lib.UMAP(n_components=50, metric="cosine",
                                random_state=42, n_neighbors=15)
    coords_50d = reducer_50d.fit_transform(all_vectors)
    # Store as a flat list so pyarrow can serialise it into parquet.
    df["embedding_50d"] = [row.tolist() for row in coords_50d]

    # ── Stage 2: 768D → 2D (for display only) ───────────────────────────
    # min_dist controls point spread: lower = tighter clusters visually.
    reducer_2d = umap_lib.UMAP(n_components=2, metric="cosine",
                               random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer_2d.fit_transform(all_vectors)
    df["projection_x"] = coords_2d[:, 0].astype(float)
    df["projection_y"] = coords_2d[:, 1].astype(float)
    return df


# ──────────────────────────────────────────────
# 9. CLUSTER LABEL GENERATION
#
# CURRENT APPROACH: Vocabulary-first with KeyBERT fallback
#   For each cluster, compute the mean of its SPECTER2 embeddings (centroid)
#   and find the closest match in the pre-built OpenAlex topic vocabulary
#   (vocab_embeddings.npz). This produces clean, consistent, human-curated
#   labels like "Reinforcement Learning" or "Natural Language Processing"
#   rather than extracted phrases like "reward shaping policy".
#
#   If the best vocabulary match scores below MIN_VOCAB_CONFIDENCE (cosine
#   similarity), the cluster falls back to KeyBERT extraction. This handles
#   niche or emerging topics not well-covered by the OpenAlex taxonomy.
#
# PREVIOUS APPROACH: KeyBERT with abstracts as input
#   Re-enable by setting LABEL_MODE = "keybert" below, or by removing
#   vocab_embeddings.npz from the repo (the code gracefully falls back).
#
# VOCAB FILE: data/vocab_embeddings.npz
#   Downloaded from the label-vocabulary-builder repo as part of the
#   GitHub Actions workflow (see .github/workflows/update_map.yml).
#   Contains ~300 pre-embedded CS/AI topic names from OpenAlex.
# ──────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL SETTINGS
# ════════════════════════════════════════════════════════════════════
#
# ── EMBEDDING_MODEL (str) ────────────────────────────────────────────
# Which SentenceTransformer model to use for SPECTER2 embeddings.
# All three options output 768D vectors, so no downstream changes are
# needed — vocab_embeddings.npz, UMAP, and HDBSCAN all remain the same.
#
# IMPORTANT: vocab_embeddings.npz was built with specter2_base.
# If you switch to mpnet, the vocab centroid matching will be less
# accurate (different embedding space). Either:
#   a) Set LABEL_MODE = "keybert" when using mpnet, or
#   b) Regenerate vocab_embeddings.npz with the new model via
#      the label-vocabulary-builder repo.
#
# Options:
#   "specter2_base" — allenai/specter2_base (current default)
#                     Scientific domain, fast, well-tested here.
#                     768D output. vocab_embeddings.npz was built with this.
#   "mpnet"         — sentence-transformers/all-mpnet-base-v2
#                     General-purpose, not science-specific.
#                     Strong semantic similarity benchmark performance.
#                     768D output — no downstream dimension changes needed.
#
# NOTE: allenai/specter2 (adapter version) has a known PEFT compatibility
# issue with recent sentence-transformers and is excluded.
#
# Recommended experiment: try "mpnet" vs "specter2_base" to test whether
# scientific domain specialisation actually helps your clustering quality.
EMBEDDING_MODEL = "specter2_base"

_EMBEDDING_MODEL_IDS = {
    "specter2_base": "allenai/specter2_base",
    "mpnet":         "sentence-transformers/all-mpnet-base-v2",
}

if EMBEDDING_MODEL not in _EMBEDDING_MODEL_IDS:
    raise ValueError(
        f"Unknown EMBEDDING_MODEL '{EMBEDDING_MODEL}'. "
        f"Choose from: {list(_EMBEDDING_MODEL_IDS)}"
    )

EMBEDDING_MODEL_ID = _EMBEDDING_MODEL_IDS[EMBEDDING_MODEL]
print(f"▶  Embedding model: {EMBEDDING_MODEL_ID}")

# ════════════════════════════════════════════════════════════════════
# LABELING SETTINGS
# ════════════════════════════════════════════════════════════════════
#
# ── LABEL_MODE (str) ─────────────────────────────────────────────────
# Controls how cluster labels are generated.
#
# "vocab"      — OpenAlex ~300-topic vocabulary loaded from vocab_embeddings.npz.
#                Falls back to KeyBERT when cosine similarity < MIN_VOCAB_CONFIDENCE.
#                Produces specific, granular labels. Requires the .npz file.
#
# "curated_20" — 20 hand-curated categories covering modern arXiv/cs.AI research.
#                Embeddings are computed at runtime from CURATED_20_LABELS below
#                (takes ~1s for 20 strings; no .npz file needed).
#                Always assigns the best-matching category — no KeyBERT fallback.
#                The threshold (MIN_CURATED_CONFIDENCE) is intentionally low (0.3)
#                since every cluster should map to one of the 20 categories.
#                Use this to see broad thematic structure across the corpus.
#
# "keybert"    — KeyBERT extraction on cluster abstracts for all clusters.
#                Produces phrase-level labels ("reward shaping policy") rather
#                than topic names. No vocab file required. Original approach.
#
# Parallel comparison: run with "curated_20" and "vocab" on alternate days
# (or use EMBEDDING_MODE=full to force a fresh run) and compare label quality
# in the diagnostics JSON.
LABEL_MODE = "vocab"

# ── CURATED_20_LABELS ─────────────────────────────────────────────────
# 20 hand-curated categories for modern arXiv cs.AI research (Feb 2026).
# Used only when LABEL_MODE = "curated_20".
# Edit freely — these are embedded at runtime with the active EMBEDDING_MODEL,
# so no .npz regeneration is needed after changes.
#
# Grouped by theme for readability; order does not affect matching.
CURATED_20_LABELS = [
    # Core Model Architectures & Training
    "Foundation Models and Scaling Laws",
    "State Space Models",
    "Mixture of Experts",
    "Neural ODEs and Continuous-Time Models",
    # Learning Paradigms
    "Agentic AI and Autonomous Agents",
    "Reinforcement Learning from Human Feedback",
    "Multi-Agent Reinforcement Learning",
    "Self-Supervised and Contrastive Learning",
    # Specialized AI Capabilities
    "Multimodal Fusion",
    "Retrieval-Augmented Generation",
    "Mathematical Reasoning and Theorem Proving",
    "AI for Code and Software Engineering",
    # Ethics, Safety & Evaluation
    "Benchmark Design and Evaluation Methodology",
    "Explainability and Interpretability",
    "Adversarial Robustness and AI Safety",
    "AI Governance and Policy",
    # Hardware & Deployment
    "On-Device and Edge AI",
    "Efficient Inference and Quantization",
    # Domain-Specific Applications
    "AI for Science",
    "Medical and Clinical AI",
]

# ── MIN_VOCAB_CONFIDENCE (float, default: 0.75) ──────────────────────
# Minimum cosine similarity to accept a vocabulary match (LABEL_MODE="vocab").
# Higher → fewer vocab labels, more KeyBERT fallbacks (more conservative).
# Lower  → more vocab labels, accepts weaker matches.
# Recommended range: 0.65–0.85.
MIN_VOCAB_CONFIDENCE = 0.85

# ── MIN_CURATED_CONFIDENCE (float, default: 0.30) ────────────────────
# Minimum cosine similarity to accept a curated-20 match (LABEL_MODE="curated_20").
# Intentionally low — with only 20 categories every cluster should match one.
# Raise above 0.5 only if you want genuine mismatches to fall back to KeyBERT.
# Range: 0.20–0.50.
MIN_CURATED_CONFIDENCE = 0.30

# ── VOCAB_EMBEDDINGS_PATH (str) ──────────────────────────────────────
# Path to the pre-built OpenAlex vocabulary embeddings file.
# Used only when LABEL_MODE = "vocab".
# Generated by the label-vocabulary-builder repo.
VOCAB_EMBEDDINGS_PATH = "data/vocab_embeddings.npz"
# ════════════════════════════════════════════════════════════════════


def _load_vocab_embeddings(path: str):
    """
    Load pre-computed vocabulary embeddings from disk.
    Returns (embeddings, labels) or (None, None) if file not found or invalid.
    Embeddings are L2-normalised 768-dim SPECTER2 vectors, shape (N, 768).
    Used only when LABEL_MODE = "vocab".
    """
    if not os.path.exists(path):
        print(f"  [vocab] {path} not found — will use KeyBERT for all clusters.")
        return None, None
    try:
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["labels"].tolist()
        print(f"  [vocab] Loaded {len(labels)} vocabulary entries from {path}.")
        return embeddings, labels
    except Exception as e:
        print(f"  [vocab] Failed to load {path} ({e}) — will use KeyBERT for all clusters.")
        print(f"  [vocab] Run the label-vocabulary-builder workflow to regenerate vocab_embeddings.npz.")
        return None, None


def _build_curated_embeddings(labels: list, model) -> np.ndarray:
    """
    Embed CURATED_20_LABELS at runtime using the active SentenceTransformer model.
    Returns an L2-normalised (N, D) float32 matrix ready for cosine similarity.
    Called once per run when LABEL_MODE = "curated_20"; takes ~1s for 20 strings.
    """
    print(f"  [curated_20] Embedding {len(labels)} curated category labels at runtime...")
    vectors = model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)
    # L2-normalise each row so dot product == cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vectors = vectors / norms
    print(f"  [curated_20] Done. Embedding shape: {vectors.shape}")
    return vectors


def _vocab_label_for_centroid(
    centroid: np.ndarray,
    vocab_embeddings: np.ndarray,
    vocab_labels: list,
    min_confidence: float,
) -> tuple[str | None, float, list]:
    """
    Find the closest vocabulary entry to a cluster centroid via cosine similarity.

    Args:
        centroid:         Mean of the cluster's raw 768D SPECTER2 embeddings.
        vocab_embeddings: (N, 768) matrix of L2-normalised vocab embeddings.
        vocab_labels:     Corresponding label strings, length N.
        min_confidence:   Minimum cosine similarity to accept a match.

    Returns:
        (label, score, top5_candidates)
        label is None if best score < min_confidence (triggers KeyBERT fallback).
    """
    # L2-normalise the centroid for cosine similarity via dot product
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    scores = vocab_embeddings @ centroid.astype(np.float32)
    top_idx = np.argsort(scores)[::-1][:5]
    candidates = [(vocab_labels[i], float(scores[i])) for i in top_idx]

    best_label, best_score = candidates[0]
    if best_score < min_confidence:
        return None, best_score, candidates
    return best_label, best_score, candidates


def generate_keybert_labels(df: pd.DataFrame) -> tuple:
    """
    Cluster papers with HDBSCAN, then label each cluster.

    Labeling strategy (controlled by LABEL_MODE above):
      1. Compute the cluster centroid from raw 768D SPECTER2 embeddings.
      2. Find the closest OpenAlex topic via cosine similarity.
      3. If confidence ≥ MIN_VOCAB_CONFIDENCE → use vocab label.
      4. Otherwise → fall back to KeyBERT on the cluster's abstracts.

    Writes labels.parquet and adds a cluster_label column to df.
    Returns (df, labels_path).

    cluster_label is written into the parquet so the embedding-atlas
    auto-labeler uses it as its text source via --text cluster_label.
    All papers in a cluster share the same label string, so TF-IDF
    correctly extracts the cluster's topic name rather than random phrases.
    """
    from keybert import KeyBERT
    from sklearn.cluster import HDBSCAN

    print("  Generating cluster labels...")

    # 2D coords used only for placing label positions on the map.
    coords_2d = df[["projection_x", "projection_y"]].values.astype(np.float64)

    # ── Clustering input: 50D cosine space ──────────────────────────────
    # HDBSCAN clusters on 50D UMAP projections (cosine metric) for semantic
    # fidelity. The 768D vectors are used separately for centroid matching.
    #
    # ALTERNATIVE A — cluster on raw 768D SPECTER2 vectors:
    #   To enable: replace the block below with:
    #     all_vectors = np.array(df["embedding"].tolist(), dtype=np.float32)
    #     cluster_input = all_vectors
    #     cluster_metric = "cosine"
    #
    # ALTERNATIVE B — cluster on 2D display coords (original approach):
    #   To enable: replace the block below with:
    #     cluster_input = coords_2d
    #     cluster_metric = "euclidean"
    if "embedding_50d" in df.columns and df["embedding_50d"].notna().all():
        cluster_input = np.array(df["embedding_50d"].tolist(), dtype=np.float32)
        cluster_metric = "cosine"
        print("  Clustering on 50D cosine space (high-fidelity).")
    else:
        cluster_input = coords_2d
        cluster_metric = "euclidean"
        print("  Clustering on 2D coords (fallback — embedding_50d not available).")

    # ── HDBSCAN clustering settings ─────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════
    # HDBSCAN SETTINGS — all changes take effect on the next build.
    # These settings do not affect database.parquet in any way.
    # ════════════════════════════════════════════════════════════════════
    #
    # ── min_cluster_size (int, default: adaptive) ────────────────────────
    # Minimum papers required to form a cluster.
    # Lower  → more clusters, more specific labels.
    # Higher → fewer clusters, more general labels.
    # Syntax:   HDBSCAN(min_cluster_size=5, ...)
    # Range:    3–15 recommended for a ~300-paper corpus.
    #
    # ── min_samples (int, default: 4) ───────────────────────────────────
    # Lower  → more points pulled into clusters, fewer noise points.
    # Higher → stricter assignment, more noise points.
    # Syntax:   HDBSCAN(..., min_samples=4, ...)
    # Range:    1–5 recommended.
    #
    # ── cluster_selection_method (str, default: "leaf") ──────────────────
    # "leaf" — smaller, more granular clusters (current).
    # "eom"  — larger clusters of varying sizes.
    # Syntax:   HDBSCAN(..., cluster_selection_method="leaf")
    #
    # ── cluster_selection_epsilon (float, default: 0.0) ─────────────────
    # Merges clusters closer than this threshold. Raise to merge tiny clusters.
    # Syntax:   HDBSCAN(..., cluster_selection_epsilon=0.5)
    # Range:    0.0 (off) to ~2.0.
    #
    # ── alpha (float, default: 1.0) ─────────────────────────────────────
    # Higher → fewer, more stable clusters.
    # Lower  → more splits.
    # Syntax:   HDBSCAN(..., alpha=1.0)
    # Range:    0.5–2.0 typical.
    # ════════════════════════════════════════════════════════════════════
    clusterer = HDBSCAN(
        min_cluster_size=max(5, len(df) // 80),
        min_samples=2,
        metric=cluster_metric,
        cluster_selection_method="leaf",
    )
    cluster_ids = clusterer.fit_predict(cluster_input)

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"  Found {n_clusters} clusters (noise points excluded).")
    print(f"  Label mode: {LABEL_MODE}")

    # ── Raw 768D vectors — needed for centroid matching in all vocab modes ──
    # These preserve the full semantic structure that UMAP compresses away.
    all_768d = None
    if "embedding" in df.columns and LABEL_MODE in ("vocab", "curated_20"):
        all_768d = np.array(df["embedding"].tolist(), dtype=np.float32)

    # ── Load / build vocab embeddings based on LABEL_MODE ───────────────────
    vocab_embeddings, vocab_labels = None, None
    curated_confidence = MIN_CURATED_CONFIDENCE

    if LABEL_MODE == "vocab":
        # Load pre-built OpenAlex ~300-topic vocabulary from disk.
        vocab_embeddings, vocab_labels = _load_vocab_embeddings(VOCAB_EMBEDDINGS_PATH)

    elif LABEL_MODE == "curated_20":
        # Embed the 20 hand-curated category strings at runtime.
        # Uses the same model as paper embeddings so the spaces are aligned.
        from sentence_transformers import SentenceTransformer as _ST
        _model = _ST(EMBEDDING_MODEL_ID)
        vocab_embeddings = _build_curated_embeddings(CURATED_20_LABELS, _model)
        vocab_labels = CURATED_20_LABELS
        del _model   # free VRAM before KeyBERT loads (KeyBERT reloads it internally)

    # ── KeyBERT — initialised once; used as fallback for "vocab" and "keybert" mode
    # In "curated_20" mode it is only used when a cluster scores below
    # MIN_CURATED_CONFIDENCE (rare given only 20 broad categories).
    kw_model = KeyBERT(model=EMBEDDING_MODEL_ID)

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    KEYBERT_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
        # Generic AI/ML paper boilerplate — common across ALL clusters
        "model", "models", "modeling", "modeled",
        "paper", "propose", "proposed", "approach",
        "method", "methods", "task", "tasks", "performance", "results",
        "result", "work", "framework", "system", "learning", "deep",
        "based", "using", "show", "new", "novel", "training", "dataset",
        "data", "benchmark", "improve", "improved", "state", "art",
        "effective", "efficient", "robust", "demonstrate", "achieve",
        # Abstract-specific boilerplate
        "present", "introduce", "existing", "recent", "large",
        "address", "problem", "challenge", "issue", "key",
        "conduct", "evaluate", "evaluation", "study", "studies",
        "experiment", "experiments", "experimental", "empirical",
        "outperform", "outperforms", "baseline", "baselines",
        "significant", "significantly", "extensive", "comprehensive",
    ]

    vocab_count = 0
    curated_count = 0
    keybert_count = 0
    label_rows = []
    diagnostics = []   # written to label_diagnostics.json for inspection

    # Confidence threshold varies by mode
    active_confidence = (
        MIN_CURATED_CONFIDENCE if LABEL_MODE == "curated_20"
        else MIN_VOCAB_CONFIDENCE
    )

    for cid in sorted(set(cluster_ids)):
        if cid == -1:
            continue  # HDBSCAN noise points get no label

        mask = cluster_ids == cid
        label_text = None
        diag = {
            "cluster_id":    int(cid),
            "paper_count":   int(mask.sum()),
            "method":        None,
            "label":         None,
            "vocab_score":   None,
            "vocab_top10":   [],
            "keybert_top3":  [],
            "sample_titles": df.loc[mask, "title"].tolist()[:5],
        }

        # ── Step 1: Centroid matching (vocab or curated_20 modes) ────────────
        if vocab_embeddings is not None and all_768d is not None:
            centroid = all_768d[mask].mean(axis=0)
            label_text, score, candidates = _vocab_label_for_centroid(
                centroid, vocab_embeddings, vocab_labels, active_confidence
            )
            diag["vocab_score"] = round(float(score), 4)
            diag["vocab_top10"] = [(t, round(float(s), 4)) for t, s in candidates[:10]]
            if label_text:
                method_tag = "curated_20" if LABEL_MODE == "curated_20" else "vocab"
                diag["method"] = method_tag
                print(f"  Cluster {cid} ({mask.sum()} papers) → {method_tag}: '{label_text}' "
                      f"(score={score:.3f})")
                print(f"    top5: {[(t, f'{s:.3f}') for t,s in candidates[:5]]}")
                if LABEL_MODE == "curated_20":
                    curated_count += 1
                else:
                    vocab_count += 1
            else:
                print(f"  Cluster {cid} ({mask.sum()} papers) → centroid too low "
                      f"(best={score:.3f}: '{candidates[0][0]}') → KeyBERT fallback")

        # ── Step 2: KeyBERT fallback ─────────────────────────────────────────
        # Always used when LABEL_MODE = "keybert".
        # Used as fallback when centroid score is too low in "vocab" / "curated_20".
        if label_text is None:
            abstracts = df.loc[mask, "abstract"].tolist()
            combined  = " ".join(abstracts)
            # ── KeyBERT keyword extraction settings ─────────────────────────
            # keyphrase_ngram_range: (min, max) words per phrase.
            #   (1, 2) → single words and bigrams (recommended).
            #   (2, 2) → bigrams only — specific but can produce odd pairings.
            #
            # use_mmr: Maximal Marginal Relevance reduces redundancy.
            #   True  → broader, more diverse terms.
            #   False → highest-scoring terms regardless of similarity.
            #
            # diversity: only applies when use_mmr=True. Range 0.0–1.0.
            #   Lower  → keywords closer to cluster centroid (more representative).
            #   Higher → more spread out.
            keywords = kw_model.extract_keywords(
                combined,
                keyphrase_ngram_range=(2, 2),
                stop_words=KEYBERT_STOP_WORDS,
                use_mmr=True,
                diversity=0.3,
                top_n=3,
            )
            print(f"  Cluster {cid} ({mask.sum()} papers) → keybert: {keywords}")
            if not keywords:
                diag["method"] = "keybert_empty"
                diagnostics.append(diag)
                continue
            label_text = keywords[0][0].title()
            diag["keybert_top3"] = [(kw, round(float(sc), 4)) for kw, sc in keywords]
            diag["method"] = "keybert"
            keybert_count += 1

        diag["label"] = label_text
        diagnostics.append(diag)
        # Tag each paper in this cluster with the label (used below for --text)
        df.loc[df.index[mask], "_cluster_label_tmp"] = label_text
        cx = float(coords_2d[mask, 0].mean())
        cy = float(coords_2d[mask, 1].mean())
        label_rows.append({"x": cx, "y": cy, "text": label_text})

    print(f"  Labeling summary: {vocab_count} vocab | {curated_count} curated_20 | "
          f"{keybert_count} KeyBERT fallback.")

    # ── Write diagnostics report ─────────────────────────────────────────────
    # label_diagnostics.json shows exactly what happened for every cluster:
    #   - which label was chosen and why (vocab / curated_20 / keybert)
    #   - the top 10 candidates with cosine scores
    #   - the top 3 KeyBERT keywords (when used as fallback)
    #   - 5 sample paper titles from the cluster
    # Use this to compare LABEL_MODE options and tune confidence thresholds.
    diag_path = "label_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump({
            "summary": {
                "label_mode":      LABEL_MODE,
                "total_clusters":  len(diagnostics),
                "vocab_labels":    vocab_count,
                "curated_labels":  curated_count,
                "keybert_labels":  keybert_count,
                "min_confidence":  active_confidence,
                "vocab_file":      VOCAB_EMBEDDINGS_PATH if LABEL_MODE == "vocab" else "n/a",
                "vocab_loaded":    vocab_embeddings is not None,
            },
            "clusters": diagnostics,
        }, f, indent=2)
    print(f"  Wrote diagnostics to {diag_path} — inspect to tune labeling.")

    # ── Write cluster_label column back into df ──────────────────────────
    # Each paper gets the label of its cluster. Noise points (HDBSCAN -1)
    # get an empty string. This column is passed to the embedding-atlas CLI
    # via --text so its built-in TF-IDF auto-labeler sees identical strings
    # for all papers in a cluster and correctly extracts the topic name.
    if "_cluster_label_tmp" in df.columns:
        df["cluster_label"] = df["_cluster_label_tmp"].fillna("").astype(str)
        df = df.drop(columns=["_cluster_label_tmp"])
    else:
        df["cluster_label"] = ""

    labels_df = pd.DataFrame(label_rows)
    labels_path = "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    print(f"  Wrote {len(labels_df)} labels to {labels_path}.")
    return df, labels_path


# ──────────────────────────────────────────────
# 10. HTML POP-OUT PANEL
# ──────────────────────────────────────────────
def build_panel_html(run_date: str) -> str:
    return (
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
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
  #arm-tab {
    position:fixed; left:0; top:50%; transform:translateY(-50%);
    z-index:1000000; display:flex; align-items:center; justify-content:center;
    writing-mode:vertical-rl; text-orientation:mixed;
    background:rgba(15,23,42,0.90); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    color:var(--arm-accent); border:1px solid var(--arm-border); border-left:none;
    padding:18px 9px; border-radius:0 10px 10px 0; cursor:pointer;
    font-family:var(--arm-font); font-weight:600; font-size:12px;
    letter-spacing:0.12em; text-transform:uppercase;
    box-shadow:3px 0 20px rgba(0,0,0,0.4);
    transition:background 0.2s,color 0.2s,box-shadow 0.2s; user-select:none;
  }
  #arm-tab:hover { background:rgba(30,41,59,0.95); color:#93c5fd; box-shadow:4px 0 24px rgba(96,165,250,0.2); }
  #arm-panel {
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
  #arm-panel.arm-open { transform:translateY(-50%) translateX(0); }
  #arm-body { overflow-y:auto; overflow-x:hidden; padding:22px 20px 16px; flex:1; scrollbar-width:thin; scrollbar-color:#334155 transparent; }
  #arm-body::-webkit-scrollbar { width:4px; }
  #arm-body::-webkit-scrollbar-thumb { background:#334155; border-radius:4px; }
  .arm-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px; }
  .arm-title { font-size:15px; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em; line-height:1.3; margin:0; }
  .arm-title span { color:var(--arm-accent); }
  #arm-close { background:none; border:none; color:var(--arm-muted); cursor:pointer; font-size:17px; line-height:1; padding:2px 4px; border-radius:4px; transition:color 0.15s,background 0.15s; flex-shrink:0; margin-left:8px; }
  #arm-close:hover { color:#f1f5f9; background:rgba(255,255,255,0.07); }
  .arm-byline { font-size:12px; color:var(--arm-muted); margin-bottom:16px; }
  .arm-byline a { color:var(--arm-accent); text-decoration:none; font-weight:500; }
  .arm-byline a:hover { color:#93c5fd; text-decoration:underline; }
  .arm-divider { border:none; border-top:1px solid var(--arm-border); margin:14px 0; }
  .arm-section { font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#475569; margin:0 0 8px; }
  .arm-p { color:#94a3b8; margin:0 0 10px; font-size:12.5px; }
  .arm-p a { color:var(--arm-accent); text-decoration:none; }
  .arm-p a:hover { color:#93c5fd; text-decoration:underline; }
  .arm-tip { background:var(--arm-accent-dim); border:1px solid rgba(96,165,250,0.22); border-radius:8px; padding:10px 12px; font-size:12px; color:#bfdbfe; margin-bottom:12px; display:flex; gap:8px; align-items:flex-start; }
  .arm-tip-icon { flex-shrink:0; font-size:14px; margin-top:1px; }
  .arm-legend-row { display:flex; align-items:flex-start; gap:9px; margin-bottom:8px; font-size:12px; color:#94a3b8; }
  .arm-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; margin-top:4px; }
  .arm-dot-enhanced { background:#f59e0b; box-shadow:0 0 6px rgba(245,158,11,0.5); }
  .arm-dot-std { background:#6366f1; box-shadow:0 0 6px rgba(99,102,241,0.4); }
  .arm-legend-label { font-weight:600; color:#cbd5e1; }
  .arm-book { display:flex; align-items:center; gap:10px; background:rgba(255,255,255,0.03); border:1px solid var(--arm-border); border-radius:8px; padding:10px 12px; text-decoration:none; transition:background 0.2s,border-color 0.2s; margin-bottom:8px; }
  .arm-book:hover { background:rgba(96,165,250,0.07); border-color:rgba(96,165,250,0.3); }
  .arm-book-icon { font-size:22px; flex-shrink:0; }
  .arm-book-text { display:flex; flex-direction:column; }
  .arm-book-title { font-size:12px; font-weight:600; color:#e2e8f0; line-height:1.3; margin-bottom:2px; }
  .arm-book-sub { font-size:11px; color:var(--arm-accent); }
  #arm-footer { padding:10px 20px 14px; border-top:1px solid var(--arm-border); display:flex; align-items:center; gap:7px; flex-shrink:0; }
  .arm-status-dot { width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 6px rgba(34,197,94,0.7); flex-shrink:0; animation:arm-pulse 2.5s ease-in-out infinite; }
  @keyframes arm-pulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }
  .arm-status-text { font-size:11px; color:#475569; font-family:var(--arm-font); }
  .arm-status-text strong { color:#64748b; font-weight:500; }
</style>

<button id="arm-tab" onclick="document.getElementById('arm-panel').classList.toggle('arm-open')" aria-label="Open info panel">About</button>

<div id="arm-panel" role="complementary" aria-label="About this atlas">
  <div id="arm-body">
    <div class="arm-header">
      <p class="arm-title">The <span>AI Research</span> Atlas</p>
      <button id="arm-close" onclick="document.getElementById('arm-panel').classList.remove('arm-open')" aria-label="Close panel">&#x2715;</button>
    </div>
    <div class="arm-byline">By <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" rel="noopener">Lee Fischman</a></div>

    <p class="arm-p">A live semantic atlas of recent AI research from arXiv (cs.AI), rebuilt daily. Each point is a paper. Nearby points share similar topics &mdash; clusters surface naturally from the embedding space and are labelled by their most distinctive terms.</p>
    <p class="arm-p">Powered by <a href="https://apple.github.io/embedding-atlas/" target="_blank" rel="noopener">Apple Embedding Atlas</a> and SPECTER2 scientific embeddings.</p>

    <hr class="arm-divider">

    <div class="arm-tip"><span class="arm-tip-icon">&#x1F4A1;</span><span>Set color to <strong>Reputation</strong> to see higher reputation scoring.</span></div>

  <div class="arm-tip"><span class="arm-tip-icon">&#x1F4A1;</span><span>Set color to <strong>Author Tier</strong>; more authors tends to be better.</span></div>

  <div class="arm-tip"><span class="arm-tip-icon">&#x1F4A1;</span><span>Set color to <strong>author_seniority</strong> to highlight papers with established researchers.</span></div>

  <div class="arm-tip"><span class="arm-tip-icon">&#x1F310;</span><span>Set color to <strong>institution_country</strong> to see the geographic spread of AI research.</span></div>

  <div class="arm-tip"><span class="arm-tip-icon">&#x1F3DB;</span><span>Set color to <strong>institution_type</strong> to compare academic vs industry research.</span></div>

  <div class="arm-tip"><span class="arm-tip-icon">&#x1F9EC;</span><span>Set color to <strong>openalex_subfield</strong> to see research areas from the OpenAlex taxonomy.</span></div>

    <hr class="arm-divider">

    <p class="arm-section">Books by the author</p>
    <a class="arm-book" href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" rel="noopener">
      <span class="arm-book-icon">&#x1F4D8;</span>
      <span class="arm-book-text"><span class="arm-book-title">Building Deep Learning Products</span><span class="arm-book-sub">Available on Amazon &#x2192;</span></span>
    </a>

    <hr class="arm-divider">

    <p class="arm-section">How to use</p>
    <p class="arm-p">Click any point to read its abstract and open the PDF on arXiv. Use the search bar to find papers by keyword or phrase. Drag to pan; scroll or pinch to zoom.</p>
  </div>
  <div id="arm-footer">
    <div class="arm-status-dot"></div>
    <span class="arm-status-text">Last updated <strong>""" + run_date + """ UTC</strong></span>
  </div>
</div>
"""
    )


# ──────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    clear_docs_contents("docs")

    now      = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    # Load & prune existing rolling DB
    existing_df  = load_existing_db()
    is_first_run = existing_df.empty and not os.path.exists(DB_PATH)
    days_back    = 15 if is_first_run else 2

    if is_first_run:
        print("  First run — pre-filling with last 5 days of arXiv papers.")
    else:
        print(f"  Loaded {len(existing_df)} existing papers from rolling DB.")

    # ── One-time migration: fix rows poisoned by the batch-400 bug ───────
    # An earlier build used a pre-filled result dict, causing rows that got
    # HTTP 400 errors to be marked openalex_fetched=True with empty institution
    # lists.  Reset any such rows to False so they are retried this run.
    # This guard is harmless once all rows have real data: the condition
    # (fetched=True AND empty names) will simply never match.
    if "openalex_fetched" in existing_df.columns and "openalex_institution_names" in existing_df.columns:
        bad_mask = (
            (existing_df["openalex_fetched"] == True) &
            (existing_df["openalex_institution_names"].apply(
                lambda v: not isinstance(v, list) or len(v) == 0
            ))
        )
        if bad_mask.any():
            existing_df.loc[bad_mask, "openalex_fetched"] = False
            print(f"  Migration: reset openalex_fetched for {bad_mask.sum()} rows "
                  f"with empty institution data (will re-query this run).")

    # Load run state — used to skip arXiv fetch if already ran today
    run_state   = load_run_state()
    today_date  = now.strftime("%Y-%m-%d")
    already_ran = run_state.get("last_fetch_date") == today_date

    if already_ran:
        print(f"  arXiv already fetched today ({today_date}) — skipping fetch, "
              f"reusing {len(existing_df)} papers from rolling DB.")
        df     = existing_df.copy()
        new_df = pd.DataFrame()   # empty — no new papers this run
    else:
        # Fetch from arXiv
        # Set a descriptive User-Agent as required by arXiv's API policy.
        # Without this, new clients may receive HTTP 406 rejections.
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent",
            "ai-research-atlas/1.0 (https://github.com/lfischman/ai-research-atlas; "
            "mailto:lee.fischman@gmail.com)")]
        urllib.request.install_opener(opener)

        client = arxiv.Client(page_size=100, delay_seconds=10)
        search = arxiv.Search(
            query=(
                f"cat:cs.AI AND submittedDate:"
                f"[{(now - timedelta(days=days_back)).strftime('%Y%m%d%H%M')}"
                f" TO {now.strftime('%Y%m%d%H%M')}]"
            ),
            max_results=ARXIV_MAX,
        )

        results = fetch_arxiv(client, search)
        if not results:
            print("  No results returned from arXiv. Skipping build.")
            exit(0)

        print(f"  Fetched {len(results)} papers from arXiv.")

        # Build new-papers DataFrame
        today_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for r in results:
            title    = r.title
            abstract = r.summary
            scrubbed = scrub_model_words(f"{title}. {title}. {abstract}")
            # label_text: title only (repeated for TF-IDF weight), used for cluster
            # labels in incremental mode where --text doesn't affect embeddings.
            label_text = scrub_model_words(f"{title}. {title}. {title}.")
            rows.append({
                "title":        title,
                "abstract":     abstract,
                "text":         scrubbed,
                "label_text":   label_text,
                "url":          r.pdf_url,
                "id":           r.entry_id.split("/")[-1],
                "author_count": len(r.authors),
                "author_tier":  categorize_authors(len(r.authors)),
                "date_added":   r.published.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

        new_df = pd.DataFrame(rows)

        # ── OpenAlex institution lookup ──────────────────────────────────────
        openalex_data = fetch_openalex_data(new_df["id"].tolist())
        new_df = apply_openalex_columns(new_df, openalex_data)
        new_df["Reputation"] = new_df.apply(calculate_reputation, axis=1)

        # Merge into rolling DB
        df = merge_papers(existing_df, new_df)
        df = df.drop(columns=["group"], errors="ignore")
        print(f"  Rolling DB: {len(df)} papers after merge.")

        # Record successful fetch so re-runs today skip arXiv
        save_run_state({**run_state, "last_fetch_date": today_date})

    # Ensure OpenAlex columns exist for all rows (including those loaded from
    # an older parquet that predates this schema).
    for col in ("openalex_institution_names", "openalex_institution_countries",
                "openalex_institution_types"):
        if col not in df.columns:
            df[col] = None
    if "openalex_fetched" not in df.columns:
        df["openalex_fetched"] = False

    # Backfill: query OpenAlex for any rows that haven't been fetched yet.
    # new_df is empty when arXiv was already fetched today, so this covers
    # all unfetched rows in that case too.
    new_ids        = set(new_df["id"].tolist()) if not new_df.empty else set()
    unfetched_mask = (df["openalex_fetched"] == False) & (~df["id"].isin(new_ids))
    unfetched_ids  = df.loc[unfetched_mask, "id"].tolist()
    if unfetched_ids:
        print(f"  Backfilling OpenAlex data for {len(unfetched_ids)} existing rows...")
        backfill_data = fetch_openalex_data(unfetched_ids)
        df = apply_openalex_columns(df, backfill_data)
        # Recalculate Reputation for rows that got new institution data
        newly_fetched = df["openalex_fetched"] & df["id"].isin(unfetched_ids)
        if newly_fetched.any():
            df.loc[newly_fetched, "Reputation"] = df.loc[newly_fetched].apply(
                calculate_reputation, axis=1
            )
            print(f"  Re-scored Reputation for {newly_fetched.sum()} backfilled rows.")

    # Backfill Reputation for older rows that predate the column.
    if "Reputation" not in df.columns or df["Reputation"].isna().any():
        missing = df["Reputation"].isna() if "Reputation" in df.columns else pd.Series([True] * len(df))
        if missing.any():
            print(f"  Backfilling Reputation for {missing.sum()} older rows...")
            df.loc[missing, "Reputation"] = df.loc[missing].apply(calculate_reputation, axis=1)

    # ── Author seniority (Step 2) ────────────────────────────────────────
    df = apply_author_seniority(df)

    # ── Country & institution type dimensions (Steps 3 & 4) ──────────────
    # No additional API calls needed — data is already in the parquet from
    # the Step 1 OpenAlex fetch. Just derives scalar columns from the lists.
    df = apply_geo_and_type_columns(df)

    # Re-score Reputation now that all signals are populated
    df["Reputation"] = df.apply(calculate_reputation, axis=1)

    # Embed & project (incremental mode only)
    labels_path = None
    if EMBEDDING_MODE == "incremental":
        print("  Incremental mode: embedding new papers only.")
        df = embed_and_project(df)
        df, labels_path = generate_keybert_labels(df)

    # Save rolling DB
    # Full mode: drop projection columns so CLI always recomputes a fresh layout.
    if EMBEDDING_MODE == "full":
        # Full mode: drop raw vectors so the CLI always recomputes a fresh layout.
        # cluster_label is kept — it's written by generate_keybert_labels in
        # incremental mode, so it won't exist in full mode (no-op).
        save_df = df.drop(columns=["embedding", "projection_x", "projection_y"], errors="ignore")
    else:
        save_df = df

    # Normalize date_added to a consistent ISO string.
    # load_existing_db() converts it to datetime; new rows arrive as strings.
    # After merge the column is mixed-type, which pyarrow refuses to serialize.
    save_df = save_df.copy()
    save_df["date_added"] = pd.to_datetime(save_df["date_added"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Normalize OpenAlex list columns: replace None/NaN with empty lists so
    # pyarrow can infer a consistent list[string] type for the column.
    for col in ("openalex_institution_names", "openalex_institution_countries",
                "openalex_institution_types", "openalex_author_ids",
                "openalex_keywords"):
        if col in save_df.columns:
            save_df[col] = save_df[col].apply(
                lambda v: v if isinstance(v, list) else []
            )
    # openalex_fetched: ensure bool (not object)
    if "openalex_fetched" in save_df.columns:
        save_df["openalex_fetched"] = save_df["openalex_fetched"].fillna(False).astype(bool)

    save_df.to_parquet(DB_PATH, index=False)
    print(f"  Saved {len(save_df)} papers to {DB_PATH}.")

    # Build the atlas
    print(f"  Building atlas ({EMBEDDING_MODE} mode)...")

    if EMBEDDING_MODE == "incremental":
        # Incremental mode: embeddings are pre-computed via --x/--y.
        # --text cluster_label feeds embedding-atlas's built-in TF-IDF
        # auto-labeler. Since every paper in a cluster shares the same label
        # string (e.g. all 32 papers in cluster 7 have "Reinforcement Learning
        # in Robotics"), TF-IDF extracts that phrase as the cluster label.
        # --labels also passes our positioned labels.parquet as a secondary
        # layer; remove it if it causes duplicate labels in the UI.
        atlas_cmd = [
            "embedding-atlas", DB_PATH,
            "--x",          "projection_x",
            "--y",          "projection_y",
            "--text",       "cluster_label",
            "--labels",     labels_path,
            "--export-application", "site.zip",
        ]
    else:
        # Full mode: --text feeds both SPECTER2 embeddings and TF-IDF labels,
        # so we keep the full scrubbed text to preserve embedding quality.
        # To improve label quality in full mode, switch to incremental mode.
        atlas_cmd = [
            "embedding-atlas", DB_PATH,
            "--text",       "text",
            "--model",      EMBEDDING_MODEL_ID,
            # "--stop-words", STOP_WORDS_PATH,  # omitted: NLTK default stop words are used
            "--export-application", "site.zip",
        ]

    subprocess.run(atlas_cmd, check=True)
    os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")

    # Config override
    config_path = "docs/data/metadata.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            conf = json.load(f)

        conf["name_column"]  = "title"
        conf["label_column"] = "title"
        conf["color_by"]     = "Reputation"

        # labelDensityThreshold controls how dense a cluster must be to receive
        # a floating label. Value is relative to the max density (0.0–1.0).
        # Raise it to suppress labels on small/sparse clusters.
        # Lower it to label more clusters including sparse ones.
        # conf["labelDensityThreshold"] = 0.1  # default is ~0.05

        # column_mappings: keys are parquet column names, values are display
        # names shown in the color dropdown and popup. Order controls dropdown order.
        conf["column_mappings"] = {
            "title":                      "Title",
            "abstract":                   "Abstract",
            "Reputation":                 "Reputation",
            "author_count":               "Author Count",
            "author_tier":                "Author Tier",
            "author_seniority":           "Seniority",
            "institution_country":        "Country",
            "institution_type":           "Institution Type",
            "openalex_subfield":          "Research Area",
            "openalex_topic":             "Topic",
            "openalex_keywords":          "Keywords",
            "url":                        "URL",
            "openalex_institution_names": "Institutions",
            "cluster_label":              "Research Topic",
        }

        with open(config_path, "w") as f:
            json.dump(conf, f, indent=4)
        print("  Config updated.")
    else:
        print("  docs/data/config.json not found — skipping config override.")

    # Inject pop-out panel
    index_file = "docs/index.html"
    if os.path.exists(index_file):
        panel_html = build_panel_html(run_date)
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("</body>", panel_html + "\n</body>") \
            if "</body>" in content else content + panel_html
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Info panel injected into index.html.")
    else:
        print("  docs/index.html not found — skipping panel injection.")

    print("  Atlas sync complete!")
