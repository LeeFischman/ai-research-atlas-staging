#!/usr/bin/env python3
# update_significant.py
# ──────────────────────────────────────────────────────────────────────────────
# AI Research Atlas — Significant Papers weekly maintenance script.
#
# Discovers and maintains a pool of highly-cited cs.AI papers from the last
# 15–180 days using the Semantic Scholar search API. Writes significant.parquet,
# which the daily pipeline (update_map_v2.py) loads and merges at Stage 1.
#
# Run schedule: Monday 06:00 UTC (0 6 * * 1) — see weekly_significant.yml.
# Also triggerable via workflow_dispatch for testing.
#
# Pipeline
# ────────
# 1. Load existing significant.parquet (if any)
# 2. Load database.parquet to get arXiv IDs already in the recent window
#    (those papers don't need to be duplicated in the significant pool)
# 3. Query S2 for top-cited cs.AI papers in the 15–180 day window,
#    sorted by citationCount:desc. Paginate until influentialCitationCount
#    of the last paper in a batch drops below SIGNIFICANT_MIN_INFLUENTIAL.
# 4. Keep top SIGNIFICANT_POOL_SIZE papers by influentialCitationCount.
# 5. Apply retirement logic (two-strike system; age ceiling at 180 days).
# 6. Enrich newly-added papers with author h-indices (OpenAlex) and
#    update ss_cache.json with S2 data already fetched during discovery.
# 7. Write significant.parquet.
#
# significant.parquet schema (mirrors database.parquet + one extra column):
#   id, title, abstract, text, label_text, url,
#   author_count, author_tier, authors_list, date_added,
#   author_hindices, Prominence,
#   ss_citation_count, ss_influential_citations, ss_tldr,
#   publication_date,
#   paper_source ("Significant"),
#   significant_strikes (int — consecutive absences from top-N)
#
# Columns added by the daily pipeline (not stored here):
#   embedding, embedding_50d, projection_x/y  (Stage 2)
#   group_id_v2, projection_v2_x/y             (Stages 3-4)
#   CitationTier                               (after Stage 1d)
#
# Normal run:
#   python update_significant.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from atlas_utils import (
    DB_PATH,
    scrub_model_words,
    categorize_authors,
    calculate_prominence,
    load_author_cache,
    save_author_cache,
    fetch_author_hindices,
    load_ss_cache,
    save_ss_cache,
    _arxiv_id_base,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SIGNIFICANT_PATH             = "significant.parquet"

# Pool size and pagination
SIGNIFICANT_POOL_SIZE        = 75    # max papers kept in the significant set
SIGNIFICANT_MIN_INFLUENTIAL  = 3     # stop paginating when page min drops below this
SIGNIFICANT_PAGE_SIZE        = 100   # papers per S2 API request
SIGNIFICANT_MAX_PAGES        = 10    # hard safety limit on pagination

# Date window
SIGNIFICANT_LOOKBACK_DAYS    = 180   # max age: papers older than this are retired
SIGNIFICANT_LOOKFORWARD_DAYS = 15    # min age: younger papers are in the recent window

# Retirement
SIGNIFICANT_STRIKES_LIMIT    = 2     # retire after this many consecutive absences

# S2 search
_S2_SEARCH_URL    = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_SEARCH_FIELDS = (
    "externalIds,title,abstract,authors,"
    "citationCount,influentialCitationCount,tldr,publicationDate"
)
# Broad AI query -- fieldsOfStudy + date range do the real filtering
_S2_SEARCH_QUERY  = (
    "artificial intelligence OR machine learning OR deep learning "
    "OR neural network OR large language model"
)


# ══════════════════════════════════════════════════════════════════════════════
# S2 DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _s2_search_page(
    offset: int,
    date_from_str: str,
    date_to_str: str,
    headers: dict,
) -> dict:
    """Fetch one page of S2 paper search results.

    Uses publicationDateOrYear for server-side date filtering and
    sort=citationCount:desc to front-load the most impactful papers.
    """
    params = {
        "query":                 _S2_SEARCH_QUERY,
        "fields":                _S2_SEARCH_FIELDS,
        "publicationDateOrYear": f"{date_from_str}:{date_to_str}",
        "fieldsOfStudy":         "Computer Science",
        "sort":                  "citationCount:desc",
        "limit":                 str(SIGNIFICANT_PAGE_SIZE),
        "offset":                str(offset),
    }
    url = f"{_S2_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def _parse_s2_paper(paper: dict) -> dict | None:
    """Extract a normalised paper dict from one S2 search result entry.

    Returns None if the paper has no arXiv ID or no publication date.
    """
    if not paper:
        return None

    external  = paper.get("externalIds") or {}
    # S2 uses "ArXiv" (capital X) for the arXiv field
    arxiv_id  = external.get("ArXiv") or external.get("arxiv")
    if not arxiv_id:
        return None

    pub_date  = (paper.get("publicationDate") or "").strip()
    if not pub_date:
        return None

    tldr_text = ""
    if isinstance(paper.get("tldr"), dict):
        tldr_text = paper["tldr"].get("text", "") or ""

    authors      = paper.get("authors") or []
    authors_list = [a.get("name", "") for a in authors if a.get("name")]

    title    = (paper.get("title")    or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    scrubbed = scrub_model_words(f"{title}. {title}. {abstract}")

    return {
        "id":                      _arxiv_id_base(arxiv_id),
        "title":                   title,
        "abstract":                abstract,
        "text":                    scrubbed,
        "label_text":              scrub_model_words(f"{title}. {title}. {title}."),
        "url":                     f"https://arxiv.org/pdf/{arxiv_id}",
        "author_count":            len(authors_list),
        "author_tier":             categorize_authors(len(authors_list)),
        "authors_list":            authors_list,
        "ss_citation_count":       int(paper.get("citationCount")             or 0),
        "ss_influential_citations":int(paper.get("influentialCitationCount")  or 0),
        "ss_tldr":                 tldr_text,
        "publication_date":        pub_date,
    }


def discover_candidates(
    db_ids: set,
    date_from: datetime,
    date_to: datetime,
    ss_cache: dict,
) -> list[dict]:
    """Query S2 for top-cited AI papers in the lookback window.

    Paginates until either:
      - The minimum influentialCitationCount on a page drops below
        SIGNIFICANT_MIN_INFLUENTIAL (normal termination), or
      - SIGNIFICANT_MAX_PAGES pages have been fetched (safety limit).

    Filters:
      - Must have an arXiv ID
      - Must have a publication date within [date_from, date_to]
      - Must NOT already be in database.parquet (those are "Recent")

    Updates ss_cache in-place with fresh citation data so the daily pipeline
    won't re-fetch these papers within the normal TTL window.

    Returns the top SIGNIFICANT_POOL_SIZE papers sorted by
    influentialCitationCount descending.
    """
    ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers    = {
        "Content-Type": "application/json",
        "User-Agent": (
            "ai-research-atlas/2.0 "
            "(https://github.com/LeeFischman/ai-research-atlas; "
            "mailto:lee.fischman@gmail.com)"
        ),
    }
    if ss_api_key:
        headers["x-api-key"] = ss_api_key

    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str   = date_to.strftime("%Y-%m-%d")
    inter_page_sleep = 0.4 if ss_api_key else 1.2

    candidates: list[dict] = []
    seen_ids: set[str]     = set()

    print(f"  Date window: {date_from_str} to {date_to_str}")

    for page_num in range(SIGNIFICANT_MAX_PAGES):
        offset = page_num * SIGNIFICANT_PAGE_SIZE
        print(f"  S2 search page {page_num + 1}/{SIGNIFICANT_MAX_PAGES} "
              f"(offset={offset})...")

        try:
            data   = _s2_search_page(offset, date_from_str, date_to_str, headers)
            papers = data.get("data", []) or []
            total  = data.get("total", "?")
        except Exception as e:
            print(f"  S2 search error at page {page_num + 1}: {e}")
            break

        if not papers:
            print(f"  No more results (total reported: {total}).")
            break

        page_accepted = 0
        page_min_inf  = float("inf")
        page_max_inf  = 0

        for paper in papers:
            parsed = _parse_s2_paper(paper)
            if parsed is None:
                continue

            inf          = parsed["ss_influential_citations"]
            page_min_inf = min(page_min_inf, inf)
            page_max_inf = max(page_max_inf, inf)

            # Post-filter: exact date range (server filter may be approximate)
            if (parsed["publication_date"] < date_from_str
                    or parsed["publication_date"] > date_to_str):
                continue

            pid = parsed["id"]
            if pid in db_ids or pid in seen_ids:
                continue

            seen_ids.add(pid)
            candidates.append(parsed)
            page_accepted += 1

        page_min_display = page_min_inf if page_min_inf < float("inf") else 0
        print(f"    Accepted {page_accepted} papers "
              f"(page influential: min={page_min_display}, max={page_max_inf})")

        # Normal stop: influential citations have dropped below threshold
        if page_min_display < SIGNIFICANT_MIN_INFLUENTIAL and page_num > 0:
            print(f"  Stopping: page min influential ({page_min_display}) "
                  f"< threshold ({SIGNIFICANT_MIN_INFLUENTIAL}).")
            break

        # Also stop if we've exhausted the result set
        total_int = int(total) if str(total).isdigit() else 999999
        if offset + SIGNIFICANT_PAGE_SIZE >= total_int:
            print(f"  All {total} results exhausted.")
            break

        time.sleep(inter_page_sleep)

    print(f"\n  Discovery complete: {len(candidates)} candidates found.")

    # Sort by influential citations desc, keep top N
    candidates.sort(key=lambda p: p["ss_influential_citations"], reverse=True)
    pool = candidates[:SIGNIFICANT_POOL_SIZE]
    if pool:
        print(f"  Keeping top {len(pool)} — "
              f"influential range: {pool[-1]['ss_influential_citations']}"
              f"–{pool[0]['ss_influential_citations']}")

    # Update ss_cache so daily pipeline skips re-fetching these papers
    now_iso = datetime.now(timezone.utc).isoformat()
    for p in pool:
        ss_cache[p["id"]] = {
            "citation_count":             p["ss_citation_count"],
            "influential_citation_count": p["ss_influential_citations"],
            "tldr":                       p["ss_tldr"],
            "fetched_at":                 now_iso,
        }

    return pool


# ══════════════════════════════════════════════════════════════════════════════
# RETIREMENT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def apply_retirement(
    new_candidates: list[dict],
    existing_sig: pd.DataFrame | None,
    date_from_str: str,
) -> pd.DataFrame:
    """Apply the two-strike retirement system.

    Rules applied in order:
      1. Papers with publication_date < date_from_str → passive age retirement
      2. Papers in new AND existing → keep, reset strikes to 0, update S2 data
      3. Papers in new only → add with strikes = 0
      4. Papers in existing only → increment strikes;
         retire if strikes >= SIGNIFICANT_STRIKES_LIMIT

    Returns a new DataFrame representing the updated significant pool.
    """
    new_by_id: dict[str, dict] = {p["id"]: p for p in new_candidates}
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    kept: list[dict] = []
    n_new = n_reset = n_struck = n_retired = n_aged = 0

    if existing_sig is not None and not existing_sig.empty:
        existing_ids = set(existing_sig["id"].tolist())

        for _, row in existing_sig.iterrows():
            eid      = row["id"]
            pub_date = str(row.get("publication_date", "") or "").strip()

            # 1. Passive age retirement
            if pub_date and pub_date < date_from_str:
                print(f"  Age-retired ({pub_date}): "
                      f"'{str(row.get('title', eid))[:55]}'")
                n_aged += 1
                continue

            if eid in new_by_id:
                # 2. Paper returned to top-N: reset strikes, refresh S2 data
                d = row.to_dict()
                fresh = new_by_id[eid]
                d.update({
                    "ss_citation_count":        fresh["ss_citation_count"],
                    "ss_influential_citations": fresh["ss_influential_citations"],
                    "ss_tldr":                  fresh["ss_tldr"],
                    "significant_strikes":      0,
                    "paper_source":             "Significant",
                })
                kept.append(d)
                n_reset += 1
            else:
                # 4. Paper absent: increment strike
                strikes = int(row.get("significant_strikes", 0)) + 1
                if strikes >= SIGNIFICANT_STRIKES_LIMIT:
                    print(f"  Strike-retired (strikes={strikes}): "
                          f"'{str(row.get('title', eid))[:55]}'")
                    n_retired += 1
                else:
                    d = row.to_dict()
                    d["significant_strikes"] = strikes
                    d["paper_source"]        = "Significant"
                    kept.append(d)
                    print(f"  Strike {strikes}/{SIGNIFICANT_STRIKES_LIMIT}: "
                          f"'{str(row.get('title', eid))[:55]}'")
                    n_struck += 1
    else:
        existing_ids = set()

    # 3. Genuinely new papers
    for p in new_candidates:
        if p["id"] not in existing_ids:
            p["date_added"]          = now_str
            p["significant_strikes"] = 0
            p["paper_source"]        = "Significant"
            p["author_hindices"]     = None       # populated by enrich_new_papers
            p["Prominence"]          = "Unverified"
            kept.append(p)
            n_new += 1

    print(f"\n  Retirement summary: "
          f"{n_new} added, {n_reset} retained, "
          f"{n_struck} on strike, {n_retired} strike-retired, {n_aged} age-retired.")

    return pd.DataFrame(kept).reset_index(drop=True) if kept else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_new_papers(
    df: pd.DataFrame,
    prev_ids: set,
    author_cache: dict,
) -> pd.DataFrame:
    """Fetch author h-indices (OpenAlex) for newly-added papers only.

    Papers already in the pool keep their existing author_hindices.
    Prominence is recomputed for all rows after enrichment.
    """
    df = df.copy()
    if "author_hindices" not in df.columns:
        df["author_hindices"] = None

    new_mask = ~df["id"].isin(prev_ids)
    n_new    = int(new_mask.sum())
    print(f"  Enriching {n_new} new paper(s) with author h-indices (OpenAlex)...")

    for idx in df.index[new_mask]:
        authors = df.at[idx, "authors_list"]
        if not isinstance(authors, list) or not authors:
            df.at[idx, "author_hindices"] = []
            continue
        df.at[idx, "author_hindices"] = fetch_author_hindices(authors, author_cache)

    # Recompute Prominence for all rows so any h-index updates are reflected
    df["Prominence"] = df.apply(calculate_prominence, axis=1)
    tier_counts = df["Prominence"].value_counts()
    for tier in ["Elite", "Enhanced", "Emerging", "Unverified"]:
        print(f"    {tier}: {tier_counts.get(tier, 0)}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    now      = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    print("=" * 60)
    print(f"  AI Research Atlas -- Significant Papers Weekly Scan")
    print(f"  {run_date} UTC")
    print(f"  Pool size        : {SIGNIFICANT_POOL_SIZE}")
    print(f"  Lookback         : {SIGNIFICANT_LOOKBACK_DAYS} days")
    print(f"  Min age          : {SIGNIFICANT_LOOKFORWARD_DAYS} days")
    print(f"  Min influential  : {SIGNIFICANT_MIN_INFLUENTIAL}")
    print(f"  Strikes limit    : {SIGNIFICANT_STRIKES_LIMIT}")
    print("=" * 60)

    # ── Date window ──────────────────────────────────────────────────────────
    date_from = now - timedelta(days=SIGNIFICANT_LOOKBACK_DAYS)
    date_to   = now - timedelta(days=SIGNIFICANT_LOOKFORWARD_DAYS)

    # ── Load existing significant pool ───────────────────────────────────────
    print("\n▶  Loading existing significant pool...")
    if os.path.exists(SIGNIFICANT_PATH):
        existing_sig = pd.read_parquet(SIGNIFICANT_PATH)
        print(f"  Loaded {len(existing_sig)} papers from {SIGNIFICANT_PATH}.")
        prev_sig_ids = set(existing_sig["id"].tolist())
    else:
        existing_sig = None
        prev_sig_ids = set()
        print("  No existing significant.parquet -- starting fresh.")

    # ── Load database.parquet to get recent-window IDs ───────────────────────
    print("\n▶  Loading recent-window IDs from database.parquet...")
    db_ids: set[str] = set()
    if os.path.exists(DB_PATH):
        db_df  = pd.read_parquet(DB_PATH, columns=["id"])
        db_ids = {_arxiv_id_base(str(i)) for i in db_df["id"].tolist()}
        print(f"  {len(db_ids)} recent-window papers to exclude.")
    else:
        print("  database.parquet not found -- no exclusions applied.")

    # ── Load caches ──────────────────────────────────────────────────────────
    ss_cache     = load_ss_cache()
    author_cache = load_author_cache()

    # ── Stage 1: Discover candidates via S2 ─────────────────────────────────
    print("\n▶  Stage 1 -- Discovering candidates via Semantic Scholar...")
    candidates = discover_candidates(db_ids, date_from, date_to, ss_cache)

    if not candidates:
        print("  No candidates found. Keeping existing pool unchanged.")
        if existing_sig is not None and not existing_sig.empty:
            existing_sig.to_parquet(SIGNIFICANT_PATH, index=False)
            print(f"  Wrote {len(existing_sig)} papers -> {SIGNIFICANT_PATH} (unchanged).")
        save_ss_cache(ss_cache)
        print("\n✓  update_significant.py complete (no new candidates).")
        raise SystemExit(0)

    # ── Stage 2: Retirement logic ────────────────────────────────────────────
    print("\n▶  Stage 2 -- Applying retirement logic...")
    sig_df = apply_retirement(
        new_candidates = candidates,
        existing_sig   = existing_sig,
        date_from_str  = date_from.strftime("%Y-%m-%d"),
    )

    if sig_df.empty:
        print("  Significant pool is empty after retirement. Nothing to write.")
        save_ss_cache(ss_cache)
        raise SystemExit(0)

    # ── Stage 3: Author h-index enrichment (new papers only) ─────────────────
    print("\n▶  Stage 3 -- Author h-index enrichment...")
    sig_df = enrich_new_papers(sig_df, prev_sig_ids, author_cache)

    save_author_cache(author_cache)
    save_ss_cache(ss_cache)
    print(f"  Author cache: {len(author_cache)} entries.  "
          f"SS cache: {len(ss_cache)} entries.")

    # ── Write significant.parquet ─────────────────────────────────────────────
    print(f"\n▶  Writing {len(sig_df)} papers -> {SIGNIFICANT_PATH}...")

    if "date_added" in sig_df.columns:
        sig_df["date_added"] = pd.to_datetime(
            sig_df["date_added"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    sig_df.to_parquet(SIGNIFICANT_PATH, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n-- Pool summary --")
    if "ss_influential_citations" in sig_df.columns:
        inf_s = sig_df["ss_influential_citations"].fillna(0).astype(int)
        print(f"  Influential citations: max={inf_s.max()}, "
              f"median={inf_s.median():.0f}, min={inf_s.min()}")
    for tier in ["Elite", "Enhanced", "Emerging", "Unverified"]:
        print(f"  Prominence {tier}: {(sig_df['Prominence'] == tier).sum()}")
    for s in sorted(sig_df["significant_strikes"].unique()):
        print(f"  Strikes={int(s)}: {(sig_df['significant_strikes'] == s).sum()} papers")

    print("\n✓  update_significant.py complete.")
