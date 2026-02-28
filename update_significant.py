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
    fetch_semantic_scholar_data,
    oai_fetch_ids_for_range,
    fetch_arxiv_metadata,
    _arxiv_id_base,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SIGNIFICANT_PATH          = "significant.parquet"
SIG_CANDIDATES_PATH       = "sig_candidates.json"

# How many papers to track in the candidate pool for weekly citation refresh.
# Must be <= 500 so the refresh fits in a single S2 batch call.
SIG_CANDIDATES_POOL_SIZE  = 500

# How many top candidates (by influential citations) enter significant.parquet.
SIGNIFICANT_POOL_SIZE     = 75

# Date window
SIGNIFICANT_LOOKBACK_DAYS    = 150   # max age: papers older than this are retired
SIGNIFICANT_LOOKFORWARD_DAYS = 15    # min age: younger papers are in the recent window

# Retirement
SIGNIFICANT_STRIKES_LIMIT    = 2     # retire after this many consecutive absences


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE POOL STATE
#
# sig_candidates.json persists the top-SIG_CANDIDATES_POOL_SIZE papers between
# weekly runs, enabling incremental delta fetches instead of full re-scans.
#
# Schema:
#   last_fetched_date : YYYY-MM-DD — the `until` date of the last OAI fetch.
#                       Next run fetches from this date forward (delta).
#   pool              : list of {id, ss_citation_count, ss_influential_citations}
#                       sorted by ss_citation_count desc, capped at
#                       SIG_CANDIDATES_POOL_SIZE entries.
# ══════════════════════════════════════════════════════════════════════════════

def load_sig_candidates() -> dict | None:
    """Load the candidate pool state from disk. Returns None on first run."""
    if os.path.exists(SIG_CANDIDATES_PATH):
        with open(SIG_CANDIDATES_PATH) as f:
            return json.load(f)
    return None


def save_sig_candidates(state: dict) -> None:
    """Persist the candidate pool state to disk."""
    with open(SIG_CANDIDATES_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY  (arXiv OAI-PMH → S2 batch citations)
#
# Replaces the previous S2 search-based approach, which was unreliable:
#   • S2's /paper/search endpoint returned total=0 for multi-term OR queries
#   • fieldsOfStudy filter also returned total=0
#   • sort=citationCount:desc was undocumented and unreliable
#
# New approach:
#   1. Fetch cs.AI paper IDs from arXiv OAI-PMH (authoritative source).
#      First run: full SIGNIFICANT_LOOKBACK_DAYS window (~150 days, ~15-30k IDs).
#      Subsequent runs: delta from last_fetched_date (typically ~200 new IDs/week).
#   2. S2 batch lookup for new IDs → get citation counts (proven reliable).
#   3. Merge new IDs into existing candidate pool, keep top-SIG_CANDIDATES_POOL_SIZE
#      by citation count.
#   4. Weekly citation refresh: re-fetch S2 citations for the full top-500 in one
#      batch call. Catches citation growth on existing candidates.
#   5. Select top-SIGNIFICANT_POOL_SIZE by influential citations → significant pool.
#   6. For new entrants not in existing significant.parquet: fetch full metadata
#      (title, abstract, authors, publication_date) from arXiv Search API.
# ══════════════════════════════════════════════════════════════════════════════

def _build_paper_dict(arxiv_paper, ss_cache: dict) -> dict:
    """Convert an _ArxivPaper + cached S2 data into a significant pool schema dict."""
    base_id      = _arxiv_id_base(arxiv_paper.entry_id.split("/")[-1])
    title        = arxiv_paper.title
    abstract     = arxiv_paper.summary
    authors_list = [a.name for a in arxiv_paper.authors]
    scrubbed     = scrub_model_words(f"{title}. {title}. {abstract}")
    ss_entry     = ss_cache.get(base_id, {})

    return {
        "id":                      base_id,
        "title":                   title,
        "abstract":                abstract,
        "text":                    scrubbed,
        "label_text":              scrub_model_words(f"{title}. {title}. {title}."),
        "url":                     arxiv_paper.pdf_url,
        "author_count":            len(authors_list),
        "author_tier":             categorize_authors(len(authors_list)),
        "authors_list":            authors_list,
        "ss_citation_count":       int(ss_entry.get("citation_count",             0)),
        "ss_influential_citations":int(ss_entry.get("influential_citation_count", 0)),
        "ss_tldr":                 ss_entry.get("tldr", "") or "",
        "publication_date":        arxiv_paper.publication_date,
    }


def discover_candidates(
    db_ids:       set,
    date_from:    datetime,
    date_to:      datetime,
    ss_cache:     dict,
    existing_sig: "pd.DataFrame | None",
) -> list[dict]:
    """Discover top-SIGNIFICANT_POOL_SIZE cs.AI candidates for the significant pool.

    Uses arXiv OAI-PMH for authoritative cs.AI paper discovery and S2 batch
    endpoint for citation data. Maintains a persistent candidate pool in
    sig_candidates.json for efficient incremental weekly updates.

    Parameters
    ----------
    db_ids       : arXiv IDs already in database.parquet (recent window) — excluded
    date_from    : oldest publication date to consider (today - LOOKBACK_DAYS)
    date_to      : newest publication date to consider (today - LOOKFORWARD_DAYS)
    ss_cache     : shared S2 cache dict (mutated in-place)
    existing_sig : current significant.parquet DataFrame, or None

    Returns
    -------
    List of paper dicts ready for apply_retirement. Existing-pool papers include
    only citation fields (apply_retirement reads the rest from significant.parquet).
    New entrants include full metadata fetched from arXiv.
    """
    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str   = date_to.strftime("%Y-%m-%d")
    existing_ids  = set(existing_sig["id"].tolist()) if existing_sig is not None else set()

    # ── Load or initialise candidate pool ────────────────────────────────────
    state = load_sig_candidates()
    if state is None:
        print("  No sig_candidates.json found — performing full backfill "
              f"({SIGNIFICANT_LOOKBACK_DAYS}-day window).")
        fetch_from = date_from_str
        pool_by_id: dict[str, dict] = {}
    else:
        fetch_from  = state["last_fetched_date"]
        pool_by_id  = {p["id"]: p for p in state.get("pool", [])}
        print(f"  Loaded {len(pool_by_id)} existing candidates from "
              f"{SIG_CANDIDATES_PATH} (last fetched: {fetch_from}).")

    # ── Step 1: Fetch new arXiv IDs (delta or full) ───────────────────────────
    if fetch_from < date_to_str:
        print(f"\n  Step 1 — OAI-PMH fetch: {fetch_from} to {date_to_str}")
        new_ids = oai_fetch_ids_for_range(fetch_from, date_to_str)
        # Exclude papers already in the recent window or already in candidate pool
        new_ids = [i for i in new_ids
                   if i not in db_ids and i not in pool_by_id]
        print(f"  {len(new_ids)} new IDs after excluding recent-window "
              f"and existing candidates.")
    else:
        new_ids = []
        print(f"\n  Step 1 — No new IDs to fetch "
              f"(last_fetched_date {fetch_from} >= date_to {date_to_str}).")

    # ── Step 2: S2 batch lookup for new IDs ──────────────────────────────────
    if new_ids:
        print(f"\n  Step 2 — S2 citation lookup for {len(new_ids)} new IDs...")
        fetch_semantic_scholar_data(new_ids, ss_cache)
        for aid in new_ids:
            entry = ss_cache.get(aid, {})
            pool_by_id[aid] = {
                "id":                      aid,
                "ss_citation_count":       int(entry.get("citation_count",             0)),
                "ss_influential_citations":int(entry.get("influential_citation_count", 0)),
            }
    else:
        print(f"\n  Step 2 — No new IDs to look up in S2.")

    # ── Step 3: Rank and keep top-SIG_CANDIDATES_POOL_SIZE ───────────────────
    ranked = sorted(pool_by_id.values(),
                    key=lambda p: p["ss_citation_count"], reverse=True)
    top_pool = ranked[:SIG_CANDIDATES_POOL_SIZE]
    print(f"\n  Step 3 — Candidate pool: {len(pool_by_id)} total → "
          f"keeping top {len(top_pool)} by citation count.")
    if top_pool:
        print(f"  Citation range in pool: "
              f"{top_pool[-1]['ss_citation_count']}–{top_pool[0]['ss_citation_count']}")

    # ── Step 4: Weekly citation refresh for full top-500 (single batch) ──────
    top_pool_ids = [p["id"] for p in top_pool]
    print(f"\n  Step 4 — Refreshing S2 citations for top {len(top_pool_ids)} "
          f"candidates (single batch call)...")
    fetch_semantic_scholar_data(top_pool_ids, ss_cache)

    # Update pool with refreshed counts
    for p in top_pool:
        entry = ss_cache.get(p["id"], {})
        p["ss_citation_count"]        = int(entry.get("citation_count",             0))
        p["ss_influential_citations"] = int(entry.get("influential_citation_count", 0))

    # ── Persist updated candidate pool ───────────────────────────────────────
    save_sig_candidates({
        "last_fetched_date": date_to_str,
        "pool":              top_pool,
    })
    print(f"  Saved {len(top_pool)} candidates to {SIG_CANDIDATES_PATH}.")

    # ── Step 5: Select top-SIGNIFICANT_POOL_SIZE by influential citations ─────
    top_sig = sorted(top_pool,
                     key=lambda p: p["ss_influential_citations"], reverse=True)
    top_sig = top_sig[:SIGNIFICANT_POOL_SIZE]
    print(f"\n  Step 5 — Top {len(top_sig)} by influential citations: "
          f"range {top_sig[-1]['ss_influential_citations'] if top_sig else 0}"
          f"–{top_sig[0]['ss_influential_citations'] if top_sig else 0}")

    # ── Step 6: Fetch arXiv metadata for new entrants ─────────────────────────
    need_metadata = [p["id"] for p in top_sig if p["id"] not in existing_ids]
    metadata_map: dict[str, object] = {}
    if need_metadata:
        print(f"\n  Step 6 — Fetching arXiv metadata for "
              f"{len(need_metadata)} new entrants...")
        arxiv_papers = fetch_arxiv_metadata(need_metadata)
        for ap in arxiv_papers:
            bid = _arxiv_id_base(ap.entry_id.split("/")[-1])
            metadata_map[bid] = ap
        print(f"  Metadata retrieved for {len(metadata_map)}/{len(need_metadata)} papers.")
    else:
        print(f"\n  Step 6 — All {len(top_sig)} candidates already in significant pool "
              f"— no metadata fetch needed.")

    # ── Step 7: Build candidate dicts for apply_retirement ───────────────────
    candidates = []
    n_existing = n_new = n_skipped = 0

    for p in top_sig:
        aid      = p["id"]
        ss_entry = ss_cache.get(aid, {})

        if aid in existing_ids:
            # Already in pool — pass citation update only; apply_retirement
            # reads all other fields from the existing significant.parquet row
            candidates.append({
                "id":                      aid,
                "ss_citation_count":       p["ss_citation_count"],
                "ss_influential_citations":p["ss_influential_citations"],
                "ss_tldr":                 ss_entry.get("tldr", "") or "",
            })
            n_existing += 1
        elif aid in metadata_map:
            candidates.append(_build_paper_dict(metadata_map[aid], ss_cache))
            n_new += 1
        else:
            print(f"  Warning: no arXiv metadata for {aid} — skipping.")
            n_skipped += 1

    print(f"\n  Discovery complete: {n_existing} existing, {n_new} new, "
          f"{n_skipped} skipped (no metadata).")

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  TEST HARNESS — REMOVE AFTER VALIDATION                     ║
    # ╚══════════════════════════════════════════════════════════════╝
    print("\n  ── TEST HARNESS: DISCOVERY SUMMARY ──────────────────────")

    # ── OAI fetch quality ─────────────────────────────────────────
    print(f"\n  OAI-PMH fetch:")
    print(f"    New IDs fetched         : {len(new_ids)}")
    print(f"    Candidate pool size     : {len(top_pool)} (cap={SIG_CANDIDATES_POOL_SIZE})")
    print(f"    Pool was pre-existing   : {state is not None}")
    if state is not None:
        print(f"    Last fetched date       : {state.get('last_fetched_date', '?')}")
        print(f"    Delta fetch range       : {fetch_from} → {date_to_str}")

    # ── S2 citation quality ───────────────────────────────────────
    print(f"\n  S2 citation data (top-{len(top_sig)} significant candidates):")
    cites     = [p["ss_citation_count"]        for p in top_sig]
    inf_cites = [p["ss_influential_citations"] for p in top_sig]
    n_zero_cites = sum(1 for c in cites if c == 0)
    n_zero_inf   = sum(1 for c in inf_cites if c == 0)
    if cites:
        print(f"    Citation count   — max={max(cites)}, "
              f"median={sorted(cites)[len(cites)//2]}, "
              f"min={min(cites)}, zero={n_zero_cites}")
        print(f"    Influential cits — max={max(inf_cites)}, "
              f"median={sorted(inf_cites)[len(inf_cites)//2]}, "
              f"min={min(inf_cites)}, zero={n_zero_inf}")

    # ── Sample top-10 significant candidates ─────────────────────
    print(f"\n  Top 10 candidates by influential citations:")
    for i, p in enumerate(top_sig[:10], 1):
        title_preview = ""
        if p["id"] in metadata_map:
            title_preview = metadata_map[p["id"]].title[:55]
        elif p["id"] in existing_ids and existing_sig is not None:
            row = existing_sig[existing_sig["id"] == p["id"]]
            if not row.empty:
                title_preview = str(row.iloc[0].get("title", ""))[:55]
        print(f"    {i:2}. {p['id']}  "
              f"cites={p['ss_citation_count']:4}  "
              f"inf={p['ss_influential_citations']:3}  "
              f"{title_preview}")

    # ── Metadata fetch quality ────────────────────────────────────
    print(f"\n  Metadata fetch (new entrants):")
    print(f"    Needed metadata         : {len(need_metadata)}")
    print(f"    Successfully retrieved  : {len(metadata_map)}")
    if need_metadata:
        n_missing_meta = len(need_metadata) - len(metadata_map)
        print(f"    Missing (will be skipped): {n_missing_meta}")
        if metadata_map:
            sample_meta = list(metadata_map.values())[:3]
            print(f"    Sample new entrant titles:")
            for ap in sample_meta:
                print(f"      - {ap.title[:70]}")
                print(f"        pub_date={ap.publication_date}  "
                      f"authors={len(ap.authors)}")

    # ── sig_candidates.json state ────────────────────────────────
    if os.path.exists(SIG_CANDIDATES_PATH):
        with open(SIG_CANDIDATES_PATH) as _f:
            _saved = json.load(_f)
        print(f"\n  sig_candidates.json:")
        print(f"    last_fetched_date : {_saved.get('last_fetched_date', '?')}")
        print(f"    pool size         : {len(_saved.get('pool', []))}")
        _pool = _saved.get("pool", [])
        if _pool:
            print(f"    citation range    : "
                  f"{_pool[-1]['ss_citation_count']}–{_pool[0]['ss_citation_count']}")

    # ── Candidate breakdown for apply_retirement ─────────────────
    print(f"\n  Candidates passed to apply_retirement:")
    print(f"    Existing (citation update only) : {n_existing}")
    print(f"    New (full metadata)             : {n_new}")
    print(f"    Skipped (no metadata)           : {n_skipped}")
    print(f"    Total                           : {len(candidates)}")

    print("  ── END TEST HARNESS ──────────────────────────────────────")
    # ╔══════════════════════════════════════════════════════════════╗
    # ║  END TEST HARNESS                                            ║
    # ╚══════════════════════════════════════════════════════════════╝

    return candidates

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

    # Also retry existing papers whose authors aren't in cache — these are
    # likely 429 failures from a previous run (429 failures are not cached,
    # so they appear as cache misses here).
    def _has_uncached_authors(row) -> bool:
        authors = row.get("authors_list", [])
        if not isinstance(authors, list) or not authors:
            return False
        return any(
            author_cache.get(a.strip().lower()) is None
            for a in authors if a.strip()
        )

    retry_mask = df.index.isin(
        df[~new_mask].index[df[~new_mask].apply(_has_uncached_authors, axis=1)]
    )
    n_retry = int(retry_mask.sum())

    print(f"  Enriching {n_new} new paper(s) with author h-indices (OpenAlex)...")
    if n_retry:
        print(f"  Retrying {n_retry} existing paper(s) with uncached authors "
              f"(likely prior 429 failures)...")

    for idx in df.index[new_mask | retry_mask]:
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
# STAGE 4 — WEEKLY CITATION REFRESH
# ══════════════════════════════════════════════════════════════════════════════

def refresh_recent_citations(ss_cache: dict) -> None:
    """Force-refresh S2 citation data for all papers in database.parquet.

    Called unconditionally from __main__ regardless of whether Stage 1
    found any new significant candidates. This ensures recent-window papers
    receive a coordinated weekly citation update even on weeks where the
    significant pool is unchanged.
    """
    print("\n▶  Stage 4 -- Weekly citation refresh for recent-window papers...")

    if not os.path.exists(DB_PATH):
        print(f"  {DB_PATH} not found — skipping recent-window refresh.")
        return

    db_full    = pd.read_parquet(DB_PATH, columns=["id"])
    db_all_ids = [str(i) for i in db_full["id"].tolist()]
    print(f"  {len(db_all_ids)} recent-window papers to refresh.")

    if not db_all_ids:
        print("  database.parquet is empty — nothing to refresh.")
        return

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  TEST HARNESS — REMOVE AFTER VALIDATION                     ║
    # ╚══════════════════════════════════════════════════════════════╝
    print("\n  ── TEST HARNESS: PRE-FETCH SNAPSHOT ──────────────────────")
    base_ids = [_arxiv_id_base(aid) for aid in db_all_ids]

    before = {}
    n_pre_cached = n_pre_missing = n_pre_no_signal = 0
    for bid in base_ids:
        entry = ss_cache.get(bid)
        if entry is None:
            before[bid] = None
            n_pre_missing += 1
        else:
            before[bid] = {
                "fetched_at":                 entry.get("fetched_at", ""),
                "citation_count":             int(entry.get("citation_count", 0)),
                "influential_citation_count": int(entry.get("influential_citation_count", 0)),
                "has_tldr":                   bool((entry.get("tldr") or "").strip()),
            }
            n_pre_cached += 1
            if (before[bid]["citation_count"] == 0
                    and before[bid]["influential_citation_count"] == 0
                    and not before[bid]["has_tldr"]):
                n_pre_no_signal += 1

    print(f"  Papers in DB          : {len(base_ids)}")
    print(f"  Already in ss_cache   : {n_pre_cached}")
    print(f"  Missing from ss_cache : {n_pre_missing}")
    print(f"  In cache, zero signal : {n_pre_no_signal}  "
          f"(no citations + no TLDR — likely brand-new)")

    sample_ids = [
        bid for bid in base_ids
        if before.get(bid) and before[bid]["citation_count"] > 0
    ][:5]
    if sample_ids:
        print(f"\n  Sample (up to 5 papers with existing citations):")
        for bid in sample_ids:
            b = before[bid]
            print(f"    {bid}  citations={b['citation_count']}  "
                  f"influential={b['influential_citation_count']}  "
                  f"fetched_at={b['fetched_at']}")
    else:
        print("  No cached papers with citations found — all are new or zero.")
    print("  ── END PRE-FETCH SNAPSHOT ────────────────────────────────\n")

    fetch_semantic_scholar_data(db_all_ids, ss_cache)
    save_ss_cache(ss_cache)
    print(f"  Citation refresh complete. SS cache now has {len(ss_cache)} entries.")

    print("\n  ── TEST HARNESS: POST-FETCH COMPARISON ───────────────────")
    n_timestamp_updated = n_timestamp_unchanged = 0
    n_newly_cached = n_still_null = 0
    n_citations_increased = n_citations_unchanged = n_citations_decreased = 0
    changed_examples = []

    for bid in base_ids:
        after_entry = ss_cache.get(bid)
        if after_entry is None:
            n_still_null += 1
            continue

        after = {
            "fetched_at":                 after_entry.get("fetched_at", ""),
            "citation_count":             int(after_entry.get("citation_count", 0)),
            "influential_citation_count": int(after_entry.get("influential_citation_count", 0)),
        }

        b = before.get(bid)
        if b is None:
            n_newly_cached += 1
            n_timestamp_updated += 1
        else:
            if after["fetched_at"] != b["fetched_at"]:
                n_timestamp_updated += 1
            else:
                n_timestamp_unchanged += 1

            delta = after["citation_count"] - b["citation_count"]
            if delta > 0:
                n_citations_increased += 1
                if len(changed_examples) < 5:
                    changed_examples.append((bid, b["citation_count"],
                                              after["citation_count"], delta))
            elif delta < 0:
                n_citations_decreased += 1
            else:
                n_citations_unchanged += 1

    print(f"  Timestamps updated    : {n_timestamp_updated} / {len(base_ids)}")
    print(f"  Timestamps unchanged  : {n_timestamp_unchanged}  "
          f"(unexpected — fetch should always write new fetched_at)")
    print(f"  Newly added to cache  : {n_newly_cached}  "
          f"(were missing before fetch)")
    print(f"  Still null after fetch: {n_still_null}  "
          f"(should be 0 — fetch writes zeros for unindexed papers)")
    print(f"  Citation count increased : {n_citations_increased}")
    print(f"  Citation count unchanged : {n_citations_unchanged}")
    print(f"  Citation count decreased : {n_citations_decreased}  "
          f"(should be 0 — citations don't go backwards)")

    if changed_examples:
        print(f"\n  Papers with citation count increases (up to 5):")
        for bid, before_c, after_c, delta in changed_examples:
            print(f"    {bid}  {before_c} → {after_c}  (+{delta})")
    else:
        print("\n  No citation count increases observed.")
        print("  This is expected if papers are very recent (few days old).")

    if sample_ids:
        print(f"\n  Spot-check: same {len(sample_ids)} sample papers after fetch:")
        for bid in sample_ids:
            a = ss_cache.get(bid, {})
            print(f"    {bid}  citations={a.get('citation_count', '?')}  "
                  f"influential={a.get('influential_citation_count', '?')}  "
                  f"fetched_at={a.get('fetched_at', '?')}")

    print("  ── END POST-FETCH COMPARISON ─────────────────────────────")
    # ╔══════════════════════════════════════════════════════════════╗
    # ║  END TEST HARNESS                                            ║
    # ╚══════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    now      = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    print("=" * 60)
    print(f"  AI Research Atlas -- Significant Papers Weekly Scan")
    print(f"  {run_date} UTC")
    print(f"  Significant pool size : {SIGNIFICANT_POOL_SIZE}")
    print(f"  Candidate pool size   : {SIG_CANDIDATES_POOL_SIZE}")
    print(f"  Lookback              : {SIGNIFICANT_LOOKBACK_DAYS} days")
    print(f"  Min age               : {SIGNIFICANT_LOOKFORWARD_DAYS} days")
    print(f"  Strikes limit         : {SIGNIFICANT_STRIKES_LIMIT}")
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

    # ── Stage 1: Discover candidates via arXiv OAI-PMH + S2 batch ──────────────
    print("\n▶  Stage 1 -- Discovering candidates via arXiv + Semantic Scholar...")
    candidates = discover_candidates(db_ids, date_from, date_to, ss_cache, existing_sig)

    if not candidates:
        print("  No candidates found. Keeping existing pool unchanged.")
        if existing_sig is not None and not existing_sig.empty:
            existing_sig.to_parquet(SIGNIFICANT_PATH, index=False)
            print(f"  Wrote {len(existing_sig)} papers -> {SIGNIFICANT_PATH} (unchanged).")
        save_ss_cache(ss_cache)
        refresh_recent_citations(ss_cache)
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

    # ── Stage 4: Weekly citation refresh ─────────────────────────────────────
    refresh_recent_citations(ss_cache)

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
