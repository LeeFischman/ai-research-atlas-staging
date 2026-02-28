#!/usr/bin/env python3
# daily_significant.py
# ──────────────────────────────────────────────────────────────────────────────
# AI Research Atlas — Daily significant-papers maintenance job.
#
# Runs every day at 3 AM EST (8 AM UTC) — after the main daily atlas build
# (03:00 UTC) has committed fresh database.parquet.
#
# Two tasks:
#   1. AUTHOR BACKFILL — find authors in significant.parquet whose h-index
#      is not yet in author_cache.json, fetch up to OA_DAILY_LIMIT per run
#      via OpenAlex, recompute Prominence, write significant.parquet.
#
#   2. S2 CITATION REFRESH — re-fetch citation counts for all 500 candidates
#      in sig_candidates.json via a single Semantic Scholar batch call, then
#      re-rank so the next weekly run works from fresh numbers.
#      Also updates ss_cache.json for any papers in significant.parquet.
#
# Scheduling rationale
# ────────────────────
# Author enrichment is decoupled from weekly discovery because the first
# weekly run with 75 new entrants requires ~75 × avg_authors OpenAlex calls.
# Even at 1 req/s (post-429 slow mode) this can exceed GitHub Actions' 6-hour
# limit. Spreading across daily runs with a hard cap of 900 calls/day keeps
# each job well inside limits.
#
# S2 citation refresh moved here from the weekly job so citation counts
# (and therefore tier rankings) update daily rather than weekly.
#
# Normal run:
#   OPENALEX_API_KEY=... SEMANTIC_SCHOLAR_API_KEY=... python daily_significant.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime, timezone

import pandas as pd

from atlas_utils import (
    DB_PATH,
    calculate_prominence,
    load_author_cache,
    save_author_cache,
    fetch_author_hindices,
    load_ss_cache,
    save_ss_cache,
    fetch_semantic_scholar_data,
    _arxiv_id_base,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SIGNIFICANT_PATH     = "significant.parquet"
SIG_CANDIDATES_PATH  = "sig_candidates.json"

# Max unique author name lookups sent to OpenAlex per daily run.
# Free key budget: $1/day at $0.001/search = 1,000 searches.
# 900 leaves 10% headroom for other OpenAlex usage.
OA_DAILY_LIMIT = 900

# Author cache TTL (days) — must match atlas_utils.AUTHOR_CACHE_TTL_DAYS
from atlas_utils import AUTHOR_CACHE_TTL_DAYS


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — AUTHOR BACKFILL
# ══════════════════════════════════════════════════════════════════════════════

def _uncached_authors(sig_df: pd.DataFrame, author_cache: dict) -> list[str]:
    """Return unique author names from sig_df that are missing or stale in cache.

    'Missing' = not in cache at all.
    'Stale'   = fetched_at is beyond AUTHOR_CACHE_TTL_DAYS.
    """
    from datetime import timedelta
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=AUTHOR_CACHE_TTL_DAYS)
    ).isoformat()

    seen: set[str] = set()
    uncached: list[str] = []

    for authors in sig_df["authors_list"]:
        if not isinstance(authors, list):
            continue
        for name in authors:
            name = name.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            entry = author_cache.get(key)
            if entry is None or entry.get("fetched_at", "") < cutoff_ts:
                uncached.append(name)

    return uncached


def run_author_backfill(sig_df: pd.DataFrame, author_cache: dict) -> pd.DataFrame:
    """Fetch h-indices for up to OA_DAILY_LIMIT uncached authors.

    Recomputes Prominence for the entire pool after enrichment.
    Returns the updated DataFrame.
    """
    uncached = _uncached_authors(sig_df, author_cache)
    total_pending = len(uncached)

    if not uncached:
        print("  All authors already cached — nothing to fetch.")
        return sig_df

    batch = uncached[:OA_DAILY_LIMIT]
    remainder = total_pending - len(batch)

    print(f"  Uncached authors      : {total_pending}")
    print(f"  Fetching this run     : {len(batch)} (cap={OA_DAILY_LIMIT})")
    if remainder:
        print(f"  Remaining for future  : {remainder} (will be processed in subsequent runs)")

    # fetch_author_hindices mutates author_cache in-place and handles
    # 429 backoff + global throttle detection internally.
    fetch_author_hindices(batch, author_cache)

    # Now rebuild author_hindices for every paper from the updated cache
    print("  Rebuilding author_hindices for all papers...")
    from atlas_utils import _safe_hindices
    from datetime import timedelta
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=AUTHOR_CACHE_TTL_DAYS)
    ).isoformat()

    def _hindices_from_cache(authors_list) -> list[int]:
        if not isinstance(authors_list, list):
            return []
        result = []
        for name in authors_list:
            name = name.strip()
            if not name:
                result.append(0)
                continue
            entry = author_cache.get(name.lower())
            if entry and entry.get("fetched_at", "") >= cutoff_ts:
                result.append(int(entry.get("hindex", 0)))
            else:
                result.append(0)
        return result

    sig_df = sig_df.copy()
    sig_df["author_hindices"] = sig_df["authors_list"].apply(_hindices_from_cache)

    # Recompute Prominence for all papers
    sig_df["Prominence"] = sig_df.apply(calculate_prominence, axis=1)
    tier_counts = sig_df["Prominence"].value_counts()
    print("  Prominence after enrichment:")
    for tier in ["Elite", "Enhanced", "Emerging", "Unverified"]:
        print(f"    {tier}: {tier_counts.get(tier, 0)}")

    return sig_df


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — S2 CITATION REFRESH
# ══════════════════════════════════════════════════════════════════════════════

def run_citation_refresh(sig_df: pd.DataFrame, ss_cache: dict) -> pd.DataFrame:
    """Re-fetch S2 citation counts for all candidates + significant papers.

    Two batch calls:
      - All IDs in sig_candidates.json (up to 500) — keeps candidate rankings fresh.
      - Any significant paper IDs not already covered above.

    Updates sig_candidates.json with new citation counts.
    Updates significant.parquet ss_citation_count and ss_influential_citations columns.
    Returns updated sig_df.
    """
    # ── Collect all IDs to refresh ────────────────────────────────────────────
    candidate_ids: list[str] = []
    if os.path.exists(SIG_CANDIDATES_PATH):
        with open(SIG_CANDIDATES_PATH) as f:
            state = json.load(f)
        candidate_ids = [p["id"] for p in state.get("pool", [])]
        print(f"  Candidate pool: {len(candidate_ids)} IDs loaded from {SIG_CANDIDATES_PATH}.")
    else:
        print(f"  {SIG_CANDIDATES_PATH} not found — skipping candidate refresh.")

    # Add any significant paper IDs not already in the candidate list
    sig_ids = set(_arxiv_id_base(str(i)) for i in sig_df["id"].tolist())
    candidate_id_set = set(candidate_ids)
    extra_ids = [i for i in sig_ids if i not in candidate_id_set]
    all_ids = candidate_ids + extra_ids

    if not all_ids:
        print("  No IDs to refresh.")
        return sig_df

    print(f"  Refreshing S2 citations for {len(all_ids)} papers "
          f"({len(candidate_ids)} candidates + {len(extra_ids)} sig-only)...")
    fetch_semantic_scholar_data(all_ids, ss_cache)
    save_ss_cache(ss_cache)
    print(f"  SS cache updated: {len(ss_cache)} entries.")

    # ── Update sig_candidates.json with fresh citation counts ─────────────────
    if os.path.exists(SIG_CANDIDATES_PATH) and candidate_ids:
        with open(SIG_CANDIDATES_PATH) as f:
            state = json.load(f)
        updated_candidates = []
        for p in state.get("candidates", []):
            bid = _arxiv_id_base(p["id"])
            entry = ss_cache.get(bid)
            if entry:
                p["ss_citation_count"]        = entry.get("citation_count", 0)
                p["ss_influential_citations"] = entry.get("influential_citation_count", 0)
            updated_candidates.append(p)
        # Re-rank by citation count so weekly job works from fresh numbers
        updated_candidates.sort(key=lambda x: x.get("ss_citation_count", 0), reverse=True)
        state["candidates"] = updated_candidates
        with open(SIG_CANDIDATES_PATH, "w") as f:
            json.dump(state, f, indent=2)
        cites = [p.get("ss_citation_count", 0) for p in updated_candidates]
        print(f"  sig_candidates.json updated: "
              f"citation range {min(cites)}–{max(cites)} across {len(cites)} papers.")

    # ── Update sig_df with fresh citation counts ──────────────────────────────
    sig_df = sig_df.copy()
    for col, key in [("ss_citation_count", "citation_count"),
                     ("ss_influential_citations", "influential_citation_count"),
                     ("ss_tldr", "tldr")]:
        if col not in sig_df.columns:
            sig_df[col] = None

    def _update_row(row):
        bid   = _arxiv_id_base(str(row["id"]))
        entry = ss_cache.get(bid)
        if entry:
            row = row.copy()
            row["ss_citation_count"]        = entry.get("citation_count", 0)
            row["ss_influential_citations"] = entry.get("influential_citation_count", 0)
            if entry.get("tldr"):
                row["ss_tldr"] = entry["tldr"]
        return row

    sig_df = sig_df.apply(_update_row, axis=1)

    if "ss_influential_citations" in sig_df.columns:
        inf_s = sig_df["ss_influential_citations"].fillna(0).astype(int)
        print(f"  Significant pool citations — "
              f"max={inf_s.max()}, median={inf_s.median():.0f}, min={inf_s.min()}")

    return sig_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    now      = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    print("=" * 60)
    print(f"  AI Research Atlas -- Daily Significant Papers Maintenance")
    print(f"  {run_date} UTC")
    print(f"  OpenAlex daily cap    : {OA_DAILY_LIMIT} author lookups")
    print(f"  Author cache TTL      : {AUTHOR_CACHE_TTL_DAYS} days")
    print("=" * 60)

    # ── Load significant pool ─────────────────────────────────────────────────
    if not os.path.exists(SIGNIFICANT_PATH):
        print(f"\n  {SIGNIFICANT_PATH} not found — nothing to do.")
        print("  (Weekly job hasn't run yet, or pool is empty.)")
        raise SystemExit(0)

    sig_df = pd.read_parquet(SIGNIFICANT_PATH)
    print(f"\n  Loaded {len(sig_df)} papers from {SIGNIFICANT_PATH}.")

    if sig_df.empty:
        print("  Pool is empty — nothing to do.")
        raise SystemExit(0)

    # Ensure authors_list column exists and is list type
    if "authors_list" not in sig_df.columns:
        print("  authors_list column missing — cannot enrich authors.")
        sig_df["authors_list"] = [[] for _ in range(len(sig_df))]

    # ── Load caches ───────────────────────────────────────────────────────────
    author_cache = load_author_cache()
    ss_cache     = load_ss_cache()

    # ── Task 1: Author backfill ───────────────────────────────────────────────
    print(f"\n▶  Task 1 -- Author h-index backfill (OpenAlex, cap={OA_DAILY_LIMIT})...")
    sig_df = run_author_backfill(sig_df, author_cache)
    save_author_cache(author_cache)
    print(f"  Author cache saved: {len(author_cache)} entries.")

    # ── Task 2: S2 citation refresh ───────────────────────────────────────────
    print(f"\n▶  Task 2 -- S2 citation refresh...")
    sig_df = run_citation_refresh(sig_df, ss_cache)

    # ── Write significant.parquet ─────────────────────────────────────────────
    print(f"\n▶  Writing {len(sig_df)} papers -> {SIGNIFICANT_PATH}...")

    if "date_added" in sig_df.columns:
        sig_df["date_added"] = pd.to_datetime(
            sig_df["date_added"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    sig_df.to_parquet(SIGNIFICANT_PATH, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n-- Summary --")
    still_uncached = len(_uncached_authors(sig_df, author_cache))
    print(f"  Authors still pending : {still_uncached}"
          + (" (will continue tomorrow)" if still_uncached else " ✓ fully enriched"))
    tier_counts = sig_df["Prominence"].value_counts() if "Prominence" in sig_df.columns else {}
    for tier in ["Elite", "Enhanced", "Emerging", "Unverified"]:
        print(f"  Prominence {tier}: {tier_counts.get(tier, 0)}")

    print("\n✓  daily_significant.py complete.")
