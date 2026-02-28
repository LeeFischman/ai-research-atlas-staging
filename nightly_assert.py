#!/usr/bin/env python3
# nightly_assert.py
# ──────────────────────────────────────────────────────────────────────────────
# AI Research Atlas — Nightly pipeline health assertions.
#
# Runs after the main daily build (update_map_v2.py) completes. Reads the
# committed artefacts and checks that the pipeline produced sane outputs.
#
# Exit codes
# ──────────
#   0   All checks passed (PASS + WARN only)
#   1   One or more FAIL checks
#
# GitHub Actions annotations
# ──────────────────────────
# When running inside a GitHub Actions job (GITHUB_ACTIONS=true), WARN checks
# emit ::warning:: annotations and FAIL checks emit ::error:: annotations.
# These appear inline in the Actions run summary and as PR check annotations.
#
# Usage
# ─────
#   python nightly_assert.py               # normal run
#   python nightly_assert.py --strict      # treat WARN as FAIL (exit 1)
#
# Workflow integration (daily.yml)
# ────────────────────────────────
#   - name: Run nightly assertions
#     run: python nightly_assert.py
#     # Do not add continue-on-error: true — let this gate the build.
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — mirror atlas_utils.py and update_map_v2.py exactly
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH              = "database.parquet"
SIGNIFICANT_PATH     = "significant.parquet"
SIG_CANDIDATES_PATH  = "sig_candidates.json"
GROUP_NAMES_CACHE    = "group_names_v2.json"
AUTHOR_CACHE_PATH    = "author_cache.json"
SS_CACHE_PATH        = "ss_cache.json"

RETENTION_DAYS             = 14
ARXIV_MAX                  = 400
GROUP_COUNT_MAX            = 20
GROUP_COUNT_MIN_WARN       = 3     # fewer than this is suspicious
DOMINANT_GROUP_WARN_PCT    = 0.60  # one group owning > 60% of papers is suspicious
DOMINANT_GROUP_FAIL_PCT    = 0.80  # one group owning > 80% is almost certainly broken

SIGNIFICANT_POOL_SIZE      = 75
SIGNIFICANT_POOL_MIN       = 20    # below this something went badly wrong
SIGNIFICANT_POOL_MAX       = 80    # above this retirement logic is not firing
SIGNIFICANT_LOOKBACK_DAYS  = 150
SIGNIFICANT_STRIKES_LIMIT  = 2

SIG_CANDIDATES_POOL_SIZE   = 500
SIG_CANDIDATES_MIN         = 50    # below this the pool is depleted
SIG_CANDIDATES_STALE_DAYS  = 10    # last_fetched_date older than this is a WARN

DB_MIN_PAPERS_WARN         = 50    # below this on a weekday is suspicious
DB_MAX_PAPERS_WARN         = 600   # above this pruning may have stopped working

SS_CACHE_TLDR_NULL_WARN    = 0.50  # > 50% of papers with no TLDR is worth flagging

# Columns that must be present and fully non-null in database.parquet
DB_REQUIRED_NONNULL = [
    "id", "title", "abstract", "url",
    "projection_v2_x", "projection_v2_y", "group_id_v2",
    "Prominence", "paper_source", "date_added",
    "ss_citation_count", "ss_influential_citations",
]
# Columns that must be present but may have some nulls or valid empty strings.
# CitationTier uses "" as a valid value (uncited recent papers), so it lives here.
DB_REQUIRED_PRESENT = [
    "CitationTier",
    "text", "label_text", "author_count", "author_tier",
    "authors_list", "author_hindices", "ss_tldr",
]

# Columns that must be present in significant.parquet
SIG_REQUIRED_NONNULL = [
    "id", "title", "abstract", "url", "paper_source",
    "ss_citation_count", "ss_influential_citations",
    "significant_strikes",
]
SIG_REQUIRED_PRESENT = [
    "authors_list", "Prominence", "publication_date", "ss_tldr",
]

VALID_PROMINENCE   = {"Elite", "Enhanced", "Emerging", "Unverified"}
VALID_PAPER_SOURCE = {"Recent", "Significant"}
VALID_CITATION_TIER = {"Very Highly Cited", "Highly Cited", "Cited", ""}
VALID_STRIKES       = {0, 1}

IN_GHA = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


# ══════════════════════════════════════════════════════════════════════════════
# RESULT MODEL
# ══════════════════════════════════════════════════════════════════════════════

Status = Literal["PASS", "WARN", "FAIL"]

@dataclass
class Result:
    status:  Status
    section: str
    message: str


@dataclass
class Checker:
    """Collects results and emits them at the end."""
    results: list[Result] = field(default_factory=list)

    def record(self, status: Status, section: str, message: str) -> None:
        self.results.append(Result(status, section, message))

    def ok(self, section: str, message: str) -> None:
        self.record("PASS", section, message)

    def warn(self, section: str, message: str) -> None:
        self.record("WARN", section, message)

    def fail(self, section: str, message: str) -> None:
        self.record("FAIL", section, message)

    def summary(self, strict: bool = False) -> int:
        """Print all results and return exit code (0 or 1)."""
        n_pass = sum(1 for r in self.results if r.status == "PASS")
        n_warn = sum(1 for r in self.results if r.status == "WARN")
        n_fail = sum(1 for r in self.results if r.status == "FAIL")

        icons = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}
        # Group by section for readable output
        current_section = None
        for r in self.results:
            if r.section != current_section:
                current_section = r.section
                print(f"\n  [{r.section}]")
            line = f"    {icons[r.status]} {r.message}"
            print(line)
            if IN_GHA:
                if r.status == "FAIL" or (r.status == "WARN" and strict):
                    print(f"::error title=Atlas {r.section}::{r.message}")
                elif r.status == "WARN":
                    print(f"::warning title=Atlas {r.section}::{r.message}")

        print(f"\n  {'═'*56}")
        print(f"  PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}")
        failed = n_fail > 0 or (strict and n_warn > 0)
        print(f"  Status: {'FAILED' if failed else 'OK'}")
        print(f"  {'═'*56}")
        return 1 if failed else 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_parquet(path: str) -> pd.DataFrame | None:
    """Return DataFrame or None if file is missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        return None


def _load_json(path: str) -> dict | list | None:
    """Return parsed JSON or None if file is missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _null_rate(series: pd.Series) -> float:
    """Fraction of rows that are null, NaN, or empty string."""
    null_mask = series.isna()
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        null_mask = null_mask | (series.astype(str).str.strip() == "")
    return float(null_mask.mean())


def _is_weekday(dt: datetime) -> bool:
    """True if dt is Monday-Friday UTC."""
    return dt.weekday() < 5     # 0=Mon … 4=Fri


# ══════════════════════════════════════════════════════════════════════════════
# SECTION CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_database(c: Checker, now: datetime) -> pd.DataFrame | None:
    """Assert database.parquet is healthy. Returns the DataFrame on success."""
    sec = "database.parquet"

    df = _load_parquet(DB_PATH)
    if df is None:
        c.fail(sec, f"{DB_PATH} is missing or unreadable — build may have crashed.")
        return None

    n = len(df)
    c.ok(sec, f"File present: {n} rows.")

    # ── Row count ─────────────────────────────────────────────────────────────
    if n == 0:
        c.fail(sec, "Zero rows — pipeline produced an empty database.")
        return df  # continue other checks with empty df

    if n < DB_MIN_PAPERS_WARN:
        if _is_weekday(now):
            c.warn(sec, f"Only {n} papers on a weekday — expected ≥{DB_MIN_PAPERS_WARN}.")
        else:
            c.ok(sec, f"{n} papers (weekend — thin window is normal).")
    elif n > DB_MAX_PAPERS_WARN:
        c.warn(sec, f"{n} papers exceeds {DB_MAX_PAPERS_WARN} — pruning may have stopped.")
    else:
        c.ok(sec, f"Row count {n} within expected range.")

    # ── Required non-null columns ─────────────────────────────────────────────
    missing_cols = [col for col in DB_REQUIRED_NONNULL if col not in df.columns]
    if missing_cols:
        c.fail(sec, f"Missing required columns: {missing_cols}")
    else:
        c.ok(sec, f"All {len(DB_REQUIRED_NONNULL)} non-null columns present.")

    present_missing = [col for col in DB_REQUIRED_PRESENT if col not in df.columns]
    if present_missing:
        c.warn(sec, f"Expected columns missing: {present_missing}")

    if n == 0:
        return df

    # ── Null checks on critical columns ───────────────────────────────────────
    for col in DB_REQUIRED_NONNULL:
        if col not in df.columns:
            continue  # already flagged above
        nr = _null_rate(df[col])
        if nr > 0:
            if nr > 0.10:
                c.fail(sec, f"Column '{col}' has {nr:.1%} null/empty values.")
            else:
                c.warn(sec, f"Column '{col}' has {nr:.1%} null/empty values.")
        # else: no news is good news

    # ── TLDR null rate (allowed to be high — new papers don't have it yet) ────
    if "ss_tldr" in df.columns:
        nr_tldr = _null_rate(df["ss_tldr"])
        if nr_tldr > SS_CACHE_TLDR_NULL_WARN:
            c.warn(sec, f"ss_tldr is null for {nr_tldr:.1%} of papers "
                        f"(S2 may not have indexed them yet — fine for recent papers).")
        else:
            c.ok(sec, f"ss_tldr coverage: {1 - nr_tldr:.1%}.")

    # ── Duplicate IDs ─────────────────────────────────────────────────────────
    if "id" in df.columns:
        n_dupes = int(df["id"].duplicated().sum())
        if n_dupes > 0:
            c.fail(sec, f"{n_dupes} duplicate paper IDs — merge_papers dedup may be broken.")
        else:
            c.ok(sec, "No duplicate IDs.")

    # ── paper_source values ───────────────────────────────────────────────────
    if "paper_source" in df.columns:
        bad_sources = set(df["paper_source"].unique()) - VALID_PAPER_SOURCE
        if bad_sources:
            c.fail(sec, f"Unexpected paper_source values: {bad_sources}")
        recent_count = int((df["paper_source"] == "Recent").sum())
        sig_count    = int((df["paper_source"] == "Significant").sum())
        c.ok(sec, f"paper_source: {recent_count} Recent, {sig_count} Significant.")
        if recent_count == 0:
            c.warn(sec, "No Recent papers in database — arXiv fetch may have returned nothing.")

    # ── Prominence distribution ───────────────────────────────────────────────
    if "Prominence" in df.columns:
        bad_prom = set(df["Prominence"].unique()) - VALID_PROMINENCE
        if bad_prom:
            c.fail(sec, f"Unexpected Prominence values: {bad_prom}")
        pct_unverified = float((df["Prominence"] == "Unverified").mean())
        if pct_unverified == 1.0:
            c.fail(sec, "All papers are 'Unverified' — OpenAlex h-index lookup may have failed entirely.")
        elif pct_unverified > 0.90:
            c.warn(sec, f"{pct_unverified:.1%} papers 'Unverified' — author cache may be sparse.")
        else:
            elite    = int((df["Prominence"] == "Elite").sum())
            enhanced = int((df["Prominence"] == "Enhanced").sum())
            emerging = int((df["Prominence"] == "Emerging").sum())
            unverif  = int((df["Prominence"] == "Unverified").sum())
            c.ok(sec, f"Prominence: Elite={elite}, Enhanced={enhanced}, "
                      f"Emerging={emerging}, Unverified={unverif}.")

    # ── CitationTier distribution ─────────────────────────────────────────────
    if "CitationTier" in df.columns:
        bad_ct = set(df["CitationTier"].unique()) - VALID_CITATION_TIER
        if bad_ct:
            c.fail(sec, f"Unexpected CitationTier values: {bad_ct}")
        else:
            tier_counts = df["CitationTier"].value_counts().to_dict()
            c.ok(sec, f"CitationTier: Very Highly Cited={tier_counts.get('Very Highly Cited', 0)}, "
                      f"Highly Cited={tier_counts.get('Highly Cited', 0)}, "
                      f"Cited={tier_counts.get('Cited', 0)}, "
                      f"uncited={tier_counts.get('', 0)}.")
        # If significant pool is present, we should have Very Highly Cited papers
        if "paper_source" in df.columns:
            has_sig = (df["paper_source"] == "Significant").any()
            if has_sig and (df["CitationTier"] == "Very Highly Cited").sum() == 0:
                c.warn(sec, "Significant papers present but no 'Very Highly Cited' tier — "
                            "CITATION_TIER_TOP_PCT split may not have fired.")

    # ── Group structure ───────────────────────────────────────────────────────
    if "group_id_v2" in df.columns:
        n_groups = int(df["group_id_v2"].nunique())

        if n_groups == 0:
            c.fail(sec, "No groups assigned — Haiku grouping step produced no output.")
        elif n_groups > GROUP_COUNT_MAX:
            c.fail(sec, f"{n_groups} groups exceeds GROUP_COUNT_MAX={GROUP_COUNT_MAX} — "
                        "merge step failed.")
        elif n_groups < GROUP_COUNT_MIN_WARN:
            c.warn(sec, f"Only {n_groups} groups — Haiku may have collapsed too aggressively.")
        else:
            c.ok(sec, f"{n_groups} groups (within expected 3–{GROUP_COUNT_MAX}).")

        # Dominant group check
        if n_groups > 0:
            group_fracs = df["group_id_v2"].value_counts(normalize=True)
            max_frac    = float(group_fracs.iloc[0])
            dom_gid     = int(group_fracs.index[0])
            if max_frac > DOMINANT_GROUP_FAIL_PCT:
                c.fail(sec, f"Group {dom_gid} holds {max_frac:.1%} of papers — "
                            f"nearly all papers collapsed into one group.")
            elif max_frac > DOMINANT_GROUP_WARN_PCT:
                c.warn(sec, f"Group {dom_gid} holds {max_frac:.1%} of papers — "
                            f"unusually dominant (>{DOMINANT_GROUP_WARN_PCT:.0%}).")
            else:
                c.ok(sec, f"Group size distribution OK (largest: {max_frac:.1%}).")

    # ── Projection coordinates populated ─────────────────────────────────────
    for col in ("projection_v2_x", "projection_v2_y"):
        if col in df.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                c.fail(sec, f"{n_nan} NaN values in '{col}' — layout step incomplete.")

    # ── Date recency: all Recent papers should be within RETENTION_DAYS ───────
    if "date_added" in df.columns and "paper_source" in df.columns:
        try:
            dates = pd.to_datetime(df.loc[df["paper_source"] == "Recent", "date_added"],
                                   utc=True, errors="coerce")
            cutoff = now - timedelta(days=RETENTION_DAYS + 1)  # +1 day grace
            stale  = int((dates < cutoff).sum())
            if stale > 0:
                c.warn(sec, f"{stale} Recent papers have date_added older than "
                            f"{RETENTION_DAYS + 1} days — pruning may not have run.")
            else:
                c.ok(sec, "All Recent papers are within the retention window.")
        except Exception:
            pass  # date parse failure is already covered by null check above

    return df


def check_significant(c: Checker, db_df: pd.DataFrame | None, now: datetime) -> pd.DataFrame | None:
    """Assert significant.parquet is healthy. Returns the DataFrame or None."""
    sec = "significant.parquet"

    if not os.path.exists(SIGNIFICANT_PATH):
        c.warn(sec, f"{SIGNIFICANT_PATH} not found — weekly job may not have run yet.")
        return None

    df = _load_parquet(SIGNIFICANT_PATH)
    if df is None:
        c.fail(sec, f"{SIGNIFICANT_PATH} is unreadable.")
        return None

    n = len(df)
    c.ok(sec, f"File present: {n} rows.")

    if n == 0:
        c.fail(sec, "Significant pool is empty.")
        return df

    if n < SIGNIFICANT_POOL_MIN:
        c.fail(sec, f"Only {n} papers — expected ≥{SIGNIFICANT_POOL_MIN}. "
                    f"Pool may have been over-retired.")
    elif n > SIGNIFICANT_POOL_MAX:
        c.warn(sec, f"{n} papers exceeds {SIGNIFICANT_POOL_MAX} — "
                    f"retirement logic may not be firing.")
    else:
        c.ok(sec, f"Pool size {n} within expected range ({SIGNIFICANT_POOL_MIN}–{SIGNIFICANT_POOL_MAX}).")

    # ── Required columns ──────────────────────────────────────────────────────
    missing_cols = [col for col in SIG_REQUIRED_NONNULL if col not in df.columns]
    if missing_cols:
        c.fail(sec, f"Missing required columns: {missing_cols}")

    present_missing = [col for col in SIG_REQUIRED_PRESENT if col not in df.columns]
    if present_missing:
        c.warn(sec, f"Expected columns missing: {present_missing}")

    # ── paper_source must all be "Significant" ────────────────────────────────
    if "paper_source" in df.columns:
        bad = df[df["paper_source"] != "Significant"]
        if len(bad) > 0:
            c.fail(sec, f"{len(bad)} rows have paper_source != 'Significant'.")
        else:
            c.ok(sec, "All rows have paper_source='Significant'.")

    # ── Strikes values must be 0 or 1 ────────────────────────────────────────
    if "significant_strikes" in df.columns:
        bad_strikes = set(df["significant_strikes"].unique()) - VALID_STRIKES
        if bad_strikes:
            c.warn(sec, f"Unexpected significant_strikes values: {bad_strikes} "
                        f"(expected 0 or 1).")
        else:
            n_strike1 = int((df["significant_strikes"] == 1).sum())
            c.ok(sec, f"Strikes distribution: 0={n - n_strike1}, 1={n_strike1}.")

    # ── Citation counts — significant papers should have citations ────────────
    if "ss_citation_count" in df.columns:
        n_zero = int((df["ss_citation_count"].fillna(0) == 0).sum())
        pct_zero = n_zero / n if n else 0.0
        if pct_zero > 0.50:
            c.fail(sec, f"{pct_zero:.1%} of significant papers have zero citations — "
                        f"S2 lookup may have failed or pool contains un-cited papers.")
        elif pct_zero > 0.20:
            c.warn(sec, f"{pct_zero:.1%} of significant papers have zero citations.")
        else:
            cites = df["ss_citation_count"].fillna(0).astype(int)
            c.ok(sec, f"Citations: max={cites.max()}, median={int(cites.median())}, "
                      f"zero={n_zero}/{n}.")

    # ── Prominence: too many Unverified means daily job is behind ─────────────
    if "Prominence" in df.columns:
        pct_unverif = float((df["Prominence"] == "Unverified").mean())
        if pct_unverif > 0.30:
            c.warn(sec, f"{pct_unverif:.1%} of significant papers are 'Unverified' — "
                        f"daily enrichment job may be behind.")
        else:
            c.ok(sec, f"Prominence enrichment: {1 - pct_unverif:.1%} verified.")

    # ── Age check — no paper older than SIGNIFICANT_LOOKBACK_DAYS ────────────
    if "publication_date" in df.columns:
        cutoff_str = (now - timedelta(days=SIGNIFICANT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        valid_dates = df["publication_date"].dropna().astype(str)
        over_age    = int((valid_dates < cutoff_str).sum())
        if over_age > 0:
            c.warn(sec, f"{over_age} paper(s) have publication_date older than "
                        f"{SIGNIFICANT_LOOKBACK_DAYS} days — age-retirement may have missed them.")
        else:
            c.ok(sec, f"All papers are within the {SIGNIFICANT_LOOKBACK_DAYS}-day lookback window.")

    # ── No overlap with Recent window in database.parquet ────────────────────
    if db_df is not None and not db_df.empty and "paper_source" in db_df.columns:
        recent_ids = set(db_df.loc[db_df["paper_source"] == "Recent", "id"].tolist())
        sig_ids    = set(df["id"].tolist())
        overlap    = recent_ids & sig_ids
        if overlap:
            c.warn(sec, f"{len(overlap)} IDs appear in both the Significant pool and "
                        f"the Recent window — deduplication step may have been skipped.")
        else:
            c.ok(sec, "No ID overlap between Significant pool and Recent window.")

    return df


def check_sig_candidates(c: Checker, now: datetime) -> None:
    """Assert sig_candidates.json is healthy."""
    sec = "sig_candidates.json"

    if not os.path.exists(SIG_CANDIDATES_PATH):
        c.warn(sec, f"{SIG_CANDIDATES_PATH} not found — weekly job has not run yet.")
        return

    state = _load_json(SIG_CANDIDATES_PATH)
    if state is None:
        c.fail(sec, "File is unreadable or not valid JSON.")
        return

    c.ok(sec, "File present and parseable.")

    # ── Correct key name: "pool" not "candidates" ─────────────────────────────
    if "candidates" in state and "pool" not in state:
        c.fail(sec, "State uses key 'candidates' instead of 'pool' — "
                    "the daily refresh bug may have been reintroduced.")
    elif "pool" not in state:
        c.fail(sec, "Missing 'pool' key — file may be malformed.")
    else:
        pool = state["pool"]
        n    = len(pool)
        if n < SIG_CANDIDATES_MIN:
            c.fail(sec, f"Pool has only {n} entries (expected ≥{SIG_CANDIDATES_MIN}) — "
                        "OAI discovery may have failed.")
        elif n > SIG_CANDIDATES_POOL_SIZE:
            c.warn(sec, f"Pool has {n} entries (cap={SIG_CANDIDATES_POOL_SIZE}) — "
                        "truncation step may have been skipped.")
        else:
            c.ok(sec, f"Pool size: {n} (cap={SIG_CANDIDATES_POOL_SIZE}).")

        # Check pool entries have citation data
        if pool:
            cites = [p.get("ss_citation_count", 0) for p in pool]
            n_zero = sum(1 for c_ in cites if c_ == 0)
            c.ok(sec, f"Citation range in pool: {min(cites)}–{max(cites)}, {n_zero} zero.")

    # ── last_fetched_date freshness ────────────────────────────────────────────
    last_date_str = state.get("last_fetched_date")
    if not last_date_str:
        c.warn(sec, "Missing 'last_fetched_date' — first-run state may be corrupt.")
    else:
        try:
            last_dt = datetime.strptime(last_date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            age_days = (now - last_dt).days
            if age_days > SIG_CANDIDATES_STALE_DAYS:
                c.warn(sec, f"last_fetched_date={last_date_str} is {age_days} days ago "
                            f"(>{SIG_CANDIDATES_STALE_DAYS}) — weekly job may have missed a run.")
            else:
                c.ok(sec, f"last_fetched_date={last_date_str} ({age_days} days ago).")
        except ValueError:
            c.warn(sec, f"last_fetched_date='{last_date_str}' is not a valid YYYY-MM-DD date.")


def check_group_names(c: Checker, db_df: pd.DataFrame | None) -> None:
    """Assert group_names_v2.json exists and covers all groups in the database."""
    sec = "group_names_v2.json"

    if not os.path.exists(GROUP_NAMES_CACHE):
        c.fail(sec, f"{GROUP_NAMES_CACHE} missing — offline mode will crash on next run.")
        return

    raw = _load_json(GROUP_NAMES_CACHE)
    if raw is None or not isinstance(raw, dict):
        c.fail(sec, "File is unreadable or not a JSON object.")
        return

    c.ok(sec, f"File present: {len(raw)} group names.")

    if db_df is None or db_df.empty or "group_id_v2" not in db_df.columns:
        return

    db_gids     = set(int(g) for g in db_df["group_id_v2"].unique())
    cached_gids = set(int(k) for k in raw.keys())
    missing     = db_gids - cached_gids

    if missing:
        c.fail(sec, f"Groups {sorted(missing)} appear in database.parquet but "
                    f"have no entry in group_names_v2.json — Atlas labels will show 'Group N'.")
    else:
        c.ok(sec, f"All {len(db_gids)} group IDs in database have cached names.")


def check_caches(c: Checker) -> None:
    """Assert author_cache.json and ss_cache.json are present and non-trivial."""
    for path, sec, min_entries in [
        (AUTHOR_CACHE_PATH, "author_cache.json", 100),
        (SS_CACHE_PATH,     "ss_cache.json",     100),
    ]:
        if not os.path.exists(path):
            c.warn(sec, f"{path} not found — will be created on next full run.")
            continue

        raw = _load_json(path)
        if raw is None or not isinstance(raw, dict):
            c.fail(sec, "File is unreadable or not a JSON object.")
            continue

        n = len(raw)
        if n < min_entries:
            c.warn(sec, f"Only {n} entries (expected ≥{min_entries}) — "
                        "cache may have been cleared or is freshly initialized.")
        else:
            c.ok(sec, f"{n} entries.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    strict = "--strict" in sys.argv
    now    = datetime.now(timezone.utc)

    print("═" * 60)
    print(f"  AI Research Atlas — Nightly Assertions")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if strict:
        print("  Mode: STRICT (WARN treated as FAIL)")
    print("═" * 60)

    c = Checker()

    db_df  = check_database(c, now)
    sig_df = check_significant(c, db_df, now)
    check_sig_candidates(c, now)
    check_group_names(c, db_df)
    check_caches(c)

    return c.summary(strict=strict)


if __name__ == "__main__":
    sys.exit(main())
