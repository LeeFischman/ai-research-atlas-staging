#!/usr/bin/env python3
# update_map_v2.py
# ──────────────────────────────────────────────────────────────────────────────
# AI Research Atlas — v2 pipeline.
#
# Key difference from update_map.py (v1):
#   v1: HDBSCAN groups papers by SPECTER2 geometry, Haiku labels each cluster.
#   v2: Haiku groups papers by research meaning, geometry follows from those groups.
#
# Pipeline (4 stages + build)
# ──────────────────────────────────────────────
# 1. FETCH       arXiv → new papers
# 2. EMBED       SPECTER2 incremental embed + UMAP → projection_x/y for direction vectors
# 3. GROUP       Single Haiku call → group_id_v2 + group_name per paper (12-18 groups)
#                Haiku names the groups as it forms them; no second labeling call needed.
#                If Haiku returns > GROUP_COUNT_MAX groups, excess groups are merged
#                by repeatedly absorbing the closest pair (SPECTER2 distance).
# 4. LAYOUT      MDS on group-to-group SPECTER2 distances → 2D group centroids,
#                multiplied by LAYOUT_SCALE for human-readable coordinate range.
#                Within each group: scatter papers using SPECTER2 direction vectors
#                + variance-proportional scatter scale, also × LAYOUT_SCALE.
#                Writes projection_v2_x / projection_v2_y.
# 5. BUILD       embedding-atlas CLI + deploy
#
# Projection columns written (coexist with v1 columns in database.parquet):
#   group_id_v2     — Haiku group assignment (int, post-merge)
#   projection_v2_x — v2 layout x
#   projection_v2_y — v2 layout y
#
# Offline mode (re-runs layout + build only, skipping all API/embed work):
#   OFFLINE_MODE=true python update_map_v2.py
#   Requires: database.parquet with group_id_v2 + embedding columns, and
#             group_names_v2.json saved from a previous normal run.
#
# Normal run:
#   ANTHROPIC_API_KEY=sk-... python update_map_v2.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import random
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from math import sqrt

import numpy as np
import pandas as pd

import arxiv

# Shared utilities — does NOT modify update_map.py
from atlas_utils import (
    DB_PATH,
    ARXIV_MAX,
    _strip_urls,
    scrub_model_words,
    calculate_reputation,
    categorize_authors,
    load_existing_db,
    merge_papers,
    fetch_arxiv,
    embed_and_project,
    build_and_deploy_atlas,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — all tuning knobs in one place
# ══════════════════════════════════════════════════════════════════════════════

# ── Offline mode ─────────────────────────────────────────────────────────────
# When True: skip arXiv fetch, SPECTER2 embedding, and Haiku API calls entirely.
# Re-runs MDS layout, scatter, and Atlas build using data already in the parquet.
# Set via env var:  OFFLINE_MODE=true python update_map_v2.py
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "false").strip().lower() == "true"

# Cache file written after every successful Haiku grouping call.
# Loaded automatically in offline mode.
GROUP_NAMES_CACHE = "group_names_v2.json"

# ── Haiku grouping ───────────────────────────────────────────────────────────

# Target group count range sent to Haiku in the prompt.
# If Haiku returns more than GROUP_COUNT_MAX groups, excess groups are merged
# down automatically — no retry needed for over-grouping.
# Recommended range: 10-20.
GROUP_COUNT_MIN = 12
GROUP_COUNT_MAX = 18

# Characters of each abstract sent to Haiku.
# Recommended range: 150-400.  Shorter = cheaper; longer = better grouping.
ABSTRACT_GROUPING_CHARS = 300

# Retry policy for the Haiku grouping call.
# Covers both 529 overload errors and hard parse failures.
# Wait schedule: GROUPING_RETRY_BASE_WAIT * 2^(attempt-1) seconds.
# With base=60 and 5 retries: 60, 120, 240, 480 s between attempts.
GROUPING_MAX_RETRIES     = 5
GROUPING_RETRY_BASE_WAIT = 60   # seconds; recommended range: 30-120

# ── Layout scaling ────────────────────────────────────────────────────────────
# MDS raw output is in a small range (typically ±0.15 for ~15 groups on
# SPECTER2 cosine distances, which cluster between 0.05 and 0.25).
# LAYOUT_SCALE multiplies ALL v2 coordinates — centroid positions AND
# within-group scatter distances — so the final layout is visible in Atlas.
#
# With the run that produced the attached log:
#   Raw centroid range: ±0.17   →  ×20  →  ±3.4
#   Scatter radius:     ~0.06   →  ×20  →  ~1.2
#   Total canvas:       ~0.34   →  ×20  →  ~6.8  (comfortably label-able)
#
# Recommended range: 15-30.  Increase if clusters still look cramped in Atlas.
LAYOUT_SCALE = 20.0

# ── Within-group scatter ─────────────────────────────────────────────────────
# Scatter radius is expressed as a fraction of the median nearest-neighbour
# centroid distance, so it auto-scales to however spread the MDS layout is.
#
# radius = SCATTER_FRACTION
#         * median_nearest_centroid_dist
#         * (1 + group_variance * VARIANCE_AMPLIFIER)
#
# SCATTER_FRACTION = 0.35 means each paper is placed at most ~35% of the way
# to its nearest neighbouring cluster centroid.  Papers at the edge of a group
# (high intra-group cosine distance) get a larger fraction; papers at the core
# get a smaller fraction.  This prevents clouds from overlapping at any scale.
# Recommended range: 0.25-0.50.
#
# VARIANCE_AMPLIFIER loosely-coupled groups spread a bit wider than tight ones.
# Recommended range: 1.0-3.0.
SCATTER_FRACTION   = 0.35
VARIANCE_AMPLIFIER = 2.0

# ── Layout tuning guide ──────────────────────────────────────────────────────
# Quick reference for adjusting the visual appearance of the atlas:
#
# Clusters overlap / labels still central:
#   → Lower SCATTER_FRACTION (0.35 → 0.25)
#     Clouds tighten; more breathing room between groups.
#
# Clusters too sparse / papers far from their label:
#   → Raise SCATTER_FRACTION (0.35 → 0.45)
#
# Merge step absorbing groups that feel semantically distinct:
#   → Raise GROUP_COUNT_MAX (18 → 20 or 22)
#     Haiku's natural granularity may warrant a higher ceiling.
#
# Haiku consistently returns too few groups (< GROUP_COUNT_MIN):
#   → Lower GROUP_COUNT_MIN (12 → 10), or reduce ABSTRACT_GROUPING_CHARS
#     so Haiku sees less detail and forms coarser groups.
#
# Ready for production (fresh papers + re-grouping):
#   → Remove OFFLINE_MODE from the workflow YAML.
#     First normal run re-embeds, calls Haiku, and commits updated
#     group_names_v2.json to the repo for future offline runs.

# ── Atlas CLI ────────────────────────────────────────────────────────────────
# Projection column names written to database.parquet.
# Must match --x / --y args passed to the Atlas CLI.
PROJ_X_COL = "projection_v2_x"
PROJ_Y_COL = "projection_v2_y"

# ── Haiku model ──────────────────────────────────────────────────────────────
HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — HAIKU GROUPING
# ══════════════════════════════════════════════════════════════════════════════

_GROUPING_SYSTEM = (
    "You are a research taxonomy expert. You will be given a list of AI research "
    "paper titles and abstracts. Your task is to group them into thematically "
    "coherent clusters based on shared research methodology, problem formulation, "
    "or application domain.\n\n"
    "Rules:\n"
    "- Assign every paper to exactly one group.\n"
    f"- Use between {GROUP_COUNT_MIN} and {GROUP_COUNT_MAX} groups total. "
    "Do not create more or fewer.\n"
    "- Group papers by what makes them intellectually related, not just surface "
    "topic keywords. Papers that share a methodology or theoretical framework "
    "should be grouped together even if their application domains differ.\n"
    "- Give each group a concise 3-6 word name capturing the shared intellectual "
    "thread (methodology or framing, not just topic keywords). "
    "Prefer noun phrases: 'Sparse Reward Policy Learning' not 'Papers About RL'.\n"
    "- Respond ONLY with a JSON array. No preamble, no explanation, no markdown "
    "fences. The array must have exactly one entry per paper in the same order "
    "as the input, with this structure:\n"
    '[{"index": 0, "group_id": 0, "group_name": "Sparse Reward Policy Learning"}, '
    '{"index": 1, "group_id": 3, "group_name": "Multimodal Instruction Tuning"}, ...]\n'
    "- group_id must be an integer from 0 to (number_of_groups - 1).\n"
    "- group_name must be identical for every entry that shares the same group_id."
)


def _build_grouping_user_message(df: pd.DataFrame) -> str:
    lines = [f"Assign each of the following {len(df)} papers to a group. "
             "Return JSON only.\n"]
    for i, row in df.iterrows():
        title            = str(row["title"]).strip()
        abstract         = _strip_urls(str(row.get("abstract", ""))).strip()
        abstract_snippet = abstract[:ABSTRACT_GROUPING_CHARS]
        lines.append(f"[{i}] Title: {title}\nAbstract: {abstract_snippet}")
    return "\n\n".join(lines)


def _parse_grouping_response(
    text: str, n_papers: int
) -> tuple[dict[int, int], dict[int, str]] | None:
    """Parse and validate Haiku's JSON grouping response.

    Hard failures → return None → trigger retry:
      - Non-JSON or non-list
      - Wrong entry count
      - Missing / non-integer fields
      - Fewer than GROUP_COUNT_MIN groups

    Soft acceptance → log warning, return result:
      - More than GROUP_COUNT_MAX groups (caller merges excess groups down)

    group_name drift: majority vote across all entries for each group_id.
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return None

    if not isinstance(data, list):
        print(f"    Expected JSON list, got {type(data).__name__}.")
        return None

    if len(data) != n_papers:
        print(f"    Expected {n_papers} entries, got {len(data)}.")
        return None

    mapping: dict[int, int] = {}
    name_votes: dict[int, dict[str, int]] = {}

    for entry in data:
        if not isinstance(entry, dict) or "index" not in entry or "group_id" not in entry:
            print(f"    Malformed entry: {entry}")
            return None
        idx = entry["index"]
        gid = entry["group_id"]
        if not isinstance(idx, int) or not isinstance(gid, int) or gid < 0:
            print(f"    Non-integer or negative values: index={idx}, group_id={gid}")
            return None
        mapping[idx] = gid
        name = str(entry.get("group_name", "")).strip().title()
        if name:
            name_votes.setdefault(gid, {})
            name_votes[gid][name] = name_votes[gid].get(name, 0) + 1

    missing = set(range(n_papers)) - set(mapping.keys())
    if missing:
        print(f"    Missing paper indices: {sorted(missing)[:10]}...")
        return None

    n_groups = len(set(mapping.values()))
    if n_groups < GROUP_COUNT_MIN:
        print(f"    Got {n_groups} groups — below minimum {GROUP_COUNT_MIN}. Rejecting.")
        return None
    if n_groups > GROUP_COUNT_MAX:
        print(f"    Got {n_groups} groups — above maximum {GROUP_COUNT_MAX}. "
              "Will merge excess groups after parsing.")

    group_names: dict[int, str] = {}
    for gid in set(mapping.values()):
        votes = name_votes.get(gid, {})
        if votes:
            winner = max(votes, key=votes.__getitem__)
            if len(votes) > 1:
                print(f"    Group {gid}: name drift resolved → '{winner}' "
                      f"(from {dict(votes)})")
            group_names[gid] = winner
        else:
            group_names[gid] = f"Group {gid}"
            print(f"    Group {gid}: no name in response — fallback '{group_names[gid]}'")

    return mapping, group_names


def _merge_excess_groups(
    mapping: dict[int, int],
    group_names: dict[int, str],
    df: pd.DataFrame,
    target_max: int,
) -> tuple[dict[int, int], dict[int, str]]:
    """Merge excess groups down to target_max by repeatedly absorbing the
    closest pair (smallest mean inter-group SPECTER2 cosine distance).
    The larger group keeps its name; remaps group_ids to 0..n-1 afterward.
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    n_groups = len(set(mapping.values()))
    if n_groups <= target_max:
        return mapping, group_names

    print(f"\n  Merging {n_groups} → {target_max} groups (absorbing closest pairs)...")

    all_vecs     = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
    all_pairwise = cosine_distances(all_vecs)

    # group_id → list of integer row positions
    active: dict[int, list[int]] = {}
    for pos, gid in enumerate(mapping.values()):
        active.setdefault(gid, []).append(pos)

    def mean_inter(a, b):
        return float(all_pairwise[np.ix_(a, b)].mean())

    while len(active) > target_max:
        gids = sorted(active.keys())
        best_dist, best_pair = float("inf"), None
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                d = mean_inter(active[gids[i]], active[gids[j]])
                if d < best_dist:
                    best_dist, best_pair = d, (gids[i], gids[j])

        ga, gb = best_pair
        keep   = ga if len(active[ga]) >= len(active[gb]) else gb
        absorb = gb if keep == ga else ga
        print(f"    Merge group {absorb} ('{group_names[absorb]}', n={len(active[absorb])}) "
              f"→ group {keep} ('{group_names[keep]}', n={len(active[keep])})  "
              f"dist={best_dist:.4f}")

        active[keep].extend(active.pop(absorb))
        del group_names[absorb]
        for pos in active[keep]:
            mapping[pos] = keep

    # Remap group_ids to clean 0..n-1
    id_remap    = {old: new for new, old in enumerate(sorted(active.keys()))}
    new_mapping = {pos: id_remap[g] for pos, g in mapping.items()}
    new_names   = {id_remap[old]: name for old, name in group_names.items()}
    print(f"  Merge complete: {len(active)} groups remaining.")
    return new_mapping, new_names


def haiku_group_papers(
    df: pd.DataFrame,
    client,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Stage 3: Haiku assigns every paper to a group and names each group.

    Returns (df, group_names):
      df          — with group_id_v2 column added
      group_names — {group_id: label_text}

    Saves group_names to GROUP_NAMES_CACHE for offline mode reuse.
    Falls back to HDBSCAN if all Haiku attempts hard-fail.
    """
    df = df.reset_index(drop=True)
    n  = len(df)
    print(f"\n▶  Stage 3 — Haiku grouping + naming ({n} papers)...")

    user_msg      = _build_grouping_user_message(df)
    approx_tokens = len(user_msg) // 4
    print(f"  Grouping prompt ≈ {approx_tokens:,} tokens "
          f"(abstracts truncated to {ABSTRACT_GROUPING_CHARS} chars each).")

    result = None
    for attempt in range(1, GROUPING_MAX_RETRIES + 1):
        print(f"  Haiku grouping call — attempt {attempt}/{GROUPING_MAX_RETRIES}...")
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=8192,   # 250 entries × ~55 chars ≈ 3.4K tokens; Haiku max for headroom
                system=_GROUPING_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            print(f"  Response length: {len(raw)} chars.")
            result = _parse_grouping_response(raw, n)
            if result is not None:
                mapping, group_names = result
                print(f"  ✓ Grouping parsed ({len(set(mapping.values()))} groups "
                      f"before any merging).")
                break
        except Exception as e:
            err_str = str(e)
            is_529  = "529" in err_str or "overloaded" in err_str.lower()
            label   = "API overloaded (529)" if is_529 else "API error"
            print(f"  {label} on attempt {attempt}: {e}")

        if attempt < GROUPING_MAX_RETRIES:
            wait = GROUPING_RETRY_BASE_WAIT * (2 ** (attempt - 1))
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    if result is None:
        print("  ✗ All Haiku attempts failed. Falling back to HDBSCAN.")
        mapping     = _hdbscan_fallback_grouping(df)
        group_names = {gid: f"Group {gid}" for gid in set(mapping.values())}
    else:
        # Merge down if Haiku returned more groups than the maximum
        if len(set(mapping.values())) > GROUP_COUNT_MAX:
            mapping, group_names = _merge_excess_groups(
                mapping, group_names, df, target_max=GROUP_COUNT_MAX
            )

    df["group_id_v2"] = [mapping[i] for i in range(n)]
    group_counts = df["group_id_v2"].value_counts().sort_index()
    for gid, count in group_counts.items():
        print(f"    Group {gid:>2} ({count:>3} papers): '{group_names.get(gid, '?')}'")
    print(f"  Total: {len(group_counts)} groups, {n} papers assigned.")

    # Cache for offline mode
    with open(GROUP_NAMES_CACHE, "w") as f:
        json.dump({str(k): v for k, v in group_names.items()}, f, indent=2)
    print(f"  Group names cached to {GROUP_NAMES_CACHE}.")

    return df, group_names


def _load_group_names_cache(df: pd.DataFrame | None = None) -> dict[int, str]:
    """Load group names from cache (used in offline mode).

    If the cache file is missing, falls back to deriving generic names from
    the group_id_v2 column of df (if provided) rather than crashing.
    """
    if os.path.exists(GROUP_NAMES_CACHE):
        with open(GROUP_NAMES_CACHE) as f:
            raw = json.load(f)
        names = {int(k): v for k, v in raw.items()}
        print(f"  Loaded {len(names)} group names from {GROUP_NAMES_CACHE}.")
        return names

    # Cache missing — fall back to generic names derived from the parquet
    print(f"  WARNING: Group names cache '{GROUP_NAMES_CACHE}' not found.")
    if df is not None and "group_id_v2" in df.columns:
        gids  = sorted(int(g) for g in df["group_id_v2"].unique())
        names = {gid: f"Group {gid}" for gid in gids}
        print(f"  Using generic names for {len(names)} groups. "
              "Run once in normal mode to generate meaningful labels.")
        return names
    raise FileNotFoundError(
        f"Group names cache '{GROUP_NAMES_CACHE}' not found and no dataframe "
        "provided as fallback. Run once in normal mode first."
    )


def _hdbscan_fallback_grouping(df: pd.DataFrame) -> dict[int, int]:
    """Fallback HDBSCAN grouping if all Haiku calls fail."""
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("  Fallback HDBSCAN grouping...")

    if "embedding_50d" in df.columns and df["embedding_50d"].notna().all():
        X, metric = np.array(df["embedding_50d"].tolist(), dtype=np.float32), "cosine"
        print("  Clustering on 50D SPECTER2 (cosine).")
    elif "embedding" in df.columns and df["embedding"].notna().all():
        X = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
        metric = "cosine"
        print("  Clustering on 768D SPECTER2 (cosine).")
    else:
        print("  No embedding available — random group assignment.")
        return {i: i % GROUP_COUNT_MIN for i in range(len(df))}

    for mcs in [max(3, len(df) // 30), max(3, len(df) // 40), 3]:
        clusterer  = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                             metric=metric, cluster_selection_method="leaf")
        labels     = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  HDBSCAN min_cluster_size={mcs} → {n_clusters} clusters.")
        if GROUP_COUNT_MIN <= n_clusters <= GROUP_COUNT_MAX + 5:
            break

    unique_clusters = [c for c in sorted(set(labels)) if c != -1]
    if not unique_clusters:
        return {i: 0 for i in range(len(df))}

    centroids  = np.array([X[labels == c].mean(axis=0) for c in unique_clusters])
    noise_mask = labels == -1
    if noise_mask.any():
        from sklearn.metrics.pairwise import cosine_distances as _cd
        nearest = _cd(X[noise_mask], centroids).argmin(axis=1)
        labels  = labels.copy()
        labels[noise_mask] = [unique_clusters[j] for j in nearest]
        print(f"  Reassigned {noise_mask.sum()} noise points.")

    id_map = {old: new for new, old in enumerate(unique_clusters)}
    return {i: id_map.get(int(labels[i]), 0) for i in range(len(df))}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MDS LAYOUT + WITHIN-GROUP SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def compute_mds_centroids(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Stage 4a: MDS on group-to-group SPECTER2 distances → scaled 2D centroids.

    Raw MDS output is multiplied by LAYOUT_SCALE so centroid coordinates land
    in a useful range for Atlas rendering (see LAYOUT_SCALE config comment).

    Returns: {group_id: (cx, cy)} — coordinates already include LAYOUT_SCALE.
    """
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("\n▶  Stage 4a — MDS between-group layout...")

    if "embedding" not in df.columns or df["embedding"].isna().any():
        raise RuntimeError("SPECTER2 'embedding' column missing or has NaNs.")

    all_vecs  = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
    group_ids = sorted(df["group_id_v2"].unique())
    n_groups  = len(group_ids)
    print(f"  Computing {n_groups}×{n_groups} group distance matrix...")

    # group_id → list of integer row positions in df
    group_pos: dict[int, list[int]] = {gid: [] for gid in group_ids}
    for pos, gid in enumerate(df["group_id_v2"].tolist()):
        group_pos[gid].append(pos)

    all_pairwise = cosine_distances(all_vecs)

    group_dist = np.zeros((n_groups, n_groups), dtype=np.float32)
    for i, gid_i in enumerate(group_ids):
        for j, gid_j in enumerate(group_ids):
            if i == j:
                continue
            group_dist[i, j] = float(
                all_pairwise[np.ix_(group_pos[gid_i], group_pos[gid_j])].mean()
            )
    group_dist = (group_dist + group_dist.T) / 2
    np.fill_diagonal(group_dist, 0.0)

    print(f"  Group dist: min={group_dist.min():.4f}, "
          f"max={group_dist.max():.4f}, mean={group_dist.mean():.4f}")

    mds = MDS(n_components=2, metric=True, dissimilarity="precomputed",
              random_state=42, n_init=4, max_iter=500, normalized_stress="auto")
    raw_coords   = mds.fit_transform(group_dist)          # shape (n_groups, 2)
    group_coords = raw_coords * LAYOUT_SCALE               # ← scale applied here
    print(f"  MDS stress: {mds.stress_:.6f}  (lower is better)")
    print(f"  LAYOUT_SCALE={LAYOUT_SCALE}×  raw range "
          f"[{raw_coords.min():.3f}, {raw_coords.max():.3f}] → "
          f"scaled [{group_coords.min():.2f}, {group_coords.max():.2f}]")

    centroids: dict[int, tuple[float, float]] = {}
    for i, gid in enumerate(group_ids):
        cx, cy = float(group_coords[i, 0]), float(group_coords[i, 1])
        centroids[gid] = (cx, cy)
        print(f"  Group {gid:>2}: centroid=({cx:+.2f}, {cy:+.2f}), "
              f"n={len(group_pos[gid])}")

    return centroids


def scatter_within_groups(
    df: pd.DataFrame,
    centroids: dict[int, tuple[float, float]],
    group_names: dict[int, str],
) -> pd.DataFrame:
    """Stage 4b: Place each paper around its MDS centroid.

    Scatter radius is expressed as a fraction of the median nearest-neighbour
    centroid distance, so it auto-scales regardless of LAYOUT_SCALE or how
    semantically similar the groups are to each other.

    Position formula:
      direction    = unit vector from SPECTER2 group centroid → paper's UMAP pos
      base_radius  = median_nn_centroid_dist * SCATTER_FRACTION
      scatter_dist = base_radius
                   * (paper_mean_intra_dist / group_mean_intra_dist)
                   * (1 + group_variance * VARIANCE_AMPLIFIER)
      final_pos    = mds_centroid + direction * scatter_dist

    Normalising by group_mean_intra_dist ensures papers at the edge of a group
    scatter proportionally further than papers at the core, without the raw
    cosine distance magnitude inflating the radius when groups are loose.

    Writes: projection_v2_x, projection_v2_y
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("\n▶  Stage 4b — Within-group scatter (fraction of centroid spacing)...")

    df = df.copy()
    df["projection_v2_x"] = np.nan
    df["projection_v2_y"] = np.nan

    all_vecs   = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
    specter2_x = df["projection_x"].values
    specter2_y = df["projection_y"].values

    # ── Compute median nearest-neighbour centroid distance ───────────────────
    # This is the reference scale for scatter radius.
    gids       = sorted(centroids.keys())
    c_array    = np.array([centroids[g] for g in gids])   # (n_groups, 2)
    nn_dists   = []
    for i in range(len(gids)):
        dists_to_others = [
            sqrt((c_array[i,0]-c_array[j,0])**2 + (c_array[i,1]-c_array[j,1])**2)
            for j in range(len(gids)) if j != i
        ]
        nn_dists.append(min(dists_to_others))
    median_nn_dist = float(np.median(nn_dists))
    base_radius    = median_nn_dist * SCATTER_FRACTION
    print(f"  Median nearest-neighbour centroid dist: {median_nn_dist:.3f}")
    print(f"  Base scatter radius (SCATTER_FRACTION={SCATTER_FRACTION}): "
          f"{base_radius:.3f}  ({SCATTER_FRACTION*100:.0f}% of centroid spacing)")

    for gid in gids:
        mask      = df["group_id_v2"] == gid
        positions = df.index[mask].tolist()
        pos_array = [df.index.get_loc(p) for p in positions]
        n_g       = len(positions)
        name      = group_names.get(gid, f"Group {gid}")
        mds_cx, mds_cy = centroids[gid]

        if n_g == 1:
            df.at[positions[0], "projection_v2_x"] = mds_cx
            df.at[positions[0], "projection_v2_y"] = mds_cy
            print(f"  Group {gid:>2} ('{name}'): singleton at centroid.")
            continue

        g_vecs         = all_vecs[pos_array]
        pairwise       = cosine_distances(g_vecs)
        mean_dists     = pairwise.mean(axis=1)          # per-paper mean distance
        group_mean     = float(mean_dists.mean())       # group-level mean
        upper          = pairwise[np.triu_indices(n_g, k=1)]
        group_variance = float(upper.std()) if len(upper) > 0 else 0.0
        variance_boost = 1.0 + group_variance * VARIANCE_AMPLIFIER

        # Effective radius for this group
        eff_radius = base_radius * variance_boost
        print(f"  Group {gid:>2} ('{name}'): "
              f"n={n_g:>3}, var={group_variance:.4f}, "
              f"eff_radius={eff_radius:.3f}, "
              f"centroid=({mds_cx:+.2f}, {mds_cy:+.2f})")

        sp_cx = specter2_x[pos_array].mean()
        sp_cy = specter2_y[pos_array].mean()

        for local_i, (df_idx, pos_i) in enumerate(zip(positions, pos_array)):
            dx     = specter2_x[pos_i] - sp_cx
            dy     = specter2_y[pos_i] - sp_cy
            length = sqrt(dx * dx + dy * dy)
            if length < 1e-8:
                angle  = random.uniform(0, 2 * 3.14159265)
                dx, dy = float(np.cos(angle)), float(np.sin(angle))
                length = 1.0
            # Scale per-paper distance relative to group mean so core papers
            # cluster tightly and edge papers drift proportionally further out
            rel_dist     = (mean_dists[local_i] / group_mean) if group_mean > 1e-8 else 1.0
            scatter_dist = eff_radius * rel_dist
            df.at[df_idx, "projection_v2_x"] = mds_cx + (dx / length) * scatter_dist
            df.at[df_idx, "projection_v2_y"] = mds_cy + (dy / length) * scatter_dist

    n_placed = df["projection_v2_x"].notna().sum()
    x_range  = (df["projection_v2_x"].min(), df["projection_v2_x"].max())
    y_range  = (df["projection_v2_y"].min(), df["projection_v2_y"].max())
    print(f"\n  Placed {n_placed}/{len(df)} papers.")
    print(f"  Final layout bounds: x=[{x_range[0]:.2f}, {x_range[1]:.2f}], "
          f"y=[{y_range[0]:.2f}, {y_range[1]:.2f}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WRITE LABELS PARQUET
# ══════════════════════════════════════════════════════════════════════════════

def write_labels_parquet(
    df: pd.DataFrame,
    labels: dict[int, str],
    centroids: dict[int, tuple[float, float]],
    labels_path: str = "labels_v2.parquet",
) -> str:
    """Write labels.parquet for the Atlas CLI --labels flag.

    Label x/y are the mean of each group's actual projection_v2_x/y positions —
    the visual centre of the dot cloud after scatter — not the MDS centroid.
    This keeps labels anchored to where their papers actually landed, regardless
    of how far the scatter pushed them from the centroid.
    """
    print(f"\n▶  Writing labels parquet ({len(centroids)} labels)...")
    rows = []
    for gid in sorted(centroids.keys()):
        label_text = labels.get(gid, f"Group {gid}")
        mask       = df["group_id_v2"] == gid
        n_papers   = int(mask.sum())
        # Mean of actual paper positions — tracks where dots landed after scatter
        cx = float(df.loc[mask, "projection_v2_x"].mean())
        cy = float(df.loc[mask, "projection_v2_y"].mean())
        rows.append({"x": cx, "y": cy, "text": label_text,
                     "level": 0, "priority": 10})
        print(f"  [{gid:>2}] '{label_text}' @ ({cx:+.2f}, {cy:+.2f}), {n_papers} papers")

    labels_df = pd.DataFrame(rows)
    labels_df.to_parquet(labels_path, index=False)
    print(f"  Wrote {len(labels_df)} labels → {labels_path}.")
    return labels_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import anthropic

    now      = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    print("=" * 60)
    print(f"  AI Research Atlas v2 — {run_date} UTC")
    print(f"  OFFLINE_MODE       : {OFFLINE_MODE}")
    print(f"  GROUP_COUNT        : {GROUP_COUNT_MIN}–{GROUP_COUNT_MAX}")
    print(f"  LAYOUT_SCALE       : {LAYOUT_SCALE}")
    print(f"  SCATTER_FRACTION   : {SCATTER_FRACTION}")
    print(f"  VARIANCE_AMPLIFIER : {VARIANCE_AMPLIFIER}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    # OFFLINE MODE — re-run layout + build only
    # ══════════════════════════════════════════════════════════════════════════
    if OFFLINE_MODE:
        print("\n▶  OFFLINE MODE — loading existing data, skipping all API calls...")

        df = load_existing_db()
        if df.empty:
            raise RuntimeError(
                "No database.parquet found. Run once in normal mode first."
            )
        if "group_id_v2" not in df.columns:
            raise RuntimeError(
                "database.parquet has no group_id_v2 column. "
                "Run once in normal mode to populate it, then retry offline."
            )
        if "embedding" not in df.columns or df["embedding"].isna().any():
            raise RuntimeError(
                "database.parquet is missing SPECTER2 embeddings. "
                "Run once in normal mode to embed, then retry offline."
            )
        if "projection_x" not in df.columns or df["projection_x"].isna().any():
            raise RuntimeError(
                "database.parquet is missing projection_x/y (SPECTER2 UMAP). "
                "Run once in normal mode first."
            )

        group_names = _load_group_names_cache(df)

        # Patch any group IDs in the parquet that are missing from the cache
        parquet_gids  = set(int(g) for g in df["group_id_v2"].unique())
        missing_names = parquet_gids - set(group_names.keys())
        if missing_names:
            print(f"  WARNING: {len(missing_names)} group IDs have no cached name "
                  f"{missing_names} — using 'Group N' fallback.")
            for gid in missing_names:
                group_names[gid] = f"Group {gid}"

        print(f"  Loaded {len(df)} papers, "
              f"{df['group_id_v2'].nunique()} groups, "
              f"{len(group_names)} cached names.")

    # ══════════════════════════════════════════════════════════════════════════
    # NORMAL MODE — full pipeline
    # ══════════════════════════════════════════════════════════════════════════
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to GitHub repo secrets "
                "and expose it in the workflow YAML under env:."
            )
        haiku_client = anthropic.Anthropic(api_key=api_key)

        # ── Stage 1: Load & prune rolling DB ────────────────────────────────
        print("\n▶  Stage 1 — Loading rolling database...")
        existing_df  = load_existing_db()
        is_first_run = existing_df.empty and not os.path.exists(DB_PATH)
        days_back    = 5 if is_first_run else 2

        if is_first_run:
            print("  First run — pre-filling with last 5 days of arXiv papers.")
        else:
            print(f"  Loaded {len(existing_df)} existing papers.")

        # ── Stage 1b: arXiv fetch ────────────────────────────────────────────
        print("\n▶  Stage 1b — Fetching from arXiv...")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent",
            "ai-research-atlas/2.0 (https://github.com/LeeFischman/ai-research-atlas; "
            "mailto:lee.fischman@gmail.com)")]
        urllib.request.install_opener(opener)

        arxiv_client = arxiv.Client(page_size=100, delay_seconds=10)
        search = arxiv.Search(
            query=(
                f"cat:cs.AI AND submittedDate:"
                f"[{(now - timedelta(days=days_back)).strftime('%Y%m%d%H%M')}"
                f" TO {now.strftime('%Y%m%d%H%M')}]"
            ),
            max_results=ARXIV_MAX,
        )
        results = fetch_arxiv(arxiv_client, search)

        if not results:
            if existing_df.empty:
                print("  No arXiv results and no existing DB. Exiting.")
                exit(0)
            print(f"  No new papers (weekend / dry spell). "
                  f"Rebuilding from {len(existing_df)} existing papers.")

        if results:
            print(f"  Fetched {len(results)} papers from arXiv.")
            today_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            rows = []
            for r in results:
                title      = r.title
                abstract   = r.summary
                scrubbed   = scrub_model_words(f"{title}. {title}. {abstract}")
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
                    "date_added":   today_str,
                })
            new_df = pd.DataFrame(rows)
            new_df["Reputation"] = new_df.apply(calculate_reputation, axis=1)
            df = merge_papers(existing_df, new_df)
            df = df.drop(columns=["group", "group_id_v2"], errors="ignore")
            print(f"  Rolling DB: {len(df)} papers after merge.")
        else:
            df = existing_df.drop(columns=["group", "group_id_v2"], errors="ignore")

        # Backfill reputation for older rows
        if "Reputation" not in df.columns or df["Reputation"].isna().any():
            missing = (df["Reputation"].isna() if "Reputation" in df.columns
                       else pd.Series([True] * len(df)))
            if missing.any():
                print(f"  Backfilling Reputation for {missing.sum()} rows...")
                df.loc[missing, "Reputation"] = df.loc[missing].apply(
                    calculate_reputation, axis=1)

        # ── Stage 2: SPECTER2 embed + UMAP ──────────────────────────────────
        print("\n▶  Stage 2 — SPECTER2 embedding + UMAP...")
        df = embed_and_project(df, model_name="specter2")

        # ── Stage 3: Haiku grouping + naming ────────────────────────────────
        df, group_names = haiku_group_papers(df, haiku_client)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGES 4-5: Layout + build — run in both normal AND offline modes
    # ══════════════════════════════════════════════════════════════════════════

    # ── Stage 4a: MDS centroids ──────────────────────────────────────────────
    centroids = compute_mds_centroids(df)

    # ── Stage 4b: Within-group scatter ───────────────────────────────────────
    df = scatter_within_groups(df, centroids, group_names)

    # ── Write labels parquet ─────────────────────────────────────────────────
    labels_path = write_labels_parquet(df, group_names, centroids)

    # ── Save rolling DB ───────────────────────────────────────────────────────
    print("\n▶  Saving rolling database...")
    save_df = df.copy()
    save_df["date_added"] = pd.to_datetime(
        save_df["date_added"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if "projection_v2_x" not in save_df.columns:
        raise RuntimeError("projection_v2_x missing — layout step failed.")

    save_df.to_parquet(DB_PATH, index=False)
    print(f"  Saved {len(save_df)} papers to {DB_PATH}.")
    proj_cols = [c for c in save_df.columns if "projection" in c or "group_id" in c]
    print(f"  Projection / group columns: {proj_cols}")

    # ── Stage 5: Build + deploy ───────────────────────────────────────────────
    print("\n▶  Stage 5 — Building atlas...")
    build_and_deploy_atlas(
        db_path     = DB_PATH,
        proj_x_col  = PROJ_X_COL,
        proj_y_col  = PROJ_Y_COL,
        labels_path = labels_path,
        run_date    = run_date,
    )

    print("\n✓  update_map_v2.py complete.")
