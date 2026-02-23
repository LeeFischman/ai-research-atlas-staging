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
# 4. LAYOUT      MDS on group-to-group SPECTER2 distances → 2D group centroids
#                Within each group: scatter papers using SPECTER2 direction vectors
#                + variance-proportional scatter scale.
#                Writes projection_v2_x / projection_v2_y.
# 5. BUILD       embedding-atlas CLI + deploy
#
# Projection columns written (coexist with v1 columns in database.parquet):
#   group_id_v2     — Haiku group assignment (int)
#   projection_v2_x — v2 layout x
#   projection_v2_y — v2 layout y
#
# Run standalone:
#   ANTHROPIC_API_KEY=sk-... python update_map_v2.py
#
# Or via GitHub Actions workflow_dispatch with EMBEDDING_MODEL=v2.
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

# ── Haiku grouping ───────────────────────────────────────────────────────────

# Target number of groups Haiku will produce.
# Haiku receives "between GROUP_COUNT_MIN and GROUP_COUNT_MAX groups".
# Recommended range: 10-20.  Lower = more general labels; higher = more specific.
GROUP_COUNT_MIN = 12
GROUP_COUNT_MAX = 18

# Characters of each abstract sent to Haiku during the grouping call.
# Haiku reads title + first ABSTRACT_GROUPING_CHARS characters of abstract.
# Recommended range: 250-400.  Shorter = cheaper; longer = better grouping.
ABSTRACT_GROUPING_CHARS = 300

# Maximum Haiku retries for the grouping call (covers both 529 overload and
# JSON parse / validation errors). Each retry waits GROUPING_RETRY_BASE_WAIT
# seconds, doubling each time. With base=60 and 5 retries: 60, 120, 240, 480.
# Recommended: 5 retries. Fewer risks giving up during API load spikes.
GROUPING_MAX_RETRIES     = 5
GROUPING_RETRY_BASE_WAIT = 60   # seconds; recommended range: 30-120

# ── Within-group scatter ─────────────────────────────────────────────────────

# Base scatter scale in 2D MDS coordinate space.
# Papers at mean cosine distance D from their group centroid are placed
# SCATTER_SCALE_BASE * (1 + group_variance * VARIANCE_AMPLIFIER) * D from
# the MDS centroid.  Tune SCATTER_SCALE_BASE so clusters don't overlap.
# Recommended range: 1.0-4.0.  Start at 2.0 and adjust.
SCATTER_SCALE_BASE = 2.0

# Amplifies scatter for high-variance (loosely-coupled) groups.
# A group with pairwise cosine-distance std of 0.3 gets scatter scale
# SCATTER_SCALE_BASE * (1 + 0.3 * VARIANCE_AMPLIFIER).
# Recommended range: 1.0-5.0.
VARIANCE_AMPLIFIER = 3.0

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

# System prompt for the grouping call.
# Key design rationale (from context doc):
#   - "between N and M groups" lets Haiku reflect natural structure while
#     keeping the number bounded for reliable JSON parsing.
#   - "same order as input" + index field makes parsing robust even if Haiku
#     reorders entries.
#   - Explicit example structure and "No markdown fences" prevent ```json wrappers.
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
    """Format all papers as the user message for the grouping call."""
    lines = [f"Assign each of the following {len(df)} papers to a group. "
             "Return JSON only.\n"]
    for i, row in df.iterrows():
        title    = str(row["title"]).strip()
        abstract = _strip_urls(str(row.get("abstract", ""))).strip()
        abstract_snippet = abstract[:ABSTRACT_GROUPING_CHARS]
        lines.append(f"[{i}] Title: {title}\nAbstract: {abstract_snippet}")
    return "\n\n".join(lines)


def _parse_grouping_response(
    text: str, n_papers: int
) -> tuple[dict[int, int], dict[int, str]] | None:
    """Parse and validate Haiku's JSON grouping response.

    Returns (index_to_group_id, group_id_to_name) or None if invalid.

    Validates:
      - Valid JSON list
      - Exactly n_papers entries
      - All group_ids are non-negative integers
      - All indices 0..n_papers-1 are present
      - group_name present on at least one entry per group_id

    group_name consistency: if the same group_id appears with different names
    (Haiku occasionally drifts), the most common name wins.
    """
    # Strip markdown fences in case Haiku ignores the instruction
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
    # Collect all names seen per group_id to resolve drift with majority vote
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

    # Check all paper indices are present
    missing = set(range(n_papers)) - set(mapping.keys())
    if missing:
        print(f"    Missing paper indices: {sorted(missing)[:10]}...")
        return None

    n_groups = len(set(mapping.values()))
    if not (GROUP_COUNT_MIN <= n_groups <= GROUP_COUNT_MAX):
        print(f"    Got {n_groups} groups — outside [{GROUP_COUNT_MIN}, {GROUP_COUNT_MAX}].")
        print("    Accepting anyway (Haiku's grouping may be more natural than the target range).")

    # Resolve group names: majority-vote winner per group_id; fallback if absent
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


def haiku_group_papers(
    df: pd.DataFrame,
    client,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Stage 3: Send all papers to Haiku in a single call.

    Returns (df, group_names) where:
      df          — original dataframe with group_id_v2 column added
      group_names — {group_id: label_text} ready for write_labels_parquet

    The DataFrame index must be a clean integer range (0..n-1).
    Call df.reset_index(drop=True) before this function if needed.

    Adds column: group_id_v2 (int)

    Falls back to HDBSCAN clustering from atlas_utils if all Haiku attempts fail,
    with generic "Group N" names.
    """
    df = df.reset_index(drop=True)
    n  = len(df)
    print(f"\n▶  Stage 3 — Haiku grouping + naming ({n} papers)...")

    user_msg = _build_grouping_user_message(df)

    # Estimate token count roughly (4 chars ≈ 1 token)
    approx_tokens = len(user_msg) // 4
    print(f"  Grouping prompt ≈ {approx_tokens:,} tokens "
          f"(abstracts truncated to {ABSTRACT_GROUPING_CHARS} chars each).")

    result = None
    for attempt in range(1, GROUPING_MAX_RETRIES + 1):
        print(f"  Haiku grouping call — attempt {attempt}/{GROUPING_MAX_RETRIES}...")
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=8192,  # 250 entries × ~50 chars ≈ 3K tokens; use Haiku max for headroom
                system=_GROUPING_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            print(f"  Response length: {len(raw)} chars.")
            result = _parse_grouping_response(raw, n)
            if result is not None:
                mapping, group_names = result
                print(f"  ✓ Grouping parsed successfully "
                      f"({len(set(mapping.values()))} groups).")
                break
        except Exception as e:
            err_str = str(e)
            is_529  = "529" in err_str or "overloaded" in err_str.lower()
            label   = "API overloaded (529)" if is_529 else f"API error"
            print(f"  {label} on attempt {attempt}: {e}")

        if attempt < GROUPING_MAX_RETRIES:
            # Use a long base wait so 529 overload errors have time to clear.
            # 2 ** (attempt-1) doubles the wait each retry: 60, 120, 240, 480s.
            wait = GROUPING_RETRY_BASE_WAIT * (2 ** (attempt - 1))
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    if result is None:
        print("  ✗ All Haiku grouping attempts failed. "
              "Falling back to HDBSCAN clustering.")
        mapping     = _hdbscan_fallback_grouping(df)
        group_names = {gid: f"Group {gid}" for gid in set(mapping.values())}

    df["group_id_v2"] = [mapping[i] for i in range(n)]
    group_counts = df["group_id_v2"].value_counts().sort_index()
    for gid, count in group_counts.items():
        print(f"    Group {gid:>2} ({count:>3} papers): '{group_names.get(gid, '?')}'")
    print(f"  Total: {len(group_counts)} groups, {n} papers assigned.")
    return df, group_names


def _hdbscan_fallback_grouping(df: pd.DataFrame) -> dict[int, int]:
    """Fallback: HDBSCAN on 50D SPECTER2 embeddings if Haiku grouping fails.

    Produces 12-18 groups approximately by tuning min_cluster_size.
    Papers in noise cluster (-1) are assigned to the nearest cluster centroid.
    """
    from sklearn.cluster import HDBSCAN
    import numpy as np
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("  Fallback HDBSCAN grouping...")

    if "embedding_50d" in df.columns and df["embedding_50d"].notna().all():
        X = np.array(df["embedding_50d"].tolist(), dtype=np.float32)
        metric = "cosine"
        print("  Clustering on 50D SPECTER2 (cosine).")
    elif "embedding" in df.columns and df["embedding"].notna().all():
        X = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
        metric = "cosine"
        print("  Clustering on 768D SPECTER2 (cosine).")
    else:
        # Worst-case: random assignment into GROUP_COUNT_MIN groups
        print("  No embedding available — random group assignment.")
        n_groups = GROUP_COUNT_MIN
        return {i: i % n_groups for i in range(len(df))}

    # Try progressively smaller min_cluster_size to hit the target group count
    for mcs in [max(3, len(df) // 30), max(3, len(df) // 40), 3]:
        clusterer = HDBSCAN(
            min_cluster_size=mcs, min_samples=3,
            metric=metric, cluster_selection_method="leaf",
        )
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  HDBSCAN min_cluster_size={mcs} → {n_clusters} clusters.")
        if GROUP_COUNT_MIN <= n_clusters <= GROUP_COUNT_MAX + 5:
            break

    # Assign noise points (-1) to nearest cluster centroid
    unique_clusters = [c for c in sorted(set(labels)) if c != -1]
    if not unique_clusters:
        print("  HDBSCAN produced no clusters — assigning all to group 0.")
        return {i: 0 for i in range(len(df))}

    centroids = np.array([
        X[labels == c].mean(axis=0) for c in unique_clusters
    ])
    noise_mask = labels == -1
    if noise_mask.any():
        noise_vecs = X[noise_mask]
        dists_to_centroids = cosine_distances(noise_vecs, centroids)
        nearest = dists_to_centroids.argmin(axis=1)
        labels = labels.copy()
        labels[noise_mask] = [unique_clusters[j] for j in nearest]
        print(f"  Reassigned {noise_mask.sum()} noise points to nearest cluster.")

    # Remap cluster IDs to 0..n_clusters-1
    id_map = {old: new for new, old in enumerate(unique_clusters)}
    return {i: id_map.get(int(labels[i]), 0) for i in range(len(df))}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MDS LAYOUT + WITHIN-GROUP SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def compute_mds_centroids(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Stage 4a: MDS on group-to-group SPECTER2 distances → 2D centroids.

    Uses classical MDS (metric=True) which preserves global distance ratios.
    Appropriate here because:
      - We have few points (~15 groups) so MDS is fast and stable.
      - We want the between-group distances to be faithfully represented in 2D.

    Returns: {group_id: (centroid_x, centroid_y)}
    """
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("\n▶  Stage 4a — MDS between-group layout...")

    # Use raw 768D SPECTER2 vectors for distance computation (most precise)
    if "embedding" not in df.columns or df["embedding"].isna().any():
        raise RuntimeError(
            "SPECTER2 'embedding' column missing or has NaNs. "
            "Run embed_and_project() before compute_mds_centroids()."
        )

    all_vecs = sk_normalize(
        np.array(df["embedding"].tolist(), dtype=np.float32)
    )  # shape (n, 768)

    group_ids = sorted(df["group_id_v2"].unique())
    n_groups  = len(group_ids)
    print(f"  Computing {n_groups}×{n_groups} group distance matrix...")

    # Build positional index lookup: group_id → list of integer row positions
    group_positions: dict[int, list[int]] = {gid: [] for gid in group_ids}
    for pos, gid in enumerate(df["group_id_v2"].tolist()):
        group_positions[gid].append(pos)

    # Full pairwise cosine distance matrix (reused for per-pair means)
    all_pairwise = cosine_distances(all_vecs)  # shape (n, n)

    # Group-to-group distance matrix: mean inter-group pairwise cosine distance
    group_dist = np.zeros((n_groups, n_groups), dtype=np.float32)
    for i, gid_i in enumerate(group_ids):
        for j, gid_j in enumerate(group_ids):
            if i == j:
                continue
            pos_i = group_positions[gid_i]
            pos_j = group_positions[gid_j]
            sub   = all_pairwise[np.ix_(pos_i, pos_j)]
            group_dist[i, j] = float(sub.mean())

    # Ensure exact symmetry (floating-point drift)
    group_dist = (group_dist + group_dist.T) / 2
    np.fill_diagonal(group_dist, 0.0)

    print(f"  Group dist matrix: min={group_dist.min():.4f}, "
          f"max={group_dist.max():.4f}, mean={group_dist.mean():.4f}")

    # Classical MDS — fit group centroids in 2D
    mds = MDS(
        n_components=2,
        metric=True,           # classical / metric MDS
        dissimilarity="precomputed",
        random_state=42,
        n_init=4,              # run 4 initialisations, keep best
        max_iter=500,
        normalized_stress="auto",
    )
    group_coords = mds.fit_transform(group_dist)   # shape (n_groups, 2)
    stress = mds.stress_
    print(f"  MDS stress: {stress:.6f}  "
          f"(lower is better; < 0.05 is excellent for ~15 points)")

    centroids: dict[int, tuple[float, float]] = {}
    for i, gid in enumerate(group_ids):
        cx, cy = float(group_coords[i, 0]), float(group_coords[i, 1])
        n_papers = len(group_positions[gid])
        centroids[gid] = (cx, cy)
        print(f"  Group {gid:>2}: centroid=({cx:+.3f}, {cy:+.3f}), "
              f"n={n_papers}")

    return centroids


def scatter_within_groups(
    df: pd.DataFrame,
    centroids: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """Stage 4b: Place each paper around its MDS centroid.

    Position formula
    ────────────────
    For each paper p in group g:
      1. Direction: unit vector from the SPECTER2 group centroid toward
         paper p's existing SPECTER2 UMAP position (projection_x/y).
         Papers at group edges in SPECTER2 space drift further from the MDS
         centroid — naturally placing straddlers between groups.
      2. Distance: paper's mean cosine distance to other group members,
         scaled by a group-specific scatter factor:
           scatter_scale = SCATTER_SCALE_BASE
                         * (1 + group_variance * VARIANCE_AMPLIFIER)
         Tight groups (low variance) → tight scatter.
         Loose groups (high variance) → more spread.
      3. final_pos = mds_centroid + direction * distance * scatter_scale

    Requires columns: embedding (768D), projection_x, projection_y, group_id_v2
    Writes columns:   projection_v2_x, projection_v2_y
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("\n▶  Stage 4b — Within-group scatter (variance-proportional scale)...")

    df = df.copy()
    df["projection_v2_x"] = np.nan
    df["projection_v2_y"] = np.nan

    # Normalised 768D SPECTER2 vectors (whole corpus, for cosine distance)
    all_vecs = sk_normalize(
        np.array(df["embedding"].tolist(), dtype=np.float32)
    )

    # SPECTER2 UMAP positions — used as scatter direction hints
    specter2_x = df["projection_x"].values
    specter2_y = df["projection_y"].values

    group_ids = sorted(df["group_id_v2"].unique())

    for gid in group_ids:
        mask      = df["group_id_v2"] == gid
        positions = df.index[mask].tolist()   # DataFrame index values
        pos_array = [df.index.get_loc(p) for p in positions]  # integer positions

        g_vecs = all_vecs[pos_array]   # shape (n_group, 768)
        n_g    = len(positions)

        # ── Per-group cosine distance matrix ────────────────────────────────
        if n_g == 1:
            # Singleton group — place at centroid
            cx, cy = centroids[gid]
            df.at[positions[0], "projection_v2_x"] = cx
            df.at[positions[0], "projection_v2_y"] = cy
            print(f"  Group {gid:>2}: singleton paper placed at centroid.")
            continue

        pairwise = cosine_distances(g_vecs)  # shape (n_g, n_g)

        # Mean distance of each paper to its group peers
        mean_dists = pairwise.mean(axis=1)   # shape (n_g,)

        # Group variance = std of upper-triangle pairwise distances
        upper = pairwise[np.triu_indices(n_g, k=1)]
        group_variance = float(upper.std()) if len(upper) > 0 else 0.0

        scatter_scale = SCATTER_SCALE_BASE * (1.0 + group_variance * VARIANCE_AMPLIFIER)

        print(f"  Group {gid:>2}: n={n_g:>3}, "
              f"variance={group_variance:.4f}, "
              f"scatter_scale={scatter_scale:.3f}")

        # ── SPECTER2 group centroid in 2D UMAP space ─────────────────────
        sp_cx = specter2_x[pos_array].mean()
        sp_cy = specter2_y[pos_array].mean()

        # ── MDS centroid for this group ──────────────────────────────────
        mds_cx, mds_cy = centroids[gid]

        # ── Place each paper ─────────────────────────────────────────────
        for local_i, (df_idx, pos_i) in enumerate(zip(positions, pos_array)):
            # Direction: from SPECTER2 group centroid toward this paper's UMAP pos
            dx = specter2_x[pos_i] - sp_cx
            dy = specter2_y[pos_i] - sp_cy
            length = sqrt(dx * dx + dy * dy)

            if length < 1e-8:
                # Paper sits at exact SPECTER2 centroid → random direction
                angle  = random.uniform(0, 2 * 3.14159265)
                dx, dy = np.cos(angle), np.sin(angle)
                length = 1.0

            scatter_dist = float(mean_dists[local_i]) * scatter_scale

            df.at[df_idx, "projection_v2_x"] = mds_cx + (dx / length) * scatter_dist
            df.at[df_idx, "projection_v2_y"] = mds_cy + (dy / length) * scatter_dist

    n_placed = df["projection_v2_x"].notna().sum()
    print(f"  Placed {n_placed}/{len(df)} papers in v2 layout.")

    x_range = (df["projection_v2_x"].min(), df["projection_v2_x"].max())
    y_range = (df["projection_v2_y"].min(), df["projection_v2_y"].max())
    print(f"  Layout bounds: x=[{x_range[0]:.3f}, {x_range[1]:.3f}], "
          f"y=[{y_range[0]:.3f}, {y_range[1]:.3f}]")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4b/5 — WRITE LABELS PARQUET
# ══════════════════════════════════════════════════════════════════════════════

def write_labels_parquet(
    df: pd.DataFrame,
    labels: dict[int, str],
    centroids: dict[int, tuple[float, float]],
    labels_path: str = "labels_v2.parquet",
) -> str:
    """Write labels.parquet for the Atlas CLI --labels flag.

    Label positions are the MDS centroids (in v2 coordinate space),
    which is where the cluster visually lives after scatter.

    Returns the path to the written file.
    """
    rows = []
    for gid, (cx, cy) in centroids.items():
        label_text = labels.get(gid, f"Group {gid}")
        n_papers   = int((df["group_id_v2"] == gid).sum())
        rows.append({
            "x":        cx,
            "y":        cy,
            "text":     label_text,
            "level":    0,
            "priority": 10,
        })
        print(f"  Label '{label_text}' → ({cx:.3f}, {cy:.3f}), {n_papers} papers")

    labels_df = pd.DataFrame(rows)
    labels_df.to_parquet(labels_path, index=False)
    print(f"  Wrote {len(labels_df)} labels to {labels_path}.")
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
    print(f"  GROUP_COUNT_MIN={GROUP_COUNT_MIN}, GROUP_COUNT_MAX={GROUP_COUNT_MAX}")
    print(f"  SCATTER_SCALE_BASE={SCATTER_SCALE_BASE}, VARIANCE_AMPLIFIER={VARIANCE_AMPLIFIER}")
    print("=" * 60)

    # ── Anthropic client ────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to GitHub repo secrets "
            "and expose it in the workflow YAML under env:."
        )
    haiku_client = anthropic.Anthropic(api_key=api_key)

    # ── Stage 1: Load & prune existing DB ───────────────────────────────────
    print("\n▶  Stage 1 — Loading rolling database...")
    existing_df  = load_existing_db()
    is_first_run = existing_df.empty and not os.path.exists(DB_PATH)
    days_back    = 5 if is_first_run else 2

    if is_first_run:
        print("  First run — pre-filling with last 5 days of arXiv papers.")
    else:
        print(f"  Loaded {len(existing_df)} existing papers from rolling DB.")

    # ── Stage 1b: arXiv fetch ───────────────────────────────────────────────
    print("\n▶  Stage 1b — Fetching from arXiv...")

    # Set descriptive User-Agent as required by arXiv API policy
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
            print("  No results from arXiv and no existing DB. Cannot build. Exiting.")
            exit(0)
        print(f"  No new papers from arXiv (weekend or dry spell). "
              f"Rebuilding atlas from {len(existing_df)} existing papers.")

    if results:
        print(f"  Fetched {len(results)} papers from arXiv.")
        today_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for r in results:
            title    = r.title
            abstract = r.summary
            scrubbed = scrub_model_words(f"{title}. {title}. {abstract}")
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
        # Remove legacy grouping columns from v1 (group, group_id_v2 if stale)
        df = df.drop(columns=["group", "group_id_v2"], errors="ignore")
        print(f"  Rolling DB: {len(df)} papers after merge.")
    else:
        df = existing_df.drop(columns=["group", "group_id_v2"], errors="ignore")

    # Backfill reputation for older rows
    if "Reputation" not in df.columns or df["Reputation"].isna().any():
        missing = (df["Reputation"].isna()
                   if "Reputation" in df.columns
                   else pd.Series([True] * len(df)))
        if missing.any():
            print(f"  Backfilling Reputation for {missing.sum()} rows...")
            df.loc[missing, "Reputation"] = df.loc[missing].apply(
                calculate_reputation, axis=1
            )

    # ── Stage 2: SPECTER2 embed + UMAP ──────────────────────────────────────
    print("\n▶  Stage 2 — SPECTER2 embedding + UMAP...")
    df = embed_and_project(df, model_name="specter2")
    # After this:
    #   df["embedding"]     — 768D SPECTER2 vectors (for cosine distances)
    #   df["embedding_50d"] — 50D UMAP (for potential future use)
    #   df["projection_x"]  — 2D UMAP x (for scatter direction vectors)
    #   df["projection_y"]  — 2D UMAP y (for scatter direction vectors)

    # ── Stage 3: Haiku grouping + naming ────────────────────────────────────
    df, group_names = haiku_group_papers(df, haiku_client)

    # ── Stage 4a: MDS between-group layout ──────────────────────────────────
    centroids = compute_mds_centroids(df)

    # ── Stage 4b: Within-group scatter ──────────────────────────────────────
    df = scatter_within_groups(df, centroids)

    # ── Write labels parquet ─────────────────────────────────────────────────
    # group_names came from Stage 3 — no second API call needed.
    print("\n▶  Writing labels parquet...")
    labels_path = write_labels_parquet(df, group_names, centroids)

    # ── Save rolling DB ──────────────────────────────────────────────────────
    print("\n▶  Saving rolling database...")
    save_df = df.copy()
    save_df["date_added"] = pd.to_datetime(
        save_df["date_added"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Sanity check: ensure v2 projection columns are present
    if "projection_v2_x" not in save_df.columns:
        raise RuntimeError("projection_v2_x missing — layout step failed.")

    save_df.to_parquet(DB_PATH, index=False)
    print(f"  Saved {len(save_df)} papers to {DB_PATH}.")

    cols = [c for c in save_df.columns if "projection" in c or "embedding" in c]
    print(f"  Projection / embedding columns in parquet: {cols}")

    # ── Stage 6: Build + deploy ──────────────────────────────────────────────
    print("\n▶  Stage 6 — Building atlas...")
    build_and_deploy_atlas(
        db_path    = DB_PATH,
        proj_x_col = PROJ_X_COL,
        proj_y_col = PROJ_Y_COL,
        labels_path = labels_path,
        run_date   = run_date,
    )

    print("\n✓  update_map_v2.py complete.")
