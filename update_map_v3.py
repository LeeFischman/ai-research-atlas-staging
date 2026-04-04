#!/usr/bin/env python3
# update_map_v3.py
# ──────────────────────────────────────────────────────────────────────────────
# AI Research Atlas — v3 pipeline.
#
# Key difference from update_map.py (v1):
#   v1: HDBSCAN groups papers by SPECTER2 geometry, Haiku labels each cluster.
#   v2: Haiku groups papers by research meaning, geometry follows from those groups.
#   v3: Adds secondary_tags (multi-label) and end-of-run dynamic-group reconciliation.
#
# Pipeline (4 stages + build)
# ──────────────────────────────────────────────
# 1. FETCH       arXiv → new papers
# 2. EMBED       SPECTER2 incremental embed + UMAP → projection_x/y for direction vectors
# 3. GROUP       Single Haiku call → group_id_v3 + group_name per paper (12-18 groups)
#                Haiku names the groups as it forms them; no second labeling call needed.
#                If Haiku returns > GROUP_COUNT_MAX groups, excess groups are merged
#                by repeatedly absorbing the closest pair (SPECTER2 distance).
#                Returns secondary_tags: 0-2 additional groups per paper (for Grid View).
#                End-of-run reconciliation merges near-duplicate dynamic groups.
# 4. LAYOUT      MDS on group-to-group SPECTER2 distances → 2D group centroids,
#                multiplied by LAYOUT_SCALE for human-readable coordinate range.
#                Within each group: scatter papers using SPECTER2 direction vectors
#                + variance-proportional scatter scale, also × LAYOUT_SCALE.
#                Writes projection_v3_x / projection_v3_y.
# 5. BUILD       embedding-atlas CLI + deploy
#
# Projection columns written (coexist with v1 columns in database.parquet):
#   group_id_v3     — Haiku group assignment (int, post-merge)
#   projection_v3_x — v3 layout x
#   projection_v3_y — v3 layout y
#   secondary_tags  — list of 0-2 additional group names per paper (Grid View only)
#
# Offline mode (re-runs layout + build only, skipping all API/embed work):
#   OFFLINE_MODE=true python update_map_v2.py
#   Requires: database.parquet with group_id_v3 + embedding columns, and
#             group_names_v3.json saved from a previous normal run.
#
# Normal run:
#   ANTHROPIC_API_KEY=sk-... python update_map_v3.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from math import sqrt

import numpy as np
import pandas as pd

# Shared utilities — does NOT modify update_map.py
from atlas_utils import (
    DB_PATH,
    ARXIV_MAX,
    RETENTION_DAYS,
    _strip_urls,
    scrub_model_words,
    calculate_prominence,
    calculate_citation_tier,
    categorize_authors,
    load_author_cache,
    save_author_cache,
    fetch_author_hindices,
    load_existing_db,
    merge_papers,
    fetch_arxiv_oai,
    embed_and_project,
    build_and_deploy_atlas,
    # Semantic Scholar
    SS_CACHE_TTL_DAYS,
    load_ss_cache,
    save_ss_cache,
    fetch_semantic_scholar_data,
    _arxiv_id_base,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — all tuning knobs in one place
# ══════════════════════════════════════════════════════════════════════════════

# ── Offline mode ─────────────────────────────────────────────────────────────
# When True: skip arXiv fetch, SPECTER2 embedding, and Haiku API calls entirely.
# Re-runs MDS layout, scatter, and Atlas build using data already in the parquet.
# Set via env var:  OFFLINE_MODE=true python update_map_v2.py
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "false").strip().lower() == "true"

# When True: re-fetch OpenAlex h-indices for ALL papers, including those that
# already have an author_hindices list (e.g. empty lists from a first run).
# Set via env var:  BACKFILL_HINDICES=true
# Remove after one successful backfill run.
BACKFILL_HINDICES = os.environ.get("BACKFILL_HINDICES", "false").strip().lower() == "true"

# When True: skip the OpenAlex API fetch entirely (Stage 1c).
# Existing author_hindices values carry forward from the loaded parquet.
# New papers get author_hindices=[] and Prominence="Unverified" — they will
# be enriched on the next normal run.
# Prominence is still recomputed from whatever author_hindices is in the data.
# Use when: OpenAlex daily quota is exhausted but the rest of the pipeline
# must still run.
# Set via env var:  OPENALEX_OFFLINE_MODE=true python update_map_v2.py
OPENALEX_OFFLINE_MODE = os.environ.get("OPENALEX_OFFLINE_MODE", "false").strip().lower() == "true"

# Cache file written after every successful Haiku grouping call.
# Loaded automatically in offline mode.
GROUP_NAMES_CACHE = "group_names_v3.json"

# ── Haiku grouping — two-pass + persistent taxonomy ─────────────────────────
#
# Pass 1: All papers assigned to stable categories (IDs 0-13) using titles.
#         Papers that do not clearly fit a stable category are flagged uncertain.
# Pass 2: Uncertain papers assigned to existing or new dynamic categories with
#         per-paper confidence ratings. Informed by persistent dynamic taxonomy.
# Review: Optional targeted call for papers in groups with declining confidence.
#
# Token budget at 2000 papers:
#   Pass 1 input:  ~2000 × 80 chars ÷ 4 ≈ 40K tokens (one call, under TPM limit)
#   Pass 1 output: ~2000 × 8 chars  ÷ 4 ≈ 4K tokens  (under 8192 ceiling)
#   Pass 2 input:  ≤400 papers × ~400 chars ÷ 4 ≈ 40K tokens
#   Pass 2 output: ≤400 × ~30 chars ÷ 4 ≈ 3K tokens

GROUP_STABLE_COUNT = 14    # fixed; IDs 0-13

# Hard cap on total groups (stable + dynamic).
GROUP_COUNT_MAX = 60

# Characters of each abstract sent to Haiku in pass 2.
ABSTRACT_GROUPING_CHARS = 300

# Maximum papers Haiku may flag as uncertain in pass 1.
# Papers above this cap are force-assigned to their pass 1 stable best-guess.
PASS1_UNCERTAIN_CAP = 400

# Pass 1 is batched to keep output arrays short and reliable.
# At 100 papers: ~100 integers output ≈ 300 tokens — trivially reliable.
# Input: ~100 titles × ~90 chars ÷ 4 ≈ 2K tokens. ~21 batches at 2000 papers.
PASS1_BATCH_SIZE = 100

# Retry policy — shared across all Haiku calls in Stage 3.
# Wait schedule: GROUPING_RETRY_BASE_WAIT × 2^(attempt-1) seconds.
GROUPING_MAX_RETRIES     = 5
GROUPING_RETRY_BASE_WAIT = 60   # seconds

# ── Dynamic taxonomy persistence ──────────────────────────────────────────────
# Committed to repo alongside group_names_v3.json.
TAXONOMY_PATH              = "dynamic_taxonomy.json"
TAXONOMY_CONFIDENCE_WINDOW = 5    # rolling window length (runs)
TAXONOMY_RETIRE_DAYS       = 14   # absent this many days → retired from taxonomy
# Review threshold modulation — groups with fewer runs are exempt.
# Tightens as the group matures toward a full confidence window.
#   1-2 runs : exempt (too early to judge)
#   3 runs   : 0.35
#   4 runs   : 0.45
#   5+ runs  : 0.60
TAXONOMY_REVIEW_THRESHOLDS = {1: None, 2: None, 3: 0.35, 4: 0.45}
TAXONOMY_REVIEW_THRESHOLD_MATURE = 0.60

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
LAYOUT_X_SCALE     = 1.6   # stretch x for a wider canvas (>1.0 = wider)
LAYOUT_Y_SCALE     = 0.7   # compress y to fit landscape view (<1.0 = shorter)

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
# Too many dynamic groups (noise-like topics appearing):
#   → Lower GROUP_DYNAMIC_MAX (6 → 4)
#
# Dynamic groups too coarse (interesting edges being merged into stable):
#   → Raise GROUP_DYNAMIC_MAX (6 → 8)
#
# Merge step absorbing groups that feel semantically distinct:
#   → Raise GROUP_COUNT_MAX (20 → 24)
#
# Ready for production (fresh papers + re-grouping):
#   → Remove OFFLINE_MODE from the workflow YAML.
#     First normal run re-embeds, calls Haiku, and commits updated
#     group_names_v3.json to the repo for future offline runs.

# ── Atlas CLI ────────────────────────────────────────────────────────────────
# Projection column names written to database.parquet.
# Must match --x / --y args passed to the Atlas CLI.
PROJ_X_COL = "projection_v3_x"
PROJ_Y_COL = "projection_v3_y"

# ── Significant papers ───────────────────────────────────────────────────────
# Written by the weekly update_significant.py script.
# Loaded at Stage 1 and merged with the recent-window papers.
SIGNIFICANT_PATH = "significant.parquet"

# ── Haiku model ──────────────────────────────────────────────────────────────
HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — HAIKU GROUPING  (two-pass + persistent dynamic taxonomy)
# ══════════════════════════════════════════════════════════════════════════════

_STABLE_BUCKETS = """STABLE CATEGORIES (use only those that have papers today — skip empty ones):

ID 1 — Computer Vision
  Image and video understanding, generation, and editing. Object detection,
  segmentation, recognition, diffusion models for images, 3D reconstruction,
  video generation and understanding. Primary contribution is visual.

ID 2 — Multimodal Learning
  Systems jointly processing two or more modalities (vision+language,
  audio+language, etc.). VLMs, vision-language alignment, captioning,
  visual QA, audio-language models. Primary contribution is cross-modal fusion.

ID 3 — Reinforcement Learning
  RL algorithms, policy optimization, reward modeling, multi-agent RL,
  offline RL, model-based RL, exploration strategies, game-playing agents.

ID 4 — Robotics & Embodied AI
  Physical robots, manipulation, locomotion, sim-to-real transfer,
  embodied agents in 3D environments, navigation, dexterous control.

ID 5 — Safety, Alignment & Ethics
  AI safety, alignment research, red-teaming, jailbreaks, fairness, bias,
  interpretability for safety, societal impact, governance, privacy.

ID 6 — Theory, Optimization & Efficient ML
  Mathematical foundations: convergence proofs, generalization bounds,
  learning theory. Model compression, quantization, pruning, efficient
  architectures, hardware-aware ML, federated learning.

ID 7 — Domain Applications
  AI applied to a specific external field as the primary contribution:
  medicine, biology, climate, law, finance, education, scientific discovery.
  The application domain — not the AI method — is the main result.

ID 8 — Generative Models & Synthesis
  Generative modelling as the primary contribution: diffusion models, GANs,
  VAEs, flow-based models, autoregressive generation of images, video, 3D,
  molecules, or other structured data. Distinct from Computer Vision when the
  core contribution is the generative method rather than visual understanding.

ID 9 — Speech & Audio Processing
  Spoken language, audio signals, and music. Speech recognition, synthesis,
  voice conversion, speaker identification, audio generation, sound event
  detection, music information retrieval. Primary modality is audio.

ID 10 — Graph Learning & Networks
  Graph neural networks, knowledge graphs, network analysis, link prediction,
  node classification, graph generation, and geometric deep learning on
  non-Euclidean data structures.

ID 11 — Data, Benchmarks & Evaluation
  Datasets, evaluation frameworks, and measurement methodology as the primary
  contribution: new benchmarks, annotation pipelines, evaluation metrics,
  leaderboard analysis, reproducibility studies, dataset audits.

ID 12 — Human-AI Interaction
  Interfaces, user experience, and the human side of AI systems: conversational
  agents, explainability for end-users, HCI studies, accessibility, human
  factors, collaborative AI tools, trust and transparency research.

ID 13 — Planning & Search
  Planning algorithms, search-based reasoning, symbolic AI, combinatorial
  optimisation, constraint satisfaction, automated reasoning, and hybrid
  neuro-symbolic approaches where planning or search is the core contribution.

ID 0 — Language Models & Pretraining
  ONLY use for papers whose PRIMARY contribution is the language model itself:
  pretraining objectives, model architecture, fine-tuning methods, RLHF,
  inference-time scaling, context length, or LLM-specific efficiency (e.g.
  quantization, distillation of LLMs).
  NOT this category: agents, RAG, prompt engineering, reasoning chains,
  instruction following, tool use, LLM applications, or any paper where an
  LLM is used as a component rather than studied as the primary subject.
  If unsure whether the LLM or its application is the main contribution,
  mark the paper uncertain."""


# Canonical names for stable buckets — used to normalise Haiku's output.
_STABLE_BUCKET_NAMES: dict[int, str] = {
    0:  "Language Models & Pretraining",
    1:  "Computer Vision",
    2:  "Multimodal Learning",
    3:  "Reinforcement Learning",
    4:  "Robotics & Embodied AI",
    5:  "Safety, Alignment & Ethics",
    6:  "Theory, Optimization & Efficient ML",
    7:  "Domain Applications",
    8:  "Generative Models & Synthesis",
    9:  "Speech & Audio Processing",
    10: "Graph Learning & Networks",
    11: "Data, Benchmarks & Evaluation",
    12: "Human-AI Interaction",
    13: "Planning & Search",
}


# ── System prompts ────────────────────────────────────────────────────────────

_PASS1_SYSTEM = (
    "You are a research taxonomy expert. Assign each AI research paper "
    "(provided by title only) to exactly one stable category (IDs 0-13). "
    "For papers that do not clearly fit any stable category, still provide "
    "your best stable assignment AND add the index to 'uncertain'.\n\n"
    + _STABLE_BUCKETS + "\n\n"
    "ASSIGNMENT RULES:\n"
    "- When a paper could fit multiple categories, assign it to the one where "
    "its PRIMARY CONTRIBUTION lies — the thing the authors would consider their "
    "main result.\n"
    "- Consider ALL categories before settling on one. Categories are listed "
    "with ID 0 last deliberately — exhaust all other options before using it.\n"
    "- ID 0 (Language Models & Pretraining) is ONLY for papers studying the "
    "model itself. Papers using LLMs as a tool belong elsewhere or in uncertain.\n"
    "- Mark a paper 'uncertain' if it uses LLMs as a component but its main "
    "contribution is agentic, applied, or not clearly core model research. "
    "Uncertain papers receive specialized dynamic-group review.\n"
    "- Prefer stable categories: only mark uncertain if genuinely ambiguous.\n\n"
    "OUTPUT FORMAT — respond ONLY with JSON, no preamble, no markdown fences:\n"
    '{"assignments": [3, 0, 7, 1, ...], "uncertain": [4, 17, ...], '
    '"secondary": {"2": [6], "5": [3, 8]}}\n'
    "- 'assignments': a JSON array of length N where assignments[i] is the\n"
    "  stable group_id (0-13) for paper i. Index position = paper index.\n"
    "- 'uncertain': list of paper indices where no stable category fits well.\n"
    "- 'secondary': SPARSE dict — only include papers that clearly straddle a\n"
    "  second stable category. Key = string index, value = list of 1-2 stable\n"
    "  group_ids (0-13) that are also highly relevant. Omit papers with no\n"
    "  strong secondary fit. Most papers should NOT appear here.\n"
    "- Every paper index 0 to N-1 must be represented in 'assignments'."
)

_PASS2_SYSTEM = (
    "You are a research taxonomy expert specialising in emerging and cross-cutting "
    "AI research topics. You will be given papers that do not clearly fit standard "
    "AI research categories. Assign each paper to the best available group and "
    "rate your confidence.\n\n"
    + _STABLE_BUCKETS + "\n\n"
    "CONFIDENCE RATING:\n"
    "- 'high': the paper clearly belongs to this group; you are confident.\n"
    "- 'low': the paper fits imperfectly, or you are genuinely uncertain.\n\n"
    "CREATING NEW DYNAMIC CATEGORIES:\n"
    "- Only create a new group when papers genuinely share an intellectual thread "
    "not covered by any existing group (stable or dynamic).\n"
    "- Name each with a concise 3-6 word noun phrase capturing the shared theme.\n"
    "- Prefer assigning to existing dynamic groups over creating new ones.\n\n"
    "OUTPUT FORMAT — respond ONLY with JSON, no preamble, no markdown fences:\n"
    '{"groups": {"14": "Continual Learning & Adaptation", "22": "New Topic Name"},\n'
    ' "assignments": {"0": {"group": 14, "confidence": "high", "secondary": [3]}, '
    '"1": {"group": 22, "confidence": "low", "secondary": []}, ...}}\n'
    "- 'groups': ALL group IDs used (existing and new) mapped to their names.\n"
    "- 'assignments': every paper index mapped to {group, confidence, secondary}.\n"
    "- 'secondary': list of 0-2 additional group IDs (stable or dynamic) that the\n"
    "  paper also clearly belongs to. Must differ from the primary 'group'. Only\n"
    "  include when confidence is genuinely high for that secondary fit. Use [].\n"
    "  for most papers — secondary tags should be rare, not the default.\n"
    "- For stable groups (0-13), use canonical names exactly as listed above.\n"
    "- For new dynamic groups, use a concise 3-6 word noun phrase."
)


# ── Low-level API helper ──────────────────────────────────────────────────────

def _haiku_call(client, system: str, user: str, max_tokens: int) -> str | None:
    """Single Haiku API call. Returns raw response text or None on exception."""
    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        err_str = str(e)
        is_529  = "529" in err_str or "overloaded" in err_str.lower()
        label   = "API overloaded (529)" if is_529 else "API error"
        print(f"    {label}: {e}")
        return None


# ── Dynamic taxonomy I/O ──────────────────────────────────────────────────────

def load_dynamic_taxonomy() -> dict:
    """Load dynamic_taxonomy.json. Returns empty taxonomy if file is missing."""
    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH) as f:
            t = json.load(f)
        n = len(t.get("groups", {}))
        print(f"  Loaded dynamic taxonomy: {n} groups "
              f"(next_id={t.get('next_id', 14)}).")
        return t
    return {"next_id": 14, "groups": {}}


def save_dynamic_taxonomy(taxonomy: dict) -> None:
    with open(TAXONOMY_PATH, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"  Dynamic taxonomy saved: {len(taxonomy.get('groups', {}))} groups.")


def seed_dynamic_taxonomy_from_cache(today: str) -> dict:
    """Seed dynamic_taxonomy.json from group_names_v3.json on first run.

    Imports all IDs >= 14 with empty confidence_history so they start as
    1-run-old groups and receive appropriate breathing room.
    """
    if not os.path.exists(GROUP_NAMES_CACHE):
        print(f"  No {GROUP_NAMES_CACHE} found — starting with empty taxonomy.")
        return {"next_id": 14, "groups": {}}

    with open(GROUP_NAMES_CACHE) as f:
        raw = json.load(f)

    groups: dict[str, dict] = {}
    max_id = 13
    for k, name in raw.items():
        gid = int(k)
        if gid >= 14:
            groups[str(gid)] = {
                "name":               name,
                "created":            today,
                "last_seen":          today,
                "confidence_history": [],
                "paper_count_history": [],
            }
            max_id = max(max_id, gid)

    next_id  = max_id + 1
    taxonomy = {"next_id": next_id, "groups": groups}
    print(f"  Seeded dynamic taxonomy from {GROUP_NAMES_CACHE}: "
          f"{len(groups)} groups, next_id={next_id}.")
    return taxonomy


def _review_threshold(n_runs: int) -> float | None:
    """Maturity-modulated review threshold.

    Returns None if the group has too little history to judge.
    Tightens progressively as the group matures toward TAXONOMY_CONFIDENCE_WINDOW.
    """
    return TAXONOMY_REVIEW_THRESHOLDS.get(
        n_runs,
        TAXONOMY_REVIEW_THRESHOLD_MATURE if n_runs >= 5 else None,
    )


# ── Pass 1: stable assignment ─────────────────────────────────────────────────

def _build_pass1_message(df: pd.DataFrame) -> str:
    n = len(df)
    lines = [
        f"Assign each of the following {n} AI research papers to a stable "
        "category using the title alone. The ONLY valid group IDs are 0 through 13 — "
        "do NOT use any other numbers as group IDs.\n"
        "For papers that do not clearly fit any stable category, still provide your "
        "best stable assignment AND include its position (0-based) in 'uncertain'.\n"
        "Return JSON only — no preamble, no markdown:\n"
        '{"assignments": [3, 0, 7, 1, ...], "uncertain": [4, 17, ...]}\n'
        f"- 'assignments': array of exactly {n} integers, each between 0 and 13 inclusive.\n"
        "- 'uncertain': list of 0-based positions where no stable category fits well.\n"
        "- Every value in 'assignments' MUST be 0-13. Never use the paper's position as a group ID.\n"
    ]
    # No numeric prefixes — they confuse position with group ID
    for _, row in df.iterrows():
        lines.append(str(row['title']).strip())
    return "\n".join(lines)


def _parse_pass1_response(
    text: str, n_papers: int
) -> tuple[dict[int, int], list[int]] | None:
    """Parse pass 1 response.

    Returns (assignments, uncertain_indices, secondary) or None on hard failure.
      assignments:       {paper_idx: stable_group_id}  — all N papers
      uncertain_indices: [paper_idx, ...]               — subset for pass 2
      secondary:         {paper_idx: [gid, ...]}        — sparse, stable IDs only
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return None

    if not isinstance(data, dict) or "assignments" not in data:
        print("    Missing 'assignments' key.")
        return None

    assignments_raw = data["assignments"]
    uncertain_raw   = data.get("uncertain", [])

    if not isinstance(assignments_raw, list):
        print(f"    'assignments' must be a list, got {type(assignments_raw).__name__}.")
        return None
    if not isinstance(uncertain_raw, list):
        print(f"    'uncertain' must be a list, got {type(uncertain_raw).__name__}.")
        return None

    # Allow short arrays: Haiku occasionally drops a few trailing entries.
    # Hard-fail only if the array is more than 10% short (likely garbage).
    n_received = len(assignments_raw)
    if n_received > n_papers:
        print(f"    'assignments' length {n_received} > n_papers {n_papers} "
              f"— truncating excess.")
        assignments_raw = assignments_raw[:n_papers]
    elif n_received < int(n_papers * 0.90):
        print(f"    'assignments' length {n_received} is >10% short of "
              f"n_papers {n_papers} — hard fail.")
        return None

    assignments: dict[int, int] = {}
    for i, v in enumerate(assignments_raw):
        if not isinstance(v, int) or v < 0 or v > 13:
            print(f"    Invalid stable group_id {v!r} at index {i} "
                  f"(must be integer 0-13).")
            return None
        assignments[i] = v

    # Fill any gaps (trailing entries Haiku dropped) with group 0
    # and route them to uncertain for pass 2 review.
    gap_indices: list[int] = []
    for i in range(len(assignments_raw), n_papers):
        assignments[i] = 0
        gap_indices.append(i)
    if gap_indices:
        print(f"    {len(gap_indices)} trailing paper(s) not returned by Haiku "
              f"— defaulted to group 0, routed to uncertain.")

    uncertain: list[int] = list(gap_indices)  # gaps go to uncertain first
    seen: set[int]       = set(gap_indices)
    for idx in uncertain_raw:
        try:
            i = int(idx)
        except (ValueError, TypeError):
            print(f"    Non-integer uncertain index: {idx!r}")
            return None
        if i not in assignments:
            print(f"    Uncertain index {i} not in assignments — skipping.")
            continue
        if i not in seen:
            uncertain.append(i)
            seen.add(i)

    # Parse secondary sparse dict (new in v3)
    secondary_raw = data.get("secondary", {})
    secondary: dict[int, list[int]] = {}
    if isinstance(secondary_raw, dict):
        valid_stable = set(range(14))
        for k, v in secondary_raw.items():
            try:
                idx = int(k)
            except (ValueError, TypeError):
                continue
            if idx not in assignments:
                continue
            primary_gid = assignments[idx]
            if not isinstance(v, list):
                continue
            cleaned = [
                t for t in v
                if isinstance(t, int) and t in valid_stable and t != primary_gid
            ]
            if cleaned:
                secondary[idx] = cleaned[:2]

    n_certain = n_papers - len(uncertain)
    print(f"    Pass 1: {n_certain} certain → stable, "
          f"{len(uncertain)} uncertain → pass 2. "
          f"{len(secondary)} papers with secondary tags.")
    return assignments, uncertain, secondary


# ── Pass 2: dynamic grouping ──────────────────────────────────────────────────

def _build_pass2_message(
    uncertain_df: pd.DataFrame,
    taxonomy: dict,
    next_id: int,
    all_known_dynamic: dict[int, str],
) -> str:
    """Build pass 2 user message.

    all_known_dynamic: {group_id: name} for all currently-known dynamic groups
    (taxonomy + any already created this run, e.g. during review rebuild).
    """
    if all_known_dynamic:
        dyn_lines = [
            "EXISTING DYNAMIC CATEGORIES (prefer these if the paper fits well):"
        ]
        for gid in sorted(all_known_dynamic.keys()):
            dyn_lines.append(f"  ID {gid} — {all_known_dynamic[gid]}")
        dyn_block = "\n".join(dyn_lines)
    else:
        dyn_block = (
            "EXISTING DYNAMIC CATEGORIES: None yet — "
            "create new ones as needed."
        )

    lines = [
        f"The following {len(uncertain_df)} papers did not clearly fit standard "
        "stable AI research categories. Assign each to an existing dynamic "
        f"category, a stable category (IDs 0-13) if appropriate, or create a new "
        f"dynamic category (use IDs starting at {next_id}).\n"
        f"{dyn_block}\n"
        "- Prefer existing groups over creating new ones.\n"
        "- Mark confidence 'high' if the fit is clear, 'low' if uncertain.\n"
        "Return JSON only — no preamble, no markdown:\n"
        '{"groups": {"14": "Continual Learning & Adaptation", '
        '"22": "New Topic Name"},\n'
        ' "assignments": {"0": {"group": 14, "confidence": "high"}, '
        '"1": {"group": 22, "confidence": "low"}, ...}}\n'
        "Every paper index 0 to N-1 must appear in 'assignments'.\n"
    ]

    for local_i, (_, row) in enumerate(uncertain_df.iterrows()):
        title    = str(row["title"]).strip()
        abstract = _strip_urls(str(row.get("abstract", ""))).strip()
        snippet  = abstract[:ABSTRACT_GROUPING_CHARS]
        lines.append(f"\n[{local_i}] Title: {title}\nAbstract: {snippet}")

    return "\n".join(lines)


def _parse_pass2_response(
    text: str,
    n_papers: int,
    existing_dynamic_ids: set[int],
) -> tuple[dict[int, dict], dict[int, str]] | None:
    """Parse pass 2 / review response.

    Returns (assignments, new_group_names) or None on hard failure.
      assignments:     {local_idx: {"group": int, "confidence": str, "secondary": [int]}}
      new_group_names: {group_id: name}  — only groups NOT in existing_dynamic_ids
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    # Truncate at last } to strip any trailing explanation Haiku appends after the JSON
    last_brace = text.rfind("}")
    if last_brace != -1:
        text = text[:last_brace + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return None

    if not isinstance(data, dict) or "groups" not in data or "assignments" not in data:
        print("    Missing 'groups' or 'assignments' key.")
        return None

    groups_raw      = data["groups"]
    assignments_raw = data["assignments"]

    if not isinstance(groups_raw, dict) or not isinstance(assignments_raw, dict):
        print("    'groups' and 'assignments' must be dicts.")
        return None

    # Parse groups dict
    groups: dict[int, str] = {}
    for k, v in groups_raw.items():
        try:
            gid = int(k)
        except (ValueError, TypeError):
            print(f"    Non-integer group_id in 'groups': {k!r}")
            return None
        groups[gid] = str(v).strip()

    # Parse assignments
    assignments: dict[int, dict] = {}
    for k, v in assignments_raw.items():
        try:
            local_i = int(k)
        except (ValueError, TypeError):
            print(f"    Non-integer paper index: {k!r}")
            return None

        if not isinstance(v, dict) or "group" not in v or "confidence" not in v:
            print(f"    Assignment {local_i} missing 'group' or 'confidence'.")
            return None

        gid  = v["group"]
        conf = v["confidence"]

        if not isinstance(gid, int) or gid < 0:
            print(f"    Invalid group_id {gid!r} at index {local_i}.")
            return None

        if conf not in ("high", "low"):
            print(f"    Invalid confidence '{conf}' at index {local_i} "
                  f"— treating as 'low'.")
            conf = "low"

        # Auto-fill stable groups missing from groups dict
        if gid <= 13:
            if gid not in groups:
                groups[gid] = _STABLE_BUCKET_NAMES.get(gid, f"Group {gid}")
        elif gid not in groups:
            print(f"    group_id {gid} used in assignments but missing from 'groups'.")
            return None

        # Secondary tags (new in v3) — optional, default []
        raw_sec = v.get("secondary", [])
        secondary_ids: list[int] = []
        if isinstance(raw_sec, list):
            for t in raw_sec:
                if isinstance(t, int) and t != gid and t >= 0:
                    secondary_ids.append(t)
        secondary_ids = secondary_ids[:2]

        assignments[local_i] = {"group": gid, "confidence": conf,
                                 "secondary": secondary_ids}

    missing = set(range(n_papers)) - set(assignments.keys())
    if missing:
        print(f"    Missing paper indices: {sorted(missing)[:10]}...")
        return None

    used_gids    = {a["group"] for a in assignments.values()}
    n_stable     = sum(1 for g in used_gids if g <= 13)
    n_exist_dyn  = sum(1 for g in used_gids if g >= 14 and g in existing_dynamic_ids)
    n_new_dyn    = sum(1 for g in used_gids if g >= 14 and g not in existing_dynamic_ids)
    n_high       = sum(1 for a in assignments.values() if a["confidence"] == "high")
    n_low        = len(assignments) - n_high

    print(f"    {n_stable} stable, {n_exist_dyn} existing dynamic, "
          f"{n_new_dyn} new dynamic. "
          f"Confidence: {n_high} high / {n_low} low.")

    new_group_names = {
        gid: name for gid, name in groups.items()
        if gid >= 14 and gid not in existing_dynamic_ids
    }
    return assignments, new_group_names


# ── Taxonomy update ───────────────────────────────────────────────────────────

def _update_taxonomy(
    taxonomy: dict,
    run_group_stats: dict[int, dict],
    today: str,
) -> dict:
    """Update dynamic taxonomy with this run's results.

    run_group_stats: {group_id: {"name": str, "n_high": int, "n_low": int}}

    Updates confidence_history and last_seen for groups used today.
    Adds new groups. Retires groups absent for TAXONOMY_RETIRE_DAYS.
    """
    groups    = taxonomy.get("groups", {})
    today_dt  = datetime.strptime(today, "%Y-%m-%d")

    for gid, stats in run_group_stats.items():
        gid_str = str(gid)
        n_total = stats["n_high"] + stats["n_low"]
        conf_rate = stats["n_high"] / n_total if n_total > 0 else 0.0

        if gid_str in groups:
            g = groups[gid_str]
            g["last_seen"] = today
            g["name"]      = stats["name"]   # keep name current
            hist = g.get("confidence_history", [])
            hist.append(round(conf_rate, 3))
            g["confidence_history"]  = hist[-TAXONOMY_CONFIDENCE_WINDOW:]
            cnt_hist = g.get("paper_count_history", [])
            cnt_hist.append(n_total)
            g["paper_count_history"] = cnt_hist[-TAXONOMY_CONFIDENCE_WINDOW:]
        else:
            groups[gid_str] = {
                "name":                stats["name"],
                "created":             today,
                "last_seen":           today,
                "confidence_history":  [round(conf_rate, 3)],
                "paper_count_history": [n_total],
            }

    # Retire groups absent for TAXONOMY_RETIRE_DAYS
    to_retire = []
    for gid_str, g in groups.items():
        try:
            last_dt  = datetime.strptime(g.get("last_seen", "2000-01-01"), "%Y-%m-%d")
            age_days = (today_dt - last_dt).days
            if age_days >= TAXONOMY_RETIRE_DAYS:
                to_retire.append(gid_str)
        except ValueError:
            pass

    for gid_str in to_retire:
        print(f"  Retiring group {gid_str} "
              f"('{groups[gid_str]['name']}') — absent ≥{TAXONOMY_RETIRE_DAYS} days.")
        del groups[gid_str]

    taxonomy["groups"] = groups

    n_used    = len(run_group_stats)
    n_retired = len(to_retire)
    n_active  = len(groups)
    print(f"  Taxonomy update: {n_used} groups active today, "
          f"{n_retired} retired, {n_active} total.")
    return taxonomy


# ── Review call ───────────────────────────────────────────────────────────────

def _build_review_message(
    review_df: pd.DataFrame,
    current_assignments: dict[int, int],
    all_known_dynamic: dict[int, str],
    weak_group_names: dict[int, str],
) -> str:
    """Build review call message for low-confidence papers in weak groups."""
    weak_ctx = ", ".join(
        f"ID {gid} '{name}'" for gid, name in sorted(weak_group_names.items())
    )

    stable_block = "\n".join(
        f"  ID {gid} — {name}"
        for gid, name in sorted(_STABLE_BUCKET_NAMES.items())
    )
    dyn_block_lines = [
        f"  ID {gid} — {name}"
        for gid, name in sorted(all_known_dynamic.items())
    ]
    dyn_block = "\n".join(dyn_block_lines) if dyn_block_lines else "  (none yet)"

    lines = [
        f"The following {len(review_df)} papers were assigned to groups with "
        f"declining confidence ({weak_ctx}). Please reassign each paper to its "
        "best fit. You may use any stable or existing dynamic group, or create a "
        "new one if genuinely warranted.\n"
        f"STABLE CATEGORIES:\n{stable_block}\n"
        f"EXISTING DYNAMIC CATEGORIES:\n{dyn_block}\n"
        "Mark confidence 'high' if the fit is clear, 'low' if still uncertain.\n"
        "Return JSON only — same format as before:\n"
        '{"groups": {"14": "name", ...}, '
        '"assignments": {"0": {"group": 7, "confidence": "high"}, ...}}\n'
    ]

    for local_i, (_, row) in enumerate(review_df.iterrows()):
        title       = str(row["title"]).strip()
        abstract    = _strip_urls(str(row.get("abstract", ""))).strip()
        snippet     = abstract[:ABSTRACT_GROUPING_CHARS]
        current_gid = current_assignments.get(local_i, -1)
        current_nm  = weak_group_names.get(current_gid, f"Group {current_gid}")
        lines.append(
            f"\n[{local_i}] Current group: '{current_nm}'\n"
            f"Title: {title}\nAbstract: {snippet}"
        )

    return "\n".join(lines)


# ── Group names cache ─────────────────────────────────────────────────────────

def _save_group_names_cache(group_names: dict[int, str]) -> None:
    with open(GROUP_NAMES_CACHE, "w") as f:
        json.dump({str(k): v for k, v in group_names.items()}, f, indent=2)
    print(f"  Group names cached to {GROUP_NAMES_CACHE}.")


# ── Excess-group merge (safety fallback) ──────────────────────────────────────

def _merge_excess_groups(
    mapping: dict[int, int],
    group_names: dict[int, str],
    df: pd.DataFrame,
    target_max: int,
) -> tuple[dict[int, int], dict[int, str]]:
    """Merge excess groups down to target_max by absorbing closest pairs
    (smallest mean inter-group SPECTER2 cosine distance).
    The larger group keeps its name; IDs are remapped to 0..n-1 afterward.
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    n_groups = len(set(mapping.values()))
    if n_groups <= target_max:
        return mapping, group_names

    print(f"\n  Merging {n_groups} → {target_max} groups (absorbing closest pairs)...")

    all_vecs     = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
    all_pairwise = cosine_distances(all_vecs)

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
        print(f"    Merge group {absorb} ('{group_names[absorb]}', "
              f"n={len(active[absorb])}) → group {keep} "
              f"('{group_names[keep]}', n={len(active[keep])})  "
              f"dist={best_dist:.4f}")

        active[keep].extend(active.pop(absorb))
        del group_names[absorb]
        for pos in active[keep]:
            mapping[pos] = keep

    id_remap    = {old: new for new, old in enumerate(sorted(active.keys()))}
    new_mapping = {pos: id_remap[g] for pos, g in mapping.items()}
    new_names   = {id_remap[old]: name for old, name in group_names.items() if old in id_remap}
    print(f"  Merge complete: {len(active)} groups remaining.")
    return new_mapping, new_names


# ── End-of-run reconciliation ─────────────────────────────────────────────────

def reconcile_dynamic_groups(
    group_names: dict[int, str],
    df: pd.DataFrame,
    final_assignment: dict[int, int],
    client,
) -> dict[int, int]:
    """Ask Haiku to identify near-duplicate dynamic groups and return a merge map.

    Only operates on dynamic groups (IDs >= 14).
    Returns {absorb_id: keep_id} — may be empty if no merges warranted.
    Called after all pass 1 / pass 2 / review assignments are finalised,
    before MDS layout.
    """
    dynamic = {gid: name for gid, name in group_names.items() if gid >= 14}
    if len(dynamic) < 4:
        print("  Reconciliation: fewer than 4 dynamic groups — skipping.")
        return {}

    counts: dict[int, int] = {}
    for gid in final_assignment.values():
        counts[gid] = counts.get(gid, 0) + 1

    lines = [
        f"  {gid}: \"{name}\" ({counts.get(gid, 0)} papers)"
        for gid, name in sorted(dynamic.items())
    ]
    prompt = (
        "Below is a list of dynamic AI research topic groups with paper counts.\n"
        "Identify pairs that are near-duplicates or clearly redundant "
        "(same concept, different phrasing — e.g. 'LLM Reasoning' vs "
        "'Reasoning in LLMs'). Be conservative: only merge when the names are "
        "unambiguously the same concept. Do NOT merge groups that are merely "
        "related or overlapping.\n\n"
        "GROUPS:\n" + "\n".join(lines) + "\n\n"
        "Respond ONLY with a JSON array of merge operations, or [] if none.\n"
        "Each operation: {\"keep_id\": <int>, \"absorb_id\": <int>}\n"
        "keep_id survives; absorb_id is deleted and its papers move to keep_id.\n"
        "Prefer keeping the group with more papers.\n"
        "Example: [{\"keep_id\": 14, \"absorb_id\": 19}]"
    )

    print(f"\n  Reconciliation — checking {len(dynamic)} dynamic groups for duplicates...")
    raw = _haiku_call(client, "", prompt, max_tokens=1024)
    if raw is None:
        print("  Reconciliation: API call failed — skipping.")
        return {}

    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    try:
        ops = json.loads(raw)
    except json.JSONDecodeError:
        print("  Reconciliation: JSON parse failed — skipping.")
        return {}

    if not isinstance(ops, list):
        print("  Reconciliation: unexpected response format — skipping.")
        return {}

    remap: dict[int, int] = {}
    for op in ops:
        keep   = op.get("keep_id")
        absorb = op.get("absorb_id")
        if (isinstance(keep, int) and isinstance(absorb, int)
                and keep in dynamic and absorb in dynamic
                and keep != absorb and absorb not in remap):
            remap[absorb] = keep
            print(f"  Reconciliation: merging {absorb} "
                  f"(\"{group_names[absorb]}\") → {keep} "
                  f"(\"{group_names[keep]}\")")

    if not remap:
        print("  Reconciliation: no merges needed.")
    return remap


# ── Main Stage 3 entry point ──────────────────────────────────────────────────

def haiku_group_papers(
    df: pd.DataFrame,
    client,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Stage 3: Two-pass Haiku grouping with persistent dynamic taxonomy.

    Pass 1 — All papers, titles only (single call):
      Assigns each paper to a stable category (0-13).
      Uncertain papers are flagged for pass 2.

    Pass 2 — Uncertain papers, with abstracts (single call):
      Assigns to existing dynamic groups (from taxonomy) or creates new ones.
      Returns per-paper confidence ratings.

    Review — Optional targeted call:
      Papers in groups whose rolling confidence has fallen below the
      maturity-modulated threshold are re-evaluated by Haiku.

    Taxonomy is loaded, updated, and saved each run. Groups absent for
    TAXONOMY_RETIRE_DAYS are retired automatically.

    Returns (df, group_names).
    """
    df    = df.reset_index(drop=True)
    n     = len(df)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\n▶  Stage 3 — Haiku grouping + naming ({n} papers, two-pass)...")

    # ── Load or seed dynamic taxonomy ────────────────────────────────────────
    if os.path.exists(TAXONOMY_PATH):
        taxonomy = load_dynamic_taxonomy()
    else:
        print(f"  {TAXONOMY_PATH} not found — seeding from {GROUP_NAMES_CACHE}.")
        taxonomy = seed_dynamic_taxonomy_from_cache(today)
        save_dynamic_taxonomy(taxonomy)

    existing_dynamic_ids: set[int] = {
        int(k) for k in taxonomy.get("groups", {}).keys()
    }
    all_known_dynamic: dict[int, str] = {
        int(k): v["name"] for k, v in taxonomy.get("groups", {}).items()
    }

    # ── Pass 1: stable assignment (batched, titles only) ─────────────────────
    # Batched to keep each output array short and reliable.
    # Each batch returns an array of integers; merge by concatenation.
    import math as _math
    n_p1_batches = _math.ceil(n / PASS1_BATCH_SIZE)
    print(f"\n  Pass 1 — stable assignment ({n} papers, titles only, "
          f"{n_p1_batches} batch{'es' if n_p1_batches > 1 else ''})...")

    stable_assignments: dict[int, int] = {}
    uncertain_indices:  list[int]      = []
    p1_secondary_map:   dict[int, list[int]] = {}   # paper_idx → secondary gids (pass 1)
    p1_failed = False

    for b in range(n_p1_batches):
        b_start  = b * PASS1_BATCH_SIZE
        b_end    = min(b_start + PASS1_BATCH_SIZE, n)
        b_size   = b_end - b_start
        batch_df = df.iloc[b_start:b_end].reset_index(drop=True)
        p1_msg   = _build_pass1_message(batch_df)
        approx_tk = len(p1_msg) // 4
        print(f"  Batch {b+1}/{n_p1_batches} "
              f"(papers {b_start}–{b_end-1}, ≈{approx_tk:,} tokens)...")

        b_result = None
        for attempt in range(1, GROUPING_MAX_RETRIES + 1):
            print(f"    Attempt {attempt}/{GROUPING_MAX_RETRIES}...")
            raw = _haiku_call(client, _PASS1_SYSTEM, p1_msg, max_tokens=4096)
            if raw is not None:
                print(f"    Response length: {len(raw)} chars.")
                b_result = _parse_pass1_response(raw, b_size)
                if b_result is not None:
                    print(f"    ✓ Parsed.")
                    break
                else:
                    print(f"    Full response dump:\n{raw}")

            if attempt < GROUPING_MAX_RETRIES:
                wait = GROUPING_RETRY_BASE_WAIT * (2 ** (attempt - 1))
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

        if b_result is None:
            print(f"  ✗ Pass 1 batch {b+1} failed — falling back to HDBSCAN.")
            p1_failed = True
            break

        b_assignments, b_uncertain, b_secondary = b_result
        # Merge into global: offset local indices by batch start
        for local_i, gid in b_assignments.items():
            stable_assignments[b_start + local_i] = gid
        for local_i in b_uncertain:
            uncertain_indices.append(b_start + local_i)
        for local_i, sec in b_secondary.items():
            p1_secondary_map[b_start + local_i] = sec

    if p1_failed:
        print("  ✗ Pass 1 failed — falling back to HDBSCAN.")
        fallback    = _hdbscan_fallback_grouping(df)
        group_names = {gid: f"Group {gid}" for gid in set(fallback.values())}
        df["group_id_v3"] = [fallback[i] for i in range(n)]
        _save_group_names_cache(group_names)
        return df, group_names

    n_uncertain_total = len(uncertain_indices)
    n_certain_total   = n - n_uncertain_total
    print(f"  Pass 1 complete: {n_certain_total} certain → stable, "
          f"{n_uncertain_total} uncertain → pass 2.")

    # Apply uncertain cap
    if len(uncertain_indices) > PASS1_UNCERTAIN_CAP:
        n_over = len(uncertain_indices) - PASS1_UNCERTAIN_CAP
        print(f"  ⚠ {len(uncertain_indices)} uncertain exceeds cap ({PASS1_UNCERTAIN_CAP}). "
              f"Force-assigning {n_over} papers to stable best-guess.")
        uncertain_indices = uncertain_indices[:PASS1_UNCERTAIN_CAP]

    # ── Pass 2: dynamic grouping (uncertain papers, with abstracts) ───────────
    next_id      = taxonomy.get("next_id", 14)
    uncertain_df = df.iloc[uncertain_indices].reset_index(drop=True)
    n_uncertain  = len(uncertain_df)

    p2_assignments:  dict[int, dict]   = {}
    new_group_names: dict[int, str]    = {}

    if n_uncertain == 0:
        print("\n  Pass 2 — skipped (all papers assigned to stable categories).")
    else:
        print(f"\n  Pass 2 — dynamic grouping ({n_uncertain} papers, with abstracts)...")
        p2_msg         = _build_pass2_message(uncertain_df, taxonomy, next_id,
                                               all_known_dynamic)
        approx_tokens2 = len(p2_msg) // 4
        print(f"  Prompt ≈ {approx_tokens2:,} tokens.")

        p2_result = None
        for attempt in range(1, GROUPING_MAX_RETRIES + 1):
            print(f"  Attempt {attempt}/{GROUPING_MAX_RETRIES}...")
            raw = _haiku_call(client, _PASS2_SYSTEM, p2_msg, max_tokens=8192)
            if raw is not None:
                print(f"  Response length: {len(raw)} chars.")
                p2_result = _parse_pass2_response(raw, n_uncertain,
                                                   existing_dynamic_ids)
                if p2_result is not None:
                    print(f"  ✓ Pass 2 parsed.")
                    break
            if attempt < GROUPING_MAX_RETRIES:
                wait = GROUPING_RETRY_BASE_WAIT * (2 ** (attempt - 1))
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

        if p2_result is None:
            print("  ✗ Pass 2 failed — uncertain papers kept at stable best-guess.")
            p2_assignments  = {}
            new_group_names = {}
        else:
            p2_assignments, new_group_names = p2_result
            # Update next_id and known dynamic map for any new groups
            if new_group_names:
                new_max = max(new_group_names.keys())
                taxonomy["next_id"] = max(taxonomy.get("next_id", 14), new_max + 1)
                all_known_dynamic.update(new_group_names)

    # ── Merge pass 1 + pass 2 assignments into final mapping ─────────────────
    final_assignment: dict[int, int] = {}
    confidence_map:   dict[int, str] = {}
    secondary_map:    dict[int, list[int]] = {}   # global paper_idx → secondary gids

    for i in range(n):
        final_assignment[i] = stable_assignments[i]
        confidence_map[i]   = "high"   # stable assignments are implicitly high
        secondary_map[i]    = p1_secondary_map.get(i, [])

    for local_i, asgn in p2_assignments.items():
        global_i = uncertain_indices[local_i]
        final_assignment[global_i] = asgn["group"]
        confidence_map[global_i]   = asgn["confidence"]
        secondary_map[global_i]    = asgn.get("secondary", [])

    # ── Build group_names from all sources ────────────────────────────────────
    group_names: dict[int, str] = {}
    for gid in set(final_assignment.values()):
        if gid <= 13:
            group_names[gid] = _STABLE_BUCKET_NAMES.get(gid, f"Group {gid}")
        elif gid in all_known_dynamic:
            group_names[gid] = all_known_dynamic[gid]
        else:
            group_names[gid] = f"Emerging Topic {gid}"

    # ── Compute per-group confidence stats for taxonomy update ────────────────
    def _build_run_stats(
        final_asgn: dict[int, int],
        conf_map: dict[int, str],
        gnames: dict[int, str],
    ) -> dict[int, dict]:
        stats: dict[int, dict] = {}
        for gi, gid in final_asgn.items():
            if gid < 14:
                continue
            conf = conf_map.get(gi, "high")
            if gid not in stats:
                stats[gid] = {"name": gnames.get(gid, f"Group {gid}"),
                               "n_high": 0, "n_low": 0}
            if conf == "high":
                stats[gid]["n_high"] += 1
            else:
                stats[gid]["n_low"] += 1
        return stats

    run_group_stats = _build_run_stats(final_assignment, confidence_map, group_names)

    # ── Review: identify weak groups, re-evaluate their low-confidence papers ─
    weak_group_ids: set[int] = set()
    for gid, stats in run_group_stats.items():
        gid_str  = str(gid)
        existing = taxonomy.get("groups", {}).get(gid_str, {})
        hist     = existing.get("confidence_history", [])
        n_total  = stats["n_high"] + stats["n_low"]
        conf_rate = stats["n_high"] / n_total if n_total > 0 else 0.0
        projected = (hist + [conf_rate])[-TAXONOMY_CONFIDENCE_WINDOW:]
        n_runs    = len(projected)
        threshold = _review_threshold(n_runs)
        if threshold is None:
            continue
        rolling_mean = sum(projected) / len(projected)
        if rolling_mean < threshold:
            weak_group_ids.add(gid)
            print(f"  ⚠ Group {gid} ('{stats['name']}'): "
                  f"rolling confidence {rolling_mean:.2f} < "
                  f"threshold {threshold:.2f} (n_runs={n_runs}) "
                  f"— flagged for review.")

    if weak_group_ids:
        review_global_idxs = [
            i for i, gid in final_assignment.items()
            if gid in weak_group_ids and confidence_map.get(i, "high") == "low"
        ]

        if review_global_idxs:
            weak_gnames = {
                gid: group_names[gid]
                for gid in weak_group_ids if gid in group_names
            }
            review_df  = df.iloc[review_global_idxs].reset_index(drop=True)
            l2g        = {li: gi for li, gi in enumerate(review_global_idxs)}
            local_curr = {li: final_assignment[gi] for li, gi in l2g.items()}

            print(f"\n  Review — {len(review_global_idxs)} low-confidence papers "
                  f"from {len(weak_group_ids)} weak groups...")
            rev_msg    = _build_review_message(
                review_df, local_curr, all_known_dynamic, weak_gnames
            )
            approx_tkr = len(rev_msg) // 4
            print(f"  Review prompt ≈ {approx_tkr:,} tokens.")

            rev_result = None
            for attempt in range(1, GROUPING_MAX_RETRIES + 1):
                print(f"  Review attempt {attempt}/{GROUPING_MAX_RETRIES}...")
                raw = _haiku_call(client, _PASS2_SYSTEM, rev_msg, max_tokens=4096)
                if raw is not None:
                    print(f"  Response length: {len(raw)} chars.")
                    # existing_dynamic_ids for review = all currently known
                    all_known_now = existing_dynamic_ids | set(new_group_names.keys())
                    rev_result    = _parse_pass2_response(
                        raw, len(review_df), all_known_now
                    )
                    if rev_result is not None:
                        print("  ✓ Review parsed.")
                        break
                if attempt < GROUPING_MAX_RETRIES:
                    wait = GROUPING_RETRY_BASE_WAIT * (2 ** (attempt - 1))
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

            if rev_result is not None:
                rev_assignments, rev_new_groups = rev_result
                # Apply review overrides
                for li, asgn in rev_assignments.items():
                    gi = l2g[li]
                    final_assignment[gi] = asgn["group"]
                    confidence_map[gi]   = asgn["confidence"]
                    secondary_map[gi]    = asgn.get("secondary", [])
                # Add any new groups from review
                group_names.update(rev_new_groups)
                all_known_dynamic.update(rev_new_groups)
                if rev_new_groups:
                    rev_max = max(rev_new_groups.keys())
                    taxonomy["next_id"] = max(
                        taxonomy.get("next_id", 14), rev_max + 1
                    )
                # Recompute run stats after review overrides
                run_group_stats = _build_run_stats(
                    final_assignment, confidence_map, group_names
                )
                print(f"  Review complete: {len(rev_assignments)} papers "
                      f"re-evaluated.")
            else:
                print("  ✗ Review failed — keeping pass 2 assignments.")

    # ── Update and save dynamic taxonomy ─────────────────────────────────────
    taxonomy = _update_taxonomy(taxonomy, run_group_stats, today)
    save_dynamic_taxonomy(taxonomy)

    # ── Safety: merge down if over GROUP_COUNT_MAX ────────────────────────────
    if len(set(final_assignment.values())) > GROUP_COUNT_MAX:
        final_assignment, group_names = _merge_excess_groups(
            final_assignment, group_names, df, target_max=GROUP_COUNT_MAX
        )

    # ── End-of-run reconciliation: merge near-duplicate dynamic groups ────────
    remap = reconcile_dynamic_groups(group_names, df, final_assignment, client)
    if remap:
        for i in range(n):
            gid = final_assignment[i]
            if gid in remap:
                final_assignment[i] = remap[gid]
            # Remap secondary IDs too; remove any that collapse to the new primary
            old_sec = secondary_map.get(i, [])
            new_primary = final_assignment[i]
            new_sec = [remap.get(t, t) for t in old_sec]
            new_sec = [t for t in new_sec if t != new_primary]
            secondary_map[i] = list(dict.fromkeys(new_sec))[:2]  # dedup + cap
        # Remove absorbed group names
        for absorb_id in remap:
            group_names.pop(absorb_id, None)
        print(f"  Reconciliation applied: {len(remap)} group(s) absorbed.")

    # ── Apply to DataFrame ────────────────────────────────────────────────────
    df["group_id_v3"] = [final_assignment[i] for i in range(n)]

    # secondary_tags: list of group name strings (empty list for most papers)
    def _sec_names(i: int) -> list[str]:
        return [
            group_names.get(gid, f"Group {gid}")
            for gid in secondary_map.get(i, [])
            if gid in group_names
        ]
    df["secondary_tags"] = [_sec_names(i) for i in range(n)]
    n_with_secondary = (df["secondary_tags"].apply(len) > 0).sum()
    print(f"  {n_with_secondary}/{n} papers have ≥1 secondary tag.")

    group_counts = df["group_id_v3"].value_counts().sort_index()
    for gid, count in group_counts.items():
        print(f"    Group {gid:>2} ({count:>3} papers): "
              f"'{group_names.get(gid, '?')}'")
    print(f"  Total: {len(group_counts)} groups, {n} papers assigned.")

    # Guard: ensure group_names covers every ID in final_assignment.
    # The review call can route papers to existing-taxonomy groups that were
    # never in pass 1/2 assignments, so those IDs won't be in group_names yet.
    for _gid in set(final_assignment.values()):
        if _gid not in group_names:
            if _gid <= 13:
                group_names[_gid] = _STABLE_BUCKET_NAMES.get(_gid, f"Group {_gid}")
            elif _gid in all_known_dynamic:
                group_names[_gid] = all_known_dynamic[_gid]
            else:
                group_names[_gid] = f"Emerging Topic {_gid}"
    _save_group_names_cache(group_names)
    return df, group_names

def _load_group_names_cache(df: pd.DataFrame | None = None) -> dict[int, str]:
    """Load group names from cache (used in offline mode).

    If the cache file is missing, falls back to deriving generic names from
    the group_id_v3 column of df (if provided) rather than crashing.
    """
    if os.path.exists(GROUP_NAMES_CACHE):
        with open(GROUP_NAMES_CACHE) as f:
            raw = json.load(f)
        names = {int(k): v for k, v in raw.items()}
        print(f"  Loaded {len(names)} group names from {GROUP_NAMES_CACHE}.")
        return names

    # Cache missing — fall back to generic names derived from the parquet
    print(f"  WARNING: Group names cache '{GROUP_NAMES_CACHE}' not found.")
    if df is not None and "group_id_v3" in df.columns:
        gids  = sorted(int(g) for g in df["group_id_v3"].unique())
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
        return {i: i % 8 for i in range(len(df))}

    for mcs in [max(3, len(df) // 30), max(3, len(df) // 40), 3]:
        clusterer  = HDBSCAN(min_cluster_size=mcs, min_samples=3,
                             metric=metric, cluster_selection_method="leaf")
        labels     = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  HDBSCAN min_cluster_size={mcs} → {n_clusters} clusters.")
        if 5 <= n_clusters <= GROUP_COUNT_MAX + 5:
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
    group_ids = sorted(df["group_id_v3"].unique())
    n_groups  = len(group_ids)
    print(f"  Computing {n_groups}×{n_groups} group distance matrix...")

    # group_id → list of integer row positions in df
    group_pos: dict[int, list[int]] = {gid: [] for gid in group_ids}
    for pos, gid in enumerate(df["group_id_v3"].tolist()):
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

    mds = MDS(n_components=2, metric_mds=True, metric="precomputed",
              init="random", random_state=42, n_init=4, max_iter=500,
              normalized_stress="auto")
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

    Writes: projection_v3_x, projection_v3_y
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize as sk_normalize

    print("\n▶  Stage 4b — Within-group scatter (fraction of centroid spacing)...")

    df = df.copy()
    df["projection_v3_x"] = np.nan
    df["projection_v3_y"] = np.nan

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
        mask      = df["group_id_v3"] == gid
        positions = df.index[mask].tolist()
        pos_array = [df.index.get_loc(p) for p in positions]
        n_g       = len(positions)
        name      = group_names.get(gid, f"Group {gid}")
        mds_cx, mds_cy = centroids[gid]

        if n_g == 1:
            df.at[positions[0], "projection_v3_x"] = mds_cx
            df.at[positions[0], "projection_v3_y"] = mds_cy
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
            df.at[df_idx, "projection_v3_x"] = mds_cx + (dx / length) * scatter_dist
            df.at[df_idx, "projection_v3_y"] = mds_cy + (dy / length) * scatter_dist

    n_placed = df["projection_v3_x"].notna().sum()
    x_range  = (df["projection_v3_x"].min(), df["projection_v3_x"].max())
    y_range  = (df["projection_v3_y"].min(), df["projection_v3_y"].max())
    df["projection_v3_x"] = df["projection_v3_x"] * LAYOUT_X_SCALE
    df["projection_v3_y"] = df["projection_v3_y"] * LAYOUT_Y_SCALE
    x_range  = (df["projection_v3_x"].min(), df["projection_v3_x"].max())
    y_range  = (df["projection_v3_y"].min(), df["projection_v3_y"].max())
    print(f"\n  Placed {n_placed}/{len(df)} papers.")
    print(f"  Final layout bounds (after aspect scaling): x=[{x_range[0]:.2f}, {x_range[1]:.2f}], "
          f"y=[{y_range[0]:.2f}, {y_range[1]:.2f}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WRITE LABELS PARQUET
# ══════════════════════════════════════════════════════════════════════════════

def write_labels_parquet(
    df: pd.DataFrame,
    labels: dict[int, str],
    centroids: dict[int, tuple[float, float]],
    labels_path: str = "labels_v3.parquet",
) -> str:
    """Write labels.parquet for the Atlas CLI --labels flag.

    Label x/y are the mean of each group's actual projection_v3_x/y positions —
    the visual centre of the dot cloud after scatter — not the MDS centroid.
    This keeps labels anchored to where their papers actually landed, regardless
    of how far the scatter pushed them from the centroid.
    """
    print(f"\n▶  Writing labels parquet ({len(centroids)} labels)...")
    rows = []
    for gid in sorted(centroids.keys()):
        label_text = labels.get(gid, f"Group {gid}")
        mask       = df["group_id_v3"] == gid
        n_papers   = int(mask.sum())
        # Mean of actual paper positions — tracks where dots landed after scatter
        cx = float(df.loc[mask, "projection_v3_x"].mean())
        cy = float(df.loc[mask, "projection_v3_y"].mean())
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
    print(f"  BACKFILL_HINDICES  : {BACKFILL_HINDICES}")
    print(f"  OPENALEX_OFFLINE   : {OPENALEX_OFFLINE_MODE}")
    print(f"  GROUP_COUNT        : up to {GROUP_COUNT_MAX} ({GROUP_STABLE_COUNT} stable + dynamic)")
    print(f"  PASS1_BATCH_SIZE   : {PASS1_BATCH_SIZE}")
    print(f"  PASS1_UNCERTAIN_CAP: {PASS1_UNCERTAIN_CAP}")
    print(f"  TAXONOMY_PATH      : {TAXONOMY_PATH}")
    print(f"  LAYOUT_SCALE       : {LAYOUT_SCALE}")
    print(f"  SCATTER_FRACTION   : {SCATTER_FRACTION}")
    print(f"  VARIANCE_AMPLIFIER : {VARIANCE_AMPLIFIER}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    # OFFLINE MODE — re-run layout + build only
    # ══════════════════════════════════════════════════════════════════════════
    if OFFLINE_MODE:
        print("\n▶  OFFLINE MODE — loading existing data, skipping all API calls...")

        df = load_existing_db(bypass_pruning=True)
        if df.empty:
            raise RuntimeError(
                "database.parquet is empty or missing. "
                "Run once in normal mode (remove OFFLINE_MODE from the YAML) to populate it."
            )
        if "group_id_v3" not in df.columns:
            raise RuntimeError(
                "database.parquet has no group_id_v3 column. "
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

        # Migrate Reputation → Prominence column name if present from older runs
        if "Reputation" in df.columns and "Prominence" not in df.columns:
            df = df.rename(columns={"Reputation": "Prominence"})
            print("  Migrated 'Reputation' column -> 'Prominence'.")

        # Recompute Prominence tier values if they look like the old two-value system
        if "Prominence" in df.columns:
            old_values = {"Reputation Enhanced", "Reputation Std"}
            if set(df["Prominence"].dropna().unique()).issubset(old_values):
                print("  Recomputing Prominence tiers (old two-value system detected)...")
                df["Prominence"] = df.apply(calculate_prominence, axis=1)
                tier_counts = df["Prominence"].value_counts()
                for tier in ["Elite", "Enhanced", "Emerging", "Unverified"]:
                    print(f"    {tier}: {tier_counts.get(tier, 0)}")

        # Ensure paper_source exists (needed for CitationTier)
        if "paper_source" not in df.columns:
            df["paper_source"] = "Recent"

        # Note: significant.parquet is NOT loaded in offline mode.
        # Significant papers lack embeddings and group_id_v3, so they cannot
        # participate in Stage 4 (MDS layout). Offline mode is for layout/UI
        # iteration only — CitationTier is still computed for Recent papers
        # (all "Cited" or blank) so the column exists in the build.

        # Ensure CitationTier exists; recompute so layout reflects current pool
        print("\n▶  (Offline) Computing CitationTier (Recent papers only)...")
        if "ss_citation_count" not in df.columns:
            df["ss_citation_count"] = 0
        if "ss_influential_citations" not in df.columns:
            df["ss_influential_citations"] = 0
        df["CitationTier"] = calculate_citation_tier(df)

        group_names = _load_group_names_cache(df)

        # Patch any group IDs in the parquet that are missing from the cache
        parquet_gids  = set(int(g) for g in df["group_id_v3"].unique())
        missing_names = parquet_gids - set(group_names.keys())
        if missing_names:
            print(f"  WARNING: {len(missing_names)} group IDs have no cached name "
                  f"{missing_names} — using 'Group N' fallback.")
            for gid in missing_names:
                group_names[gid] = f"Group {gid}"

        print(f"  Loaded {len(df)} papers, "
              f"{df['group_id_v3'].nunique()} groups, "
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
        print("\n▶  Stage 1 -- Loading rolling database...")

        # ── Load database.parquet (Recent papers only) ───────────────────────
        if os.path.exists(DB_PATH):
            existing_df = pd.read_parquet(DB_PATH)
        else:
            existing_df = pd.DataFrame()

        # Ensure paper_source column exists and default to "Recent"
        if not existing_df.empty and "paper_source" not in existing_df.columns:
            existing_df["paper_source"] = "Recent"

        # Remove any Significant papers that leaked into database.parquet from
        # a prior run — they'll be re-added cleanly from significant.parquet below
        if not existing_df.empty and "paper_source" in existing_df.columns:
            before_sig = len(existing_df)
            existing_df = existing_df[
                existing_df["paper_source"] != "Significant"
            ].copy().reset_index(drop=True)
            removed = before_sig - len(existing_df)
            if removed:
                print(f"  Removed {removed} Significant rows from database.parquet "
                      f"(will reload from significant.parquet).")

        # Prune Recent papers older than RETENTION_DAYS
        if not existing_df.empty and "date_added" in existing_df.columns:
            cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
            existing_df["date_added"] = pd.to_datetime(
                existing_df["date_added"], utc=True, errors="coerce"
            )
            before = len(existing_df)
            existing_df = existing_df[
                existing_df["date_added"] >= cutoff
            ].reset_index(drop=True)
            pruned = before - len(existing_df)
            if pruned:
                print(f"  Pruned {pruned} Recent papers older than {RETENTION_DAYS} days.")

        # Migrate Reputation → Prominence column name if present from older runs
        if "Reputation" in existing_df.columns and "Prominence" not in existing_df.columns:
            existing_df = existing_df.rename(columns={"Reputation": "Prominence"})
            print("  Migrated 'Reputation' column -> 'Prominence'.")

        print(f"  Loaded {len(existing_df)} existing Recent papers.")

        # ── Load and merge significant.parquet ──────────────────────────────
        if os.path.exists(SIGNIFICANT_PATH):
            sig_df = pd.read_parquet(SIGNIFICANT_PATH)
            sig_df["paper_source"] = "Significant"
            n_sig = len(sig_df)
            # Remove Significant papers that are also in the recent window
            # (prefer Recent so they get pruned naturally when they age out)
            sig_df = sig_df[~sig_df["id"].isin(existing_df["id"])].reset_index(drop=True)
            if len(sig_df) < n_sig:
                print(f"  {n_sig - len(sig_df)} Significant paper(s) already in "
                      f"recent window -- skipping duplicates.")
            existing_df = pd.concat([existing_df, sig_df], ignore_index=True)
            print(f"  Merged {len(sig_df)} Significant papers "
                  f"(total: {len(existing_df)}).")
        else:
            print("  No significant.parquet found -- running without Significant papers.")

        is_first_run = existing_df.empty and not os.path.exists(DB_PATH)
        days_back = 5 if is_first_run else 1

        if is_first_run:
            print("  First run — pre-filling with last 5 days of arXiv papers.")
        else:
            print(f"  Loaded {len(existing_df)} existing papers.")

        # ── Stage 1b: arXiv fetch (OAI-PMH announcement-date) ───────────────
        print("\n▶  Stage 1b — Fetching from arXiv (OAI-PMH)...")
        results = fetch_arxiv_oai(days_back=days_back, max_results=ARXIV_MAX)

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
                    "authors_list": [a.name for a in r.authors],
                })
            new_df = pd.DataFrame(rows)
            new_df["Prominence"]   = "Unverified"   # placeholder; recalculated in 1c
            new_df["paper_source"] = "Recent"
            df = merge_papers(existing_df, new_df)
            df = df.drop(columns=["group", "group_id_v3"], errors="ignore")
            print(f"  Rolling DB: {len(df)} papers after merge.")
        else:
            df = existing_df.drop(columns=["group", "group_id_v3"], errors="ignore")
            # Ensure paper_source is set for all rows
            if "paper_source" not in df.columns:
                df["paper_source"] = "Recent"

        # ── Stage 1c: Author h-index enrichment ─────────────────────────────
        print("\n▶  Stage 1c — Author h-index enrichment (OpenAlex)...")
        if "author_hindices" not in df.columns:
            df["author_hindices"] = None
        if "authors_list" not in df.columns:
            df["authors_list"] = None

        if OPENALEX_OFFLINE_MODE:
            print("  OPENALEX_OFFLINE_MODE=true — skipping OpenAlex fetch.")
            print("  Existing author_hindices carried forward; new papers get [].")
            # Fill any missing entries (new papers added this run)
            df["author_hindices"] = df["author_hindices"].apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            author_cache = load_author_cache()

            if BACKFILL_HINDICES:
                # Re-fetch for all rows, including those with empty lists
                needs_hindex = pd.Series([True] * len(df), index=df.index)
                print(f"  BACKFILL_HINDICES=true — re-fetching all {len(df)} papers.")
            else:
                # Normal mode: only fetch for rows missing h-index data
                needs_hindex = df["author_hindices"].isna() | df["author_hindices"].apply(
                    lambda x: isinstance(x, list) and len(x) == 0
                )

            n_needs = needs_hindex.sum()
            print(f"  {n_needs} papers need h-index lookup"
                  f" ({len(df) - n_needs} already have data).")

            for idx in df.index[needs_hindex]:
                authors = df.at[idx, "authors_list"]
                if not isinstance(authors, list) or len(authors) == 0:
                    df.at[idx, "author_hindices"] = []
                    continue
                df.at[idx, "author_hindices"] = fetch_author_hindices(authors, author_cache)

            save_author_cache(author_cache)
            print(f"  h-index enrichment complete. Cache now has {len(author_cache)} entries.")

        # Compute Prominence from whatever author_hindices is in the data
        # (runs in both normal and OPENALEX_OFFLINE_MODE)
        df["Prominence"] = df.apply(calculate_prominence, axis=1)
        tier_counts = df["Prominence"].value_counts()
        for tier in ["Elite", "Enhanced", "Emerging", "Unverified"]:
            print(f"  {tier}: {tier_counts.get(tier, 0)}")

        # ── Stage 1d: Semantic Scholar enrichment ────────────────────────────
        print("\n▶  Stage 1d — Semantic Scholar enrichment...")

        # Ensure columns exist (default values; populated below from cache)
        for col, default in [
            ("ss_citation_count",        0),
            ("ss_influential_citations", 0),
            ("ss_tldr",                  ""),
        ]:
            if col not in df.columns:
                df[col] = default

        ss_cache = load_ss_cache()

        # Fetch a paper if any of these are true:
        #   (a) not in cache at all
        #   (b) cache entry is older than SS_CACHE_TTL_DAYS (normal TTL)
        #   (c) cache entry has zero signal — citation_count=0, tldr="" —
        #       meaning S2 hadn't indexed it yet; retry every run until it
        #       appears (brand-new papers typically land in S2 within days)
        cutoff_ss = (
            datetime.now(timezone.utc) - timedelta(days=SS_CACHE_TTL_DAYS)
        ).isoformat()

        def _needs_ss_fetch(arxiv_id):
            """Return a reason string if the paper needs a fetch, else None."""
            base  = _arxiv_id_base(arxiv_id)
            entry = ss_cache.get(base)
            if entry is None:
                return "not cached"
            if entry.get("fetched_at", "") < cutoff_ss:
                return "stale (TTL expired)"
            no_signal = (
                int(entry.get("citation_count",             0)) == 0
                and int(entry.get("influential_citation_count", 0)) == 0
                and not (entry.get("tldr") or "").strip()
            )
            if no_signal:
                return "unindexed (no signal yet)"
            return None

        reason_map = {
            aid: _needs_ss_fetch(aid)
            for aid in df["id"].tolist()
        }
        arxiv_ids_to_fetch = [aid for aid, reason in reason_map.items()
                               if reason is not None]
        n_cached    = len(df) - len(arxiv_ids_to_fetch)
        n_unindexed = sum(1 for r in reason_map.values()
                          if r == "unindexed (no signal yet)")
        n_stale     = sum(1 for r in reason_map.values()
                          if r == "stale (TTL expired)")
        n_missing   = sum(1 for r in reason_map.values() if r == "not cached")
        print(f"  {len(arxiv_ids_to_fetch)} papers need S2 fetch "
              f"({n_cached} skipped: already have data within TTL). "
              f"Breakdown: {n_missing} missing, {n_stale} stale, "
              f"{n_unindexed} not yet indexed.")

        if arxiv_ids_to_fetch:
            fetch_semantic_scholar_data(arxiv_ids_to_fetch, ss_cache)
            save_ss_cache(ss_cache)
            print(f"  S2 cache now has {len(ss_cache)} entries.")
        else:
            print(f"  All papers found in S2 cache — no fetch needed.")

        # Apply cache values to DataFrame (covers all papers, not just re-fetched)
        for idx in df.index:
            base  = _arxiv_id_base(df.at[idx, "id"])
            entry = ss_cache.get(base)
            if entry:
                df.at[idx, "ss_citation_count"]        = int(entry.get("citation_count",             0))
                df.at[idx, "ss_influential_citations"]  = int(entry.get("influential_citation_count", 0))
                df.at[idx, "ss_tldr"]                   = entry.get("tldr", "") or ""

        n_with_tldr  = (df["ss_tldr"].astype(str).str.len() > 0).sum()
        n_with_cites = (df["ss_citation_count"] > 0).sum()
        print(f"  S2 enrichment complete: "
              f"{n_with_cites}/{len(df)} papers with citations, "
              f"{n_with_tldr}/{len(df)} with TLDRs.")

        # ── Stage 1e: CitationTier ────────────────────────────────────────────
        print("\n▶  Stage 1e -- Computing CitationTier...")
        df["CitationTier"] = calculate_citation_tier(df)

        # ── Stage 1f: Recency label ───────────────────────────────────────────
        # Categorical column for sidebar color-by shortcut.
        # "Today" / "Yesterday" / "Earlier" based on date_added.
        print("\n▶  Stage 1f -- Computing Recency...")
        _today_utc = datetime.now(timezone.utc).date()
        def _recency_label(date_added_val):
            try:
                d = pd.to_datetime(date_added_val, utc=True).date()
                delta = (_today_utc - d).days
                if delta == 0:  return "Today"
                if delta == 1:  return "Yesterday"
                return "Earlier"
            except Exception:
                return "Earlier"
        df["recency"] = df["date_added"].apply(_recency_label)
        recency_counts = df["recency"].value_counts()
        print(f"  Recency: Today={recency_counts.get('Today', 0)}, "
              f"Yesterday={recency_counts.get('Yesterday', 0)}, "
              f"Earlier={recency_counts.get('Earlier', 0)}.")

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

    # ── Write groups.json for LinkedIn post automation ────────────────────────
    # Sorted by group_id so the most-populated stable buckets come first.
    # Read by linkedin_post.js to generate the daily Haiku caption.
    # groups_json_path = "groups.json"
    # group_names_list = [group_names[gid] for gid in sorted(group_names.keys())]
    # with open(groups_json_path, "w") as _f:
    #     json.dump(group_names_list, _f, indent=2)
    # print(f"  Wrote {len(group_names_list)} group names → {groups_json_path}")

    # ── Save rolling DB ───────────────────────────────────────────────────────
    print("\n▶  Saving rolling database...")
    save_df = df.copy()
    save_df["date_added"] = pd.to_datetime(
        save_df["date_added"], utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if "projection_v3_x" not in save_df.columns:
        raise RuntimeError("projection_v3_x missing — layout step failed.")

    save_df.to_parquet(DB_PATH, index=False)
    print(f"  Saved {len(save_df)} papers to {DB_PATH}.")
    proj_cols = [c for c in save_df.columns if "projection" in c or "group_id" in c
                 or c == "secondary_tags"]
    print(f"  Projection / group / secondary columns: {proj_cols}")

    # ── Stage 5: Build + deploy ───────────────────────────────────────────────
    print("\n▶  Stage 5 — Building atlas...")
    build_and_deploy_atlas(
        db_path     = DB_PATH,
        proj_x_col  = PROJ_X_COL,
        proj_y_col  = PROJ_Y_COL,
        labels_path = labels_path,
        run_date    = run_date,
    )

    print("\n✓  update_map_v3.py complete.")
