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
from datetime import datetime, timedelta, timezone

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DB_PATH         = "database.parquet"
STOP_WORDS_PATH = "stop_words.csv"
RETENTION_DAYS  = 4       # papers older than this are pruned each run
ARXIV_MAX       = 250     # max papers fetched per arXiv query

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
# 3. REPUTATION SCORING
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

    # Institution match — strongest signal
    if INSTITUTION_PATTERN.search(full_text):
        score += 3

    # Public codebase — indicates reproducibility commitment
    if any(k in full_text for k in ["github.com", "huggingface.co"]):
        score += 2

    # Author count — larger teams tend to reflect institutional resources
    score += author_reputation_score(row["author_count"])

    return "Reputation Enhanced" if score >= 3 else "Reputation Std"


# ──────────────────────────────────────────────
# 4. ROLLING DATABASE
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
# 5. ARXIV FETCH WITH EXPONENTIAL BACKOFF
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
# 6. INCREMENTAL EMBEDDING
#    Embeds only papers missing vectors, then re-projects
#    ALL papers with UMAP for a globally coherent layout.
# ──────────────────────────────────────────────
def embed_and_project(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    import umap as umap_lib

    model = SentenceTransformer("allenai/specter2_base")

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
# 7. CLUSTER LABEL GENERATION
#
# CURRENT APPROACH: KeyBERT (option 1)
#   Uses semantic keyword extraction via SPECTER2 to find the most
#   representative 1-2 word phrase for each spatial cluster. Produces
#   meaningful labels like "federated learning" or "vision-language"
#   rather than raw TF-IDF frequency winners.
#
# ALTERNATIVES (if you want to change approach):
#
#   Option 2 — LLM-generated labels via Anthropic API (highest quality):
#     After clustering, send the top-N titles per cluster to claude-haiku
#     and ask for a 2-4 word descriptive label. Best results but adds
#     latency and API cost. Requires ANTHROPIC_API_KEY in repo secrets.
#     Rough implementation: collect cluster titles → call anthropic.Anthropic()
#     client → parse response → write labels parquet.
#
#   Option 3 — Bigrams in label_text (lowest effort):
#     Pre-process label_text to append bigrams joined with underscores
#     (e.g. "reinforcement_learning", "graph_neural") before the atlas
#     build. TF-IDF then has more distinctive tokens to rank. No new
#     dependencies. Add to the label_text construction in the row builder:
#       import nltk; nltk.download("punkt")
#       from nltk import ngrams, word_tokenize
#       words = word_tokenize(title.lower())
#       bigrams = ["_".join(b) for b in ngrams(words, 2)]
#       label_text = title + " " + " ".join(bigrams)
#
#   Option 4 — Pre-computed labels file from external tool:
#     Generate a CSV/parquet with columns x, y, text (optional: level,
#     priority) using any method, then pass via --labels to the CLI.
#     This is the escape hatch if none of the above satisfy.
# ──────────────────────────────────────────────
def generate_keybert_labels(df: pd.DataFrame) -> str:
    """
    Cluster the 2D projection with HDBSCAN, then use KeyBERT + SPECTER2
    to extract the best 1-2 word phrase for each cluster. Writes a
    labels.parquet file and returns its path for the --labels CLI flag.
    """
    from keybert import KeyBERT
    from sklearn.cluster import HDBSCAN

    print("  Generating KeyBERT cluster labels...")

    # 2D coords used only for computing label centroid positions for display.
    coords_2d = df[["projection_x", "projection_y"]].values.astype(np.float64)

    # ── Clustering input: 50D cosine space ──────────────────────────────
    # CURRENT APPROACH: cluster on the 50D UMAP projection (stored in
    # embedding_50d) using cosine metric. This is far more semantically
    # accurate than clustering on 2D display coords, which lose structure
    # during the final compression step.
    #
    # ALTERNATIVE A — cluster on raw 768D SPECTER2 vectors:
    #   More faithful to the original embedding space but slower and
    #   HDBSCAN struggles with very high dimensions (curse of dimensionality).
    #   Use 50D as a middle ground (current approach) unless you have
    #   a strong reason to use the full vectors.
    #   To enable: replace the embedding_50d block below with:
    #     all_vectors = np.array(df["embedding"].tolist(), dtype=np.float32)
    #     cluster_input = all_vectors
    #     cluster_metric = "cosine"
    #
    # ALTERNATIVE B — cluster on 2D display coords (original approach):
    #   Simple but lossy. Fine for small corpora but misses structure
    #   compressed away during the 768D → 2D projection.
    #   To enable: replace the block below with:
    #     cluster_input = coords_2d
    #     cluster_metric = "euclidean"
    if "embedding_50d" in df.columns and df["embedding_50d"].notna().all():
        cluster_input = np.array(df["embedding_50d"].tolist(), dtype=np.float32)
        cluster_metric = "cosine"
        print("  Clustering on 50D cosine space (high-fidelity).")
    else:
        # Fallback for runs where 50D projection isn't yet stored.
        cluster_input = coords_2d
        cluster_metric = "euclidean"
        print("  Clustering on 2D coords (fallback — embedding_50d not available).")

    # ── HDBSCAN clustering settings ─────────────────────────────────────
    # min_cluster_size: minimum papers for a group to become a cluster.
    #   Higher → fewer, larger, more general clusters (fewer labels).
    #   Lower  → more, smaller, more specific clusters (more labels).
    #   Recommended range: 3–15. Default: 5.
    #
    # min_samples: how conservative cluster assignment is.
    #   Lower  → more points assigned to clusters (less noise/unlabelled).
    #   Higher → stricter, more points treated as noise.
    #   Recommended range: 1–5. Default: 3.
    #
    # cluster_selection_method:
    #   "eom"  (default) — Excess of Mass; finds clusters of varying sizes.
    #   "leaf" — selects leaf nodes of the condensed tree; smaller, more
    #            granular clusters. Try this if clusters feel too coarse.
    #   To enable leaf mode: add cluster_selection_method="leaf" below.
    #
    # Adaptive min_cluster_size (scales with corpus size):
    #   Rather than a fixed value, scale to corpus size so the number of
    #   clusters stays roughly proportional as the rolling DB grows/shrinks.
    #   To enable: replace min_cluster_size=5 with:
    #     min_cluster_size=max(5, len(df) // 40)
    #
    # Soft clustering (reduces unlabelled noise points):
    #   HDBSCAN supports fuzzy cluster membership — points near boundaries
    #   get fractional membership rather than being forced into one cluster
    #   or marked as noise. Requires prediction_data=True and a separate
    #   soft assignment step via hdbscan.all_points_membership_vectors().
    #   Note: soft clustering is not currently implemented here; enable
    #   prediction_data=True below as a first step if you want to explore it.
    # ════════════════════════════════════════════════════════════════════
    # HDBSCAN SETTINGS — all changes take effect on the next build.
    # These settings do not affect database.parquet in any way.
    # ════════════════════════════════════════════════════════════════════
    #
    # ── min_cluster_size (int, default: 5) ──────────────────────────────
    # Minimum number of papers required to form a cluster.
    # Lower  → more clusters, more specific labels.
    # Higher → fewer clusters, more general labels.
    # Syntax:   HDBSCAN(min_cluster_size=5, ...)
    # Range:    3–15 recommended for a ~300 paper corpus.
    #
    # ── min_samples (int, default: 4) ───────────────────────────────────
    # Controls how conservative cluster assignment is.
    # Lower  → more points pulled into clusters, fewer noise points.
    # Higher → stricter assignment, more points marked as noise.
    # Syntax:   HDBSCAN(..., min_samples=4, ...)
    # Range:    1–5 recommended.
    #
    # ── cluster_selection_method (str, default: "eom") ──────────────────
    # "eom"  — Excess of Mass; finds clusters of varying sizes (default).
    # "leaf" — Selects leaf nodes; smaller, more granular clusters.
    #          Try "leaf" if clusters feel too coarse.
    # Syntax:   HDBSCAN(..., cluster_selection_method="leaf")
    #
    # ── metric (str, set automatically above) ───────────────────────────
    # "cosine"    — used when clustering on 50D embeddings (current).
    # "euclidean" — used as fallback when clustering on 2D coords.
    # Do not change this directly; it is set by the cluster_metric variable.
    #
    # ── Adaptive min_cluster_size (scales with corpus size) ─────────────
    # Keeps cluster count roughly proportional as the rolling DB changes.
    # Syntax:   HDBSCAN(min_cluster_size=max(5, len(df) // 40), ...)
    #
    # ── cluster_selection_epsilon (float, default: 0.0) ─────────────────
    # Merges clusters closer than this distance threshold. Useful if you
    # see many tiny clusters that should logically be one topic.
    # Syntax:   HDBSCAN(..., cluster_selection_epsilon=0.5)
    # Range:    0.0 (off) to ~2.0; tune by inspecting cluster count.
    #
    # ── alpha (float, default: 1.0) ─────────────────────────────────────
    # Controls how aggressively clusters are split during extraction.
    # Higher → fewer, more stable clusters.
    # Lower  → more splits, more clusters.
    # Syntax:   HDBSCAN(..., alpha=1.0)
    # Range:    0.5–2.0 typical.
    # ════════════════════════════════════════════════════════════════════
    clusterer = HDBSCAN(min_cluster_size=max(5, len(df) // 40), min_samples=4, metric=cluster_metric, cluster_selection_method="leaf")
    cluster_ids = clusterer.fit_predict(cluster_input)

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"  Found {n_clusters} clusters (noise points excluded).")

    # KeyBERT with SPECTER2 — reuses the cached HuggingFace download.
    kw_model = KeyBERT(model="allenai/specter2_base")

    label_rows = []
    for cid in sorted(set(cluster_ids)):
        if cid == -1:
            continue  # HDBSCAN noise points get no label
        mask = cluster_ids == cid
        # Use label_text (scrubbed, title-only) rather than raw title so
        # "model/models" variants are already stripped before KeyBERT sees them.
        titles = df.loc[mask, "label_text"].tolist()
        combined = " ".join(titles)

        # Extend scikit-learn English stop words with AI boilerplate that
        # KeyBERT would otherwise rank highly across all clusters.
        # Note: stop_words="english" passes a string to CountVectorizer which
        # uses sklearn's list internally — we extend that list explicitly here.
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        KEYBERT_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
            "model", "models", "modeling", "modeled",
            "paper", "propose", "proposed", "approach",
            "method", "methods", "task", "tasks", "performance", "results",
            "result", "work", "framework", "system", "learning", "deep",
            "based", "using", "show", "new", "novel", "training", "dataset",
            "data", "benchmark", "improve", "improved", "state", "art",
            "effective", "efficient", "robust", "demonstrate", "achieve",
        ]

        # ── KeyBERT keyword extraction settings ─────────────────────────
        # keyphrase_ngram_range: (min, max) words per keyword phrase.
        #   (1, 1) → single words only e.g. "robotics", "safety" — more general.
        #   (1, 2) → single words and bigrams e.g. "medical imaging" — more specific.
        #   (2, 2) → bigrams only — specific but can produce odd pairings.
        #
        # use_mmr: Maximal Marginal Relevance reduces redundancy between keywords.
        #   True  → picks broader, more diverse terms across the cluster.
        #   False → picks the highest-scoring terms regardless of similarity.
        #
        # diversity: only applies when use_mmr=True. Range 0.0–1.0.
        #   Lower  → keywords closer to cluster centroid (more representative).
        #   Higher → keywords more spread out (more diverse but less focused).
        #
        # top_n: how many candidate keywords to extract (kept for debug printing).
        keywords = kw_model.extract_keywords(
            combined,
            keyphrase_ngram_range=(2, 2),
            stop_words=KEYBERT_STOP_WORDS,
            use_mmr=True,
            diversity=0.3,
            top_n=3,
        )
        print(f"  Cluster {cid} ({mask.sum()} papers) top keywords: {keywords}")
        print(f"  Sample input (first 200 chars): {combined[:200]}")
        if not keywords:
            continue

        label_text = keywords[0][0].title()
        cx = float(coords_2d[mask, 0].mean())
        cy = float(coords_2d[mask, 1].mean())
        label_rows.append({"x": cx, "y": cy, "text": label_text})

    labels_df = pd.DataFrame(label_rows)
    labels_path = "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    print(f"  Wrote {len(labels_df)} labels to {labels_path}.")
    return labels_path


# ──────────────────────────────────────────────
# 8. HTML POP-OUT PANELS  (About + Shortcuts)
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

  /* ── Tab strip ─────────────────────────────────────────────────────── */
  #arm-tabs {
    position:fixed; left:0; top:50%; transform:translateY(-50%);
    z-index:1000000; display:flex; flex-direction:column; gap:4px;
  }
  .arm-tab-btn {
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
  .arm-tab-btn:first-child { border-radius:0 10px 0 0; }
  .arm-tab-btn:last-child  { border-radius:0 0 10px 0; }
  .arm-tab-btn:only-child  { border-radius:0 10px 10px 0; }
  .arm-tab-btn:hover { background:rgba(30,41,59,0.95); color:#93c5fd; box-shadow:4px 0 24px rgba(96,165,250,0.2); }
  .arm-tab-btn.arm-tab-active { background:rgba(30,41,59,0.98); color:#93c5fd; }

  /* ── Shared panel chrome ────────────────────────────────────────────── */
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
  .arm-panel-footer { padding:10px 20px 14px; border-top:1px solid var(--arm-border); display:flex; align-items:center; gap:7px; flex-shrink:0; }

  /* ── Shared typography ──────────────────────────────────────────────── */
  .arm-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px; }
  .arm-title { font-size:15px; font-weight:700; color:#f1f5f9; letter-spacing:-0.01em; line-height:1.3; margin:0; }
  .arm-title span { color:var(--arm-accent); }
  .arm-close-btn { background:none; border:none; color:var(--arm-muted); cursor:pointer; font-size:17px; line-height:1; padding:2px 4px; border-radius:4px; transition:color 0.15s,background 0.15s; flex-shrink:0; margin-left:8px; }
  .arm-close-btn:hover { color:#f1f5f9; background:rgba(255,255,255,0.07); }
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
  /* ── Book cards ─────────────────────────────────────────────────────── */
  .arm-book { display:flex; align-items:center; gap:10px; background:rgba(255,255,255,0.03); border:1px solid var(--arm-border); border-radius:8px; padding:10px 12px; text-decoration:none; transition:background 0.2s,border-color 0.2s; margin-bottom:8px; }
  .arm-book:hover { background:rgba(96,165,250,0.07); border-color:rgba(96,165,250,0.3); }
  .arm-book-icon { font-size:22px; flex-shrink:0; }
  .arm-book-text { display:flex; flex-direction:column; }
  .arm-book-title { font-size:12px; font-weight:600; color:#e2e8f0; line-height:1.3; margin-bottom:2px; }
  .arm-book-sub { font-size:11px; color:var(--arm-accent); }

  /* ── Status footer ───────────────────────────────────────────────────── */
  .arm-status-dot { width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 6px rgba(34,197,94,0.7); flex-shrink:0; animation:arm-pulse 2.5s ease-in-out infinite; }
  @keyframes arm-pulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }
  .arm-status-text { font-size:11px; color:#475569; font-family:var(--arm-font); }
  .arm-status-text strong { color:#64748b; font-weight:500; }

  /* ── Shortcuts panel ─────────────────────────────────────────────────── */
  .arm-shortcut-row {
    display:grid; grid-template-columns:auto 1fr; align-items:center;
    gap:10px 14px; margin-bottom:11px;
  }
  .arm-key-group { display:flex; align-items:center; gap:5px; flex-shrink:0; }
  .arm-key {
    display:inline-flex; align-items:center; justify-content:center;
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15);
    border-bottom:2px solid rgba(0,0,0,0.35); border-radius:5px;
    padding:3px 7px; font-size:11px; font-weight:600; font-family:var(--arm-font);
    color:#cbd5e1; white-space:nowrap; line-height:1.4;
    box-shadow:0 1px 0 rgba(0,0,0,0.3);
  }
  .arm-key-sep { color:#475569; font-size:10px; }
  .arm-shortcut-desc { font-size:12px; color:#94a3b8; line-height:1.45; }
  .arm-shortcut-desc strong { color:#cbd5e1; font-weight:600; }
  .arm-shortcut-icon { font-size:15px; line-height:1; }
</style>

<script>
  function armToggle(panelId, tabId) {
    var allPanels = document.querySelectorAll('.arm-panel');
    var allTabs   = document.querySelectorAll('.arm-tab-btn');
    var panel = document.getElementById(panelId);
    var isOpen = panel.classList.contains('arm-open');
    // Close everything first
    allPanels.forEach(function(p) { p.classList.remove('arm-open'); });
    allTabs.forEach(function(t)   { t.classList.remove('arm-tab-active'); });
    // Toggle the clicked panel unless it was already open
    if (!isOpen) {
      panel.classList.add('arm-open');
      document.getElementById(tabId).classList.add('arm-tab-active');
    }
  }
  function armClose(panelId, tabId) {
    document.getElementById(panelId).classList.remove('arm-open');
    document.getElementById(tabId).classList.remove('arm-tab-active');
  }
</script>

<!-- ── Tab strip ───────────────────────────────────────────────── -->
<div id="arm-tabs">
  <button id="arm-tab-about"     class="arm-tab-btn" onclick="armToggle('arm-panel-about','arm-tab-about')"     aria-label="Open About panel">About</button>
  <button id="arm-tab-shortcuts" class="arm-tab-btn" onclick="armToggle('arm-panel-shortcuts','arm-tab-shortcuts')" aria-label="Open Shortcuts panel">Keys</button>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     ABOUT PANEL
═══════════════════════════════════════════════════════════════ -->
<div id="arm-panel-about" class="arm-panel" role="complementary" aria-label="About this atlas">
  <div class="arm-body">

    <div class="arm-header">
      <p class="arm-title">The <span>AI Research</span> Atlas</p>
      <button class="arm-close-btn" onclick="armClose('arm-panel-about','arm-tab-about')" aria-label="Close panel">&#x2715;</button>
    </div>
    <div class="arm-byline">By <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" rel="noopener">Lee Fischman</a></div>

    <p class="arm-p">A live semantic map of recent AI research from arXiv (cs.AI), rebuilt every day at 14:00 UTC. Each point is a paper. The map shows a rolling 4-day window &mdash; up to 250 papers per day.</p>

    <hr class="arm-divider">

    <p class="arm-section">How the map is organized</p>
    <p class="arm-p">Papers are embedded with <a href="https://allenai.org/blog/specter" target="_blank" rel="noopener">SPECTER2</a>, a model trained on scientific citation graphs. <strong>Nearby points are papers a researcher in that subfield would read together</strong> &mdash; they cluster by intellectual proximity, not just surface keywords. A paper on RL safety and one on multi-agent coordination may sit close even if they share few words.</p>
    <p class="arm-p">Cluster labels are extracted with KeyBERT, finding the most distinctive 2-word phrase in each group's titles.</p>

    <hr class="arm-divider">

    <p class="arm-section">Color dimensions</p>
    <div class="arm-legend-row">
      <div class="arm-dot arm-dot-enhanced"></div>
      <div><span class="arm-legend-label">Reputation Enhanced</span><br>Paper is from a recognized institution, lab, or has a public codebase.</div>
    </div>
    <div class="arm-legend-row">
      <div class="arm-dot arm-dot-std"></div>
      <div><span class="arm-legend-label">Reputation Std</span><br>No strong institutional signal detected in the abstract text.</div>
    </div>
    <p class="arm-p" style="margin-top:8px;">Switch the <strong>Color by</strong> dropdown to <strong>author_tier</strong> to see team size (1–3, 4–7, 8+ authors), which loosely correlates with resource backing.</p>

    <hr class="arm-divider">

    <p class="arm-section">Powered by</p>
    <p class="arm-p"><a href="https://apple.github.io/embedding-atlas/" target="_blank" rel="noopener">Apple Embedding Atlas</a> &bull; SPECTER2 &bull; UMAP &bull; HDBSCAN &bull; KeyBERT &bull; arXiv API</p>

    <hr class="arm-divider">

    <p class="arm-section">Books by the author</p>
    <a class="arm-book" href="https://www.amazon.com/dp/B0GMVH6P2W" target="_blank" rel="noopener">
      <span class="arm-book-icon">&#x1F4D8;</span>
      <span class="arm-book-text">
        <span class="arm-book-title">Building Deep Learning Products</span>
        <span class="arm-book-sub">Available on Amazon &#x2192;</span>
      </span>
    </a>

  </div>
  <div class="arm-panel-footer">
    <div class="arm-status-dot"></div>
    <span class="arm-status-text">Last updated <strong>""" + run_date + """ UTC</strong></span>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════
     SHORTCUTS PANEL
═══════════════════════════════════════════════════════════════ -->
<div id="arm-panel-shortcuts" class="arm-panel" role="complementary" aria-label="Keyboard and mouse shortcuts">
  <div class="arm-body">

    <div class="arm-header">
      <p class="arm-title"><span>Shortcuts</span> &amp; Controls</p>
      <button class="arm-close-btn" onclick="armClose('arm-panel-shortcuts','arm-tab-shortcuts')" aria-label="Close panel">&#x2715;</button>
    </div>

    <p class="arm-section" style="margin-top:6px;">Navigation</p>

    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-shortcut-icon">🖱️</span><span class="arm-key">Drag</span></div>
      <div class="arm-shortcut-desc"><strong>Pan</strong> the map</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-shortcut-icon">🖱️</span><span class="arm-key">Scroll</span></div>
      <div class="arm-shortcut-desc"><strong>Zoom</strong> in / out</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-shortcut-icon">🤌</span><span class="arm-key">Pinch</span></div>
      <div class="arm-shortcut-desc"><strong>Zoom</strong> on touch / trackpad</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Dbl&nbsp;click</span></div>
      <div class="arm-shortcut-desc"><strong>Reset</strong> zoom &amp; pan to fit</div>
    </div>

    <hr class="arm-divider">
    <p class="arm-section">Selecting papers</p>

    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-shortcut-icon">🖱️</span><span class="arm-key">Click</span></div>
      <div class="arm-shortcut-desc"><strong>Select</strong> a paper — shows title, abstract, and PDF link</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Click</span><span class="arm-key-sep">+</span><span class="arm-key">empty</span></div>
      <div class="arm-shortcut-desc"><strong>Deselect</strong> — click any blank area</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Esc</span></div>
      <div class="arm-shortcut-desc"><strong>Dismiss</strong> the detail panel</div>
    </div>

    <hr class="arm-divider">
    <p class="arm-section">Search &amp; filter</p>

    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-shortcut-icon">🔍</span><span class="arm-key">Search&nbsp;bar</span></div>
      <div class="arm-shortcut-desc"><strong>Filter by keyword</strong> — matches titles and abstracts; non-matching points dim</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Enter</span></div>
      <div class="arm-shortcut-desc"><strong>Commit search</strong> &amp; highlight matches</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Esc</span></div>
      <div class="arm-shortcut-desc"><strong>Clear search</strong> and restore all points</div>
    </div>

    <hr class="arm-divider">
    <p class="arm-section">Color &amp; labels</p>

    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Color&nbsp;by</span></div>
      <div class="arm-shortcut-desc">Switch between <strong>Reputation</strong>, <strong>author_tier</strong>, or other columns to recolor the map</div>
    </div>
    <div class="arm-shortcut-row">
      <div class="arm-key-group"><span class="arm-key">Labels</span></div>
      <div class="arm-shortcut-desc">Floating cluster labels appear automatically; zoom in to reveal more granular ones</div>
    </div>

  </div>
  <div class="arm-panel-footer">
    <div class="arm-status-dot"></div>
    <span class="arm-status-text">Tip: open <strong>About</strong> for color legend details</span>
  </div>
</div>
"""
    )


# ──────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    clear_docs_contents("docs")

    now      = datetime.now(timezone.utc)
    run_date = now.strftime("%B %d, %Y")

    # Load & prune existing rolling DB
    existing_df  = load_existing_db()
    is_first_run = existing_df.empty and not os.path.exists(DB_PATH)
    days_back    = 5 if is_first_run else 2

    if is_first_run:
        print("  First run — pre-filling with last 5 days of arXiv papers.")
    else:
        print(f"  Loaded {len(existing_df)} existing papers from rolling DB.")

    # Fetch from arXiv
        # Set a descriptive User-Agent as required by arXiv's API policy.
    # Without this, new clients may receive HTTP 406 rejections.
    import urllib.request
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
            "date_added":   today_str,
        })

    new_df = pd.DataFrame(rows)
    new_df["Reputation"] = new_df.apply(calculate_reputation, axis=1)

    # Merge into rolling DB
    df = merge_papers(existing_df, new_df)
    df = df.drop(columns=["group"], errors="ignore")  # remove legacy column name
    print(f"  Rolling DB: {len(df)} papers after merge.")

    # Backfill Reputation for older rows that predate the column.
    if "Reputation" not in df.columns or df["Reputation"].isna().any():
        missing = df["Reputation"].isna() if "Reputation" in df.columns else pd.Series([True] * len(df))
        if missing.any():
            print(f"  Backfilling Reputation for {missing.sum()} older rows...")
            df.loc[missing, "Reputation"] = df.loc[missing].apply(calculate_reputation, axis=1)

    # Embed & project (incremental mode only)
    labels_path = None
    if EMBEDDING_MODE == "incremental":
        print("  Incremental mode: embedding new papers only.")
        df = embed_and_project(df)
        labels_path = generate_keybert_labels(df)

    # Save rolling DB
    # Full mode: drop projection columns so CLI always recomputes a fresh layout.
    if EMBEDDING_MODE == "full":
        save_df = df.drop(columns=["embedding", "projection_x", "projection_y"], errors="ignore")
    else:
        save_df = df

    # Normalize date_added to a consistent ISO string.
    # load_existing_db() converts it to datetime; new rows arrive as strings.
    # After merge the column is mixed-type, which pyarrow refuses to serialize.
    save_df = save_df.copy()
    save_df["date_added"] = pd.to_datetime(save_df["date_added"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    save_df.to_parquet(DB_PATH, index=False)
    print(f"  Saved {len(save_df)} papers to {DB_PATH}.")

    # Build the atlas
    print(f"  Building atlas ({EMBEDDING_MODE} mode)...")

    if EMBEDDING_MODE == "incremental":
        # Incremental mode: embeddings are pre-computed via --x/--y.
        # KeyBERT labels are passed via --labels, bypassing TF-IDF entirely.
        # --text is intentionally omitted so the CLI cannot generate competing
        # automatic TF-IDF labels that would override our KeyBERT labels.
        atlas_cmd = [
            "embedding-atlas", DB_PATH,
            "--x",          "projection_x",
            "--y",          "projection_y",
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
            "--model",      "allenai/specter2_base",
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

        conf.setdefault("column_mappings", {}).update({
            "title":        "title",
            "abstract":     "abstract",
            "Reputation":   "Reputation",
            "author_count": "author_count",
            "author_tier":  "author_tier",
            "url":          "url",
        })

        with open(config_path, "w") as f:
            json.dump(conf, f, indent=4)
        print("  Config updated.")
    else:
        print("  docs/data/config.json not found — skipping config override.")

    # Inject About + Shortcuts pop-out panels
    index_file = "docs/index.html"
    if os.path.exists(index_file):
        panels_html = build_panel_html(run_date)
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("</body>", panels_html + "\n</body>") \
            if "</body>" in content else content + panels_html
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("  About + Shortcuts panels injected into index.html.")
    else:
        print("  docs/index.html not found — skipping panel injection.")

    print("  Atlas sync complete!")
