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

# Embedding model is controlled by the EMBEDDING_MODEL env var.
# "specter2" — allenai/specter2_base. Groups by citation-graph proximity.
#              Columns: embedding, embedding_50d, projection_x, projection_y.
# "sbert"    — all-mpnet-base-v2. Groups by surface semantic similarity.
#              Columns: embedding_sbert, embedding_sbert_50d,
#                       projection_sbert_x, projection_sbert_y.
# "hybrid"   — Blends SPECTER2 + SBERT + TF-IDF distance matrices, then
#              runs UMAP on the blended matrix. Both model's vectors are
#              embedded incrementally and stored separately. Projection
#              written to projection_hybrid_x, projection_hybrid_y.
#              Weights controlled by HYBRID_WEIGHT_* env vars below.
# All column sets coexist in database.parquet — switching is a one-line
# env change with no re-embedding required.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "specter2").strip().lower()
if EMBEDDING_MODEL not in ("specter2", "sbert", "hybrid"):
    raise ValueError(f"EMBEDDING_MODEL must be 'specter2', 'sbert', or 'hybrid', got: '{EMBEDDING_MODEL}'")

# ── Hybrid blend weights ──────────────────────────────────────────────────
# Only used when EMBEDDING_MODEL=hybrid. Weights are auto-normalised so they
# don't need to sum to 1.0. Set HYBRID_WEIGHT_SBERT=0.0 to exclude SBERT.
#
# HYBRID_WEIGHT_SPECTER2 — citation-graph proximity. Groups by who cites whom.
# HYBRID_WEIGHT_SBERT    — surface semantic similarity. Groups by topic.
#                          Set > 0.0 to enable; requires SBERT vectors (already
#                          in the parquet from the earlier SBERT build run).
# HYBRID_WEIGHT_TFIDF    — term frequency overlap. Grounds clusters in shared
#                          vocabulary. Cheap: computed fresh each run.
HYBRID_WEIGHT_SPECTER2 = 0.33
HYBRID_WEIGHT_SBERT    = 0.33
HYBRID_WEIGHT_TFIDF    = 0.33

print(f"▶  Embedding mode  : {EMBEDDING_MODE.upper()}")
print(f"▶  Embedding model : {EMBEDDING_MODEL.upper()}")
if EMBEDDING_MODEL == "hybrid":
    total = HYBRID_WEIGHT_SPECTER2 + HYBRID_WEIGHT_SBERT + HYBRID_WEIGHT_TFIDF
    print(f"▶  Hybrid weights  : SPECTER2={HYBRID_WEIGHT_SPECTER2/total:.2f}  "
          f"SBERT={HYBRID_WEIGHT_SBERT/total:.2f}  "
          f"TF-IDF={HYBRID_WEIGHT_TFIDF/total:.2f}  (normalised)")


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
#
#    Supports two embedding models controlled by EMBEDDING_MODEL env var:
#      "specter2" — allenai/specter2_base (citation-graph proximity)
#      "sbert"    — all-mpnet-base-v2 (surface semantic similarity)
#    Each model writes to its own column set and coexists in the parquet.
# ──────────────────────────────────────────────
def embed_and_project(df: pd.DataFrame, model_name: str = "specter2") -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    import umap as umap_lib

    # ── Column names per model ───────────────────────────────────────────
    # Each model has its own isolated column set so both can coexist in the
    # parquet without overwriting each other. Switching models is a one-line
    # env change — the other model's vectors remain intact.
    if model_name == "specter2":
        hf_model_id   = "allenai/specter2_base"
        col_embed      = "embedding"           # raw 768D vectors
        col_embed_50d  = "embedding_50d"       # UMAP 50D for clustering
        col_proj_x     = "projection_x"        # UMAP 2D display x
        col_proj_y     = "projection_y"        # UMAP 2D display y
        text_col       = "text"                # scrubbed title+abstract
    elif model_name == "sbert":
        hf_model_id   = "sentence-transformers/all-mpnet-base-v2"
        col_embed      = "embedding_sbert"
        col_embed_50d  = "embedding_sbert_50d"
        col_proj_x     = "projection_sbert_x"
        col_proj_y     = "projection_sbert_y"
        text_col       = "abstract"            # SBERT works well on raw abstracts
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

    print(f"  Loading embedding model: {hf_model_id}")
    model = SentenceTransformer(hf_model_id)

    # ── Incremental embedding ────────────────────────────────────────────
    # Only embed papers that don't yet have a vector for this model.
    # Papers embedded by the other model are left untouched.
    if col_embed in df.columns:
        needs_embed = df[col_embed].isna()
    else:
        df[col_embed] = None
        needs_embed = pd.Series([True] * len(df))

    n_new = needs_embed.sum()
    if n_new:
        print(f"  Embedding {n_new} new paper(s) with {model_name.upper()}...")
        idx     = df.index[needs_embed].tolist()
        texts   = df.loc[idx, text_col].tolist()
        vectors = model.encode(texts, show_progress_bar=True, batch_size=16,
                               convert_to_numpy=True)
        for i, pos in enumerate(idx):
            df.at[pos, col_embed] = vectors[i].tolist()
    else:
        print(f"  All papers already embedded with {model_name.upper()} — skipping.")

    all_vectors = np.array(df[col_embed].tolist(), dtype=np.float32)
    n = len(all_vectors)
    print(f"  Projecting {n} papers with UMAP (two-stage, {model_name.upper()})...")

    # ── Stage 1: high-D → 50D (for clustering) ──────────────────────────
    # Preserves far more structure than going straight to 2D.
    # HDBSCAN clusters in this richer space using cosine metric.
    reducer_50d = umap_lib.UMAP(n_components=50, metric="cosine",
                                random_state=42, n_neighbors=15)
    coords_50d = reducer_50d.fit_transform(all_vectors)
    df[col_embed_50d] = [row.tolist() for row in coords_50d]

    # ── Stage 2: high-D → 2D (for display only) ─────────────────────────
    # min_dist controls point spread: lower = tighter clusters visually.
    reducer_2d = umap_lib.UMAP(n_components=2, metric="cosine",
                               random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer_2d.fit_transform(all_vectors)
    df[col_proj_x] = coords_2d[:, 0].astype(float)
    df[col_proj_y] = coords_2d[:, 1].astype(float)
    return df


# ──────────────────────────────────────────────
# 6b. HYBRID DISTANCE MATRIX + PROJECTION
#     Blends SPECTER2, SBERT, and TF-IDF cosine distances into a single
#     weighted distance matrix, then runs UMAP on it to produce a layout
#     that reflects all three signals simultaneously.
# ──────────────────────────────────────────────
def compute_hybrid_distances(
    df: pd.DataFrame,
    w_specter2: float = 0.75,
    w_sbert: float    = 0.0,
    w_tfidf: float    = 0.25,
) -> np.ndarray:
    """
    Build a normalised hybrid distance matrix from up to three sources:

      SPECTER2  — cosine distances between 768D SPECTER2 vectors.
                  Captures citation-graph proximity.
      SBERT     — cosine distances between 768D SBERT vectors.
                  Captures surface semantic similarity.
      TF-IDF    — cosine distances between TF-IDF abstract vectors.
                  Captures shared vocabulary / terminology.

    Each source is independently normalised to [0, 1] before blending so
    that no single source dominates due to scale differences. Weights are
    also normalised to sum to 1.0 so the output is always in [0, 1].

    Returns:
        dist_matrix (np.ndarray, shape [n, n], float32): symmetric distance
        matrix suitable for UMAP(metric="precomputed") and
        HDBSCAN(metric="precomputed").
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

    def _normalise_dist(d: np.ndarray) -> np.ndarray:
        """Scale a distance matrix to [0, 1] by its max value."""
        max_val = d.max()
        return (d / max_val).astype(np.float32) if max_val > 0 else d.astype(np.float32)

    if w_specter2 > 0:
        if "embedding" not in df.columns or df["embedding"].isna().any():
            raise RuntimeError(
                "SPECTER2 vectors missing — run embed_and_project(model_name='specter2') first."
            )
        vecs = np.array(df["embedding"].tolist(), dtype=np.float32)
        vecs = sk_normalize(vecs)          # unit-normalise for cosine
        d = cosine_distances(vecs).astype(np.float32)
        dist += w_specter2 * _normalise_dist(d)
        print(f"  Hybrid: added SPECTER2 component (w={w_specter2:.2f}).")

    if w_sbert > 0:
        if "embedding_sbert" not in df.columns or df["embedding_sbert"].isna().any():
            raise RuntimeError(
                "SBERT vectors missing — run embed_and_project(model_name='sbert') first."
            )
        vecs = np.array(df["embedding_sbert"].tolist(), dtype=np.float32)
        vecs = sk_normalize(vecs)
        d = cosine_distances(vecs).astype(np.float32)
        dist += w_sbert * _normalise_dist(d)
        print(f"  Hybrid: added SBERT component (w={w_sbert:.2f}).")

    if w_tfidf > 0:
        # TF-IDF on cleaned abstracts — strips URLs/citation keys first.
        texts = df["abstract"].apply(_strip_urls).tolist()
        tfidf = TfidfVectorizer(
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
            ngram_range=(1, 2),
        )
        tfidf_matrix = tfidf.fit_transform(texts)
        d = cosine_distances(tfidf_matrix).astype(np.float32)
        dist += w_tfidf * _normalise_dist(d)
        print(f"  Hybrid: added TF-IDF component (w={w_tfidf:.2f}).")

    # Ensure exact symmetry and zero diagonal (floating point can drift).
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    print(f"  Hybrid distance matrix: shape={dist.shape}, "
          f"min={dist.min():.4f}, max={dist.max():.4f}, mean={dist.mean():.4f}.")
    return dist


def embed_and_project_hybrid(
    df: pd.DataFrame,
    w_specter2: float = 0.75,
    w_sbert: float    = 0.0,
    w_tfidf: float    = 0.25,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Ensure both SPECTER2 and SBERT vectors exist (embedding incrementally
    as needed), then build the hybrid distance matrix and project it to 2D
    with UMAP(metric="precomputed"). Writes projection_hybrid_x/y to df.

    Returns:
        df            — updated dataframe with projection_hybrid_x/y columns.
        dist_matrix   — the full hybrid distance matrix (n × n), returned
                        so generate_cluster_labels can reuse it without
                        recomputing.
    """
    import umap as umap_lib

    # Embed with each model that has a non-zero weight.
    # _embed_only handles incremental logic — only new papers get embedded.
    if w_specter2 > 0:
        print("  Hybrid: ensuring SPECTER2 vectors...")
        df = _embed_only(df, model_name="specter2")
    if w_sbert > 0:
        print("  Hybrid: ensuring SBERT vectors...")
        df = _embed_only(df, model_name="sbert")

    dist = compute_hybrid_distances(df, w_specter2=w_specter2,
                                    w_sbert=w_sbert, w_tfidf=w_tfidf)

    n = len(df)
    print(f"  Projecting {n} papers with UMAP on hybrid distance matrix...")
    reducer = umap_lib.UMAP(
        n_components=2,
        metric="precomputed",
        random_state=42,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
    )
    coords_2d = reducer.fit_transform(dist)
    df["projection_hybrid_x"] = coords_2d[:, 0].astype(float)
    df["projection_hybrid_y"] = coords_2d[:, 1].astype(float)
    print("  Hybrid projection written to projection_hybrid_x / projection_hybrid_y.")
    return df, dist


def _embed_only(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Embed papers missing vectors for the given model without running UMAP.
    Used by hybrid mode to populate vector columns before distance blending.
    UMAP for individual models (projection_x/y, projection_sbert_x/y) is
    NOT run in hybrid mode — only the hybrid projection is computed.
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
        model   = SentenceTransformer(hf_model_id)
        idx     = df.index[needs_embed].tolist()
        texts   = df.loc[idx, text_col].tolist()
        vectors = model.encode(texts, show_progress_bar=True, batch_size=16,
                               convert_to_numpy=True)
        for i, pos in enumerate(idx):
            df.at[pos, col_embed] = vectors[i].tolist()
    else:
        print(f"  All papers already embedded with {model_name.upper()} — skipping.")
    return df
#
# CURRENT APPROACH: Claude Haiku (Option 2)
#   Sends up to 8 abstracts per cluster to claude-haiku-4-5 and asks for
#   a 3-6 word label capturing the shared *citation thread* — the intellectual
#   reason a researcher would cite these papers together. Produces labels like
#   "Sparse Reward Policy Learning" rather than raw keyword extractions.
#   Requires ANTHROPIC_API_KEY in GitHub repo secrets.
#
# ALTERNATIVE: KeyBERT keyword extraction (Option 1 — original approach)
#   Uses semantic keyword extraction via SPECTER2 to find the most
#   representative phrase for each cluster. No API key required.
#   To restore: replace generate_cluster_labels() with the KeyBERT version
#   (see git history) and add keybert to pip install in the workflow.
#
# FALLBACK: Pre-computed labels file (Option 4)
#   Generate a parquet with columns x, y, text (optionally level, priority)
#   using any method, then pass via --labels to the CLI.
# ──────────────────────────────────────────────
def _strip_urls(text: str) -> str:
    """Remove URLs and arXiv-style citation keys from abstract text.

    Targets:
      - http/https URLs (e.g. https://github.com/user/repo)
      - Bare github.com / huggingface.co references
      - Citation-key tokens containing digits (e.g. cho2026_tokenizer,
        agentlab_main, youtu_cy06ljee1jq) — identifiers, not natural language
    """
    # Remove full URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove bare domain references
    text = re.sub(r'\b(?:github|huggingface|arxiv)\.(?:com|org|io)\S*', '', text, flags=re.IGNORECASE)
    # Remove citation-key tokens: any word containing both digits and underscores,
    # or camelCase identifiers mixed with digits (e.g. w4a4, cho2026)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    return " ".join(text.split())  # normalise whitespace


def generate_cluster_labels(
    df: pd.DataFrame,
    model_name: str = "specter2",
    precomputed_dist: np.ndarray | None = None,
) -> str:
    """
    Cluster papers with HDBSCAN then use Claude Haiku to generate a concise,
    meaningful label for each cluster. Writes a labels.parquet file and returns
    its path for the --labels CLI flag.

    Args:
        df:               Full paper dataframe with projection columns populated.
        model_name:       "specter2", "sbert", or "hybrid" — selects which
                          column set to read for 2D coords and (if
                          precomputed_dist is None) for clustering input.
        precomputed_dist: Optional precomputed n×n distance matrix. When
                          provided, HDBSCAN uses metric="precomputed" on it
                          directly. Pass the matrix returned by
                          embed_and_project_hybrid() to avoid recomputing.
    """
    import anthropic
    from sklearn.cluster import HDBSCAN

    # ── Column names per model ───────────────────────────────────────────
    if model_name in ("specter2", "hybrid"):
        col_embed_50d = "embedding_50d"
        col_proj_x    = "projection_hybrid_x" if model_name == "hybrid" else "projection_x"
        col_proj_y    = "projection_hybrid_y" if model_name == "hybrid" else "projection_y"
        prompt_framing = (
            "Papers in the same cluster were grouped because researchers in "
            "the same domain tend to cite them together — they share a "
            "methodology, theoretical framework, or problem formulation even "
            "if their surface topics differ. Capture the shared citation thread."
        )
    elif model_name == "sbert":
        col_embed_50d = "embedding_sbert_50d"
        col_proj_x    = "projection_sbert_x"
        col_proj_y    = "projection_sbert_y"
        prompt_framing = (
            "Papers in the same cluster were grouped by semantic similarity — "
            "they share a common research topic, approach, or application domain. "
            "Capture the shared topic or methodology concisely."
        )
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to GitHub repo secrets "
            "and expose it in the workflow YAML under env:."
        )

    client = anthropic.Anthropic(api_key=api_key)
    print(f"  Generating Claude Haiku cluster labels (embedding model: {model_name.upper()})...")

    # ── How many abstracts to send per cluster ───────────────────────────
    # ABSTRACTS_PER_LABEL: number of abstracts sampled per cluster for the
    # Haiku prompt. More = richer context but longer prompt.
    # Recommended range: 6–12. Default: 8.
    ABSTRACTS_PER_LABEL = 8
    #
    # ABSTRACT_CHAR_LIMIT: each abstract is truncated to this many characters
    # before being sent. Keeps prompt size predictable.
    # Recommended range: 300–600. Default: 450.
    ABSTRACT_CHAR_LIMIT = 450

    # 2D coords used for label centroid placement.
    coords_2d = df[[col_proj_x, col_proj_y]].values.astype(np.float64)

    # ── Clustering input ─────────────────────────────────────────────────
    # Priority: precomputed distance matrix > 50D embedding column > 2D fallback.
    # Hybrid mode passes precomputed_dist so HDBSCAN sees the blended distances
    # directly. Single-model modes cluster on the 50D column.
    if precomputed_dist is not None:
        cluster_input  = precomputed_dist
        cluster_metric = "precomputed"
        print(f"  Clustering on precomputed hybrid distance matrix.")
    elif col_embed_50d in df.columns and df[col_embed_50d].notna().all():
        cluster_input  = np.array(df[col_embed_50d].tolist(), dtype=np.float32)
        cluster_metric = "cosine"
        print(f"  Clustering on 50D cosine space ({model_name.upper()}).")
    else:
        cluster_input  = coords_2d
        cluster_metric = "euclidean"
        print(f"  Clustering on 2D coords (fallback — {col_embed_50d} not available).")

    # ════════════════════════════════════════════════════════════════════
    # HDBSCAN SETTINGS — all changes take effect on the next build.
    # These settings do not affect database.parquet in any way.
    # ════════════════════════════════════════════════════════════════════
    #
    # ── min_cluster_size (int, default: adaptive) ───────────────────────
    # Minimum number of papers required to form a cluster.
    # Lower  → more clusters, more specific labels.
    # Higher → fewer clusters, more general labels.
    # Syntax:   HDBSCAN(min_cluster_size=5, ...)
    # Range:    3–15 recommended for a ~300 paper corpus.
    #
    # ── min_samples (int, default: 4) ───────────────────────────────────
    # Controls how conservative cluster assignment is.
    # Lower  → more points pulled into clusters, fewer noise points.
    # Syntax:   HDBSCAN(..., min_samples=4, ...)
    # Range:    1–5 recommended.
    #
    # ── cluster_selection_method (str, default: "leaf") ─────────────────
    # "eom"  — Excess of Mass; finds clusters of varying sizes.
    # "leaf" — Selects leaf nodes; smaller, more granular clusters (current).
    #
    # ── cluster_selection_epsilon (float, default: 0.0) ─────────────────
    # Merges clusters closer than this distance threshold.
    # Syntax:   HDBSCAN(..., cluster_selection_epsilon=0.5)
    # Range:    0.0 (off) to ~2.0.
    # ════════════════════════════════════════════════════════════════════
    clusterer = HDBSCAN(
        min_cluster_size=max(5, len(df) // 40),
        min_samples=4,
        metric=cluster_metric,
        cluster_selection_method="leaf",
    )
    cluster_ids = clusterer.fit_predict(cluster_input)

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"  Found {n_clusters} clusters (noise points excluded).")

    # ── Haiku prompt ─────────────────────────────────────────────────────
    # Instructs Claude to find the shared *citation thread* — the intellectual
    # reason a researcher would cite multiple papers from this group together —
    # rather than just naming the broadest topic. This surfaces SPECTER2's
    # citation-graph groupings as human-readable explanations.
    SYSTEM_PROMPT = (
        "You are a research cartographer labeling clusters of AI papers. "
        + prompt_framing
        + "\n\nYour task: read the abstracts and return a single label of 3–6 words "
        "that captures the shared intellectual thread.\n\n"
        "Rules:\n"
        "- Focus on methodology or framing, not just topic keywords.\n"
        "- Prefer noun phrases: 'Sparse Reward Policy Learning' not 'Papers About RL'.\n"
        "- No punctuation, no quotes, no sentence structure — just the label.\n"
        "- Return ONLY the label text and nothing else."
    )

    label_rows = []
    for cid in sorted(set(cluster_ids)):
        if cid == -1:
            continue

        mask = cluster_ids == cid
        n_papers = int(mask.sum())
        cluster_coords = coords_2d[mask]
        cluster_abstracts = df.loc[mask, "abstract"].tolist()

        # Sample up to ABSTRACTS_PER_LABEL abstracts, clean and truncate each.
        sample = cluster_abstracts[:ABSTRACTS_PER_LABEL]
        cleaned = [_strip_urls(a)[:ABSTRACT_CHAR_LIMIT] for a in sample]
        numbered = "\n\n".join(f"[{i+1}] {a}" for i, a in enumerate(cleaned))
        user_msg = f"Cluster of {n_papers} papers. Abstracts:\n\n{numbered}"

        # Call Haiku with a simple retry on transient errors.
        label_text = None
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=32,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                label_text = response.content[0].text.strip().title()
                break
            except Exception as e:
                print(f"  Cluster {cid}: Haiku API attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        if not label_text:
            print(f"  Cluster {cid}: all API attempts failed — skipping label.")
            continue

        cx = float(cluster_coords[:, 0].mean())
        cy = float(cluster_coords[:, 1].mean())
        print(f"  Cluster {cid} ({n_papers} papers): '{label_text}'  @ ({cx:.2f}, {cy:.2f})")
        label_rows.append({"x": cx, "y": cy, "text": label_text,
                            "level": 0, "priority": 10})

    labels_df = pd.DataFrame(label_rows)
    labels_path = "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    print(f"  Wrote {len(labels_df)} labels to {labels_path}.")
    return labels_path
    return labels_path


# ──────────────────────────────────────────────
# 8. HTML POP-OUT PANELS  (Shortcuts + About)
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
  .arm-tile-icon { flex-shrink:0; width:18px; height:18px; color:#64748b; display:flex; align-items:center; justify-content:center; }
  .arm-tile-text { display:flex; flex-direction:column; gap:1px; }
  .arm-tile-label { font-size:12px; font-weight:600; color:#cbd5e1; line-height:1.3; transition:color 0.15s; }
  .arm-tile-sub   { font-size:11px; color:#64748b; line-height:1.3; }
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
</script>

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

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/#?state=rVZtj5s4EP4rp9F95CLICwG-VdVWOmnvdNfet2oVOTCbuHUwsofdpBH__TQGEiBkWVX9kuDxM8-M7Xk7wwsaK3UOCfizYD3zwQOSB7QkDgUkwXodrP0w9sNZHMUepHthyEJyhoB_6FQgJICHLWaZzHdOmRTLHjqyTJBg-BESKIz-hilJnW-O4MGpLzoxAx4JEkhVaQnNRoktKvAgFYQ7bVjhMxYlCVaAqvJg3nGlMJhJhtqOL1_-ffztn-tG5cGio0Jiq7CD_q9Zp1qVh9xC8rXZ8kBsLRmREnhQGnZKZiwtaa_NJtVlTtclSTT12XEjsgwZqQvMhcLjRuaWJJXuzLk4OG9HNx2pkfcBfIjeZmNeZj3pdzy9atOXkS5k2hXYcvssUfU8fUZK9875zq1fDmkxl9pI4ne7dXsodRfu9cLl8r1Z-WxkEB6D0BjGBIfK5gVT0nzVn_58fNj8_eGvB3iqPFh2nti580ehNHXeuX3UNjrro7dyDqzVFEUnHoYsly0mCqeI6mgacrCU1ddT6i4Mh9oyc8qRU24d7oeqEic0HOFnOAjzHRLYCr5JSyeGM5dSH7XSBhL4nSFu8UlkyFmEeardO9a53Tl810zlsvwMYrczuBN0OQVUVeUNLD9LReisNV-_xMqTBxZVHUgM2prS7vnjyg1Hd1uvMtsh1VfS3LhNhcJZE7vpXuQ5qhrvXStgwzPLtTkIJX-4akFk5LZ0rpza24bkq-8FT5xh7I0rL0d4qjhk48l4G5YWMXIfbpfPEvhTfL3iNKTrbDq2YIrtzeo2ZH8TLO2jtAQJmRKd7flP2e4Wz3fZ7yrc-LD4KR_a-vwu-y34xvZkMRsv_3eN9lA31ibr3lhbuWurg7mxNFkYb5rVXTM1wLFO1suxjneX-IJx3NG7ua-d8y51C3HMk9nfa8BDzuFMNJn7Ix38TkG5QhzzZB0YHwRu2tMIyvFP5vrISPEWuYM45skM7s8TN4NBZ5dbVz0NfyE3VLYj8YvE10IbarrVchYto3gVroJFtIqiMHJtaj1bxcs4jFf-IppHQRx7dZeBxJ8Fob-I1lEYhMt5OPfnYdvE-V-Jky55OlacR-366oITJ2ewe_36sZnVn4WyWFXV_w">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="9,2 11.1,6.6 16.2,7.3 12.6,10.7 13.5,15.8 9,13.4 4.5,15.8 5.4,10.7 1.8,7.3 6.9,6.6"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Reputation score</span></span>
    </a>

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/#?state=rVZbj5s6EP4r1aiPnAhyIYS3qtpKR9pz1NtbtYocmE3cOhjZw27SiP9ejYEECFlWq74kePzNN2N7bid4QmOlziAGfxIsJz54QHKPlsQ-hzhYLoOlH0bz2cRfTT1IdsKQhfgEAf_QMUeIAfcbTFOZbZ0yKZbdtWSpIMHwA8SQG_0TE5I6Wx_Ag2NXdGQGPBDEkKjCEpq1EhtU4EEiCLfasIIoaKfNOtFFRlCWHkxbzuQGU8lg2_Lm25f7d58vG6UHs5YKiY3CFvp7vU60KvaZhfhHveWB2FgyIiHwoDDslkxZ2nbovCSJpjo9rkWaIiN1jplQeFjLzJKkwp06E3vn7eCmIzXyNoAP0dmszcu0I_2Fx2dtujLSuUzaAltsHiWqjqePSMnOOf8V84IEG70c0mImtZHEL3ftdl_qLtzrBMz5e73w2UgvQHrB0Y8KDpb1Eyak-ao__Xt_t_7_w3938FB6MG89sXPnn1xpar1z86hNfFZHb-QcWIsxilY89FnOW0wUjhFV0dTnYCmrL8fUXRj2tWXqlCOn3DjcDVUljmg4wk-wF-YXxLARfJOWjgxnLqU-aqUNxPCeIW7xSaTIWYRZot07VtndOnwnRV2en0Bstwa3gs6ngLIsvZ7lR6kInbX6669YefDAoqoCiUEbU9gdf1y44eBu61mmW6TqSuobt4lQOKljN9mJLENV4b1LDax5Jpk2e6Hkb1ctiIzcFM6VY3PbEP_wveCBM4y9ceXlAA8lh-xqNN76pUUM3Ifb5bME_hhfpzj16Vqbji0YY3uxuvXZXwRLey8tQUymQGd7-ibb7eL5KvtthSsfZm_yoanPr7LfgK9sjxaz4fJ_02gHdWVttO4NtZWbtlqYK0ujhfGqWd00UwEc62i9HOp4N4nPGMcdvZr70jlvUjcQxzya_Z0G3OdsbbqZaDT3Bzr4jYJygTjm0TowPAhctacBlOMfzfWBkeIlcgdxzKMZ3J0nrgaD1i63rmoe_kZuqGyG4ieJz7k2VHer-SSaR6tFuAhm0SKKwsi1qeVksZqvwtXCn0XTKFitvKrLQOxPgtCfRcsoDML5NJz607Bp4vyvxFEXPB8rzqNmfXHBieMT2J1-_lhP649CWSzL8g8">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="6.5" cy="6" r="2.5"/><path d="M1 16c0-3.3 2.5-5 5.5-5"/><circle cx="13" cy="6" r="2.5"/><path d="M17 16c0-3.3-2.5-5-5.5-5"/><path d="M10 16c0-2.8-1.8-4.5-4-4.5"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author count</span><span class="arm-tile-sub">More authors tends to be better</span></span>
    </a>

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/#?state=rVZLj5tIEP4rUSlHYoEfGHOLookUaRIlm71FI6sNNXZv2jTqLmbMWvz3VTUwBoyHUbQXm67HV9XV9TrDExordQYx-LNgPfPBA5JHtCSOOcTBeh2s_TCKwtl6vvIgOQhDFuIzBPxDZY4QAx53mKYy2ztlUky769BSQYLFTxBDbvQ_mJDU2fYEHpR9UskIeCKIIVGFJTRbJXaowINEEO61YQVR0EGbrcVMaiOphKryYN5xKDeYSlawHY9-_rh_9_3CqDxYdFRI7BR2pP9uzolWxTGzEP9qWB6InSUjEgIPCsOuyZSptVOJLjK6HEmiqSOAW5GmyJI6x0woPG1lZklS4W6eiaPzdpTpQI28LcCX6DEb8zLtUX9j-axNn0Y6l0mXYIvdo0TV8_QRKTk45__CvCDBRi-XvDyEB9duD6ku4F4vaV6-tyufjQySZJAgw8zghNk-YUKaQ_35y_3d9tvHr3fwUHmw7Dyxc-dDrjR13rl91DZH66u3dE6s1RREJx-GKC8sBgqngOpsGmIwldXXU-ouDYfaMnXKkVNuHe6nqhIlGs7wMxyF-Q0x7ARH0lLJ4oyl1CettIEY3rOIO3wWKXIVYZZo9451hXcu3zVTuVo_g9jvDe4FvdwCqqryBpYfpSJ01pqv_8XKgwcWVZ1ILLQzhT3wxwUbTi5azzLdI9UhaSJuE6Fw1uRuchBZhqqW9y59sMGZZdochZL_um5BZOSucK6UbbQh_uV7wQNXGHvj2ssJHipO2c1kvg1bixiJh-PyXQJ_Cq_XnIZwHaZDC6bQXu1uQ_RXhaW9l5YgJlOgsz3_I9vd5vkm-12FKx8Wf-RD25_fZL8VvrI92czG2_9Noz2pK2uTfW9srNy01ZG5sjTZGK-G1U0ztYBDneyXYxPvJvCLjMOO3ox9mZw3oVsRhzxZ_b0BPMTsMN1ONFn7IxP8RkMZbFuTfWB8EbgaTyNSDn-y1kdWitfAnYhDnqzg_j5xtRh0uDy66p34J7mlsl2MnyQ-59pQM62Ws2gZbVbhKlhEqygKIzem1rPVZrkJLyt_Ec2jYLPx6ikDsT8LQn8RraMwCJfzcO7Pw3aI878SpS54R1ZcR-366oITJ2ewe_36sZnVn4WyWFXV_w">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5L9 4l5 4.5"/><path d="M5.5 8v5.5h7V8"/><path d="M7 13.5v-3h4v3"/><path d="M3 8.5h12"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author seniority</span><span class="arm-tile-sub">Highlights established researchers</span></span>
    </a>

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/#?state=rVbdj6M2EP9XqlEfaQT5IIS302lPqrSt2l7fTqvIgdnEPQdH9rAbGvG_V2NgYwhZVqe-JHg-fjMez9cFXtBYqQtIIZxF61kIAZA8oiVxPEEardfROow3i3CWhEkA2UEYstBeIMIfqk4IKeBxh3kui71TJsW0B4-WCxIsfoYUTkb_gxlJXWzPEEDVJ1WMgGeCFDJVWkKzVWKHCgLIBOFeG1aQhSVJpdPIdFmQqaCuA5h7Pp0M5pJ1rOfU1z8ff_rjyqgDWHgqJHYKPem_23OmVXksLKTfWlYAYmfJiIwggNKwdzJnakkHbRqXrkeSaJog4FbkObKkPmEhFJ63_lUKcXTejjKbe8r7AnyJHrM1L_Me9TtWr9r0aaRPMvMJttw9S1Q9T5-RsoNz_i88lSTY6PWSFgupjSR-wLHn6VNdwINe3rx9b1chGxnkySBHhsnBObN9wYw0h_rLr48P298__fYAT3UAS--JnTu_nJQm7527R-3StLl6R-fEWk1BePkwRHljMVA8BdRk0xCDqay-nlJ3aTjUlrlTTpxy53A_VZWo0HCGX-AozHdIYSc4kpYqFmcspT5rpQ2k8DOLuMMXkSNXERaZdu_YFLl3ed9M7cr9AmK_N7gX9HYLqOs6GFh-lorQWWu__hcrTwFYVE0isdDOlPbAH1dsOLtovcp8j9SEpI24zYTCWZu72UEUBapGPri2whZnVmhzFEr-67oFkZG70rlSddGG9FsYRE9cYeyNay9neKo5ZTeT-TZsLWIkHo7Ld4nCKbxecxrCeUyHFk2hvdvdhujvCkv7KC1BSqZEZ3v-Q7b95vkh-77CjQ-LH_Kh688fst8J39iebGbj7f-u0Z7UjbXJvjc2Vu7a8mRuLE02xpthdddMI-BQJ_vl2MS7C_wm47CTD2NfJ-dd6E7EIU9Wf28ADzE9ptuJJmt_ZILfaShXEYc82QfGF4Gb8XRvm5us9ZGV4j1wJ-KQJyu4v0_cLAYel0dXsxZ_JbdUdrvxi8TXkzbUTqvlLFkmm1W8ihbJKknixI2p9Wy1WW7izSpcJPMk2myCZspAGs6iOFwk6ySO4uU8nofzuBvi_K9EpUtekxXXUXe-uuDI6QXsQb9-bpf2Z6Es1nX9Hw">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="9" r="7"/><path d="M2 9h14"/><path d="M9 2c-2 2-3 4.5-3 7s1 5 3 7"/><path d="M9 2c2 2 3 4.5 3 7s-1 5-3 7"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Country</span><span class="arm-tile-sub">Geographic spread of AI research</span></span>
    </a>

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/#?state=rVZZj5swEP4r1aiPNIIcBHirqq1UaVv1eqtWkQOziVsHI9vsJo3479UYWAwhy2rVlwTP8c14PNcZHlBpLnNIwJ8F65kPHhh-QG3YoYAkWK-DtR_GYTRbxnMP0j1TRkNyhoB-zKlASAAPW8wynu-sshFEu3FoGTOMxI-QQKHkb0wNl_nmCB6c-qQTIeDRQAKpKLVBtRFsiwI8SJnBnVSkwHNtuCmthnWhqjyYOw4VCjNOCtrx6Me32zdfO0blwcJRMWwr0JH-2ZxTKcpDriH51bA8YFttFEsNeFAqco1nRC3NXqpNKsvcdEfDUdURwA3LMiRJWWDOBB437j1ydrDejjItqOLXBegSPWZjnmc96h88PUrVpxlZ8NQl6HJ7z1H0PL1Hk-6t89-xKA0jo90lNeZcKm7o9S7dHlJtwL1e0jx9b1Y-GRkkySBBhplBCbN5wNRICvXHT7c3my_vP9_AXeXB0nli6867QkjjvHP7qG2O1ldv6ZRYqykIJx-GKE8sAgqngOpsGmIQldTXU-o2DYfaPLPKkVVuHe6nqhIlGs7wMxyF-Q0x7ARH0lLJ4oyl1CettIEY3rOIO3wWKXIVYZZo9451hXcu3zVTuVo_g9jvDe4FvdwCqqryBpYfpSJ01pqv_8XKgwcWVZ1ILLQzhT3wxwUbTi5azzLdI9UhaSJuE6Fw1uRuchBZhqqW9y59sMGZZdochZL_um5BZOSucK6UbbQh_uV7wQNXGHvj2ssJHipO2c1kvg1bixiJh-PyXQJ_Cq_XnIZwHaZDC6bQXu1uQ_RXhaW9l5YgJlOgsz3_I9vd5vkm-12FKx8Wf-RD25_fZL8VvrI92czG2_9Noz2pK2uTfW9srNy01ZG5sjTZGK-G1U0ztYBDneyXYxPvJvCLjMOO3ox9mZw3oVsRhzxZ_b0BPMTsMN1ONFn7IxP8RkMZbFuTfWB8EbgaTyNSDn-y1kdWitfAnYhDnqzg_j5xtRh0uDy66p34J7mlsl2MnyQ-59pQM62Ws2gZbVbhKlhEqygKIzem1rPVZrkJLyt_Ec2jYLPx6ikDsT8LQn8RraMwCJfzcO7Pw3aI878SpS54R1ZcR-366oITJ2ewe_36sZnVn4WyWFXV_w">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="14" width="14" height="2" rx="0.5"/><rect x="6" y="7" width="2" height="7"/><rect x="10" y="7" width="2" height="7"/><path d="M2 7h14"/><path d="M5 7L9 3l4 4"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Institution</span><span class="arm-tile-sub">Academic vs industry research</span></span>
    </a>

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/#?state=rVZLj9s2EP4rxaBH1ZC8a-txC4INUGBbtElvwcKgpVmbCS0a5GjXjqH_Xgwl2ZQsrxZBLrY4j2-Gw3md4AWNlbqEDMJZFM9CCIDkDi2J3R6yKI6jOIzD-3CWzO8DyLfCkIXsBBH_0HGPkAHu1lgUstw4ZVJMe_BohSDB4gfIYG_0N8xJ6nJ1gACOfdKREfBAkEGuKktoVkqsUUEAuSDcaMMKeo-lUHhY2Wr9LFEVUNcBzD2P9gYLyRrWc-nLv4-__XNh1AHceSok1go96f_ac65VtSstZF9bVgBibcmInCCAyrBvsmBqRVttVrmuSrocSaJpQoArURTIkmf_ZWlJUuWuXoqd83aU6UCNvC3Al-gxW_Oy6FG_4_FVmz6N9F7mPuEcVY_2jJRvnfOfcV-RYKOXS1ospTaS-Pmu3R5SXcCDXtacv1eLkI0MsmSQIcPU4IxZvWBOmkP96c_Hh9XfH_56gKc6gHvviZ07f-yVJu-du0ftkrS5ekfnxFpMQXj5MEQ5sxhoOQXUZNMQg6msHk-puzQcasumOhKn3DncT1Uljmg4w0-wE-Y7ZLAWHElLRxZnLKU-aqUNZPA7i7jDJ1EgVxGWuXbv2JS4d3nfTO2K_QRiszG4EXS-BdR1HQwsP0tF6Ky1X7_EylMAFlWTSCy0NpXd8scFGw4uWq-y2CA1IWkjbnOhcNbmbr4VZYmqkQ8ujbDFmZXa7ISSP1y3IDJyXTlXjl20IfsaBtETVxh749rLAZ5qTtl0Mt-GrUWMxMNx-S5ROIXXa05DOI_p0KIptDe72xD9TWFpH6UlyMhU6GzPf8q23zzfZd9XuPLh7qd86Przu-x3wle2J5vZePu_abQndWVtsu-NjZWbtjyZK0uTjfFqWN000wg41Ml-OTbxbgL3do0oeTf2ZXLehO5EHPJk9fcG8BDTY7qdaLL2Ryb4jYZyEXHIk31gfBG4Gk8jUg5_stZHVoq3wJ2IQ56s4P4-cbUYeFweXc1S_IXcUtltxi8SX_faUDut7mfpfBHFy2SRpGGUxvHCzal4tozTMA3ju2SeLhdhGjRjBrJwFiVhcrdMwjRN5tE8XXZDnP-VOOqKl2TFddSdLy44cnYCu9WvH9uV_Vkoi3Vd_w">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 5h8l4 4-4 4H3V5z"/><line x1="7" y1="9" x2="7" y2="9" stroke-width="2" stroke-linecap="round"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">OpenAlex subfield</span><span class="arm-tile-sub">OpenAlex taxonomy</span></span>
    </a>

    <a class="arm-tile" href="https://leefischman.github.io/ai-research-atlas-staging/">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="14" height="12" rx="1.5"/><line x1="2" y1="8" x2="16" y2="8"/><line x1="6" y1="2" x2="6" y2="6"/><line x1="12" y1="2" x2="12" y2="6"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Release date</span></span>
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

    <p class="arm-p">A live mostly semantic map of recent AI research from arXiv (cs.AI), rebuilt every day across a rolling 4-day window. Each point is a paper.</p>
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


# ──────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
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
        if existing_df.empty:
            print("  No results from arXiv and no existing DB. Cannot build. Exiting.")
            exit(0)
        print(f"  No new papers from arXiv (weekend or dry spell). "
              f"Rebuilding atlas from {len(existing_df)} existing papers.")

    if results:
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
    else:
        # No new arXiv papers — use existing DB as-is (weekend / dry spell rebuild)
        df = existing_df.drop(columns=["group"], errors="ignore")

    # Backfill Reputation for older rows that predate the column.
    if "Reputation" not in df.columns or df["Reputation"].isna().any():
        missing = df["Reputation"].isna() if "Reputation" in df.columns else pd.Series([True] * len(df))
        if missing.any():
            print(f"  Backfilling Reputation for {missing.sum()} older rows...")
            df.loc[missing, "Reputation"] = df.loc[missing].apply(calculate_reputation, axis=1)

    # Backfill label_text for older rows that predate the column (e.g. weekend
    # rebuilds from existing DB, or rows added before label_text was introduced).
    if "label_text" not in df.columns or df["label_text"].isna().any():
        missing_lt = df["label_text"].isna() if "label_text" in df.columns else pd.Series([True] * len(df))
        if missing_lt.any():
            print(f"  Backfilling label_text for {missing_lt.sum()} older rows...")
            df.loc[missing_lt, "label_text"] = df.loc[missing_lt, "title"].apply(
                lambda t: scrub_model_words(f"{t}. {t}. {t}.")
            )

    # Embed & project (incremental mode only)
    labels_path      = None
    precomputed_dist = None
    if EMBEDDING_MODE == "incremental":
        if EMBEDDING_MODEL == "hybrid":
            print(f"  Incremental mode: hybrid embedding (SPECTER2 + SBERT + TF-IDF).")
            df, precomputed_dist = embed_and_project_hybrid(
                df,
                w_specter2=HYBRID_WEIGHT_SPECTER2,
                w_sbert=HYBRID_WEIGHT_SBERT,
                w_tfidf=HYBRID_WEIGHT_TFIDF,
            )
        else:
            print(f"  Incremental mode: embedding new papers only ({EMBEDDING_MODEL.upper()}).")
            df = embed_and_project(df, model_name=EMBEDDING_MODEL)
        labels_path = generate_cluster_labels(
            df,
            model_name=EMBEDDING_MODEL,
            precomputed_dist=precomputed_dist,
        )

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
    # Clear docs immediately before the build so the site is never left broken
    # if an earlier step fails or exits early (e.g. empty arXiv on weekends).
    clear_docs_contents("docs")
    print(f"  Building atlas ({EMBEDDING_MODE} mode, {EMBEDDING_MODEL.upper()})...")

    # Select the right projection columns for the Atlas CLI.
    if EMBEDDING_MODEL == "sbert":
        proj_x_col = "projection_sbert_x"
        proj_y_col = "projection_sbert_y"
    elif EMBEDDING_MODEL == "hybrid":
        proj_x_col = "projection_hybrid_x"
        proj_y_col = "projection_hybrid_y"
    else:
        proj_x_col = "projection_x"
        proj_y_col = "projection_y"

    if EMBEDDING_MODE == "incremental":
        # Incremental mode: embeddings are pre-computed via --x/--y.
        # Claude Haiku labels are passed via --labels, bypassing TF-IDF entirely.
        # --text is intentionally omitted so the CLI cannot generate competing
        # automatic TF-IDF labels that would override our Haiku labels.
        atlas_cmd = [
            "embedding-atlas", DB_PATH,
            "--x",          proj_x_col,
            "--y",          proj_y_col,
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

        # labelDensityThreshold controls label culling in Embedding Atlas.
        # WARNING: setting this too low (e.g. 0.02) kills all label display.
        # Leave unset to use Atlas defaults unless you understand the exact
        # semantics of this field in your installed Atlas version.
        # conf["labelDensityThreshold"] = 0.1

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
