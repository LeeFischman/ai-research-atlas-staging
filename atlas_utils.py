# atlas_utils.py
# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities for the AI Research Atlas pipeline.
#
# Imported by update_map_v2.py.  update_map.py is deliberately left unchanged
# so it continues to work as a standalone fallback / reference implementation.
#
# Contents
# --------
#   §1  Text helpers          scrub_model_words, _strip_urls
#   §2  Docs cleanup          clear_docs_contents
#   §3  Reputation scoring    INSTITUTION_PATTERN, calculate_reputation
#   §4  Rolling database      load_existing_db, merge_papers
#   §5  arXiv fetch           fetch_arxiv  (exponential back-off)
#   §6  Embedding / UMAP      embed_and_project, _embed_only,
#                             compute_hybrid_distances, embed_and_project_hybrid
#   §7  Atlas build + deploy  build_and_deploy_atlas
#   §8  HTML panels           build_panel_html
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH         = "database.parquet"
STOP_WORDS_PATH = "stop_words.csv"
RETENTION_DAYS  = 4      # papers older than this are pruned each run
ARXIV_MAX       = 250    # max papers fetched per arXiv query

# arXiv retry policy
BASE_WAIT   = 15
MAX_WAIT    = 480
MAX_RETRIES = 7


# ══════════════════════════════════════════════════════════════════════════════
# §1  TEXT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def scrub_model_words(text: str) -> str:
    """Remove inflections of 'model' to reduce TF-IDF noise."""
    pattern = re.compile(r'model(?:s|ing|ed|er|ers)?\b', re.IGNORECASE)
    return " ".join(pattern.sub("", text).split())


def _strip_urls(text: str) -> str:
    """Remove URLs and arXiv-style citation keys from abstract text.

    Targets:
      - http/https URLs (e.g. https://github.com/user/repo)
      - Bare github.com / huggingface.co / arxiv.org references
      - Citation-key tokens containing digits (e.g. cho2026_tokenizer,
        agentlab_main, youtu_cy06ljee1jq) — identifiers, not natural language
    """
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b(?:github|huggingface|arxiv)\.(?:com|org|io)\S*', '',
                  text, flags=re.IGNORECASE)
    # Remove citation-key tokens: word containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    return " ".join(text.split())


# ══════════════════════════════════════════════════════════════════════════════
# §2  DOCS CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def clear_docs_contents(target_dir: str) -> None:
    """Delete every file/folder inside target_dir (but keep the dir itself)."""
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


# ══════════════════════════════════════════════════════════════════════════════
# §3  REPUTATION SCORING
# ══════════════════════════════════════════════════════════════════════════════

INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    # US Universities
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
    # US National Labs & Research Institutes
    "Allen Institute", "AI2", "Allenai",
    "IBM Research", "Microsoft Research", "MSR",
    "Google Research", "Google Brain", "Google DeepMind",
    "NVIDIA Research", "Intel Labs", "Salesforce Research",
    "Adobe Research", "Baidu Research", "Amazon Science",
    "Apple ML Research",
    # US AI Labs
    "OpenAI", "Anthropic", "DeepMind", "Google DeepMind",
    "FAIR", "Meta AI", "xAI", "Mistral",
    "Cohere", "Stability AI", "Inflection AI",
    "Character AI", "Runway", "Hugging Face",
    # UK
    "Oxford", "University of Oxford",
    "Cambridge", "University of Cambridge",
    "Imperial College", "UCL", "University College London",
    "Edinburgh", "University of Edinburgh",
    "King's College London", "University of Manchester",
    "University of Bristol", "University of Warwick",
    "Alan Turing Institute",
    # Canada
    "University of Toronto", "UofT", "Vector Institute",
    "McGill", "McGill University",
    "Mila", "Montreal Institute",
    "University of Montreal",
    "University of British Columbia", "UBC",
    "University of Alberta",
    # France
    "INRIA", "ENS", "CNRS", "Sorbonne",
    "PSL University", "Paris-Saclay",
    # Germany
    "Max Planck", "MPI",
    "TU Berlin", "TU Munich", "Technical University of Munich",
    "DFKI", "Helmholtz",
    "University of Tuebingen",
    # Switzerland
    "ETH Zurich", "EPFL", "University of Zurich",
    # Netherlands
    "University of Amsterdam", "CWI", "Delft",
    # Scandinavia
    "KTH", "Chalmers", "University of Copenhagen", "DTU", "NTNU",
    # Israel
    "Technion", "Hebrew University", "Weizmann Institute",
    "Tel Aviv University", "Bar-Ilan University",
    # China
    "Tsinghua", "Tsinghua University",
    "Peking University", "PKU",
    "Shanghai AI Lab",
    "Zhejiang University",
    "USTC", "Fudan University", "Renmin University",
    "Chinese Academy of Sciences", "CAS",
    "BAAI", "Beijing Academy of Artificial Intelligence",
    "Alibaba DAMO", "Tencent AI Lab", "ByteDance", "SenseTime",
    # Japan
    "University of Tokyo", "Kyoto University",
    "RIKEN", "RIKEN AIP",
    "Osaka University", "Tohoku University",
    "Tokyo Institute of Technology", "Tokyo Tech",
    "Preferred Networks",
    # South Korea
    "KAIST", "Seoul National University", "SNU",
    "POSTECH", "Yonsei University",
    "Samsung Research", "Naver", "LG AI Research", "Kakao",
    # Singapore
    "NUS", "National University of Singapore",
    "NTU", "Nanyang Technological University",
    "AI Singapore", "ASTAR",
    # Australia
    "University of Sydney", "University of Melbourne",
    "Australian National University", "ANU",
    "University of Queensland", "CSIRO",
    # India
    "IIT", "IIT Bombay", "IIT Delhi", "IIT Madras",
    "IISc", "Indian Institute of Science",
    "Microsoft Research India",
    # Middle East
    "KAUST", "King Abdullah University",
    "Mohamed bin Zayed University", "MBZUAI",
    # Latin America
    "University of Sao Paulo", "USP", "PUC-Rio",
]) + r")\b", re.IGNORECASE)


def categorize_authors(n: int) -> str:
    if n <= 3:  return "1-3 Authors"
    if n <= 7:  return "4-7 Authors"
    return "8+ Authors"


def author_reputation_score(n: int) -> int:
    if n >= 8:  return 3
    if n >= 4:  return 1
    return 0


def calculate_reputation(row) -> str:
    score = 0
    full_text = f"{row['title']} {row['abstract']}".lower()
    if INSTITUTION_PATTERN.search(full_text):
        score += 3
    if any(k in full_text for k in ["github.com", "huggingface.co"]):
        score += 2
    score += author_reputation_score(row["author_count"])
    return "Reputation Enhanced" if score >= 3 else "Reputation Std"


# ══════════════════════════════════════════════════════════════════════════════
# §4  ROLLING DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_db(db_path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    df = pd.read_parquet(db_path)
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
    """Merge new papers into existing; duplicate arXiv IDs: new overwrites old."""
    if existing.empty:
        return new
    kept = existing[~existing["id"].isin(new["id"])]
    overwritten = len(existing) - len(kept)
    if overwritten:
        print(f"  Overwrote {overwritten} updated paper(s).")
    return pd.concat([kept, new], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# §5  arXiv FETCH WITH EXPONENTIAL BACK-OFF
# ══════════════════════════════════════════════════════════════════════════════

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
    raise RuntimeError(
        f"arXiv fetch failed after {MAX_RETRIES} attempts. Last: {last_exc}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# §6  EMBEDDING / UMAP
# ══════════════════════════════════════════════════════════════════════════════

def embed_and_project(df: pd.DataFrame, model_name: str = "specter2") -> pd.DataFrame:
    """Embed new papers and (re-)project ALL papers to 50D + 2D with UMAP.

    Model name controls which column set is written:
      "specter2" → embedding, embedding_50d, projection_x, projection_y
      "sbert"    → embedding_sbert, embedding_sbert_50d,
                   projection_sbert_x, projection_sbert_y

    All column sets coexist in the parquet — switching is a one-line env change.
    """
    from sentence_transformers import SentenceTransformer
    import umap as umap_lib

    if model_name == "specter2":
        hf_model_id  = "allenai/specter2_base"
        col_embed    = "embedding"
        col_embed_50d = "embedding_50d"
        col_proj_x   = "projection_x"
        col_proj_y   = "projection_y"
        text_col     = "text"
    elif model_name == "sbert":
        hf_model_id  = "sentence-transformers/all-mpnet-base-v2"
        col_embed    = "embedding_sbert"
        col_embed_50d = "embedding_sbert_50d"
        col_proj_x   = "projection_sbert_x"
        col_proj_y   = "projection_sbert_y"
        text_col     = "abstract"
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

    print(f"  Loading embedding model: {hf_model_id}")
    model = SentenceTransformer(hf_model_id)

    # Incremental embedding — only new papers get encoded
    if col_embed in df.columns:
        needs_embed = df[col_embed].isna()
    else:
        df[col_embed] = None
        needs_embed = pd.Series([True] * len(df))

    n_new = needs_embed.sum()
    if n_new:
        print(f"  Embedding {n_new} new paper(s) with {model_name.upper()}...")
        idx    = df.index[needs_embed].tolist()
        texts  = df.loc[idx, text_col].tolist()
        vecs   = model.encode(texts, show_progress_bar=True, batch_size=16,
                              convert_to_numpy=True)
        for i, pos in enumerate(idx):
            df.at[pos, col_embed] = vecs[i].tolist()
    else:
        print(f"  All papers already embedded with {model_name.upper()} — skipping.")

    all_vectors = np.array(df[col_embed].tolist(), dtype=np.float32)
    n = len(all_vectors)
    print(f"  Projecting {n} papers with UMAP (two-stage, {model_name.upper()})...")

    # Stage 1: 768D → 50D (for clustering)
    reducer_50d = umap_lib.UMAP(n_components=50, metric="cosine",
                                random_state=42, n_neighbors=15)
    coords_50d = reducer_50d.fit_transform(all_vectors)
    df[col_embed_50d] = [row.tolist() for row in coords_50d]

    # Stage 2: 768D → 2D (for display / direction vectors)
    reducer_2d = umap_lib.UMAP(n_components=2, metric="cosine",
                               random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer_2d.fit_transform(all_vectors)
    df[col_proj_x] = coords_2d[:, 0].astype(float)
    df[col_proj_y] = coords_2d[:, 1].astype(float)
    return df


def _embed_only(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Embed papers missing vectors without running UMAP.

    Used by hybrid mode to populate raw vector columns before distance blending.
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
        model  = SentenceTransformer(hf_model_id)
        idx    = df.index[needs_embed].tolist()
        texts  = df.loc[idx, text_col].tolist()
        vecs   = model.encode(texts, show_progress_bar=True, batch_size=16,
                              convert_to_numpy=True)
        for i, pos in enumerate(idx):
            df.at[pos, col_embed] = vecs[i].tolist()
    else:
        print(f"  All papers already embedded with {model_name.upper()} — skipping.")
    return df


def compute_hybrid_distances(
    df: pd.DataFrame,
    w_specter2: float = 0.75,
    w_sbert:    float = 0.0,
    w_tfidf:    float = 0.25,
) -> np.ndarray:
    """Build a normalised hybrid distance matrix from up to three sources.

    Each source is independently normalised to [0, 1] before blending.
    Weights are also normalised so the output is always in [0, 1].

    Returns: symmetric (n × n) float32 distance matrix.
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

    def _norm(d: np.ndarray) -> np.ndarray:
        mx = d.max()
        return (d / mx).astype(np.float32) if mx > 0 else d.astype(np.float32)

    if w_specter2 > 0:
        if "embedding" not in df.columns or df["embedding"].isna().any():
            raise RuntimeError("SPECTER2 vectors missing.")
        vecs = sk_normalize(np.array(df["embedding"].tolist(), dtype=np.float32))
        dist += w_specter2 * _norm(cosine_distances(vecs).astype(np.float32))
        print(f"  Hybrid: added SPECTER2 (w={w_specter2:.2f}).")

    if w_sbert > 0:
        if "embedding_sbert" not in df.columns or df["embedding_sbert"].isna().any():
            raise RuntimeError("SBERT vectors missing.")
        vecs = sk_normalize(np.array(df["embedding_sbert"].tolist(), dtype=np.float32))
        dist += w_sbert * _norm(cosine_distances(vecs).astype(np.float32))
        print(f"  Hybrid: added SBERT (w={w_sbert:.2f}).")

    if w_tfidf > 0:
        texts  = df["abstract"].apply(_strip_urls).tolist()
        tfidf  = TfidfVectorizer(max_features=20_000, sublinear_tf=True,
                                  min_df=2, ngram_range=(1, 2))
        tfidf_m = tfidf.fit_transform(texts)
        dist += w_tfidf * _norm(cosine_distances(tfidf_m).astype(np.float32))
        print(f"  Hybrid: added TF-IDF (w={w_tfidf:.2f}).")

    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    print(f"  Hybrid dist: shape={dist.shape}, "
          f"min={dist.min():.4f}, max={dist.max():.4f}, mean={dist.mean():.4f}.")
    return dist


def embed_and_project_hybrid(
    df: pd.DataFrame,
    w_specter2: float = 0.75,
    w_sbert:    float = 0.0,
    w_tfidf:    float = 0.25,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ensure both model vectors exist, build hybrid distance matrix, project to 2D.

    Writes projection_hybrid_x/y to df.
    Returns (df, dist_matrix) so callers can reuse the distance matrix.
    """
    import umap as umap_lib

    if w_specter2 > 0:
        df = _embed_only(df, "specter2")
    if w_sbert > 0:
        df = _embed_only(df, "sbert")

    dist = compute_hybrid_distances(df, w_specter2, w_sbert, w_tfidf)
    n    = len(df)
    print(f"  Projecting {n} papers with UMAP on hybrid distance matrix...")
    reducer = umap_lib.UMAP(
        n_components=2, metric="precomputed", random_state=42,
        n_neighbors=min(15, n - 1), min_dist=0.1,
    )
    coords_2d = reducer.fit_transform(dist)
    df["projection_hybrid_x"] = coords_2d[:, 0].astype(float)
    df["projection_hybrid_y"] = coords_2d[:, 1].astype(float)
    print("  Hybrid projection → projection_hybrid_x / projection_hybrid_y.")
    return df, dist


# ══════════════════════════════════════════════════════════════════════════════
# §7  ATLAS BUILD + DEPLOY
# ══════════════════════════════════════════════════════════════════════════════

def build_and_deploy_atlas(
    db_path:     str,
    proj_x_col:  str,
    proj_y_col:  str,
    labels_path: str,
    run_date:    str,
    docs_dir:    str = "docs",
) -> None:
    """Run the embedding-atlas CLI, unzip, patch config, inject panel HTML.

    Parameters
    ----------
    db_path     : path to the paper parquet (must contain proj_x_col / proj_y_col)
    proj_x_col  : name of the x-coordinate column in db_path
    proj_y_col  : name of the y-coordinate column in db_path
    labels_path : path to the labels parquet (columns: x, y, text)
    run_date    : human-readable date string injected into the panel footer
    docs_dir    : output directory for the built site (default "docs")
    """
    clear_docs_contents(docs_dir)
    print(f"  Building atlas (x={proj_x_col}, y={proj_y_col})...")

    atlas_cmd = [
        "embedding-atlas", db_path,
        "--x",      proj_x_col,
        "--y",      proj_y_col,
        "--labels", labels_path,
        "--export-application", "site.zip",
    ]
    subprocess.run(atlas_cmd, check=True)
    os.system(f"unzip -o site.zip -d {docs_dir}/ && touch {docs_dir}/.nojekyll")

    # ── Config override ────────────────────────────────────────────────────
    config_path = os.path.join(docs_dir, "data", "metadata.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            conf = json.load(f)

        conf["name_column"]  = "title"
        conf["label_column"] = "title"
        conf["color_by"]     = "Reputation"
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
        print(f"  {config_path} not found — skipping config override.")

    # ── Panel HTML injection ───────────────────────────────────────────────
    # Font <link> tags go in <head> to prevent FOUT (flash of unstyled text).
    # Panel CSS/JS/DOM goes before </body>.
    index_file = os.path.join(docs_dir, "index.html")
    if os.path.exists(index_file):
        font_html, panel_html = build_panel_html(run_date)
        with open(index_file, "r", encoding="utf-8") as f:
            content = f.read()
        if "</head>" in content:
            content = content.replace("</head>", font_html + "\n</head>")
        else:
            content = font_html + "\n" + content
        if "</body>" in content:
            content = content.replace("</body>", panel_html + "\n</body>")
        else:
            content += panel_html
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Info panel injected into index.html.")
    else:
        print(f"  {index_file} not found — skipping panel injection.")

    print("  Atlas build complete.")


# ══════════════════════════════════════════════════════════════════════════════
# §8  HTML PANELS
# ══════════════════════════════════════════════════════════════════════════════

def build_panel_html(run_date: str) -> tuple[str, str]:
    """Return (font_html, panel_html).

    font_html  — <link> tags injected into <head> to prevent font FOUT.
    panel_html — CSS, JS, and panel DOM injected before </body>.
    """
    font_html = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )
    panel_html = (
        """
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
  .arm-tile.arm-tile-active { background:rgba(96,165,250,0.12); border-color:rgba(96,165,250,0.4); }
  .arm-tile.arm-tile-active .arm-tile-label { color:#93c5fd; }
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

  // ── Shortcut color-by handler ─────────────────────────────────────────────
  // Atlas renders a <select> whose options have JSON-quoted values like
  // '"Reputation"', '"author_count"' etc. We find it by checking which
  // select contains a Reputation option, set its value via the native setter
  // (required for Svelte to detect the change), and fire a change event.
  function armSetColor(columnName, tileEl) {
    var colorSelect = Array.from(document.querySelectorAll('select')).find(function(sel) {
      return Array.from(sel.options).some(function(o) { return o.text.includes('Reputation'); });
    });
    if (colorSelect) {
      var nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype, 'value'
      ).set;
      nativeSetter.call(colorSelect, JSON.stringify(columnName));
      colorSelect.dispatchEvent(new Event('change', { bubbles: true }));
    }
    armClose('arm-shortcuts-panel', 'arm-shortcuts-tab');
    document.querySelectorAll('.arm-tile').forEach(function(t) { t.classList.remove('arm-tile-active'); });
    if (tileEl) { tileEl.classList.add('arm-tile-active'); }
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

    <a class="arm-tile" href="javascript:void(0)" onclick="armSetColor('Reputation', this)">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polygon points="9,2 11.1,6.6 16.2,7.3 12.6,10.7 13.5,15.8 9,13.4 4.5,15.8 5.4,10.7 1.8,7.3 6.9,6.6"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Reputation score</span><span class="arm-tile-sub">Institution &amp; open-source signals</span></span>
    </a>

    <a class="arm-tile" href="javascript:void(0)" onclick="armSetColor('author_count', this)">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="6.5" cy="6" r="2.5"/><path d="M1 16c0-3.3 2.5-5 5.5-5"/><circle cx="13" cy="6" r="2.5"/><path d="M17 16c0-3.3-2.5-5-5.5-5"/><path d="M10 16c0-2.8-1.8-4.5-4-4.5"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author count</span><span class="arm-tile-sub">More authors tends to be better</span></span>
    </a>

    <a class="arm-tile" href="javascript:void(0)" onclick="armSetColor('author_tier', this)">
      <span class="arm-tile-icon"><svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5L9 4l5 4.5"/><path d="M5.5 8v5.5h7V8"/><path d="M7 13.5v-3h4v3"/><path d="M3 8.5h12"/></svg></span>
      <span class="arm-tile-text"><span class="arm-tile-label">Author seniority</span><span class="arm-tile-sub">Highlights established researchers</span></span>
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

    <p class="arm-p">A live semantic map of recent AI research from arXiv (cs.AI), rebuilt every day across a rolling 4-day window. Each point is a paper. Groups are determined by Claude Haiku based on shared research methodology and problem formulation.</p>
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
    return font_html, panel_html
