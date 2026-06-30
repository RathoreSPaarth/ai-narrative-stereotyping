"""
Flagship piece -- Step 1: Embed every story and measure how close each one
sits to the "default" story the model reaches for, regardless of country.

WHY THIS VERSION LOOKS DIFFERENT FROM THE FIRST ONE
    The first version of this script clustered the embeddings with KMeans
    and reported what % of a country's stories fell into its single most
    common cluster ("12 shapes", later "19 shapes"). A silhouette-score
    scan across K=4..20 showed every candidate K scoring between 0.03 and
    0.04 -- essentially no real separation between clusters at any K
    (above ~0.5 is considered genuinely separated; near 0 means the
    cluster boundaries are arbitrary, not a real seam in the data).

    In other words: in this embedding space, the 11,800 stories don't
    form distinct discrete "shapes" -- they form one continuous cloud
    with soft variation, pulled toward a single dominant center. Forcing
    a cluster count onto that and reporting "K shapes" would have been a
    more confident claim than the data actually supports.

    This version drops clustering entirely. Instead it measures, for
    every story, its cosine similarity to the single global average
    embedding across all 11,800 stories -- a continuous "how close to
    the default story is this" score that doesn't require pretending
    there are discrete categories. It also measures, per country, how
    tightly that country's own 50 stories agree with each other,
    independent of whether they're close to the global default or not.

WHAT THIS DOES
    1. Reads every country's *_summaries file (falls back to the first
       part of *_stories if a summary is missing or empty).
    2. Embeds each story's text with a local sentence-transformer model.
    3. Reduces the embeddings to 2D with UMAP, for plotting -- this stays
       a continuous similarity map, not a set of discrete clusters.
    4. Computes the global centroid (the average embedding across every
       story in the dataset) and, for every story, its cosine similarity
       to that centroid -- the genericness score.
    5. For each country, computes:
         - genericness_score: average similarity of that country's 50
           stories to the GLOBAL centroid (high = very close to the
           generic default story; low = distinctive)
         - consistency_score: average similarity of that country's 50
           stories to their OWN country centroid (high = the model tells
           this country basically the same story every time; low = the
           model is more varied for this country)
         - distinctive_terms: words unusually common in this country's
           stories compared to the rest of the corpus (TF-IDF contrast)
         - a 2D centroid position, for the overview scatter
    6. Also writes the dataset's overall top vocabulary, as a baseline
       reference for "what the default story tends to contain."

OUTPUTS (in ./explorer_data/)
    stories_embedded.json   one point per story: story_id, country_code, x, y, genericness
    country_metrics.json    one row per country: genericness/consistency scores, centroid, distinctive terms
    global_vocabulary.json  top terms across the whole corpus (the "default story" baseline)

USAGE
    pip install pandas numpy scikit-learn sentence-transformers umap-learn
    1. Set DATA_ROOT below (same dataset folder you used for Phase 1).
    2. Set COUNTRIES_JSON to point at the countries.json Phase 1 produced.
    3. Run: python build_narrative_clusters.py

    Reuses the embedding cache from the earlier version if present
    (explorer_data/_embeddings_cache.npz), so this should run in well
    under a minute on a rerun -- no clustering or K-scan happening anymore.

A NOTE FOR YOUR REPORT
    Keep the console output and k_selection.json from your earlier run --
    that flat silhouette curve (0.03-0.04 across every K from 4 to 20) is
    the actual evidence for why this approach was abandoned. It's worth
    citing directly rather than just asserting "there were no clusters."
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- CONFIG: edit these ----------
DATA_ROOT = Path("./gpt-stories")
COUNTRIES_JSON = Path("./dashboard_data/countries.json")
FILE_EXT = "csv"
OUTPUT_DIR = Path("./explorer_data")
EMBEDDING_CACHE = Path("./explorer_data/_embeddings_cache.npz")
MAX_SUMMARY_FALLBACK_CHARS = 800  # used only if a story has no summary
TOP_N_TERMS = 8                   # distinctive terms kept per country
TOP_N_GLOBAL_TERMS = 25           # terms kept for the global vocabulary baseline
# -----------------------------------------


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if FILE_EXT == "tsv" else ","
    return pd.read_csv(path, sep=sep)


def find_country_files(data_root: Path, suffix: str) -> dict:
    pattern = f"*_{suffix}.{FILE_EXT}"
    found = {}
    for path in data_root.rglob(pattern):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        code = path.stem.split(f"_{suffix}")[0]
        found[code] = path
    return found


def load_story_texts() -> pd.DataFrame:
    """One row per story: story_id, country_code, text (summary, or a fallback excerpt)."""
    summary_files = find_country_files(DATA_ROOT, "summaries")
    stories_files = find_country_files(DATA_ROOT, "stories")

    rows = []
    for code, path in stories_files.items():
        try:
            stories_df = read_table(path)
        except Exception as e:
            print(f"  [warn] couldn't read stories file for {code}: {e}")
            continue

        summaries_df = None
        if code in summary_files:
            try:
                summaries_df = read_table(summary_files[code]).set_index("Story_ID")
            except Exception as e:
                print(f"  [warn] couldn't read summaries file for {code}: {e}")

        for _, story_row in stories_df.iterrows():
            story_id = story_row["Story_ID"]
            text = None
            if summaries_df is not None and story_id in summaries_df.index:
                summary_val = summaries_df.loc[story_id, "Summaries"]
                if isinstance(summary_val, str) and summary_val.strip():
                    text = summary_val.strip()
            if text is None:
                story_text = str(story_row.get("Story", ""))
                text = story_text[:MAX_SUMMARY_FALLBACK_CHARS].strip()
            if text:
                rows.append({"story_id": story_id, "country_code": code, "text": text})

    return pd.DataFrame(rows)


def embed_texts(texts: list) -> np.ndarray:
    """Returns unit-length embeddings (normalize_embeddings=True), so cosine
    similarity between any two rows is just their dot product."""
    if EMBEDDING_CACHE.exists():
        print(f"Found cached embeddings at {EMBEDDING_CACHE}, loading...")
        cached = np.load(EMBEDDING_CACHE, allow_pickle=True)
        if len(cached["texts"]) == len(texts) and (cached["texts"] == np.array(texts)).all():
            return cached["embeddings"]
        print("  cache doesn't match current texts (different dataset?) -- recomputing.")

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformer model (first run downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} stories...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(EMBEDDING_CACHE, texts=np.array(texts), embeddings=embeddings)
    return embeddings


def group_distinctive_terms(texts: list, group_ids: list, top_n: int = 8) -> dict:
    """
    For each distinct value in group_ids, finds the words/phrases most
    unusually common in that group's texts compared to every OTHER
    group's texts, via TF-IDF contrast. Used here per-country (no
    clustering assumption involved -- countries are a real, given
    grouping, not something the algorithm had to infer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    group_ids = np.array(group_ids)
    labels = {}
    for g in pd.unique(group_ids):
        mask = group_ids == g
        if mask.sum() == 0:
            labels[g] = []
            continue
        group_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        rest_mean = np.asarray(tfidf[~mask].mean(axis=0)).ravel()
        distinctiveness = group_mean - rest_mean
        top_idx = distinctiveness.argsort()[::-1][:top_n]
        labels[g] = terms[top_idx].tolist()
    return labels


def top_corpus_terms(texts: list, top_n: int = 25) -> list:
    """Most common words/phrases across the WHOLE corpus -- a baseline
    reference for what the 'default story' tends to contain."""
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2), min_df=3)
    counts = vectorizer.fit_transform(texts)
    totals = np.asarray(counts.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = totals.argsort()[::-1][:top_n]
    return [{"term": str(terms[i]), "count": int(totals[i])} for i in top_idx]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading story texts (summaries, falling back to story excerpts)...")
    df = load_story_texts()
    print(f"  -> {len(df)} stories loaded across {df['country_code'].nunique()} countries")

    if COUNTRIES_JSON.exists():
        countries = json.loads(COUNTRIES_JSON.read_text())
    else:
        print(f"  [warn] {COUNTRIES_JSON} not found -- country names will fall back to codes")
        countries = {}

    print("\nEmbedding stories...")
    embeddings = embed_texts(df["text"].tolist())

    print("\nReducing to 2D with UMAP (continuous similarity map, not discrete clusters)...")
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)
    df["x"] = coords[:, 0].astype(np.float64)
    df["y"] = coords[:, 1].astype(np.float64)

    print("\nMeasuring distance from the global 'default story'...")
    global_centroid = embeddings.mean(axis=0)
    global_centroid = global_centroid / np.linalg.norm(global_centroid)
    df["genericness"] = (embeddings @ global_centroid).astype(np.float64).round(4)  # cosine similarity (embeddings are unit vectors)

    print("Measuring each country's internal consistency...")
    self_similarity = np.zeros(len(df))
    for code, group in df.groupby("country_code"):
        idx = group.index.to_numpy()
        country_embeddings = embeddings[idx]
        country_centroid = country_embeddings.mean(axis=0)
        norm = np.linalg.norm(country_centroid)
        if norm > 0:
            country_centroid = country_centroid / norm
        self_similarity[idx] = country_embeddings @ country_centroid
    df["self_similarity"] = self_similarity.round(4)

    print("\nExtracting distinctive vocabulary per country...")
    distinctive = group_distinctive_terms(df["text"].tolist(), df["country_code"].tolist(), TOP_N_TERMS)

    print("Extracting overall corpus vocabulary (the 'default story' baseline)...")
    global_terms = top_corpus_terms(df["text"].tolist(), top_n=TOP_N_GLOBAL_TERMS)

    print("\nAggregating per-country metrics...")
    country_rows = []
    for code, group in df.groupby("country_code"):
        meta = countries.get(code, {})
        country_rows.append({
            "country_code": code,
            "country_name": meta.get("name", code),
            "demonym": meta.get("demonym", ""),
            "story_count": int(len(group)),
            "genericness_score": round(100 * float(group["genericness"].mean()), 1),
            "consistency_score": round(100 * float(group["self_similarity"].mean()), 1),
            "distinctive_terms": distinctive.get(code, []),
            "centroid_x": float(group["x"].mean()),
            "centroid_y": float(group["y"].mean()),
        })

    print("\nWriting outputs...")
    stories_out = df[["story_id", "country_code", "x", "y", "genericness"]].copy()
    stories_out["x"] = stories_out["x"].round(3)
    stories_out["y"] = stories_out["y"].round(3)
    stories_out.to_json(OUTPUT_DIR / "stories_embedded.json", orient="records")
    print(f"  -> stories_embedded.json ({len(stories_out)} rows)")

    with open(OUTPUT_DIR / "country_metrics.json", "w") as f:
        json.dump(sorted(country_rows, key=lambda r: -r["genericness_score"]), f, indent=2)
    print(f"  -> country_metrics.json ({len(country_rows)} countries)")

    with open(OUTPUT_DIR / "global_vocabulary.json", "w") as f:
        json.dump(global_terms, f, indent=2)
    print(f"  -> global_vocabulary.json ({len(global_terms)} terms)")

    most_generic = max(country_rows, key=lambda r: r["genericness_score"])
    most_distinctive = min(country_rows, key=lambda r: r["genericness_score"])
    most_consistent = max(country_rows, key=lambda r: r["consistency_score"])
    least_consistent = min(country_rows, key=lambda r: r["consistency_score"])
    print(f"\nClosest to the default story: {most_generic['country_name']} "
          f"(genericness {most_generic['genericness_score']})")
    print(f"Furthest from the default story: {most_distinctive['country_name']} "
          f"(genericness {most_distinctive['genericness_score']})")
    print(f"Most internally consistent: {most_consistent['country_name']} "
          f"(consistency {most_consistent['consistency_score']})")
    print(f"Least internally consistent: {least_consistent['country_name']} "
          f"(consistency {least_consistent['consistency_score']})")


if __name__ == "__main__":
    main()