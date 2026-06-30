"""
Regional template analysis: does gpt-4o-mini assign different narrative
templates by world region, or is it a single global default?

BACKGROUND
    Earlier analysis showed that the embedding space has no meaningful cluster
    structure (silhouette scores 0.03-0.04 across K=4..20), so a KMeans
    partition into "shapes" was not defensible. However, manual inspection of
    stories across US, Madagascar, and China suggested two distinct narrative
    registers:
      - Americas/Europe: realist homecoming, small town, grief, healing
      - Africa/Asia:     folkloric quest, village, sacred nature, guide figure

    This script tests that observation rigorously using UN M49 geographic
    regions as an EXTERNAL, PRE-DEFINED grouping (not inferred from the
    embeddings), which avoids the circularity of discovering groups from the
    same data you then use to characterize them.

METHOD
    For each UN M49 region, compute a REGIONAL CENTROID -- the average
    embedding of all stories prompted with a country from that region.
    Then, for every story, compute its cosine similarity to EACH regional
    centroid. This gives five affinity scores per story (one per region),
    and the highest one names the region whose narrative template this
    story most resembles.

    Key question: does a story's highest-affinity region match its OWN
    prompted region, or does it consistently point to a different one?

    Confusion matrix (region x region): rows = prompted region,
    columns = most-similar template region. A diagonal matrix would mean
    each region gets its own distinct template. Off-diagonal concentration
    would mean systematic template imposition.

    Regional vocabulary: TF-IDF contrast per region shows what each
    regional template actually talks about -- without any human labeling.

IMPORTANT LIMITATION
    This analysis identifies GEOGRAPHIC CORRELATION, not causal mechanism.
    The model may be responding to implied language, economic status,
    colonial history, or other correlates of geography rather than
    geography per se. Causal claims would require further controlled
    prompting experiments (e.g., same country with different implied
    linguistic context), which are outside the scope of this dataset.

OUTPUTS  (in ./explorer_data/)
    regional_affinities.json   per-country aggregated affinity scores +
                               template match/mismatch flag
    stories_regional.json      per-story affinity to each region centroid
    region_vocab.json          distinctive vocabulary per regional template
    confusion_matrix.json      region x region story-count cross-tabulation

USAGE
    python build_regional_analysis.py
    Reads the existing embedding cache; no re-embedding needed.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- CONFIG: same defaults as build_narrative_clusters.py ----------
DATA_ROOT    = Path("./gpt-stories")
COUNTRIES_JSON = Path("./dashboard_data/countries.json")
FILE_EXT     = "csv"
OUTPUT_DIR   = Path("./explorer_data")
EMBEDDING_CACHE = Path("./explorer_data/_embeddings_cache.npz")
TOP_N_REGION_TERMS = 15
# --------------------------------------------------------------------------


# ── UN M49 regional classification ────────────────────────────────────────
# Source: UN Statistics Division M49 standard.
# Covers all 236 countries in the dataset plus dependent territories.
# Antarctica is assigned "Polar" rather than left unclassified.
UN_M49_REGIONS = {
    # Africa (57)
    "DZ":"Africa","AO":"Africa","BJ":"Africa","BW":"Africa","BF":"Africa",
    "BI":"Africa","CV":"Africa","CM":"Africa","CF":"Africa","TD":"Africa",
    "KM":"Africa","CG":"Africa","CD":"Africa","CI":"Africa","DJ":"Africa",
    "EG":"Africa","GQ":"Africa","ER":"Africa","SZ":"Africa","ET":"Africa",
    "GA":"Africa","GM":"Africa","GH":"Africa","GN":"Africa","GW":"Africa",
    "KE":"Africa","LS":"Africa","LR":"Africa","LY":"Africa","MG":"Africa",
    "MW":"Africa","ML":"Africa","MR":"Africa","MU":"Africa","MA":"Africa",
    "MZ":"Africa","NA":"Africa","NE":"Africa","NG":"Africa","RW":"Africa",
    "ST":"Africa","SN":"Africa","SL":"Africa","SO":"Africa","ZA":"Africa",
    "SS":"Africa","SD":"Africa","TZ":"Africa","TG":"Africa","TN":"Africa",
    "UG":"Africa","ZM":"Africa","ZW":"Africa",
    "RE":"Africa","YT":"Africa","SH":"Africa","EH":"Africa","SC":"Africa","BM":"Americas",
    # Americas (54)
    "AG":"Americas","AR":"Americas","BS":"Americas","BB":"Americas",
    "BZ":"Americas","BO":"Americas","BR":"Americas","CA":"Americas",
    "CL":"Americas","CO":"Americas","CR":"Americas","CU":"Americas",
    "DM":"Americas","DO":"Americas","EC":"Americas","SV":"Americas",
    "GD":"Americas","GT":"Americas","GY":"Americas","HT":"Americas",
    "HN":"Americas","JM":"Americas","MX":"Americas","NI":"Americas",
    "PA":"Americas","PY":"Americas","PE":"Americas","KN":"Americas",
    "LC":"Americas","VC":"Americas","SR":"Americas","TT":"Americas",
    "US":"Americas","UY":"Americas","VE":"Americas",
    "PR":"Americas","VI":"Americas","VG":"Americas","TC":"Americas",
    "KY":"Americas","AI":"Americas","MS":"Americas","FK":"Americas",
    "GF":"Americas","GP":"Americas","MQ":"Americas","MF":"Americas",
    "BL":"Americas","PM":"Americas","AW":"Americas","CW":"Americas",
    "SX":"Americas","BQ":"Americas","GL":"Americas",
    # Asia – incl. Middle East & Central Asia (51)
    "AF":"Asia","AM":"Asia","AZ":"Asia","BH":"Asia","BD":"Asia",
    "BT":"Asia","BN":"Asia","MM":"Asia","KH":"Asia","CN":"Asia",
    "CY":"Asia","TL":"Asia","GE":"Asia","IN":"Asia","ID":"Asia",
    "IR":"Asia","IQ":"Asia","IL":"Asia","JP":"Asia","JO":"Asia",
    "KZ":"Asia","KW":"Asia","KG":"Asia","LA":"Asia","LB":"Asia",
    "MY":"Asia","MV":"Asia","MN":"Asia","NP":"Asia","KP":"Asia",
    "OM":"Asia","PK":"Asia","PS":"Asia","PH":"Asia","QA":"Asia",
    "SA":"Asia","SG":"Asia","KR":"Asia","LK":"Asia","SY":"Asia",
    "TW":"Asia","TJ":"Asia","TH":"Asia","TR":"Asia","TM":"Asia",
    "AE":"Asia","UZ":"Asia","VN":"Asia","YE":"Asia",
    "HK":"Asia","MO":"Asia",
    # Europe (51)
    "AL":"Europe","AD":"Europe","AT":"Europe","BY":"Europe","BE":"Europe",
    "BA":"Europe","BG":"Europe","HR":"Europe","CZ":"Europe","DK":"Europe",
    "EE":"Europe","FI":"Europe","FR":"Europe","DE":"Europe","GR":"Europe",
    "HU":"Europe","IS":"Europe","IE":"Europe","IT":"Europe","XK":"Europe",
    "LV":"Europe","LI":"Europe","LT":"Europe","LU":"Europe","MT":"Europe",
    "MD":"Europe","MC":"Europe","ME":"Europe","NL":"Europe","MK":"Europe",
    "NO":"Europe","PL":"Europe","PT":"Europe","RO":"Europe","RU":"Europe",
    "SM":"Europe","RS":"Europe","SK":"Europe","SI":"Europe","ES":"Europe",
    "SE":"Europe","CH":"Europe","UA":"Europe","GB":"Europe","VA":"Europe",
    "GI":"Europe","JE":"Europe","GG":"Europe","IM":"Europe",
    "FO":"Europe","AX":"Europe",
    # Oceania (27)
    "AU":"Oceania","FJ":"Oceania","KI":"Oceania","MH":"Oceania",
    "FM":"Oceania","NR":"Oceania","NZ":"Oceania","PW":"Oceania",
    "PG":"Oceania","WS":"Oceania","SB":"Oceania","TO":"Oceania",
    "TV":"Oceania","VU":"Oceania",
    "PF":"Oceania","NC":"Oceania","WF":"Oceania","GU":"Oceania",
    "MP":"Oceania","AS":"Oceania","CK":"Oceania","NU":"Oceania",
    "TK":"Oceania","NF":"Oceania","CC":"Oceania","CX":"Oceania",
    "PN":"Oceania",
    # Polar
    "AQ":"Polar",
}
# ──────────────────────────────────────────────────────────────────────────


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if FILE_EXT == "tsv" else ","
    return pd.read_csv(path, sep=sep)


def find_country_files(data_root: Path, suffix: str) -> dict:
    found = {}
    for path in data_root.rglob(f"*_{suffix}.{FILE_EXT}"):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        code = path.stem.split(f"_{suffix}")[0]
        found[code] = path
    return found


def load_story_texts() -> pd.DataFrame:
    """Must use IDENTICAL ordering logic as build_narrative_clusters.py so
    that row i in the embedding cache corresponds to row i here."""
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
                text = str(story_row.get("Story", ""))[:800].strip()
            if text:
                rows.append({"story_id": story_id, "country_code": code, "text": text})
    return pd.DataFrame(rows)


def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    if not EMBEDDING_CACHE.exists():
        raise FileNotFoundError(
            f"Embedding cache not found at {EMBEDDING_CACHE}.\n"
            "Run build_narrative_clusters.py first to generate it."
        )
    print(f"Loading embedding cache from {EMBEDDING_CACHE}...")
    cached = np.load(EMBEDDING_CACHE, allow_pickle=True)
    cache_texts = cached["texts"]
    current_texts = np.array(df["text"].tolist())
    if len(cache_texts) != len(current_texts):
        raise ValueError(
            f"Cache has {len(cache_texts)} embeddings but dataset has "
            f"{len(current_texts)} stories. Delete the cache and rerun "
            "build_narrative_clusters.py to rebuild it."
        )
    mismatches = (cache_texts != current_texts).sum()
    if mismatches > 0:
        raise ValueError(
            f"{mismatches} story texts don't match the cache. "
            "The dataset may have changed. Delete the cache and rerun "
            "build_narrative_clusters.py."
        )
    return cached["embeddings"].astype(np.float64)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def region_tfidf_vocab(texts: list, region_ids: list, top_n: int) -> dict:
    """Distinctive words per region via TF-IDF contrast."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=8000,
        ngram_range=(1, 2), min_df=3
    )
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    region_ids = np.array(region_ids)
    vocab = {}
    for region in sorted(set(region_ids)):
        mask = region_ids == region
        group_mean = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        rest_mean = np.asarray(tfidf[~mask].mean(axis=0)).ravel()
        distinctiveness = group_mean - rest_mean
        top_idx = distinctiveness.argsort()[::-1][:top_n]
        vocab[region] = [
            {"term": str(terms[i]), "score": round(float(distinctiveness[i]), 4)}
            for i in top_idx
        ]
    return vocab


def build_confusion_matrix(df: pd.DataFrame, regions: list) -> dict:
    """
    Counts how many stories from each prompted_region were most similar
    to each template_region. Normalized to row percentages.
    """
    raw = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        raw[row["prompted_region"]][row["most_similar_region"]] += 1

    result = {}
    for prompted in sorted(raw.keys()):
        total = sum(raw[prompted].values())
        result[prompted] = {
            template: {
                "count": raw[prompted][template],
                "pct": round(100 * raw[prompted][template] / total, 1)
            }
            for template in sorted(regions)
        }
        result[prompted]["_total_stories"] = total
    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading story texts...")
    df = load_story_texts()
    print(f"  -> {len(df)} stories, {df['country_code'].nunique()} countries")

    print("Loading UN M49 regional assignments...")
    unmapped = []
    df["prompted_region"] = df["country_code"].map(UN_M49_REGIONS)
    unmapped_codes = df[df["prompted_region"].isna()]["country_code"].unique()
    if len(unmapped_codes) > 0:
        print(f"  [warn] {len(unmapped_codes)} country codes have no UN M49 mapping:")
        for code in sorted(unmapped_codes):
            print(f"    {code}")
        print("  These will be excluded from regional analysis.")
        df = df[df["prompted_region"].notna()].copy()

    regions = sorted(df["prompted_region"].unique())
    print(f"  -> Regions in dataset: {regions}")

    print("\nLoading embeddings from cache...")
    # Note: load against the FULL df before filtering unmapped, to preserve
    # cache alignment -- then re-filter with the same index
    df_full = load_story_texts()
    embeddings_full = load_embeddings(df_full)
    # Re-apply the region filter using index alignment
    df_full["prompted_region"] = df_full["country_code"].map(UN_M49_REGIONS)
    mask = df_full["prompted_region"].notna()
    df = df_full[mask].copy()
    embeddings = embeddings_full[mask.values]
    print(f"  -> {len(embeddings)} stories retained after unmapped exclusion")

    print("\nComputing regional centroids...")
    regional_centroids = {}
    region_story_counts = {}
    for region in regions:
        idx = (df["prompted_region"] == region).values
        region_embeddings = embeddings[idx]
        centroid = unit(region_embeddings.mean(axis=0))
        regional_centroids[region] = centroid
        region_story_counts[region] = int(idx.sum())
        print(f"  {region:10s}: {idx.sum()} stories, centroid norm={np.linalg.norm(centroid):.3f}")

    print("\nComputing per-story affinity to each regional centroid...")
    affinity_cols = {}
    for region in regions:
        sim = embeddings @ regional_centroids[region]
        affinity_cols[f"affinity_{region}"] = sim
        df[f"affinity_{region}"] = sim

    df["most_similar_region"] = df[[f"affinity_{r}" for r in regions]].idxmax(axis=1).str.replace("affinity_", "")
    df["template_match"] = df["prompted_region"] == df["most_similar_region"]

    print(f"\n  Overall template match rate: "
          f"{df['template_match'].mean()*100:.1f}% of stories "
          f"most resemble their own region's template")
    for region in regions:
        sub = df[df["prompted_region"] == region]
        match_rate = sub["template_match"].mean() * 100
        print(f"  {region:10s}: {match_rate:.1f}% match ({len(sub)} stories)")

    print("\nExtracting regional vocabulary (this takes a moment)...")
    vocab = region_tfidf_vocab(
        df["text"].tolist(), df["prompted_region"].tolist(), TOP_N_REGION_TERMS
    )

    print("\nBuilding confusion matrix...")
    confusion = build_confusion_matrix(df, regions)

    print("\nAggregating per-country metrics...")
    if COUNTRIES_JSON.exists():
        countries_meta = json.loads(COUNTRIES_JSON.read_text())
    else:
        countries_meta = {}

    country_rows = []
    for code, group in df.groupby("country_code"):
        meta = countries_meta.get(code, {})
        region = UN_M49_REGIONS.get(code, "Unknown")
        avg_affinities = {
            region_name: round(float(group[f"affinity_{region_name}"].mean()), 4)
            for region_name in regions
        }
        most_similar = max(avg_affinities, key=avg_affinities.get)
        template_match_pct = round(float(group["template_match"].mean()) * 100, 1)
        country_rows.append({
            "country_code": code,
            "country_name": meta.get("name", code),
            "demonym": meta.get("demonym", ""),
            "prompted_region": region,
            "most_similar_region": most_similar,
            "template_match": most_similar == region,
            "template_match_pct": template_match_pct,
            "avg_affinities": avg_affinities,
        })

    country_rows.sort(key=lambda r: (r["prompted_region"], -r["template_match_pct"]))

    print("\nWriting outputs...")
    # Per-story
    story_cols = ["story_id", "country_code", "prompted_region", "most_similar_region", "template_match"] + \
                 [f"affinity_{r}" for r in regions]
    stories_out = df[story_cols].copy()
    for col in [f"affinity_{r}" for r in regions]:
        stories_out[col] = stories_out[col].round(4)
    stories_out.to_json(OUTPUT_DIR / "stories_regional.json", orient="records")
    print(f"  -> stories_regional.json ({len(stories_out)} rows)")

    with open(OUTPUT_DIR / "regional_affinities.json", "w") as f:
        json.dump(country_rows, f, indent=2)
    print(f"  -> regional_affinities.json ({len(country_rows)} countries)")

    with open(OUTPUT_DIR / "region_vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"  -> region_vocab.json ({len(vocab)} regions)")

    with open(OUTPUT_DIR / "confusion_matrix.json", "w") as f:
        json.dump(confusion, f, indent=2)
    print(f"  -> confusion_matrix.json")

    print("\n── CONFUSION MATRIX (% of stories from each region) ──────────────")
    header = f"{'Prompted':12s}" + "".join(f"{r:12s}" for r in regions)
    print(header)
    print("-" * len(header))
    for prompted in sorted(confusion.keys()):
        row_str = f"{prompted:12s}"
        for template in regions:
            pct = confusion[prompted].get(template, {}).get("pct", 0)
            marker = " *" if template == prompted else "  "
            row_str += f"{pct:>8.1f}%{marker}"
        print(row_str)
    print("\n* = diagonal (own region's template)")

    print("\n── REGIONAL VOCABULARY (top 5 distinctive terms per region) ──────")
    for region in regions:
        terms = [t["term"] for t in vocab[region][:5]]
        print(f"  {region:10s}: {', '.join(terms)}")


if __name__ == "__main__":
    main()