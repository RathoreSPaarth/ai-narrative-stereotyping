"""
Phase 1: Data Prep for the Narrative Dashboard
------------------------------------------------
Aggregates per-country files (sentiments, word_freq, noun_phrases, stories)
across all 236 countries into a handful of clean, combined datasets that
Observable Plot can read directly.

USAGE
    1. Set DATA_ROOT below to point at the folder containing your dataset.
       It will work whether your files are:
         (a) nested in one subfolder per country code, e.g.
             data/AD/AD_sentiments.csv
         (b) or all sitting flat in one folder, e.g.
             data/AD_sentiments.csv, data/AF_sentiments.csv, ...
    2. If your files are .tsv or .xlsx instead of .csv, change FILE_EXT.
    3. Run:  python prepare_dashboard_data.py
    4. Outputs land in ./dashboard_data/

OUTPUTS
    countries.json        -> {code: {name, demonym, iso, story_count}}
    sentiment_summary.csv -> long format: country, sentiment, count, %, avg_confidence
    word_freq_top.csv     -> top N words per country, ranked
    noun_phrase_top.csv   -> top N noun phrases per country, ranked

If a file is missing or a column name doesn't match for some country, the
script logs a warning and skips just that country rather than crashing —
check the console output at the end for any [warn] lines.
"""

import json
from pathlib import Path

import pandas as pd

# ---------- CONFIG: edit these ----------
DATA_ROOT = Path("./gpt-stories")  # <-- point this at the folder that directly
                                    #     contains the AD/, AE/, AF/... subfolders
FILE_EXT = "csv"                  # "csv", "tsv", or "xlsx"
OUTPUT_DIR = Path("./dashboard_data")
TOP_N_WORDS = 20
TOP_N_NOUN_PHRASES = 20
# -----------------------------------------


def read_table(path: Path) -> pd.DataFrame:
    if FILE_EXT == "xlsx":
        return pd.read_excel(path)
    sep = "\t" if FILE_EXT == "tsv" else ","
    return pd.read_csv(path, sep=sep)


def find_country_files(data_root: Path, suffix: str) -> dict:
    """
    Finds every file named <CODE>_<suffix>.<ext> anywhere under data_root,
    regardless of nested-per-country or flat folder layout.
    Skips macOS zip junk (__MACOSX folder and "._" AppleDouble files).
    Returns {country_code: filepath}.
    """
    pattern = f"*_{suffix}.{FILE_EXT}"
    found = {}
    for path in data_root.rglob(pattern):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        code = path.stem.split(f"_{suffix}")[0]
        found[code] = path
    return found


def build_country_lookup(stories_files: dict) -> dict:
    """Pulls Country_Name / Demonym / ISO code from each *_stories file (first row is enough)."""
    lookup = {}
    for code, path in stories_files.items():
        try:
            df = read_table(path)
            row = df.iloc[0]
            lookup[code] = {
                "code": code,
                "iso": str(row.get("ISO-3361", code)),
                "name": str(row.get("Country_Name", code)),
                "demonym": str(row.get("Demonym", "")),
                "story_count": int(len(df)),
            }
        except Exception as e:
            print(f"  [warn] couldn't read stories file for {code}: {e}")
    return lookup


def build_sentiment_summary(sentiment_files: dict, lookup: dict) -> pd.DataFrame:
    rows = []
    for code, path in sentiment_files.items():
        try:
            df = read_table(path)
            name = lookup.get(code, {}).get("name", code)
            total = len(df)
            grouped = df.groupby("sentiment").agg(
                count=("sentiment", "size"),
                avg_confidence=("confidence", "mean"),
            ).reset_index()
            for _, r in grouped.iterrows():
                rows.append({
                    "country_code": code,
                    "country_name": name,
                    "sentiment": r["sentiment"],
                    "count": int(r["count"]),
                    "percentage": round(100 * r["count"] / total, 2) if total else 0,
                    "avg_confidence": round(float(r["avg_confidence"]), 3),
                })
        except Exception as e:
            print(f"  [warn] sentiment file failed for {code}: {e}")
    return pd.DataFrame(rows)


def build_top_terms(term_files: dict, lookup: dict, term_col: str, count_col: str,
                     top_n: int, label: str) -> pd.DataFrame:
    rows = []
    for code, path in term_files.items():
        try:
            df = read_table(path)
            name = lookup.get(code, {}).get("name", code)
            df = df.sort_values(count_col, ascending=False).head(top_n)
            for rank, (_, r) in enumerate(df.iterrows(), start=1):
                rows.append({
                    "country_code": code,
                    "country_name": name,
                    "term": r[term_col],
                    "count": r[count_col],
                    "rank": rank,
                })
        except Exception as e:
            print(f"  [warn] {label} file failed for {code}: {e}")
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Scanning for files...")
    stories_files = find_country_files(DATA_ROOT, "stories")
    sentiment_files = find_country_files(DATA_ROOT, "sentiments")
    word_freq_files = find_country_files(DATA_ROOT, "word_freq")
    noun_phrase_files = find_country_files(DATA_ROOT, "noun_phrases")

    print(f"Found {len(stories_files)} countries with stories files.")
    print(f"Found {len(sentiment_files)} countries with sentiment files.")
    print(f"Found {len(word_freq_files)} countries with word_freq files.")
    print(f"Found {len(noun_phrase_files)} countries with noun_phrase files.")

    print("\nBuilding country lookup table...")
    lookup = build_country_lookup(stories_files)
    with open(OUTPUT_DIR / "countries.json", "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"  -> countries.json ({len(lookup)} countries)")

    print("\nBuilding sentiment summary...")
    sentiment_df = build_sentiment_summary(sentiment_files, lookup)
    sentiment_df.to_csv(OUTPUT_DIR / "sentiment_summary.csv", index=False)
    print(f"  -> sentiment_summary.csv ({len(sentiment_df)} rows)")

    print("\nBuilding top word frequencies...")
    word_df = build_top_terms(word_freq_files, lookup, "Word", "Frequency",
                               TOP_N_WORDS, "word_freq")
    word_df.to_csv(OUTPUT_DIR / "word_freq_top.csv", index=False)
    print(f"  -> word_freq_top.csv ({len(word_df)} rows)")

    print("\nBuilding top noun phrases...")
    noun_df = build_top_terms(noun_phrase_files, lookup, "Noun Phrase", "Count",
                               TOP_N_NOUN_PHRASES, "noun_phrases")
    noun_df.to_csv(OUTPUT_DIR / "noun_phrase_top.csv", index=False)
    print(f"  -> noun_phrase_top.csv ({len(noun_df)} rows)")

    print("\nDone. Outputs in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()