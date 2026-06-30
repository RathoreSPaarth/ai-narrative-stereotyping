"""
Quick spot-check: pulls real story summaries for chosen countries so you can
manually read what's behind the genericness numbers before trusting them.

For each country code below, prints its LEAST generic (most distinctive),
MEDIAN, and MOST generic story -- the actual range, not a random sample.

USAGE
    Set COUNTRIES_TO_CHECK to the country codes you want (check
    dashboard_data/countries.json if you're not sure of a code -- it's
    keyed by code, e.g. "US", "MG", "CN").
    python inspect_stories.py
"""
import json
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("./gpt-stories")
EXPLORER_DATA = Path("./explorer_data")
FILE_EXT = "csv"
COUNTRIES_TO_CHECK = ["US", "MG", "CN"]


def find_country_file(code, suffix):
    matches = list(DATA_ROOT.rglob(f"{code}_{suffix}.{FILE_EXT}"))
    return matches[0] if matches else None


def load_summaries(code):
    path = find_country_file(code, "summaries")
    if path is None:
        print(f"  [warn] no summaries file found for {code}")
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["Story_ID"], df["Summaries"]))


def main():
    stories = pd.DataFrame(json.loads((EXPLORER_DATA / "stories_embedded.json").read_text()))

    for code in COUNTRIES_TO_CHECK:
        print(f"\n{'=' * 70}\n{code}\n{'=' * 70}")
        summaries = load_summaries(code)
        country_stories = stories[stories["country_code"] == code].sort_values("genericness")
        if country_stories.empty:
            print("  no stories found for this code -- check it matches countries.json")
            continue

        picks = {
            "LEAST generic (most distinctive)": country_stories.iloc[0],
            "MEDIAN": country_stories.iloc[len(country_stories) // 2],
            "MOST generic (closest to default)": country_stories.iloc[-1],
        }
        for label, row in picks.items():
            text = summaries.get(row["story_id"], "[summary text not found]")
            print(f"\n--- {label} | story_id={row['story_id']} | genericness={row['genericness']:.3f} ---")
            print(text)


if __name__ == "__main__":
    main()