# How AI Stereotypes the World: Regional Narrative Templates in LLM-Generated Stories

Code for the paper of the same name. Applies sentence-transformer embeddings and regional centroid analysis to the publicly released dataset from Rettberg & Wigers (2025) to show that AI narrative homogenisation operates through five regional stereotypes rather than one global template.

## Dataset

This repo contains no data. Download the dataset from the Harvard Dataverse:
**Rettberg, Jill Walker & Wigers, Hermann (2025). *AI-generated stories for 236 nationalities.* Harvard Dataverse.**
`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XBMKZJ`

Unzip it so you have a folder called `gpt-stories/` with one subfolder per country code (`AD/`, `AE/`, ...), each containing six CSV files.

## Requirements

```
pip install pandas numpy scikit-learn sentence-transformers umap-learn scipy
```

## How to run

Run the four scripts in order from inside the `gpt-stories/` folder (or update `DATA_ROOT` in each script to point at it):

**Step 1 — Aggregate per-country files into combined CSVs for the dashboard**
```
python scripts/prepare_dashboard_data.py
```
Outputs: `dashboard_data/`

**Step 2 — Embed all 11,800 story summaries and compute per-country metrics**
```
python scripts/build_narrative_clusters.py
```
First run downloads the embedding model (~90 MB). Outputs: `explorer_data/`
Embeddings are cached — subsequent runs skip this step.

**Step 3 — Assign UN M49 regions and compute the confusion matrix**
```
python scripts/build_regional_analysis.py
```
Outputs: `explorer_data/confusion_matrix.json`, `regional_affinities.json`, `region_vocab.json`

**Step 4 — Run chi-square significance tests**
```
python scripts/chi_square_test.py
```
Outputs: `explorer_data/chi_square_results.json` and copy-paste LaTeX text for the paper.

### Optional: spot-check individual stories
```
python scripts/inspect_stories.py
```
Edit `COUNTRIES_TO_CHECK` inside the script. Prints the least-generic, median, and most-generic story for each selected country.

## Paper

`paper/main.tex` — compile with pdflatex. You will need to supply `paper/references.bib` with the BibTeX entries for the citations in the paper.

## Dashboard

The Observable Framework dashboard lives in `dashboard/`. To run it locally:
```
cd dashboard
npm install
npm run dev
```
Copy the JSON files from `explorer_data/` and `dashboard_data/` into `dashboard/src/data/` first.
