# Arabic Mental Health — Analysis Code

Code to reproduce the analysis in our paper on help-seeking orientation in
Arabic mental health discourse on Facebook.

## Scripts

- `analysis_pipeline.py` — the full pipeline. Running it end-to-end reproduces
  every statistic in the paper and regenerates all figures, covering the corpus
  characterisation (RQ1), the orientation and engagement analysis (RQ2), and the
  reliability analysis against the human gold standard (RQ3).
- `mental_health_analysis.py` — the earlier, partial script covering RQ2 and RQ3
  only. Kept for reference; `analysis_pipeline.py` supersedes it.

## Data

The corpus was collected through CrowdTangle, whose access is now restricted, so
the raw posts cannot be shared here. The scripts expect these files in the same
folder:

For RQ1 (raw corpus, used by `analysis_pipeline.py`):

- the three collection-stream CSVs (D1, D2, D3) as exported from CrowdTangle

For RQ2 and RQ3 (annotated data):

- `annotated_fanar_10kposts_Mentalhealth.csv` — the 10,000-post Fanar-annotated sample
- `Mental Health.xlsx - Sheet1.csv` — the 1,000-post human-annotated gold standard

## Setup

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

## Running

Full reproduction (RQ1 + RQ2 + RQ3):

```bash
python analysis_pipeline.py
```

RQ2 and RQ3 only:

```bash
python mental_health_analysis.py
```

The statistics are printed to the console and the figures are written to a
`figures/` folder.

## Output

- RQ1: corpus size and per-stream split, duplication, text/language rates,
  annual posting counts, account-type distribution, geography, post formats, and
  engagement (median/mean and per-stream medians)
- RQ2: orientation distribution across the six categories; median engagement by
  orientation with a Kruskal–Wallis test and follower-normalised check; pairwise
  comparisons with effect sizes; reaction-mix composition by orientation
- RQ3: agreement between Fanar and the human annotator (Cohen's Kappa), the
  confusion matrix, and per-category precision and recall
