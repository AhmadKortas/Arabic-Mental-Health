# Arabic Mental Health — Analysis Code

Code to reproduce the analysis in our paper on help-seeking orientation in
Arabic mental health discourse on Facebook.

The script computes the statistics reported in the paper and generates the
figures. It covers the orientation distribution and engagement analysis (RQ2)
and the reliability analysis against the human gold standard (RQ3).

## Data

The corpus was collected through CrowdTangle, whose access is now restricted, so
the raw posts cannot be shared here. The script expects the annotated files in
the same folder:

- `annotated_fanar_10kposts_Mentalhealth.csv` — the 10,000-post Fanar-annotated sample
- `Mental Health.xlsx - Sheet1.csv` — the 1,000-post human-annotated gold standard

## Setup

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

## Running

```bash
python analysis.py
```

The statistics are printed to the console and the figures are written to a
`figures/` folder.

## Output

- Orientation distribution across the six categories
- Median engagement (interactions, comments, shares) by orientation, with a
  Kruskal–Wallis test and follower-normalised check
- Pairwise comparisons with effect sizes
- Reaction-mix composition by orientation
- Agreement between Fanar and the human annotator (Cohen's Kappa), the confusion
  matrix, and per-category precision and recall
