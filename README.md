# Predicting Earthquakes Using 200 Years of Global Major Earthquakes

**Goal:** Predict earthquake magnitude category (5–5.9, 6–6.9, 7+) from location, depth, and time‑based features using historical data (1826–2026).

**Dataset:** 200 Years of Global Major Earthquakes (1826–2026) from Kaggle.
- https://www.kaggle.com/datasets/dhrubangtalukdar/200-years-of-global-major-earthquakes-18262026

## Setup

```sh
# Activating the conda environment
conda env create -f environment.yml
conda activate quake-ml

# Generate the processed dataset
python src/data_prep.py

# Start Jupyter from the repo root
jupyter notebook
```

## Results

### Best Model: Gradient Boosting

- Accuracy: 0.88
- Macro F1: 0.47
- Weighted F1: 0.86

### Analysis

- Temporal features were most important.
- Location and depth contributed moderately.
- Rare high‑magnitude classes (7+) remain harder to predict due to imbalance.

## Permutation Importance (Gradient Boosting)

We used permutation importance to identify which features most influenced magnitude predictions.

![Permutation importance (Gradient Boosting)](charts/gb_perm_importance.png)
