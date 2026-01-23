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

## Analysis

- Temporal features were most important.
- Location and depth contributed moderately.
- Rare high‑magnitude classes (7+) remain harder to predict due to imbalance.

### Exploratory Data Analysis (EDA)

Run:

```sh
python src/eda.py
```

#### Magnitude Category Distribution

![Magnitude category distribution](charts/eda_mag_category.png)

- High class imbalance: 5-5.9 magnitude earthquakes are most prevalent.
- 7+ magnitude rarity causes difficult recall.
- Macro F1 is more informative than accuracy given the skew.

#### Depth Distribution

![Depth distribution](charts/eda_depth_distribution.png)

- Depths between ~0-100 km are more prevalent than ~200 km+.
- The distribution is right-skewed with a long tail to ~700 km.

#### Spatial Distribution by Magnitude

![Spatial distribution by magnitude](charts/eda_spatial_scatter.png)

- Events cluster in low-to-mid latitudes.
- Coverage spans most longitudes, so location features are meaningful.
- Spatial clustering reflect tectonic boundaries.

### Model‑Guided Analysis

#### Permutation Importance (Gradient Boosting)

We used permutation importance to identify which features most influenced magnitude predictions.

![Permutation importance (Gradient Boosting)](charts/gb_perm_importance.png)

- Year ranks highest in this model.
- Location and depth contributions are less impactful than the year.
- Temporal dominance may reflect technology improvements in later years.

#### Confusion Matrix (Gradient Boosting)

![Normalized Confusion Matrix (Gradient Boosting)](charts/gb_confusion_matrix.png)

- The 5-5.9 class is predicted correctly most of the time (~0.99 recall).
- The 6-6.9 class is often predicted as 5-5.9, so recall is low (~0.26).
- The 7+ class has very low recall (~0.05), reflecting class imbalance.
