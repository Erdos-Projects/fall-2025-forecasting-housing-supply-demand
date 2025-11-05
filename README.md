# Forecasting Canadian Housing Supply and Demand — Fall 2025

This repository develops a data-driven framework to forecast Canadian housing adequacy at the provincial level using open data from CMHC and Statistics Canada, covering the years 1990–2025.  
The work is part of the Erdős Institute Fall 2025 Data Science Bootcamp.

---

## Project Overview

The project aims to model and forecast two related outcomes:

1. **Quarterly housing starts** (number of new dwellings started per quarter), derived from CMHC’s monthly *Seasonally Adjusted Annual Rate* (SAAR) data.  
2. **Housing Adequacy Index (HAI)**, defined as the ratio between housing supply and estimated housing need.

The Housing Adequacy Index (HAI) is computed as:

\[
HAI = \frac{\text{dwellings started in a quarter}}{\Delta \text{population} / \text{AHS}}
\]

where **AHS** (Average Household Size) is assumed to be 2.5 persons per household.  
Values of HAI below 1 indicate insufficient supply, while values above 1 suggest a housing surplus relative to population growth.

---

## Repository Structure

| File or Folder | Description |
|-----------------|-------------|
| `data/` | Contains all datasets, including the processed `housing_adequacy_dataset.csv` file. |
| `utilities/` | Helper functions for feature construction, tuning, and rolling evaluation. |
| `00_data_processing.ipynb` | Converts CMHC monthly SAAR data to quarterly starts, merges with population, and computes HAI. |
| `01_eda.ipynb` | Exploratory data analysis of housing starts, population trends, and HAI variation. |
| `02_dwelling_sameQ.ipynb` | Forecasts housing starts for the same quarter next year (H = 4). |
| `03_dwelling_nextQ.ipynb` | Forecasts housing starts for the next quarter (H = 1). |
| `04_explore_hai.ipynb` | Forecasts the Housing Adequacy Index (HAI), comparing raw and smoothed versions. |
| `plotly_map.ipynb` | Generates interactive provincial heatmaps of model metrics and HAI values. |
| `best_params_cache_*.json` | Cached hyperparameters for reproducibility and faster reruns. |

---

## Methodology

### 1. Data Preparation
- Convert monthly CMHC *SAAR* data into quarterly means.  
- Merge with Statistics Canada population data.  
- Compute quarterly population change (Δpop) and estimate required dwellings as Δpop / 2.5.  
- Derive quarterly housing starts as SAAR / 4 × 1000.

### 2. Forecast Targets
- `dwelling_starts`: the number of dwellings started each quarter.  
- `hai`: the Housing Adequacy Index, representing housing supply relative to need.

### 3. Forecast Horizons
- **H = 1** → predict the next quarter.  
- **H = 4** → predict the same quarter in the next year.

### 4. Models Evaluated
- Linear Regression  
- Ridge Regression  
- Random Forest  
- Extra Trees Regressor  
- Gradient Boosting  
- XGBoost  

Model performance is compared against a **seasonal naïve baseline** using the MASE metric.

### 5. Evaluation Metrics
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Symmetric Mean Absolute Percentage Error (sMAPE)  
- Mean Absolute Scaled Error (MASE)

Rolling evaluation is used to track model stability over time and detect performance drift.

### 6. Smoothing Experiments
HAI is volatile because of small denominators or rapid supply changes.  
The project compares raw and smoothed forecasts, using a four-quarter median smoother applied to both features and targets.

---

## Key Findings

- Models generally reproduce short-term dynamics but only slightly outperform the seasonal naïve baseline at longer horizons.  
- Median smoothing stabilizes forecasts and reduces extreme spikes in HAI but slightly delays turning points.  
- Population growth explains most of the variation in housing adequacy across provinces.  
- Ontario and British Columbia show persistent undersupply compared to smaller provinces.  
- Rolling evaluation indicates model stability improves after 2012 as more data become available.

---

## How to Run

1. Clone the repository.  
2. Ensure Python 3.10 or higher and install required libraries (`pandas`, `scikit-learn`, `xgboost`, `seaborn`, `plotly`).  
3. Run the notebooks in sequence:  

   ```
   00_data_processing.ipynb  
   01_eda.ipynb  
   02_dwelling_sameQ.ipynb or 03_dwelling_nextQ.ipynb  
   04_explore_hai.ipynb
   ```

4. Cached best parameters are automatically loaded from `best_params_cache_*.json`.  
   Delete or rename these files to force a fresh tuning run.

---

## Next Steps

- Add macroeconomic predictors such as interest rates, CPI, and building permits.  
- Explore multi-level (hierarchical) or panel models for cross-province transfer learning.  
- Build a reproducible pipeline with automatic rolling evaluation and dashboards for policy insights.

---

## Acknowledgments

This project was developed by participants of the **Erdős Institute Fall 2025 Data Science Bootcamp** as part of the *Forecasting Housing Supply and Demand* challenge.  
We thank the Erdős Institute for guidance and mentorship, and the teaching team for providing data access and modeling support.

Contributors include researchers and data scientists from multiple backgrounds, collaborating to develop transparent, reproducible forecasting tools for housing policy analysis.
