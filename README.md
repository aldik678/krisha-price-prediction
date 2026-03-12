# Astana Apartment Price Prediction

## Overview

This project predicts apartment listing prices in Astana from a structured snapshot of listings collected from krisha.kz. The task is formulated as a supervised regression problem using a compact set of tabular features such as district, building type, residential complex, year, floor, area, room count, renovation status, parking, and bathroom type.

After duplicate removal and target cleaning, the final modeling table contains **2,357 listings**.

## Data and exploratory analysis

The exploratory analysis highlights several properties of the dataset that strongly influence the modeling strategy:

- the target distribution is **right-skewed**, with a median price of **37.5 million tenge**, a mean of **48.9 million**, and an upper tail reaching **785 million**;
- apartment area is also right-skewed, with most listings concentrated in moderate sizes and a smaller set of very large apartments;
- **area** is the strongest numeric correlate of price (correlation about **0.79**), followed by **room count** (about **0.64**);
- price distributions differ meaningfully across **districts**, **building types**, and **renovation status**, indicating substantial heterogeneity across market segments.

These patterns suggest that the relationship between the available features and price is not purely linear and that a transformed target is appropriate.

## Preprocessing and feature transformations

The workflow uses a leakage-safe preprocessing pipeline:

- exact duplicates are removed before modeling;
- the target is parsed into numeric tenge values;
- area is extracted from the original text field;
- district labels are standardized;
- categorical missing values are mapped to a common `"No data"` category;
- numeric values are median-imputed;
- categorical variables are one-hot encoded with infrequent-category handling;
- additional features are created inside the pipeline, including **area per room**, **is_new_build**, and a logged version of area.

A **log transformation of the target** is used through `TransformedTargetRegressor`. This is motivated by the strong right skew of prices, the broad value range, and the heavy upper tail. Training on `log1p(price)` reduces the dominance of extreme listings, stabilizes the scale of the regression problem, and leads to better performance than fitting directly on the raw target.

## Models

The notebook compares several model families:

- **Dummy median baseline** to verify that the learned models recover nontrivial signal;
- **Ridge regression** as a regularized linear benchmark;
- **Random forest** as a nonlinear bagged-tree baseline;
- **XGBoost** as the main boosted-tree model.

This comparison is important because the EDA shows both strong size-related structure and substantial heterogeneity across market segments. Linear models can capture broad trends, but boosted trees are better suited to nonlinear patterns, threshold effects, and interactions between size, district, building type, and apartment condition.

## Model selection and evaluation

Model selection is performed on the training split with cross-validation. Both raw-target and log-target pipelines are compared, and the log-target setting improves all learnable model families.

The final tuned comparison focuses on **Ridge** and **XGBoost**. On cross-validation, the tuned XGBoost model achieves the best mean RMSE among the final candidates. On the untouched test set, it also delivers the strongest performance:

- **RMSE:** 15.24 million tenge
- **MAE:** 8.71 million tenge
- **MAPE:** 16.4%
- **R²:** 0.816

Tuned Ridge remains competitive, but it is weaker across all reported holdout metrics.

## Segment analysis

The selected final model is also evaluated on two price segments in the test set:

- **price <= 75th percentile**
- **price > 75th percentile**

This split shows a clear difference in difficulty. The lower-price segment has much lower absolute error because its target range is relatively tight. The upper-price segment has substantially larger error, which is expected because it is far more heterogeneous and covers a much broader range of prices. In the full dataset, the upper tail extends to **785 million tenge**, so higher error in the expensive segment should be interpreted in the context of much greater variability rather than as a simple model failure.

## Project structure

```text
app/
  app.py
artifacts/
  app_metadata.json
  metrics_snapshot.json
  model_pipeline.pkl
data/
  krisha_data.csv
notebooks/
  krisha_price_regression.ipynb
src/
  __init__.py
  project_utils.py
README.md
requirements.txt
```

## Saved artifacts

The artifacts/ directory stores the outputs used outside the notebook:

model_pipeline.pkl — trained preprocessing-and-model pipeline used for inference;

app_metadata.json — metadata used by the Streamlit app, including expected input structure and category information;

metrics_snapshot.json — saved evaluation summary for the selected model.

## Running the project

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```
Run the notebook from Jupyter and execute the cells in order to reproduce the analysis.
Start the Streamlit interface from the project root:
```bash
streamlit run app/app.py
```

## Data file

The analysis expects the source data at data/krisha_data.csv. The notebook and application require that file to be present in the data/ directory before execution.


## Conclusion

The final results are strong in the context of the available data. The dataset is relatively small, the feature set is compact, and several important housing-price determinants are not observed directly. Even under those constraints, the model captures a substantial share of the pricing signal.

The final XGBoost pipeline performs well because it can combine continuous size effects with nonlinear and segment-specific structure that a purely linear model cannot represent as naturally. The result is a practical housing-price regression workflow with a clear separation between exploratory analysis, model selection, holdout evaluation, and deployment-oriented artifact saving.
