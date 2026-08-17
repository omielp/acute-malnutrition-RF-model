# Technical Documentation

## Purpose

This application supports early warning and decision support for acute malnutrition by forecasting short-term district risk and visualizing spatial and temporal trends.

## Analytical Workflow

1. Load monthly district data.
2. Parse dates and standardize region/district labels.
3. Compute within-district risk using each district's own historical percentiles.
4. Compute between-district risk using peer district distributions by month.
5. Harmonize legacy diarrhea inputs where needed.
6. Sanitize impossible target and covariate values.
7. Impute missing modeled covariates with district/location-aware rules and create `*_missing` indicators.
8. Cap extreme modeled covariates with robust district/location-aware thresholds and create `*_outlier` indicators.
9. Add GAM target QC flags for suspicious spikes and duplicate district-month observations.
10. Engineer lagged, rolling, seasonal, and district-relative features.
11. Train Random Forest regression to forecast acute malnutrition case counts.
12. Train Random Forest classifiers for risk evaluation.
13. Generate 3-month recursive forecasts.
14. Convert forecasts into risk categories.
15. Visualize results through maps, charts, risk tables, and model metrics.

## Risk Definitions

Risk is categorized using percentile thresholds:

| Risk | Threshold |
|---|---|
| Low | Below 75th percentile |
| Moderate | 75th to 89th percentile |
| High | 90th to 94th percentile |
| Extreme | At or above 95th percentile |

## Within-District Risk

Within-district risk compares a district against its own previous history. This is useful for identifying unusual increases in a district even if its absolute case count is not nationally high.

## Between-District Risk

Between-district risk compares each district against other districts. This is useful for identifying districts with high burden relative to peers.

## Features

The model uses:

- calendar features: month, quarter, sine/cosine seasonality;
- lag features: previous 1, 2, and 3 months;
- rolling features: 3-, 6-, and 12-month summaries;
- trend and acceleration;
- district historical percentiles;
- threshold exceedance features;
- recent risk counts;
- climate covariates plus under-5 disease and population covariates;
- missingness-indicator features for each modeled covariate;
- outlier-indicator features for each modeled covariate.

### Final Feature Set

The final engineered feature set is shared across the Streamlit app and CHAP, with three app-only anomaly-history features.

| Feature group | Exact features |
|---|---|
| Shared temporal and target-history features | `Month`, `Quarter`, `Month_Sin`, `Month_Cos`, `Lag1`, `Lag2`, `Lag3`, `Lag1_Z`, `Roll3_Mean`, `Roll3_Std`, `Roll6_Mean`, `Seasonal_Lag12`, `Hist_Mean`, `Hist_Std`, `Hist_P90`, `Hist_P95`, `Lag1_Over_P95`, `Lag1_Ratio_P95` |
| App-only features | `Recent_WD_High_Count`, `WD_Score_Lag`, `XD_Score_Lag` |

| Covariate | Base covariate feature used by both app and CHAP | Missing flag | Outlier flag | Extra lag and rolling features |
|---|---|---|---|---|
| `mean_temperature` | `mean_temperature` | `mean_temperature_missing` | `mean_temperature_outlier` | `mean_temperature_Lag2`, `mean_temperature_Lag3`, `mean_temperature_Roll2_Mean`, `mean_temperature_Roll3_Mean` |
| `rainfall` | `rainfall` | `rainfall_missing` | `rainfall_outlier` | `rainfall_Lag2`, `rainfall_Lag3`, `rainfall_Roll2_Mean`, `rainfall_Roll3_Mean` |
| `mean_relative_humidity` | `mean_relative_humidity` | `mean_relative_humidity_missing` | `mean_relative_humidity_outlier` | `mean_relative_humidity_Lag2`, `mean_relative_humidity_Lag3`, `mean_relative_humidity_Roll2_Mean`, `mean_relative_humidity_Roll3_Mean` |
| `average_gpp` | `average_gpp` | `average_gpp_missing` | `average_gpp_outlier` | `average_gpp_Lag2`, `average_gpp_Lag3`, `average_gpp_Roll2_Mean`, `average_gpp_Roll3_Mean` |
| `malaria_confirmed` | `malaria_confirmed` | `malaria_confirmed_missing` | `malaria_confirmed_outlier` | `malaria_confirmed_Lag2`, `malaria_confirmed_Lag3`, `malaria_confirmed_Roll2_Mean`, `malaria_confirmed_Roll3_Mean` |
| `pneumonia_cases` | `pneumonia_cases` | `pneumonia_cases_missing` | `pneumonia_cases_outlier` | `pneumonia_cases_Lag2`, `pneumonia_cases_Lag3`, `pneumonia_cases_Roll2_Mean`, `pneumonia_cases_Roll3_Mean` |
| `diarrhea` | `diarrhea` | `diarrhea_missing` | `diarrhea_outlier` | `diarrhea_Lag2`, `diarrhea_Lag3`, `diarrhea_Roll2_Mean`, `diarrhea_Roll3_Mean` |
| `low_birth_weight_babies` | `low_birth_weight_babies` | `low_birth_weight_babies_missing` | `low_birth_weight_babies_outlier` | `low_birth_weight_babies_Lag2`, `low_birth_weight_babies_Lag3`, `low_birth_weight_babies_Roll2_Mean`, `low_birth_weight_babies_Roll3_Mean` |
| `population` | `population` | `population_missing` | `population_outlier` | none |

Notes:

- all base covariate features are used as lag-1 covariates, meaning the model uses the previous month's observed or imputed value;
- the extra `Lag2`, `Lag3`, `Roll2_Mean`, and `Roll3_Mean` features are applied to all modeled covariates except `population`;
- `target_outlier_qc` and `target_duplicate_qc` are QC fields and are not part of the model feature set.

## Missing Data Handling

Modeled covariates are handled with rule-based imputation rather than blanket row dropping or zero-filling:

- the GAM target is not imputed for model training;
- climate covariates are interpolated within district/location, then filled from district/location-month medians, then district/location medians;
- child-health covariates use a 1-month forward fill, then district/location-month medians, then district/location medians;
- `population` is treated as a slow-moving denominator and uses forward fill plus back fill before median fallback;
- each modeled covariate generates a companion `*_missing` feature so the model can distinguish observed values from imputed ones.

In the CHAP path, if a future covariate column is absent from prediction input, it is treated as missing rather than being forced to zero.

## Outlier Handling

Modeled covariates are treated more aggressively than the GAM target:

- impossible covariate values are converted to missing before imputation;
- extreme modeled covariates are capped within district/location using a robust median plus/minus `5 * MAD` rule;
- each modeled covariate generates a companion `*_outlier` feature so the model can distinguish typical values from capped extremes.

This approach protects the model from obvious reporting artifacts without flattening the GAM target itself.

## Target Handling

The GAM target is handled conservatively:

- negative or non-numeric target values are converted to missing and excluded from training;
- the target is not imputed for training;
- the target is not winsorized by default, because true spikes may carry operational signal;
- suspicious target spikes are tracked with `target_outlier_qc`;
- duplicate district/location-month target observations are tracked with `target_duplicate_qc`.

## Models

### Regression

Random Forest Regressor predicts acute malnutrition case counts for the next 3 months.
Forecast risk labels and operational alerts are derived from the regression forecast in the active forecasting path.

## CHAP Output Modes

The CHAP interface supports two output modes:

| Mode | How to enable | Output columns | Intended use |
|---|---|---|---|
| Default CHAP mode | default behavior | `time_period`, `location`, `sample_0` to `sample_99` | safest path for CHAP and DHIS2 ingestion |
| Enriched CHAP mode | `CHAP_INCLUDE_RISK_OUTPUT=1` with `predict.py`, or `python chap_model.py predict ... --include-risk-output` | default CHAP columns plus `point_forecast`, `wd_risk`, `xd_risk`, `Operational_Alert`, `Operational_Alert_Why`, `Composite_Risk` | testing or downstream flows that can tolerate extra columns |

In enriched CHAP mode, the added outputs are derived from the regression forecast rather than emitted by a standalone classifier:

- `point_forecast` is the mean of the `sample_*` forecast draws;
- `wd_risk` is derived from each location's own historical forecast context;
- `xd_risk` is derived by comparing forecasted locations within the same forecast month;
- `Operational_Alert` and `Operational_Alert_Why` use the same rule logic as the Streamlit app.

## Evaluation

The app uses walk-forward cross-validation because the data are temporal. Evaluation metrics include:

- MAE and RMSE for regression;
- accuracy;
- balanced accuracy;
- macro recall;
- weighted F1;
- sensitivity;
- specificity;
- precision;
- NPV;
- confusion-matrix counts.

## Correlation Analysis

Spearman rank correlation is used to assess monotonic associations between acute malnutrition and covariates. This is robust for non-normal and skewed public health data.

## Mapping

The app accepts a district GeoJSON file and maps:

- current within-district risk;
- current between-district risk;
- forecast within-district risk;
- forecast between-district risk;
- composite risk;
- 3-month risk change.

## Limitations

- Forecasts depend on data quality and reporting consistency.
- Imputed covariates can stabilize training and prediction, but large or systematic source-data gaps can still bias the model.
- Extreme GAM spikes are retained by default and only flagged for QC, so confirmed reporting artifacts still require analyst review.
- Risk categories are percentile-based and should complement, not replace, public health judgment.
- District name mismatches between CSV and GeoJSON can affect maps.
- Model outputs should be reviewed alongside field surveillance and nutrition program data.
