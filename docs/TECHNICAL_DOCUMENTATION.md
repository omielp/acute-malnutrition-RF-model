# Technical Documentation

## Purpose

This application supports early warning and decision support for acute malnutrition by forecasting short-term district risk and visualizing spatial and temporal trends.

## Analytical Workflow

1. Load monthly district data.
2. Parse dates and standardize region/district labels.
3. Compute within-district risk using each district's own historical percentiles after 9 valid prior months.
4. Harmonize legacy diarrhea inputs where needed.
5. Sanitize impossible target and covariate values.
6. Impute missing modeled covariates with district/location-aware rules and create `*_missing` indicators.
7. Cap extreme modeled covariates with robust district/location-aware thresholds and create `*_outlier` indicators.
8. Add GAM target QC for invalid observations, duplicates, and imprecise detection rates using a 95% Wilson interval width.
9. Engineer lagged, rolling, seasonal, and district-relative features.
10. Train Random Forest regression on a log1p GAM detection-rate target; inputs are not full-history standardised, preventing later observations from influencing earlier rows.
11. Derive within-district risk labels from percentile rules; no Random Forest classifier produces risk labels.

Existing CHAP artifacts created before `chap-rf-v22-finite-chap-samples` must be retrained. The prediction command rejects incompatible artifacts rather than applying an incompatible feature schema or target-quality policy.
12. Generate separate direct forecasts for horizons 1 and 3; the 3-month model does not use intermediate predictions.
13. Convert forecasts into within-district risk categories.
14. Visualize results through maps, charts, risk tables, and model metrics.

## Risk Definitions

The direct district operational alert is categorized using percentile thresholds:

| Operational Alert | Threshold |
|---|---|
| Monitor | Below 80th percentile |
| Alert | 80th to 94th percentile |
| Respond | At or above 95th percentile |

The Streamlit app offers a sensitivity selector between the default `P90 / P95` profile and `P80 / P90`. It recalculates observed and forecast alert labels, but does not alter the GAM detection-rate target. CHAP uses the standard `P90 / P95` profile for reproducible deployment output.

For Streamlit forecasting, users can select Random Forest, a causal district local-level/trend state-space model, or an MAE-weighted RF/SSM blend. The blend is an operational comparison tool until both methods are evaluated on identical rolling forecast folds; it does not use future observations.

## Within-District Risk

Within-district risk compares a district against its own previous valid GAM-detection-rate history. It is assigned after 9 valid prior months. Districts with 6–8 valid prior months are shown as `Provisional Alert History` on the observed map, without a standard alert label; districts below 6 months remain `Insufficient Alert History`.

## Suggested Actions

The within-district anomaly is the direct alert signal. `Monitor` supports routine monitoring; `Alert` prompts investigation and validation; and `Respond` prompts escalation for rapid assessment and response planning. These actions are decision-support prompts, not IPC classifications or automated response orders.

## Features

The model uses:

- sine/cosine seasonal features;
- GAM lags 1, 2, and 3, a 3-month GAM mean, and a seasonal GAM lag;
- eight lag-1 operational covariates: temperature, rainfall, GPP, malaria, diarrhoea, children assessed, reporting rate, and under-five population;
- missingness-indicator features for each selected forecast covariate;
- outlier-indicator features for each selected forecast covariate.

### Final Feature Set

The final engineered feature set is shared across the Streamlit app and CHAP. It is intentionally lean to reduce overfitting in short district time series.

| Feature group | Exact features |
|---|---|
| Temporal and GAM-history features | `Month_Sin`, `Month_Cos`, `Lag1`, `Lag2`, `Lag3`, `Roll3_Mean`, `Seasonal_Lag12` |

| Covariate | Lag-1 feature used by both app and CHAP | Missing flag | Outlier flag |
|---|---|---|---|
| `mean_temperature` | `mean_temperature` | `mean_temperature_missing` | `mean_temperature_outlier` |
| `rainfall` | `rainfall` | `rainfall_missing` | `rainfall_outlier` |
| `average_gpp` | `average_gpp` | `average_gpp_missing` | `average_gpp_outlier` |
| `malaria_confirmed_u5` | `malaria_confirmed_u5` | `malaria_confirmed_u5_missing` | `malaria_confirmed_u5_outlier` |
| `diarrhea_u5` | `diarrhea_u5` | `diarrhea_u5_missing` | `diarrhea_u5_outlier` |
| `screened_u5` | `screened_u5` | `screened_u5_missing` | `screened_u5_outlier` |
| `reporting_rate` | `reporting_rate` | `reporting_rate_missing` | `reporting_rate_outlier` |
| `population_u5` | `population_u5` | `population_u5_missing` | `population_u5_outlier` |

Notes:

- all base covariate features are used as lag-1 covariates, meaning the model uses the previous month's observed or imputed value;
- additional covariates remain available in the raw data and drivers view, but are not forecast features unless the lean schema is deliberately revised after backtesting;
- all clinical and service-delivery covariates in the feature set are under-5 based;
- `reporting_rate` is normalized to a `0-1` fraction inside preprocessing even if the source data use `0-100`.
- `target_outlier_qc` and `target_duplicate_qc` are QC fields and are not part of the model feature set.

## Missing Data Handling

Modeled covariates are handled with rule-based imputation rather than blanket row dropping or zero-filling:

- the GAM target is not imputed for model training;
- climate covariates use a 1-month forward fill, then past-only district/location-month, district/location, and global medians;
- child-health and service-delivery count covariates use a 1-month forward fill, then the same past-only fallback hierarchy;
- `reporting_rate` uses a 1-month forward fill, then the same past-only fallback hierarchy, and is clipped to `0-1`;
- `population_u5` is treated as a slow-moving denominator and uses past-only forward fill and fallback values;
- each modeled covariate generates a companion `*_missing` feature so the model can distinguish observed values from imputed ones.

In the CHAP path, if a future covariate column is absent from prediction input, it is treated as missing rather than being forced to zero.
No preprocessing step uses a later time period to fill or cap an earlier one.

## Outlier Handling

Modeled covariates are treated more aggressively than the GAM target:

- impossible covariate values are converted to missing before imputation, including reporting rates outside `0-100`;
- extreme modeled covariates are capped within district/location using a robust median plus/minus `5 * MAD` rule calculated from earlier observations only;
- each modeled covariate generates a companion `*_outlier` feature so the model can distinguish typical values from capped extremes.

This approach protects the model from obvious reporting artifacts without flattening the GAM target itself.

## Target Handling

The regression target is the observed GAM detection rate among assessed children, per 1,000 assessed children:

`GAM detection rate = GAM cases / screened_u5 * 1,000`

Rows are retained for all districts. A district-month is excluded from training and evaluation when GAM is missing or invalid, children assessed is missing or not positive, GAM exceeds children assessed, the district-month is duplicated, or its 95% Wilson interval is wider than 125 per 1,000. The interval is calculated from GAM cases and children assessed only, not from population or any future information. GAM is never imputed or proxy-filled. GAM target outliers remain QC flags because a genuine spike may carry operational signal.

## Models

### Regression

Separate Random Forest regressors predict GAM detection rate per 1,000 assessed children directly at horizons 1 and 3 months.
Forecast risk labels are derived from the regression forecast in the active forecasting path.

## CHAP Output Modes

The CHAP interface supports two output modes:

| Mode | How to enable | Output columns | Intended use |
|---|---|---|---|
| Default CHAP mode | default behavior | `time_period`, `location`, `sample_0` to `sample_99` | safest path for CHAP and DHIS2 ingestion |
| Enriched CHAP mode | `CHAP_INCLUDE_RISK_OUTPUT=1` with `predict.py`, or `python chap_model.py predict ... --include-risk-output` | default CHAP columns plus `point_forecast`, `operational_alert`, `operational_alert_code` | testing or downstream flows that can tolerate extra columns |

CHAP requires finite values in every `sample_*` column. When a district has fewer than three valid observed GAM-detection-rate months, its requested future rows therefore receive a finite fallback distribution derived only from valid GAM detection rates in the training data. It does not create synthetic GAM history, does not use the fallback as observed GAM, and does not add it to training or evaluation. The enriched fields remain `No Data`. Enriched within-district anomaly output requires 9 valid observed months.

The direct models support future months `+1` and `+3` from each district's latest valid GAM observation. For a supplied `+2` row, CHAP receives the mean of paired samples from the independent `+1` and `+3` direct models. This finite bridge is non-recursive, does not use a forecast as a model input, and retains `No Data` enriched fields rather than issuing an alert.

In enriched CHAP mode, the added outputs are derived from the regression forecast rather than emitted by a standalone classifier:

- `point_forecast` is the mean of the `sample_*` forecast draws;
- `operational_alert` is derived from each location's own historical forecast context;
- code fields are included for downstream systems that prefer numeric imports:
  - `operational_alert_code`: `0 = No Data`, `1 = Monitor`, `2 = Alert`, `3 = Respond`

### Current simple production contract

The smallest CHAP-compatible production setup keeps the existing Modeling App mapping behavior and does not require a frontend fork.

| Modeling App field | Value source today | Final behavior |
|---|---|---|
| `Quantile high` | computed by CHAP from `sample_*` | imported to DHIS2 through the standard quantile mapping |
| `Quantile mid high` | computed by CHAP from `sample_*` | imported to DHIS2 through the standard quantile mapping |
| `Median` | computed by CHAP from `sample_*` | imported to DHIS2 through the standard quantile mapping |
| `Quantile mid low` | computed by CHAP from `sample_*` | imported to DHIS2 through the standard quantile mapping |
| `Quantile low` | computed by CHAP from `sample_*` | imported to DHIS2 through the standard quantile mapping |
| `Outbreak indicator` | computed by CHAP from the chosen alert probability and the imported quantiles | binary CHAP alert channel for now |

Under this simpler first version:

- the model should be run in Default CHAP mode for production imports;
- `operational_alert` and `operational_alert_code` remain optional extra outputs for QA, exports, or a future Modeling App extension;
- exposing those extra fields as dedicated import targets will require a Modeling App and likely CHAP-core extension, because the current platform import contract is centered on quantiles plus one outbreak-indicator channel.

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
- forecast within-district risk;
- composite risk;
- direct 1-month and 3-month operational-alert maps.

## Limitations

- Forecasts depend on data quality and reporting consistency.
- Imputed covariates can stabilize training and prediction, but large or systematic source-data gaps can still bias the model.
- Extreme GAM spikes are retained by default and only flagged for QC, so confirmed reporting artifacts still require analyst review.
- Risk categories are percentile-based and should complement, not replace, public health judgment.
- District name mismatches between CSV and GeoJSON can affect maps.
- Model outputs should be reviewed alongside field surveillance and nutrition program data.
