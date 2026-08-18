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
- climate covariates plus under-5 disease, service-delivery, reporting-quality, and population covariates;
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
| `malaria_confirmed_u5` | `malaria_confirmed_u5` | `malaria_confirmed_u5_missing` | `malaria_confirmed_u5_outlier` | `malaria_confirmed_u5_Lag2`, `malaria_confirmed_u5_Lag3`, `malaria_confirmed_u5_Roll2_Mean`, `malaria_confirmed_u5_Roll3_Mean` |
| `pneumonia_cases_u5` | `pneumonia_cases_u5` | `pneumonia_cases_u5_missing` | `pneumonia_cases_u5_outlier` | `pneumonia_cases_u5_Lag2`, `pneumonia_cases_u5_Lag3`, `pneumonia_cases_u5_Roll2_Mean`, `pneumonia_cases_u5_Roll3_Mean` |
| `diarrhea_u5` | `diarrhea_u5` | `diarrhea_u5_missing` | `diarrhea_u5_outlier` | `diarrhea_u5_Lag2`, `diarrhea_u5_Lag3`, `diarrhea_u5_Roll2_Mean`, `diarrhea_u5_Roll3_Mean` |
| `low_birth_weight_babies` | `low_birth_weight_babies` | `low_birth_weight_babies_missing` | `low_birth_weight_babies_outlier` | `low_birth_weight_babies_Lag2`, `low_birth_weight_babies_Lag3`, `low_birth_weight_babies_Roll2_Mean`, `low_birth_weight_babies_Roll3_Mean` |
| `sam_admissions_u5` | `sam_admissions_u5` | `sam_admissions_u5_missing` | `sam_admissions_u5_outlier` | `sam_admissions_u5_Lag2`, `sam_admissions_u5_Lag3`, `sam_admissions_u5_Roll2_Mean`, `sam_admissions_u5_Roll3_Mean` |
| `screened_u5` | `screened_u5` | `screened_u5_missing` | `screened_u5_outlier` | `screened_u5_Lag2`, `screened_u5_Lag3`, `screened_u5_Roll2_Mean`, `screened_u5_Roll3_Mean` |
| `reporting_rate` | `reporting_rate` | `reporting_rate_missing` | `reporting_rate_outlier` | `reporting_rate_Lag2`, `reporting_rate_Lag3`, `reporting_rate_Roll2_Mean`, `reporting_rate_Roll3_Mean` |
| `population_u5` | `population_u5` | `population_u5_missing` | `population_u5_outlier` | none |

Notes:

- all base covariate features are used as lag-1 covariates, meaning the model uses the previous month's observed or imputed value;
- the extra `Lag2`, `Lag3`, `Roll2_Mean`, and `Roll3_Mean` features are applied to all modeled covariates except `population_u5`;
- all clinical and service-delivery covariates in the feature set are under-5 based, with `low_birth_weight_babies` retained as a neonatal covariate within the same under-5 scope;
- `reporting_rate` is normalized to a `0-1` fraction inside preprocessing even if the source data use `0-100`.
- `target_outlier_qc` and `target_duplicate_qc` are QC fields and are not part of the model feature set.

## Missing Data Handling

Modeled covariates are handled with rule-based imputation rather than blanket row dropping or zero-filling:

- the GAM target is not imputed for model training;
- climate covariates are interpolated within district/location, then filled from district/location-month medians, then district/location medians;
- child-health and service-delivery count covariates use a 1-month forward fill, then district/location-month medians, then district/location medians;
- `reporting_rate` uses a 1-month forward fill, then district/location-month medians, then district/location medians, and is clipped to `0-1`;
- `population_u5` is treated as a slow-moving denominator and uses forward fill plus back fill before median fallback;
- each modeled covariate generates a companion `*_missing` feature so the model can distinguish observed values from imputed ones.

In the CHAP path, if a future covariate column is absent from prediction input, it is treated as missing rather than being forced to zero.

## Outlier Handling

Modeled covariates are treated more aggressively than the GAM target:

- impossible covariate values are converted to missing before imputation, including reporting rates outside `0-100`;
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

### Target Proxy Status

Target-proxy creation is active in the training pipeline.

Current behavior:

- observed `disease_cases` or `Acut_Malnutrition` remains the preferred target whenever it is available;
- missing target rows can be filled from proxy logic based on `screened_u5`, `sam_admissions_u5`, and `reporting_rate`;
- proxy calibration is causal, so each row only uses earlier observed history when estimating GAM-to-screening and GAM-to-SAM ratios;
- proxy provenance fields are generated, including `target_source`, `target_proxy_confidence`, `target_is_observed`, and `target_training_weight`;
- CHAP and Streamlit training can include proxy-filled rows, but with lower sample weights than fully observed GAM rows;
- regression evaluation metrics remain anchored on rows with observed GAM only;
- Streamlit data-quality summaries and observed anomaly maps use observed GAM only, not proxy-filled GAM.

Implemented proxy workflow:

| Step | Intended rule |
|---|---|
| 1. Observed target | If reported GAM caseload is present, use it directly |
| 2. Reporting adjustment | Adjust `screened_u5` and `sam_admissions_u5` using district `reporting_rate`, with a conservative lower floor on the effective reporting rate |
| 3. Screening proxy | If GAM is missing and `screened_u5` is available, estimate GAM from `screened_u5 x` historical district-seasonal GAM detection rate learned from earlier observed months |
| 4. SAM proxy | If GAM is missing and `sam_admissions_u5` is available, estimate GAM from `sam_admissions_u5 x` historical district-seasonal GAM-to-SAM multiplier learned from earlier observed months |
| 5. Blended proxy | If both screening and SAM proxies are available, combine them with heavier weight on screening |
| 6. Provenance flags | Store whether the target is `observed`, `proxy_screened`, `proxy_sam_admissions`, or `proxy_combined`, plus a proxy-confidence label and training weight |

Training-weight rules:

| Target source | Base training weight | Notes |
|---|---|---|
| `observed` | `1.00` | full weight |
| `proxy_combined` | `0.60` | strongest proxy source |
| `proxy_screened` | `0.50` | screening-only proxy |
| `proxy_sam_admissions` | `0.35` | weakest direct proxy in the current implementation |

For proxy-filled rows, the base weight is further reduced when `reporting_rate` is low:

- `reporting_rate >= 0.8`: weight factor `1.00`
- `0.6 <= reporting_rate < 0.8`: weight factor `0.85`
- `reporting_rate < 0.6`: weight factor `0.70`

Recommended implementation guardrails:

- keep proxy-filled target rows out of the gold-standard evaluation set;
- compare model performance with and without proxy-filled training rows;
- treat `screened_u5` as the primary proxy for total GAM and `sam_admissions_u5` as a secondary severity proxy;
- cap or downweight reporting-rate adjustments when `reporting_rate` is very low.

## Models

### Regression

Random Forest Regressor predicts acute malnutrition case counts for the next 3 months.
Forecast risk labels and operational alerts are derived from the regression forecast in the active forecasting path.

## CHAP Output Modes

The CHAP interface supports two output modes:

| Mode | How to enable | Output columns | Intended use |
|---|---|---|---|
| Default CHAP mode | default behavior | `time_period`, `location`, `sample_0` to `sample_99` | safest path for CHAP and DHIS2 ingestion |
| Enriched CHAP mode | `CHAP_INCLUDE_RISK_OUTPUT=1` with `predict.py`, or `python chap_model.py predict ... --include-risk-output` | default CHAP columns plus `point_forecast`, `wd_risk`, `wd_risk_code`, `xd_risk`, `xd_risk_code`, `Operational_Alert`, `Operational_Alert_Code`, `Operational_Alert_Why`, `Composite_Risk`, `Composite_Risk_Code` | testing or downstream flows that can tolerate extra columns |

In enriched CHAP mode, the added outputs are derived from the regression forecast rather than emitted by a standalone classifier:

- `point_forecast` is the mean of the `sample_*` forecast draws;
- `wd_risk` is derived from each location's own historical forecast context;
- `xd_risk` is derived by comparing forecasted locations within the same forecast month;
- `Operational_Alert` and `Operational_Alert_Why` use the same rule logic as the Streamlit app.
- code fields are included for downstream systems that prefer numeric imports:
  - `Operational_Alert_Code`: `0 = No Data`, `1 = Monitor`, `2 = Alert`, `3 = Respond`
  - `wd_risk_code` and `xd_risk_code`: `0 = No Data`, `1 = Low`, `2 = Moderate`, `3 = High`, `4 = Extreme`

### No-fork Operational Alert workaround

If the Modeling App cannot be forked yet, the existing `Outbreak indicator` channel can be repurposed by enabling:

- CLI: `python chap_model.py predict ... --outbreak-indicator-mode operational_alert_code`
- env var: `CHAP_OUTBREAK_INDICATOR_MODE=operational_alert_code`

When this mode is enabled, CHAP appends:

- `outbreak_indicator`
- `outbreak_indicator_label`

Workaround semantics:

- `outbreak_indicator = Operational_Alert_Code`
- `0 = No Data`
- `1 = Monitor`
- `2 = Alert`
- `3 = Respond`

This is intentionally a workaround rather than a native platform feature. It repurposes the outbreak-indicator channel from a binary outbreak signal into a 4-state operational-alert code, so downstream interpretation must treat it as `Operational Alert`.

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
- `wd_risk`, `xd_risk`, `Operational_Alert`, `Operational_Alert_Why`, and `Composite_Risk` remain optional extra outputs for QA, exports, or future forked Modeling App work;
- exposing those extra fields as dedicated import targets will require a Modeling App and likely CHAP-core extension, because the current platform import contract is centered on quantiles plus one outbreak-indicator channel.

### Smallest Modeling App fork for Operational Alert

This repository does not contain the Modeling App source code, so the DHIS2 setup modal and import workflow must be changed in that separate app. The smallest fork is:

| Layer | Smallest change |
|---|---|
| Prediction setup modal | add one new mapping field: `Operational Alert` |
| Saved prediction setup schema | persist the chosen DHIS2 data element for `Operational_Alert_Code` |
| Import payload builder | include `Operational_Alert_Code` from enriched CHAP output when sending data values to DHIS2 |
| DHIS2 metadata | create one numeric data element for the alert code |

Recommended first implementation:

- map `Operational_Alert_Code`, not free text;
- keep quantile mappings and outbreak-indicator mappings unchanged;
- enable enriched CHAP output only for the forked Modeling App path that can consume the extra field.

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
