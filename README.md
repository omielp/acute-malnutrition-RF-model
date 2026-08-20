# Acute Malnutrition Forecasting Tool (AMFT)

AMFT is a Streamlit-based decision-support tool for district-level acute malnutrition monitoring in Uganda. It combines anomaly detection, short-term forecasting, geospatial visualization, data quality checks, and model evaluation to support early warning and operational planning.

The current version focuses on three connected outputs:

- 4-level anomaly classification for within-district and between-district monitoring
- 3-level operational alerts derived from anomaly combinations
- 3-month district forecasts of GAM caseloads with uncertainty bounds

## What Changed in the Current Tool

The README has been updated to reflect the current app behavior. The tool now uses:

- anomaly levels of `Low`, `Moderate`, `High`, and `Extreme`
- operational alerts of `Monitor`, `Alert`, and `Respond`
- 3-month recursive forecasting with `Lower_80` and `Upper_80` intervals
- district filtering across the full app
- observed and forecast map views for anomalies and operational alerts
- built-in data quality summaries for districts and months
- regression model checks inside the app
- optional IPC AMN severity labeling when direct GAM prevalence columns are available

## Core Capabilities

- Load district monthly GAM caseload data from a local path or file upload
- Load district boundaries from a local GeoJSON path or file upload
- Classify observed GAM caseloads using district-relative and peer-relative anomaly rules
- Generate operational alerts from within-district and between-district anomaly combinations
- Forecast district GAM caseloads for the next 3 months
- Display forecast uncertainty with 80% intervals
- Compare districts or regions against a national reference trend
- Explore Spearman correlations between GAM caseload and covariates
- Review regression and classification model performance
- Download classified observed data and forecast outputs

## Risk Framework

### 1. Anomaly Classification

Observed and forecast anomaly labels use percentile thresholds:

| Level | Threshold | Meaning |
|---|---|---|
| `Low` | Below 75th percentile | Within the usual range |
| `Moderate` | 75th to below 90th percentile | Higher than usual |
| `High` | 90th to below 95th percentile | Unusually high |
| `Extreme` | 95th percentile and above | Exceptionally high |

### 2. Operational Alert

Operational alerts are derived from the combination of within-district and between-district anomaly levels:

| Operational Alert | Rule | Typical Action |
|---|---|---|
| `Monitor` | Within = `Low` and Between = `Low` or `Moderate` | Routine monitoring |
| `Alert` | Within = `Low` and Between = `High` or `Extreme`, or Within = `Moderate` and Between = `Low` or `Moderate` | Heightened surveillance |
| `Respond` | Within = `High` or `Extreme`, or Within = `Moderate` and Between = `High` or `Extreme` | Immediate response |

### 3. IPC AMN Severity

If the dataset includes direct GAM prevalence columns such as WHZ- or MUAC-based prevalence, the app can also compute observed IPC AMN severity phases:

- `Acceptable`
- `Alert`
- `Serious`
- `Critical`
- `Extremely Critical`

If prevalence data are not present, the app still supports anomaly-based early warning and GAM caseload forecasting.

## How the App Works

The application follows this workflow:

1. Load monthly district GAM data.
2. Parse dates and standardize district labels.
3. Calculate within-district anomalies from each district's own historical distribution.
4. Calculate between-district anomalies from peer district and seasonal patterns.
5. Harmonize diarrhea inputs, sanitize impossible values, cap extreme modeled covariates, and create missingness and outlier indicator features.
6. Apply target QC flags for suspicious GAM spikes or duplicate district-month rows.
7. Engineer lagged, rolling, trend, seasonal, and district-relative features.
8. Train Random Forest models for regression and anomaly classification.
9. Generate 3-month recursive district forecasts.
10. Convert forecast outputs into anomaly labels and operational alerts.
11. Visualize current conditions, historical patterns, drivers, forecasts, and model checks.

## App Sections

The Streamlit app is organized into four main tabs:

### Exploratory Analysis

- Run summary
- Data quality and methodology
- Current situation summaries
- Top districts by current operational alert
- Observed anomaly and operational alert maps
- Raw classified data table

### Drivers & Context

- Spearman correlation portrait
- District or regional comparison against the national reference
- Historical anomaly trend
- Historical GAM trend

### Forecasts

- 3-month district forecast chart
- Hindcast view for prior out-of-sample predictions
- Forecast table with uncertainty bounds
- Forecast anomaly and operational alert maps

### Model Checks

- Regression metrics such as MAE, RMSE, mean error, and P90 absolute error
- Predicted vs actual plots
- Residual analysis
- Largest forecast error review
- Feature importance charts
- Classification metrics for within-district and between-district anomaly models

## System Architecture

The tool is designed for integration with DHIS2 and the Climate Health Analytics Platform (CHAP).

![System setup in the DHIS2 eco system](Screenshots/architecture.png)

## CHAP Integration

This repository now includes a CHAP-compatible model interface at the project root:

- `MLproject`
- `MLProject.yaml`
- `train.py`
- `predict.py`
- `chap_model.py`
- `pyproject.toml`

### CHAP entrypoints

The model follows the CHAP external-model contract:

- `train` takes `train_data` and `model`
- `predict` takes `historic_data`, `future_data`, `model`, and `out_file`

The `MLproject` file exposes these entrypoints so the model can be run directly by CHAP.
`MLProject.yaml` is also included for compatibility with CHAP model-template seeding flows that fetch metadata from GitHub.

### Input schema

The CHAP-facing scripts accept the standard CHAP column names:

- `time_period`
- `location`
- `disease_cases` for `train_data` and `historic_data`

Modeled covariates supported by the model:

- `mean_temperature`
- `rainfall`
- `mean_relative_humidity`
- `average_gpp`
- `malaria_confirmed_u5` for malaria cases among children under 5
- `pneumonia_cases_u5` for pneumonia cases among children under 5
- `diarrhea_u5` for diarrhoea cases among children under 5
- `low_birth_weight_babies` for low birth weight newborns
- `sam_admissions_u5` for SAM admissions among children under 5
- `screened_u5` for under-5 nutrition screening or assessment volume
- `reporting_rate` for the district health-facility reporting rate, accepted as either `0-1` or `0-100`
- `population_u5` for the district under-5 population

For this project, all clinical and service-delivery covariates are interpreted as children under 5 years of age. In practice this means:

- `malaria_confirmed_u5`, `pneumonia_cases_u5`, `diarrhea_u5`, `sam_admissions_u5`, and `screened_u5` are under-5 case-count or service-volume inputs
- `low_birth_weight_babies` is retained as a neonatal input and therefore still falls within the under-5 scope
- `reporting_rate` is a district operational-quality input and is normalized to a `0-1` fraction inside the pipeline
- `population_u5` is the under-5 population denominator

For backward compatibility with older datasets, the wrapper also accepts:

- `Region_District` as a fallback for `location`
- `Acut_Malnutrition` as a fallback for `disease_cases`
- legacy `diarrhea_acute` and `diarrhea_persistent`, which are combined into `diarrhea_u5`
- common aliases for the U5 clinical covariates, `screened_u5`, `population_u5`, and `reporting_rate`, which are harmonized to the canonical covariate names

### Missing covariate handling

The CHAP pipeline uses a causal, rule-based missing-data workflow for modeled covariates:

- GAM target values are not imputed for training
- climate, child-health, service-volume, and reporting-rate covariates use a 1-month forward fill, then past-only district/location-month, district/location, and global medians; reporting rate is clipped to `0-1`
- `population_u5` uses past-only forward fill and fallback values
- each modeled covariate also generates a companion `*_missing` feature so the model can learn whether the original value was observed or imputed

For CHAP prediction inputs, if a future covariate column is omitted entirely, the model treats it as missing rather than forcing it to zero.
No CHAP preprocessing step uses a later time period to fill or cap an earlier row.

### Outlier and target handling

Modeled covariates and GAM targets are handled differently:

- impossible covariate values are converted to missing before preprocessing, for example negative child-health counts, non-positive `population_u5`, negative `rainfall`, negative `average_gpp`, humidity outside `0-100`, or reporting rates outside `0-100`
- extreme modeled covariates are capped within district/location using a robust median plus/minus `5 * MAD` rule calculated from earlier observations only, and each modeled covariate also gets a companion `*_outlier` indicator
- GAM targets are not winsorized by default, because true spikes may be the signal of interest
- impossible GAM target values such as negative caseloads are converted to missing and excluded from training
- GAM target QC is tracked with `target_outlier_qc` and `target_duplicate_qc` flags so suspicious spikes or duplicate district-month observations can be reviewed without flattening the target series

### Target proxy status

Proxy target creation is now active in the training pipeline for rows where GAM is missing.

- observed GAM remains the preferred target whenever it is available
- if GAM is missing, the pipeline can build:
  - `gam_proxy_screened` from `screened_u5`
  - `gam_proxy_sam_admissions` from `sam_admissions_u5`
  - `gam_proxy_combined` as the default blended proxy target
- proxy calibration is causal: each month only uses earlier observed history, never future months or same-month peers
- `reporting_rate` is used to adjust `screened_u5` and `sam_admissions_u5` before proxy calculation, with a conservative floor to avoid extreme inflation
- provenance fields are generated:
  - `target_source`
  - `target_proxy_confidence`
  - `target_is_observed`
  - `target_training_weight`
- proxy-filled rows can be used for model training, but with lower training weights than observed GAM rows
- regression evaluation remains anchored on rows with observed GAM only
- data-quality summaries and "Observed Anomalies & Operational Alert" maps now use observed GAM rows only

### Output schema

The `predict` entrypoint writes a CHAP-compatible CSV with:

- `time_period`
- `location`
- `sample_0` through `sample_99`

Each `sample_*` column is one forecast draw derived from the fitted random forest, which allows CHAP to calculate uncertainty intervals.

### Current CHAP and Modeling App setup

For the simpler first version, use the default CHAP output mode in production:

| CHAP / Modeling App parameter today | Source from this model | Notes |
|---|---|---|
| `Quantile high` | derived by CHAP from `sample_*` | standard CHAP import path |
| `Quantile mid high` | derived by CHAP from `sample_*` | standard CHAP import path |
| `Median` | derived by CHAP from `sample_*` | standard CHAP import path |
| `Quantile mid low` | derived by CHAP from `sample_*` | standard CHAP import path |
| `Quantile low` | derived by CHAP from `sample_*` | standard CHAP import path |
| `Outbreak indicator` | derived by CHAP from imported quantiles plus alert probability | this is the current alert channel in CHAP, not a direct import of `Operational_Alert` |

This means the current CHAP / DHIS2 workflow works without a frontend fork, but it does not yet expose dedicated setup fields for `wd_risk`, `xd_risk`, `Operational_Alert`, `Operational_Alert_Why`, or `Composite_Risk`.

Optional enriched CHAP output is also available when needed for downstream testing:

- set `CHAP_INCLUDE_RISK_OUTPUT=1` when using `predict.py`; or
- run `python chap_model.py predict ... --include-risk-output`

When enabled, the prediction CSV appends:

- `point_forecast`
- `wd_risk`
- `wd_risk_code`
- `xd_risk`
- `xd_risk_code`
- `Operational_Alert`
- `Operational_Alert_Code`
- `Operational_Alert_Why`
- `Composite_Risk`
- `Composite_Risk_Code`

Recommended numeric coding for DHIS2:

- `Operational_Alert_Code`: `0 = No Data`, `1 = Monitor`, `2 = Alert`, `3 = Respond`
- `wd_risk_code` and `xd_risk_code`: `0 = No Data`, `1 = Low`, `2 = Moderate`, `3 = High`, `4 = Extreme`

This keeps the default CHAP sample output unchanged while allowing risk-label and operational-alert testing in environments that can tolerate extra columns. These extra columns are useful for local QA, exports, and future Modeling App fork work, but they are not part of the simple production import contract today.

### No-fork Operational Alert workaround

If you need to use the existing `Outbreak indicator` slot without forking the Modeling App, enable the workaround mode:

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

This is a pragmatic compatibility workaround. It repurposes the outbreak-indicator channel from a binary outbreak signal into a 4-state operational-alert code, so downstream users should treat it as `Operational Alert`, not as the original binary outbreak flag.

### Smallest Modeling App fork for Operational Alert

This repo does not contain the Modeling App source, so that UI/import change still has to be done in the separate app. The smallest fork is:

| Layer | Smallest change |
|---|---|
| Prediction setup modal | add a new mapping field named `Operational Alert` |
| Saved setup schema | store the selected DHIS2 data element for `Operational_Alert_Code` |
| Import payload builder | send `Operational_Alert_Code` from enriched CHAP output to DHIS2 |
| DHIS2 metadata | create one numeric data element for the alert code |

Recommended first implementation:

- map `Operational_Alert_Code`, not free text
- keep the existing quantile and outbreak-indicator mappings unchanged
- enable enriched CHAP output only in the forked path that knows how to import the extra field

### Local smoke-test example

Once dependencies are installed, the CHAP path can be exercised locally with:

```bash
python train.py /path/to/train.csv /tmp/model.pkl
python predict.py /tmp/model.pkl /path/to/historic.csv /path/to/future.csv /tmp/predictions.csv
CHAP_INCLUDE_RISK_OUTPUT=1 python predict.py /tmp/model.pkl /path/to/historic.csv /path/to/future.csv /tmp/predictions_with_risk.csv
CHAP_OUTBREAK_INDICATOR_MODE=operational_alert_code python predict.py /tmp/model.pkl /path/to/historic.csv /path/to/future.csv /tmp/predictions_with_outbreak_workaround.csv
```

## Installation

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The default local address is:

```text
http://localhost:8501
```

## Running With Sample Data

The app already points to local sample files by default:

- `data/sample/Acute_Malnutrition_data_district.csv`
- `data/sample/uganda-districts_ug.geojson`

In the sidebar you can choose either:

- `Use local CSV path` / `Use local GeoJSON path`
- `Upload CSV` / `Upload GeoJSON`

When a GeoJSON is provided, the app asks you to select the district name column used for matching.

## Data Requirements

### Required columns

| Column | Description |
|---|---|
| `Region_District` | Region and district label, for example `Acholi|Agago District` |
| `District` | District name |
| `Region` | Region name |
| `time_period` | Monthly date such as `2020-01` |
| `Acut_Malnutrition` | GAM caseload, interpreted in the app as SAM + MAM |
| `mean_temperature` | Mean temperature |
| `rainfall` | Rainfall |
| `mean_relative_humidity` | Relative humidity |
| `average_gpp` | Vegetation productivity |
| `malaria_confirmed_u5` | Malaria cases among children under 5 |
| `pneumonia_cases_u5` | Pneumonia cases among children under 5 |
| `diarrhea_u5` | Diarrhoea cases among children under 5 |
| `low_birth_weight_babies` | Low birth weight newborns |
| `sam_admissions_u5` | SAM admissions among children under 5 |
| `screened_u5` | Under-5 nutrition screening or assessment volume |
| `reporting_rate` | District health-facility reporting rate, accepted as `0-1` or `0-100` |
| `population_u5` | District under-5 population |

Missing values in these covariates are handled internally using the rule-based workflow described above. The model also creates missingness and outlier indicator features for each modeled covariate.

### Optional severity columns

Observed IPC AMN severity can be calculated if the dataset includes a direct GAM prevalence column. The app checks for common WHZ- and MUAC-based naming variants such as:

- `gam_whz`
- `gam_whz_pct`
- `gam_muac`
- `gam_muac_pct`

### GeoJSON requirements

- Upload or point to a district-level GeoJSON
- Select the GeoJSON column that contains district names
- Ensure district names match the CSV closely enough for joining

## Model Approach

### Regression

A `RandomForestRegressor` is used to forecast district GAM caseloads for the next 3 months.
Forecast anomaly labels and operational alerts are derived from the regression forecast rather than emitted by a classifier in the forecasting path.

### Features

The model pipeline uses:

- climate covariates plus under-5 disease, service-delivery, reporting-quality, and population covariates
- lagged values
- rolling summaries
- seasonality features
- district-relative thresholds
- recent anomaly history
- trend and acceleration signals

## Outputs You Can Download

- classified observed CSV
- forecast CSV
- within-district anomaly metrics CSV
- between-district anomaly metrics CSV

## Screens

### Current Situation and Observed Maps

The app shows side-by-side summaries for:

- within-district anomaly
- between-districts anomaly
- operational alert

It also renders observed district maps when a GeoJSON is provided.

![Observed maps and current situation](Screenshots/Screenshot2png.png)

### Historical Anomaly Trend

The historical anomaly view shows how district counts move across anomaly levels over time.

![Historical anomaly trend](Screenshots/Screenshot3.png)

### Spearman Correlation Portrait

The drivers view includes a rank-based correlation portrait between GAM caseload and key covariates.

![Spearman correlation portrait](Screenshots/Screenshot4.png)

### Forecasts

The forecast tab includes district forecast charts, forecast tables, and forecast maps for each horizon.

![Forecast view](Screenshots/Screenshot6.png)

## Troubleshooting

### `streamlit: command not found`

```bash
pip install -r requirements.txt
```

or

```bash
pip install streamlit
```

### Port already in use

```bash
streamlit run app.py --server.port 8502
```

### Missing package errors

```bash
pip install -r requirements.txt
```

## Project Structure

```text
.
├── app.py
├── app/
├── data/
│   └── sample/
├── docs/
├── impact_metrics/
├── Screenshots/
└── requirements.txt
```

## Additional Documentation

- Technical notes: [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)
- Impact framing: [impact_metrics/IMPACT_METRICS.md](impact_metrics/IMPACT_METRICS.md)

## Notes and Limitations

- Forecast accuracy depends on data quality, completeness, and reporting consistency.
- Rule-based imputation can stabilize training and prediction, but large or systematic source-data gaps can still bias the model.
- Extreme GAM spikes are retained by default and only flagged for QC, so known reporting artifacts should still be reviewed during analysis.
- The anomaly labels are percentile-based early-warning indicators, not direct clinical diagnosis.
- Severity phases require direct GAM prevalence data and may be unavailable in many datasets.
- GeoJSON district naming mismatches can prevent map joins.
- Model outputs should be interpreted alongside field surveillance and program context.
