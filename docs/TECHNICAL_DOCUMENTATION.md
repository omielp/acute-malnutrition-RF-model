# Technical Documentation

## Purpose

This application supports early warning and decision support for acute malnutrition by forecasting short-term district risk and visualizing spatial and temporal trends.

## Analytical Workflow

1. Load monthly district data.
2. Parse dates and standardize region/district labels.
3. Compute within-district risk using each district's own historical percentiles.
4. Compute between-district risk using peer district distributions by month.
5. Engineer lagged, rolling, seasonal, and district-relative features.
6. Train Random Forest regression to forecast acute malnutrition case counts.
7. Train Random Forest classifiers for risk evaluation.
8. Generate 3-month recursive forecasts.
9. Convert forecasts into risk categories.
10. Visualize results through maps, charts, risk tables, and model metrics.

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
- climate, disease, maternal-child health, and population covariates.

## Models

### Regression

Random Forest Regressor predicts acute malnutrition case counts for the next 3 months.

### Classification

Random Forest Classifier evaluates within-district and between-district risk labels. The classifier uses fixed risk-label ordering to avoid alphabetical label sorting errors.

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
- Risk categories are percentile-based and should complement, not replace, public health judgment.
- District name mismatches between CSV and GeoJSON can affect maps.
- Model outputs should be reviewed alongside field surveillance and nutrition program data.
