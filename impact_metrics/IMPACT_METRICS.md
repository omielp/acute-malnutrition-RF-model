# Impact Metrics

Use this document to describe the public health value of the application.

## Operational Impact

| Metric | Suggested Measurement |
|---|---|
| Districts monitored | Number of districts included in the dataset |
| Time period covered | First and last month of available data |
| Forecast horizon | Number of months forecasted |
| High-risk districts identified | Districts forecast as Alarm or Emergency |
| Regions monitored | Number of regions represented |
| Data indicators integrated | Number of climate, disease, nutrition, and population covariates |

## Model Performance Impact

| Metric | Why it matters |
|---|---|
| Sensitivity | Ability to detect true high-risk periods |
| Specificity | Ability to avoid false alarms |
| Balanced accuracy | Fairer measure when classes are imbalanced |
| Macro recall | Treats each risk class equally |
| MAE | Average forecast error in case counts |
| RMSE | Penalizes larger forecast errors |

## Public Health Use Cases

- Early identification of districts moving toward Alert, Alarm, or Emergency.
- Prioritization of nutrition surveillance.
- Targeted district or regional review.
- Comparison of district trends against national patterns.
- Support for planning outreach, commodities, staffing, and field validation.

## Example Impact Statement

This tool monitors 146 districts over 60 months and provides 3-month acute malnutrition forecasts. By combining district-specific risk thresholds, national peer comparisons, and spatial mapping, it supports earlier identification of districts requiring intensified nutrition surveillance and response.

## Evidence to Add Later

After field use, document:

- number of decisions supported;
- number of high-risk districts reviewed;
- response time improvement;
- program actions triggered;
- user feedback from nutrition/public health teams;
- comparison with routine reporting or field alerts.

