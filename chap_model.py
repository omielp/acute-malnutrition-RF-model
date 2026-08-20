import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MODEL_VERSION = "chap-rf-v9"
RISK_LEVELS = ["Low", "Moderate", "High", "Extreme"]
RISK_ORDER = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Extreme": 3,
}
RISK_EXPORT_ORDER = {
    "No Data": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Extreme": 4,
}
OPERATIONAL_ALERT_EXPORT_ORDER = {
    "No Data": 0,
    "Monitor": 1,
    "Alert": 2,
    "Respond": 3,
}
OUTBREAK_INDICATOR_MODES = {"none", "operational_alert_code"}
LEGACY_DIARRHEA_COVARIATES = ("diarrhea_acute", "diarrhea_persistent")
CLIMATE_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "mean_relative_humidity",
    "average_gpp",
]
CHILD_HEALTH_COVARIATES = [
    "malaria_confirmed_u5",
    "pneumonia_cases_u5",
    "diarrhea_u5",
    "low_birth_weight_babies",
    "sam_admissions_u5",
    "screened_u5",
]
RATE_COVARIATES = ["reporting_rate"]
SLOW_MOVING_COVARIATES = ["population_u5"]
DEFAULT_COVARIATES = [
    *CLIMATE_COVARIATES,
    *CHILD_HEALTH_COVARIATES,
    *RATE_COVARIATES,
    *SLOW_MOVING_COVARIATES,
]
SKEWED_COVARIATES = {
    *CHILD_HEALTH_COVARIATES,
    *SLOW_MOVING_COVARIATES,
}
MISSING_FLAG_SUFFIX = "_missing"
OUTLIER_FLAG_SUFFIX = "_outlier"
COUNT_COVARIATES = CHILD_HEALTH_COVARIATES
LAGGED_EXOG_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "mean_relative_humidity",
    "average_gpp",
    "malaria_confirmed_u5",
    "pneumonia_cases_u5",
    "diarrhea_u5",
    "low_birth_weight_babies",
    "sam_admissions_u5",
    "screened_u5",
    "reporting_rate",
]
COVARIATE_ALIASES = {
    "malaria_confirmed_u5": (
        "malaria_confirmed_u5",
        "malaria_confirmed",
    ),
    "pneumonia_cases_u5": (
        "pneumonia_cases_u5",
        "pneumonia_cases",
    ),
    "diarrhea_u5": (
        "diarrhea_u5",
        "diarrhea",
        *LEGACY_DIARRHEA_COVARIATES,
    ),
    "sam_admissions_u5": (
        "sam_admissions_u5",
        "sam_admissions",
        "sam_admission",
        "sam_cases_admitted",
        "sam_admitted",
    ),
    "screened_u5": (
        "screened_u5",
        "screening_assessment",
        "screening_assessments",
        "screened_children",
        "children_screened",
        "number_screened",
        "screening_assessment_u5",
    ),
    "population_u5": (
        "population_u5",
        "population",
        "under5population",
        "u5population",
    ),
    "reporting_rate": (
        "reporting_rate",
        "facility_reporting_rate",
        "district_reporting_rate",
        "reporting_completeness",
        "reporting_completeness_pct",
        "hf_reporting_rate",
    ),
}
BASE_FEATURES = [
    "Month",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Lag1",
    "Lag2",
    "Lag3",
    "Lag1_Z",
    "Roll3_Mean",
    "Roll3_Std",
    "Roll6_Mean",
    "Seasonal_Lag12",
    "Hist_Mean",
    "Hist_Std",
    "Hist_P90",
    "Hist_P95",
    "Lag1_Over_P95",
    "Lag1_Ratio_P95",
]
SCALED_FEATURES = [
    "Lag1",
    "Lag2",
    "Lag3",
    "Roll3_Mean",
    "Roll6_Mean",
    "Seasonal_Lag12",
    "Hist_Mean",
    "Hist_Std",
    "Hist_P90",
    "Hist_P95",
    "Lag1_Over_P95",
]
N_SAMPLES = 100
REPORTING_RATE_FLOOR = 0.6
PROXY_SCREEN_WEIGHT = 0.7
PROXY_SAM_WEIGHT = 0.3
PROXY_SOURCE_BASE_WEIGHTS = {
    "observed": 1.0,
    "proxy_combined": 0.6,
    "proxy_screened": 0.5,
    "proxy_sam_admissions": 0.35,
    "missing": 0.0,
}


def parse_time_period(value: str) -> pd.Timestamp:
    value = str(value).strip()
    for fmt in ("%Y-%m", "%B %Y", "%b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(value, format=fmt))
        except (TypeError, ValueError):
            pass
    return pd.to_datetime(value, errors="coerce")


def classify_risk(value: float, p50: float, p75: float, p90: float, p95: float) -> str:
    if pd.isna(value):
        return "No Data"
    if value < p75:
        return "Low"
    if value < p90:
        return "Moderate"
    if value < p95:
        return "High"
    return "Extreme"


def derive_operational_alert(
    wd_risk: str,
    xd_risk: str,
    severity_phase: str = "No Data",
    lower_severity_phase: str = "No Data",
) -> str:
    if wd_risk not in RISK_LEVELS or xd_risk not in RISK_LEVELS:
        return "No Data"
    if wd_risk == "Low" and xd_risk in {"Low", "Moderate"}:
        return "Monitor"
    if (
        (wd_risk == "Low" and xd_risk in {"High", "Extreme"})
        or (wd_risk == "Moderate" and xd_risk in {"Low", "Moderate"})
    ):
        return "Alert"
    if wd_risk in {"High", "Extreme"} or (wd_risk == "Moderate" and xd_risk in {"High", "Extreme"}):
        return "Respond"
    return "No Data"


def explain_operational_alert(
    wd_risk: str,
    xd_risk: str,
    severity_phase: str = "No Data",
    lower_severity_phase: str = "No Data",
) -> str:
    alert = derive_operational_alert(wd_risk, xd_risk, severity_phase, lower_severity_phase)
    if alert == "Monitor":
        return f"Monitor because within-district anomaly is {wd_risk} and between-districts anomaly is {xd_risk}."
    if alert == "Alert":
        return f"Alert because within-district anomaly is {wd_risk} and between-districts anomaly is {xd_risk}."
    if alert == "Respond":
        return f"Respond because within-district anomaly is {wd_risk} and between-districts anomaly is {xd_risk}."
    return "Operational alert unavailable because one or both anomaly classifications are missing."


def encode_risk_level(value: str) -> int:
    return int(RISK_EXPORT_ORDER.get(str(value), 0))


def encode_operational_alert(value: str) -> int:
    return int(OPERATIONAL_ALERT_EXPORT_ORDER.get(str(value), 0))


def normalize_outbreak_indicator_mode(value: str | None) -> str:
    mode = str(value or "none").strip().lower()
    if mode not in OUTBREAK_INDICATOR_MODES:
        raise ValueError(
            f"Unsupported outbreak indicator mode '{value}'. "
            f"Supported modes: {sorted(OUTBREAK_INDICATOR_MODES)}"
        )
    return mode


def env_flag_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def log_transform_covariate(name: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if name in SKEWED_COVARIATES:
        clipped = numeric.clip(lower=0)
        return np.log1p(clipped)
    return numeric


def sanitize_covariate_values(name: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if name in COUNT_COVARIATES:
        return numeric.mask(numeric < 0)
    if name == "reporting_rate":
        numeric = numeric.mask((numeric < 0) | (numeric > 100))
        return numeric.where(numeric <= 1, numeric / 100.0)
    if name == "population_u5":
        return numeric.mask(numeric <= 0)
    if name == "rainfall":
        return numeric.mask(numeric < 0)
    if name == "mean_relative_humidity":
        return numeric.mask((numeric < 0) | (numeric > 100))
    if name == "average_gpp":
        return numeric.mask(numeric < 0)
    return numeric


def _cap_series_with_mad(series: pd.Series, z: float = 5.0, min_obs: int = 4) -> tuple[pd.Series, pd.Series]:
    valid = series.dropna()
    flags = pd.Series(False, index=series.index, dtype=bool)
    if len(valid) < min_obs:
        return series, flags
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad < 1e-6:
        return series, flags
    lower = median - z * mad
    upper = median + z * mad
    flags = series.notna() & ((series < lower) | (series > upper))
    return series.clip(lower=lower, upper=upper), flags


def cap_outliers_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> tuple[pd.Series, pd.Series]:
    capped = pd.Series(index=df.index, dtype="float64")
    flags = pd.Series(0.0, index=df.index, dtype="float64")
    for _, index in df.groupby(group_col).groups.items():
        clipped, group_flags = _cap_series_with_mad(df.loc[index, value_col])
        capped.loc[index] = clipped
        flags.loc[index] = group_flags.astype(float)
    return capped, flags


def cap_outliers_causally(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> tuple[pd.Series, pd.Series]:
    """Cap a row using only earlier observations from its location."""
    ordered = df.sort_values([date_col, group_col]).copy()
    capped = pd.Series(np.nan, index=df.index, dtype="float64")
    flags = pd.Series(0.0, index=df.index, dtype="float64")
    history: dict[str, list[float]] = {}

    for _, batch in ordered.groupby(date_col, sort=True):
        for idx in batch.index:
            value = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
            group_key = str(df.loc[idx, group_col])
            prior_values = pd.Series(history.get(group_key, []), dtype="float64").dropna()
            if pd.isna(value) or len(prior_values) < 4:
                capped.loc[idx] = value
                continue
            median = float(prior_values.median())
            mad = float((prior_values - median).abs().median())
            if mad < 1e-6:
                capped.loc[idx] = value
                continue
            lower = median - 5.0 * mad
            upper = median + 5.0 * mad
            flags.loc[idx] = float(value < lower or value > upper)
            capped.loc[idx] = float(np.clip(value, lower, upper))

        for idx in batch.index:
            value = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
            if pd.notna(value):
                history.setdefault(str(df.loc[idx, group_col]), []).append(float(value))

    return capped, flags


def flag_outliers_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    flags = pd.Series(0.0, index=df.index, dtype="float64")
    for _, index in df.groupby(group_col).groups.items():
        _, group_flags = _cap_series_with_mad(df.loc[index, value_col])
        flags.loc[index] = group_flags.astype(float)
    return flags


def sanitize_target_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.mask(numeric < 0)


def add_target_qc_flags(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df["target_duplicate_qc"] = df.duplicated(subset=[group_col, "Date"], keep=False).astype(float)
    df["target_outlier_qc"] = flag_outliers_by_group(df, group_col, target_col)
    return df


def effective_reporting_rate(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").clip(lower=0, upper=1)
    numeric = numeric.where(numeric > 0)
    return numeric.fillna(1.0).clip(lower=REPORTING_RATE_FLOOR, upper=1.0)


def _median_or_nan(history: list[float]) -> float:
    if not history:
        return np.nan
    return float(np.median(np.asarray(history, dtype=float)))


def fill_group_month_median(
    df: pd.DataFrame,
    values: pd.Series,
    group_col: str,
    date_col: str,
) -> pd.Series:
    ordered = df.sort_values([date_col, group_col]).copy()
    filled = pd.Series(np.nan, index=df.index, dtype="float64")
    group_month_history: dict[tuple[str, int], list[float]] = {}
    group_history: dict[str, list[float]] = {}
    global_history: list[float] = []

    for current_date, batch in ordered.groupby(date_col, sort=True):
        batch_index = list(batch.index)
        for idx in batch_index:
            group_key = str(ordered.loc[idx, group_col])
            month_key = int(pd.Timestamp(current_date).month)
            estimate = _median_or_nan(group_month_history.get((group_key, month_key), []))
            if pd.isna(estimate):
                estimate = _median_or_nan(group_history.get(group_key, []))
            if pd.isna(estimate):
                estimate = _median_or_nan(global_history)
            filled.loc[idx] = estimate

        for idx in batch_index:
            value = pd.to_numeric(values.loc[idx], errors="coerce")
            if pd.isna(value):
                continue
            group_key = str(ordered.loc[idx, group_col])
            month_key = int(pd.Timestamp(current_date).month)
            group_month_history.setdefault((group_key, month_key), []).append(float(value))
            group_history.setdefault(group_key, []).append(float(value))
            global_history.append(float(value))

    return filled


def add_target_proxy_columns(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    date_col: str,
) -> pd.DataFrame:
    df = df.copy()
    observed = pd.to_numeric(df[target_col], errors="coerce")
    reporting_rate = (
        pd.to_numeric(df["reporting_rate"], errors="coerce")
        if "reporting_rate" in df.columns
        else pd.Series(1.0, index=df.index, dtype="float64")
    )
    effective_rate = effective_reporting_rate(reporting_rate)
    screened = pd.to_numeric(df["screened_u5"], errors="coerce") if "screened_u5" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
    sam = pd.to_numeric(df["sam_admissions_u5"], errors="coerce") if "sam_admissions_u5" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")

    screened_adjusted = screened / effective_rate
    sam_adjusted = sam / effective_rate

    screened_ratio_source = (observed / screened_adjusted.replace(0, np.nan)).where(observed.notna() & screened_adjusted.gt(0))
    sam_ratio_source = (observed / sam_adjusted.replace(0, np.nan)).where(observed.notna() & sam_adjusted.gt(0))

    screened_ratio = fill_group_month_median(df, screened_ratio_source, group_col, date_col)
    sam_ratio = fill_group_month_median(df, sam_ratio_source, group_col, date_col)

    gam_proxy_screened = (screened_adjusted * screened_ratio).where(screened_adjusted.gt(0)).clip(lower=0)
    gam_proxy_sam_admissions = (sam_adjusted * sam_ratio).where(sam_adjusted.gt(0)).clip(lower=0)

    gam_proxy_combined = pd.Series(np.nan, index=df.index, dtype="float64")
    both = gam_proxy_screened.notna() & gam_proxy_sam_admissions.notna()
    only_screened = gam_proxy_screened.notna() & ~gam_proxy_sam_admissions.notna()
    only_sam = gam_proxy_sam_admissions.notna() & ~gam_proxy_screened.notna()
    gam_proxy_combined.loc[both] = (
        PROXY_SCREEN_WEIGHT * gam_proxy_screened.loc[both]
        + PROXY_SAM_WEIGHT * gam_proxy_sam_admissions.loc[both]
    )
    gam_proxy_combined.loc[only_screened] = gam_proxy_screened.loc[only_screened]
    gam_proxy_combined.loc[only_sam] = gam_proxy_sam_admissions.loc[only_sam]

    target_source = pd.Series("missing", index=df.index, dtype="object")
    target_source.loc[only_sam] = "proxy_sam_admissions"
    target_source.loc[only_screened] = "proxy_screened"
    target_source.loc[both] = "proxy_combined"
    target_source.loc[observed.notna()] = "observed"

    reporting_for_conf = reporting_rate.fillna(1.0)
    sam_mask = target_source.eq("proxy_sam_admissions")
    screened_mask = target_source.eq("proxy_screened")
    combined_mask = target_source.eq("proxy_combined")
    target_proxy_confidence = pd.Series("No Data", index=df.index, dtype="object")
    target_proxy_confidence.loc[observed.notna()] = "Observed"
    target_proxy_confidence.loc[sam_mask] = "Low"
    target_proxy_confidence.loc[screened_mask] = np.where(
        reporting_for_conf.loc[screened_mask].ge(0.8),
        "Medium",
        "Low",
    )
    target_proxy_confidence.loc[combined_mask] = np.where(
        reporting_for_conf.loc[combined_mask].ge(0.8),
        "High",
        np.where(reporting_for_conf.loc[combined_mask].ge(0.6), "Medium", "Low"),
    )

    report_weight_factor = pd.Series(
        np.where(
            reporting_for_conf.ge(0.8),
            1.0,
            np.where(reporting_for_conf.ge(0.6), 0.85, 0.7),
        ),
        index=df.index,
        dtype="float64",
    )
    target_training_weight = target_source.map(PROXY_SOURCE_BASE_WEIGHTS).astype(float)
    proxy_mask = target_source.ne("observed")
    target_training_weight.loc[proxy_mask] = target_training_weight.loc[proxy_mask] * report_weight_factor.loc[proxy_mask]

    df[f"{target_col}_observed"] = observed
    df["effective_reporting_rate"] = effective_rate
    df["screened_u5_adjusted"] = screened_adjusted
    df["sam_admissions_u5_adjusted"] = sam_adjusted
    df["gam_proxy_screened"] = gam_proxy_screened
    df["gam_proxy_sam_admissions"] = gam_proxy_sam_admissions
    df["gam_proxy_combined"] = gam_proxy_combined
    df["target_source"] = target_source
    df["target_proxy_confidence"] = target_proxy_confidence
    df["target_is_observed"] = observed.notna().astype(float)
    df["target_training_weight"] = target_training_weight
    df[target_col] = observed.fillna(gam_proxy_combined)
    return df


def covariate_lag_feature_names(available_columns: list[str] | set[str]) -> list[str]:
    available = set(available_columns)
    names = []
    for covariate in LAGGED_EXOG_COVARIATES:
        if covariate in available:
            names.extend(
                [
                    f"{covariate}_Lag2",
                    f"{covariate}_Lag3",
                    f"{covariate}_Roll2_Mean",
                    f"{covariate}_Roll3_Mean",
                ]
            )
    return names


def impute_modeled_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present_covariates = [col for col in DEFAULT_COVARIATES if col in df.columns]

    for col in present_covariates:
        values = sanitize_covariate_values(col, df[col])
        df[col] = values
        df[f"{col}{MISSING_FLAG_SUFFIX}"] = values.isna().astype(float)
        df[col], df[f"{col}{OUTLIER_FLAG_SUFFIX}"] = cap_outliers_causally(df, "location", "Date", col)

    for col in [covariate for covariate in CLIMATE_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0)

    for col in [covariate for covariate in CHILD_HEALTH_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0).clip(lower=0)

    for col in [covariate for covariate in RATE_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0).clip(lower=0, upper=1)

    for col in [covariate for covariate in SLOW_MOVING_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill())
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0).clip(lower=0)

    return df


def resolve_covariate_series(df: pd.DataFrame, covariate: str) -> pd.Series | None:
    if covariate == "diarrhea_u5":
        source = pick_column(df, [covariate, "diarrhea"])
        if source is not None:
            return pd.to_numeric(df[source], errors="coerce")

        legacy_sources = [
            pick_column(df, [legacy_covariate])
            for legacy_covariate in LEGACY_DIARRHEA_COVARIATES
        ]
        legacy_sources = [source_name for source_name in legacy_sources if source_name is not None]
        if not legacy_sources:
            return None

        legacy_values = pd.concat(
            [pd.to_numeric(df[source_name], errors="coerce") for source_name in legacy_sources],
            axis=1,
        )
        return legacy_values.sum(axis=1, min_count=1)

    source = pick_column(df, [covariate, *COVARIATE_ALIASES.get(covariate, ())])
    if source is None:
        return None
    return pd.to_numeric(df[source], errors="coerce")


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_dataframe(df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input CSV is empty.")

    time_col = pick_column(df, ["time_period", "Time_Period"])
    location_col = pick_column(df, ["location", "Region_District", "District", "district"])
    target_col = pick_column(df, ["disease_cases", "Acut_Malnutrition"])

    missing = []
    if time_col is None:
        missing.append("time_period")
    if location_col is None:
        missing.append("location")
    if require_target and target_col is None:
        missing.append("disease_cases")
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    norm = pd.DataFrame(
        {
            "time_period": df[time_col].astype(str).str.strip(),
            "location": df[location_col].astype(str).str.strip(),
        }
    )
    norm["Date"] = norm["time_period"].apply(parse_time_period)
    norm = norm.dropna(subset=["Date"]).copy()

    if target_col is not None:
        norm["disease_cases"] = sanitize_target_values(df.loc[norm.index, target_col])
    elif require_target:
        raise ValueError("Observed inputs must include disease_cases.")
    else:
        norm["disease_cases"] = np.nan

    covariates = []
    for covariate in DEFAULT_COVARIATES:
        values = resolve_covariate_series(df, covariate)
        if values is None:
            continue
        norm[covariate] = pd.to_numeric(values.loc[norm.index], errors="coerce")
        covariates.append(covariate)

    norm = norm.sort_values(["location", "Date"]).reset_index(drop=True)
    norm = add_target_qc_flags(norm, group_col="location", target_col="disease_cases")
    norm = impute_modeled_covariates(norm)
    if require_target:
        norm = add_target_proxy_columns(norm, target_col="disease_cases", group_col="location", date_col="Date")
    for covariate in covariates:
        norm[covariate] = log_transform_covariate(covariate, norm[covariate])
    if require_target:
        norm = norm.dropna(subset=["disease_cases"]).copy()
    return norm


def infer_covariate_columns(df: pd.DataFrame) -> list[str]:
    covariates = [column for column in DEFAULT_COVARIATES if column in df.columns]
    flags = [
        f"{column}{MISSING_FLAG_SUFFIX}"
        for column in DEFAULT_COVARIATES
        if f"{column}{MISSING_FLAG_SUFFIX}" in df.columns
    ]
    outlier_flags = [
        f"{column}{OUTLIER_FLAG_SUFFIX}"
        for column in DEFAULT_COVARIATES
        if f"{column}{OUTLIER_FLAG_SUFFIX}" in df.columns
    ]
    return covariates + flags + outlier_flags


def transform_target(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def inverse_target(values: float | np.ndarray, scaler: dict) -> float | np.ndarray:
    return np.expm1(np.asarray(values) * scaler["target_std"] + scaler["target_mean"])


def build_fallback_scaler(cases: list[float] | np.ndarray) -> dict:
    series = np.asarray(cases, dtype=float)
    if series.size == 0:
        series = np.array([0.0], dtype=float)
    feature_mean = float(np.mean(series))
    feature_std = float(np.std(series))
    if feature_std < 1e-6:
        feature_std = 1.0
    target_log = transform_target(series)
    target_mean = float(np.mean(target_log))
    target_std = float(np.std(target_log))
    if target_std < 1e-6:
        target_std = 1.0
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def pad_history(values: list[float], min_length: int, fill_value: float) -> list[float]:
    history = list(values)
    seed = float(fill_value)
    if not history:
        history = [seed]
    while len(history) < min_length:
        history.insert(0, float(history[0]))
    return history


def build_training_features(df: pd.DataFrame, covariate_cols: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    rows = []
    scalers = {}
    derived_covariate_features = covariate_lag_feature_names(covariate_cols)

    for location in df["location"].unique():
        location_df = df[df["location"] == location].sort_values("Date").reset_index(drop=True)
        cases = location_df["disease_cases"].astype(float).to_numpy()
        if len(location_df) < 4:
            continue

        feature_mu = float(np.mean(cases[:-1]))
        feature_std = float(np.std(cases[:-1]))
        if feature_std < 1e-6:
            feature_std = 1.0

        target_log = transform_target(cases[3:])
        target_mean = float(np.mean(target_log))
        target_std = float(np.std(target_log))
        if target_std < 1e-6:
            target_std = 1.0

        scalers[location] = {
            "feature_mean": feature_mu,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }

        for i in range(3, len(location_df)):
            hist = cases[:i]
            roll3 = hist[-3:]
            roll6 = hist[-6:] if len(hist) >= 6 else hist
            hist_mean = float(np.mean(hist))
            hist_std = float(np.std(hist)) if np.std(hist) > 1e-6 else 1.0
            hist_p90, hist_p95 = np.percentile(hist, [90, 95])
            current_date = location_df.loc[i, "Date"]

            same_month_hist = location_df.loc[
                (location_df.index < i) & (location_df["Date"].dt.month == current_date.month),
                "disease_cases",
            ].astype(float)
            seasonal_lag = float(same_month_hist.iloc[-1]) if not same_month_hist.empty else float(hist[-1])

            row = {
                "location": location,
                "Date": current_date,
                "Month": current_date.month,
                "Quarter": current_date.quarter,
                "Month_Sin": np.sin(2 * np.pi * current_date.month / 12),
                "Month_Cos": np.cos(2 * np.pi * current_date.month / 12),
                "Lag1": float(hist[-1]),
                "Lag2": float(hist[-2]),
                "Lag3": float(hist[-3]),
                "Lag1_Z": (float(hist[-1]) - hist_mean) / hist_std,
                "Roll3_Mean": float(np.mean(roll3)),
                "Roll3_Std": float(np.std(roll3)),
                "Roll6_Mean": float(np.mean(roll6)),
                "Seasonal_Lag12": seasonal_lag,
                "Hist_Mean": hist_mean,
                "Hist_Std": hist_std,
                "Hist_P90": float(hist_p90),
                "Hist_P95": float(hist_p95),
                "Lag1_Over_P95": float(hist[-1] - hist_p95),
                "Lag1_Ratio_P95": float(hist[-1] / (hist_p95 + 1e-6)),
                "Target": float(cases[i]),
                "Target_Observed": float(location_df.loc[i, "disease_cases_observed"]) if pd.notna(location_df.loc[i, "disease_cases_observed"]) else np.nan,
                "Target_Is_Observed": float(location_df.loc[i, "target_is_observed"]) if "target_is_observed" in location_df.columns else 1.0,
                "Target_Training_Weight": float(location_df.loc[i, "target_training_weight"]) if "target_training_weight" in location_df.columns else 1.0,
                "Target_Transformed": float((target_log[i - 3] - target_mean) / target_std),
            }
            for covariate in covariate_cols:
                row[covariate] = float(location_df.loc[i - 1, covariate])
            for covariate in LAGGED_EXOG_COVARIATES:
                if covariate not in location_df.columns:
                    continue
                history = location_df.loc[i - 3:i - 1, covariate].astype(float).to_numpy()
                row[f"{covariate}_Lag2"] = float(history[-2])
                row[f"{covariate}_Lag3"] = float(history[-3])
                row[f"{covariate}_Roll2_Mean"] = float(np.mean(history[-2:]))
                row[f"{covariate}_Roll3_Mean"] = float(np.mean(history))
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("Not enough history to build training features. Each location needs at least 4 observations.")

    for location, scaler in scalers.items():
        mask = feature_df["location"] == location
        for feature in SCALED_FEATURES:
            feature_df.loc[mask, feature] = (
                feature_df.loc[mask, feature] - scaler["feature_mean"]
            ) / scaler["feature_std"]

    feature_cols = BASE_FEATURES + covariate_cols + derived_covariate_features
    return feature_df, feature_cols, scalers


def make_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=1,
        min_samples_split=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def train_model(train_data_path: str, model_path: str) -> None:
    df = normalize_dataframe(pd.read_csv(train_data_path), require_target=True)
    covariate_cols = infer_covariate_columns(df)
    feature_df, feature_cols, scalers = build_training_features(df, covariate_cols)

    model = make_regressor()
    sample_weight = feature_df["Target_Training_Weight"].fillna(1.0).to_numpy() if "Target_Training_Weight" in feature_df.columns else None
    model.fit(
        feature_df[feature_cols].to_numpy(),
        feature_df["Target_Transformed"].to_numpy(),
        sample_weight=sample_weight,
    )

    artifact = {
        "model_version": MODEL_VERSION,
        "feature_cols": feature_cols,
        "covariate_cols": covariate_cols,
        "scalers": scalers,
        "model": model,
    }
    with open(model_path, "wb") as file_obj:
        pickle.dump(artifact, file_obj)


def last_known_covariates(location_df: pd.DataFrame, covariate_cols: list[str]) -> dict:
    values = {}
    for covariate in covariate_cols:
        if covariate not in location_df.columns:
            values[covariate] = 0.0
            continue
        series = location_df[covariate].dropna()
        values[covariate] = float(series.iloc[-1]) if not series.empty else 0.0
    return values


def scale_feature_row(row: dict, scaler: dict) -> dict:
    scaled = row.copy()
    for feature in SCALED_FEATURES:
        scaled[feature] = (scaled[feature] - scaler["feature_mean"]) / scaler["feature_std"]
    return scaled


def sample_from_forest(model: RandomForestRegressor, feature_vector: np.ndarray, scaler: dict) -> np.ndarray:
    tree_predictions = np.array([tree.predict(feature_vector)[0] for tree in model.estimators_], dtype=float)
    tree_predictions = inverse_target(tree_predictions, scaler)
    tree_predictions = np.clip(tree_predictions, a_min=0, a_max=None)
    if len(tree_predictions) >= N_SAMPLES:
        positions = np.linspace(0, len(tree_predictions) - 1, N_SAMPLES).round().astype(int)
        return tree_predictions[positions]
    if len(tree_predictions) == 0:
        return np.zeros(N_SAMPLES)
    return np.resize(tree_predictions, N_SAMPLES)


def build_prediction_rows(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model_artifact: dict,
    include_risk_output: bool = False,
    outbreak_indicator_mode: str = "none",
) -> pd.DataFrame:
    outbreak_indicator_mode = normalize_outbreak_indicator_mode(outbreak_indicator_mode)
    include_outbreak_workaround = outbreak_indicator_mode == "operational_alert_code"
    compute_risk_outputs = include_risk_output or include_outbreak_workaround
    model = model_artifact["model"]
    covariate_cols = model_artifact["covariate_cols"]
    feature_cols = model_artifact["feature_cols"]
    scalers = model_artifact["scalers"]
    rows = []
    all_hist_vals = historic_df["disease_cases"].dropna().to_numpy(dtype=float)
    global_case_baseline = float(np.median(all_hist_vals)) if len(all_hist_vals) else 0.0
    global_scaler = build_fallback_scaler(all_hist_vals if len(all_hist_vals) else [0.0])

    locations = sorted(set(future_df["location"]))
    for location in locations:
        location_hist = historic_df[historic_df["location"] == location].sort_values("Date").reset_index(drop=True)
        location_future = future_df[future_df["location"] == location].sort_values("Date").reset_index(drop=True)
        if location_future.empty:
            continue

        hist_cases = location_hist["disease_cases"].astype(float).tolist()
        hist_cases = pad_history(hist_cases, min_length=3, fill_value=global_case_baseline)
        scaler = scalers.get(location, build_fallback_scaler(hist_cases if hist_cases else [global_case_baseline]))
        if scaler is None:
            scaler = global_scaler

        known_covariates = last_known_covariates(location_hist, covariate_cols) if not location_hist.empty else {}
        global_known_covariates = last_known_covariates(historic_df, covariate_cols) if not historic_df.empty else {}
        covariate_histories = {}
        for covariate in covariate_cols:
            location_series = (
                location_hist[covariate].dropna().astype(float).tolist()
                if covariate in location_hist.columns
                else []
            )
            fallback_covariate = known_covariates.get(covariate, global_known_covariates.get(covariate, 0.0))
            covariate_histories[covariate] = pad_history(location_series, min_length=1, fill_value=fallback_covariate)

        for i in range(len(location_future)):
            future_row = location_future.iloc[i]
            current_date = future_row["Date"]
            hist = np.array(hist_cases, dtype=float)
            roll3 = hist[-3:]
            roll6 = hist[-6:] if len(hist) >= 6 else hist
            hist_mean = float(np.mean(hist))
            hist_std = float(np.std(hist)) if np.std(hist) > 1e-6 else 1.0
            hist_p50, hist_p75, hist_p90, hist_p95 = np.percentile(hist, [50, 75, 90, 95])

            month_history = pd.Series(hist_cases[:-1] if len(hist_cases) > 1 else hist_cases)
            if len(location_hist) + i > 0:
                historic_dates = list(location_hist["Date"]) + list(location_future.loc[: i - 1, "Date"])
                same_month_values = [
                    hist_cases[pos]
                    for pos, date_value in enumerate(historic_dates)
                    if date_value.month == current_date.month
                ]
            else:
                same_month_values = []
            seasonal_lag = float(same_month_values[-1]) if same_month_values else float(hist[-1])

            feature_row = {
                "Month": current_date.month,
                "Quarter": current_date.quarter,
                "Month_Sin": np.sin(2 * np.pi * current_date.month / 12),
                "Month_Cos": np.cos(2 * np.pi * current_date.month / 12),
                "Lag1": float(hist[-1]),
                "Lag2": float(hist[-2]),
                "Lag3": float(hist[-3]),
                "Lag1_Z": (float(hist[-1]) - hist_mean) / hist_std,
                "Roll3_Mean": float(np.mean(roll3)),
                "Roll3_Std": float(np.std(roll3)),
                "Roll6_Mean": float(np.mean(roll6)),
                "Seasonal_Lag12": seasonal_lag,
                "Hist_Mean": hist_mean,
                "Hist_Std": hist_std,
                "Hist_P90": float(hist_p90),
                "Hist_P95": float(hist_p95),
                "Lag1_Over_P95": float(hist[-1] - hist_p95),
                "Lag1_Ratio_P95": float(hist[-1] / (hist_p95 + 1e-6)),
            }
            for covariate in covariate_cols:
                history = covariate_histories.get(covariate, [])
                if history:
                    feature_row[covariate] = float(history[-1])
                else:
                    feature_row[covariate] = float(known_covariates.get(covariate, global_known_covariates.get(covariate, 0.0)))
            for covariate in LAGGED_EXOG_COVARIATES:
                history = covariate_histories.get(covariate, [])
                fallback_value = feature_row.get(covariate, 0.0)
                padded_history = pad_history(history, min_length=3, fill_value=fallback_value)
                feature_row[f"{covariate}_Lag2"] = float(padded_history[-2])
                feature_row[f"{covariate}_Lag3"] = float(padded_history[-3])
                feature_row[f"{covariate}_Roll2_Mean"] = float(np.mean(padded_history[-2:]))
                feature_row[f"{covariate}_Roll3_Mean"] = float(np.mean(padded_history[-3:]))

            scaled_row = scale_feature_row(feature_row, scaler)
            feature_vector = np.array([[scaled_row[column] for column in feature_cols]], dtype=float)
            samples = sample_from_forest(model, feature_vector, scaler)
            point_forecast = float(np.mean(samples))
            hist_cases.append(point_forecast)

            for covariate in covariate_cols:
                value = future_row.get(covariate, np.nan)
                if pd.isna(value):
                    history = covariate_histories.get(covariate, [])
                    next_value = history[-1] if history else float(known_covariates.get(covariate, 0.0))
                else:
                    next_value = float(value)
                    known_covariates[covariate] = next_value
                covariate_histories.setdefault(covariate, []).append(next_value)

            if compute_risk_outputs:
                wd_risk = classify_risk(point_forecast, hist_p50, hist_p75, hist_p90, hist_p95)
            else:
                wd_risk = "No Data"

            row = {
                "time_period": str(future_row["time_period"]),
                "location": location,
                "_point_forecast_internal": point_forecast,
                "_wd_risk_internal": wd_risk,
            }
            row.update({f"sample_{index}": float(samples[index]) for index in range(len(samples))})
            if compute_risk_outputs:
                row["_wd_risk_internal"] = wd_risk
                if include_risk_output:
                    row.update(
                        {
                            "point_forecast": point_forecast,
                            "wd_risk": wd_risk,
                            "wd_risk_code": encode_risk_level(wd_risk),
                            "xd_risk": "No Data",
                            "xd_risk_code": 0,
                            "Operational_Alert": "No Data",
                            "Operational_Alert_Code": 0,
                            "Operational_Alert_Why": "",
                            "Composite_Risk": "No Data",
                            "Composite_Risk_Code": 0,
                        }
                    )
                if include_outbreak_workaround:
                    row.update(
                        {
                            "outbreak_indicator": 0,
                            "outbreak_indicator_label": "No Data",
                        }
                    )
            rows.append(row)

    if not rows:
        raise ValueError("No predictions were generated. Check that future locations exist in the historic/training data and each has at least 3 observations.")
    predictions = pd.DataFrame(rows)
    if not compute_risk_outputs or predictions.empty:
        return predictions.drop(
            columns=[col for col in ["_point_forecast_internal", "_wd_risk_internal"] if col in predictions.columns]
        )

    for time_period in predictions["time_period"].unique():
        mask = predictions["time_period"] == time_period
        point_forecasts = predictions.loc[mask, "_point_forecast_internal"].to_numpy(dtype=float)
        if len(point_forecasts) >= 3:
            xd_p50, xd_p75, xd_p90, xd_p95 = np.percentile(point_forecasts, [50, 75, 90, 95])
        else:
            xd_p50, xd_p75, xd_p90, xd_p95 = np.percentile(all_hist_vals, [50, 75, 90, 95])
        for idx in predictions[mask].index:
            point_forecast = float(predictions.loc[idx, "_point_forecast_internal"])
            if include_risk_output:
                wd_risk = str(predictions.loc[idx, "wd_risk"])
            else:
                wd_risk = str(predictions.loc[idx, "_wd_risk_internal"])
            xd_risk = classify_risk(point_forecast, xd_p50, xd_p75, xd_p90, xd_p95)
            operational_alert = derive_operational_alert(wd_risk, xd_risk)
            if include_risk_output:
                predictions.loc[idx, "xd_risk"] = xd_risk
                predictions.loc[idx, "xd_risk_code"] = encode_risk_level(xd_risk)
                predictions.loc[idx, "Operational_Alert"] = operational_alert
                predictions.loc[idx, "Operational_Alert_Code"] = encode_operational_alert(operational_alert)
                predictions.loc[idx, "Operational_Alert_Why"] = explain_operational_alert(wd_risk, xd_risk)
                predictions.loc[idx, "Composite_Risk"] = operational_alert
                predictions.loc[idx, "Composite_Risk_Code"] = encode_operational_alert(operational_alert)
            if include_outbreak_workaround:
                predictions.loc[idx, "outbreak_indicator"] = encode_operational_alert(operational_alert)
                predictions.loc[idx, "outbreak_indicator_label"] = operational_alert

    predictions = predictions.drop(
        columns=[col for col in ["_point_forecast_internal", "_wd_risk_internal"] if col in predictions.columns]
    )
    return predictions


def predict_model(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    output_path: str,
    include_risk_output: bool | None = None,
    outbreak_indicator_mode: str | None = None,
) -> None:
    if include_risk_output is None:
        include_risk_output = env_flag_true("CHAP_INCLUDE_RISK_OUTPUT")
    if outbreak_indicator_mode is None:
        outbreak_indicator_mode = os.getenv("CHAP_OUTBREAK_INDICATOR_MODE", "none")
    with open(model_path, "rb") as file_obj:
        artifact = pickle.load(file_obj)

    historic_df = normalize_dataframe(pd.read_csv(historic_data_path), require_target=True)
    future_df = normalize_dataframe(pd.read_csv(future_data_path), require_target=False)

    for covariate in artifact["covariate_cols"]:
        if covariate not in future_df.columns:
            if covariate.endswith(MISSING_FLAG_SUFFIX):
                future_df[covariate] = 1.0
            elif covariate.endswith(OUTLIER_FLAG_SUFFIX):
                future_df[covariate] = 0.0
            else:
                future_df[covariate] = np.nan

    predictions = build_prediction_rows(
        historic_df,
        future_df,
        artifact,
        include_risk_output=include_risk_output,
        outbreak_indicator_mode=outbreak_indicator_mode,
    )
    predictions.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CHAP-compatible acute malnutrition forecasting model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_data")
    train_parser.add_argument("model")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("model")
    predict_parser.add_argument("historic_data")
    predict_parser.add_argument("future_data")
    predict_parser.add_argument("out_file")
    predict_parser.add_argument(
        "--include-risk-output",
        action="store_true",
        help="Append point forecast, risk labels, and operational alert columns to the CHAP prediction output.",
    )
    predict_parser.add_argument(
        "--outbreak-indicator-mode",
        choices=sorted(OUTBREAK_INDICATOR_MODES),
        default="none",
        help="Optionally repurpose the outbreak_indicator output as Operational_Alert_Code for no-fork DHIS2 imports.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        train_model(args.train_data, args.model)
        return

    if args.command == "predict":
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        predict_model(
            args.model,
            args.historic_data,
            args.future_data,
            args.out_file,
            include_risk_output=args.include_risk_output,
            outbreak_indicator_mode=args.outbreak_indicator_mode,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
