import calendar
import hashlib
import json
import os
import warnings
from io import StringIO

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = APP_DIR if os.path.isdir(os.path.join(APP_DIR, "data", "sample")) else os.path.dirname(APP_DIR)
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "Acute_Malnutrition_data_district_2024_2025.csv")
DEFAULT_GEOJSON_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "uganda-districts_ug.geojson")
MODEL_SCHEMA_VERSION = "gam-detection-per-1000-v23-loosened-quality-history"
FORECAST_MODEL_OPTIONS = ["MAE-Weighted RF + SSM Blend", "Random Forest", "State-Space Model"]
GAM_DETECTION_CI_WIDTH_MAX = 125.0
MIN_WITHIN_HISTORY_MONTHS = 9
MIN_PROVISIONAL_HISTORY_MONTHS = 6
RISK_THRESHOLD_PROFILES = {
    "Standard: P90 / P95": (90, 95),
    "Sensitive: P80 / P90": (80, 90),
}
LEGACY_DIARRHEA_COLUMNS = ("diarrhea_acute", "diarrhea_persistent")
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
    "screened_u5",
]
RATE_COVARIATES = ["reporting_rate"]
SLOW_MOVING_COVARIATES = ["population_u5"]
MODELED_COVARIATES = CLIMATE_COVARIATES + CHILD_HEALTH_COVARIATES + RATE_COVARIATES + SLOW_MOVING_COVARIATES
CORE_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "average_gpp",
    "malaria_confirmed_u5",
    "diarrhea_u5",
    "screened_u5",
    "reporting_rate",
    "population_u5",
]
CORE_GAM_FEATURES = [
    "Month_Sin",
    "Month_Cos",
    "Lag1",
    "Lag2",
    "Lag3",
    "Roll3_Mean",
    "Seasonal_Lag12",
]
MISSING_FLAG_SUFFIX = "_missing"
OUTLIER_FLAG_SUFFIX = "_outlier"
COUNT_COVARIATES = CHILD_HEALTH_COVARIATES
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
        "diarrhea_acute",
        "diarrhea_persistent",
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

RISK_LEVELS = ["Monitor", "Alert", "Respond"]
RISK_ORDER = {
    "Monitor": 0,
    "Alert": 1,
    "Respond": 2,
}
IPC_LEVELS = ["Acceptable", "Alert", "Serious", "Critical", "Extremely Critical"]
IPC_ORDER = {
    "Acceptable": 0,
    "Alert": 1,
    "Serious": 2,
    "Critical": 3,
    "Extremely Critical": 4,
}
RISK_COLORS = {
    "Monitor": "#2ecc71",
    "Alert": "#ffd11a",
    "Respond": "#d85c4a",
    "Acceptable": "#d4edda",
    "Serious": "#fde8d0",
    "Critical": "#f8d7da",
    "Extremely Critical": "#f5c6cb",
    "No Data": "#bdbdbd",
    "GAM Not Reported": "#9aa0a6",
    "Unreliable GAM Rate": "#d1d5db",
    "Insufficient Alert History": "#94a3b8",
    "Provisional Alert History": "#60a5fa",
}

ANOMALY_GUIDANCE = {
    "Monitor": "Within the usual range",
    "Alert": "Higher than usual",
    "Respond": "Exceptionally high",
}

SKEWED_COVS = CHILD_HEALTH_COVARIATES + SLOW_MOVING_COVARIATES

FEATURE_COLS_NEW = MODELED_COVARIATES
REPORTING_RATE_FLOOR = 0.6
GAM_RATE_SCALE = 1_000.0
PROXY_SCREEN_WEIGHT = 0.7
PROXY_SOURCE_BASE_WEIGHTS = {
    "observed": 1.0,
    "proxy_combined": 0.6,
    "proxy_screened": 0.5,
    "missing": 0.0,
}

NICE_NAMES = {
    "mean_temperature": "Mean Temperature",
    "rainfall": "Rainfall",
    "mean_relative_humidity": "Relative Humidity",
    "average_gpp": "Avg GPP",
    "malaria_confirmed_u5": "Malaria Cases (<5)",
    "pneumonia_cases_u5": "Pneumonia Cases (<5)",
    "diarrhea_u5": "Diarrhoea Cases (<5)",
    "low_birth_weight_babies": "Low Birth Weight Newborns",
    "screened_u5": "Children Screened (<5)",
    "reporting_rate": "Facility Reporting Rate",
    "population_u5": "Under-5 Population",
    "mean_temperature_missing": "Mean Temperature Missing",
    "rainfall_missing": "Rainfall Missing",
    "mean_relative_humidity_missing": "Relative Humidity Missing",
    "average_gpp_missing": "Avg GPP Missing",
    "malaria_confirmed_u5_missing": "Malaria Cases (<5) Missing",
    "pneumonia_cases_u5_missing": "Pneumonia Cases (<5) Missing",
    "diarrhea_u5_missing": "Diarrhoea Cases (<5) Missing",
    "low_birth_weight_babies_missing": "Low Birth Weight Newborns Missing",
    "screened_u5_missing": "Children Screened (<5) Missing",
    "reporting_rate_missing": "Facility Reporting Rate Missing",
    "population_u5_missing": "Under-5 Population Missing",
    "mean_temperature_outlier": "Mean Temperature Outlier",
    "rainfall_outlier": "Rainfall Outlier",
    "mean_relative_humidity_outlier": "Relative Humidity Outlier",
    "average_gpp_outlier": "Avg GPP Outlier",
    "malaria_confirmed_u5_outlier": "Malaria Cases (<5) Outlier",
    "pneumonia_cases_u5_outlier": "Pneumonia Cases (<5) Outlier",
    "diarrhea_u5_outlier": "Diarrhoea Cases (<5) Outlier",
    "low_birth_weight_babies_outlier": "Low Birth Weight Newborns Outlier",
    "screened_u5_outlier": "Children Screened (<5) Outlier",
    "reporting_rate_outlier": "Facility Reporting Rate Outlier",
    "population_u5_outlier": "Under-5 Population Outlier",
    "target_outlier_qc": "GAM Target Outlier QC",
    "target_duplicate_qc": "Duplicate District-Month QC",
    "target_precision_qc": "Imprecise GAM Detection Rate QC",
    "gam_detection_ci_lower": "GAM Detection Rate 95% CI Lower",
    "gam_detection_ci_upper": "GAM Detection Rate 95% CI Upper",
    "gam_detection_ci_width": "GAM Detection Rate 95% CI Width",
    "Acut_Malnutrition": "GAM Detection Rate per 1,000 Assessed Children",
    "Acut_Malnutrition_Observed": "Observed GAM Detection Rate per 1,000 Assessed Children",
    "GAM_Cases_Observed": "Observed GAM Cases",
    "target_source": "Target Source",
    "target_proxy_confidence": "Target Proxy Confidence",
    "target_is_observed": "Observed Target Flag",
    "target_training_weight": "Target Training Weight",
}

DISPLAY_LABELS = {
    "Date": "Month",
    "time_period": "Reported Period",
    "District_Name": "District",
    "Region_Label": "Region",
    "Region_District": "Region | District",
    "wd_risk": "Operational Alert (District Anomaly)",
    "WD_Risk": "Operational Alert (District Anomaly)",
    "Severity_Prevalence_Pct": "GAM Prevalence (%)",
    "Severity_Phase": "IPC AMN Phase",
    "Lower_Severity_Phase": "Lower-Bound IPC AMN Phase",
    "Severity_Basis": "Severity Basis",
    "Denominator_Source": "Severity Source",
    "Predicted": "Forecast GAM Detection Rate per 1,000 Assessed Children",
    "Lower_80": "Lower 80% Bound",
    "Upper_80": "Upper 80% Bound",
    "Months_Reported": "Months Reported",
    "Completeness_Pct": "Completeness (%)",
    "Zero_GAM_Months": "Zero GAM Months",
    "Outlier_GAM_Months": "Outlier GAM Months",
    "Missing_Population": "Missing Population",
    "Missing_Severity": "Missing Severity Rows",
    "Districts_Reported": "Districts Reported",
    "Total_GAM_Caseload": "District GAM Detection Rate Sum",
    "Zero_GAM_Districts": "Districts with Zero GAM",
    "Reporting_Completeness_Pct": "Reporting Completeness (%)",
    "target_source": "Target Source",
    "target_proxy_confidence": "Target Proxy Confidence",
}

_EXCLUDE_FROM_FEATURES = {
    "Region_District",
    "District_Name",
    "Region_Label",
    "Display_Name",
    "District",
    "Region",
    "time_period",
    "location",
    "Date",
    "Acut_Malnutrition",
    "wd_risk",
    "wd_score",
    "wd_p80",
    "wd_p95",
    "Severity_Prevalence_Pct",
    "Severity_Phase",
    "Severity_Basis",
    "Denominator_Source",
    "month",
    "quarter",
    "Acut_Malnutrition_Observed",
    "effective_reporting_rate",
    "screened_u5_adjusted",
    "gam_proxy_screened",
    "gam_proxy_combined",
    "target_source",
    "target_proxy_confidence",
    "target_is_observed",
    "target_training_weight",
    "target_outlier_qc",
    "target_duplicate_qc",
    "target_precision_qc",
    "gam_detection_ci_lower",
    "gam_detection_ci_upper",
    "gam_detection_ci_width",
    "diarrhea_acute",
    "diarrhea_persistent",
}

# =============================================================================
# UTILITIES
# =============================================================================

def parse_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    for fmt in ("%Y-%m", "%B %Y", "%b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(s, errors="coerce")


def display_label(col: str) -> str:
    if col in DISPLAY_LABELS:
        return DISPLAY_LABELS[col]
    if col in NICE_NAMES:
        return NICE_NAMES[col]
    return str(col).replace("_", " ")


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: display_label(c) for c in df.columns})


def build_phase_count_table(values: pd.Series, levels: list[str] | None = None, label: str = "Classification") -> pd.DataFrame:
    levels = levels or RISK_LEVELS
    counts = values.value_counts().reindex(levels).fillna(0).astype(int).reset_index()
    counts.columns = [label, "District Count"]
    return counts


def render_phase_pie_chart(counts: pd.DataFrame, level_col: str = "Level"):
    chart_df = counts[counts["District Count"] > 0].copy()
    if chart_df.empty:
        st.info("No districts in the current selection.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_df[level_col],
                values=chart_df["District Count"],
                hole=0.35,
                sort=False,
                marker=dict(colors=[RISK_COLORS.get(level, "#bdbdbd") for level in chart_df[level_col]]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} districts (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)


def ordered_levels_for_col(col: str) -> list[str]:
    if col in {"Severity_Phase", "Lower_Severity_Phase"}:
        return IPC_LEVELS
    return RISK_LEVELS


def classify_risk(value: float, p80: float, p95: float) -> str:
    """Map a district-relative GAM rate to the direct three-level alert scale."""
    if pd.isna(value):
        return "No Data"
    if value < p80:
        return "Monitor"
    if value < p95:
        return "Alert"
    return "Respond"


OBSERVED_MAP_LEVELS = [
    "Monitor",
    "Alert",
    "Respond",
    "GAM Not Reported",
    "Unreliable GAM Rate",
    "Provisional Alert History",
    "Insufficient Alert History",
    "No Data",
]


def add_observed_map_status(current_df: pd.DataFrame, full_df: pd.DataFrame, current_date) -> pd.DataFrame:
    """Expose why an observed district-month cannot receive an alert label."""
    result = current_df.copy()
    prior_valid = (
        full_df[
            (full_df["Date"] < current_date)
            & full_df["target_is_observed"].fillna(0).gt(0.5)
        ]
        .groupby("District_Name")
        .size()
    )
    result["Prior_Valid_GAM_Months"] = result["District_Name"].map(prior_valid).fillna(0).astype(int)
    result["Observed_Map_Status"] = result["wd_risk"]
    missing_gam = result["target_exclusion_reason"].eq("missing_gam")
    invalid_rate = result["Observed_Map_Status"].isna() & ~missing_gam & result["target_exclusion_reason"].ne("")
    insufficient_history = result["Observed_Map_Status"].isna() & ~missing_gam & ~invalid_rate
    result.loc[missing_gam, "Observed_Map_Status"] = "GAM Not Reported"
    result.loc[invalid_rate, "Observed_Map_Status"] = "Unreliable GAM Rate"
    provisional_history = insufficient_history & result["Prior_Valid_GAM_Months"].ge(MIN_PROVISIONAL_HISTORY_MONTHS)
    result.loc[provisional_history, "Observed_Map_Status"] = "Provisional Alert History"
    result.loc[insufficient_history, "Observed_Map_Status"] = "Insufficient Alert History"
    result.loc[provisional_history, "Observed_Map_Status"] = "Provisional Alert History"
    return result


def classify_ipc_amn_phase(prevalence_pct: float) -> str:
    """Classify GAM prevalence with IPC AMN phase thresholds."""
    if pd.isna(prevalence_pct):
        return "No Data"
    if prevalence_pct < 5:
        return "Acceptable"
    if prevalence_pct < 10:
        return "Alert"
    if prevalence_pct < 15:
        return "Serious"
    if prevalence_pct < 30:
        return "Critical"
    return "Extremely Critical"


def risk_score(label: str) -> int:
    return RISK_ORDER.get(label, -1)


def severity_score(label: str) -> int:
    return {
        "Acceptable": 0,
        "Alert": 1,
        "Serious": 2,
        "Critical": 3,
        "Extremely Critical": 3,
    }.get(label, -1)


def normalise_col_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_matching_col(columns: list[str], aliases: list[str]) -> str | None:
    norm_map = {normalise_col_name(col): col for col in columns}
    for alias in aliases:
        col = norm_map.get(alias)
        if col is not None:
            return col
    return None


def get_severity_setup(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    whz_col = find_matching_col(columns, [
        "gamwhz", "gamwhzpct", "gamwhzpercent", "gamwhzprevalence",
        "gamprevalence", "gampct", "gampercent", "globalacutemalnutritionprevalence",
    ])
    muac_col = find_matching_col(columns, [
        "gammuac", "gammuacpct", "gammuacpercent", "gammuacprevalence",
    ])
    total_pop_col = find_matching_col(columns, ["populationu5", "population", "totalpopulation", "under5population", "u5population"])

    if whz_col:
        return {
            "observed_method": "gam_whz",
            "prevalence_col": whz_col,
            "denominator_col": None,
            "population_col": total_pop_col,
            "basis": f"GAM prevalence from `{whz_col}`",
            "note": "Standards-based severity uses direct GAM prevalence, aligned to IPC AMN thresholds.",
            "forecast_method": "unavailable",
            "forecast_note": "Forecast severity phase unavailable because the model predicts GAM detection rate rather than direct GAM prevalence.",
        }
    if muac_col:
        return {
            "observed_method": "gam_muac",
            "prevalence_col": muac_col,
            "denominator_col": None,
            "population_col": total_pop_col,
            "basis": f"GAM MUAC prevalence from `{muac_col}`",
            "note": "Standards-based severity uses direct GAM MUAC prevalence. IPC recommends MUAC-based phase use only where WHZ is unavailable and supported by convergence of evidence.",
            "forecast_method": "unavailable",
            "forecast_note": "Forecast severity phase unavailable because the model predicts GAM detection rate rather than direct GAM prevalence.",
        }
    return {
        "observed_method": "unavailable",
        "prevalence_col": None,
        "denominator_col": None,
        "population_col": total_pop_col,
        "basis": "Unavailable",
        "note": "Standards-based severity requires direct GAM prevalence (WHZ/MUAC). This dataset currently supports anomaly-based warning labels and GAM-detection-rate forecasting, but not standards-based severity calculation.",
        "forecast_method": "unavailable",
        "forecast_note": "Forecast severity phase unavailable because the model predicts GAM detection rate rather than direct GAM prevalence.",
    }


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def get_denominator_source_label(severity_setup: dict) -> str:
    method = severity_setup["observed_method"]
    if method == "gam_whz":
        return f"Reported GAM prevalence `{severity_setup['prevalence_col']}`"
    if method == "gam_muac":
        return f"Reported GAM MUAC prevalence `{severity_setup['prevalence_col']}`"
    return "Unavailable"


def get_denominator_source_short(severity_setup: dict) -> str:
    method = severity_setup["observed_method"]
    if method in {"gam_whz", "gam_muac"}:
        return "Direct GAM"
    return "Unavailable"


def build_methodology_df(df: pd.DataFrame) -> pd.DataFrame:
    severity_setup = get_severity_setup(df)
    rows = [
        {
            "Component": "Outcome",
            "Definition": "Observed GAM detection rate: GAM cases / children assessed x 1,000.",
        },
        {
            "Component": "Within-district anomaly",
            "Definition": "Compares each district with its own historical distribution after at least 9 valid prior months.",
        },
        {
            "Component": "Target reliability",
            "Definition": "Excludes observed rates with a 95% Wilson interval wider than 125 per 1,000; the GAM value is not imputed.",
        },
        {
            "Component": "Data note",
            "Definition": severity_setup["note"],
        },
    ]
    return pd.DataFrame(rows)


def build_alert_actions_df(p80: int, p95: int) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Operational Alert": "Monitor",
            "Percentile": f"<{p80}th percentile",
            "Interpretation": "Within the usual range",
            "Suggested action": "Routine monitoring",
        },
        {
            "Operational Alert": "Alert",
            "Percentile": f"{p80}th-<{p95}th percentile",
            "Interpretation": "Higher than usual",
            "Suggested action": "Investigate and validate the signal",
        },
        {
            "Operational Alert": "Respond",
            "Percentile": f">={p95}th percentile",
            "Interpretation": "Exceptionally high",
            "Suggested action": "Escalate for rapid assessment and response planning",
        },
    ])


def _count_iqr_outliers(values: pd.Series) -> int:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 4:
        return 0
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    upper = q3 + 1.5 * iqr
    return int((vals > upper).sum())


def compute_data_quality_tables(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    severity_setup = get_severity_setup(df)
    pop_col = severity_setup["population_col"]
    target_col = "Acut_Malnutrition_Observed" if "Acut_Malnutrition_Observed" in df.columns else "Acut_Malnutrition"
    districts = sorted(df["District_Name"].unique())
    expected_months = len(pd.period_range(df["Date"].min(), df["Date"].max(), freq="M"))
    total_districts = len(districts)

    district_rows = []
    for district in districts:
        d = df[df["District_Name"] == district].sort_values("Date")
        observed_target = pd.to_numeric(d[target_col], errors="coerce")
        observed_months = int(d.loc[observed_target.notna(), "Date"].nunique())
        pop_missing = 0
        duplicate_target_rows = int(d.duplicated(subset=["Date"]).sum())
        outlier_target_rows = (
            int(pd.to_numeric(d["target_outlier_qc"], errors="coerce").fillna(0).sum())
            if "target_outlier_qc" in d.columns
            else _count_iqr_outliers(observed_target)
        )
        if pop_col is not None and pop_col in d.columns:
            missing_col = f"{pop_col}{MISSING_FLAG_SUFFIX}"
            if missing_col in d.columns:
                pop_missing = int(pd.to_numeric(d[missing_col], errors="coerce").fillna(0).sum())
            else:
                pop_missing = int(pd.to_numeric(d[pop_col], errors="coerce").isna().sum())

        district_rows.append({
            "District": district,
            "Months_Reported": observed_months,
            "Completeness_Pct": round(observed_months / expected_months * 100, 1) if expected_months else np.nan,
            "Zero_GAM_Months": int(observed_target.eq(0).sum()),
            "Outlier_GAM_Months": outlier_target_rows,
            "Imprecise_Target_Rows": int(pd.to_numeric(d.get("target_precision_qc", pd.Series(0.0, index=d.index)), errors="coerce").fillna(0).sum()),
            "Duplicate_Target_Rows": duplicate_target_rows,
            "Missing_Population": pop_missing,
            "Missing_Severity": int(d["Severity_Prevalence_Pct"].isna().sum()),
            "Denominator_Source": d["Denominator_Source"].iloc[0] if "Denominator_Source" in d.columns and not d.empty else "Unavailable",
        })

    district_quality = pd.DataFrame(district_rows).sort_values(
        ["Completeness_Pct", "Duplicate_Target_Rows", "Missing_Severity", "Outlier_GAM_Months"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    monthly = (
        df.groupby("Date", sort=True)
        .apply(
            lambda g: pd.Series({
                "Districts_Reported": int(g.loc[pd.to_numeric(g[target_col], errors="coerce").notna(), "District_Name"].nunique()),
                "Total_GAM_Caseload": float(pd.to_numeric(g[target_col], errors="coerce").sum()),
                "Zero_GAM_Districts": int(pd.to_numeric(g[target_col], errors="coerce").eq(0).sum()),
                "Missing_Severity": int(g["Severity_Prevalence_Pct"].isna().sum()),
                "Target_Outlier_Rows": int(pd.to_numeric(g["target_outlier_qc"], errors="coerce").fillna(0).sum()) if "target_outlier_qc" in g.columns else 0,
                "Imprecise_Target_Rows": int(pd.to_numeric(g.get("target_precision_qc", pd.Series(0.0, index=g.index)), errors="coerce").fillna(0).sum()),
                "Duplicate_Target_Rows": int(pd.to_numeric(g["target_duplicate_qc"], errors="coerce").fillna(0).sum()) if "target_duplicate_qc" in g.columns else 0,
            })
        )
        .reset_index()
        .sort_values("Date")
    )
    monthly["Reporting_Completeness_Pct"] = (monthly["Districts_Reported"] / total_districts * 100).round(1) if total_districts else np.nan
    monthly["Date"] = monthly["Date"].dt.strftime("%Y-%m")

    duplicate_rows = int(df.duplicated(subset=["District_Name", "Date"]).sum())
    summary = {
        "expected_months": expected_months,
        "median_completeness": float(district_quality["Completeness_Pct"].median()) if not district_quality.empty else 0.0,
        "zero_gam_months": int(district_quality["Zero_GAM_Months"].sum()) if not district_quality.empty else 0,
        "outlier_gam_months": int(district_quality["Outlier_GAM_Months"].sum()) if not district_quality.empty else 0,
        "imprecise_target_rows": int(pd.to_numeric(df.get("target_precision_qc", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).sum()),
        "missing_severity_rows": int(df["Severity_Prevalence_Pct"].isna().sum()),
        "duplicate_rows": duplicate_rows,
    }
    return summary, district_quality, monthly


def compute_target_proxy_tables(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    source_series = (
        df["target_source"].fillna("missing")
        if "target_source" in df.columns
        else pd.Series("observed", index=df.index, dtype="object")
    )
    confidence_series = (
        df["target_proxy_confidence"].fillna("No Data")
        if "target_proxy_confidence" in df.columns
        else pd.Series("Observed", index=df.index, dtype="object")
    )

    summary = {
        "observed_rows": int(source_series.eq("observed").sum()),
        "proxy_rows": int(source_series.ne("observed").sum()),
        "proxy_combined_rows": int(source_series.eq("proxy_combined").sum()),
        "proxy_screened_rows": int(source_series.eq("proxy_screened").sum()),
    }

    proxy_table = (
        pd.DataFrame({
            "Target_Source": source_series,
            "Target_Proxy_Confidence": confidence_series,
        })
        .value_counts()
        .rename("Rows")
        .reset_index()
        .sort_values(["Rows", "Target_Source"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary, proxy_table


def apply_log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def transform_regression_target(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def inverse_regression_target(values: float | np.ndarray, scaler: dict | None = None) -> float | np.ndarray:
    """Restore the causal log1p GAM target; scaler is retained for call compatibility."""
    restored = np.expm1(np.asarray(values))
    restored = np.clip(restored, a_min=0, a_max=None)
    if np.isscalar(values):
        return float(restored)
    return restored


def guess_name_col(columns: list[str]) -> str:
    keywords = ["district", "dist", "name", "admin", "adm"]
    non_geom = [c for c in columns if c != "geometry"]
    for kw in keywords:
        for col in non_geom:
            if kw in col.lower():
                return col
    return non_geom[0] if non_geom else ""


def safe_geojson(gdf: gpd.GeoDataFrame) -> dict:
    g = gdf.copy()
    g["geometry"] = g["geometry"].simplify(0.01, preserve_topology=True)
    for col in g.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(g[col]):
            g[col] = g[col].astype(str)
        elif g[col].dtype == object:
            g[col] = g[col].fillna("Unknown").astype(str)
        elif pd.api.types.is_float_dtype(g[col]):
            g[col] = g[col].fillna(0).replace([np.inf, -np.inf], 0)
    return json.loads(g.to_json(na="drop", show_bbox=False))


def colour_risk_df(df_styler, columns):
    risk_bg = {
        "Monitor": RISK_COLORS["Monitor"],
        "Alert": RISK_COLORS["Alert"],
        "Respond": RISK_COLORS["Respond"],
        "Extremely Critical": "#f5c6cb",
        "Critical": "#f8d7da",
        "Serious": "#fde8d0",
        "Acceptable": "#d4edda",
        "No Data": "#f0f0f0",
    }
    risk_fg = {
        "Monitor": "#0b2e13",
        "Alert": "#4d3d00",
        "Respond": "#ffffff",
        "Extremely Critical": "#58151c",
        "Critical": "#721c24",
        "Serious": "#8a4500",
        "Acceptable": "#155724",
        "No Data": "#555",
    }

    def _style(val):
        if val in risk_bg:
            return f"background-color:{risk_bg[val]};color:{risk_fg[val]};font-weight:bold"
        return ""

    return df_styler.map(_style, subset=columns)


def compute_spearman_correlations(data_slice: pd.DataFrame):
    target = "Acut_Malnutrition"
    candidates = [c for c in FEATURE_COLS_NEW if c in data_slice.columns]
    rows = []

    for col in candidates:
        pair = data_slice[[target, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 6 or pair[target].nunique() < 2 or pair[col].nunique() < 2:
            continue
        rho = pair[target].corr(pair[col], method="spearman")
        if pd.notna(rho):
            rows.append({
                "Variable": col,
                "Display": NICE_NAMES.get(col, col),
                "Spearman rho": float(rho),
                "Abs rho": float(abs(rho)),
                "N": int(len(pair)),
            })

    return pd.DataFrame(rows).sort_values("Abs rho", ascending=False).reset_index(drop=True)


def draw_spearman_radial(corr_df: pd.DataFrame):
    if corr_df.empty:
        return None

    plot_df = corr_df.head(10).copy()
    n = len(plot_df)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = plot_df["Abs rho"].to_numpy()
    signed = plot_df["Spearman rho"].to_numpy()
    labels = plot_df["Display"].tolist()

    fig, ax = plt.subplots(
        figsize=(7.1, 7.1),
        subplot_kw={"projection": "polar"},
        facecolor="#f7f4ef",
    )
    ax.set_facecolor("#f7f4ef")

    max_r = max(0.40, float(vals.max()) + 0.06)
    for rv in np.arange(0.1, max_r, 0.1):
        ax.plot(
            np.linspace(0, 2 * np.pi, 360),
            [rv] * 360,
            color="#2f2f2f",
            lw=0.55,
            alpha=0.15,
            zorder=1,
        )
        ax.text(
            np.pi / 2,
            rv + 0.006,
            f"r={rv:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555",
            fontfamily="monospace",
            alpha=0.80,
        )

    pos_color = "#08aeca"
    neg_color = "#ef4056"
    width = 2 * np.pi / n * 0.58

    def wrap_label(label: str, width: int = 12) -> str:
        words = str(label).split()
        if not words:
            return str(label)
        lines = []
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if len(trial) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        if len(lines) > 2:
            lines = lines[:2]
            if not lines[-1].endswith("..."):
                lines[-1] = lines[-1][: max(0, width - 3)].rstrip() + "..."
        return "\n".join(lines)

    for angle, rho_abs, rho, label in zip(angles, vals, signed, labels):
        color = pos_color if rho >= 0 else neg_color
        for glow_width, alpha in [(width * 2.7, 0.05), (width * 1.8, 0.10), (width, 0.95)]:
            ax.bar(
                angle,
                rho_abs,
                width=glow_width,
                bottom=0.035,
                color=color,
                alpha=alpha,
                edgecolor="none",
                zorder=3,
            )
        ax.scatter(
            angle,
            rho_abs + 0.035,
            s=70,
            color=color,
            edgecolors="#111",
            linewidths=1.1,
            zorder=6,
        )
        ax.text(
            angle,
            rho_abs + 0.075,
            f"{rho:+.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
            fontfamily="monospace",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground="#f7f4ef")],
        )
        short = wrap_label(label)
        ax.text(
            angle,
            max(rho_abs + 0.11, 0.18),
            short,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#303040",
            fontweight="bold",
            fontfamily="monospace",
            linespacing=1.05,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="#f7f4ef")],
        )

    for size, alpha in [(430, 0.04), (260, 0.10), (130, 0.22), (70, 0.95)]:
        ax.scatter(0, 0, s=size, color="#f5c518", alpha=alpha, zorder=9)
    ax.text(
        0,
        0,
        "malnutrition",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#101010",
        fontfamily="monospace",
        zorder=10,
    )

    ax.set_ylim(0, max_r + 0.14)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.05, right=0.95)
    return fig


def render_compare_time_series(df: pd.DataFrame, reference_label: str = "All Districts"):
    st.subheader(f"Compare Districts or Regions Against {reference_label} Reference")
    compare_mode = st.radio(
        "Compare by",
        ["Districts", "Regions"],
        horizontal=True,
        key="compare_mode",
    )

    national = (
        df.groupby("Date")["Acut_Malnutrition"]
        .agg(Total="sum", Mean="mean", Median="median")
        .reset_index()
        .sort_values("Date")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national["Date"],
        y=national["Total"],
        mode="lines",
        name=f"{reference_label} - Total (right axis)",
        yaxis="y2",
        line=dict(color="#34495e", width=2.2, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=national["Date"],
        y=national["Mean"],
        mode="lines",
        name=f"{reference_label} - Mean (left axis)",
        yaxis="y1",
        line=dict(color="#95a5a6", width=2.8),
        hovertemplate="<b>%{x|%b %Y}</b><br>Mean per district: %{y:,.1f}<extra></extra>",
    ))

    palette = px.colors.qualitative.Bold
    if compare_mode == "Districts":
        options = sorted(df["Display_Name"].unique())
        defaults = options[: min(3, len(options))]
        selected = st.multiselect("Select districts", options=options, default=defaults, key="compare_districts")
        for i, display_name in enumerate(selected):
            sub = df[df["Display_Name"] == display_name].sort_values("Date")
            fig.add_trace(go.Scatter(
                x=sub["Date"],
                y=sub["Acut_Malnutrition"],
                mode="lines+markers",
                name=display_name,
                yaxis="y1",
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x|%b %Y}</b><br>Cases: %{y:,.0f}<extra></extra>",
            ))
        y1_title = "Cases per District"
    else:
        options = sorted(df["Region_Label"].dropna().unique())
        defaults = options[: min(3, len(options))]
        selected = st.multiselect("Select regions", options=options, default=defaults, key="compare_regions")
        region_month = (
            df[df["Region_Label"].isin(selected)]
            .groupby(["Date", "Region_Label"])["Acut_Malnutrition"]
            .agg("mean")
            .reset_index()
            .sort_values("Date")
        )
        for i, region in enumerate(selected):
            sub = region_month[region_month["Region_Label"] == region]
            fig.add_trace(go.Scatter(
                x=sub["Date"],
                y=sub["Acut_Malnutrition"],
                mode="lines+markers",
                name=region,
                yaxis="y1",
                line=dict(color=palette[i % len(palette)], width=2.2),
                marker=dict(size=5),
                hovertemplate="<b>%{x|%b %Y}</b><br>Mean cases per district: %{y:,.1f}<extra></extra>",
            ))
        y1_title = "Mean Cases per District in Region"

    fig.update_layout(
        height=520,
        hovermode="x unified",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title=y1_title, showgrid=True, gridcolor="#ececec"),
        yaxis2=dict(title=f"Total - {reference_label}", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_historical_risk_trend(df: pd.DataFrame, scope_label: str = "All Districts"):
    st.subheader(f"Historical Anomaly Trend - {scope_label}")
    risk_col = "wd_risk"

    monthly = (
        df.groupby(["Date", risk_col]).size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_LEVELS, fill_value=0)
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    total = monthly[RISK_LEVELS].sum(axis=1).replace(0, np.nan)
    for lvl in RISK_LEVELS:
        monthly[f"{lvl}_pct"] = monthly[lvl] / total * 100

    monthly["_n0"] = 0
    cumulative = monthly["_n0"]
    for i, lvl in enumerate(RISK_LEVELS, start=1):
        cumulative = cumulative + monthly[lvl]
        monthly[f"_n{i}"] = cumulative

    focus_level = RISK_LEVELS[-1]
    monthly[f"{focus_level}_6m"] = monthly[f"{focus_level}_pct"].rolling(6, min_periods=1, center=True).mean()
    total_max = float(monthly[f"_n{len(RISK_LEVELS)}"].max())
    monthly[f"{focus_level}_6m_scaled"] = monthly[f"{focus_level}_6m"] / 100 * total_max

    fig = go.Figure()
    for i, lvl in enumerate(RISK_LEVELS, start=1):
        bottom = f"_n{i - 1}"
        top = f"_n{i}"
        color = RISK_COLORS[lvl]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=monthly["Date"],
            y=monthly[top],
            mode="lines",
            line=dict(color=color, width=0.8),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([monthly["Date"], monthly["Date"][::-1]]),
            y=pd.concat([monthly[top], monthly[bottom][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.78)",
            line=dict(color="rgba(0,0,0,0)"),
            name=lvl,
            customdata=np.stack([
                monthly[lvl],
                monthly[f"{lvl}_pct"].fillna(0).round(1),
            ], axis=-1).tolist() * 2,
            hovertemplate="<b>%{x|%b %Y}</b><br>" + lvl + ": %{customdata[0]} districts (%{customdata[1]}%)<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=monthly["Date"],
        y=monthly[f"{focus_level}_6m_scaled"],
        mode="lines",
        name=f"{focus_level} % (6m avg)",
        line=dict(color="#c0392b", width=2.4, dash="dash"),
        customdata=monthly[f"{focus_level}_6m"].fillna(0),
        hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{focus_level} trend: %{{customdata:.1f}}%<extra></extra>",
    ))

    peak_series = monthly[f"{focus_level}_pct"].dropna() if not monthly.empty else pd.Series(dtype=float)
    if not peak_series.empty:
        peak_idx = peak_series.idxmax()
        peak_date = monthly.loc[peak_idx, "Date"]
        peak_pct = monthly.loc[peak_idx, f"{focus_level}_pct"]
        fig.add_vline(x=peak_date, line_dash="dot", line_color="#e74c3c", line_width=1.8, opacity=0.75)
        fig.add_annotation(
            x=peak_date,
            y=total_max * 1.04,
            text=f"Peak: {peak_pct:.0f}% {focus_level}",
            showarrow=False,
            font=dict(size=10, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="#e74c3c",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        height=500,
        xaxis_title="",
        yaxis_title="Number of Districts",
        hovermode="x unified",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if monthly.empty:
        latest_month = "No data"
        avg_e = np.nan
        peak_e = np.nan
        recent = np.nan
        prior = np.nan
    else:
        latest_month = monthly["Date"].iloc[-1].strftime("%b %Y")
        avg_e = monthly[f"{focus_level}_pct"].mean()
        peak_e = monthly[f"{focus_level}_pct"].max()
        recent = monthly[f"{focus_level}_pct"].tail(3).mean()
        prior = (
            monthly[f"{focus_level}_pct"].iloc[-9:-3].mean()
            if len(monthly) >= 9 else monthly[f"{focus_level}_pct"].mean()
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest month", latest_month)
    k2.metric(f"Avg % {focus_level}", f"{avg_e:.1f}%" if pd.notna(avg_e) else "N/A")
    k3.metric(f"Peak {focus_level}", f"{peak_e:.1f}%" if pd.notna(peak_e) else "N/A")
    k4.metric(
        "Recent 3m trend",
        "Rising" if pd.notna(recent) and pd.notna(prior) and recent > prior else "Falling" if pd.notna(recent) and pd.notna(prior) else "N/A",
        delta=f"{recent - prior:+.1f}pp" if pd.notna(recent) and pd.notna(prior) else None,
        delta_color="inverse",
    )


# =============================================================================
# DATA LOADING AND DISTRICT RISK CLASSIFICATION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, _hash: str, risk_thresholds: tuple[int, int]) -> pd.DataFrame:
    content = file_bytes.decode("utf-8")
    return load_data_from_df(pd.read_csv(StringIO(content)), risk_thresholds)


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, risk_thresholds: tuple[int, int]) -> pd.DataFrame:
    return load_data_from_df(pd.read_csv(path), risk_thresholds)


def find_column(df: pd.DataFrame, candidates: tuple[str, ...] | list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def harmonize_diarrhea_covariate(df: pd.DataFrame) -> pd.DataFrame:
    legacy_cols = [col for col in LEGACY_DIARRHEA_COLUMNS if col in df.columns]
    if "diarrhea_u5" not in df.columns:
        direct_source = find_column(df, ["diarrhea_u5", "diarrhea"])
        if direct_source is not None:
            df["diarrhea_u5"] = pd.to_numeric(df[direct_source], errors="coerce")
    if "diarrhea_u5" not in df.columns and legacy_cols:
        legacy_values = pd.concat(
            [pd.to_numeric(df[col], errors="coerce") for col in legacy_cols],
            axis=1,
        )
        df["diarrhea_u5"] = legacy_values.sum(axis=1, min_count=1)
    if legacy_cols:
        df = df.drop(columns=legacy_cols)
    return df


def harmonize_optional_covariates(df: pd.DataFrame) -> pd.DataFrame:
    for canonical, aliases in COVARIATE_ALIASES.items():
        if canonical in df.columns:
            continue
        source = find_column(df, aliases)
        if source is not None:
            df[canonical] = df[source]
    return df


def _district_month_median(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> pd.Series:
    return df.groupby([group_col, df[date_col].dt.month])[value_col].transform("median")


def _district_median(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    return df.groupby(group_col)[value_col].transform("median")


def _fill_remaining(series: pd.Series) -> pd.Series:
    median_value = series.median(skipna=True)
    fill_value = 0.0 if pd.isna(median_value) else float(median_value)
    return series.fillna(fill_value)


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


def cap_outliers_causally(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    value_col: str,
) -> tuple[pd.Series, pd.Series]:
    """Cap a covariate using only earlier values from the same district."""
    ordered = df.sort_values([date_col, group_col])
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
            lower, upper = median - 5.0 * mad, median + 5.0 * mad
            flags.loc[idx] = float(value < lower or value > upper)
            capped.loc[idx] = float(np.clip(value, lower, upper))

        # Add the raw current-month observations only after all rows are capped.
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
    """Retain observed GAM only; proxy targets are deliberately disabled."""
    df = df.copy()
    observed = pd.to_numeric(df[target_col], errors="coerce")
    df[f"{target_col}_Observed"] = observed
    df["target_source"] = np.where(observed.notna(), "observed", "missing")
    df["target_proxy_confidence"] = np.where(observed.notna(), "Observed", "No Data")
    df["target_is_observed"] = observed.notna().astype(float)
    df["target_training_weight"] = df["target_is_observed"]
    df[target_col] = observed
    return df


def wilson_interval_per_1000(cases: pd.Series, assessed: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return 95% Wilson interval bounds for an observed detection rate."""
    cases = pd.to_numeric(cases, errors="coerce")
    assessed = pd.to_numeric(assessed, errors="coerce")
    valid = cases.notna() & cases.ge(0) & assessed.gt(0) & cases.le(assessed)
    proportion = (cases / assessed).where(valid)
    z = 1.959963984540054
    denominator = 1.0 + (z ** 2 / assessed.where(valid))
    center = (proportion + z ** 2 / (2.0 * assessed.where(valid))) / denominator
    half_width = z * np.sqrt(
        (proportion * (1.0 - proportion) + z ** 2 / (4.0 * assessed.where(valid)))
        / assessed.where(valid)
    ) / denominator
    return (center - half_width) * GAM_RATE_SCALE, (center + half_width) * GAM_RATE_SCALE


def add_gam_rate_target(df: pd.DataFrame, target_col: str, assessed_col: str) -> pd.DataFrame:
    """Create an observed-only GAM detection-rate target without filling missing GAM."""
    df = df.copy()
    cases = pd.to_numeric(df[target_col], errors="coerce")
    assessed = pd.to_numeric(df[assessed_col], errors="coerce") if assessed_col in df.columns else pd.Series(np.nan, index=df.index)
    duplicate = df.get("target_duplicate_qc", pd.Series(0.0, index=df.index)).astype(bool)
    basic_valid = cases.notna() & cases.ge(0) & assessed.gt(0) & cases.le(assessed) & ~duplicate
    ci_lower, ci_upper = wilson_interval_per_1000(cases, assessed)
    ci_width = ci_upper - ci_lower
    imprecise = basic_valid & ci_width.gt(GAM_DETECTION_CI_WIDTH_MAX)
    valid = basic_valid & ~imprecise

    reason = pd.Series("", index=df.index, dtype="object")
    reason.loc[cases.isna()] = "missing_gam"
    reason.loc[cases.notna() & assessed.isna()] = "missing_children_assessed"
    reason.loc[cases.notna() & assessed.notna() & assessed.le(0)] = "invalid_children_assessed"
    reason.loc[cases.notna() & assessed.gt(0) & cases.gt(assessed)] = "gam_exceeds_children_assessed"
    reason.loc[duplicate] = "duplicate_district_month"
    reason.loc[imprecise] = "imprecise_detection_rate"

    rate = (cases / assessed * GAM_RATE_SCALE).where(valid)
    df["GAM_Cases_Observed"] = cases
    df["children_assessed_observed"] = assessed
    df["gam_detection_ci_lower"] = ci_lower
    df["gam_detection_ci_upper"] = ci_upper
    df["gam_detection_ci_width"] = ci_width
    df["target_precision_qc"] = imprecise.astype(float)
    df["Acut_Malnutrition_Observed"] = rate
    df["target_valid"] = valid.astype(float)
    df["target_exclusion_reason"] = reason
    df["target_source"] = np.where(valid, "observed", "excluded")
    df["target_proxy_confidence"] = np.where(valid, "Observed", "No Data")
    df["target_is_observed"] = valid.astype(float)
    df["target_training_weight"] = valid.astype(float)
    df[target_col] = rate
    return df


def covariate_lag_feature_names(available_columns: list[str] | set[str]) -> list[str]:
    return []


def impute_modeled_covariates(df: pd.DataFrame, group_col: str, date_col: str) -> pd.DataFrame:
    df = df.copy()
    present_covariates = [col for col in MODELED_COVARIATES if col in df.columns]

    for col in present_covariates:
        values = sanitize_covariate_values(col, df[col])
        df[col] = values
        df[f"{col}{MISSING_FLAG_SUFFIX}"] = values.isna().astype(float)
        df[col], df[f"{col}{OUTLIER_FLAG_SUFFIX}"] = cap_outliers_causally(df, group_col, date_col, col)

    for col in [covariate for covariate in CLIMATE_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby(group_col)[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], group_col, date_col))
        df[col] = df[col].fillna(0.0)

    for col in [covariate for covariate in CHILD_HEALTH_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby(group_col)[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], group_col, date_col))
        df[col] = df[col].fillna(0.0).clip(lower=0)

    for col in [covariate for covariate in RATE_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby(group_col)[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], group_col, date_col))
        df[col] = df[col].fillna(0.0).clip(lower=0, upper=1)

    for col in [covariate for covariate in SLOW_MOVING_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby(group_col)[col].transform(lambda series: series.ffill())
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], group_col, date_col))
        df[col] = df[col].fillna(0.0).clip(lower=0)

    return df


def load_data_from_df(
    df: pd.DataFrame,
    risk_thresholds: tuple[int, int] = RISK_THRESHOLD_PROFILES["Standard: P90 / P95"],
) -> pd.DataFrame:
    required = ["time_period", "Acut_Malnutrition", "Region_District"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = harmonize_diarrhea_covariate(df.copy())
    df = harmonize_optional_covariates(df)
    if "screened_u5" not in df.columns:
        raise ValueError("Missing required column: screened_u5 (children assessed).")

    df["Date"] = df["time_period"].apply(parse_date)
    df["Acut_Malnutrition"] = sanitize_target_values(df["Acut_Malnutrition"])
    df = df.dropna(subset=["Date"]).copy()

    raw_rd = df["Region_District"].astype(str).str.strip()
    df["Region_District"] = raw_rd

    if "District" in df.columns:
        df["District_Name"] = df["District"].astype(str).str.strip()
    else:
        df["District_Name"] = raw_rd.str.split("|").str[-1].str.strip()

    has_pipe = raw_rd.str.contains("|", regex=False)
    df["Region_Label"] = ""
    df.loc[has_pipe, "Region_Label"] = raw_rd[has_pipe].str.split("|").str[0].str.strip()
    if "Region" in df.columns:
        df.loc[~has_pipe, "Region_Label"] = df.loc[~has_pipe, "Region"].astype(str).str.strip()
    else:
        df.loc[~has_pipe, "Region_Label"] = "Unknown"

    df["Display_Name"] = df["Region_Label"] + " - " + df["District_Name"]
    df = df.sort_values(["District_Name", "Date"]).reset_index(drop=True)
    df = add_target_qc_flags(df, group_col="District_Name", target_col="Acut_Malnutrition")
    df = add_gam_rate_target(df, target_col="Acut_Malnutrition", assessed_col="screened_u5")
    df = impute_modeled_covariates(df, group_col="District_Name", date_col="Date")
    p80_threshold, p95_threshold = risk_thresholds

    severity_setup = get_severity_setup(df)
    df["Severity_Basis"] = severity_setup["basis"]
    df["Severity_Prevalence_Pct"] = np.nan
    df["Severity_Phase"] = "No Data"
    df["Denominator_Source"] = get_denominator_source_label(severity_setup)
    if severity_setup["observed_method"] in {"gam_whz", "gam_muac"}:
        prevalence = pd.to_numeric(df[severity_setup["prevalence_col"]], errors="coerce")
        df["Severity_Prevalence_Pct"] = prevalence
        df["Severity_Phase"] = prevalence.apply(classify_ipc_amn_phase)

    df[["wd_p80", "wd_p95"]] = np.nan
    df["wd_risk"] = "No Data"

    for district in df["District_Name"].unique():
        mask = df["District_Name"] == district
        d = df[mask].sort_values("Date")
        for idx in d.index:
            value = df.loc[idx, "Acut_Malnutrition"]
            if pd.isna(value):
                continue
            past = d.loc[d["Date"] < df.loc[idx, "Date"], "Acut_Malnutrition"].dropna()
            if len(past) >= MIN_WITHIN_HISTORY_MONTHS:
                p80, p95 = np.percentile(past, [p80_threshold, p95_threshold])
            else:
                continue
            df.loc[idx, ["wd_p80", "wd_p95"]] = p80, p95
            df.loc[idx, "wd_risk"] = classify_risk(value, p80, p95)

    df["wd_score"] = df["wd_risk"].map(RISK_ORDER).fillna(0).astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter

    return df.reset_index(drop=True)


# =============================================================================
# FEATURE ENGINEERING  (pruned)
# =============================================================================
#
# Removed lower-importance / redundant engineered features:
# Lag2_Z, Lag3_Z, Roll6_Max, Roll12_Mean, Roll12_Max, Trend, Acceleration,
# Hist_P75, Lag1_Over_P75, Lag1_Over_P90, Lag1_Ratio_P75, Lag1_Ratio_P90.
# Hist_P90 is retained to capture elevated conditions below the Respond threshold.

@st.cache_data(show_spinner=False)
def build_features(_df: pd.DataFrame, horizon: int = 1):
    if horizon not in (1, 3):
        raise ValueError("Direct forecast horizon must be 1 or 3 months.")
    records = []

    for district in _df["District_Name"].unique():
        d = _df[_df["District_Name"] == district].sort_values("Date").reset_index(drop=True)
        adm = d["Acut_Malnutrition"].astype(float).values

        eligible_indices = [
            i for i in range(2, len(d) - horizon)
            if np.isfinite(adm[i - 2:i + 1]).all() and np.isfinite(adm[i + horizon])
        ]
        for i in eligible_indices:
            target_index = i + horizon
            hist = adm[:i + 1][np.isfinite(adm[:i + 1])]
            roll3 = adm[i - 2:i + 1]
            lag1 = float(adm[i])
            lag2 = float(adm[i - 1])
            lag3 = float(adm[i - 2])

            same_month_hist = d.loc[
                (d.index <= i) & (d["Date"].dt.month == d.loc[target_index, "Date"].month),
                "Acut_Malnutrition",
            ].dropna().astype(float)
            seasonal_lag_12 = float(same_month_hist.iloc[-1]) if len(same_month_hist) else lag1

            row = {
                "District": district,
                "Date": d.loc[target_index, "Date"],
                "Forecast_Horizon": horizon,
                "Month_Sin": np.sin(2 * np.pi * d.loc[target_index, "Date"].month / 12),
                "Month_Cos": np.cos(2 * np.pi * d.loc[target_index, "Date"].month / 12),
                "Lag1": lag1,
                "Lag2": lag2,
                "Lag3": lag3,
                "Roll3_Mean": float(np.mean(roll3)),
                "Seasonal_Lag12": seasonal_lag_12,
                "Target_Adm": adm[target_index],
                "Target_Adm_Observed": float(d.loc[target_index, "Acut_Malnutrition_Observed"]) if pd.notna(d.loc[target_index, "Acut_Malnutrition_Observed"]) else np.nan,
                "Target_Source": d.loc[target_index, "target_source"] if "target_source" in d.columns else "observed",
                "Target_Proxy_Confidence": d.loc[target_index, "target_proxy_confidence"] if "target_proxy_confidence" in d.columns else "Observed",
                "Target_Is_Observed": float(d.loc[target_index, "target_is_observed"]) if "target_is_observed" in d.columns else 1.0,
                "Target_Training_Weight": float(d.loc[target_index, "target_training_weight"]) if "target_training_weight" in d.columns else 1.0,
                "Target_WD_Risk": d.loc[target_index, "wd_risk"],
            }

            for covariate in CORE_COVARIATES:
                for feature in (covariate, f"{covariate}{MISSING_FLAG_SUFFIX}", f"{covariate}{OUTLIER_FLAG_SUFFIX}"):
                    if feature in d.columns:
                        row[feature] = float(d.loc[i - 1, feature])

            records.append(row)

    feat_df = pd.DataFrame(records)
    required_cols = [
        "District",
        "Date",
        "Forecast_Horizon",
        "Target_Adm",
        "Target_WD_Risk",
    ]
    feat_df = feat_df.dropna(subset=[col for col in required_cols if col in feat_df.columns]).reset_index(drop=True)
    skewed_feature_cols = [covariate for covariate in CORE_COVARIATES if covariate in SKEWED_COVS]
    feat_df = apply_log_transform(feat_df, skewed_feature_cols)

    # Random forests do not require scaled inputs. Keeping the raw lag features
    # and a log1p target means a row is independent of later district outcomes.
    scalers = {
        district: {"feature_mean": 0.0, "feature_std": 1.0, "target_mean": 0.0, "target_std": 1.0}
        for district in feat_df["District"].unique()
    }
    feat_df["Target_Adm_Transformed"] = transform_regression_target(feat_df["Target_Adm"].astype(float))

    for col in feat_df.columns:
        if col not in {"District", "Date", "Target_WD_Risk"}:
            if pd.api.types.is_integer_dtype(feat_df[col]):
                feat_df[col] = feat_df[col].astype("float64")

    targets = {
        "District",
        "Date",
        "Target_Adm",
        "Target_Adm_Observed",
        "Target_Adm_Transformed",
        "Target_Source",
        "Target_Proxy_Confidence",
        "Target_Is_Observed",
        "Target_Training_Weight",
        "Target_WD_Risk",
    }
    covariate_features = [
        feature
        for covariate in CORE_COVARIATES
        for feature in (covariate, f"{covariate}{MISSING_FLAG_SUFFIX}", f"{covariate}{OUTLIER_FLAG_SUFFIX}")
        if feature in feat_df.columns
    ]
    feature_cols = CORE_GAM_FEATURES + covariate_features
    return feat_df, feature_cols, scalers


# =============================================================================
# WALK-FORWARD CV AND MODELS
# =============================================================================

@st.cache_data(show_spinner=False)
def wf_splits(_feat_df: pd.DataFrame, n_splits: int = 6, min_months: int = 12):
    dates = sorted(_feat_df["Date"].unique())
    n = len(dates)
    if n < min_months + 2:
        return []

    cutoff = dates[0] + pd.DateOffset(months=min_months)
    min_i = next((i for i, d in enumerate(dates) if d >= cutoff), n - 1)
    step = max(1, (n - min_i) // n_splits)

    splits = []
    for i in range(n_splits):
        ci = min_i + i * step
        if ci >= n - 1:
            break
        te = min(ci + step, n - 1)
        tr_i = _feat_df[_feat_df["Date"].isin(dates[:ci])].index
        te_i = _feat_df[_feat_df["Date"].isin(dates[ci:te])].index
        if len(tr_i) >= 10 and len(te_i) >= 5:
            splits.append((tr_i, te_i))
    return splits


def make_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=700,
        max_depth=14,
        min_samples_leaf=1,
        min_samples_split=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


@st.cache_data(show_spinner=False)
def run_regression_cv(_feat_df: pd.DataFrame, feature_cols: list[str], _scalers: dict):
    valid = _feat_df.dropna(subset=["Target_Adm_Transformed"]).copy().reset_index(drop=True)
    X = valid[feature_cols].values
    y = valid["Target_Adm_Transformed"].values
    weights = valid["Target_Training_Weight"].fillna(1.0).values if "Target_Training_Weight" in valid.columns else None
    splits = wf_splits(valid)
    if not splits:
        return {"error": "Not enough data for regression CV."}

    mae_cv, rmse_cv, resid_rows = [], [], []
    for tr_i, te_i in splits:
        tr_p = valid.index.get_indexer(tr_i)
        tr_p = tr_p[tr_p >= 0]
        te_p = valid.index.get_indexer(te_i)
        te_p = te_p[te_p >= 0]
        if not len(tr_p) or not len(te_p):
            continue

        model = make_regressor()
        model.fit(X[tr_p], y[tr_p], sample_weight=weights[tr_p] if weights is not None else None)
        pred_transformed = model.predict(X[te_p])
        meta = valid.iloc[te_p]
        pred_raw = np.array([
            inverse_regression_target(pred_transformed[j], _scalers[meta.iloc[j]["District"]])
            for j in range(len(te_p))
        ])
        observed_mask = meta["Target_Is_Observed"].fillna(0).to_numpy(dtype=float) > 0.5
        if not observed_mask.any():
            continue
        actual_raw = meta.loc[observed_mask, "Target_Adm_Observed"].values
        pred_eval = pred_raw[observed_mask]

        mae_cv.append(mean_absolute_error(actual_raw, pred_eval))
        rmse_cv.append(np.sqrt(mean_squared_error(actual_raw, pred_eval)))

        observed_positions = np.flatnonzero(observed_mask)
        for j in observed_positions:
            residual = meta.iloc[j]["Target_Adm_Observed"] - pred_raw[j]
            resid_rows.append({
                "District": meta.iloc[j]["District"],
                "Date": meta.iloc[j]["Date"],
                "WD_Risk": meta.iloc[j]["Target_WD_Risk"],
                "Target_Source": meta.iloc[j]["Target_Source"],
                "Actual": meta.iloc[j]["Target_Adm_Observed"],
                "Predicted": pred_raw[j],
                "Residual": residual,
                "Signed_Error": pred_raw[j] - meta.iloc[j]["Target_Adm_Observed"],
                "Absolute_Error": abs(residual),
            })

    final_model = make_regressor()
    final_model.fit(X, y, sample_weight=weights)
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_,
        "Display": [display_label(feature) for feature in feature_cols],
    }).sort_values("Importance", ascending=False)
    residuals = pd.DataFrame(resid_rows)

    return {
        "cv_mae": float(np.mean(mae_cv)),
        "cv_mae_std": float(np.std(mae_cv)),
        "cv_rmse": float(np.mean(rmse_cv)),
        "cv_mean_error": float(residuals["Signed_Error"].mean()) if not residuals.empty else np.nan,
        "cv_median_ae": float(residuals["Absolute_Error"].median()) if not residuals.empty else np.nan,
        "cv_p90_ae": float(residuals["Absolute_Error"].quantile(0.90)) if not residuals.empty else np.nan,
        "feat_imp": feat_imp,
        "residuals": residuals,
        "n_folds": len(mae_cv),
        "model": final_model,
        "target_transform": "log1p GAM detection rate per 1,000 assessed children",
    }


# =============================================================================
# FORECASTING
# =============================================================================

class Forecaster:
    # Engineered features computed at inference time. Keep in sync with build_features.
    BASE = CORE_GAM_FEATURES

    def __init__(self, horizon: int):
        self.horizon = horizon

    def fit(
        self,
        feat_df: pd.DataFrame,
        feature_cols: list[str],
        scalers: dict,
        risk_thresholds: tuple[int, int],
    ):
        self.feature_cols = feature_cols
        self.scalers = scalers
        self.risk_thresholds = risk_thresholds
        self.extra = [c for c in feature_cols if c not in self.BASE]

        valid = feat_df.dropna(subset=["Target_Adm_Transformed"]).copy()
        self.extra_vals = {}
        for d in valid["District"].unique():
            last = valid[valid["District"] == d].iloc[-1]
            self.extra_vals[d] = {c: float(last.get(c, 0.0)) for c in self.extra}

        self.reg = make_regressor()
        sample_weight = valid["Target_Training_Weight"].fillna(1.0).values if "Target_Training_Weight" in valid.columns else None
        self.reg.fit(valid[feature_cols].values, valid["Target_Adm_Transformed"].values, sample_weight=sample_weight)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        severity_setup = get_severity_setup(df)
        forecast_origin = df["Date"].max()
        p80_threshold, p95_threshold = self.risk_thresholds

        for district in df["District_Name"].unique():
            d = df[df["District_Name"] == district].dropna(subset=["Acut_Malnutrition"]).sort_values("Date")
            if len(d) < 3:
                continue

            scaler = self.scalers.get(district)
            if scaler is None:
                continue
            hist_adm = list(d["Acut_Malnutrition"].astype(float).values)
            has_sufficient_within_history = len(hist_adm) >= MIN_WITHIN_HISTORY_MONTHS
            extra_histories = {
                feature: list(d[feature].astype(float).values)
                for feature in self.extra
                if feature in d.columns
            }
            last_date = d["Date"].iloc[-1]
            if last_date != forecast_origin:
                continue
            extras = self.extra_vals.get(district, {c: 0.0 for c in self.extra})
            fd = forecast_origin + pd.DateOffset(months=self.horizon)
            hist = np.array(hist_adm, dtype=float)
            roll3 = hist[-3:]
            risk_p80, risk_p95 = np.percentile(hist, self.risk_thresholds)
            lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]

            base_raw = {
                "Month_Sin": np.sin(2 * np.pi * fd.month / 12),
                "Month_Cos": np.cos(2 * np.pi * fd.month / 12),
                "Lag1": lag1,
                "Lag2": lag2,
                "Lag3": lag3,
                "Roll3_Mean": roll3.mean(),
                "Seasonal_Lag12": hist[-12] if len(hist) >= 12 else lag1,
            }
            dynamic_extras = extras.copy()
            for feature, history in extra_histories.items():
                if not history:
                    continue
                value = float(history[-1])
                dynamic_extras[feature] = float(np.log1p(max(value, 0.0))) if feature in SKEWED_COVS else value

            fv = np.array([[{**base_raw, **dynamic_extras}.get(c, 0.0) for c in self.feature_cols]])
            pred_transformed = float(self.reg.predict(fv)[0])
            pred_raw = round(float(inverse_regression_target(pred_transformed, scaler)), 2)

            tree_preds = np.array([t.predict(fv)[0] for t in self.reg.estimators_])
            tree_preds_raw = inverse_regression_target(tree_preds, scaler)
            lo, hi = np.percentile(tree_preds_raw, [10, 90])
            lo = max(0.0, round(float(lo), 2))
            hi = max(lo, round(float(hi), 2))

            wd_risk = classify_risk(pred_raw, risk_p80, risk_p95) if has_sufficient_within_history else "No Data"
            rows.append({
                "District": district,
                "Date": fd,
                "Month_Year": fd.strftime("%B %Y"),
                "Step": self.horizon,
                "Predicted": pred_raw,
                "Lower_80": lo,
                "Upper_80": hi,
                "WD_Risk": wd_risk,
                "Severity_Prevalence_Pct": np.nan,
                "Severity_Phase": "No Data",
                "Lower_Severity_Phase": "No Data",
                "Severity_Basis": severity_setup["forecast_note"],
                "Last_Observed_Date": last_date,
                "Forecast_Gap_Months": 0,
            })

        result = pd.DataFrame(rows)
        # Direct models emit only their requested horizon; no intermediate prediction is reused.
        result = result[result["Date"] > forecast_origin].copy()
        if result.empty:
            return result

        return result


@st.cache_data(show_spinner=False)
def run_forecast(_feat_df, feature_cols, _scalers, _df, data_key, risk_thresholds: tuple[int, int], horizon: int):
    fc = Forecaster(horizon).fit(_feat_df, feature_cols, _scalers, risk_thresholds)
    return fc.predict(_df)


def _ssm_one_step_predictions(values: np.ndarray) -> list[tuple[float, float]]:
    """Causal local-level/trend filter used for the state-space benchmark."""
    if len(values) < 4:
        return []
    level, trend = float(values[0]), 0.0
    predictions = []
    for value in values[1:]:
        predicted = max(0.0, level + trend)
        predictions.append((predicted, float(value)))
        innovation = float(value) - predicted
        level = predicted + 0.35 * innovation
        trend = trend + 0.12 * innovation
    return predictions


@st.cache_data(show_spinner=False)
def run_ssm_forecast(_df: pd.DataFrame, risk_thresholds: tuple[int, int]) -> tuple[pd.DataFrame, float]:
    rows, errors = [], []
    origin = _df["Date"].max()
    for district, district_df in _df.groupby("District_Name"):
        observed = district_df.dropna(subset=["Acut_Malnutrition"]).sort_values("Date")
        history = observed["Acut_Malnutrition"].astype(float).to_numpy()
        if len(history) < 3:
            continue
        errors.extend(abs(prediction - actual) for prediction, actual in _ssm_one_step_predictions(history))
        level, trend = float(history[0]), 0.0
        for value in history[1:]:
            predicted = level + trend
            innovation = float(value) - predicted
            level = predicted + 0.35 * innovation
            trend = trend + 0.12 * innovation
        residual_sd = float(np.std([actual - prediction for prediction, actual in _ssm_one_step_predictions(history)]))
        residual_sd = max(residual_sd, 1.0)
        last_date = observed["Date"].iloc[-1]
        if last_date != origin:
            continue
        p90, p95 = np.percentile(history, risk_thresholds)
        for horizon in (1, 3):
            forecast_date = origin + pd.DateOffset(months=horizon)
            point = max(0.0, level + trend * min(horizon, 3))
            rows.append({
                "District": district,
                "Date": forecast_date,
                "Month_Year": forecast_date.strftime("%B %Y"),
                "Step": horizon,
                "Predicted": round(point, 2),
                "Lower_80": round(max(0.0, point - 1.28 * residual_sd * np.sqrt(horizon)), 2),
                "Upper_80": round(point + 1.28 * residual_sd * np.sqrt(horizon), 2),
                "WD_Risk": classify_risk(point, p90, p95) if len(history) >= MIN_WITHIN_HISTORY_MONTHS else "No Data",
                "Alert_P90": p90,
                "Alert_P95": p95,
                "Model_Source": "State-Space Model",
                "Last_Observed_Date": last_date,
                "Forecast_Gap_Months": 0,
            })
    forecast = pd.DataFrame(rows)
    if not forecast.empty:
        forecast = forecast[forecast["Date"] > origin].copy()
    return forecast, float(np.mean(errors)) if errors else np.nan


def build_ensemble_forecast(rf_df: pd.DataFrame, ssm_df: pd.DataFrame, rf_mae: float, ssm_mae: float) -> tuple[pd.DataFrame, float]:
    if rf_df.empty or not np.isfinite(rf_mae):
        return ssm_df.copy(), 0.0
    if ssm_df.empty or not np.isfinite(ssm_mae):
        return rf_df.copy(), 1.0
    rf_weight = ssm_mae / max(rf_mae + ssm_mae, 1e-6)
    merged = rf_df.merge(ssm_df, on=["District", "Date"], suffixes=("_RF", "_SSM"))
    rows = []
    for _, row in merged.iterrows():
        point = rf_weight * row["Predicted_RF"] + (1 - rf_weight) * row["Predicted_SSM"]
        p90, p95 = row["Alert_P90"], row["Alert_P95"]
        rows.append({
            "District": row["District"], "Date": row["Date"], "Month_Year": row["Month_Year_RF"], "Step": row["Step_RF"],
            "Predicted": round(point, 2),
            "Lower_80": round(rf_weight * row["Lower_80_RF"] + (1 - rf_weight) * row["Lower_80_SSM"], 2),
            "Upper_80": round(rf_weight * row["Upper_80_RF"] + (1 - rf_weight) * row["Upper_80_SSM"], 2),
            "WD_Risk": classify_risk(point, p90, p95) if pd.notna(p90) else "No Data",
            "Alert_P90": p90, "Alert_P95": p95, "Model_Source": f"MAE-Weighted RF + SSM Blend (RF {rf_weight:.0%})",
        })
    return pd.DataFrame(rows), rf_weight


# =============================================================================
# MAPS
# =============================================================================

def load_and_match_geodf(geo_file, df: pd.DataFrame, name_col: str):
    gdf = gpd.read_file(geo_file)
    gdf = gdf.rename(columns={name_col: "District_Name"})
    gdf["District_Name"] = gdf["District_Name"].astype(str).str.strip()

    csv_lower = {n.lower(): n for n in df["District_Name"].unique()}
    remap = {
        gn: csv_lower[gn.lower()]
        for gn in gdf["District_Name"].unique()
        if gn.lower() in csv_lower and gn != csv_lower[gn.lower()]
    }
    if remap:
        gdf["District_Name"] = gdf["District_Name"].replace(remap)

    matched = len(set(gdf["District_Name"]) & set(df["District_Name"]))
    return gdf, matched


def render_map(gdf, data, risk_col, title, hover_cols=None, levels=None):
    merged = gdf.merge(data, on="District_Name", how="left")
    levels = levels or [*ordered_levels_for_col(risk_col), "No Data"]
    if risk_col in merged.columns:
        merged[risk_col] = pd.Categorical(
            merged[risk_col].fillna("No Data"),
            categories=levels,
            ordered=True,
        )
    gj = safe_geojson(merged)
    hover_cols = hover_cols or []
    hover_data = {c: True for c in hover_cols if c in merged.columns}
    hover_data[risk_col] = True

    fig = px.choropleth_mapbox(
        merged,
        geojson=gj,
        locations=merged.index,
        color=risk_col,
        color_discrete_map=RISK_COLORS,
        category_orders={risk_col: levels},
        labels={risk_col: display_label(risk_col)},
        zoom=5.1,
        center={"lat": 1.37, "lon": 32.29},
        mapbox_style="carto-positron",
        hover_name="District_Name",
        hover_data=hover_data,
        title=title,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=360)
    st.plotly_chart(fig, use_container_width=True)


def render_delta_map(gdf, fc_map, risk_col, months):
    if len(months) < 2:
        return

    first, last = months[0], months[-1]
    first_df = fc_map[fc_map["Month_Year"] == first][["District_Name", risk_col]].rename(columns={risk_col: "Risk_First"})
    last_df = fc_map[fc_map["Month_Year"] == last][["District_Name", risk_col]].rename(columns={risk_col: "Risk_Last"})
    delta = first_df.merge(last_df, on="District_Name", how="inner")
    delta["Score_First"] = delta["Risk_First"].map(RISK_ORDER).fillna(0).astype(int)
    delta["Score_Last"] = delta["Risk_Last"].map(RISK_ORDER).fillna(0).astype(int)
    delta["Delta"] = delta["Score_Last"] - delta["Score_First"]
    delta["Delta_Label"] = delta["Delta"].apply(lambda x: f"+{x}" if x > 0 else str(x))
    delta["Direction"] = delta["Delta"].apply(lambda x: "Worsening" if x > 0 else ("Improving" if x < 0 else "No change"))

    merged = gdf.merge(delta, on="District_Name", how="left")
    merged["Delta_Label"] = merged["Delta_Label"].fillna("0")
    color_map = {
        "-4": "#08306b",
        "-3": "#08519c",
        "-2": "#3182bd",
        "-1": "#9ecae1",
        "0": "#e0e0e0",
        "+1": "#fdae6b",
        "+2": "#e6550d",
        "+3": "#a63603",
        "+4": "#7f2704",
    }
    gj = safe_geojson(merged)
    fig = px.choropleth_mapbox(
        merged,
        geojson=gj,
        locations=merged.index,
        color="Delta_Label",
        color_discrete_map=color_map,
        zoom=5.1,
        center={"lat": 1.37, "lon": 32.29},
        mapbox_style="carto-positron",
        hover_name="District_Name",
        hover_data={"Risk_First": True, "Risk_Last": True, "Direction": True, "Delta_Label": True},
        title=f"Risk Change: {first} to {last}",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=380)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# APP
# =============================================================================

def inject_app_styles():
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: #eef4f8;
            padding: 0.35rem;
            border-radius: 0.9rem;
            border: 1px solid #d4e0e8;
            margin-bottom: 0.85rem;
        }

        button[data-baseweb="tab"] {
            min-height: 2.8rem;
            padding: 0.55rem 1rem;
            border-radius: 0.7rem;
            border: 1px solid transparent;
            background: #f8fbfd;
            color: #234;
            font-size: 1rem;
            font-weight: 700;
            transition: all 0.18s ease;
        }

        button[data-baseweb="tab"]:hover {
            background: #e3eef6;
            color: #12344d;
            border-color: #b8cfde;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: #1f5f8b;
            color: #ffffff;
            border-color: #1f5f8b;
            box-shadow: 0 6px 16px rgba(31, 95, 139, 0.18);
        }

        button[data-baseweb="tab"] p {
            font-size: 1rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Acute Malnutrition Forecasting Tool",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_app_styles()
    st.title("Acute Malnutrition Forecasting Tool")
    st.caption("A tool for district anomaly monitoring and forecasting")

    with st.sidebar:
        st.header("Data")
        csv_mode = st.radio("CSV source", ["Use local CSV path", "Upload CSV"], horizontal=False)
        csv_path = st.text_input("CSV path", value=DEFAULT_CSV_PATH, disabled=(csv_mode != "Use local CSV path"))
        csv_file = None
        if csv_mode == "Upload CSV":
            csv_file = st.file_uploader("Upload CSV", type="csv")

        st.header("GeoJSON")
        geo_mode = st.radio("GeoJSON source", ["Use local GeoJSON path", "Upload GeoJSON"], horizontal=False)
        geo_path = st.text_input("GeoJSON path", value=DEFAULT_GEOJSON_PATH, disabled=(geo_mode != "Use local GeoJSON path"))
        geo_file = None
        geo_source = geo_path if geo_mode == "Use local GeoJSON path" else None
        if geo_mode == "Upload GeoJSON":
            geo_file = st.file_uploader("Upload district GeoJSON", type=["json", "geojson"])
            geo_source = geo_file
        geo_name_col = None
        if geo_source:
            try:
                preview = gpd.read_file(geo_source)
                if hasattr(geo_source, "seek"):
                    geo_source.seek(0)
                non_geom = [c for c in preview.columns if c != "geometry"]
                auto_guess = guess_name_col(list(preview.columns))
                geo_name_col = st.selectbox(
                    "District name column in GeoJSON",
                    options=non_geom,
                    index=non_geom.index(auto_guess) if auto_guess in non_geom else 0,
                )
            except Exception as e:
                st.warning(f"Could not preview GeoJSON: {e}")

        st.markdown("---")
        st.markdown("**Anomaly threshold profile**")
        threshold_profile = st.selectbox(
            "Risk sensitivity",
            options=list(RISK_THRESHOLD_PROFILES),
            help="Changes percentile bands for observed and forecast anomaly labels. The GAM detection-rate target is unchanged.",
        )
        risk_thresholds = RISK_THRESHOLD_PROFILES[threshold_profile]
        forecast_model_choice = st.selectbox(
            "Forecast model",
            FORECAST_MODEL_OPTIONS,
            help="The blend weights Random Forest and the causal state-space model by their available historical MAE estimates.",
        )
        p80_threshold, p95_threshold = risk_thresholds
        st.markdown("**Risk rule**")
        st.caption(
            "Within-district compares each district with its own historical GAM detection-rate distribution. "
            f"Operational Alert uses Monitor (<{p80_threshold}th percentile), "
            f"Alert ({p80_threshold}th-<{p95_threshold}th percentile), and "
            f"Respond (>={p95_threshold}th percentile)."
        )

    try:
        if csv_mode == "Upload CSV":
            if csv_file is None:
                st.info("Upload your CSV to start.")
                return
            raw = csv_file.read()
            source_key = file_hash(raw)
            data_key = f"{source_key}|{threshold_profile}|{forecast_model_choice}"
            df = load_data_from_bytes(raw, source_key, risk_thresholds)
        else:
            data_key = f"{csv_path}|{threshold_profile}|{forecast_model_choice}"
            df = load_data_from_path(csv_path, risk_thresholds)
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        return

    with st.spinner("Engineering district-relative features..."):
        feat_df, feature_cols, scalers = build_features(df, horizon=1)
        feat_df_h3, feature_cols_h3, scalers_h3 = build_features(df, horizon=3)

    if (
        st.session_state.get("_data_key") != data_key
        or st.session_state.get("_model_schema_version") != MODEL_SCHEMA_VERSION
    ):
        with st.spinner("Training direct 1- and 3-month Random Forest forecasts..."):
            st.session_state["reg_eval"] = run_regression_cv(feat_df, feature_cols, scalers)
            st.session_state["reg_eval_h3"] = run_regression_cv(feat_df_h3, feature_cols_h3, scalers_h3)
            rf_forecast = pd.concat([
                run_forecast(feat_df, feature_cols, scalers, df, data_key, risk_thresholds, horizon=1),
                run_forecast(feat_df_h3, feature_cols_h3, scalers_h3, df, data_key, risk_thresholds, horizon=3),
            ], ignore_index=True)
            ssm_forecast, ssm_mae = run_ssm_forecast(df, risk_thresholds)
            if forecast_model_choice == "Random Forest":
                st.session_state["fc_df"] = rf_forecast
            elif forecast_model_choice == "State-Space Model":
                st.session_state["fc_df"] = ssm_forecast
            else:
                ensemble_parts, ensemble_weights = [], {}
                for horizon, evaluation in ((1, st.session_state["reg_eval"]), (3, st.session_state["reg_eval_h3"])):
                    blended, weight = build_ensemble_forecast(
                        rf_forecast[rf_forecast["Step"] == horizon],
                        ssm_forecast[ssm_forecast["Step"] == horizon],
                        evaluation.get("cv_mae", np.nan),
                        ssm_mae,
                    )
                    ensemble_parts.append(blended)
                    ensemble_weights[horizon] = weight
                st.session_state["fc_df"] = pd.concat(ensemble_parts, ignore_index=True)
                st.session_state["ensemble_rf_weight"] = ensemble_weights
            st.session_state["ssm_mae"] = ssm_mae
            st.session_state["_data_key"] = data_key
            st.session_state["_model_schema_version"] = MODEL_SCHEMA_VERSION

    reg_eval = st.session_state["reg_eval"]
    fc_df = st.session_state["fc_df"]

    district_scope = (
        df[["District_Name", "Display_Name"]]
        .drop_duplicates()
        .sort_values("Display_Name")
        .reset_index(drop=True)
    )
    all_districts = district_scope["District_Name"].tolist()
    all_display_districts = district_scope["Display_Name"].tolist()
    display_to_district = dict(zip(district_scope["Display_Name"], district_scope["District_Name"]))
    district_to_display = dict(zip(district_scope["District_Name"], district_scope["Display_Name"]))

    selected_display_districts = st.multiselect(
        "District filter (applies across the tool)",
        ["All"] + all_display_districts,
        default=["All"],
        key="global_district_filter",
    )
    use_all_districts = (not selected_display_districts) or ("All" in selected_display_districts)
    active_display_districts = all_display_districts if use_all_districts else selected_display_districts
    active_districts = all_districts if use_all_districts else [display_to_district[name] for name in active_display_districts]
    scope_label = "All Districts" if use_all_districts else active_display_districts[0] if len(active_display_districts) == 1 else f"{len(active_display_districts)} Selected Districts"

    view_df = df[df["District_Name"].isin(active_districts)].copy()
    view_feat_df = feat_df[feat_df["District"].isin(active_districts)].copy()
    view_fc_df = fc_df[fc_df["District"].isin(active_districts)].copy()
    method_df = build_methodology_df(view_df)
    dq_summary, dq_district, dq_monthly = compute_data_quality_tables(view_df)
    proxy_summary, proxy_table = compute_target_proxy_tables(view_df)

    if use_all_districts:
        scoped_reg_eval = reg_eval
    else:
        scoped_reg_eval = run_regression_cv(view_feat_df, feature_cols, scalers)

    st.success(
        f"{len(view_df):,} records | {view_df['District_Name'].nunique()} districts | "
        f"{view_df['Date'].min().strftime('%b %Y')} to {view_df['Date'].max().strftime('%b %Y')}"
    )
    if not use_all_districts:
        st.caption(f"Current district filter: {', '.join(active_display_districts)}")
    st.info("Streamlit displays the direct district operational-alert scale: Monitor, Alert, Respond.")

    tab_explore, tab_drivers, tab_forecast, tab_eval = st.tabs([
        "Exploratory Analysis",
        "Drivers & Context",
        "Forecasts",
        "Model Checks",
    ])

    with tab_explore:
        with st.expander("Run Summary", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(view_df):,}")
            c2.metric("Districts", view_df["District_Name"].nunique())
            c3.metric("Months", view_df["Date"].nunique())
            c4.metric("Feature rows", f"{len(view_feat_df):,}")

        with st.expander("Data Quality & Method", expanded=True):
            mtab, dtab, mtab2, ptab = st.tabs(["Methodology", "District Quality", "Monthly Quality", "Target Proxies"])

            with mtab:
                k1, k2, k3 = st.columns(3)
                k1.metric("Outcome", "GAM detection rate")
                k2.metric("Operational Alert", "3 levels")
                k3.metric("Expected months", dq_summary["expected_months"])
                st.caption("Early-warning labels are anomaly-based and use district-relative percentile thresholds.")

                left, right = st.columns([1.0, 2.0])
                with left:
                    st.markdown("**Anomaly Scale**")
                    st.dataframe(
                        colour_risk_df(build_alert_actions_df(*risk_thresholds).style, ["Operational Alert"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                with right:
                    st.markdown("**How the Tool Works**")
                    st.dataframe(method_df, use_container_width=True, hide_index=True)
                st.caption("Suggested actions guide review and escalation; they are not an IPC classification or an automated response order.")

            with dtab:
                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("Median completeness", f"{dq_summary['median_completeness']:.1f}%")
                q2.metric("Zero GAM months", f"{dq_summary['zero_gam_months']:,}")
                q3.metric("Outlier GAM months", f"{dq_summary['outlier_gam_months']:,}")
                q4.metric("Duplicate district-months", f"{dq_summary['duplicate_rows']:,}")
                q5.metric("Imprecise GAM-rate rows", f"{dq_summary['imprecise_target_rows']:,}")
                st.dataframe(rename_for_display(dq_district), use_container_width=True, hide_index=True, height=360)

            with mtab2:
                q5, q6 = st.columns(2)
                q5.metric("Missing severity rows", f"{dq_summary['missing_severity_rows']:,}")
                q6.metric("Latest month completeness", f"{dq_monthly['Reporting_Completeness_Pct'].iloc[-1]:.1f}%" if not dq_monthly.empty else "0.0%")
                st.dataframe(rename_for_display(dq_monthly), use_container_width=True, hide_index=True, height=320)

            with ptab:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Observed targets", f"{proxy_summary['observed_rows']:,}")
                p2.metric("Proxy targets", f"{proxy_summary['proxy_rows']:,}")
                p3.metric("Combined proxy", f"{proxy_summary['proxy_combined_rows']:,}")
                p4.metric("Screened proxy", f"{proxy_summary['proxy_screened_rows']:,}")
                st.caption("Observed GAM rows remain the gold-standard evaluation set. Proxy-filled rows can contribute to training with reduced weights.")
                st.dataframe(rename_for_display(proxy_table), use_container_width=True, hide_index=True, height=240)

        situation_dates = sorted(view_df["Date"].dropna().unique(), reverse=True)
        with st.expander("Current Situation", expanded=True):
            situation_date = st.selectbox(
                "Current situation month",
                situation_dates,
                format_func=lambda value: pd.Timestamp(value).strftime("%B %Y"),
                key="current_situation_month",
                help="Uses one common month for every district instead of mixing each district's latest available record.",
            )
            situation_df = view_df[view_df["Date"] == situation_date].copy()
            observed_situation = situation_df[
                pd.to_numeric(situation_df["Acut_Malnutrition_Observed"], errors="coerce").notna()
            ].copy()
            observed_map = add_observed_map_status(situation_df, view_df, situation_date)
            st.caption(
                f"{pd.Timestamp(situation_date).strftime('%B %Y')}: "
                f"{len(observed_situation)} districts have valid observed GAM detection rates."
            )
            st.markdown("**Operational Alert (District Anomaly)**")
            wd_table = build_phase_count_table(situation_df["wd_risk"], RISK_LEVELS, "Level")
            render_phase_pie_chart(wd_table, "Level")

        with st.expander("Districts by Current Operational Alert", expanded=True):
            anomaly_latest = situation_df[["District_Name", "wd_risk"]].copy()
            anomaly_latest["_risk_rank"] = anomaly_latest["wd_risk"].map(RISK_ORDER).fillna(-1)
            anomaly_latest = (
                anomaly_latest
                .sort_values(["_risk_rank", "District_Name"], ascending=[False, True])
                .drop(columns=["_risk_rank"])
            )
            anomaly_latest = rename_for_display(anomaly_latest)
            st.dataframe(
                colour_risk_df(
                    anomaly_latest.head(15).style,
                    ["Operational Alert (District Anomaly)"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Observed District Operational Alert", expanded=True):
            if not geo_source or geo_name_col is None:
                st.info("Provide your district GeoJSON in the sidebar to show observed maps.")
            else:
                try:
                    if hasattr(geo_source, "seek"):
                        geo_source.seek(0)
                    gdf, matched = load_and_match_geodf(geo_source, view_df, geo_name_col)
                    if hasattr(geo_source, "seek"):
                        geo_source.seek(0)
                    st.caption(f"Matched {matched} GeoJSON districts to CSV districts.")
                    if matched == 0:
                        st.error("No matching district names. Select the correct GeoJSON district name column.")
                    else:
                        latest_period = pd.Timestamp(situation_date).strftime("%B %Y")
                        status_counts = observed_map["Observed_Map_Status"].value_counts()
                        st.caption(
                            f"{status_counts.get('GAM Not Reported', 0)} GAM not reported | "
                            f"{status_counts.get('Unreliable GAM Rate', 0)} unreliable rate | "
                            f"{status_counts.get('Provisional Alert History', 0)} provisional history | "
                            f"{status_counts.get('Insufficient Alert History', 0)} insufficient alert history"
                        )
                        render_map(
                            gdf,
                            observed_map[["District_Name", "Observed_Map_Status", "Acut_Malnutrition_Observed", "target_exclusion_reason", "Prior_Valid_GAM_Months"]],
                            "Observed_Map_Status",
                            f"Operational Alert (District Anomaly) - {latest_period}",
                            ["Acut_Malnutrition_Observed", "target_exclusion_reason", "Prior_Valid_GAM_Months"],
                            levels=OBSERVED_MAP_LEVELS,
                        )
                except Exception as e:
                    st.error(f"Observed map error: {e}")
                    st.exception(e)

        with st.expander("Raw Classified Data", expanded=False):
            show_cols = [
                "Region_District", "District_Name", "Region_Label", "time_period", "Date",
                "Acut_Malnutrition", "Acut_Malnutrition_Observed", "GAM_Cases_Observed", "children_assessed_observed",
                "gam_detection_ci_lower", "gam_detection_ci_upper", "gam_detection_ci_width", "target_precision_qc",
                "target_exclusion_reason", "target_source", "target_proxy_confidence", "wd_risk",
            ] + [c for c in FEATURE_COLS_NEW if c in df.columns]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(rename_for_display(view_df[show_cols]), use_container_width=True, height=420, hide_index=True)
            st.download_button("Download classified CSV", view_df[show_cols].to_csv(index=False), "classified_acute_malnutrition.csv")

    with tab_forecast:
        if view_fc_df.empty:
            with st.expander("3-Month District Forecast", expanded=True):
                st.warning("No forecast produced.")
        else:
            with st.expander("3-Month District Forecast", expanded=True):
                st.success(", ".join(view_fc_df["Month_Year"].unique().tolist()))
                st.caption("Forecast maps and tables use the within-district early-warning anomaly scale.")
                resid_source = scoped_reg_eval.get("residuals", pd.DataFrame()).copy() if "error" not in scoped_reg_eval else pd.DataFrame()

                hindcast_plot = pd.DataFrame()
                hindcast_months = 0
                if use_all_districts or len(active_districts) > 1:
                    if not resid_source.empty:
                        hindcast_plot = (
                            resid_source
                            .groupby("Date")[["Actual", "Predicted"]]
                            .sum()
                            .reset_index()
                            .sort_values("Date")
                        )
                else:
                    selected_forecast_district = active_districts[0]
                    selected_forecast_display = district_to_display.get(selected_forecast_district, selected_forecast_district)
                    if not resid_source.empty:
                        hindcast_plot = (
                            resid_source[resid_source["District"] == selected_forecast_district][["Date", "Actual", "Predicted"]]
                            .sort_values("Date")
                            .reset_index(drop=True)
                        )

                max_hindcast = int(hindcast_plot["Date"].nunique()) if not hindcast_plot.empty else 0
                if max_hindcast > 0:
                    hindcast_months = st.slider(
                        "Hindcast months before forecast",
                        min_value=0,
                        max_value=max_hindcast,
                        value=min(6, max_hindcast),
                        step=1,
                        help="Shows out-of-sample historical predictions before the forward 3-month forecast.",
                    )
                    if hindcast_months > 0:
                        hindcast_plot = hindcast_plot.tail(hindcast_months).copy()
                        st.caption("Hindcast shows model predictions for prior months before the forward forecast.")
                    else:
                        hindcast_plot = hindcast_plot.iloc[0:0].copy()

                if use_all_districts or len(active_districts) > 1:
                    hist_plot = (
                        view_df
                        .groupby("Date")["Acut_Malnutrition"]
                        .sum()
                        .reset_index()
                    )
                    hist_label = "Historical"
                    fc_plot = (
                        view_fc_df
                        .groupby("Date")[["Predicted", "Lower_80", "Upper_80"]]
                        .sum()
                        .reset_index()
                    )
                    chart_title = scope_label
                    yaxis_title = "GAM detection rate per 1,000 assessed children"
                    forecast_marker = None
                else:
                    hist_window = max(18, hindcast_months + 6) if hindcast_months > 0 else 18
                    hist_plot = (
                        view_df[view_df["District_Name"] == selected_forecast_district]
                        .sort_values("Date")
                        .tail(hist_window)[["Date", "Acut_Malnutrition", "wd_risk"]]
                        .reset_index(drop=True)
                    )
                    hist_label = "Historical"
                    fc_plot = (
                        view_fc_df[view_fc_df["District"] == selected_forecast_district]
                        .sort_values("Date")[["Date", "Predicted", "Lower_80", "Upper_80", "WD_Risk"]]
                        .reset_index(drop=True)
                    )
                    chart_title = selected_forecast_display
                    yaxis_title = "Cases"
                    forecast_color_col = "WD_Risk" if "WD_Risk" in fc_plot.columns else None
                    forecast_marker = dict(
                        symbol="diamond",
                        size=11,
                        color=[RISK_COLORS.get(r, "#999") for r in fc_plot[forecast_color_col]] if forecast_color_col else "#999",
                    )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_plot["Date"],
                    y=hist_plot["Acut_Malnutrition"],
                    mode="lines+markers",
                    name=hist_label,
                    marker=dict(color=[RISK_COLORS.get(r, "#999") for r in hist_plot["wd_risk"]]) if "wd_risk" in hist_plot.columns else None,
                ))
                if not hindcast_plot.empty:
                    fig.add_trace(go.Scatter(
                        x=hindcast_plot["Date"],
                        y=hindcast_plot["Predicted"],
                        mode="lines+markers",
                        name="Hindcast",
                        line=dict(color="#6c757d", dash="dot", width=2.5),
                        marker=dict(color="#6c757d", size=8),
                        customdata=hindcast_plot["Actual"],
                        hovertemplate="<b>%{x|%b %Y}</b><br>Hindcast: %{y:,.0f}<br>Actual: %{customdata:,.0f}<extra></extra>",
                    ))
                fig.add_trace(go.Scatter(
                    x=list(fc_plot["Date"]) + list(fc_plot["Date"])[::-1],
                    y=list(fc_plot["Upper_80"]) + list(fc_plot["Lower_80"])[::-1],
                    fill="toself",
                    fillcolor="rgba(52,152,219,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="80% interval",
                ))
                fig.add_trace(go.Scatter(
                    x=fc_plot["Date"],
                    y=fc_plot["Predicted"],
                    mode="lines+markers",
                    name="Forecast",
                    marker=forecast_marker,
                    line=dict(color="#3498db", dash="dash", width=3),
                ))
                fig.update_layout(height=460, title=chart_title, yaxis_title=yaxis_title, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("District Forecast Table", expanded=True):
                display = view_fc_df.copy()
                display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
                st.caption("Operational Alert uses the selected percentile profile and each district's own prior history.")
                risk_cols = [c for c in ["WD_Risk"] if c in display.columns]
                display_cols = [
                    "District", "Date", "Predicted", "Lower_80", "Upper_80",
                    "WD_Risk", "Model_Source",
                ]
                display_cols = [c for c in display_cols if c in display.columns]
                display_view = rename_for_display(display[display_cols])
                display_risk_cols = [display_label(c) for c in risk_cols if c in display_cols]
                st.dataframe(colour_risk_df(display_view.style, display_risk_cols), use_container_width=True, hide_index=True, height=480)
                if use_all_districts:
                    download_name = "acute_malnutrition_forecast.csv"
                elif len(active_districts) == 1:
                    download_name = f"acute_malnutrition_forecast_{active_districts[0]}.csv"
                else:
                    download_name = "acute_malnutrition_forecast_selected_districts.csv"
                st.download_button("Download forecast CSV", display.to_csv(index=False), download_name)

            with st.expander("Forecast District Operational Alert Maps", expanded=True):
                if not use_all_districts:
                    st.caption(f"Showing forecast maps for: {', '.join(active_display_districts)}.")
                if not geo_source or geo_name_col is None:
                    st.info("Provide your district GeoJSON in the sidebar to show forecast maps.")
                else:
                    try:
                        if hasattr(geo_source, "seek"):
                            geo_source.seek(0)
                        gdf, matched = load_and_match_geodf(geo_source, view_df, geo_name_col)
                        if hasattr(geo_source, "seek"):
                            geo_source.seek(0)
                        st.caption(f"Matched {matched} GeoJSON districts to CSV districts.")
                        if matched == 0:
                            st.error("No matching district names. Select the correct GeoJSON district name column.")
                        else:
                            fc_map = view_fc_df.rename(columns={"District": "District_Name"}).copy()
                            if fc_map.empty:
                                st.info("No forecast map data available for the selected district.")
                            else:
                                forecast_districts = fc_map["District_Name"].nunique()
                                st.caption(
                                    f"Forecasts are available for {forecast_districts} of {matched} mapped districts. "
                                    "Grey districts have insufficient valid observed GAM-detection-rate history; provisional-history districts are shown separately."
                                )
                                horizons = fc_map[["Step", "Month_Year"]].drop_duplicates().sort_values("Step")
                                map_cols = [("WD_Risk", "Operational Alert (District Anomaly)")]
                                available_cols = [(risk_col, col_label) for risk_col, col_label in map_cols if risk_col in fc_map.columns]
                                if not available_cols:
                                    st.info("No forecast anomaly column available for mapping.")
                                else:
                                    for _, forecast_period in horizons.iterrows():
                                        horizon = int(forecast_period["Step"])
                                        month = forecast_period["Month_Year"]
                                        st.markdown(f"**Horizon {horizon}: {month}**")
                                        row_cols = st.columns(len(available_cols))
                                        md = fc_map[fc_map["Month_Year"] == month]
                                        for col, (risk_col, col_label) in zip(row_cols, available_cols):
                                            hover_cols = ["Predicted", "Lower_80", "Upper_80"]
                                            with col:
                                                render_map(
                                                    gdf,
                                                    md[[c for c in ["District_Name", risk_col, "Predicted", "Lower_80", "Upper_80", "Severity_Prevalence_Pct"] if c in md.columns]],
                                                    risk_col,
                                                    f"{col_label} - {month}",
                                                    hover_cols,
                                                )
                    except Exception as e:
                        st.error(f"Forecast map error: {e}")
                        st.exception(e)

    with tab_drivers:
        with st.expander("Spearman Correlation Portrait", expanded=True):
            st.caption("Rank-based correlation between GAM detection rate per 1,000 assessed children and covariates. Blue means positive association; red means negative association.")
            corr_slice = view_df
            if len(active_districts) == 1:
                st.caption(f"{len(corr_slice)} monthly records for {active_display_districts[0]}.")

            corr_df = compute_spearman_correlations(corr_slice)
            fig_corr = draw_spearman_radial(corr_df)
            if fig_corr is None:
                st.info("Not enough numeric variation for Spearman correlation in the selected scope.")
            else:
                st.pyplot(fig_corr, use_container_width=True)
                plt.close(fig_corr)
                show_corr = corr_df[["Display", "Spearman rho", "N"]].rename(columns={"Display": "Variable"})
                st.dataframe(show_corr.round(3), use_container_width=True, hide_index=True)

        with st.expander("Compare Districts or Regions Against National Reference", expanded=True):
            render_compare_time_series(view_df, scope_label)

        with st.expander(f"Historical Anomaly Trend - {scope_label}", expanded=True):
            render_historical_risk_trend(view_df, scope_label)

        with st.expander("Historical Trend", expanded=True):
            nat = view_df.groupby("Date")["Acut_Malnutrition"].sum().reset_index()
            fig = px.line(nat, x="Date", y="Acut_Malnutrition", markers=True, title=f"District GAM Detection Rate Sum per 1,000 Assessed Children - {scope_label}")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    with tab_eval:
        with st.expander("Regression: Case Count Forecast", expanded=True):
            if "error" in scoped_reg_eval:
                st.warning(scoped_reg_eval["error"])
            else:
                mean_error = scoped_reg_eval.get("cv_mean_error", np.nan)
                p90_ae = scoped_reg_eval.get("cv_p90_ae", np.nan)
                target_transform = scoped_reg_eval.get("target_transform", "GAM detection rate scaling")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("MAE", f"{scoped_reg_eval['cv_mae']:.1f}", delta=f"+/- {scoped_reg_eval['cv_mae_std']:.1f}", delta_color="off")
                r2.metric("RMSE", f"{scoped_reg_eval['cv_rmse']:.1f}")
                r3.metric("Mean Error", f"{mean_error:.1f}" if pd.notna(mean_error) else "N/A")
                r4.metric("P90 Abs Error", f"{p90_ae:.1f}" if pd.notna(p90_ae) else "N/A")
                r5.metric("CV folds", scoped_reg_eval["n_folds"])
                st.caption(
                    f"Regression target transform: {target_transform}. "
                    "Mean error above 0 means the model tends to overpredict."
                )
                if not use_all_districts:
                    st.caption(f"Model checks are filtered to: {', '.join(active_display_districts)}")

                resid = scoped_reg_eval["residuals"].copy()
                if not use_all_districts and not resid.empty:
                    resid = resid[resid["District"].isin(active_districts)].copy()
                if not resid.empty:
                    tabs = st.tabs(["Predicted vs Actual", "Residuals", "Largest Errors", "Feature Importance"])

                    with tabs[0]:
                        fig = px.scatter(
                            resid,
                            x="Actual",
                            y="Predicted",
                            color="WD_Risk",
                            color_discrete_map=RISK_COLORS,
                            category_orders={"WD_Risk": RISK_LEVELS},
                            hover_data=["District", "Date", "Absolute_Error"],
                            title=f"Predicted vs Actual - {scope_label}",
                        )
                        max_val = max(resid["Actual"].max(), resid["Predicted"].max()) * 1.05
                        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)

                    with tabs[1]:
                        fig = px.scatter(
                            resid,
                            x="Actual",
                            y="Signed_Error",
                            color="WD_Risk",
                            color_discrete_map=RISK_COLORS,
                            category_orders={"WD_Risk": RISK_LEVELS},
                            hover_data=["District", "Date", "Predicted", "Absolute_Error"],
                            title=f"Signed Error vs Actual - {scope_label}",
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(height=450, yaxis_title="Predicted - Actual")
                        st.plotly_chart(fig, use_container_width=True)

                    with tabs[2]:
                        largest_errors = (
                            resid.sort_values("Absolute_Error", ascending=False)
                            .head(15)
                            .copy()
                        )
                        largest_errors["Date"] = pd.to_datetime(largest_errors["Date"]).dt.strftime("%Y-%m")
                        st.dataframe(
                            largest_errors[[
                                "District", "Date", "Actual", "Predicted",
                                "Signed_Error", "Absolute_Error", "WD_Risk",
                            ]].round(2),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with tabs[3]:
                        fig_fi = px.bar(
                            scoped_reg_eval["feat_imp"].head(15),
                            x="Importance",
                            y="Display",
                            orientation="h",
                            title=f"Top Features - Regression ({scope_label})",
                        )
                        fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=430)
                        st.plotly_chart(fig_fi, use_container_width=True)

if __name__ == "__main__":
    main()
