import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MODEL_VERSION = "chap-rf-v1"
DEFAULT_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "mean_relative_humidity",
    "average_gpp",
    "malaria_confirmed",
    "pneumonia_cases",
    "pregnant_women_with_Anaemia",
    "diarrhea_acute",
    "low_birth_weight_babies",
    "diarrhea_persistent",
    "population",
]
SKEWED_COVARIATES = {
    "malaria_confirmed",
    "pneumonia_cases",
    "pregnant_women_with_Anaemia",
    "diarrhea_acute",
    "low_birth_weight_babies",
    "diarrhea_persistent",
    "population",
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


def parse_time_period(value: str) -> pd.Timestamp:
    value = str(value).strip()
    for fmt in ("%Y-%m", "%B %Y", "%b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(value, format=fmt))
        except (TypeError, ValueError):
            pass
    return pd.to_datetime(value, errors="coerce")


def log_transform_covariate(name: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if name in SKEWED_COVARIATES:
        clipped = numeric.clip(lower=0)
        return np.log1p(clipped)
    return numeric


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
        norm["disease_cases"] = pd.to_numeric(df.loc[norm.index, target_col], errors="coerce")
    elif require_target:
        raise ValueError("Observed inputs must include disease_cases.")
    else:
        norm["disease_cases"] = np.nan

    covariates = []
    for covariate in DEFAULT_COVARIATES:
        source = pick_column(df, [covariate])
        if source is None:
            continue
        norm[covariate] = log_transform_covariate(covariate, df.loc[norm.index, source])
        covariates.append(covariate)

    norm = norm.sort_values(["location", "Date"]).reset_index(drop=True)
    if require_target:
        norm = norm.dropna(subset=["disease_cases"]).copy()
    return norm


def infer_covariate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_COVARIATES if column in df.columns]


def transform_target(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def inverse_target(values: float | np.ndarray, scaler: dict) -> float | np.ndarray:
    return np.expm1(np.asarray(values) * scaler["target_std"] + scaler["target_mean"])


def build_training_features(df: pd.DataFrame, covariate_cols: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    rows = []
    scalers = {}

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
                "Target_Transformed": float((target_log[i - 3] - target_mean) / target_std),
            }
            for covariate in covariate_cols:
                row[covariate] = float(location_df.loc[i, covariate]) if pd.notna(location_df.loc[i, covariate]) else 0.0
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

    feature_cols = BASE_FEATURES + covariate_cols
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
    model.fit(feature_df[feature_cols].to_numpy(), feature_df["Target_Transformed"].to_numpy())

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
) -> pd.DataFrame:
    model = model_artifact["model"]
    covariate_cols = model_artifact["covariate_cols"]
    feature_cols = model_artifact["feature_cols"]
    scalers = model_artifact["scalers"]
    rows = []

    locations = sorted(set(future_df["location"]))
    for location in locations:
        location_hist = historic_df[historic_df["location"] == location].sort_values("Date").reset_index(drop=True)
        location_future = future_df[future_df["location"] == location].sort_values("Date").reset_index(drop=True)
        if len(location_hist) < 3:
            continue
        scaler = scalers.get(location)
        if scaler is None:
            continue

        hist_cases = location_hist["disease_cases"].astype(float).tolist()
        known_covariates = last_known_covariates(location_hist, covariate_cols)

        for i in range(len(location_future)):
            future_row = location_future.iloc[i]
            current_date = future_row["Date"]
            hist = np.array(hist_cases, dtype=float)
            roll3 = hist[-3:]
            roll6 = hist[-6:] if len(hist) >= 6 else hist
            hist_mean = float(np.mean(hist))
            hist_std = float(np.std(hist)) if np.std(hist) > 1e-6 else 1.0
            hist_p90, hist_p95 = np.percentile(hist, [90, 95])

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
                value = future_row.get(covariate, np.nan)
                if pd.isna(value):
                    value = known_covariates.get(covariate, 0.0)
                else:
                    value = float(value)
                    known_covariates[covariate] = value
                feature_row[covariate] = value

            scaled_row = scale_feature_row(feature_row, scaler)
            feature_vector = np.array([[scaled_row[column] for column in feature_cols]], dtype=float)
            samples = sample_from_forest(model, feature_vector, scaler)
            point_forecast = float(np.mean(samples))
            hist_cases.append(point_forecast)

            row = {
                "time_period": str(future_row["time_period"]),
                "location": location,
            }
            row.update({f"sample_{index}": float(samples[index]) for index in range(len(samples))})
            rows.append(row)

    if not rows:
        raise ValueError("No predictions were generated. Check that future locations exist in the historic/training data and each has at least 3 observations.")
    return pd.DataFrame(rows)


def predict_model(model_path: str, historic_data_path: str, future_data_path: str, output_path: str) -> None:
    with open(model_path, "rb") as file_obj:
        artifact = pickle.load(file_obj)

    historic_df = normalize_dataframe(pd.read_csv(historic_data_path), require_target=True)
    future_df = normalize_dataframe(pd.read_csv(future_data_path), require_target=False)

    for covariate in artifact["covariate_cols"]:
        if covariate not in future_df.columns:
            future_df[covariate] = np.nan

    predictions = build_prediction_rows(historic_df, future_df, artifact)
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
        predict_model(args.model, args.historic_data, args.future_data, args.out_file)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
