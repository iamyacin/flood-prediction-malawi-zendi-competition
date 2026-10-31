"""Train a flood-extent model for the Zindi Malawi challenge.

The competition data is not included. Download ``Train.csv`` and
``SampleSubmission.csv`` from Zindi (if you are an authorised participant),
place them in ``data/``, then run:

    python train.py --model rf --cv 5

The script prints cross-validation RMSE and writes a submission CSV.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RAINFALL_PATTERN = re.compile(
    r"^precip\s+(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
DEFAULT_STATIC = "elevation,LC_Type1_mode"
DEFAULT_CATEGORICAL = "LC_Type1_mode"


def parse_events(columns) -> dict[int, list[str]]:
    """Group weekly precipitation columns into flood events by date gaps."""
    found = []
    for column in columns:
        match = RAINFALL_PATTERN.match(str(column).strip())
        if not match:
            continue
        start = date.fromisoformat(match.group(1))
        end = date.fromisoformat(match.group(2))
        found.append((column, start, end))

    if not found:
        raise ValueError("No precipitation columns found. Expected names like "
                         "'precip 2019-01-20 - 2019-01-27'.")

    groups: list[list[tuple[str, date, date]]] = [[found[0]]]
    for item in found[1:]:
        if item[1] - groups[-1][-1][2] > timedelta(days=60):
            groups.append([item])
        else:
            groups[-1].append(item)

    events: dict[int, list[str]] = {}
    for group in groups:
        year = max(item[2] for item in group).year
        if year in events:
            raise ValueError(
                f"More than one rainfall block ends in {year}; "
                "rename the columns or edit parse_events()."
            )
        events[year] = [item[0] for item in group]
    return events


def rainfall_features(frame: pd.DataFrame, columns: list[str], raw: bool) -> pd.DataFrame:
    """Turn one event's weekly rainfall into model features."""
    values = frame[columns].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    if raw:
        names = [f"rain_w{index + 1:02d}" for index in range(values.shape[1])]
        return pd.DataFrame(values, columns=names, index=frame.index)

    return pd.DataFrame(
        {
            "rain_total": values.sum(axis=1),
            "rain_mean": values.mean(axis=1),
            "rain_std": values.std(axis=1),
            "rain_max": values.max(axis=1),
            "rain_last4w": values[:, -min(4, values.shape[1]) :].sum(axis=1),
            "rain_last8w": values[:, -min(8, values.shape[1]) :].sum(axis=1),
            "rain_first4w": values[:, : min(4, values.shape[1])].sum(axis=1),
            "rain_wet_weeks": (values >= 0.1).sum(axis=1).astype(float),
        },
        index=frame.index,
    )


def build_features(
    frame: pd.DataFrame,
    rainfall_columns: list[str],
    static_columns: list[str],
    coordinates: bool,
    raw_rainfall: bool,
) -> pd.DataFrame:
    """Build a design matrix with identical column names for both events."""
    features = rainfall_features(frame, rainfall_columns, raw_rainfall)
    for column in static_columns:
        features[column] = frame[column].to_numpy()
    if coordinates:
        features["X"] = frame["X"].to_numpy()
        features["Y"] = frame["Y"].to_numpy()
    return features


def make_model(
    name: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    n_estimators: int,
    seed: int,
) -> Pipeline:
    """Build a preprocessing pipeline and regressor."""
    categorical = [column for column in categorical_columns if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]

    scale = name == "ridge"
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )

    if name == "rf":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "hist_gb":
        estimator = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif name == "tree":
        estimator = DecisionTreeRegressor(random_state=seed)
    else:
        estimator = Ridge(alpha=1.0)

    return Pipeline([("preprocessor", ColumnTransformer(transformers)), ("model", estimator)])


def cross_validate(model: Pipeline, features: pd.DataFrame, target: np.ndarray, cv: int, seed: int) -> float | None:
    """Print and return out-of-fold RMSE for a random K-fold split."""
    if cv == 0:
        return None
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    predictions = np.full(len(target), np.nan)
    for train_index, test_index in splitter.split(features):
        fold_model = clone(model)
        fold_model.fit(features.iloc[train_index], target[train_index])
        predictions[test_index] = fold_model.predict(features.iloc[test_index])
    predictions = np.clip(predictions, 0.0, 1.0)
    rmse = float(np.sqrt(mean_squared_error(target, predictions)))
    return rmse


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a flood-extent model and write a Zindi submission."
    )
    parser.add_argument("--train", default="data/Train.csv", help="Training CSV path.")
    parser.add_argument(
        "--sample", default="data/SampleSubmission.csv", help="Sample submission CSV path."
    )
    parser.add_argument(
        "--output", default="submissions/submission.csv", help="Output submission path."
    )
    parser.add_argument("--model", choices=("rf", "extra_trees", "hist_gb", "tree", "ridge"), default="rf")
    parser.add_argument("--cv", type=int, default=5, help="Random K-fold count; 0 skips CV.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Trees or boosting rounds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-col", default="Square_ID")
    parser.add_argument("--target-col", default="target_2015")
    parser.add_argument("--target-year", type=int, default=2015)
    parser.add_argument("--predict-year", type=int, default=None)
    parser.add_argument("--static-cols", default=DEFAULT_STATIC)
    parser.add_argument("--categorical-cols", default=DEFAULT_CATEGORICAL)
    parser.add_argument("--coordinates", action="store_true", help="Use X/Y as features.")
    parser.add_argument(
        "--raw-rainfall",
        action="store_true",
        help="Use the 17 weekly values instead of aggregated rainfall features.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cv != 0 and args.cv < 2:
        raise ValueError("--cv must be 0 (skip) or at least 2.")

    train_path = Path(args.train)
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Download Train.csv from Zindi "
            "and place it in data/."
        )
    frame = pd.read_csv(train_path)

    static_columns = parse_list(args.static_cols)
    categorical_columns = parse_list(args.categorical_cols)
    unknown_categorical = [column for column in categorical_columns if column not in static_columns]
    if unknown_categorical:
        raise ValueError(
            "Categorical columns must also be listed in --static-cols: "
            + ", ".join(unknown_categorical)
        )
    required = [args.id_col, args.target_col, *static_columns]
    if args.coordinates:
        required.extend(["X", "Y"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing column(s): {', '.join(missing)}")

    events = parse_events(frame.columns)
    predict_year = args.predict_year
    if predict_year is None:
        later = sorted(year for year in events if year > args.target_year)
        if not later:
            raise ValueError("Could not infer the prediction year; use --predict-year.")
        predict_year = later[-1]
    if args.target_year not in events or predict_year not in events:
        raise ValueError(
            f"Rainfall blocks found for {sorted(events)}, but target/predict years are "
            f"{args.target_year}/{predict_year}."
        )

    train_features = build_features(
        frame,
        events[args.target_year],
        static_columns,
        args.coordinates,
        args.raw_rainfall,
    )
    predict_features = build_features(
        frame,
        events[predict_year],
        static_columns,
        args.coordinates,
        args.raw_rainfall,
    )
    if list(train_features.columns) != list(predict_features.columns):
        raise ValueError("Training and prediction rainfall blocks have different widths.")

    target = np.clip(frame[args.target_col].to_numpy(dtype=float), 0.0, 1.0)
    model = make_model(
        args.model,
        list(train_features.columns),
        categorical_columns,
        args.n_estimators,
        args.seed,
    )

    print(f"Training rows: {len(frame):,} | features: {train_features.shape[1]}")
    print(f"Events: {args.target_year} (train) -> {predict_year} (predict)")
    rmse = cross_validate(model, train_features, target, args.cv, args.seed)
    if rmse is not None:
        print(f"{args.cv}-fold CV RMSE: {rmse:.5f}")

    model.fit(train_features, target)
    predictions = np.clip(model.predict(predict_features), 0.0, 1.0)
    submission = pd.DataFrame(
        {
            args.id_col: frame[args.id_col].to_numpy(),
            f"target_{predict_year}": predictions,
        }
    )

    sample_path = Path(args.sample)
    if sample_path.is_file():
        sample = pd.read_csv(sample_path)
        expected = set(sample[args.id_col])
        if expected == set(submission[args.id_col]):
            submission = submission.set_index(args.id_col).loc[sample[args.id_col]].reset_index()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(
        f"Wrote {len(submission):,} predictions to {output_path} "
        f"(mean {predictions.mean():.3f}, min {predictions.min():.3f}, "
        f"max {predictions.max():.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
