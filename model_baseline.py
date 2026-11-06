"""Fit LASSO on the training CSV and evaluate once on the held-out test CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_pair(training_path, test_path):
    train = pd.read_csv(training_path)
    test = pd.read_csv(test_path)
    if "y" not in train or "y" not in test:
        raise ValueError("Both files must contain the y target column.")
    if list(train.columns) != list(test.columns):
        raise ValueError("Training and test columns must match in the same order.")
    train = train.apply(pd.to_numeric, errors="raise")
    test = test.apply(pd.to_numeric, errors="raise")
    if train.empty or test.empty or train.isna().any().any() or test.isna().any().any():
        raise ValueError("Training and test data must be nonempty and contain no missing values.")
    if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError("Training and test data must be finite.")
    return train, test


def evaluate(train, test, seed=42):
    if train.shape[1] < 2 or train.shape[0] < 5 or test.empty:
        raise ValueError("Require predictors, at least five training rows, and test rows.")
    X_train, y_train = train.drop(columns="y"), train["y"]
    X_test, y_test = test.drop(columns="y"), test["y"]
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test predictors must match.")
    folds = min(5, len(train))
    model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=folds, random_state=seed, max_iter=10000),
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    regressor = model.named_steps["lassocv"]
    coefficients = regressor.coef_
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "predictors": X_train.shape[1],
        "alpha": float(regressor.alpha_),
        "selected_predictors": int(np.count_nonzero(coefficients)),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": float(r2_score(y_test, predicted)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", nargs="?", type=Path, default=Path("training_data.csv"))
    parser.add_argument("test", nargs="?", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = evaluate(*load_pair(args.training, args.test), seed=args.seed)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
