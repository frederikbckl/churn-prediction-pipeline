#!/usr/bin/env python3
"""
Customer churn prediction pipeline.

For TUM course "Business Analytics and Machine Learning" (midterm assignment)
Merges several relational CSVs on a shared customer ID. 
Preprocessing, classification pipeline, decision threshold on balanced accuracy
"""

import pandas as pd
import numpy as np
from functools import reduce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ID_COL = "customer_id"
TARGET_COL = "churn"

DATA_PATHS = [
    "data/churn_analysis.csv",
    "data/customers.csv",
    "data/payment_info.csv",
    "data/service_options.csv",
]

MODEL = "hgb"  # "logreg", "rf", or "hgb"
OUT_PATH = "submission.csv"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]


def read_and_merge(paths: list[str]) -> pd.DataFrame:   # Left-join every CSV in `paths` onto the first one via ID_COL
    dfs = [pd.read_csv(p) for p in paths]

    def merge_two(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        if ID_COL not in b.columns:
            raise ValueError(f"Missing {ID_COL} in one file")
        return a.merge(b, on=ID_COL, how="left")

    return reduce(merge_two, dfs)


def drop_high_cardinality_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_uniques: int = 200,
    max_unique_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]: # Drop non-numeric columns that are effectively free text or hidden IDs
    X_train = X_train.copy()
    X_test = X_test.copy()
    n = len(X_train)

    drop = []
    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            nun = X_train[c].nunique(dropna=True)
            if nun >= max_uniques or (nun / max(n, 1)) >= max_unique_ratio:
                drop.append(c)

    if drop:
        X_train = X_train.drop(columns=drop)
        X_test = X_test.drop(columns=drop)

    return X_train, X_test, drop


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: # merge source files and split into labeled and unlabeled rows
    df = read_and_merge(DATA_PATHS)
    if ID_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError("Merged data must contain the ID and target columns")

    train_df = df[df[TARGET_COL].notna()].copy()
    test_df = df[df[TARGET_COL].isna()].copy()

    for name, frame in [("train", train_df), ("test", test_df)]:
        if frame[ID_COL].isna().any():
            raise ValueError(f"Missing IDs in {name}")
        if frame[ID_COL].nunique(dropna=False) != len(frame):
            raise ValueError(f"Duplicate IDs in {name}")

    y = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
    if y.isna().any():
        raise ValueError("Label column must be clean 0/1")
    y = y.astype(int)

    X_train = train_df.drop(columns=[ID_COL, TARGET_COL])
    X_test = test_df.drop(
        columns=[ID_COL] + ([TARGET_COL] if TARGET_COL in test_df.columns else [])
    )

    X_train, X_test, dropped = drop_high_cardinality_categoricals(X_train, X_test)
    if dropped:
        print("[INFO] Dropped high-cardinality categoricals:", dropped)

    X_test = X_test.reindex(columns=X_train.columns)
    test_ids = test_df[ID_COL].copy()

    return X_train, y, X_test, test_ids


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute + scale numeric columns; most-frequent-impute + one-hot categoricals."""
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def build_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
        )
    raise ValueError(f"Unknown MODEL: {name}")


def best_threshold_by_balanced_accuracy(
    model, X_val: np.ndarray, y_val: pd.Series
) -> tuple[float, float]:
    """Sweep THRESHOLDS on validation probabilities and return the best (threshold, score)."""
    val_probs = model.predict_proba(X_val)[:, 1]
    scores = [
        balanced_accuracy_score(y_val, (val_probs >= t).astype(int)) for t in THRESHOLDS
    ]
    best_idx = int(np.argmax(scores))
    return THRESHOLDS[best_idx], scores[best_idx]


def main():
    X_train, y, X_test, test_ids = load_data()

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    print("Number of features after preprocessing:", len(feature_names))
    print("First 50 feature names:")
    for name in feature_names[:50]:
        print(name)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_processed, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(MODEL)
    model.fit(X_tr, y_tr)

    # threshold selection on validation set (balanced accuracy)
    if hasattr(model, "predict_proba"):
        best_threshold, bacc = best_threshold_by_balanced_accuracy(model, X_val, y_val)
        print(f"[INFO] Best threshold on validation: {best_threshold} (BA={bacc:.4f})")
    else:
        best_threshold = None
        val_pred = model.predict(X_val)
        bacc = balanced_accuracy_score(y_val, val_pred)
        print(f"[INFO] Validation BA (no predict_proba): {bacc:.4f}")

    # retrain on all labeled data before predicting the held-out rows
    model.fit(X_train_processed, y)
    if best_threshold is not None:
        test_probs = model.predict_proba(X_test_processed)[:, 1]
        test_pred = (test_probs >= best_threshold).astype(int)
    else:
        test_pred = model.predict(X_test_processed).astype(int)

    out = pd.DataFrame({"id": test_ids.values, "prediction": test_pred})
    out.to_csv(OUT_PATH, index=False)

    print(f"train={len(y)}  test={len(test_ids)}  pos_rate={y.mean():.3f}")
    print(f"val_bal_acc={bacc:.4f}  wrote={OUT_PATH}")
    print(out.head())


if __name__ == "__main__":
    main()
