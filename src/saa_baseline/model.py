"""Frozen 21-Oct random forest configuration. No search or test-time retuning."""
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from .data import CHANNELS, CUT_TRAIN, features, sha256

THRESHOLD = 0.75
RF_PARAMETERS = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 30,
                 "class_weight": "balanced", "random_state": 42, "n_jobs": 2}


def train(frame, labels):
    if "UTC" not in frame or len(frame) == 0 or frame.UTC.isna().any() or frame.UTC.duplicated().any() or (frame.UTC >= CUT_TRAIN).any():
        raise ValueError("Training input contains validation/test timestamps or is empty.")
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(frame) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training labels must align and contain both binary classes.")
    estimator = RandomForestClassifier(**RF_PARAMETERS)
    estimator.fit(features(frame), y)
    return estimator


def predict(estimator, frame):
    p = estimator.predict_proba(features(frame))[:, list(estimator.classes_).index(1)]
    return p, p >= THRESHOLD


def save_model(estimator, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)
    record = {**metadata, "model_sha256": sha256(path), "features": list(CHANNELS),
              "feature_transform": "log1p", "threshold": THRESHOLD,
              "parameters": RF_PARAMETERS, "sklearn": sklearn.__version__,
              "warning": "joblib/pickle is executable: load only your own trusted artifacts"}
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(record, indent=2) + "\n")


def load_model(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if metadata["model_sha256"] != sha256(path):
        raise ValueError("Model artifact hash mismatch.")
    if metadata["sklearn"] != sklearn.__version__:
        raise ValueError("scikit-learn version differs; reproduce the pinned environment.")
    if metadata["threshold"] != THRESHOLD or metadata["features"] != list(CHANNELS):
        raise ValueError("Model schema differs from frozen baseline.")
    estimator = joblib.load(path)  # Only caller-owned, trusted artifact paths.
    if any(estimator.get_params()[k] != v for k, v in RF_PARAMETERS.items()):
        raise ValueError("Loaded model parameters differ from the frozen baseline.")
    return estimator, metadata
