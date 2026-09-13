from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, fbeta_score, hamming_loss, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from travel_agent.quality import dataset_hash

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "01_data" / "raw" / "intent_router_dataset.csv"
CHALLENGE_PATH = PROJECT_ROOT / "01_data" / "raw" / "intent_router_challenge.csv"
OUT_DIR = PROJECT_ROOT / "06_results" / "models"
MODEL_PATH = OUT_DIR / "intent_router.joblib"
MODEL_METADATA_PATH = OUT_DIR / "intent_router_metadata.json"
METRICS_PATH = OUT_DIR / "intent_router_metrics.json"


class RouterModelCompatibilityError(RuntimeError):
    """Raised when a persisted sklearn router cannot safely run in this environment."""


def _labels(s: str) -> list[str]:
    return [x for x in str(s).split("|") if x]


def _runtime_metadata() -> dict[str, str]:
    """Runtime versions that materially affect pickle/joblib compatibility.

    scikit-learn explicitly does not support loading persisted estimators across
    different sklearn versions. Python major/minor is also recorded because pickle
    implementation details can change between Python runtimes. We *do not* force a
    package downgrade/upgrade; if these values change, the setup simply retrains the
    small local router once in the user's current environment.
    """
    return {
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": getattr(joblib, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def _read_metrics_file() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_model_metadata() -> dict[str, Any] | None:
    if not MODEL_METADATA_PATH.exists():
        return None
    try:
        data = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _safe_joblib_load(path: Path) -> Any:
    """Load a trusted local artifact while converting version warnings to errors.

    This prevents noisy InconsistentVersionWarning messages followed by obscure
    internal-module errors such as ``ModuleNotFoundError: _loss``. Callers can then
    turn the failure into a clean "retrain required" state.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(path)


def _best_thresholds(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    grid = np.arange(0.20, 0.76, 0.05)
    for idx, label in enumerate(classes):
        best_t, best_f = 0.45, -1.0
        for t in grid:
            pred = (probs[:, idx] >= t).astype(int)
            # Tool routing penalizes unnecessary calls, so threshold selection is
            # precision-oriented (F0.5) rather than pure F1.
            f = fbeta_score(y_true[:, idx], pred, beta=0.5, zero_division=0)
            if f > best_f:
                best_t, best_f = float(round(t, 2)), float(f)
        thresholds[label] = best_t
    return thresholds


def _apply_thresholds(probs: np.ndarray, classes: list[str], thresholds: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(probs, dtype=int)
    for idx, label in enumerate(classes):
        out[:, idx] = (probs[:, idx] >= thresholds.get(label, 0.45)).astype(int)
    return out


def train_router(*, write_outputs: bool = True) -> tuple[dict[str, Any], Any]:
    df = pd.read_csv(DATA_PATH)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train["tool_labels"].map(_labels))
    y_val = mlb.transform(val["tool_labels"].map(_labels))
    y_test = mlb.transform(test["tool_labels"].map(_labels))
    classes = [str(x) for x in mlb.classes_]

    # A word-level 1–3 gram TF-IDF representation scales efficiently to the
    # 240k-query corpus while retaining phrase/context information. Unicode accent
    # normalization helps accented/unaccented Hungarian surface forms; deterministic
    # routing guardrails handle high-value suffix/negation edge cases.
    features = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 3), min_df=3, max_features=35000,
        sublinear_tf=True, strip_accents="unicode"
    )
    model = Pipeline([
        ("features", features),
        ("clf", OneVsRestClassifier(SGDClassifier(loss="log_loss", max_iter=35, tol=1e-3, alpha=2e-5, class_weight="balanced", random_state=42), n_jobs=1)),
    ])
    model.fit(train["query"], y_train)

    val_prob = model.predict_proba(val["query"])
    thresholds = _best_thresholds(y_val, val_prob, classes)
    val_pred = _apply_thresholds(val_prob, classes, thresholds)
    test_prob = model.predict_proba(test["query"])
    test_pred = _apply_thresholds(test_prob, classes, thresholds)

    runtime = _runtime_metadata()
    metrics: dict[str, Any] = {
        "train_rows": int(len(train)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
        "labels": classes, "thresholds": thresholds,
        "validation_micro_f1": float(f1_score(y_val, val_pred, average="micro", zero_division=0)),
        "validation_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        "test_micro_f1": float(f1_score(y_test, test_pred, average="micro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_hamming_loss": float(hamming_loss(y_test, test_pred)),
        "dataset_sha256": dataset_hash(DATA_PATH),
        "feature_model": "word_tfidf_1_3_unicode + OVR_SGD_logistic",
        "training_runtime": runtime,
    }

    p, r, f, support = precision_recall_fscore_support(y_test, test_pred, average=None, zero_division=0)
    metrics["per_label"] = {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
        for i, label in enumerate(classes)
    }

    if CHALLENGE_PATH.exists():
        challenge = pd.read_csv(CHALLENGE_PATH)
        y_challenge = mlb.transform(challenge["tool_labels"].map(_labels))
        challenge_prob = model.predict_proba(challenge["query"])
        challenge_pred = _apply_thresholds(challenge_prob, classes, thresholds)
        metrics.update({
            "challenge_rows": int(len(challenge)),
            "challenge_micro_f1": float(f1_score(y_challenge, challenge_pred, average="micro", zero_division=0)),
            "challenge_macro_f1": float(f1_score(y_challenge, challenge_pred, average="macro", zero_division=0)),
            "challenge_hamming_loss": float(hamming_loss(y_challenge, challenge_pred)),
        })

    model_metadata = {
        "artifact_format": 2,
        "dataset_sha256": metrics["dataset_sha256"],
        **runtime,
    }
    artifact = {
        "pipeline": model,
        "mlb": mlb,
        "thresholds": thresholds,
        "threshold": 0.45,
        "metrics": metrics,
        "dataset_sha256": metrics["dataset_sha256"],
        "artifact_metadata": model_metadata,
    }

    if write_outputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        # Write the small JSON sidecar *after* the model dump. model_status reads this
        # before unpickling, so incompatible sklearn versions never touch the pickle.
        MODEL_METADATA_PATH.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report = classification_report(y_test, test_pred, target_names=classes, zero_division=0)
        (OUT_DIR / "intent_router_classification_report.txt").write_text(report, encoding="utf-8")
        _write_chart(metrics)
    return metrics, artifact


def _write_chart(metrics: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    labels = ["Validation micro-F1", "Held-out test micro-F1", "Indirect challenge micro-F1"]
    values = [metrics["validation_micro_f1"], metrics["test_micro_f1"], metrics.get("challenge_micro_f1", 0.0)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.set_title("Intent router generalization")
    ax.tick_params(axis="x", rotation=10)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.015, f"{value:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(OUT_DIR / "router_generalization.png", dpi=160); plt.close(fig)


def model_status() -> dict[str, Any]:
    """Return router status without blindly unpickling an incompatible sklearn model."""
    current_dataset = dataset_hash(DATA_PATH)
    metrics = _read_metrics_file()
    runtime = _runtime_metadata()

    if not MODEL_PATH.exists():
        return {
            "exists": False,
            "stale": True,
            "compatible": False,
            "reason": "model_missing",
            "dataset_sha256": current_dataset,
            "metrics": metrics,
        }

    metadata = _read_model_metadata()
    if metadata is None:
        # v1.2.0 and earlier shipped a bare joblib file. Do not attempt to load it:
        # it may have been produced by another sklearn version and can fail deep
        # inside pickle with private-module imports such as `_loss`.
        return {
            "exists": True,
            "stale": True,
            "compatible": False,
            "reason": "model_metadata_missing",
            "dataset_sha256": current_dataset,
            "metrics": metrics,
            "message": "Legacy model artifact detected. Retrain once in the current environment.",
        }

    trained_dataset = metadata.get("dataset_sha256")
    runtime_mismatches: dict[str, dict[str, str | None]] = {}
    for key in ("scikit_learn_version", "python_major_minor"):
        trained_value = metadata.get(key)
        current_value = runtime.get(key)
        if trained_value != current_value:
            runtime_mismatches[key] = {"trained": trained_value, "current": current_value}

    if runtime_mismatches:
        return {
            "exists": True,
            "stale": True,
            "compatible": False,
            "reason": "runtime_version_mismatch",
            "runtime_mismatches": runtime_mismatches,
            "dataset_sha256": current_dataset,
            "model_dataset_sha256": trained_dataset,
            "metrics": metrics,
            "message": "Saved sklearn model was trained in a different runtime. Retraining is required once; packages do not need to be downgraded.",
        }

    if trained_dataset != current_dataset:
        return {
            "exists": True,
            "stale": True,
            "compatible": True,
            "reason": "dataset_changed",
            "dataset_sha256": current_dataset,
            "model_dataset_sha256": trained_dataset,
            "metrics": metrics,
        }

    try:
        artifact = _safe_joblib_load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - convert persistence failures into state
        return {
            "exists": True,
            "stale": True,
            "compatible": False,
            "reason": "model_load_failed",
            "dataset_sha256": current_dataset,
            "model_dataset_sha256": trained_dataset,
            "metrics": metrics,
            "load_error": f"{type(exc).__name__}: {exc}",
            "message": "Saved router cannot be loaded safely. Retraining is required once.",
        }

    artifact_metrics = artifact.get("metrics", {}) if isinstance(artifact, dict) else {}
    return {
        "exists": True,
        "stale": False,
        "compatible": True,
        "reason": "current",
        "dataset_sha256": current_dataset,
        "model_dataset_sha256": trained_dataset,
        "metrics": artifact_metrics or metrics,
        "runtime": runtime,
    }


def load_router_artifact() -> dict[str, Any]:
    """Load the current router or raise a concise, actionable compatibility error."""
    status = model_status()
    if not status.get("exists"):
        raise FileNotFoundError(
            f"Intent router model not found: {MODEL_PATH}. Run SETUP_AND_START_UI.bat or 04_scripts/06_train_intent_router.py first."
        )
    if status.get("stale") or not status.get("compatible", False):
        reason = status.get("reason", "unknown")
        detail = status.get("message", "Retraining is required.")
        raise RouterModelCompatibilityError(f"Intent router is not usable ({reason}). {detail}")
    try:
        artifact = _safe_joblib_load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise RouterModelCompatibilityError(
            f"Intent router could not be loaded safely ({type(exc).__name__}: {exc}). Retrain it in the current environment."
        ) from exc
    if not isinstance(artifact, dict):
        raise RouterModelCompatibilityError("Intent router artifact has an unexpected format. Retraining is required.")
    return artifact
