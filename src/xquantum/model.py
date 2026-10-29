"""Train a CPU baseline; export numeric weights without executable serialization."""

import hashlib
import importlib.metadata
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss

from xquantum.dataset import audit_manifest, sample_path
from xquantum.imaging import FEATURE_VERSION, decode_image, image_features, quality_issues
from xquantum.schemas import ClinicalRecord


def feature_vector(image, clinical: ClinicalRecord, multimodal: bool) -> np.ndarray:
    visual = image_features(image)
    if not multimodal:
        return visual
    clinical_values = np.asarray(clinical.numeric_values())
    return np.concatenate([visual, clinical_values, np.isnan(clinical_values).astype(float)])


def normalize(features, median, mean, scale):
    return (np.where(np.isnan(features), median, features) - mean) / scale


def train(
    manifest: Path,
    output: Path,
    *,
    bootstrap: bool = False,
    multimodal: bool = False,
    epochs: int = 50,
    seed: int = 42,
) -> dict:
    if epochs < 1 or epochs > 10000:
        raise ValueError("epochs must be between 1 and 10000")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; preserve previous run artifacts")
    rows = audit_manifest(manifest, bootstrap=bootstrap)
    if bootstrap and multimodal:
        raise ValueError("Bootstrap sources have no verified paired clinical data")
    splits = np.array([row.split for row in rows])
    training = splits == "train"
    if not training.any():
        raise ValueError("No training samples")
    if not bootstrap and set(splits) != {"train", "validation", "test"}:
        raise ValueError("Reviewed research requires explicit train, validation and test splits")
    labels = np.array([row.label for row in rows])
    classes = np.unique(labels[training])
    if len(classes) < 2:
        raise ValueError("At least two training classes are required")
    if set(labels) != set(classes):
        raise ValueError("Held-out label has no training examples")
    if multimodal:
        for label in classes:
            paired = [
                row
                for row in rows
                if row.split == "train"
                and row.label == label
                and np.isfinite(row.clinical.numeric_values()).any()
            ]
            if len(paired) < 2:
                raise ValueError(
                    "Multimodal training requires two paired clinical records per class"
                )
    features = np.stack(
        [
            feature_vector(
                decode_image(sample_path(manifest, row).read_bytes()), row.clinical, multimodal
            )
            for row in rows
        ]
    )
    train_features = features[training]
    median = np.array(
        [
            np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0
            for col in train_features.T
        ]
    )
    filled = np.where(np.isnan(train_features), median, train_features)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale < 1e-8] = 1
    x = normalize(features, median, mean, scale)
    y = labels[training]
    counts = Counter(y)
    sample_weights = np.array([len(y) / (len(classes) * counts[label]) for label in y])
    classifier = SGDClassifier(
        loss="log_loss", alpha=0.01, learning_rate="constant", eta0=0.01, random_state=seed
    )
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        classifier.partial_fit(
            x[training][order], y[order], classes=classes, sample_weight=sample_weights[order]
        )
        losses.append(
            {
                "epoch": epoch + 1,
                "training_log_loss": float(
                    log_loss(y, classifier.predict_proba(x[training]), labels=classes)
                ),
            }
        )
    metrics = None
    if not bootstrap:
        metrics = {}
        for split in ("validation", "test"):
            mask = splits == split
            probabilities = classifier.predict_proba(x[mask])
            predictions = classes[np.argmax(probabilities, axis=1)]
            metrics[split] = {
                "samples": int(mask.sum()),
                "class_support": dict(Counter(labels[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(labels[mask], predictions)),
                "macro_f1": float(
                    f1_score(
                        labels[mask], predictions, labels=classes, average="macro", zero_division=0
                    )
                ),
                "log_loss": float(log_loss(labels[mask], probabilities, labels=classes)),
                "confusion_matrix": confusion_matrix(
                    labels[mask], predictions, labels=classes
                ).tolist(),
            }
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "weights.npz"
    np.savez_compressed(
        weights_path,
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        median=median,
        mean=mean,
        scale=scale,
        classes=classes,
    )
    report = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "epochs": epochs,
        "mode": "bootstrap" if bootstrap else "reviewed_research",
        "multimodal": multimodal,
        "architecture": "color-gradient-grid + standardized SGD logistic OvR baseline",
        "clinically_validated": False,
        "training_samples": int(training.sum()),
        "training_class_counts": dict(counts),
        "classes": classes.tolist(),
        "training_history": losses,
        "held_out_metrics": metrics,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "Pillow", "pydantic")
        },
        "limitations": [
            "Uncalibrated research scores are not diagnostic probabilities",
            "No validated scalp detector or out-of-distribution detector",
            "No treatment recommendations; specialist review always required",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class ResearchModel:
    def __init__(self, arrays, report):
        self.arrays = arrays
        self.report = report

    @classmethod
    def load(cls, directory: Path):
        report = json.loads((directory / "report.json").read_text())
        if report.get("schema_version") != 1 or report.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported model schema or feature version")
        raw = (directory / "weights.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != report["weights_sha256"]:
            raise ValueError("Model checksum mismatch")
        with np.load(directory / "weights.npz", allow_pickle=False) as file:
            arrays = {name: file[name].copy() for name in file.files}
        return cls(arrays, report)

    def analyze(self, raw: bytes, clinical: ClinicalRecord | None = None) -> dict:
        image = decode_image(raw)
        clinical = clinical or ClinicalRecord()
        issues = quality_issues(image)
        scores = []
        a = self.arrays
        if not issues:
            features = feature_vector(image, clinical, self.report["multimodal"])
            z = (
                normalize(features, a["median"], a["mean"], a["scale"]) @ a["coef"].T
                + a["intercept"]
            )
            probabilities = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            if len(a["classes"]) == 2:
                probabilities = np.array([1 - probabilities[0], probabilities[0]])
            else:
                probabilities /= probabilities.sum()
            scores = sorted(
                [
                    {"label": str(label), "score": float(score)}
                    for label, score in zip(a["classes"], probabilities, strict=True)
                ],
                key=lambda row: row["score"],
                reverse=True,
            )
        observed = [
            name
            for name in ("pain", "pustules", "loss_of_follicular_openings")
            if getattr(clinical, name) is True
        ]
        return {
            "status": "abstain",
            "clinically_validated": False,
            "reason": "Image quality failed"
            if issues
            else "Research model requires specialist review",
            "model_id": self.report["weights_sha256"],
            "mode": self.report["mode"],
            "quality_issues": issues,
            "research_scores": scores,
            "clinical_context_used_by_model": bool(self.report["multimodal"] and not issues),
            "observations_for_specialist_review": observed,
            "limitations": self.report["limitations"],
        }
