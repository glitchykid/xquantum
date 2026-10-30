"""Research metrics with one equally weighted prediction per held-out patient."""

import numpy as np


def _metrics(labels, scores, classes):
    truth = np.array([classes.index(label) for label in labels])
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    per_class, recalls, f1s = {}, [], []
    for i, label in enumerate(classes):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        fp = int(matrix[:, i].sum()) - tp
        fn = support - tp
        tn = len(labels) - tp - fp - fn
        recall = tp / support if support else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = {
            "support": support,
            "sensitivity": recall,
            "precision": precision,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "f1": f1,
        }
        recalls.append(recall)
        f1s.append(f1)
    return {
        "samples": len(labels),
        "class_support": {k: v["support"] for k, v in per_class.items()},
        "balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None,
        "macro_f1": float(np.mean(f1s)),
        "log_loss": float(
            -np.log(np.clip(scores[np.arange(len(truth)), truth], np.finfo(float).eps, 1)).mean()
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def evaluate(labels, probabilities, patient_ids, classes, *, seed=42) -> dict:
    """Average scores within patient; stratified patient bootstrap, 1,000 resamples.

    A patient must have one adjudicated primary label for this single-label task.
    No patient IDs, individual predictions or clinical records enter the report.
    """
    labels, patient_ids, classes = list(labels), list(patient_ids), list(classes)
    scores = np.asarray(probabilities, dtype=float)
    if (
        not labels
        or len(classes) < 2
        or len(set(classes)) != len(classes)
        or len(patient_ids) != len(labels)
        or scores.shape != (len(labels), len(classes))
    ):
        raise ValueError("Inconsistent evaluation dimensions or classes")
    if (
        not all(isinstance(c, str) and c for c in classes)
        or not set(labels).issubset(classes)
        or not all(isinstance(p, str) and p.strip() for p in patient_ids)
    ):
        raise ValueError("Known labels and nonempty patient IDs are required")
    if (
        not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or not np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("Probabilities must be finite, bounded, and sum to one")
    groups = {}
    for i, patient in enumerate(patient_ids):
        groups.setdefault(patient, []).append(i)
    patient_labels, patient_scores = [], []
    for indices in groups.values():
        diagnoses = {labels[i] for i in indices}
        if len(diagnoses) != 1:
            raise ValueError("Conflicting primary labels for one patient; adjudication required")
        patient_labels.append(labels[indices[0]])
        patient_scores.append(scores[indices].mean(axis=0))
    patient_scores = np.asarray(patient_scores)
    result = _metrics(labels, scores, classes)
    primary = _metrics(patient_labels, patient_scores, classes)
    result.update(
        patients=len(groups),
        class_order=classes,
        primary_analysis="patient_level",
        patient_level=primary,
        warnings=[],
    )
    primary["confidence_intervals"] = None
    support = primary["class_support"]
    if any(support[c] == 0 for c in classes):
        result["warnings"].append("Missing held-out classes; balanced accuracy is undefined")
    if any(support[c] < 2 for c in classes):
        result["warnings"].append(
            "Intervals withheld: require at least two independent patients per class; "
            "this minimum is not a clinical sample-size justification"
        )
        return result
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(np.array(patient_labels) == c) for c in classes]
    names = ("balanced_accuracy", "macro_f1", "log_loss")
    draws = []
    for _ in range(1000):
        indices = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        metrics = _metrics([patient_labels[i] for i in indices], patient_scores[indices], classes)
        draws.append([metrics[name] for name in names])
    bounds = np.percentile(np.asarray(draws), [2.5, 97.5], axis=0, method="linear")
    primary["confidence_intervals"] = {
        "method": "stratified_patient_bootstrap_percentile",
        "level": 0.95,
        "resamples": 1000,
        "seed": seed,
        **{
            name: {"lower": float(bounds[0, i]), "upper": float(bounds[1, i])}
            for i, name in enumerate(names)
        },
    }
    result["warnings"].append(
        "Intervals condition on observed class counts and fixed model weights; "
        "they do not establish external validity or account for training uncertainty"
    )
    return result
