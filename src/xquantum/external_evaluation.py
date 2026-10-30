"""Evaluate fixed research weights on an audited, independent reviewed cohort."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from xquantum.dataset import audit_manifest, sample_path
from xquantum.evaluation import evaluate
from xquantum.imaging import decode_image, perceptual_hash
from xquantum.model import ResearchModel


def _fingerprints(manifest, row):
    raw = sample_path(manifest, row).read_bytes()
    image = decode_image(raw)
    pixel_hash = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
    return row.sha256, pixel_hash, perceptual_hash(image)


def _check_independence(manifest, cohort, development_manifest, development):
    patients = {row.patient_id for row in development if row.patient_id is not None}
    sources = {row.source_group for row in development}
    reference = [_fingerprints(development_manifest, row) for row in development]
    content = {item[0] for item in reference}
    pixels = {item[1] for item in reference}
    for row in cohort:
        if row.patient_id in patients or row.source_group in sources:
            raise ValueError("Patient or source-group overlap with development cohort")
        sha, pixel_hash, fingerprint = _fingerprints(manifest, row)
        if sha in content or pixel_hash in pixels:
            raise ValueError("Exact duplicate of a development image")
        if any((fingerprint ^ other[2]).bit_count() <= 4 for other in reference):
            raise ValueError("Possible near-duplicate development overlap; review required")


def evaluate_saved_model(
    model_dir: Path, manifest: Path, development_manifest: Path, output: Path, *, seed: int = 42
) -> dict:
    """Require the original development manifest bound by the training report.

    Check against all original train/validation/test records, not only fitted rows.
    The output contains aggregate metrics; no patient IDs or individual predictions.
    """
    if output.exists():
        raise ValueError("Evaluation output already exists; use a new report path")
    model = ResearchModel.load(model_dir)
    development_hash = hashlib.sha256(development_manifest.read_bytes()).hexdigest()
    if development_hash != model.report["manifest_sha256"]:
        raise ValueError("Development manifest does not match the model training report")
    development = audit_manifest(
        development_manifest, bootstrap=model.report["mode"] == "bootstrap"
    )
    cohort = audit_manifest(manifest, bootstrap=False)
    if {row.split for row in cohort} != {"test"}:
        raise ValueError("Independent evaluation requires a test-only reviewed manifest")
    classes = model.arrays["classes"].tolist()
    if not {row.label for row in cohort}.issubset(classes):
        raise ValueError("Evaluation label is outside the saved model class scope")
    _check_independence(manifest, cohort, development_manifest, development)
    probabilities = []
    for row in cohort:
        result = model.analyze(sample_path(manifest, row).read_bytes(), row.clinical)
        if result["quality_issues"]:
            raise ValueError("Evaluation image failed quality checks; no metrics were written")
        scores = {item["label"]: item["score"] for item in result["research_scores"]}
        probabilities.append([scores[label] for label in classes])
    metrics = evaluate(
        [row.label for row in cohort],
        probabilities,
        [row.patient_id for row in cohort],
        classes,
        seed=seed,
    )
    warnings = [
        "Metrics describe uncalibrated research scores; diagnostic outputs still abstain",
        "Group and image checks cannot establish independence from unknown or re-uploaded patients",
        "Freeze the cohort before analysis; repeated test-set selection invalidates its role",
    ]
    if any(row.patient_id is None for row in development):
        warnings.append(
            "Development patient identities are incomplete; patient disjointness is unverified"
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "independent_saved_model",
        "clinically_validated": False,
        "score_metrics_only": True,
        "model_id": model.report["weights_sha256"],
        "development_manifest_sha256": development_hash,
        "evaluation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "clinical_context_used_by_model": model.report["multimodal"],
        "restricted_evaluation_source_count": sum(row.use_scope != "open" for row in cohort),
        "model_release_review_required": model.report.get("model_release_review_required", False)
        or any(row.use_scope != "open" for row in cohort),
        "metrics": metrics,
        "warnings": warnings,
        "quality_policy": "fail_entire_evaluation_on_any_quality_issue",
        "overlap_checks": [
            "patient_id",
            "source_group",
            "file_sha256",
            "decoded_pixel_sha256",
            "perceptual_hash_distance_le_4",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
