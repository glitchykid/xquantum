import base64
import hashlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from xquantum.api import create_app
from xquantum.dataset import audit_manifest
from xquantum.imaging import decode_image, image_features
from xquantum.model import ResearchModel, train
from xquantum.schemas import ClinicalRecord, Sample


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i in range(12):
        pixels = np.random.default_rng(i).integers(0, 256, (160, 192, 3), dtype=np.uint8)
        pixels[:, :, i % 2] //= 3
        path = tmp_path / f"image_{i}.png"
        Image.fromarray(pixels).save(path)
        rows.append(
            dict(
                image_path=path.name,
                label=["alopecia_areata", "tinea_capitis"][i % 2],
                patient_id=f"p{i}",
                source_group=f"s{i}",
                split="train",
                source_url="https://example.org/image",
                license="CC BY 4.0",
                rights_basis="Test fixture",
                label_status="specialist_reviewed",
                reviewer_id="test-specialist",
                photo_type="scalp",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                clinical={"age_years": 30 + i, "duration_months": i + 1},
            )
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def rewrite(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_clinical_missing_values_are_unknown_and_units_are_explicit():
    assert ClinicalRecord().age_years is None
    r = ClinicalRecord(labs={"ferritin_ng_ml": 14.0}, itching=False)
    assert r.labs.ferritin_ng_ml == 14
    assert r.itching is False
    for value in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValidationError):
            ClinicalRecord(labs={"ferritin_ng_ml": value})
    with pytest.raises(ValidationError):
        ClinicalRecord(labs={"ferritin": 14})
    with pytest.raises(ValidationError):
        ClinicalRecord(age_years=200)


def test_images_reject_invalid_oversized_and_tiny_input():
    for raw in [b"", b"not an image", b"x" * (10 * 1024 * 1024 + 1)]:
        with pytest.raises(ValueError):
            decode_image(raw)
    b = io.BytesIO()
    Image.new("RGB", (8, 8)).save(b, "PNG")
    with pytest.raises(ValueError):
        decode_image(b.getvalue())


def test_image_features_are_finite_and_shape_stable(dataset):
    path, rows = dataset
    im = decode_image((path.parent / rows[0]["image_path"]).read_bytes())
    assert image_features(im).shape == (432,)
    assert np.isfinite(image_features(im)).all()


def test_manifest_detects_patient_source_and_duplicate_leakage(dataset):
    path, rows = dataset
    assert len(audit_manifest(path, bootstrap=False)) == 12
    for field in ["patient_id", "source_group"]:
        edited = json.loads(json.dumps(rows))
        edited[1]["split"] = "test"
        edited[1][field] = edited[0][field]
        rewrite(path, edited)
        with pytest.raises(ValueError, match="leakage"):
            audit_manifest(path, bootstrap=False)
    edited = json.loads(json.dumps(rows))
    edited[1]["image_path"] = edited[0]["image_path"]
    edited[1]["sha256"] = edited[0]["sha256"]
    rewrite(path, edited)
    with pytest.raises(ValueError, match="duplicate"):
        audit_manifest(path, bootstrap=False)


def test_unreviewed_sources_cannot_be_used_for_reviewed_training(dataset):
    path, rows = dataset
    rows[0]["label_status"] = "source_only"
    rewrite(path, rows)
    with pytest.raises(ValueError, match="review"):
        audit_manifest(path, bootstrap=False)
    assert len(audit_manifest(path, bootstrap=True)) == 12


def test_training_inference_round_trip_and_bootstrap_abstention(dataset, tmp_path):
    path, rows = dataset
    out = tmp_path / "run"
    report = train(path, out, bootstrap=True, epochs=5)
    assert report["training_samples"] == 12
    assert report["held_out_metrics"] is None
    model = ResearchModel.load(out)
    raw = (path.parent / rows[0]["image_path"]).read_bytes()
    result = model.analyze(raw, ClinicalRecord(age_years=40))
    assert result["status"] == "abstain"
    assert result["clinically_validated"] is False
    assert result["clinical_context_used_by_model"] is False
    assert len(result["research_scores"]) == 2
    assert sum(x["score"] for x in result["research_scores"]) == pytest.approx(1)
    assert result == ResearchModel.load(out).analyze(raw, ClinicalRecord(age_years=40))


def test_reviewed_training_evaluates_only_held_out_patients(dataset, tmp_path):
    path, rows = dataset
    for i, row in enumerate(rows):
        row["split"] = "train" if i < 8 else ("validation" if i < 10 else "test")
    rewrite(path, rows)
    report = train(path, tmp_path / "run", bootstrap=False, multimodal=True, epochs=5)
    assert report["training_samples"] == 8
    assert report["held_out_metrics"]["test"]["samples"] == 2
    assert report["held_out_metrics"]["test"]["patients"] == 2
    assert report["held_out_metrics"]["test"]["primary_analysis"] == "patient_level"
    result = ResearchModel.load(tmp_path / "run").analyze(
        (path.parent / rows[0]["image_path"]).read_bytes(), ClinicalRecord(age_years=35)
    )
    assert result["clinical_context_used_by_model"] is True
    assert result["status"] == "abstain"


def test_api_handles_valid_and_invalid_requests_without_persisting_patient_data(dataset, tmp_path):
    path, rows = dataset
    out = tmp_path / "run"
    train(path, out, bootstrap=True, epochs=2)
    with TestClient(create_app(out)) as client:
        assert client.get("/health").json()["model_loaded"] is True
        raw = (path.parent / rows[0]["image_path"]).read_bytes()
        payload = {"image_base64": base64.b64encode(raw).decode(), "clinical": {"age_years": 35}}
        response = client.post("/v1/analyze", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "abstain"
        assert response.headers["cache-control"] == "no-store"
        for bad in ["garbage!", base64.b64encode(b"not image").decode()]:
            assert client.post("/v1/analyze", json={"image_base64": bad}).status_code == 422
        assert (
            client.post("/v1/analyze", json={**payload, "clinical": {"age_years": -1}}).status_code
            == 422
        )
    assert not list(out.glob("*.png"))


def test_sample_paths_cannot_escape_manifest_directory(dataset):
    path, rows = dataset
    rows[0]["image_path"] = "../outside.png"
    rewrite(path, rows)
    with pytest.raises(ValueError, match="outside"):
        audit_manifest(path, bootstrap=True)
    with pytest.raises(ValidationError):
        Sample.model_validate({**rows[1], "label": "made_up"})


@pytest.mark.parametrize(
    "change,match",
    [
        ({"photo_type": "unknown"}, "photo_type"),
        ({"sha256": "0" * 64}, "checksum"),
    ],
)
def test_manifest_rejects_invalid_reviewed_records(dataset, change, match):
    path, rows = dataset
    rows[0].update(change)
    rewrite(path, rows)
    with pytest.raises(ValueError, match=match):
        audit_manifest(path, bootstrap=False)


def test_empty_manifest_and_bootstrap_holdout_are_rejected(dataset):
    path, rows = dataset
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        audit_manifest(path, bootstrap=True)
    rows[0]["split"] = "test"
    rewrite(path, rows)
    with pytest.raises(ValueError, match="Bootstrap"):
        audit_manifest(path, bootstrap=True)


def test_model_rejects_corruption_and_incompatible_features(dataset, tmp_path):
    path, rows = dataset
    out = tmp_path / "run"
    train(path, out, bootstrap=True, epochs=2)
    (out / "weights.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        ResearchModel.load(out)
    report = json.loads((out / "report.json").read_text())
    report["feature_version"] = "unknown"
    (out / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Unsupported"):
        ResearchModel.load(out)


def test_low_quality_image_suppresses_scores(dataset, tmp_path):
    path, _ = dataset
    out = tmp_path / "run"
    train(path, out, bootstrap=True, epochs=2)
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(b, "PNG")
    result = ResearchModel.load(out).analyze(b.getvalue())
    assert result["research_scores"] == []
    assert set(result["quality_issues"]) == {"low_resolution", "low_contrast", "extreme_exposure"}


@pytest.mark.parametrize(
    "options,match",
    [
        ({"bootstrap": True, "epochs": 0}, "epochs"),
        ({"bootstrap": True, "multimodal": True}, "paired"),
        ({"bootstrap": False}, "explicit"),
    ],
)
def test_invalid_training_configuration_is_rejected(dataset, tmp_path, options, match):
    path, _ = dataset
    with pytest.raises(ValueError, match=match):
        train(path, tmp_path / "run", **options)


def test_run_artifacts_are_not_overwritten(dataset, tmp_path):
    path, _ = dataset
    out = tmp_path / "run"
    train(path, out, bootstrap=True, epochs=2)
    with pytest.raises(ValueError, match="empty"):
        train(path, out, bootstrap=True, epochs=2)


def test_three_class_numpy_inference_and_deterministic_training(dataset, tmp_path):
    path, rows = dataset
    for i, row in enumerate(rows):
        row["label"] = ["alopecia_areata", "tinea_capitis", "unaffected"][i % 3]
    rewrite(path, rows)
    a, b = tmp_path / "a", tmp_path / "b"
    train(path, a, bootstrap=True, epochs=3)
    train(path, b, bootstrap=True, epochs=3)
    ma, mb = ResearchModel.load(a), ResearchModel.load(b)
    np.testing.assert_array_equal(ma.arrays["coef"], mb.arrays["coef"])
    raw = (path.parent / rows[0]["image_path"]).read_bytes()
    result = ma.analyze(raw)
    assert len(result["research_scores"]) == 3
    assert sum(r["score"] for r in result["research_scores"]) == pytest.approx(1)


def test_api_rejects_oversized_streamed_body(dataset, tmp_path):
    from xquantum.api import MAX_REQUEST_BYTES

    path, _ = dataset
    out = tmp_path / "run"
    train(path, out, bootstrap=True, epochs=2)
    with TestClient(create_app(out)) as client:
        response = client.post("/v1/analyze", content=b"x" * (MAX_REQUEST_BYTES + 1))
        assert response.status_code == 413


def test_cli_runs_training_auditing_schema_and_analysis(dataset, tmp_path, capsys, monkeypatch):
    import uvicorn

    from xquantum import cli

    path, rows = dataset
    out = tmp_path / "run"
    assert cli.main(["audit", "--manifest", str(path), "--bootstrap"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 12
    assert (
        cli.main(
            ["train", "--manifest", str(path), "--bootstrap", "--output", str(out), "--epochs", "2"]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["schema", "--kind", "sample"]) == 0
    assert "properties" in json.loads(capsys.readouterr().out)
    clinical = tmp_path / "clinical.json"
    clinical.write_text('{"age_years": 30}')
    assert (
        cli.main(
            [
                "analyze",
                "--model",
                str(out),
                "--image",
                str(path.parent / rows[0]["image_path"]),
                "--clinical",
                str(clinical),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "abstain"
    monkeypatch.setattr(cli, "fetch_bootstrap", lambda _: {"downloaded": 10})
    assert cli.main(["fetch-bootstrap", "--output", str(tmp_path)]) == 0
    capsys.readouterr()
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    assert cli.main(["serve", "--model", str(out)]) == 0
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["access_log"] is False
    assert cli.main(["audit", "--manifest", str(tmp_path / "absent")]) == 1
    assert "XQuantum:" in capsys.readouterr().err


def test_downloader_preserves_attribution_and_excludes_unlicensed_files(tmp_path, monkeypatch):
    from xquantum import ingest

    catalog = {
        "alopecia_areata": ["aa1.jpg", "aa2.jpg"],
        "tinea_capitis": ["t1.jpg", "t2.jpg", "bad.jpg"],
    }
    monkeypatch.setattr(ingest, "CATALOG", catalog)

    def fake_fetch(url):
        if "api.php" in url:
            label = "aa" if "aa1" in url else "t"
            pages = {}
            for i in range(3 if label == "t" else 2):
                title = f"{label}{i}.jpg"
                pages[str(i)] = {
                    "title": title,
                    "imageinfo": [
                        {
                            "url": f"https://upload.wikimedia.org/{title}",
                            "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{title}",
                            "extmetadata": {
                                "LicenseShortName": {
                                    "value": "All rights reserved" if i == 2 else "CC BY 4.0"
                                },
                                "Artist": {"value": "<b>Research Author</b>"},
                            },
                        }
                    ],
                }
            return json.dumps({"query": {"pages": pages}}).encode()
        pixels = np.random.default_rng(sum(url.encode())).integers(
            0, 256, (160, 160, 3), dtype=np.uint8
        )
        b = io.BytesIO()
        Image.fromarray(pixels).save(b, "PNG")
        return b.getvalue()

    monkeypatch.setattr(ingest, "fetch_bytes", fake_fetch)
    report = ingest.fetch_bootstrap(tmp_path)
    assert report["downloaded"] == 4
    assert len(report["errors"]) == 1
    assert len(audit_manifest(tmp_path / "manifest.jsonl", bootstrap=True)) == 4
    attribution = json.loads((tmp_path / "attribution.json").read_text())
    assert attribution[0]["artist"] == "Research Author"
    with pytest.raises(ValueError, match="already exists"):
        ingest.fetch_bootstrap(tmp_path)


def test_downloader_rejects_untrusted_hosts_redirects_and_oversized_files(monkeypatch):
    from xquantum import ingest

    for url in ["http://commons.wikimedia.org/a", "https://localhost/a", "https://example.org/a"]:
        with pytest.raises(ValueError, match="HTTPS Wikimedia"):
            ingest.fetch_bytes(url)

    class Response(io.BytesIO):
        url = "https://upload.wikimedia.org/file"

    monkeypatch.setattr(ingest.urllib.request, "urlopen", lambda *a, **kw: Response(b"abc"))
    assert ingest.fetch_bytes("https://upload.wikimedia.org/file") == b"abc"
    with pytest.raises(ValueError, match="limit"):
        ingest.fetch_bytes("https://upload.wikimedia.org/file", max_bytes=2)
    Response.url = "https://example.org/file"
    with pytest.raises(ValueError, match="redirect"):
        ingest.fetch_bytes("https://upload.wikimedia.org/file")
