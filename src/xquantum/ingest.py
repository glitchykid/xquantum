"""Fetch an explicit, small open-license catalog, never search-engine pseudo-labels."""

import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from xquantum.imaging import MAX_BYTES, decode_image
from xquantum.schemas import Sample

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "XQuantumResearch/0.1 (https://github.com/glitchykid/XQuantum)"
CATALOG = {
    "alopecia_areata": [
        "Alopecia areata 0001.jpg",
        "Alopecia areata 1.jpg",
        "Alopecia areata 2.jpg",
        "Alopecia areata bald spot.jpg",
        "Alopecia areata head.jpg",
    ],
    "tinea_capitis": [
        "4809 lores.jpg",
        "Fungal infection on head scalp.jpg",
        "Teigne - Tinea capitis.jpg",
        "Teigne tondante enfant.jpg",
        "Tinea capitis.jpg",
    ],
}
ALLOWED_LICENSES = {
    "Public domain",
    "CC0",
    "CC BY 2.0",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 2.0",
    "CC BY-SA 2.5",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
}


def fetch_bytes(url: str, max_bytes: int = MAX_BYTES) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "commons.wikimedia.org",
        "upload.wikimedia.org",
    }:
        raise ValueError("Downloader permits HTTPS Wikimedia hosts only")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        final = urllib.parse.urlsplit(response.url)
        if final.scheme != "https" or final.hostname not in {
            "commons.wikimedia.org",
            "upload.wikimedia.org",
        }:
            raise ValueError("Unexpected image redirect")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Download exceeds byte limit")
    return raw


def plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", "", value)).strip()


def fetch_bootstrap(destination: Path) -> dict:
    """Write train-only data, per-file attribution, and a transparent failure log."""
    if (destination / "manifest.jsonl").exists():
        raise ValueError("Manifest already exists; use a new destination to preserve provenance")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir(exist_ok=True)
    samples, attribution, errors, seen = [], [], [], set()
    for label, titles in CATALOG.items():
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "titles": "|".join("File:" + t for t in titles),
        }
        metadata = json.loads(fetch_bytes(COMMONS_API + "?" + urllib.parse.urlencode(params)))
        for page in metadata.get("query", {}).get("pages", {}).values():
            title = page.get("title", "unknown")
            try:
                info = page["imageinfo"][0]
                ext = info["extmetadata"]
                license_name = ext["LicenseShortName"]["value"]
                if license_name not in ALLOWED_LICENSES:
                    raise ValueError(f"Unsupported license: {license_name}")
                raw = fetch_bytes(info["url"])
                image = decode_image(raw)
                # Re-encode pixels only: remove EXIF, GPS, comments and embedded metadata.
                clean = image.copy()
                clean.info.clear()
                image_id = hashlib.sha256(title.encode()).hexdigest()[:16]
                relative = f"images/{image_id}.jpg"
                file = destination / relative
                clean.save(file, "JPEG", quality=92)
                sha256 = hashlib.sha256(file.read_bytes()).hexdigest()
                if sha256 in seen:
                    file.unlink()
                    raise ValueError("Duplicate image content")
                seen.add(sha256)
                sample = Sample(
                    image_path=relative,
                    label=label,
                    patient_id=None,
                    source_group="wikimedia_unknown_patient_cohort",
                    source_url=info["descriptionurl"],
                    license=license_name,
                    rights_basis="Public Wikimedia license recorded; research bootstrap only",
                    sha256=sha256,
                    photo_type="scalp",
                )
                samples.append(sample.model_dump(mode="json"))
                attribution.append(
                    {
                        "image_path": relative,
                        "title": title,
                        "source_page": info["descriptionurl"],
                        "original_url": info["url"],
                        "original_sha256": hashlib.sha256(raw).hexdigest(),
                        "license": license_name,
                        "license_url": ext.get("LicenseUrl", {}).get("value"),
                        "artist": plain(
                            ext.get("Artist", {}).get("value", "Unknown; see source page")
                        ),
                        "credit": plain(ext.get("Credit", {}).get("value", "See source page")),
                        "description": plain(ext.get("ImageDescription", {}).get("value", "")),
                        "modifications": "JPEG re-encoded; metadata removed; label retained",
                        "retrieved_at": datetime.now(UTC).isoformat(),
                    }
                )
            except (KeyError, IndexError, ValueError, OSError) as exc:
                errors.append({"title": title, "error": str(exc)})
    (destination / "manifest.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
    (destination / "attribution.json").write_text(json.dumps(attribution, indent=2) + "\n")
    report = {
        "downloaded": len(samples),
        "errors": errors,
        "class_counts": {label: sum(s["label"] == label for s in samples) for label in CATALOG},
        "label_quality": "source_only",
        "patient_identity_known": False,
        "evaluation_permitted": False,
    }
    (destination / "ingestion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(report["class_counts"][label] < 2 for label in CATALOG):
        raise ValueError(
            "Insufficient photos per class; inspect ingestion_report.json before training"
        )
    return report
