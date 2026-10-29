"""Audit provenance, content integrity and evaluation leakage before training."""

import hashlib
from pathlib import Path

from xquantum.imaging import decode_image, perceptual_hash
from xquantum.schemas import Sample


def sample_path(manifest: Path, sample: Sample) -> Path:
    root = manifest.resolve().parent
    target = (root / sample.image_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Image path is outside manifest directory")
    return target


def audit_manifest(manifest: Path, *, bootstrap: bool) -> list[Sample]:
    rows = [
        Sample.model_validate_json(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Manifest is empty")
    patients, sources, seen_content, seen_pixels = {}, {}, set(), set()
    hashes = []
    for row in rows:
        if not bootstrap:
            if row.label_status != "specialist_reviewed" or not row.patient_id:
                raise ValueError("Reviewed training requires specialist review and patient_id")
            if row.photo_type == "unknown":
                raise ValueError("Reviewed training requires known photo_type")
        elif row.split != "train":
            raise ValueError(
                "Bootstrap images must all be train; unknown patients prevent evaluation"
            )
        for key, mapping in [(row.patient_id, patients), (row.source_group, sources)]:
            if key is not None and key in mapping and mapping[key] != row.split:
                raise ValueError("Patient or source-group leakage between splits")
            mapping[key] = row.split
        raw = sample_path(manifest, row).read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        if content_hash != row.sha256:
            raise ValueError("Image checksum mismatch")
        image = decode_image(raw)
        pixel_hash = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        if content_hash in seen_content or pixel_hash in seen_pixels:
            raise ValueError("Exact duplicate image in manifest")
        seen_content.add(content_hash)
        seen_pixels.add(pixel_hash)
        fingerprint = perceptual_hash(image)
        for other, split in hashes:
            if split != row.split and (fingerprint ^ other).bit_count() <= 4:
                raise ValueError("Possible near-duplicate leakage between splits; review required")
        hashes.append((fingerprint, row.split))
    return rows
