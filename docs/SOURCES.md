# Sources and exact APIs consulted

Checked 2026-10-30. These sources inform the design; they do not provide evidence
that XQuantum is clinically validated.

## Medical and dataset sources

- American Academy of Dermatology, [alopecia areata diagnosis and treatment](https://www.aad.org/public/diseases/hair-loss/types/alopecia/treatment):
  specialist examination, dermoscopy, history and selectively indicated investigations.
- Lee, Solam (2020), [Clinically Applicable Deep Learning Framework for Measuring the
  Extent of Hair Loss in Patients with Alopecia Areata](https://data.mendeley.com/datasets/75k76546ms/1),
  DOI 10.17632/75k76546ms.1. This release provides annotations; it explicitly excludes
  patient clinical photographs. It was not used as image training data.
- [Wikimedia Commons API](https://www.mediawiki.org/wiki/API:Imageinfo) supplies each
  selected file's attribution, description page and license metadata. Exact source
  records for downloaded photos are in `artifacts/bootstrap/attribution.json`.
- [Wikimedia reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia)
  explains per-file licensing and reuse obligations. The downloader uses an explicit
  catalog and license allowlist, not blanket permission for arbitrary medical images.

## Software documentation

- [scikit-learn 1.9.1 SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html):
  `loss="log_loss"`, explicit `partial_fit` epochs, class list and sample weighting.
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [NumPy 2.5 array archives](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html):
  numeric arrays loaded using `allow_pickle=False`.
- [Pillow 13.0 ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html):
  EXIF transposition and image padding.
- [Pydantic model validation](https://docs.pydantic.dev/latest/concepts/models/):
  explicit field validation, forbidden extra keys, finite numerical values.
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/) and
  [Ruff formatting](https://docs.astral.sh/ruff/formatter/).

Direct versions were verified using PyPI package metadata before implementation.
The exact resolved local environment is frozen in `requirements-lock.txt`, and model
reports retain the actual training library versions.

