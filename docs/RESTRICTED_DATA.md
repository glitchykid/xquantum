# Restricted and non-free image sources

Non-free images are supported when the intended AI research use is authorized.
Do not treat educational display rights or a paid single-image licence as automatic
training permission. The project has not purchased a dataset or contacted a rights
holder. No restricted photos have been added to the public bootstrap.

## Manifest and training

For a locally supplied licensed dataset, specify these additional Sample fields:

```json
{
  "license": "Provider-specific AI research agreement",
  "use_scope": "licensed_research",
  "permission_reference": "Internal agreement REF-001; applicable AI-use scope",
  "image_redistribution_allowed": false
}
```

These are extra fields in the full manifest contract, not a complete sample.
`use_scope` is `open`, `noncommercial_research`, or `licensed_research`. An unrecognized
licence cannot silently use the `open` category. Restricted categories require a
nonempty `permission_reference`; this records an operator's attestation and does not
verify the agreement's legal sufficiency. Keep the actual agreement in the authorized
source system; never embed confidential contracts, credentials or signed URLs in a
manifest. The reference must cover the actual intended training/evaluation use.

`image_redistribution_allowed` defaults to false. Set it true only when the applicable
terms permit redistribution with their required obligations. Research use does not
imply public image distribution. Existing reviewed-data auditing still requires
specialist labels and patient grouping; licensing does not upgrade label quality.

Training reports record `restricted_source_count`, `source_images_redistributable`
and `model_release_review_required`. Any restricted source triggers model-release
review. These fields are informational, not an automated legal review or enforcement
by the release workflow. Keep restricted datasets and resulting models in secure,
ignored directories until their use and release terms have been reviewed.

## Source assessment, checked 2026-10-30

| Source | Finding | Current project status |
|---|---|---|
| [DermNet image licence](https://dermnetnz.org/image-licence) | Website images explicitly excluded from AI training/testing; a paid licensable AI dataset is offered | Licensing candidate; no website images downloaded |
| [ISIC archive licence FAQ](https://www.isic-archive.com/blank-1) | Terms vary per image, including CC-BY-NC; permissions beyond the licence require rights-holder authorization | Source reference only; skin-lesion labels have not been substituted for hair/scalp diagnoses |

Select a disease-relevant cohort and obtain its actual use terms before importing
photos. The Wikimedia downloader continues to use its explicit open-image catalog;
it is not a general-purpose scraper. Licensed dataset files must be supplied through
the provider's authorized delivery mechanism. No payment or external contact is
initiated by this source assessment.
