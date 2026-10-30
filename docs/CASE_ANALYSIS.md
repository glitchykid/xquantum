# Multiple scalp views for one case

Submit one to six distinct views of the same patient, captured for the same case.
The software cannot verify patient identity, anatomy, viewpoint or capture time.
Do not combine photographs from different people or mix unvalidated modalities.

```bash
xquantum analyze-case --model artifacts/bootstrap \
  --image /secure/vertex.jpg /secure/temporal.jpg \
  --clinical examples/clinical.json
```

`POST /v1/analyze-case` accepts `images_base64` (a list of base64 strings) and
optional `clinical` context. The original `/v1/analyze` endpoint is unchanged.
The response contains ordered `views`, zero-based `view_index`, each view's quality
issues and research scores, and combined `research_scores`. The combination is an
arithmetic mean across all views, using the same rule as patient-level evaluation.
A single view gives the same scores as the single-image endpoint.

If any view fails the existing quality gates, combined scores are suppressed; the
response still describes each view for review. Malformed images or exact duplicate
decoded pixels reject the entire request. Recompressed or near-duplicate views may
not be recognized; the caller must select distinct, clinically relevant views.
No poor-quality view is silently dropped or replaced.

Limits: six views, 10 MiB total decoded file bytes, 40 million total decoded pixels,
and the existing per-image limits. HTTP JSON bodies remain capped at 14,100,000
bytes, including base64 and context. Oversized HTTP bodies return 413; invalid case
content returns 422. The CLI uses the same case validation. Analysis persists no
images, filenames or clinical context. View indices preserve input order without
returning local paths.

Every case output abstains from diagnosis. Combining photographs does not establish
clinical validity, improve measured accuracy, detect unfamiliar diseases or create
new training evidence. Shared clinical context affects scores only for a model
actually trained with paired clinical data.
