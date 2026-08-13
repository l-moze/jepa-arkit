# Local data layout

This directory is ignored except for this file. Never commit licensed media, biometric data,
credentials, model assets, or generated feature caches.

Expected local resources:

```text
data/
  models/face_landmarker.task
  raw/<dataset_id>/...
  releases/<release_id>/manifest.jsonl
  features/<feature_release_id>/metadata.json
```

The Face Landmarker model currently used for E00 has SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` and was retrieved from the
official MediaPipe model storage on 2026-08-13. The binary remains local and is not redistributed.

The local RAVDESS research release is stored under `raw/ravdess/`. Its 24 source archives are
retained beside `release_v1/`, which contains 1,440 full audio-video speech clips and a
source-addressable `inventory.jsonl`. RAVDESS is research-only in this project; see
`configs/data/ravdess_rights_registry.json`.
