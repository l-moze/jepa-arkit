# Resource intake for the first real gate

The codebase can validate contracts with synthetic fixtures, but E00 and later gates require
external evidence. Place resources under `data/` (ignored by Git); do not commit licensed media,
personal data, credentials, or Unreal assets.

## D0A inputs

1. Source inventory: one row per dataset in
   `configs/data/licensing_matrix.template.csv`, with saved license/consent evidence and SHA-256.
2. Candidate media: 16 kHz mono WAV plus source video where pseudo-label generation is required.
3. Identity metadata: stable speaker, face identity, source, language, and withdrawal identifiers.
4. Product-track recordings: consent must explicitly cover model training, derivatives, retention,
   withdrawal, and the intended deployment. Research-only data cannot initialize product weights.

Do not assume MEAD, VOCASET, or any third-party corpus is eligible. The audit blocks every
`unverified` rights profile.

## E00 inputs

- At least 200 clips stratified by identity, expression, pose, occlusion, and articulation.
- Two annotators plus one adjudicator.
- Two to three hours selected for refined Gold labels.
- A pinned face solver version and its model/license files.

The optional `solver` dependency and a versioned MediaPipe Face Landmarker adapter are implemented.
The official model asset remains local under `data/models/face_landmarker.task`; its expected hash
is recorded in `data/README.md`. The adapter uses Face Landmarker blendshape categories and blocks
output when the canonical schema has an undeclared/missing category. This is still a pseudo-label
solver: E00 must validate it against refined Gold and UE renders before D0B.

## Audio feature inputs

The training interface expects precomputed, timestamped feature tensors. Before extraction,
record model ID, revision/hash, layer aggregation, frame rate, dtype, normalization, and rights.
The 8 GB local GPU is sufficient for smoke tests. Full WavLM/HuBERT extraction and formal runs
need a measured resource report; the research plan's upper compatibility target remains 24 GB.

## Unreal Engine inputs

- Exact UE 5.6 patch and MetaHuman/plugin versions.
- A commandlet-capable project with one validation level.
- Two complete MetaHumans, one approximately 80%-coverage ARKit character, and one full-body
  MetaHuman with an existing body animation.
- Asset paths for the neutral pose, face rig, head-neck constraints, and output sequences.

UE 5.8 is a separate migration target. It does not replace the 5.6 acceptance profile without a
full A0 rerun.

## Secrets and large files

Credentials belong in the local secret store or environment variables. Large features should be
sample-addressable shards with a `clip_id -> withdrawal_key -> shard/row` index so deletion can
rewrite only affected shards and revoke their descendants.
