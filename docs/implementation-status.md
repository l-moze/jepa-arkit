# Implementation status

Updated: 2026-08-13

## Completed locally

- Project package and locked CUDA PyTorch environment (`torch 2.13.0+cu132`).
- Canonical ARKit curve schema with 52 curves, 7 head state values, semantic groups, ranges,
  round-trip checks, and explicit MediaPipe -> canonical -> UE boundary.
- D0A/D0B manifest audit: split leakage, source/identity/withdrawal overlap, motion archive
  validation, quality fields, rights ancestry, file fingerprints, and sample-addressable withdrawal.
- Dataset catalog generated from `数据集目录可补充.md`; 14 candidates are classified by role and
  every unverified source has explicit blockers.
- Official evidence snapshots for VOCASET, MMHead, and RAVDESS. All three are research-only under
  the observed terms; none is a product-track candidate.
- Full RAVDESS audio-video speech release: 24 checksum-verified archives, 24 actors, and 1,440
  source-traceable real clips. All clips have been solved into relative-head canonical motion and
  16 kHz audio. This is a `D0P` Silver machine-label pilot, not E00/D0B Gold.
- MediaPipe Face Landmarker adapter and local official model asset hash. Missing canonical outputs
  or zero-face clips block label export.
- E01 direct causal model, Motion-JEPA stride-1 model, EMA target, random and causal-future masks,
  confidence-weighted losses, latent statistics, deterministic checkpoints, and identity-balanced
  sampler.
- Sample-addressable feature store with metadata, timestamp alignment, and withdrawal index.
- UE-independent character profile validator, provenance sidecar schema, failure report generator,
  environment/resource report, CLI, tests, and CI workflow.
- Exact-center WavLM Base v2 features (266,110 frames, 768-dim FP16) were extracted. E01 direct
  candidates were compared under the same identity-disjoint release and the full 256-dim/6-layer
  model was selected.
- The 2.44 GiB UniTalker release candidate was downloaded and CRC-audited (8,002 entries). It is
  quarantined rather than trained because it has no umbrella license and source-specific rights
  and annotation-space conversions remain unresolved.
- Real E10 Motion-JEPA and E11 Audio-Motion-JEPA training completed with random-span and 15-frame
  causal objectives. Teacher-forced E11 gains are not treated as pure-audio deployment gains.
- E03 regional conditional residual training completed for three seeds. Its `sample=False` output
  exactly equals E01 for every reported group, all seeds keep sampled mouth motion invariant, and
  the total parameter ratio is 1.045. Validation-median seed2 is the selected offline style pilot;
  perceptual diversity remains unproven.
- A real held-out clip was exported to canonical 30 fps NPZ with validated provenance metadata.
- Frozen streaming reference protocol was exercised on a real 36.13-second concatenated held-out
  trace: 1,084 contiguous frames, 217 overlapping chunks, 80 ms look-ahead, 1 s history, and
  42.5x real-time on the local GPU. Fixed-window right-aligned context with left neutral padding
  reduced chunk-boundary median jump to 1.174x the interior step.
- UE-independent 30-to-60 fps export uses linear curve/translation interpolation and shortest-path
  quaternion SLERP. A real held-out clip round-trips exactly at source frames (curve/translation
  error 0; quaternion error below `6e-8`).

## Verified results

| Gate | Result | Evidence |
|---|---|---|
| D0A synthetic fixture | Passed | 25 records, no audit issues |
| T0 direct infrastructure smoke | Passed | 40 GPU steps, 94.99% loss reduction, audio-shuffle ratio 1.28, silence ratio 1.23 |
| T0 JEPA infrastructure smoke | Passed | 40 GPU steps, 70.89% loss reduction, deterministic parameter hash |
| E10 representation smoke | Blocked | effective-rank ratio 8.79%, below the 25% research threshold; this is not treated as a JEPA success |
| D0A real candidate catalog | Acquired | RAVDESS 1.0.0 downloaded from Zenodo; 24/24 MD5 checks pass |
| D0P RAVDESS pilot | Passed, research-only | 1,440 clips, 24 actor-disjoint identities, Silver labels; single-source warning |
| E01 real direct baseline | Pilot passed, non-comparable | corrected eval-mode test mouth MAE 0.04164; 500 ms shift ratio 1.128; silence ratio 1.161 |
| E03 regional residual | Architecture passed, perception pending | 3 seeds preserve E01 mean exactly; sampled mouth max difference 0; selected seed2 by validation median |
| E10 Motion-JEPA | Pilot passed, non-comparable | causal validation loss 0.237; effective-rank ratio 0.284 |
| E11 Audio-Motion-JEPA | Pilot passed, non-comparable | audio-only pretrained mouth MAE 0.04990 vs random 0.05094; no product-level win |
| D0B/E00 | Blocked | requires an authorized dataset, 200 clips, refined Gold labels, and UE render validation |
| A0/E20 | Blocked | no UE 5.6 project or character assets detected on this machine |
| Streaming reference | Passed locally | real trace is contiguous and boundary ratio is 1.174; UE Live Link/AnimBP runtime still external |
| 30 -> 60 fps reference | Passed locally | independent NumPy exporter and round-trip report; UE-side playback validation still external |

Synthetic smoke artifacts are marked `synthetic_non_comparable` and cannot be used as research
results or product checkpoints.

## Immediate external inputs

1. Run E00 on 200 clips with two annotators and one adjudicator before calling any model D0B/Gold.
2. Accept and download VOCASET or MMHead for a true 3D motion anchor; RAVDESS labels are
   MediaPipe-derived Silver supervision, not Gold motion capture.
3. Provide a second authorized source or enough identities for the D0B split rules; a single
   dataset with one source is not silently treated as broad generalization.
4. Provide two annotators, one adjudicator, and the 2--3 hour Gold
   refinement allocation.
5. Provide the UE 5.6 exact patch project and four A0 character profiles/assets.
6. Provide product-track self-recorded/consented data separately if a product checkpoint is wanted.

## Current model recommendation

- Streaming and deterministic offline: `runs/e01_ravdess_v2/best_checkpoint.pt`.
- Optional offline style sampling: `runs/e03_regional_ravdess_seed2/best_checkpoint.pt`, with
  `regional_temperature=0.75` and `head_temperature=0.5` as conservative pilot defaults.
- E10 remains a representation result, not a deployable audio model.
- E11 is not promoted until a pure-audio, multi-seed result clears the preregistered gate.
