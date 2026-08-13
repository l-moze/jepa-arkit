# JEPA-ARKit

Auditable research implementation for speech-driven ARKit facial motion. The repository
implements the readiness gates in `docs/jepa-arkit-research-plan.md` before expensive model
comparisons are allowed.

## Current milestone

The local research pipeline now runs end to end on real, identity-disjoint RAVDESS data:

- 24 official archives, 1,440 audio-video clips, checksum verification, canonical MediaPipe
  motion labels, and WavLM Base FP16 features.
- E01 direct causal baseline, E10 Motion-JEPA, E11 Audio-Motion-JEPA, and E03 regional
  conditional residual training.
- Offline canonical export, a frozen streaming reference, deterministic 30-to-60 fps
  interpolation, provenance sidecars, and failure diagnostics.
- Three-seed E03 evaluation. Its deterministic mean exactly preserves E01 while seeded style
  sampling changes only eyes, brows, gaze, and head motion.

The current streaming and deterministic recommendation is E01. E03 is an optional offline style
layer pending perceptual and Unreal Engine validation. E11 is not promoted because the current
pure-audio pilot does not show a clear deployment-level win.

All current real checkpoints are `research-only` and `pilot_non_comparable`: RAVDESS is
CC BY-NC-SA 4.0 and the labels are Silver machine supervision. They are not product checkpoints.

## Quick start

```powershell
uv sync --extra dev
uv run jepa-arkit status --output artifacts/project_status.json
uv run jepa-arkit evaluate-direct --config configs/experiments/e01_ravdess_v2.yaml `
  --checkpoint runs/e01_ravdess_v2/best_checkpoint.pt --split test
uv run jepa-arkit infer-regional `
  --config configs/experiments/e03_regional_ravdess_seed2.yaml `
  --checkpoint runs/e03_regional_ravdess_seed2/best_checkpoint.pt `
  --audio path/to/mono_16khz_pcm.wav --output artifacts/inference/regional.npz `
  --sampling-seed 17 --regional-temperature 0.75 --head-temperature 0.5
uv run pytest
```

The 2.44 GiB UniTalker release candidate is also downloaded and CRC-audited, but it remains in
quarantine because the archive has no umbrella training license and several upstream sources are
gated or research-only. See `docs/implementation-status.md` for the exact remaining gates.

## Repository layout

```
configs/        # YAML experiment configs (configs/experiments) + JSON contracts (configs/contracts)
docs/           # research plan, implementation status, dataset guide, top-venue survey, evidence, sessions
src/jepa_arkit/ # package source: models, training, data pipeline, solver, streaming, features
tests/          # pytest suite
data/           # local-only data store (gitignored except data/README.md)
runs/           # local training outputs (gitignored)
artifacts/      # local reports and exported results (gitignored)
```

Key documents:

- [docs/jepa-arkit-research-plan.md](docs/jepa-arkit-research-plan.md) - research plan, gates, and dual-track governance
- [docs/implementation-status.md](docs/implementation-status.md) - current milestone and gate status
- [docs/dataset-download-guide.md](docs/dataset-download-guide.md) - how to re-fetch all datasets
- [docs/top-venue-survey-2022-2026.md](docs/top-venue-survey-2022-2026.md) - CVPR/ICCV/ECCV landscape survey
