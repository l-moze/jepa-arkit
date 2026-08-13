# E03 regional residual pilot

Updated: 2026-08-13

E03 adds a conditional stochastic residual only to eyes, brows, gaze, and head motion. The frozen
E01 branch remains the deterministic mean for all 59 canonical dimensions, so sampling cannot
change lip sync or the nose/cheek controls by construction.

## Three-seed result

| Run | Validation stochastic variance | Test stochastic variance | Test head variance |
|---|---:|---:|---:|
| `e03_regional_ravdess_v1` | 0.002511 | 0.002658 | 0.007399 |
| `e03_regional_ravdess_seed2` | 0.002604 | 0.002598 | 0.007964 |
| `e03_regional_ravdess_seed3` | 0.006047 | 0.006574 | 0.022299 |

All runs use 5,222,485 parameters, 1.04514 times the E01 parameter count. Every run exactly
matches the corrected E01 test mean metrics and has a sampled-mouth maximum absolute difference
of zero. Seed2 is selected because it is closest to the median validation stochastic variance;
the selection does not use test metrics.

The architecture gate passes, but the perceptual diversity gate remains pending. Non-zero output
variance alone does not establish naturalness. Head variance is seed-sensitive, so deployment
uses separately recorded regional and head temperatures.

## Export invariant

For a real RAVDESS clip and the selected checkpoint:

- Two exports with sampling seed 17 are byte-identical NPZ files.
- Changing the seed leaves mouth and nose/cheek controls exactly unchanged.
- Mean absolute seed-to-seed differences are 0.0210 for eyes/brows, 0.0108 for gaze, 0.00264 for
  head quaternion components, and 0.0436 cm for head translation.

Evidence:

- `artifacts/e03_regional_three_seed_summary.json`
- `artifacts/e03_regional_export_invariants.json`
- `artifacts/inference/e03_seed2_style17_a.provenance.json`
