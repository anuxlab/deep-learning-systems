# Model card — digits MLP (notebook 24)

**Model:** 64-256-128-10 MLP, 50,826 parameters, trained with Adam for 60 epochs on 1,200 8×8 digit images (scikit-learn `load_digits`).
**Intended use:** teaching/demo of evaluation practice. **Not** for any real decision-making.

## Metrics (held-out 597 images)
| metric | value |
|---|---|
| clean accuracy (fp32) | 0.980 |
| int8-quantised accuracy | 0.980 |
| accuracy under FGSM ε=0.15 (standard) | 0.157 |
| accuracy under FGSM ε=0.15 (adversarially trained variant) | 0.787 |
| expected calibration error (raw → temperature-scaled, other model) | 0.030 → 0.017 |

## Limitations & caveats
- Tiny, clean, low-resolution dataset; results do not transfer to natural images or distribution shift.
- Brittle to small adversarial perturbations unless adversarially trained (which costs clean accuracy).
- Single seed; confidence intervals not reported for these figures (see section 2 for how to compute them).
- Fairness metrics were illustrated on a *synthetic* two-group problem and say nothing about this model.
