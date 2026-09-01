# 2x2 pilot report (redesigned: post-hoc deterministic rendering)

Worlds: 30

## Gate 1: validity >= 95%
{'validity_rate': 1.0, 'pass': True}
PASS

## Gate 2: renderer template separation (sanity check, not model-dependent)
gap > 0.15
{'mean_within_style_cosine': 0.759355426897846, 'mean_cross_style_cosine': 0.408005182170381, 'gap': 0.35135024472746496, 'pass': True}
PASS

## Gate 3: Delta_BC inversion (threshold-independent mechanism gate)
Delta_BC = E[cos(B)] - E[cos(C)], margin > 0.1, bootstrap CI lower bound > 0
{'mean_cos_B': 0.7586835882641139, 'mean_cos_C': 0.5163586320279293, 'delta_bc': 0.24232495623618455, 'delta_bc_ci95': [0.21577799363533068, 0.27030197885644647], 'n_worlds_b': 30, 'n_worlds_c': 30, 'pass': True}
PASS

## Note: estimate-quality equivalence across styles
Not applicable in this design -- there is a single frozen elicitation; the estimate is identical regardless of which representation is rendered downstream, by construction (render.py never reads the estimate).

## Overall: ALL GATES PASS

STOP here for human review before running the balanced dedup calibration or the full 2x2 evaluation.
