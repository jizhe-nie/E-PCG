"""Paired comparisons with one estimand across tables and confidence intervals.

Every table in the paper reports the MEAN OVER SEEDS of a per-seed sensitivity /
specificity. The paired differences used to be computed on seed-AVERAGED probabilities
(a 3-seed ensemble) -- a different estimand, so the point estimate of a difference did
not equal the difference of the table entries (editorial review, E66). From E55 on, the
re-run scripts use this module instead:

    point  = mean over seeds of [metric(arm A, seed) - metric(arm B, seed)]
    CI     = percentile bootstrap over SUBJECTS; one resample is applied to every seed,
             so seed-to-seed variation is averaged, not resampled.

metric(arm, seed, idx) -> (sensitivity, specificity) on the subjects idx; each script
supplies it with its own threshold rule (re-derived inside the resample when the rule is
'90% sensitivity on this arm's clean scores').
"""

from __future__ import annotations

import numpy as np


def boots(y, B=2000, seed=0):
    """Subject resamples that contain both classes."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < B:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(i)
    return out


def paired(metric, a, b, seeds, n, bs):
    """Seed-mean paired difference of (sen, spe) between arms a and b, with bootstrap CI."""
    full = np.arange(n)
    diff = lambda idx: np.mean([np.subtract(metric(a, s, idx), metric(b, s, idx)) for s in seeds], 0)
    pt = diff(full)
    D = np.array([diff(i) for i in bs])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"sen": round(float(pt[0]), 4), "sen_ci": q(D[:, 0]), "spe": round(float(pt[1]), 4), "spe_ci": q(D[:, 1]),
            "estimand": "mean over seeds of per-seed difference; bootstrap over subjects"}


def own_threshold_metric(P, y, clean_cond):
    """metric for scripts whose threshold is 90% sensitivity on the arm's own clean scores.
    P[(seed, arm, cond)] -> scores over all subjects (already restricted to evaluated ones).
    Arms are passed as (arm, cond) tuples."""
    def m(arm_cond, s, idx):
        arm, cond = arm_cond
        yb = y[idx]
        pc, p = P[(s, arm, clean_cond)][idx], P[(s, arm, cond)][idx]
        t = np.sort(pc[yb == 1])[::-1][int(np.ceil(0.9 * yb.sum())) - 1]
        return (p[yb == 1] >= t).mean(), (p[yb == 0] < t).mean()
    return m
