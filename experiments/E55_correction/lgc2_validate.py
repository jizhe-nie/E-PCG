#!/usr/bin/env python
"""E55 on LGC2-Net -- the pre-specified primary correction on the published model.

    python experiments/E55_correction/lgc2_validate.py maps      # write shift maps
    bash   experiments/E55_correction/run_lgc2.sh               # CPU inference, ~1 h per map
    python experiments/E55_correction/lgc2_validate.py analyze

Only the PRIMARY method (bayes_anchor) is run on LGC2-Net, under the same per-site offsets
as E50 (none, U(-100,100) per site, +80, -80 ms). Raw (uncorrected) conditions are
already on disk from E46/E50 and are reused. The comparison method is E50's symmetric 50
ms band on LGC2-Net; note that E50 and E55 each trained their own delay operator (same
data, same seed, GPU nondeterminism), so the two corrections are not computed from
bit-identical offset estimates.
Map entries are (offset_ms - correction_ms) * 2 samples at LGC2-Net's 2 kHz.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
EXP = HERE.parent
LGC = EXP.parent / "external" / "lgc2net" / "LGC2_Net_HFrEF"
SITES = ("APEX", "LLSB", "LUSB", "RUSB")
CONDS = {"none": "none", "random": "random", "fixed+80": "fixed+80", "fixed-80": "fixed-80"}
SEED = 20261004


def _subjects():
    spec = importlib.util.spec_from_file_location("e12", EXP / "E12_pinet" / "run.py")
    m = importlib.util.module_from_spec(spec); sys.modules["e12"] = m; spec.loader.exec_module(m)
    return [str(s) for s in m.build_cache(False)["subjects"]]


def maps():
    z = np.load(HERE / "e55_corr.npz")
    subs = _subjects()
    (HERE / "lgc2_maps").mkdir(exist_ok=True)
    for c in CONDS:
        M = z[f"off__{c}"] - np.round(z[f"bayes_anchor__{c}"])
        m = {f"{s}_{site}": int(2 * M[i, j]) for i, s in enumerate(subs)
             for j, site in enumerate(SITES) if int(M[i, j])}
        (HERE / "lgc2_maps" / f"{c}_bayesanchor.json").write_text(json.dumps(m), encoding="utf-8")
        print(f"{c:9s} {len(m)} entries, mean |shift| {np.mean(np.abs(list(m.values()))) / 2 if m else 0:.1f} ms")


def load(p):
    return {r["subject"]: (int(r["label"]), float(r["probability"]))
            for r in csv.DictReader(open(p, encoding="utf-8"))}


def analyze():
    A = load(LGC / "per_subject_predictions_ALIGNED.csv")
    subs = sorted(A)
    y = np.array([A[s][0] for s in subs]); pa = np.array([A[s][1] for s in subs])
    thr = np.sort(pa[y == 1])[::-1][int(np.ceil(0.9 * y.sum())) - 1]
    runs = {
        "raw": {"none": LGC / "per_subject_predictions_ALIGNED.csv",
                "random": EXP / "E50_resync/lgc2_preds/random_raw.csv",
                "fixed+80": EXP / "E46_lgc2net_direction/preds/PCGp080.csv",
                "fixed-80": EXP / "E46_lgc2net_direction/preds/PCGm080.csv"},
        "band50": {"none": EXP / "E50_resync/lgc2_preds/none_deadband.csv",
                   "random": EXP / "E50_resync/lgc2_preds/random_deadband.csv",
                   "fixed+80": EXP / "E50_resync/lgc2_preds/fixed+80_deadband.csv",
                   "fixed-80": EXP / "E50_resync/lgc2_preds/fixed-80_deadband.csv"},
        "bayes_anchor": {c: HERE / "lgc2_preds" / f"{c}_bayesanchor.csv" for c in CONDS}}
    P = {}
    for m, files in runs.items():
        for c, fp in files.items():
            if fp.exists():
                R = load(fp); P[(m, c)] = np.array([R[s][1] for s in subs])
    res = {"n": len(subs), "n_pos": int(y.sum()), "threshold": "aligned 90% sensitivity", "runs": {}}
    for (m, c), p in P.items():
        res["runs"].setdefault(m, {})[c] = {"auroc": round(float(roc_auc_score(y, p)), 4),
                                            "sen": round(float((p[y == 1] >= thr).mean()), 4),
                                            "spe": round(float((p[y == 0] < thr).mean()), 4)}
    rng = np.random.default_rng(SEED)
    for c in CONDS:
        if ("bayes_anchor", c) in P and ("band50", c) in P:
            a, b = P[("bayes_anchor", c)], P[("band50", c)]
            D = []
            for _ in range(2000):
                i = rng.integers(0, len(y), len(y))
                yb = y[i]
                if 0 < yb.sum() < len(i):
                    D.append([(a[i][yb == 1] >= thr).mean() - (b[i][yb == 1] >= thr).mean(),
                              (a[i][yb == 0] < thr).mean() - (b[i][yb == 0] < thr).mean()])
            D = np.array(D)
            res.setdefault("primary_minus_band50", {})[c] = {
                "sen": round(float((a[y == 1] >= thr).mean() - (b[y == 1] >= thr).mean()), 4),
                "sen_ci": [round(float(np.percentile(D[:, 0], q)), 4) for q in (2.5, 97.5)],
                "spe": round(float((a[y == 0] < thr).mean() - (b[y == 0] < thr).mean()), 4),
                "spe_ci": [round(float(np.percentile(D[:, 1], q)), 4) for q in (2.5, 97.5)]}
    for m, v in res["runs"].items():
        print(m, {c: (r["sen"], r["spe"], r["auroc"]) for c, r in v.items()})
    print(json.dumps(res.get("primary_minus_band50", {}), indent=1))
    (HERE / "e55_lgc2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    {"maps": maps, "analyze": analyze}[sys.argv[1]]()
