#!/usr/bin/env python
"""E50 phase 2 -- LGC2-Net under the same desynchronisation and dead-band correction.

    python experiments/E50_resync/lgc2_analyze.py

Threshold frozen at the ALIGNED 90%-sensitivity point (E46). Raw +-80 ms come from E46.
First checks that the shift-map hook actually fired: in the clean+dead-band run, only
subjects that have a non-zero map entry may change.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
EXP = HERE.parent
LGC = EXP.parent / "external" / "lgc2net" / "LGC2_Net_HFrEF"
SEED = 20261002


def load(p):
    return {r["subject"]: (int(r["label"]), float(r["probability"]))
            for r in csv.DictReader(open(p, encoding="utf-8"))}


A = load(LGC / "per_subject_predictions_ALIGNED.csv")
subs = sorted(A)
y = np.array([A[s][0] for s in subs])
pa = np.array([A[s][1] for s in subs])
thr = np.sort(pa[y == 1])[::-1][int(np.ceil(0.9 * y.sum())) - 1]
RUNS = {"aligned": pa}
for nm, f in (("fixed+80_raw", EXP / "E46_lgc2net_direction/preds/PCGp080.csv"),
              ("fixed-80_raw", EXP / "E46_lgc2net_direction/preds/PCGm080.csv")):
    R = load(f); RUNS[nm] = np.array([R[s][1] for s in subs])
for f in sorted((HERE / "lgc2_preds").glob("*.csv")):
    R = load(f); RUNS[f.stem] = np.array([R[s][1] for s in subs])

# hook check
m = json.loads((HERE / "lgc2_maps" / "none_deadband.json").read_text())
in_map = {k.rsplit("_", 1)[0] for k in m}
changed = {s for s, a, b in zip(subs, pa, RUNS["none_deadband"]) if abs(a - b) > 1e-9}
check = {"subjects_in_map": len(in_map), "subjects_changed": len(changed),
         "changed_outside_map": len(changed - in_map)}
print("hook check:", check)

rng = np.random.default_rng(SEED)
bs = [b for b in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if 0 < y[b].sum() < len(b)]
res = {"n": len(y), "n_pos": int(y.sum()), "hook_check": check, "runs": {}}
for nm, p in RUNS.items():
    spe = (p[y == 0] < thr).mean()
    res["runs"][nm] = {"auroc": round(float(roc_auc_score(y, p)), 4),
                       "sen": round(float((p[y == 1] >= thr).mean()), 4),
                       "spe": round(float(spe), 4),
                       "spe_ci": [round(float(np.percentile([(p[b][y[b] == 0] < thr).mean()
                                                             for b in bs], q)), 4) for q in (2.5, 97.5)]}
    r = res["runs"][nm]
    print(f"  {nm:20s} AUROC {r['auroc']:.4f}  sen {100 * r['sen']:5.1f}  spe {100 * r['spe']:5.1f} "
          f"[{100 * r['spe_ci'][0]:.1f}, {100 * r['spe_ci'][1]:.1f}]")
(HERE / "e50_lgc2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
