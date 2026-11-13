#!/usr/bin/env python
"""E65 -- device self-calibration in front of the published LGC2-Net (frozen official weights).

    python experiments/E65_lgc2_selfcal/run.py maps          # GPU: operator, c_hat, d_hat -> shift maps
    bash   experiments/E65_lgc2_selfcal/run_lgc2.sh          # CPU: LGC2-Net inference per map
    python experiments/E65_lgc2_selfcal/run.py analyze

WHY. For deployed models that cannot be retrained, the draft offered only the 50 ms
tolerance band (E50), which trades sensitivity for specificity. E63b/E64 showed that a
device's constant latency can be estimated from its other patients and removed before
classification. That step needs no access to the model's internals, so it applies to
LGC2-Net as a front end.

SCENARIO. E61's uncalibrated wireless-audio offsets (same draws). Maps (offset minus
correction, ms -> 2 kHz samples): audio_raw, audio_band50 (E50 rule on the operator's
d_hat at the injected offset), audio_selfcal (E63b estimator, N = 20 other test-fold
subjects at cohort prevalence), audio_calib (true constant latency), and none_selfcal
(self-calibration on a device without latency: the cost on clean input). The clean
reference is LGC2-Net's aligned prediction file. LGC2-Net's hook applies the shift with
np.roll, as in E46/E50.

PRIMARY (stated before running): audio, selfcal minus band50, paired sensitivity and
specificity at the threshold frozen on aligned data at 90% sensitivity (E50/E55 LGC2
convention), bootstrap over subjects. Secondary: selfcal minus calib; none_selfcal minus aligned.
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
ROOT = EXP.parent
LGC = ROOT / "external" / "lgc2net" / "LGC2_Net_HFrEF"
sys.path.insert(0, str(ROOT))
SITES = ("APEX", "LLSB", "LUSB", "RUSB")
GRID = np.arange(-40, 251, 5)
GC = np.arange(-40, 251, 10)
N_POOL = 20
SEED = 20261010
MAPS = ["audio_raw", "audio_band50", "audio_selfcal", "audio_calib", "none_selfcal"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def maps():
    import torch
    E61 = _load("e61_run", "E61_sync_value/run.py")
    E50, E55 = E61.E50, E61.E55

    class A:
        op_epochs = 40
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = E61.E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    Y = E50.emat_targets(subs)
    off, c = E61.scenarios(len(y), np.random.default_rng(E61.SEED))["audio"]
    rng = np.random.default_rng(SEED)
    CH = {k: np.full(len(y), np.nan) for k in ("none", "audio")}
    DH = np.full((len(y), 4), np.nan)
    for f in range(5):
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        op = E50.train_op(X, Y, tr, A, dev)
        mu, k = E50.calibrate(op, X, tr, dev)
        idx = [(i, j) for i in te for j in range(4)]
        idx_tr = [(i, j) for i in tr for j in range(4)]
        G = np.empty((len(te), 4, len(GRID)))
        for g, o in enumerate(GRID):
            tau, _ = E50.op_tau(op, X, idx, np.full(len(idx), int(o)), dev)
            G[:, :, g] = tau.reshape(-1, 4) - mu
        gt = np.array([np.median(E50.op_tau(op, X, idx_tr, np.full(len(idx_tr), int(o)), dev)[0].reshape(-1, 4) - mu)
                       for o in GC])
        keep = np.concatenate([[True], np.diff(np.maximum.accumulate(gt)) > 0])
        pos = {i: n for n, i in enumerate(te)}
        for n, i in enumerate(te):
            DH[i] = [np.interp(off[i, j], GRID, G[n, j]) / k for j in range(4)]      # E50's d_hat scale
            for cond in ("none", "audio"):
                cand = te[te != i]
                pool = rng.choice(cand, size=N_POOL, replace=False)
                jit = rng.uniform(-30, 30, (N_POOL, 4)) if cond == "audio" else np.zeros((N_POOL, 4))
                cd = c[i, 0] if cond == "audio" else 0.0
                vals = [np.interp(cd + jit[m, j], GRID, G[pos[p], j]) for m, p in enumerate(pool) for j in range(4)]
                CH[cond][i] = float(np.interp(np.median(vals), gt[keep], GC[keep]))
        print(f"  fold {f} done", flush=True)
    DH = np.clip(DH, -E50.CLIP_MS, E50.CLIP_MS)
    corr = {"audio_raw": np.zeros_like(DH), "audio_band50": E55.band(DH, 50, 50),
            "audio_selfcal": CH["audio"][:, None] * np.ones((1, 4)), "audio_calib": c}
    shifts = {m: off - np.round(v) for m, v in corr.items()}
    shifts["none_selfcal"] = -np.round(CH["none"][:, None] * np.ones((1, 4)))
    (HERE / "lgc2_maps").mkdir(exist_ok=True)
    for m, S in shifts.items():
        mp = {f"{s}_{site}": int(2 * S[i, j]) for i, s in enumerate(map(str, subs))
              for j, site in enumerate(SITES) if int(S[i, j])}
        (HERE / "lgc2_maps" / f"{m}.json").write_text(json.dumps(mp), encoding="utf-8")
        print(f"{m:14s} {len(mp)} entries, mean |shift| {np.mean(np.abs(S)):.1f} ms, max {np.abs(S).max():.0f} ms")
    e = CH["audio"] - c[:, 0]
    meta = {"c_hat_audio_bias_ms": round(float(e.mean()), 1), "c_hat_audio_mae_ms": round(float(np.abs(e).mean()), 1),
            "c_hat_clean_bias_ms": round(float(CH["none"].mean()), 1), "c_hat_clean_mae_ms": round(float(np.abs(CH["none"]).mean()), 1)}
    print(meta)
    (HERE / "e65_maps_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(HERE / "e65_shifts.npz", c_hat_audio=CH["audio"], c_hat_none=CH["none"], dhat=DH, off=off, c=c,
             subjects=np.array(list(map(str, subs))))


def _read(p):
    return {r["subject"]: (int(r["label"]), float(r["probability"])) for r in csv.DictReader(open(p, encoding="utf-8"))}


def analyze():
    Al = _read(LGC / "per_subject_predictions_ALIGNED.csv")
    subs = sorted(Al)
    y = np.array([Al[s][0] for s in subs]); pa = np.array([Al[s][1] for s in subs])
    thr = np.sort(pa[y == 1])[::-1][int(np.ceil(0.9 * y.sum())) - 1]
    P = {"aligned": pa}
    for m in MAPS:
        fp = HERE / "lgc2_preds" / f"{m}.csv"
        if fp.exists():
            R = _read(fp); P[m] = np.array([R[s][1] for s in subs])
    ss = lambda p, i: ((p[i][y[i] == 1] >= thr).mean(), (p[i][y[i] == 0] < thr).mean())
    full = np.arange(len(y))
    res = {"n": len(subs), "n_pos": int(y.sum()), "threshold": "aligned 90% sensitivity",
           "runs": {m: {"auroc": round(float(roc_auc_score(y, p)), 4), "sen": round(float(ss(p, full)[0]), 4),
                        "spe": round(float(ss(p, full)[1]), 4)} for m, p in P.items()}}
    rng = np.random.default_rng(SEED)
    boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if 0 < y[i].sum() < len(i)]
    res["paired"] = {}
    for a, b in (("audio_selfcal", "audio_band50"), ("audio_selfcal", "audio_calib"), ("audio_selfcal", "audio_raw"),
                 ("none_selfcal", "aligned")):
        if a in P and b in P:
            D = np.array([[u - v for u, v in zip(ss(P[a], i), ss(P[b], i))] for i in boots])
            pa_, pb_ = ss(P[a], full), ss(P[b], full)
            res["paired"][f"{a}-{b}"] = {"sen": round(float(pa_[0] - pb_[0]), 4),
                                         "sen_ci": [round(float(np.percentile(D[:, 0], q)), 4) for q in (2.5, 97.5)],
                                         "spe": round(float(pa_[1] - pb_[1]), 4),
                                         "spe_ci": [round(float(np.percentile(D[:, 1], q)), 4) for q in (2.5, 97.5)]}
    for m, r in res["runs"].items():
        print(f"  {m:14s} AUROC {r['auroc']:.3f}  sen {100 * r['sen']:5.1f}  spe {100 * r['spe']:5.1f}")
    for k_, v in res["paired"].items():
        print(f"  {k_:28s} sen {100 * v['sen']:+5.1f} [{100 * v['sen_ci'][0]:+.1f}, {100 * v['sen_ci'][1]:+.1f}]  "
              f"spe {100 * v['spe']:+5.1f} [{100 * v['spe_ci'][0]:+.1f}, {100 * v['spe_ci'][1]:+.1f}]")
    (HERE / "e65_lgc2_selfcal.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    {"maps": maps, "analyze": analyze}[sys.argv[1]]()
