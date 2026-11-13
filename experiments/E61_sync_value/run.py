#!/usr/bin/env python
"""E61 -- the value of synchronisation information: how much of the specificity lost to
an ECG/PCG offset comes back if the offset is KNOWN, and how precisely must it be known?

    python experiments/E61_sync_value/run.py --smoke
    python experiments/E61_sync_value/run.py --seeds 3

WHY. E50/E55/E59/E60: every correction computed from the recording itself lies on one
sensitivity-specificity frontier, because a single recording cannot separate a device
offset from a pathological R->S1 prolongation. The way past that limit is information
from outside the recording (protocol-level clock sync, a factory latency calibration, a
sync tap). This experiment prices that information, and replaces the arbitrary
U(-100, 100) ms stress test with offset scenarios anchored in published figures.

OFFSET SCENARIOS (ms, positive = PCG later; per-site draws, seeded; PRE-SPECIFIED)
  synced       U(-2, 2)         application-layer BLE sync: mean abs error 0.07-0.48 ms,
                                95th pct < 1.8 ms (PMC10144216)
  ble_naive    U(0, 15)         receive-time stamps: delay "near zero up to one connection
                                interval", 10-15 ms in that work (PMC10144216)
  audio        c + U(-30, 30)   wireless audio path, c ~ U(40, 200) per SUBJECT (one device
                                per subject): HFP ~40 ms, A2DP typically 150-200 ms, range
                                34-300 ms; per-recording jitter +-30 ms is an assumption
  stress       U(-100, 100)     E50's draw (same seed), kept for continuity

INFORMATION ARMS (correction subtracted from the injected offset before scoring)
  raw          nothing
  band50       E50's software correction (symmetric 50 ms tolerance on d_hat)
  calib        the scenario's constant component is known (factory or one-time tap
               calibration): audio -> c, ble_naive -> 7.5, synced/stress -> 0
  oracle_s     the per-recording offset is known with Gaussian error s in {0, 5, 10, 20, 40}
               ms (s=0: perfect sync). The error is also applied to clean input, where the
               true offset is 0 -- a real sync system has its error everywhere.

PRIMARY ESTIMANDS (stated before running)
  1. audio scenario: calib - band50 and oracle_10 - band50, paired sensitivity and
     specificity differences, bootstrap over subjects.
  2. required precision: the largest s in the grid at which, under audio, specificity is
     within 3 points of raw clean specificity and sensitivity is no more than 2 points below
     raw clean sensitivity (descriptive; resolution limits as in E60 apply).
Thresholds: each arm's own clean scores at 90% sensitivity (E55 convention). PINet and
the operator are retrained per fold exactly as in E55, so raw/band50 under stress
reproduce E55.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path(__file__).parent
EXP = OUT.parent
SIGMAS = [0, 5, 10, 20, 40]
CLIP = 250                     # E50 MARGIN 300: |offset - correction| must stay <= 250
SEED = 20261005


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E55 = _load("e55_run", "E55_correction/run.py")
E50, E49 = E55.E50, E55.E49
band = E55.band


def scenarios(n, rng):
    """(offset (n,4) int ms, known constant component (n,4) float ms) per scenario."""
    rs = np.random.default_rng(E50.SEED)                  # E50's exact stress draw
    sc = {"none": (np.zeros((n, 4), int), np.zeros((n, 4)))}
    sc["synced"] = (rng.integers(-2, 3, (n, 4)), np.zeros((n, 4)))
    sc["ble_naive"] = (rng.integers(0, 16, (n, 4)), np.full((n, 4), 7.5))
    c = rng.uniform(40, 200, (n, 1)) * np.ones((1, 4))
    sc["audio"] = (np.round(c + rng.uniform(-30, 30, (n, 4))).astype(int), c)
    sc["stress"] = (rs.integers(-E50.RAND_MS, E50.RAND_MS + 1, (n, 4)), np.zeros((n, 4)))
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--op-epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.op_epochs, args.seeds = 2, 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    LV = (d["lvef"] * 100).astype(np.float32)
    STI = np.nan_to_num(d["sti"].astype(np.float32))
    MSK = d["sti_mask"].astype(bool)
    Y = E50.emat_targets(subs)
    rng = np.random.default_rng(SEED)
    SC = scenarios(len(y), rng)
    NOISE = {s: rng.normal(0, s, (len(y), 4)) for s in SIGMAS}      # shared across scenarios
    arms = ["raw", "band50", "calib"] + [f"oracle_{s}" for s in SIGMAS]
    folds = [0] if args.smoke else range(5)
    seeds = [E55.SEED + 101 * i for i in range(args.seeds)]
    P = {(s, a, c): np.full(len(y), np.nan) for s in seeds for a in arms for c in SC}
    CORR = {(a, c): np.full((len(y), 4), np.nan) for a in arms for c in SC}
    DH = {c: np.full((len(y), 4), np.nan) for c in SC}

    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        op = E50.train_op(X, Y, tr, args, device)
        mu, k = E50.calibrate(op, X, tr, device)
        for c, (off, const) in SC.items():
            o = off[te]
            idx = [(i, j) for i in te for j in range(4)]
            tau, _ = E50.op_tau(op, X, idx, o.reshape(-1), device)
            dh = np.clip((tau.reshape(-1, 4) - mu) / k, -CLIP, CLIP)
            DH[c][te] = dh
            corr = {"raw": np.zeros_like(dh), "band50": band(dh, 50, 50), "calib": const[te]}
            for s in SIGMAS:
                corr[f"oracle_{s}"] = o + NOISE[s][te]
            for a in arms:
                CORR[(a, c)][te] = corr[a]
        for s in seeds:
            net = E55.train_seed(X, y, LV, STI, MSK, tr, cfg, s, device)
            for c, (off, _) in SC.items():
                for a in arms:
                    resid = np.clip(off[te] - np.round(CORR[(a, c)][te]).astype(int), -CLIP, CLIP)
                    P[(s, a, c)][te] = E50.pinet_prob(net, X, te, resid, cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    k_ = ~np.isnan(P[(seeds[0], "raw", "none")])
    yy = y[k_]
    hf = yy == 1
    nthr = int(np.ceil(0.9 * yy.sum())) - 1
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "arms": {},
           "offset_summary_ms": {c: [round(float(v), 1) for v in np.percentile(o[k_], [5, 50, 95])]
                                 for c, (o, _) in SC.items()}}
    for a in arms:
        res["arms"][a] = {}
        for c in SC:
            rows = []
            for s in seeds:
                pc, p = P[(s, a, "none")][k_], P[(s, a, c)][k_]
                t = np.sort(pc[hf])[::-1][nthr]
                rows.append([roc_auc_score(yy, p), (p[hf] >= t).mean(), (p[~hf] < t).mean()])
            rows = np.array(rows)
            res["arms"][a][c] = {key: [round(float(rows[:, i].mean()), 4), round(float(rows[:, i].std()), 4)]
                                 for i, key in enumerate(("auroc", "sen", "spe"))}

    PS = _load("pairstats", "_pairstats.py")
    Pk = {kk: v[k_] for kk, v in P.items()}
    metric = PS.own_threshold_metric(Pk, yy, "none")
    bs = PS.boots(yy, seed=SEED)
    res["primary"] = {}
    for c in ("audio", "stress", "ble_naive"):
        for a in ("calib", "oracle_10", "oracle_0"):
            res["primary"][f"{c}:{a}-band50"] = PS.paired(metric, (a, c), ("band50", c), seeds, len(yy), bs)
    np.savez_compressed(OUT / "e61_scores.npz", y=yy, **{f"{s}__{a}__{c}": v for (s, a, c), v in Pk.items()})
    ref = res["arms"]["raw"]["none"]
    ok = [s for s in SIGMAS if res["arms"][f"oracle_{s}"]["audio"]["spe"][0] >= ref["spe"][0] - 0.03
          and res["arms"][f"oracle_{s}"]["audio"]["sen"][0] >= ref["sen"][0] - 0.02]
    res["required_precision_ms"] = max(ok) if ok else None

    conds = list(SC)
    print(f"\n{len(seeds)} seed(s); own-threshold sen/spe (%)   offsets p5/p50/p95: {res['offset_summary_ms']}")
    print(f"{'arm':10s} " + "  ".join(f"{c:>11s}" for c in conds))
    for a in arms:
        print(f"{a:10s} " + "  ".join(f"{100 * res['arms'][a][c]['sen'][0]:5.1f}/{100 * res['arms'][a][c]['spe'][0]:4.1f}"
                                      for c in conds))
    print("\npaired differences vs band50 (seed-averaged scores):")
    for k, v in res["primary"].items():
        print(f"  {k:26s} sen {100 * v['sen']:+5.1f} [{100 * v['sen_ci'][0]:+.1f}, {100 * v['sen_ci'][1]:+.1f}]  "
              f"spe {100 * v['spe']:+5.1f} [{100 * v['spe_ci'][0]:+.1f}, {100 * v['spe_ci'][1]:+.1f}]")
    print(f"\nrequired precision (audio, within 3 pts spe / 2 pts sen of clean): {res['required_precision_ms']} ms")
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e61_sync_value{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)},
                                                              indent=2), encoding="utf-8")
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
