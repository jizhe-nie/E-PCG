#!/usr/bin/env python
"""E81 -- 实验1：时间失配下的关键对照实验（按导师方案执行，方案文件 2026-10-30）。

    python experiments/E81_controlled/run.py --smoke      # 1 fold, 1 training seed: pipeline + self-checks
    python experiments/E81_controlled/run.py              # 5 folds x seeds 0,1,2

QUESTIONS. (1) Does device-level calibration reduce false positives while keeping sensitivity?
(2) Is the full method better than simpler alternatives? Main cohort only; software-injected offsets.

METHODS (A-E share one PINet; H, I share one DO-PINet)
  A  no calibration: conv baseline PINet on the raw inputs
  B  known common offset: the injected common offset is removed, per-recording jitter kept (reference)
  C  traditional peak method: per beat, envelope peak in [-200, 300] ms around each R peak, median over beats;
     device offset from 20 pool patients via the training response curve; PCG shifted back; then A
  D  traditional cross-correlation: R-peak unit-impulse train vs the same envelope, normalised cross-correlation
     over [-200, 300] ms, maximum; same pool calibration; then A          (second priority, run here too)
  E  learned delay operator (DelayOp): same pool calibration; then A
  F  late-fusion PINet (ECG and PCG encoded and pooled separately), no calibration
  G  random time-shift training: A's architecture, training batches get an independent U(-100, 100) ms PCG
     offset per site with probability 1 (the paper's fixed "pure augmentation" configuration); no calibration
  H  full method: 20-patient device calibration (E's estimate) + dual-mode DO-PINet ON (paper's main config)
  I  H's weights without device calibration; per-recording alignment and ON mode kept
  J  ECG only: late-fusion architecture with the PCG channel always zero (same four sites' ECG)
SCENARIOS  S0 no offset (C, D, E, H still calibrate); S1 every PCG +80 ms; S2 per simulated device (= one fold)
  common offset ~ U(40, 200) ms, plus independent U(-30, 30) ms jitter per patient and site; 20 simulations,
  simulation k uses seed 20261030 + k. ECG never moves. Offsets rounded to whole ms (1 kHz samples).
SPLITS (list seed 20261030, identical for all methods)
  the dataset's patient-level five folds; no independent validation set exists, so 20% of each fold's training
  part is drawn per class (stratified) as validation; each test fold: 20 random patients form the calibration
  pool (never scored, labels never used), the rest are scored. Models and the delay operator are trained on the
  remaining training patients only. All sites and windows of a patient stay in one group.
CALIBRATION RULE (training data only, no disease labels)
  mu_site = median over training patients' clean recordings of the measured delay at that site;
  response curve g(o) = median over training site-recordings of (tau at injected offset o) - mu_site,
  o = -40..250 ms in 10 ms steps (the existing E63b/E65 procedure), restricted to its monotone part;
  device offset = g^{-1}(median of the pool's valid (tau - mu_site)) by linear interpolation; a median outside
  the curve's range is clamped to the end point and counted as out-of-range ("越界"); if no pool site gives a
  valid measurement the whole pool fails and the method falls back to no calibration (counted, not dropped).
  C/D: a site fails with < 3 R peaks or < 3 valid beats (C: the envelope maximum must be interior to the window);
  failures are logged, scored patients are never removed. All measurers read the same 10 s window
  (10-20 s of the buffer); the PCG at offset o is read o ms earlier in the buffer, never circularly shifted.
THRESHOLD (validation set, no offset; locked before scoring)
  the validation threshold whose sensitivity is closest to 90%, ties -> higher specificity; A-E use A's,
  H and I use I's (no device calibration on clean validation data), F, G, J their own.
METRICS  each (scenario, simulation, seed): pool the five folds' scored patients, then sensitivity,
  specificity, AUROC, MCC; S0 and S1 averaged over 3 seeds, S2 over 3 seeds x 20 simulations.
SELF-CHECKS  S0: A and B identical; J identical across S0, S1 and every S2 simulation.
OUTPUTS (experiments/E81_controlled/)  lists.json, predictions.csv, confusion.csv, calibration.csv,
  config.json (thresholds per fold/seed/method, curves and ranges, G's augmentation), failures.log, summary.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from epcg.preprocess.rpeak import assess_ecg  # noqa: E402
from epcg.preprocess.sti_panel import BAND_S12, _env  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
LIST_SEED = 20261030
TRAIN_SEEDS = [0, 1, 2]
N_POOL, N_SIM = 20, 20
GC = np.arange(-40, 251, 10)
WIN_LO, WIN_HI = -200, 300
FS = 1000
METHODS = list("ABCDEFGHIJ")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E52 = _load("e52_run", "E52_dual_mode/run.py")
E50 = E52.E50
E77 = _load("e77_run", "E77_seconds_offset/run.py")
E79 = _load("e79_run", "E79_modality_dropout/run.py")
E56 = _load("e56_run", "E56_desync_aug/run.py")
OP_START, OP_WIN = E50.OP_START, E50.OP_WIN


# ----------------------------------------------------------------------------- splits
def make_lists(y, fold):
    rng = np.random.default_rng(LIST_SEED)
    L = {}
    for f in range(5):
        tr_all, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        val = np.concatenate([rng.choice(tr_all[y[tr_all] == c], size=int(round(0.2 * (y[tr_all] == c).sum())), replace=False)
                              for c in (0, 1)])
        pool = rng.choice(te, size=N_POOL, replace=False)
        L[f] = {"train": np.setdiff1d(tr_all, val), "val": np.sort(val), "pool": np.sort(pool), "score": np.setdiff1d(te, pool)}
    return L


def make_offsets(n):
    """scenario -> list of (sim id, per-fold common offset (5,), jitter (n, 4))."""
    S = {"S0": [(0, np.zeros(5), np.zeros((n, 4), int))], "S1": [(0, np.full(5, 80.0), np.zeros((n, 4), int))], "S2": []}
    for k in range(1, N_SIM + 1):
        r = np.random.default_rng(LIST_SEED + k)
        S["S2"].append((k, np.round(r.uniform(40, 200, 5)), np.round(r.uniform(-30, 30, (n, 4))).astype(int)))
    return S


# ----------------------------------------------------------------------------- traditional measurers (C, D)
class Traditional:
    """R peaks and 20-400 Hz envelopes cached once; tau for any site at any PCG offset by indexing."""

    def __init__(self, X):
        self.X = X
        self.R, self.ENV = {}, {}

    def prep(self, ids):
        for i in ids:
            for j in range(4):
                if (i, j) in self.R:
                    continue
                q, pk = assess_ecg(self.X[i, j, 0].astype(float), FS)
                self.R[(i, j)] = np.asarray(pk, int) if (q.usable and len(pk) >= 3) else None
                e = _env(self.X[i, j, 1].astype(float), FS, BAND_S12)
                self.ENV[(i, j)] = ((e - e.mean()) / (e.std() + 1e-9)).astype(np.float32)

    def tau(self, i, j, o, kind):
        """(tau ms or nan, failure reason or ''). ECG window [OP_START, OP_START+OP_WIN); PCG at ECG time t is env[t-o]."""
        pk = self.R[(i, j)]
        if pk is None:
            return np.nan, "rpeak"
        env = self.ENV[(i, j)]
        lo_p, hi_p = OP_START - o, OP_START - o + OP_WIN               # PCG samples visible in the window
        r = pk[(pk >= OP_START) & (pk < OP_START + OP_WIN)]
        r = r[(r + WIN_LO - o >= lo_p) & (r + WIN_HI - o < hi_p)]     # whole search range inside the window
        if len(r) < 3:
            return np.nan, "beats"
        lags = np.arange(WIN_LO, WIN_HI + 1)
        seg = np.stack([env[rr + WIN_LO - o: rr + WIN_HI - o + 1] for rr in r])     # beats x lags
        if kind == "C":
            am = seg.argmax(1)
            ok = (am > 0) & (am < len(lags) - 1)                       # interior maximum only
            if ok.sum() < 3:
                return np.nan, "beats"
            return float(np.median(lags[am[ok]])), ""
        prof = seg.mean(0)                                             # impulse train x z-scored envelope
        prof = (prof - prof.mean()) / (prof.std() + 1e-9)
        return float(lags[int(prof.argmax())]), ""


# ----------------------------------------------------------------------------- calibration
def build_curve(measure, train_ids):
    """mu_site (4,), curve gt over GC, monotone mask; measure(ids, offsets(n,4)) -> tau (n,4) with nan = failed."""
    t0 = measure(train_ids, np.zeros((len(train_ids), 4), int))
    mu = np.array([np.nanmedian(t0[:, j]) for j in range(4)])
    gt = np.array([np.nanmedian(measure(train_ids, np.full((len(train_ids), 4), int(o))) - mu[None]) for o in GC])
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(gt)) > 0])
    return mu, gt, keep


def estimate(measure, pool, offs_pool, mu, gt, keep):
    """device offset from the pool -> (c_hat, out_of_range, pool_failed)."""
    v = (measure(pool, offs_pool) - mu[None]).ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, False, True
    med = float(np.median(v))
    lo, hi = gt[keep].min(), gt[keep].max()
    return float(np.interp(med, gt[keep], GC[keep])), bool(med < lo or med > hi), False


# ----------------------------------------------------------------------------- thresholds and metrics
def pick_threshold(p, yv):
    """validation threshold with sensitivity closest to 90%; ties -> higher specificity."""
    best = None
    for t in np.unique(p[yv == 1]):
        se, sp = (p[yv == 1] >= t).mean(), (p[yv == 0] < t).mean()
        key = (abs(se - 0.9), -sp)
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1]


def metrics(yt, p, pred):
    tp, fn = int((pred & (yt == 1)).sum()), int((~pred & (yt == 1)).sum())
    tn, fp = int((~pred & (yt == 0)).sum()), int((pred & (yt == 0)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "sen": tp / (tp + fn), "spe": tn / (tn + fp),
            "auroc": roc_auc_score(yt, p), "mcc": (tp * tn - fp * fn) / den if den else 0.0}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--op-epochs", type=int, default=40)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    cfg = dict(E50.E49.E44.CFG)
    d = E50.E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    Y = E50.emat_targets(subs)
    L = make_lists(y, fold)
    OFF = make_offsets(len(y))
    folds = [0] if args.smoke else list(range(5))
    seeds = TRAIN_SEEDS[:1] if args.smoke else TRAIN_SEEDS
    tag = "_smoke" if args.smoke else ""
    odir = OUT / ("smoke" if args.smoke else "full")
    odir.mkdir(exist_ok=True)
    (odir / "lists.json").write_text(json.dumps({f: {k: [str(subs[i]) for i in v] for k, v in L[f].items()} for f in L},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    TR = Traditional(X)
    fail_log = open(odir / "failures.log", "w", encoding="utf-8")
    pred_rows, cal_rows, config = [], [], {"thresholds": {}, "curves": {},
                                          "G_augmentation": {"range_ms": [-100, 100], "per": "site, independently per batch",
                                                             "probability": 1.0},
                                          "curve_offsets_ms": GC.tolist(), "search_window_ms": [WIN_LO, WIN_HI],
                                          "interpolation": "linear on the monotone part; outside -> clamped and counted"}
    P = {}                                    # (seed, scen, sim, method) -> probabilities over all patients (nan = not scored)
    for f in folds:
        tr, va, pool, sc = (L[f][k] for k in ("train", "val", "pool", "score"))
        TR.prep(np.concatenate([tr, pool, sc, va]))
        # ---- delay operator (E, H, I): trained on this fold's training patients only
        op = E50.train_op(X, Y, tr, args, device)
        mu_op, k_op = E50.calibrate(op, X, tr, device)

        def m_op(ids, offs):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, _ = E50.op_tau(op, X, ix, np.asarray(offs).reshape(-1).astype(int), device)
            return tau.reshape(-1, 4)

        def m_trad(kind):
            def m(ids, offs):
                out = np.full((len(ids), 4), np.nan)
                for a, i in enumerate(ids):
                    for j in range(4):
                        t, why = TR.tau(i, j, int(offs[a, j]), kind)
                        out[a, j] = t
                        if why:
                            fail_log.write(f"fold {f} method {kind} subject {subs[i]} site {j} offset {int(offs[a, j])}: {why}\n")
                return out
            return m

        meas = {"C": m_trad("C"), "D": m_trad("D"), "E": m_op}
        curves = {m: build_curve(fn, tr) for m, fn in meas.items()}
        config["curves"][f] = {m: {"mu_site": c[0].tolist(), "g": c[1].tolist(), "monotone_range_ms": [float(c[1][c[2]].min()), float(c[1][c[2]].max())]}
                               for m, c in curves.items()}
        # ---- device calibration for every scenario/simulation (independent of the training seed)
        CH = {}
        for scen, sims in OFF.items():
            for k, cvec, jit in sims:
                c_true = float(cvec[f])
                offs_pool = (c_true + jit[pool]).astype(int)
                for m, fn in meas.items():
                    ch, oor, pf = estimate(fn, pool, offs_pool, *curves[m])
                    CH[(scen, k, m)] = ch
                    for s in seeds:
                        cal_rows.append({"fold": f, "seed": s, "sim": k, "scenario": scen, "method": m,
                                         "pool_ids": " ".join(str(subs[i]) for i in pool), "c_true_ms": c_true,
                                         "c_hat_ms": round(ch, 2), "out_of_range": int(oor), "pool_failed": int(pf)})
        # ---- DO-PINet timing features from training patients
        def measure_dh(ids, offs):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, sg = E50.op_tau(op, X, ix, np.asarray(offs).reshape(-1).astype(int), device)
            return np.clip((tau.reshape(-1, 4) - mu_op[None]) / k_op, -E52.CLIP_MS, E52.CLIP_MS), sg.reshape(-1, 4)

        dh_tr, sg_tr = measure_dh(tr, np.zeros((len(tr), 4), int))
        fm = np.array([dh_tr.mean(), sg_tr.mean()]); fsd = np.array([dh_tr.std(), sg_tr.std()]) + 1e-6
        feat = lambda dh, sg: ((np.stack([dh, sg], -1) - fm) / fsd).astype(np.float32)
        T2 = np.zeros((len(y), 4, 2), np.float32); T2[tr] = feat(dh_tr, sg_tr)
        A_al = np.zeros((len(y), 4), int); A_al[tr] = np.round(dh_tr).astype(int)

        def on_inputs(ids, offs):
            dh, sg = measure_dh(ids, offs)
            return feat(dh, sg), np.clip(np.round(dh).astype(int), offs - 290, offs + 290)

        for s in seeds:
            nets = {"pinet": E52.train(X, y, tr, T2, None, None, cfg, s, device),
                    "dual": E52.train(X, y, tr, T2, A_al, "dual", cfg, s, device),
                    "late": E77.train_late(X, y, tr, T2, cfg, s, device),
                    "aug": E56.train_aug(X, y, tr, 100, cfg, s, device),
                    "ecg": E79.train_with(E79.variant(force_zero=1), X, y, tr, cfg, s, device)}
            (odir / "weights").mkdir(exist_ok=True)
            for nm, net in nets.items():
                torch.save(net.state_dict(), odir / "weights" / f"fold{f}_seed{s}_{nm}.pt")
            if s == seeds[0]:
                torch.save(op.state_dict(), odir / "weights" / f"fold{f}_delayop.pt")
            plain = lambda net, ids, offs: E52.predict(nets[net], X, ids, offs, None, None, None, cfg, device)

            def on_pred(ids, offs):
                Tf, Al = on_inputs(ids, offs)
                return E52.predict(nets["dual"], X, ids, offs, Al, Tf, 1.0, cfg, device)

            # ---- thresholds on the clean validation set, locked
            z = np.zeros((len(va), 4), int)
            thr = {"A": pick_threshold(plain("pinet", va, z), y[va]), "F": pick_threshold(plain("late", va, z), y[va]),
                   "G": pick_threshold(plain("aug", va, z), y[va]), "J": pick_threshold(plain("ecg", va, z), y[va]),
                   "I": pick_threshold(on_pred(va, z), y[va])}
            for m in "BCDE":
                thr[m] = thr["A"]
            thr["H"] = thr["I"]
            config["thresholds"][f"fold{f}_seed{s}"] = thr
            # ---- scoring
            for scen, sims in OFF.items():
                for k, cvec, jit in sims:
                    c_true = float(cvec[f])
                    offs = (c_true + jit[sc]).astype(int)
                    shift = lambda c: (offs - int(round(c))).astype(int)
                    probs = {"A": plain("pinet", sc, offs), "B": plain("pinet", sc, shift(c_true)),
                             "C": plain("pinet", sc, shift(CH[(scen, k, "C")])),
                             "D": plain("pinet", sc, shift(CH[(scen, k, "D")])),
                             "E": plain("pinet", sc, shift(CH[(scen, k, "E")])),
                             "F": plain("late", sc, offs), "G": plain("aug", sc, offs),
                             "H": on_pred(sc, shift(CH[(scen, k, "E")])), "I": on_pred(sc, offs),
                             "J": plain("ecg", sc, offs)}
                    for m, p in probs.items():
                        key = (s, scen, k, m)
                        if key not in P:
                            P[key] = (np.full(len(y), np.nan), np.full(len(y), np.nan))
                        P[key][0][sc] = p
                        P[key][1][sc] = thr[m]
                        for a, i in enumerate(sc):
                            pred_rows.append({"patient_id": str(subs[i]), "fold": f, "seed": s, "sim": k, "scenario": scen,
                                              "method": m, "label": int(y[i]), "prob": round(float(p[a]), 6),
                                              "threshold": round(thr[m], 6), "pred": int(p[a] >= thr[m])})
            print(f"  fold {f} seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    fail_log.close()

    # ---- merge folds, metrics per (seed, scenario, sim, method)
    conf = []
    for (s, scen, k, m), (p, t) in P.items():
        ok = np.isfinite(p)
        r = metrics(y[ok], p[ok], p[ok] >= t[ok])
        conf.append({"seed": s, "scenario": scen, "sim": k, "method": m, "n": int(ok.sum()), "n_pos": int(y[ok].sum()), **r})
    summ = {}
    for scen in OFF:
        summ[scen] = {}
        for m in METHODS:
            rows = [r for r in conf if r["scenario"] == scen and r["method"] == m]
            summ[scen][m] = {kk: float(np.mean([r[kk] for r in rows])) for kk in ("sen", "spe", "auroc", "mcc")}
            summ[scen][m]["runs"] = len(rows)
    diffs = {}
    for scen, a, b in (("S2", "E", "A"), ("S2", "E", "C"), ("S2", "H", "I"), ("S2", "H", "F"), ("S2", "H", "G"),
                       ("S2", "H", "J"), ("S0", "E", "A"), ("S0", "H", "I"), ("S2", "E", "D")):
        diffs[f"{scen}:{a}-{b}"] = {kk: 100 * (summ[scen][a][kk] - summ[scen][b][kk]) for kk in ("sen", "spe")}
    calerr = {}
    for scen in OFF:
        for m in "CDE":
            rows = [r for r in cal_rows if r["scenario"] == scen and r["method"] == m and r["seed"] == seeds[0]]
            e = np.array([r["c_hat_ms"] - r["c_true_ms"] for r in rows])
            calerr[f"{scen}:{m}"] = {"mean_error_ms": float(e.mean()), "mae_ms": float(np.abs(e).mean()),
                                     "out_of_range": f"{sum(r['out_of_range'] for r in rows)}/{len(rows)}",
                                     "pool_failed": f"{sum(r['pool_failed'] for r in rows)}/{len(rows)}"}
    # ---- self-checks
    chk = {}
    for s in seeds:
        pa, pb = P[(s, "S0", 0, "A")][0], P[(s, "S0", 0, "B")][0]
        chk[f"S0 A==B seed{s}"] = bool(np.nanmax(np.abs(pa - pb)) < 1e-6)
        pj0 = P[(s, "S0", 0, "J")][0]
        dj = max(np.nanmax(np.abs(P[(s, scen, k, "J")][0] - pj0)) for scen, sims in OFF.items() for k, _, _ in sims)
        chk[f"J identical across scenarios seed{s}"] = bool(dj < 1e-5)
        chk[f"J max prob diff seed{s}"] = float(dj)
    scored = np.isfinite(P[(seeds[0], "S0", 0, "A")][0])
    res = {"folds": folds, "seeds": seeds, "n_scored": int(scored.sum()), "n_pos": int(y[scored].sum()),
           "n_neg": int((y[scored] == 0).sum()), "summary": summ, "diffs_pct_points": diffs,
           "calibration_error": calerr, "self_checks": chk, "elapsed_min": round((time.time() - t0) / 60, 1)}
    (odir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (odir / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, rows in (("predictions.csv", pred_rows), ("confusion.csv", conf), ("calibration.csv", cal_rows)):
        with open(odir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(json.dumps({k: res[k] for k in ("n_scored", "n_pos", "self_checks", "calibration_error")}, indent=1, ensure_ascii=False))
    for scen in OFF:
        print(scen)
        for m in METHODS:
            v = summ[scen][m]
            print(f"  {m}  sen {100 * v['sen']:5.1f}  spe {100 * v['spe']:5.1f}  AUROC {v['auroc']:.3f}  MCC {v['mcc']:.3f}")
    print({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in diffs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
