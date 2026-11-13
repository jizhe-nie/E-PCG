#!/usr/bin/env python
"""E50 -- can the delay operator rescue a classifier from desynchronisation?

    python experiments/E50_resync/run.py --smoke
    python experiments/E50_resync/run.py

This is the experiment that connects the two halves of the paper. E46 and E49 showed
that the classifiers (LGC2-Net, PINet) depend on ECG-PCG alignment without measuring it:
an 80 ms offset drops specificity at a frozen 90%-sensitivity threshold from ~90% to
27-39%. The delay operator does measure the relative delay (tracking 0.83-0.84, E47).
So it should be able to detect a device offset and undo it before classification.

PROTOCOL (per outer fold, the dataset authors' five; nothing from fold f is used to fit
anything applied to fold f)

  1. Train the delay operator on training subjects' site windows (R->S1 target, E47
     recipe), with its lag window widened to [-200, 300] ms so that an offset that
     moves S1 before R is still representable.
  2. On training subjects only: mu_site = mean raw tau per auscultation site on clean
     windows, and the slope k of tau against injected offsets {-80,-40,+40,+80} ms
     (soft-argmax shrinks shifts, E47 tracking 0.84). The offset estimate for any
     window is then  d_hat = (tau - mu_site) / k,  clipped to +-150 ms.
  3. Train PINet (classification only, E44 recipe). Freeze its 90%-sensitivity
     threshold on the clean out-of-fold predictions.
  4. Score the held-out fold under: clean | offset | offset + correction |
     clean + correction. The last one is the control: correction must not damage
     synchronised input.

Offsets are applied and estimated PER SITE -- in this dataset each auscultation site is a
separate recording, so a device offset need not be shared, and the estimator is not
allowed to pool sites. Conditions: fixed -80/-40/+40/+80 ms on every site, and an
independent U(-100, 100) ms draw per site (seeded), the realistic case.

WHAT WOULD MAKE THIS FAIL, STATED FIRST. tau carries the subject's own R->S1 interval
(SD ~28 ms) on top of the offset, so even a perfect operator leaves a residual of that
order. E49 says PINet tolerates about +-40 ms at clean-level specificity. If the residual
exceeds that, correction will not restore specificity, and the paper says so.
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcg.models.emat_arms import DelayOp  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
SEED = 20261002
FS = 1000.0
SITES = ("APEX", "LLSB", "LUSB", "RUSB")
FIXED_MS = [-80, -40, 40, 80]
RAND_MS = 100
CLIP_MS = 150
# Dead-band correction, pre-specified (not tuned): leave |d_hat| <= DEADBAND_MS alone and
# correct only the excess. 50 ms ~ 2 SD of the between-subject R->S1 spread (28 ms), so
# a genuinely prolonged interval (HFrEF mean d_hat on clean input: +30 ms in the full-
# correction run) is kept, while an instrument offset outside physiology is pulled back.
DEADBAND_MS = 50
MARGIN = 300            # buffer kept free at each end: |offset - correction| <= 250
OP_WIN = 10000          # 10 s operator window
OP_START = 10000        # operator window position inside the 30 s buffer


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E49 = _load("e49_run", "E49_pinet_timing/run.py")
E49.MARGIN = MARGIN     # PINet training crops avoid the same margin used at test time


def emat_targets(subjects):
    tgt = {}
    for r in csv.DictReader(open(EXP / "E11_deploy" / "e11_per_site.csv", encoding="utf-8")):
        if r["split"] != "primary" or r.get("emat_ms") in (None, "", "None"):
            continue
        v = float(r["emat_ms"])
        if np.isfinite(v) and 0.0 < v < 250.0:
            tgt[(r["subject"], r["site"])] = v
    T = np.full((len(subjects), 4), np.nan, np.float32)
    for i, s in enumerate(subjects):
        for j, site in enumerate(SITES):
            T[i, j] = tgt.get((str(s), site), np.nan)
    return T


# ------------------------------------------------------------------ delay operator
def op_windows(X, idx, offs):
    """(n, 2, OP_WIN) operator inputs. idx: list of (subject, site); offs: PCG offset in
    samples (positive = PCG later, read from earlier in the buffer)."""
    out = np.empty((len(idx), 2, OP_WIN), np.float32)
    for k, ((i, j), o) in enumerate(zip(idx, offs)):
        out[k, 0] = X[i, j, 0, OP_START:OP_START + OP_WIN]
        out[k, 1] = X[i, j, 1, OP_START - o:OP_START - o + OP_WIN]
    return out


def train_op(X, Y, tr_subj, args, device):
    pairs = [(i, j) for i in tr_subj for j in range(4) if np.isfinite(Y[i, j])]
    y = torch.tensor([Y[i, j] for i, j in pairs])
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    m = DelayOp(FS, lag_min_ms=-200.0, lag_max_ms=300.0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    bs = 32
    steps = max(args.op_epochs * int(np.ceil(len(pairs) / bs)), 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.25)
    mu, sd = float(y.mean()), float(y.std() + 1e-6)
    T = X.shape[-1]
    for ep in range(args.op_epochs):
        m.train()
        perm = rng.permutation(len(pairs))
        for b in range(0, len(perm), bs):
            jj = perm[b:b + bs]
            s = int(rng.integers(MARGIN, T - OP_WIN - MARGIN))      # random 10 s crop
            xb = torch.from_numpy(np.stack([X[pairs[k][0], pairs[k][1], :, s:s + OP_WIN]
                                            for k in jj])).to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss((m(xb)["pred"] - mu) / sd, (y[jj].to(device) - mu) / sd)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    return m


@torch.no_grad()
def op_tau(m, X, idx, offs, device, bs=64):
    taus, sig = [], []
    for b in range(0, len(idx), bs):
        xb = torch.from_numpy(op_windows(X, idx[b:b + bs], offs[b:b + bs])).to(device)
        o = m(xb)
        taus.append(o["tau_ms"].float().cpu().numpy()); sig.append(o["sigma_ms"].float().cpu().numpy())
    return np.concatenate(taus), np.concatenate(sig)


def calibrate(m, X, tr_subj, device):
    """mu per site and slope k, from training subjects only."""
    idx = [(i, j) for i in tr_subj for j in range(4)]
    t0, _ = op_tau(m, X, idx, np.zeros(len(idx), int), device)
    site = np.array([j for _, j in idx])
    mu = np.array([np.median(t0[site == j]) for j in range(4)])
    dx, dy = [], []
    for d in FIXED_MS:
        t, _ = op_tau(m, X, idx, np.full(len(idx), d), device)
        dx.append(np.full(len(idx), d)); dy.append(t - t0)
    dx, dy = np.concatenate(dx), np.concatenate(dy)
    k = float((dx * dy).sum() / (dx * dx).sum())
    return mu, k


# ------------------------------------------------------------------ PINet scoring
@torch.no_grad()
def pinet_prob(m, X, te, pcg_off, cfg, device):
    """3-crop TTA as E49, with a PER-SITE PCG offset (n_te, 4) in samples."""
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    P = np.zeros(len(te))
    for b in range(0, len(te), cfg["bs"]):
        jj = te[b:b + cfg["bs"]]
        acc = 0.0
        for t in range(cfg["tta"]):
            o = MARGIN + int((T - n) * t / max(cfg["tta"] - 1, 1))
            w = np.empty((len(jj), 4, 2, n), np.float32)
            for a, i in enumerate(jj):
                for j in range(4):
                    po = int(pcg_off[b + a, j])
                    w[a, j, 0] = X[i, j, 0, o:o + n]
                    w[a, j, 1] = X[i, j, 1, o - po:o - po + n]
            acc = acc + torch.softmax(m(torch.from_numpy(w).to(device))["logits"].float(), -1)[:, 1]
        P[b:b + len(jj)] = (acc / cfg["tta"]).cpu().numpy()
    return P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.op_epochs = 2, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    LV = (d["lvef"] * 100).astype(np.float32)
    STI = np.nan_to_num(d["sti"].astype(np.float32))
    MSK = d["sti_mask"].astype(bool)
    Y = emat_targets(subs)
    rng = np.random.default_rng(SEED)
    conds = {"none": np.zeros((len(y), 4), int)}
    for s in FIXED_MS:
        conds[f"fixed{s:+d}"] = np.full((len(y), 4), s, int)
    conds["random"] = rng.integers(-RAND_MS, RAND_MS + 1, size=(len(y), 4))
    folds = [0] if args.smoke else range(5)

    MODES = ("raw", "corr", "deadband")
    P = {f"{c}|{mode}": np.full(len(y), np.nan) for c in conds for mode in MODES}
    D_hat = {c: np.full((len(y), 4), np.nan) for c in conds}
    SIG = {c: np.full((len(y), 4), np.nan) for c in conds}
    calib = []
    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        op = train_op(X, Y, tr, args, device)
        mu, k = calibrate(op, X, tr, device)
        calib.append({"fold": f, "mu_site_ms": [round(float(v), 2) for v in mu], "k": round(k, 4)})
        net, _, _ = E49.train(X, y, LV, STI, MSK, tr, cfg, 0.0, 0.0, device)
        idx = [(i, j) for i in te for j in range(4)]
        for c, off in conds.items():
            o = off[te]                                                  # (n_te, 4)
            tau, sg = op_tau(op, X, idx, o.reshape(-1), device)
            dh = np.clip((tau.reshape(-1, 4) - mu[None]) / k, -CLIP_MS, CLIP_MS)
            D_hat[c][te] = dh; SIG[c][te] = sg.reshape(-1, 4)
            P[f"{c}|raw"][te] = pinet_prob(net, X, te, o, cfg, device)
            P[f"{c}|corr"][te] = pinet_prob(net, X, te, o - np.round(dh).astype(int), cfg, device)
            db = np.sign(dh) * np.maximum(np.abs(dh) - DEADBAND_MS, 0.0)
            P[f"{c}|deadband"][te] = pinet_prob(net, X, te, o - np.round(db).astype(int), cfg, device)
        print(f"  fold {f}: k={k:.3f}  mu={np.round(mu, 1).tolist()}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    k_ = ~np.isnan(P["none|raw"])
    yy = y[k_]
    clean = P["none|raw"][k_]
    thr = np.sort(clean[yy == 1])[::-1][int(np.ceil(0.9 * yy.sum())) - 1]
    rb = np.random.default_rng(SEED)
    boots = [b for b in (rb.integers(0, k_.sum(), k_.sum()) for _ in range(2000))
             if 0 < yy[b].sum() < len(b)]

    def score(p):
        return {"auroc": round(float(roc_auc_score(yy, p)), 4),
                "sen": round(float((p[yy == 1] >= thr).mean()), 4),
                "spe": round(float((p[yy == 0] < thr).mean()), 4)}

    # Diagnostic: the offset estimate on CLEAN input is the subject's own deviation in
    # R->S1 -- if it separates HFrEF, offset and pathology are confounded.
    dc = D_hat["none"][k_].mean(1)
    res_conf = {"clean_dhat_mean_hfref_ms": round(float(dc[yy == 1].mean()), 2),
                "clean_dhat_mean_other_ms": round(float(dc[yy == 0].mean()), 2),
                "clean_dhat_auroc_for_hfref": round(float(roc_auc_score(yy, dc)), 4)}
    # The deployable variant of each correction mode: threshold re-frozen on that mode's
    # own clean out-of-fold scores (a device that always corrects is calibrated that way).
    recal = {}
    for mode in ("corr", "deadband"):
        pc_ = P[f"none|{mode}"][k_]
        t_ = np.sort(pc_[yy == 1])[::-1][int(np.ceil(0.9 * yy.sum())) - 1]
        recal[mode] = {c: {"sen": round(float((P[f"{c}|{mode}"][k_][yy == 1] >= t_).mean()), 4),
                           "spe": round(float((P[f"{c}|{mode}"][k_][yy == 0] < t_).mean()), 4)}
                       for c in conds}
    res = {"confound": res_conf, "recalibrated_threshold": recal,
           "deadband_ms": DEADBAND_MS, "n": int(k_.sum()), "n_pos": int(yy.sum()), "calibration": calib,
           "threshold_frozen_on": "clean out-of-fold, 90% sensitivity", "conditions": {}}
    print(f"\n{'condition':10s} {'mode':5s}  AUROC   sen    spe    | offset residual "
          f"MAE (ms)  | sigma med")
    for c, off in conds.items():
        r = {}
        for mode in MODES:
            p = P[f"{c}|{mode}"][k_]
            r[mode] = score(p)
            r[mode]["spe_ci"] = [round(float(np.percentile(
                [(p[b][yy[b] == 0] < thr).mean() for b in boots], q)), 4) for q in (2.5, 97.5)]
        resid = (D_hat[c][k_] - off[k_]).ravel()
        r["offset_residual_mae_ms"] = round(float(np.abs(resid).mean()), 2)
        r["offset_residual_sd_ms"] = round(float(resid.std()), 2)
        r["offset_bias_ms"] = round(float(resid.mean()), 2)
        r["frac_residual_within_40ms"] = round(float((np.abs(resid) <= 40).mean()), 4)
        r["sigma_median_ms"] = round(float(np.median(SIG[c][k_])), 2)
        res["conditions"][c] = r
        for mode in MODES:
            s = r[mode]
            print(f"{c:10s} {mode:5s}  {s['auroc']:.4f}  {100 * s['sen']:5.1f}  {100 * s['spe']:5.1f}"
                  + (f"  | {r['offset_residual_mae_ms']:6.1f} (bias {r['offset_bias_ms']:+.1f}, "
                     f"<=40 ms {100 * r['frac_residual_within_40ms']:.0f}%) | {r['sigma_median_ms']:.1f}"
                     if mode == "corr" else ""), flush=True)
    (OUT / f"e50_resync{'_smoke' if args.smoke else ''}.json").write_text(
        json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2),
        encoding="utf-8")
    np.savez(OUT / f"e50_oof{'_smoke' if args.smoke else ''}.npz",
             **{k.replace("|", "__"): v for k, v in P.items()},
             **{f"dhat__{c}": v for c, v in D_hat.items()},
             **{f"off__{c}": v for c, v in conds.items()})
    print(f"\nconfound: {res_conf}")
    for mode, v in recal.items():
        print(f"recalibrated ({mode}): " + "  ".join(
            f"{c} {100 * r['sen']:.0f}/{100 * r['spe']:.0f}" for c, r in v.items()))
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
