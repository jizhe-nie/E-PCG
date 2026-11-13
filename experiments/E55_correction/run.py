#!/usr/bin/env python
"""E55 -- better offset correction: where the 50 ms tolerance band loses sensitivity, and
what replaces a hand-set threshold.

    python experiments/E55_correction/run.py --smoke
    python experiments/E55_correction/run.py --seeds 3

WHY. In E50 the symmetric 50 ms tolerance band cost sensitivity (random +-100 ms offsets:
91.7% -> 86.7%). The reason is measurable: on clean input the operator's offset estimate
d_hat is roughly symmetric in non-HFrEF sites (5-95%: -44..+43 ms) but has a long right
tail in HFrEF (95th percentile +118 ms; 27% of HFrEF sites exceed +50 ms). Under random
offsets the band therefore "corrects" HFrEF sites by +17 ms on average while their true
device offset averages -2 ms -- it removes about 19 ms of genuine pathological prolongation.

PRE-SPECIFIED BEFORE RUNNING (do not edit after results exist)
  primary method    BAYES+ANCHOR: correction = posterior mean of the device offset d given
                    the anchored residual r = d_hat - a, where a is the subject's expected
                    interval predicted from offset-invariant information, the residual's
                    pathological part q has the empirical distribution of training-fold
                    residuals (cross-fitted), and the device prior is d ~ N(0, 60 ms).
                    60 ms is the order of typical wireless-audio latency spread; it is a
                    deployment assumption, stated, not tuned.
  primary criterion under per-site U(-100, 100) ms offsets, paired differences in sensitivity
                    and specificity (primary minus symmetric 50 ms band), threshold frozen
                    on each method's own clean out-of-fold scores at 90% sensitivity.
  sensitivity analyses (reported, never used to choose):
                    raw, full correction, symmetric band D in {20,30,40,50,60,80,100},
                    asymmetric band (D- = |2.5th pct|, D+ = 97.5th pct of training clean
                    d_hat, data-derived), BAYES without anchor.

THE ANCHOR. A late-fusion regressor that pools ECG and PCG features SEPARATELY before
combining them, so a relative shift of the two streams leaves its input statistics
unchanged; it predicts the clean d_hat of a site from what each modality shows on its own
(rate, QRS, sound intensity, S2, ...). Pathology that also changes those quantities is
thereby explained, and only the part of d_hat they cannot account for is attributed to the
device. It is trained per outer fold on training subjects; its training residuals are
cross-fitted (3 inner folds) so that q is not optimistically narrow.

Everything else follows E50: operator trained per fold on training subjects (R->S1),
mu_site and slope k calibrated on training subjects, PINet trained per seed, corrections
applied per site, offsets fixed -80/-40/+40/+80 ms or U(-100, 100) per site.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcg.models.emat_arms import Trunk  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
SEED = 20261004
SIGMA_D = 60.0
BANDS = [20, 30, 40, 50, 60, 80, 100]
PRIMARY = "bayes_anchor"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E50 = _load("e50_run", "E50_resync/run.py")
E49 = E50.E49
MARGIN, CLIP_MS = E50.MARGIN, E50.CLIP_MS


# ------------------------------------------------------------------ anchor (offset-invariant)
class LateAnchor(nn.Module):
    """ECG and PCG pooled separately, then combined: invariant to their relative shift."""

    def __init__(self, dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.e, self.p = Trunk(dim, dropout), Trunk(dim, dropout)
        self.head = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x):
        return self.head(torch.cat([self.e.pooled(x[:, 0]), self.p.pooled(x[:, 1])], -1)).squeeze(-1)


def fit_anchor(W, t, epochs, device, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = LateAnchor().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    bs = 64
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, pct_start=0.25,
                                              total_steps=max(epochs * int(np.ceil(len(t) / bs)), 1))
    mu, sd = float(t.mean()), float(t.std() + 1e-6)
    Wt, tt = torch.from_numpy(W), torch.from_numpy(t.astype(np.float32))
    for _ in range(epochs):
        m.train()
        perm = rng.permutation(len(t))
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss((m(Wt[j].to(device)) - mu) / sd, (tt[j].to(device) - mu) / sd)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    return m


@torch.no_grad()
def predict_anchor(m, W, device, bs=128):
    return np.concatenate([m(torch.from_numpy(W[b:b + bs]).to(device)).float().cpu().numpy()
                           for b in range(0, len(W), bs)])


# ------------------------------------------------------------------ corrections
def band(d, lo, hi):
    """Correct only the part of d outside [-lo, +hi]."""
    return np.where(d > hi, d - hi, np.where(d < -lo, d + lo, 0.0))


class Posterior:
    """E[d | r] for r = q + d, q ~ KDE(training residuals), d ~ N(0, SIGMA_D)."""

    def __init__(self, q):
        q = np.asarray(q, float)
        bw = 1.06 * q.std() * len(q) ** (-1 / 5)
        self.u = np.arange(-700.0, 700.5, 1.0)
        dens = np.zeros_like(self.u)
        for b in range(0, len(q), 512):
            dens += np.exp(-0.5 * ((self.u[:, None] - q[None, b:b + 512]) / bw) ** 2).sum(1)
        self.f = dens / dens.sum()
        self.d = np.arange(-300.0, 300.5, 1.0)
        self.prior = np.exp(-0.5 * (self.d / SIGMA_D) ** 2)

    def __call__(self, r):
        r = np.asarray(r, float)
        like = np.interp(r.ravel()[:, None] - self.d[None, :], self.u, self.f, left=0, right=0)
        w = like * self.prior[None, :] + 1e-300
        return ((w * self.d).sum(1) / w.sum(1)).reshape(r.shape)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--op-epochs", type=int, default=40)
    ap.add_argument("--anchor-epochs", type=int, default=25)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.op_epochs, args.anchor_epochs, args.seeds = 2, 2, 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    LV = (d["lvef"] * 100).astype(np.float32)
    STI = np.nan_to_num(d["sti"].astype(np.float32))
    MSK = d["sti_mask"].astype(bool)
    Y = E50.emat_targets(subs)
    rng = np.random.default_rng(E50.SEED)            # identical offsets to E50
    conds = {"none": np.zeros((len(y), 4), int)}
    for s in E50.FIXED_MS:
        conds[f"fixed{s:+d}"] = np.full((len(y), 4), s, int)
    conds["random"] = rng.integers(-E50.RAND_MS, E50.RAND_MS + 1, size=(len(y), 4))
    methods = ["raw", "full"] + [f"band{b}" for b in BANDS] + ["asym", "bayes", PRIMARY]
    folds = [0] if args.smoke else range(5)
    seeds = [SEED + 101 * i for i in range(args.seeds)]
    P = {(s, m, c): np.full(len(y), np.nan) for s in seeds for m in methods for c in conds}
    CORR = {(m, c): np.full((len(y), 4), np.nan) for m in methods for c in conds}
    DH = {c: np.full((len(y), 4), np.nan) for c in conds}
    meta = []

    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        op = E50.train_op(X, Y, tr, args, device)
        mu, k = E50.calibrate(op, X, tr, device)

        def measure(ii, off):
            idx = [(i, j) for i in ii for j in range(4)]
            tau, _ = E50.op_tau(op, X, idx, off.reshape(-1), device)
            W = E50.op_windows(X, idx, off.reshape(-1))
            return np.clip((tau.reshape(-1, 4) - mu) / k, -CLIP_MS, CLIP_MS), W

        dh_tr, W_tr = measure(tr, np.zeros((len(tr), 4), int))
        t_tr = dh_tr.reshape(-1)
        # Cross-fitted anchor residuals on the training fold, then a final anchor.
        q_res = np.zeros_like(t_tr)
        groups = np.repeat(np.arange(len(tr)), 4)
        for a_, b_ in KFold(3, shuffle=True, random_state=SEED).split(np.arange(len(tr))):
            ia, ib = np.isin(groups, a_), np.isin(groups, b_)
            am = fit_anchor(W_tr[ia], t_tr[ia], args.anchor_epochs, device)
            q_res[ib] = t_tr[ib] - predict_anchor(am, W_tr[ib], device)
        anchor = fit_anchor(W_tr, t_tr, args.anchor_epochs, device)
        post_anchor, post_plain = Posterior(q_res), Posterior(t_tr)
        lo, hi = -np.percentile(t_tr, 2.5), np.percentile(t_tr, 97.5)
        meta.append({"fold": f, "k": round(k, 3), "asym_lo": round(float(lo), 1),
                     "asym_hi": round(float(hi), 1), "train_dhat_sd": round(float(t_tr.std()), 1),
                     "anchor_resid_sd": round(float(q_res.std()), 1),
                     "anchor_r2": round(float(1 - q_res.var() / t_tr.var()), 3)})
        print(f"  fold {f}: anchor explains {100 * meta[-1]['anchor_r2']:.0f}% of clean d_hat variance; "
              f"asym band [-{lo:.0f}, +{hi:.0f}] ms", flush=True)

        for c, off in conds.items():
            o = off[te]
            dh, W = measure(te, o)
            DH[c][te] = dh
            a = predict_anchor(anchor, W, device).reshape(-1, 4)
            corr = {"raw": np.zeros_like(dh), "full": dh,
                    "asym": band(dh, lo, hi), "bayes": post_plain(dh),
                    PRIMARY: post_anchor(dh - a)}
            for b in BANDS:
                corr[f"band{b}"] = band(dh, b, b)
            for m in methods:
                CORR[(m, c)][te] = np.clip(corr[m], -CLIP_MS, CLIP_MS)
        for s in seeds:
            net = train_seed(X, y, LV, STI, MSK, tr, cfg, s, device)
            for c, off in conds.items():
                for m in methods:
                    P[(s, m, c)][te] = E50.pinet_prob(net, X, te, off[te] - np.round(CORR[(m, c)][te]).astype(int),
                                                      cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- evaluation
    k_ = ~np.isnan(P[(seeds[0], "raw", "none")])
    yy = y[k_]
    nthr = int(np.ceil(0.9 * yy.sum())) - 1
    thr_of = lambda p: np.sort(p[yy == 1])[::-1][nthr]
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "sigma_d_ms": SIGMA_D,
           "primary": PRIMARY, "fold_meta": meta, "methods": {}}
    for m in methods:
        res["methods"][m] = {}
        for c in conds:
            rows = []
            for s in seeds:
                pc, p = P[(s, m, "none")][k_], P[(s, m, c)][k_]
                pr, t_raw = P[(s, "raw", "none")][k_], None
                t_own, t_raw = thr_of(pc), thr_of(pr)
                rows.append([roc_auc_score(yy, p), (p[yy == 1] >= t_own).mean(), (p[yy == 0] < t_own).mean(),
                             (p[yy == 1] >= t_raw).mean(), (p[yy == 0] < t_raw).mean()])
            rows = np.array(rows)
            res["methods"][m][c] = {key: [round(float(rows[:, i].mean()), 4), round(float(rows[:, i].std()), 4)]
                                    for i, key in enumerate(("auroc", "sen", "spe", "sen_rawthr", "spe_rawthr"))}
        dcl = DH["none"][k_]; cr = CORR[(m, "random")][k_]
        hf = yy == 1
        res["methods"][m]["pathology_removed_hfref_ms"] = round(float((cr[hf] - (DH["random"][k_][hf] - dcl[hf])).mean()), 2) if m != "raw" else 0.0

    # primary criterion: seed-mean paired difference (same estimand as the tables), random condition
    PS = _load("pairstats", "_pairstats.py")
    Pk = {kk: v[k_] for kk, v in P.items()}
    metric = PS.own_threshold_metric(Pk, yy, "none")
    pr = PS.paired(metric, (PRIMARY, "random"), ("band50", "random"), seeds, len(yy), PS.boots(yy, seed=SEED))
    res["primary_criterion"] = {"condition": "random U(-100,100) per site", "estimand": pr["estimand"],
                                "sen_primary_minus_band50": pr["sen"], "sen_ci": pr["sen_ci"],
                                "spe_primary_minus_band50": pr["spe"], "spe_ci": pr["spe_ci"]}
    np.savez_compressed(OUT / "e55_scores.npz", y=yy, **{f"{s}__{m}__{c}": v for (s, m, c), v in Pk.items()})

    print(f"\n{len(seeds)} seed(s); own-threshold sen/spe (%), mean over seeds")
    hdr = "  ".join(f"{c:>13s}" for c in conds)
    print(f"{'method':14s} {hdr}   HFrEF pathology removed")
    for m in methods:
        cells = "  ".join(f"{100 * res['methods'][m][c]['sen'][0]:5.1f}/{100 * res['methods'][m][c]['spe'][0]:5.1f}"
                          for c in conds)
        print(f"{m:14s} {cells}   {res['methods'][m]['pathology_removed_hfref_ms']:+.1f} ms", flush=True)
    pcr = res["primary_criterion"]
    print(f"\nPRIMARY ({PRIMARY}) minus band50, random offsets: sen {100 * pcr['sen_primary_minus_band50']:+.1f} "
          f"[{100 * pcr['sen_ci'][0]:+.1f}, {100 * pcr['sen_ci'][1]:+.1f}]  spe {100 * pcr['spe_primary_minus_band50']:+.1f} "
          f"[{100 * pcr['spe_ci'][0]:+.1f}, {100 * pcr['spe_ci'][1]:+.1f}]")
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e55_correction{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)},
                                                              indent=2), encoding="utf-8")
    np.savez(OUT / f"e55_corr{tag}.npz", **{f"{m}__{c}": v for (m, c), v in CORR.items()},
             **{f"dhat__{c}": v for c, v in DH.items()}, **{f"off__{c}": v for c, v in conds.items()})
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


def train_seed(X, y, LV, STI, MSK, tr, cfg, seed, device):
    """E49.train with a different seed (E49 fixes SEED at module level)."""
    old = E49.SEED
    E49.SEED = seed
    try:
        net, _, _ = E49.train(X, y, LV, STI, MSK, tr, cfg, 0.0, 0.0, device)
    finally:
        E49.SEED = old
    return net


if __name__ == "__main__":
    raise SystemExit(main())
