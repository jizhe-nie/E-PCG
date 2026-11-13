#!/usr/bin/env python
"""E77 -- does the OFF mode survive SECONDS-scale clock offsets, and does a late-fusion head that is invariant
by construction (the unimodal-branch design of HPA-Net) close the gap?

    python experiments/E77_seconds_offset/run.py --smoke
    python experiments/E77_seconds_offset/run.py --seeds 3

WHY. E68 found that real independent clocks differ by SECONDS. The dual-mode OFF branch is invariant only
because the delay operator first re-aligns the PCG by its estimate d_hat, and the operator searches lags in
[-200, 300] ms with d_hat clipped at +-150 ms. Beyond that range the alignment cannot work, so OFF has never
been tested where independent clocks actually live.

ARMS (same five folds, same recipe as E52, 3 seeds; operator trained once per fold as in E52)
  pinet      conv baseline (ECG and PCG stacked from the first convolution)
  dual:on    dual-mode DO-PINet, timing flag 1
  dual:off   dual-mode DO-PINet, timing flag 0 (operator-aligned morphology)
  late       late-fusion PINet: ECG and PCG get separate site encoders (in_ch 1, dim/2 each), each pooled over
             time (mean + max) BEFORE concatenation; site embedding, site gate, trunk and head as PINet. A
             relative ECG-PCG shift cannot reach any representation, so it is invariant by construction
             (up to which samples fall inside the crop). Trained on unaligned clean data, no operator.
CONDITIONS  PCG offset (positive = PCG later), the same for all four sites unless stated:
  none; +-80 ms (E52 range, for continuity); +-500, +-1000, +-2000, +-3000 ms; random per site U(-3000, 3000) ms
CROPS  three test-time crops of E52 length (0.7 x 29.4 s); positions as in E52, clipped per subject into the
  range where both the ECG and the offset PCG crop lie inside the 30 s buffer. For |offset| <= 300 ms this
  is exactly E52's placement.
THRESHOLD  frozen per seed at 90% sensitivity on each arm's own clean scores (as everywhere in the paper);
  paired differences by experiments/_pairstats.py (seed-mean of per-seed differences, subject bootstrap).

PRE-SPECIFIED CRITERIA (reported either way)
  C1 late is offset-invariant: for every condition, specificity(cond) - specificity(none) >= -2 points
     (point estimate of the paired difference)
  C2 late keeps clean accuracy: clean AUROC >= 0.92 (seed mean)
  C3 hypothesis about OFF (the reason for the experiment): dual:off loses >= 10 points of specificity at
     |offset| >= 1000 ms (point estimate, at least one sign); if it does NOT, OFF is reported as robust to
     seconds-scale offsets and the late head is not needed for that purpose
  descriptive: fraction of d_hat at the +-150 ms clip per condition; dual:on and pinet under every condition
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from epcg.models.pinet import PINet, SiteEncoder  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
sys.path.insert(0, str(EXP))
from _pairstats import boots, own_threshold_metric, paired  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E52 = _load("e52_run", "E52_dual_mode/run.py")
E50 = E52.E50
MARGIN, CLIP_MS = E52.MARGIN, E50.CLIP_MS
BASE_SEED = 20261019
SECONDS_MS = [500, 1000, 2000, 3000]


class LateNet(PINet):
    """Late-fusion PINet: separate ECG / PCG site encoders, pooled over time before concatenation."""

    def __init__(self, timing: bool = False, dim: int = 128, dropout: float = 0.3):
        super().__init__(dim=dim, dropout=dropout)
        assert not timing
        self.encoder = None
        self.enc_e = SiteEncoder(dim // 2, in_ch=1, dropout=dropout)
        self.enc_p = SiteEncoder(dim // 2, in_ch=1, dropout=dropout)

    def forward(self, x, t=None):
        B, S = x.shape[:2]
        z = x.reshape(B * S, *x.shape[2:])
        f = torch.cat([self.enc_e(z[:, :1]), self.enc_p(z[:, 1:])], -1).reshape(B, S, -1)
        f = f + self.site_emb(torch.arange(S, device=x.device))[None]
        w = torch.softmax(self.site_gate, dim=0)[None, :, None]
        return {"logits": self.head_cls(self.trunk((f * w).sum(1)))}


def train_late(X, y, tr, T2, cfg, seed, device):
    """E52's training loop unchanged, with LateNet in place of E52.Net (no timing, no alignment)."""
    net = E52.Net
    E52.Net = LateNet
    try:
        return E52.train(X, y, tr, T2, None, None, cfg, seed, device)
    finally:
        E52.Net = net


@torch.no_grad()
def predict(m, X, te, offs, align, T2, flag, cfg, device):
    """E52.predict with crops clipped per subject into the valid range for large offsets."""
    L = X.shape[-1]
    T = L - 2 * MARGIN
    n = int(T * cfg["crop"])
    base = [MARGIN + int((T - n) * t / max(cfg["tta"] - 1, 1)) for t in range(cfg["tta"])]
    P = np.zeros(len(te))
    for b in range(0, len(te), cfg["bs"]):
        jj = te[b:b + cfg["bs"]]
        sl = slice(b, b + len(jj))
        off = offs[sl] - (align[sl] if align is not None else 0)
        tf = (torch.from_numpy(E52.with_flag(T2[sl], np.full((len(jj), 4, 1), flag, np.float32))).to(device)
              if flag is not None else None)
        acc = 0.0
        for t in range(cfg["tta"]):
            w = np.empty((len(jj), 4, 2, n), np.float32)
            for a, i in enumerate(jj):
                lo, hi = max(0, int(off[a].max())), min(L, L + int(off[a].min())) - n
                o = int(np.clip(base[t], lo, hi))
                w[a] = E52.E51.windows(X, [i], o, n, off[a:a + 1])[0]
            acc = acc + torch.softmax(m(torch.from_numpy(w).to(device), tf)["logits"].float(), -1)[:, 1]
        P[sl] = (acc / cfg["tta"]).cpu().numpy()
    return P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--op-epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E50.E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.op_epochs, args.seeds = 2, 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E50.E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    Y = E50.emat_targets(subs)
    print(f"late-fusion PINet {sum(p.numel() for p in LateNet().parameters()):,} parameters; "
          f"PINet {sum(p.numel() for p in PINet().parameters()):,}", flush=True)
    rng = np.random.default_rng(BASE_SEED)
    conds = {"none": np.zeros((len(y), 4), int)}
    for s in [-80, 80] + [v for s_ in SECONDS_MS for v in (-s_, s_)]:
        conds[f"fixed{s:+d}"] = np.full((len(y), 4), s, int)
    conds["random3s"] = rng.integers(-3000, 3001, size=(len(y), 4))
    folds = [0] if args.smoke else range(5)
    seeds = [BASE_SEED + 101 * i for i in range(args.seeds)]
    arms = ["pinet", "dual:on", "dual:off", "late"]
    P = {(s, e, c): np.full(len(y), np.nan) for s in seeds for e in arms for c in conds}
    clip_frac = {c: [] for c in conds}
    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        op = E50.train_op(X, Y, tr, args, device)
        mu, k = E50.calibrate(op, X, tr, device)

        def measure(ii, off):
            idx = [(i, j) for i in ii for j in range(4)]
            tau, sg = E50.op_tau(op, X, idx, off.reshape(-1), device)
            return np.clip((tau.reshape(-1, 4) - mu[None]) / k, -CLIP_MS, CLIP_MS), sg.reshape(-1, 4)

        dh_tr, sg_tr = measure(tr, np.zeros((len(tr), 4), int))
        fm = np.array([dh_tr.mean(), sg_tr.mean()]); fs = np.array([dh_tr.std(), sg_tr.std()]) + 1e-6
        feat = lambda dh, sg: ((np.stack([dh, sg], -1) - fm) / fs).astype(np.float32)
        T2 = np.zeros((len(y), 4, 2), np.float32); T2[tr] = feat(dh_tr, sg_tr)
        A = np.zeros((len(y), 4), int); A[tr] = np.round(dh_tr).astype(int)
        meas = {c: measure(te, off[te]) for c, off in conds.items()}
        for c in conds:
            clip_frac[c].append(float((np.abs(meas[c][0]) >= CLIP_MS - 1e-6).mean()))
        for s in seeds:
            nets = {"pinet": E52.train(X, y, tr, T2, None, None, cfg, s, device),
                    "dual": E52.train(X, y, tr, T2, A, "dual", cfg, s, device),
                    "late": train_late(X, y, tr, T2, cfg, s, device)}
            for c, off in conds.items():
                dh, sg = meas[c]
                Tte, Ate = feat(dh, sg), np.round(dh).astype(int)
                P[(s, "pinet", c)][te] = predict(nets["pinet"], X, te, off[te], None, Tte, None, cfg, device)
                P[(s, "dual:on", c)][te] = predict(nets["dual"], X, te, off[te], Ate, Tte, 1.0, cfg, device)
                P[(s, "dual:off", c)][te] = predict(nets["dual"], X, te, off[te], Ate, Tte, 0.0, cfg, device)
                P[(s, "late", c)][te] = predict(nets["late"], X, te, off[te], None, Tte, None, cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    k_ = ~np.isnan(P[(seeds[0], "pinet", "none")])
    yy = y[k_]
    Pk = {key: v[k_] for key, v in P.items()}
    metric = own_threshold_metric(Pk, yy, "none")
    bs = boots(yy, B=200 if args.smoke else 2000, seed=BASE_SEED)
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "conditions": list(conds),
           "dhat_clip_fraction": {c: round(float(np.mean(v)), 4) for c, v in clip_frac.items()},
           "eval": {}, "paired_vs_clean": {}}
    for e in arms:
        res["eval"][e], res["paired_vs_clean"][e] = {}, {}
        for c in conds:
            ss = [metric((e, c), s, np.arange(len(yy))) for s in seeds]
            au = [roc_auc_score(yy, Pk[(s, e, c)]) for s in seeds]
            res["eval"][e][c] = {"auroc": [round(float(np.mean(au)), 4), round(float(np.std(au)), 4)],
                                 "sen": [round(float(np.mean([v[0] for v in ss])), 4), round(float(np.std([v[0] for v in ss])), 4)],
                                 "spe": [round(float(np.mean([v[1] for v in ss])), 4), round(float(np.std([v[1] for v in ss])), 4)]}
            if c != "none":
                res["paired_vs_clean"][e][c] = paired(metric, (e, c), (e, "none"), seeds, len(yy), bs)
        print(f"{e:9s} " + "  ".join(f"{c}:{100 * res['eval'][e][c]['spe'][0]:.1f}" for c in conds), flush=True)
    pv = res["paired_vs_clean"]
    res["C1_late_invariant"] = bool(all(pv["late"][c]["spe"] >= -0.02 for c in conds if c != "none"))
    res["C2_late_clean_auroc"] = bool(res["eval"]["late"]["none"]["auroc"][0] >= 0.92)
    big = [c for c in conds if c.startswith("fixed") and abs(int(c[5:])) >= 1000]
    res["C3_off_fails_at_seconds"] = bool(any(pv["dual:off"][c]["spe"] <= -0.10 for c in big))
    print({k: res[k] for k in ("C1_late_invariant", "C2_late_clean_auroc", "C3_off_fails_at_seconds", "dhat_clip_fraction")})
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e77_seconds{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)},
                                                           indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"e77_scores{tag}.npz", y=yy, **{f"{s}__{e}__{c}": v for (s, e, c), v in Pk.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
