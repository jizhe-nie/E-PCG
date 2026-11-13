#!/usr/bin/env python
"""E51 -- DO-PINet: the delay operator inside the classifier, as synchroniser and as
measurement layer.

    python experiments/E51_do_pinet/run.py --smoke
    python experiments/E51_do_pinet/run.py

WHY THIS DESIGN. E31 put a frozen operator in place of the convolutional path and
reached AUROC 0.663 -- the operator cannot replace the network. E50 found two things the
operator CAN do: its offset estimate on clean input is itself a disease marker (AUROC
0.81 alone; HFrEF +30 ms vs -1 ms), and it can align the PCG to the ECG. So here it does
both, alongside PINet's convolutional path rather than instead of it:

  1. SYNCHRONISER. Per site, d_hat = (tau - mu_site) / k (E50 calibration, training folds
     only). The PCG is read d_hat later in the buffer -- shifted back to the site's
     canonical R->S1 -- before PINet's convolutions. The CNN then sees morphology at a
     fixed phase, independent of any device offset.
  2. MEASUREMENT LAYER. Aligning removes the timing from the waveform, so it is given
     back explicitly: (d_hat, sigma), standardised on the training fold, enter each
     site's feature vector through a small MLP. Timing now reaches the decision as one
     readable number per site instead of implicitly through the waveform.

PREDICTION THIS MAKES. PINet and LGC2-Net respond SYMMETRICALLY to desynchronisation
(earlier and later both push toward HFrEF, E46/E49). A model whose only route to timing
is the measured d_hat should respond ANTISYMMETRICALLY: PCG later -> longer measured
interval -> more HFrEF; PCG earlier -> less. That is what "reads the interval" means
for a classifier.

Four variants, one training recipe (E44 CFG), authors' five folds, operator trained per
fold on training subjects only:
    pinet         baseline
    feat          + (d_hat, sigma) features, no alignment
    align         alignment, no features
    do_pinet      alignment + features
Evaluation as E50: frozen 90%-sensitivity threshold from each variant's own clean OOF,
offsets fixed -80/-40/+40/+80 and random U(-100,100) per site, signed logit change.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcg.models.pinet import PINet, loss_fn  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
SEED = 20261003
VARIANTS = {"pinet": (False, False), "feat": (False, True),
            "align": (True, False), "do_pinet": (True, True)}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E50 = _load("e50_run", "E50_resync/run.py")
MARGIN, FIXED_MS, RAND_MS, CLIP_MS = E50.MARGIN, E50.FIXED_MS, E50.RAND_MS, E50.CLIP_MS
E13 = E50.E49.E13


class DOPINet(PINet):
    """PINet whose site features also receive the operator's (d_hat, sigma)."""

    def __init__(self, use_timing: bool, dim: int = 128, dropout: float = 0.3):
        super().__init__(dim=dim, dropout=dropout)
        self.use_timing = use_timing
        if use_timing:
            self.timing = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 2 * dim))

    def forward(self, x, t=None):
        B, S = x.shape[:2]
        f = self.encoder(x.reshape(B * S, *x.shape[2:])).reshape(B, S, -1)
        if self.use_timing:
            f = f + self.timing(t)
        f = f + self.site_emb(torch.arange(S, device=x.device))[None]
        w = torch.softmax(self.site_gate, dim=0)[None, :, None]
        pooled = self.trunk((f * w).sum(1))
        return {"logits": self.head_cls(pooled), "lvef": self.head_lvef(pooled).squeeze(-1),
                "sti": self.head_sti(f)}


def windows(X, ii, o, n, pcg_off):
    """(len(ii), 4, 2, n): ECG from o, PCG from o - pcg_off[a, j] (samples)."""
    w = np.empty((len(ii), 4, 2, n), np.float32)
    for a, i in enumerate(ii):
        for j in range(4):
            po = int(pcg_off[a, j])
            w[a, j, 0] = X[i, j, 0, o:o + n]
            w[a, j, 1] = X[i, j, 1, o - po:o - po + n]
    return w


def train(X, y, tr, offs, T_feat, align, use_t, cfg, device):
    """offs: (N,4) PCG offset already present in the data (0 for clean training);
    align: subtract round(d_hat) from it; use_t: feed T_feat (N,4,2) standardised."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    g = torch.Generator(device=device).manual_seed(SEED)
    cg = torch.Generator().manual_seed(SEED)
    m = DOPINet(use_t, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(cfg["epochs"] * int(np.ceil(len(tr) / cfg["bs"])), 1),
        pct_start=0.25)
    yt = torch.from_numpy(y).long()
    wc = torch.tensor([1.0, float((yt[tr] == 0).sum()) / max(int((yt[tr] == 1).sum()), 1)],
                      device=device)
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    aug_cfg = dict(cfg, crop=1.0)
    dummy = torch.zeros(1)
    for ep in range(cfg["epochs"]):
        m.train()
        perm = np.random.default_rng(SEED + ep).permutation(tr)
        for b in range(0, len(perm), cfg["bs"]):
            jj = perm[b:b + cfg["bs"]]
            o = MARGIN + int(rng.integers(0, T - n + 1))
            off = offs[jj] - (align[jj] if align is not None else 0)
            xb = torch.from_numpy(windows(X, jj, o, n, off)).to(device)
            xb = E13.augment(xb, aug_cfg, g, cg)
            tf = torch.from_numpy(T_feat[jj]).to(device) if use_t else None
            opt.zero_grad(set_to_none=True)
            out = m(xb, tf)
            loss, _ = loss_fn(out, yt[jj].to(device), dummy, dummy, torch.zeros(1, dtype=torch.bool),
                              wc, 0.0, 0.0)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    return m


@torch.no_grad()
def predict(m, X, te, offs, align, T_feat, use_t, cfg, device):
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    P = np.zeros(len(te))
    for b in range(0, len(te), cfg["bs"]):
        jj = te[b:b + cfg["bs"]]
        sl = slice(b, b + len(jj))
        off = offs[sl] - (align[sl] if align is not None else 0)
        tf = torch.from_numpy(T_feat[sl]).to(device) if use_t else None
        acc = 0.0
        for t in range(cfg["tta"]):
            o = MARGIN + int((T - n) * t / max(cfg["tta"] - 1, 1))
            xb = torch.from_numpy(windows(X, jj, o, n, off)).to(device)
            acc = acc + torch.softmax(m(xb, tf)["logits"].float(), -1)[:, 1]
        P[sl] = (acc / cfg["tta"]).cpu().numpy()
    return P


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op-epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E50.E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.op_epochs = 2, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E50.E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    Y = E50.emat_targets(subs)
    rng = np.random.default_rng(SEED)
    conds = {"none": np.zeros((len(y), 4), int)}
    for s in FIXED_MS:
        conds[f"fixed{s:+d}"] = np.full((len(y), 4), s, int)
    conds["random"] = rng.integers(-RAND_MS, RAND_MS + 1, size=(len(y), 4))
    folds = [0] if args.smoke else range(5)
    P = {(v, c): np.full(len(y), np.nan) for v in VARIANTS for c in conds}
    DH = np.full((len(y), 4), np.nan)

    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        op = E50.train_op(X, Y, tr, args, device)
        mu, k = E50.calibrate(op, X, tr, device)

        def measure(ii, off):
            idx = [(i, j) for i in ii for j in range(4)]
            tau, sg = E50.op_tau(op, X, idx, off.reshape(-1), device)
            dh = np.clip((tau.reshape(-1, 4) - mu[None]) / k, -CLIP_MS, CLIP_MS)
            return dh, sg.reshape(-1, 4)

        dh_tr, sg_tr = measure(tr, np.zeros((len(tr), 4), int))
        # Standardisation of the timing features from the training fold only.
        fm = np.array([dh_tr.mean(), sg_tr.mean()]); fs = np.array([dh_tr.std(), sg_tr.std()]) + 1e-6
        feat = lambda dh, sg: ((np.stack([dh, sg], -1) - fm) / fs).astype(np.float32)
        Ttr = np.zeros((len(y), 4, 2), np.float32); Ttr[tr] = feat(dh_tr, sg_tr)
        Atr = np.zeros((len(y), 4), int); Atr[tr] = np.round(dh_tr).astype(int)
        nets = {v: train(X, y, tr, np.zeros((len(y), 4), int), Ttr,
                         Atr if al else None, ut, cfg, device)
                for v, (al, ut) in VARIANTS.items()}
        for c, off in conds.items():
            o = off[te]
            dh, sg = measure(te, o)
            if c == "none":
                DH[te] = dh
            Tte, Ate = feat(dh, sg), np.round(dh).astype(int)
            for v, (al, ut) in VARIANTS.items():
                P[(v, c)][te] = predict(nets[v], X, te, o, Ate if al else None, Tte, ut, cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    k_ = ~np.isnan(P[("pinet", "none")])
    yy = y[k_]
    rb = np.random.default_rng(SEED)
    bs = [b for b in (rb.integers(0, k_.sum(), k_.sum()) for _ in range(2000)) if 0 < yy[b].sum() < len(b)]
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "variants": {}}
    for v in VARIANTS:
        pc = P[(v, "none")][k_]
        thr = np.sort(pc[yy == 1])[::-1][int(np.ceil(0.9 * yy.sum())) - 1]
        rv = {}
        for c in conds:
            p = P[(v, c)][k_]
            dl = logit(p) - logit(pc)
            rv[c] = {"auroc": round(float(roc_auc_score(yy, p)), 4),
                     "sen": round(float((p[yy == 1] >= thr).mean()), 4),
                     "spe": round(float((p[yy == 0] < thr).mean()), 4),
                     "spe_ci": [round(float(np.percentile([(p[b][yy[b] == 0] < thr).mean() for b in bs], q)), 4)
                                for q in (2.5, 97.5)],
                     "dlogit": round(float(dl.mean()), 3),
                     "dlogit_pos": round(float(dl[yy == 1].mean()), 3),
                     "dlogit_neg": round(float(dl[yy == 0].mean()), 3)}
        a80, m80 = rv["fixed+80"]["dlogit"], rv["fixed-80"]["dlogit"]
        rv["antisymmetric_80"] = bool(np.sign(a80) != np.sign(m80))
        res["variants"][v] = rv
        print(f"\n=== {v} ===  (+80 vs -80 antisymmetric: {rv['antisymmetric_80']})")
        for c, r in rv.items():
            if c == "antisymmetric_80":
                continue
            print(f"  {c:9s} AUROC {r['auroc']:.4f}  sen {100 * r['sen']:5.1f}  spe {100 * r['spe']:5.1f} "
                  f"[{100 * r['spe_ci'][0]:.1f}, {100 * r['spe_ci'][1]:.1f}]  dlogit {r['dlogit']:+.2f} "
                  f"(pos {r['dlogit_pos']:+.2f}, neg {r['dlogit_neg']:+.2f})", flush=True)
    res["clean_dhat_auroc"] = round(float(roc_auc_score(yy, DH[k_].mean(1))), 4)
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e51_do_pinet{tag}.json").write_text(
        json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2), encoding="utf-8")
    np.savez(OUT / f"e51_oof{tag}.npz", **{f"{v}__{c}": p for (v, c), p in P.items()}, dhat_clean=DH)
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
