#!/usr/bin/env python
"""E52 -- one DO-PINet, two deployment modes; and E51 repeated over seeds.

    python experiments/E52_dual_mode/run.py --smoke
    python experiments/E52_dual_mode/run.py --seeds 3

E51 found a trade-off between two variants of the same architecture:
    do_pinet   timing enters only through the operator's measured d_hat: the model reads
               the interval (antisymmetric response), AUROC 0.936, fragile to offsets
    align      timing removed (PCG aligned, no d_hat): robust to offsets, AUROC 0.934
Which one to deploy depends on whether the acquisition is hardware-synchronised. This
asks whether ONE set of weights can serve both: the timing input gets an on/off flag,
and during training the timing is switched off for a random half of the samples
(features zeroed, flag 0). At deployment the flag is set by the acquisition mode.

    dual:on    synchronised device -- measured d_hat used
    dual:off   unsynchronised device -- timing ignored, aligned morphology only

Success = dual:on close to do_pinet and dual:off close to align, on clean data and
under offsets. pinet / align / do_pinet are retrained in the same run, so E51 is
repeated over seeds; every number is reported as mean +- SD across seeds.
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
BASE_SEED = 20261004
P_OFF = 0.5


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E51 = _load("e51_run", "E51_do_pinet/run.py")
E50 = E51.E50
MARGIN, FIXED_MS, RAND_MS, CLIP_MS = E50.MARGIN, E50.FIXED_MS, E50.RAND_MS, E50.CLIP_MS
E13 = E51.E13
# name -> (align, timing mode)   timing: None | "on" | "dual"
NETS = {"pinet": (False, None), "align": (True, None), "do_pinet": (True, "on"),
        "dual": (True, "dual")}
EVAL = [("pinet", "pinet", None), ("align", "align", None), ("do_pinet", "do_pinet", 1.0),
        ("dual:on", "dual", 1.0), ("dual:off", "dual", 0.0)]


class Net(PINet):
    """PINet + optional (d_hat, sigma, flag) timing input. flag = 0 means 'no timing'."""

    def __init__(self, timing: bool, dim: int = 128, dropout: float = 0.3):
        super().__init__(dim=dim, dropout=dropout)
        self.use_t = timing
        if timing:
            self.timing = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 2 * dim))

    def forward(self, x, t=None):
        B, S = x.shape[:2]
        f = self.encoder(x.reshape(B * S, *x.shape[2:])).reshape(B, S, -1)
        if self.use_t:
            f = f + self.timing(t)
        f = f + self.site_emb(torch.arange(S, device=x.device))[None]
        w = torch.softmax(self.site_gate, dim=0)[None, :, None]
        pooled = self.trunk((f * w).sum(1))
        return {"logits": self.head_cls(pooled)}


def with_flag(T2, flag):
    """(N,4,2) standardised features -> (N,4,3); flag 0 zeroes the features too."""
    T3 = np.concatenate([T2 * flag, np.full(T2.shape[:2] + (1,), flag, np.float32)], -1)
    return T3.astype(np.float32)


def train(X, y, tr, T2, align, mode, cfg, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    cg = torch.Generator().manual_seed(seed)
    m = Net(mode is not None, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(cfg["epochs"] * int(np.ceil(len(tr) / cfg["bs"])), 1),
        pct_start=0.25)
    yt = torch.from_numpy(y).long()
    wc = torch.tensor([1.0, float((yt[tr] == 0).sum()) / max(int((yt[tr] == 1).sum()), 1)],
                      device=device)
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    aug = dict(cfg, crop=1.0)
    dummy = torch.zeros(1)
    zero_off = np.zeros((len(y), 4), int)
    for ep in range(cfg["epochs"]):
        m.train()
        perm = np.random.default_rng(seed + ep).permutation(tr)
        for b in range(0, len(perm), cfg["bs"]):
            jj = perm[b:b + cfg["bs"]]
            o = MARGIN + int(rng.integers(0, T - n + 1))
            off = zero_off[jj] - (align[jj] if align is not None else 0)
            xb = E13.augment(torch.from_numpy(E51.windows(X, jj, o, n, off)).to(device), aug, g, cg)
            tf = None
            if mode is not None:
                flag = np.ones((len(jj), 1, 1), np.float32)
                if mode == "dual":
                    flag = (rng.random((len(jj), 1, 1)) >= P_OFF).astype(np.float32)
                tf = torch.from_numpy(with_flag(T2[jj], np.broadcast_to(flag, (len(jj), 4, 1)))).to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(m(xb, tf), yt[jj].to(device), dummy, dummy,
                              torch.zeros(1, dtype=torch.bool), wc, 0.0, 0.0)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    return m


@torch.no_grad()
def predict(m, X, te, offs, align, T2, flag, cfg, device):
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    P = np.zeros(len(te))
    for b in range(0, len(te), cfg["bs"]):
        jj = te[b:b + cfg["bs"]]
        sl = slice(b, b + len(jj))
        off = offs[sl] - (align[sl] if align is not None else 0)
        tf = (torch.from_numpy(with_flag(T2[sl], np.full((len(jj), 4, 1), flag, np.float32))).to(device)
              if flag is not None else None)
        acc = 0.0
        for t in range(cfg["tta"]):
            o = MARGIN + int((T - n) * t / max(cfg["tta"] - 1, 1))
            xb = torch.from_numpy(E51.windows(X, jj, o, n, off)).to(device)
            acc = acc + torch.softmax(m(xb, tf)["logits"].float(), -1)[:, 1]
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
    rng = np.random.default_rng(BASE_SEED)
    conds = {"none": np.zeros((len(y), 4), int)}
    for s in FIXED_MS:
        conds[f"fixed{s:+d}"] = np.full((len(y), 4), s, int)
    conds["random"] = rng.integers(-RAND_MS, RAND_MS + 1, size=(len(y), 4))
    folds = [0] if args.smoke else range(5)
    seeds = [BASE_SEED + 101 * i for i in range(args.seeds)]
    P = {(s, e, c): np.full(len(y), np.nan) for s in seeds for e, _, _ in EVAL for c in conds}

    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        # The operator is trained once per fold; its seed is not what is being varied.
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
        for s in seeds:
            nets = {nm: train(X, y, tr, T2, A if al else None, mode, cfg, s, device)
                    for nm, (al, mode) in NETS.items()}
            for c, off in conds.items():
                dh, sg = meas[c]
                Tte, Ate = feat(dh, sg), np.round(dh).astype(int)
                for e, nm, flag in EVAL:
                    al = NETS[nm][0]
                    P[(s, e, c)][te] = predict(nets[nm], X, te, off[te], Ate if al else None,
                                               Tte, flag, cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    k_ = ~np.isnan(P[(seeds[0], "pinet", "none")])
    yy = y[k_]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    per_seed = {}
    for s in seeds:
        for e, _, _ in EVAL:
            pc = P[(s, e, "none")][k_]
            thr = np.sort(pc[yy == 1])[::-1][int(np.ceil(0.9 * yy.sum())) - 1]
            for c in conds:
                p = P[(s, e, c)][k_]
                per_seed[(s, e, c)] = {"auroc": roc_auc_score(yy, p),
                                       "sen": (p[yy == 1] >= thr).mean(),
                                       "spe": (p[yy == 0] < thr).mean(),
                                       "dlogit": (lg(p) - lg(pc)).mean()}
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "p_off": P_OFF, "eval": {}}
    print(f"\n{len(seeds)} seed(s); mean +- SD across seeds; threshold frozen per seed on clean OOF")
    for e, _, _ in EVAL:
        res["eval"][e] = {}
        print(f"\n=== {e} ===")
        for c in conds:
            v = {m_: [per_seed[(s, e, c)][m_] for s in seeds] for m_ in ("auroc", "sen", "spe", "dlogit")}
            r = {m_: [round(float(np.mean(x)), 4), round(float(np.std(x)), 4)] for m_, x in v.items()}
            res["eval"][e][c] = r
            print(f"  {c:9s} AUROC {r['auroc'][0]:.4f}+-{r['auroc'][1]:.4f}  sen {100 * r['sen'][0]:5.1f}+-"
                  f"{100 * r['sen'][1]:.1f}  spe {100 * r['spe'][0]:5.1f}+-{100 * r['spe'][1]:.1f}  "
                  f"dlogit {r['dlogit'][0]:+.2f}+-{r['dlogit'][1]:.2f}", flush=True)
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e52_dual{tag}.json").write_text(
        json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2), encoding="utf-8")
    np.savez(OUT / f"e52_oof{tag}.npz", **{f"{s}__{e}__{c}": p for (s, e, c), p in P.items()})
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
