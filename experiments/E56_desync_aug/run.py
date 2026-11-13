#!/usr/bin/env python
"""E56 -- the baseline a reviewer will ask for: train with random desynchronisation.

    python experiments/E56_desync_aug/run.py --smoke
    python experiments/E56_desync_aug/run.py --seeds 3

The machine-learning default for "the input may be misaligned" is augmentation: shift the
PCG relative to the ECG by a random amount during training, so the model learns not to
depend on the alignment. E52 compared four ways of handling desynchronisation
(convolutional baseline, align-only, DO-PINet, dual-mode on/off) but never this one.

  aug100   convolutional baseline (PINet), each training batch gets an independent
           U(-100, 100) ms PCG offset per site -- the same distribution as the evaluation's
           random condition, i.e. the most favourable choice for augmentation
  aug40    the same with U(-40, 40) ms, a milder augmentation that keeps more timing

Expectation stated before running: augmentation buys robustness by teaching the network
to discard relative timing, so it should land near dual:off / align-only -- robust under
offsets, without the antisymmetric interval-reading of dual:on. If it instead matched
dual:on on clean data while staying robust, DO-PINet would have no reason to exist.

Seeds, folds, offsets, the frozen 90%-sensitivity threshold and the evaluation code are
E52's, so the numbers are directly comparable with e52_dual.json.
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

from epcg.models.pinet import loss_fn  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
AUGS = {"aug100": 100, "aug40": 40}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E52 = _load("e52_run", "E52_dual_mode/run.py")
E51, E13 = E52.E51, E52.E13
MARGIN = E52.MARGIN


def train_aug(X, y, tr, max_ms, cfg, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    cg = torch.Generator().manual_seed(seed)
    m = E52.Net(False, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(cfg["epochs"] * int(np.ceil(len(tr) / cfg["bs"])), 1),
        pct_start=0.25)
    yt = torch.from_numpy(y).long()
    wc = torch.tensor([1.0, float((yt[tr] == 0).sum()) / max(int((yt[tr] == 1).sum()), 1)], device=device)
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    aug = dict(cfg, crop=1.0)
    dummy = torch.zeros(1)
    for ep in range(cfg["epochs"]):
        m.train()
        perm = np.random.default_rng(seed + ep).permutation(tr)
        for b in range(0, len(perm), cfg["bs"]):
            jj = perm[b:b + cfg["bs"]]
            o = MARGIN + int(rng.integers(0, T - n + 1))
            off = rng.integers(-max_ms, max_ms + 1, size=(len(jj), 4))      # 1 kHz: ms = samples
            xb = E13.augment(torch.from_numpy(E51.windows(X, jj, o, n, off)).to(device), aug, g, cg)
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(m(xb, None), yt[jj].to(device), dummy, dummy,
                              torch.zeros(1, dtype=torch.bool), wc, 0.0, 0.0)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E52.E50.E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.seeds = 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E52.E50.E49.E44._E12.build_cache(False)
    X, y, fold = d["X"], d["y"], d["fold"]
    rng = np.random.default_rng(E52.BASE_SEED)          # identical offsets to E52
    conds = {"none": np.zeros((len(y), 4), int)}
    for s in E52.FIXED_MS:
        conds[f"fixed{s:+d}"] = np.full((len(y), 4), s, int)
    conds["random"] = rng.integers(-E52.RAND_MS, E52.RAND_MS + 1, size=(len(y), 4))
    folds = [0] if args.smoke else range(5)
    seeds = [E52.BASE_SEED + 101 * i for i in range(args.seeds)]
    P = {(s, a, c): np.full(len(y), np.nan) for s in seeds for a in AUGS for c in conds}
    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        T2 = np.zeros((len(y), 4, 2), np.float32)
        for s in seeds:
            for a, mx in AUGS.items():
                net = train_aug(X, y, tr, mx, cfg, s, device)
                for c, off in conds.items():
                    P[(s, a, c)][te] = E52.predict(net, X, te, off[te], None, T2, None, cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)

    k_ = ~np.isnan(P[(seeds[0], "aug100", "none")])
    yy = y[k_]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "eval": {}}
    for a in AUGS:
        res["eval"][a] = {}
        for c in conds:
            rows = []
            for s in seeds:
                pc, p = P[(s, a, "none")][k_], P[(s, a, c)][k_]
                thr = np.sort(pc[yy == 1])[::-1][int(np.ceil(0.9 * yy.sum())) - 1]
                rows.append([roc_auc_score(yy, p), (p[yy == 1] >= thr).mean(), (p[yy == 0] < thr).mean(),
                             (lg(p) - lg(pc)).mean()])
            rows = np.array(rows)
            res["eval"][a][c] = {m_: [round(float(rows[:, i].mean()), 4), round(float(rows[:, i].std()), 4)]
                                 for i, m_ in enumerate(("auroc", "sen", "spe", "dlogit"))}
    e52 = json.loads((EXP / "E52_dual_mode/e52_dual.json").read_text(encoding="utf-8"))["eval"]
    print(f"\n{len(seeds)} seed(s); AUROC clean | spe random | spe +80 | spe -80 | dlogit -80/+80")
    rows = {**{k: e52[k] for k in ("pinet", "align", "dual:on", "dual:off")}, **res["eval"]}
    for k, E in rows.items():
        print(f"  {k:9s} {E['none']['auroc'][0]:.3f} | {100 * E['random']['spe'][0]:5.1f} | "
              f"{100 * E['fixed+80']['spe'][0]:5.1f} | {100 * E['fixed-80']['spe'][0]:5.1f} | "
              f"{E['fixed-80']['dlogit'][0]:+.2f}/{E['fixed+80']['dlogit'][0]:+.2f}", flush=True)
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e56_desync_aug{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)},
                                                             indent=2), encoding="utf-8")
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
