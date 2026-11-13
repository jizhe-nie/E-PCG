#!/usr/bin/env python
"""E13 -- find the compact CNN's actual ceiling.

    python experiments/E13_tune/run.py --smoke
    python experiments/E13_tune/run.py sweep

The 0.926 from E12 came from a model that was never tuned once: 40 epochs, dim 128,
no augmentation, no schedule, no ensembling. It was written as the control rung for
PI-Net's ablation and got treated as a baseline when the auxiliary heads failed. It is
not a baseline -- it is our architecture, and how much of the 0.034 gap to the
published 0.960 is architectural rather than simply unspent effort is currently
unknown.

Augmentations are chosen for what this signal actually is, not from a vision recipe:

  random crop     a 30 s recording is a sample of an ongoing process; training on
                  random sub-windows is the one augmentation guaranteed to be
                  label-preserving here
  amplitude       stethoscope contact pressure varies between recordings and has
                  nothing to do with the diagnosis
  site dropout    E11 showed one or two sites carry nearly everything, so a model
                  that survives losing a site is closer to how it would be deployed
  gaussian noise  ordinary robustness

Deliberately NOT used: time reversal (inverts systole and diastole), pitch shift
(moves S1/S2/S3 out of their diagnostic bands), and mixup on the raw waveform (a
linear blend of two hearts is not a heart).

Everything is scored on the authors' own five folds with subject-level separation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcg.models.pinet import PINet, SITES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E12_pinet"))
from importlib import import_module  # noqa: E402

OUT = Path(__file__).parent
SEED = 20260908


def load_cache():
    spec = import_module("run") if False else None
    sys.path.insert(0, str(OUT.parent / "E12_pinet"))
    from run import build_cache  # noqa: E402
    return build_cache(False)


def augment(x: torch.Tensor, cfg: dict, gen: torch.Generator,
            cpu_gen: torch.Generator) -> torch.Tensor:
    """x: (B, S, 2, T). Applied on GPU, per batch."""
    B, S, C, T = x.shape
    if cfg["crop"] < 1.0:
        n = int(T * cfg["crop"])
        # randint returns a CPU tensor, so it needs the CPU generator; randn
        # below writes to the device and needs the device one.
        off = int(torch.randint(0, T - n + 1, (1,), generator=cpu_gen).item())
        x = x[..., off:off + n]
    if cfg["amp"] > 0:
        s = 1.0 + cfg["amp"] * (torch.rand(B, S, 1, 1, device=x.device) * 2 - 1)
        x = x * s
    if cfg["site_drop"] > 0:
        m = (torch.rand(B, S, 1, 1, device=x.device) > cfg["site_drop"]).float()
        # Never drop every site: a bag with no evidence has no label to predict.
        m = torch.where(m.sum(1, keepdim=True) == 0, torch.ones_like(m), m)
        x = x * m
    if cfg["noise"] > 0:
        x = x + cfg["noise"] * torch.randn(x.shape, device=x.device, generator=gen)
    return x


def run_fold(d, tr, te, cfg, device):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    cpu_gen = torch.Generator().manual_seed(cfg["seed"])
    m = PINet(dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(cfg["epochs"] *
        int(np.ceil(len(tr) / cfg["bs"])), 1), pct_start=0.25)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    X = torch.from_numpy(d["X"]); y = torch.from_numpy(d["y"]).long()
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max(int((y[tr] == 1).sum()), 1)],
                     device=device)

    for ep in range(cfg["epochs"]):
        m.train()
        perm = np.random.default_rng(cfg["seed"] + ep).permutation(tr)
        for i in range(0, len(perm), cfg["bs"]):
            j = perm[i:i + cfg["bs"]]
            xb = X[j].to(device)
            xb = augment(xb, cfg, gen, cpu_gen)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = F.cross_entropy(m(xb)["logits"], y[j].to(device), weight=w,
                                       label_smoothing=cfg["smooth"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

    m.eval()
    P = np.zeros(len(te))
    with torch.no_grad():
        for i in range(0, len(te), cfg["bs"]):
            j = te[i:i + cfg["bs"]]
            xb = X[j].to(device)
            # Test-time crops averaged: the same invariance the crop augmentation
            # trains for, used at inference instead of discarded.
            acc = 0.0
            n_tta = cfg["tta"]
            T = xb.shape[-1]; n = int(T * cfg["crop"]) if cfg["crop"] < 1 else T
            for t in range(n_tta):
                off = 0 if n == T else int((T - n) * t / max(n_tta - 1, 1))
                with torch.autocast(device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    acc = acc + torch.softmax(
                        m(xb[..., off:off + n])["logits"].float(), -1)[:, 1]
            P[i:i + len(j)] = (acc / n_tta).cpu().numpy()
    return P


def evaluate(d, cfg, device, seeds=(0,)):
    y, fold = d["y"], d["fold"]
    oof = np.zeros((len(seeds), len(y)))
    for si, s in enumerate(seeds):
        c = {**cfg, "seed": SEED + s}
        for f in range(5):
            te = np.flatnonzero(fold == f); tr = np.flatnonzero(fold != f)
            oof[si, te] = run_fold(d, tr, te, c, device)
    p = oof.mean(0)
    rng = np.random.default_rng(SEED)
    b = [roc_auc_score(y[k], p[k]) for k in
         (rng.integers(0, len(y), len(y)) for _ in range(2000))
         if len(np.unique(y[k])) > 1]
    return (round(float(roc_auc_score(y, p)), 4),
            [round(float(np.percentile(b, 2.5)), 4),
             round(float(np.percentile(b, 97.5)), 4)],
            round(float(average_precision_score(y, p)), 4), p)


BASE = dict(dim=128, dropout=0.3, lr=3e-4, wd=1e-2, bs=16, epochs=40,
            crop=1.0, amp=0.0, site_drop=0.0, noise=0.0, smooth=0.0, tta=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="sweep")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_cache()
    print(f"{len(d['y'])} subjects, {int(d['y'].sum())} HFrEF, device={device}\n")

    steps = [
        ("E12 as published", {}),
        ("+ random crop 0.7 + TTA", dict(crop=0.7, tta=3)),
        ("+ amplitude jitter", dict(crop=0.7, tta=3, amp=0.3)),
        ("+ site dropout", dict(crop=0.7, tta=3, amp=0.3, site_drop=0.25)),
        ("+ noise + smoothing", dict(crop=0.7, tta=3, amp=0.3, site_drop=0.25,
                                     noise=0.05, smooth=0.05)),
        ("+ longer (120 ep)", dict(crop=0.7, tta=3, amp=0.3, site_drop=0.25,
                                   noise=0.05, smooth=0.05, epochs=120)),
    ]
    if args.smoke:
        steps = steps[:2]
        BASE["epochs"] = 5

    res, best, best_cfg = {}, 0.0, None
    for name, delta in steps:
        cfg = {**BASE, **delta}
        t0 = time.time()
        auc, ci, apr, _ = evaluate(d, cfg, device)
        res[name] = {"cfg": {k: cfg[k] for k in delta} or "baseline",
                     "auroc": auc, "ci": ci, "auprc": apr,
                     "seconds": round(time.time() - t0, 1)}
        flag = ""
        if auc > best:
            best, best_cfg, flag = auc, cfg, "  <- best"
        print(f"  {name:26s} AUROC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"AUPRC {apr:.3f}  ({res[name]['seconds']:.0f}s){flag}")

    if not args.smoke and best_cfg:
        print("\n=== seed ensemble on the best configuration (3 seeds) ===")
        t0 = time.time()
        auc, ci, apr, _ = evaluate(d, best_cfg, device, seeds=(0, 1, 2))
        res["best + 3-seed ensemble"] = {"auroc": auc, "ci": ci, "auprc": apr,
                                         "seconds": round(time.time() - t0, 1)}
        print(f"  AUROC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  AUPRC {apr:.3f}  "
              f"({time.time() - t0:.0f}s)")

    (OUT / "e13_results.json").write_text(json.dumps(
        {"results": res, "base": BASE,
         "reference": {"E12 untuned": 0.926, "LGC2-Net published": 0.960,
                       "acoustic panel": 0.886}}, indent=2, default=str),
        encoding="utf-8")
    print(f"\nreference: E12 untuned 0.926 | acoustic panel 0.886 | "
          f"LGC2-Net published 0.960")
    print(f"wrote {OUT / 'e13_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
