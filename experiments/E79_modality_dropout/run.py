#!/usr/bin/env python
"""E79 -- graceful degradation when one modality is missing or broken; modality-dropout training (HPA-Net, zero-fill).

    python experiments/E79_modality_dropout/run.py --smoke
    python experiments/E79_modality_dropout/run.py --seeds 3

WHY. In portable acquisition the PCG channel is the fragile one (microphone not in contact, dropped wireless
audio). HPA-Net trains with modality dropout (missing modality zero-filled). The late-fusion PINet of E77 is the
natural candidate: each modality has its own encoder, so a missing channel removes one half of the features
instead of corrupting every convolution. Inputs are z-scored per channel, so a zero channel is "the mean".

ARMS (same five folds and E52 recipe, 3 seeds; no operator, no timing input)
  pinet     early fusion (conv baseline)
  late      late-fusion PINet (E77)
  late_md   late-fusion PINet trained with modality dropout: per training sample, the ECG channel is zeroed with
            probability 0.15 and the PCG channel with probability 0.15 (never both)
  ecg_ref   late-fusion PINet with the PCG channel always zero (train and test) -- the ECG-only reference
  pcg_ref   likewise with the ECG channel always zero -- the PCG-only reference
TEST CONDITIONS (all four sites)  none | pcg_zero | ecg_zero | pcg_noise (PCG replaced by unit-variance Gaussian
  noise, fixed seed: a disconnected microphone that still records hiss)
THRESHOLD  frozen per seed at 90% sensitivity on each arm's own clean scores; AUROC reported as seed mean.

PRE-SPECIFIED (reported either way)
  PRIMARY   AUROC under pcg_zero, late_md minus late (seed-mean paired difference, subject bootstrap);
            supported if the 95% CI lies above 0
  C1        late_md clean AUROC non-inferior to late: difference >= -0.01 (point estimate)
  C2        graceful degradation: late_md AUROC under pcg_zero >= ecg_ref clean AUROC - 0.02, and under
            ecg_zero >= pcg_ref clean AUROC - 0.02
  descriptive  sensitivity / specificity at the frozen threshold under every condition (a missing channel shifts
            the score distribution, so the clean threshold may no longer sit at 90% sensitivity); pinet as the
            early-fusion reference
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
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


E77 = _load("e77_run", "E77_seconds_offset/run.py")
E52, E50 = E77.E52, E77.E50
BASE_SEED = 20261020
P_DROP = 0.15
CONDS = ["none", "pcg_zero", "ecg_zero", "pcg_noise"]


def variant(md: bool = False, force_zero: int | None = None):
    """LateNet subclass: modality dropout while training (md) and/or one channel always zero (force_zero)."""
    class V(E77.LateNet):
        def forward(self, x, t=None):
            if force_zero is not None:
                x = x.clone(); x[:, :, force_zero] = 0
            elif md and self.training:
                u = torch.rand(x.shape[0], device=x.device)
                x = x.clone()
                x[u < P_DROP, :, 0] = 0
                x[(u >= P_DROP) & (u < 2 * P_DROP), :, 1] = 0
            return super().forward(x, t)
    return V


def train_with(cls, X, y, tr, cfg, seed, device):
    net = E52.Net
    E52.Net = cls
    try:
        return E52.train(X, y, tr, None, None, None, cfg, seed, device)
    finally:
        E52.Net = net


class Cond(torch.nn.Module):
    """Wraps a trained model and corrupts the input for a test condition."""
    def __init__(self, m, cond):
        super().__init__()
        self.m, self.cond = m, cond
        self.g = torch.Generator(device="cpu").manual_seed(BASE_SEED)

    def forward(self, x, t=None):
        if self.cond != "none":
            x = x.clone()
            if self.cond == "pcg_zero":
                x[:, :, 1] = 0
            elif self.cond == "ecg_zero":
                x[:, :, 0] = 0
            elif self.cond == "pcg_noise":
                x[:, :, 1] = torch.randn(x[:, :, 1].shape, generator=self.g).to(x.device)
        return self.m(x, t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E50.E49.E44.CFG)
    if args.smoke:
        cfg["epochs"], args.seeds = 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E50.E49.E44._E12.build_cache(False)
    X, y, fold = d["X"], d["y"], d["fold"]
    arms = {"pinet": None, "late": variant(), "late_md": variant(md=True),
            "ecg_ref": variant(force_zero=1), "pcg_ref": variant(force_zero=0)}
    folds = [0] if args.smoke else range(5)
    seeds = [BASE_SEED + 101 * i for i in range(args.seeds)]
    P = {(s, a, c): np.full(len(y), np.nan) for s in seeds for a in arms for c in CONDS}
    zero = np.zeros((len(y), 4), int)
    for f in folds:
        te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
        for s in seeds:
            for a, cls in arms.items():
                m = (E52.train(X, y, tr, None, None, None, cfg, s, device) if cls is None
                     else train_with(cls, X, y, tr, cfg, s, device))
                for c in CONDS:
                    P[(s, a, c)][te] = E77.predict(Cond(m, c), X, te, zero[te], None, None, None, cfg, device)
        print(f"  fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    k_ = ~np.isnan(P[(seeds[0], "pinet", "none")])
    yy = y[k_]
    Pk = {key: v[k_] for key, v in P.items()}
    metric = own_threshold_metric(Pk, yy, "none")
    bs = boots(yy, B=200 if args.smoke else 2000, seed=BASE_SEED)
    full = np.arange(len(yy))
    auc = lambda a, c, idx: np.mean([roc_auc_score(yy[idx], Pk[(s, a, c)][idx]) for s in seeds])
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "p_drop": P_DROP, "eval": {}}
    for a in arms:
        res["eval"][a] = {}
        for c in CONDS:
            ss = [metric((a, c), s, full) for s in seeds]
            au = [roc_auc_score(yy, Pk[(s, a, c)]) for s in seeds]
            res["eval"][a][c] = {"auroc": [round(float(np.mean(au)), 4), round(float(np.std(au)), 4)],
                                 "sen": round(float(np.mean([v[0] for v in ss])), 4),
                                 "spe": round(float(np.mean([v[1] for v in ss])), 4)}
        print(f"{a:8s} " + "  ".join(f"{c}: AUROC {res['eval'][a][c]['auroc'][0]:.3f} sen {100 * res['eval'][a][c]['sen']:.1f} "
                                      f"spe {100 * res['eval'][a][c]['spe']:.1f}" for c in CONDS), flush=True)
    pt = auc("late_md", "pcg_zero", full) - auc("late", "pcg_zero", full)
    D = [auc("late_md", "pcg_zero", i) - auc("late", "pcg_zero", i) for i in bs]
    res["primary_auroc_pcg_zero_md_minus_late"] = {"diff": round(float(pt), 4),
                                                   "ci": [round(float(np.percentile(D, 2.5)), 4), round(float(np.percentile(D, 97.5)), 4)]}
    res["primary_supported"] = bool(res["primary_auroc_pcg_zero_md_minus_late"]["ci"][0] > 0)
    ev = res["eval"]
    res["C1_md_clean_noninferior"] = bool(ev["late_md"]["none"]["auroc"][0] - ev["late"]["none"]["auroc"][0] >= -0.01)
    res["C2_graceful"] = bool(ev["late_md"]["pcg_zero"]["auroc"][0] >= ev["ecg_ref"]["none"]["auroc"][0] - 0.02 and
                              ev["late_md"]["ecg_zero"]["auroc"][0] >= ev["pcg_ref"]["none"]["auroc"][0] - 0.02)
    res["paired_spe_vs_clean"] = {a: {c: paired(metric, (a, c), (a, "none"), seeds, len(yy), bs) for c in CONDS[1:]}
                                  for a in ("late", "late_md", "pinet")}
    print({k: res[k] for k in ("primary_auroc_pcg_zero_md_minus_late", "primary_supported", "C1_md_clean_noninferior", "C2_graceful")})
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e79_md{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2),
                                          encoding="utf-8")
    np.savez_compressed(OUT / f"e79_scores{tag}.npz", y=yy, **{f"{s}__{a}__{c}": v for (s, a, c), v in Pk.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
