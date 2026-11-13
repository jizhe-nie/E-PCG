#!/usr/bin/env python
"""E74 -- PACFNet (Li et al., PeerJ Comput Sci 2025) reproduced on the main cohort's patient-level five folds.

    python experiments/E74_pacfnet/run.py --smoke
    python experiments/E74_pacfnet/run.py --seeds 3

SOURCE. Official code github.com/lightyLi/PACFNet-for-CVDs-detection (Apache-2.0; TensorFlow), ported layer for
layer to PyTorch below (dual_stream_model, down_sampling, CMR_module). Training follows training.py/config.py:
Adam lr 1e-3 with clipnorm 5, categorical cross-entropy, 10% of the training data held out as validation,
ReduceLROnPlateau(val_loss, factor 0.1, patience 5, min 1e-6), EarlyStopping(val_loss, patience 20, restore
best), at most 100 epochs, batch 32; class weights total / (2 n_class) as computed in training.py.

ADAPTATION TO THIS DATASET (stated before running)
  * beats: the original segments S1-to-S1 from hand-corrected PCG annotations, which this dataset does not
    have; beats are cut R-to-R from the ECG (assess_ecg R peaks) instead, one beat per sample, each modality
    resampled to 2000 points (cubic) exactly as time_scale_normalize_signal does, z-scored per beat.
  * subject score: mean predicted HFrEF probability over all beats of all four sites.
  * validation split: 10% of the training-fold SUBJECTS (not beats, to avoid leakage between beats of one
    subject); the test fold is never used for any decision.
  * evaluation: as every model in the paper -- clean out-of-fold AUROC, sensitivity/specificity at each
    seed's own 90%-sensitivity threshold, mean +- SD over seeds; plus the random U(-100,100) ms per-site and
    +80 ms PCG offsets (PCG beat taken from the offset position), descriptive.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.interpolate as si
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from epcg.preprocess.rpeak import assess_ecg  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
FS = 1000
L = 2000
SEED = 20261019
KS, LAYER_N, DROP = 7, 64, 0.5


# ----------------------------------------------------------------------------- PACFNet (PyTorch port)
class CBR(nn.Sequential):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__(nn.Conv1d(cin, cout, k, stride, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU())


class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f1, self.f2 = nn.Linear(c, c // 8), nn.Linear(c // 8, c)

    def forward(self, x):
        w = torch.sigmoid(self.f2(F.relu(self.f1(x.mean(-1)))))
        return x * w[:, :, None]


class Res(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.b = nn.Sequential(CBR(c, c, k), CBR(c, c, k), SE(c))

    def forward(self, x):
        return x + self.b(x)


class Down(nn.Module):
    """down_sampling(): optional concat with a pooled input, CBR (stride), two SE-residual blocks."""
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.cbr = CBR(cin, cout, k, stride)
        self.res = nn.Sequential(Res(cout, k), Res(cout, k))

    def forward(self, x, extra=None):
        if extra is not None:
            x = torch.cat([x, extra], 1)
        return self.res(self.cbr(x))


class CMR(nn.Module):
    """CMR_module(): spatial attention map (2 channels) + shared channel attention, then BN-ReLU-Conv fusion."""
    def __init__(self, c, k, c_last=0):
        super().__init__()
        cc = 2 * c
        self.sa = nn.Sequential(nn.Conv1d(cc, cc // 4, 1), nn.BatchNorm1d(cc // 4), nn.ReLU(),
                                nn.Conv1d(cc // 4, cc // 4, 16, padding="same"), nn.BatchNorm1d(cc // 4), nn.ReLU(),
                                nn.Conv1d(cc // 4, 2, 1))
        self.ca1, self.ca2 = nn.Linear(cc, c // 8), nn.Linear(c // 8, c)
        self.bn = nn.BatchNorm1d(cc + c_last)
        self.conv = nn.Conv1d(cc + c_last, c, k, padding=k // 2)

    def forward(self, a, b, last=None):
        cat = torch.cat([a, b], 1)
        m = torch.sigmoid(self.sa(cat))
        wa, wb = a * m[:, 0:1], b * m[:, 1:2]
        ch = torch.sigmoid(self.ca2(F.relu(self.ca1(cat.mean(-1)))))[:, :, None]
        wa, wb = wa * ch, wb * ch
        x = torch.cat([wa, wb] + ([last] if last is not None else []), 1)
        return self.conv(F.relu(self.bn(x)))


class PACFNet(nn.Module):
    def __init__(self, n=LAYER_N, k=KS):
        super().__init__()
        self.e1, self.p1 = Down(1, n, k, 1), Down(1, n, k, 1)
        self.e2, self.p2 = Down(n, 2 * n, k, 5), Down(n, 2 * n, k, 5)
        self.e3, self.p3 = Down(2 * n + 1, 3 * n, k, 5), Down(2 * n + 1, 3 * n, k, 5)
        self.e4, self.p4 = Down(3 * n + 1, 4 * n, k, 5), Down(3 * n + 1, 4 * n, k, 5)
        self.c1 = CMR(n, k)
        self.c2 = CMR(2 * n, k, 2 * n)
        self.c3 = CMR(3 * n, k, 3 * n)
        self.c4 = CMR(4 * n, k, 4 * n)
        self.t1, self.t2, self.t3 = nn.Conv1d(n, 2 * n, 1), nn.Conv1d(2 * n, 3 * n, 1), nn.Conv1d(3 * n, 4 * n, 1)
        self.head = nn.Sequential(nn.BatchNorm1d(4 * n), nn.ReLU(), nn.Conv1d(4 * n, 256, 3, 2, padding=1),
                                  nn.BatchNorm1d(256), nn.ReLU())
        self.cls = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(DROP), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Dropout(DROP), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(DROP), nn.Linear(32, 2))

    def forward(self, ecg, pcg):
        pool = lambda x, p: F.avg_pool1d(x, p)
        e1, p1 = self.e1(ecg), self.p1(pcg)
        c1 = self.c1(e1, p1)
        e2, p2 = self.e2(e1), self.p2(p1)
        c2 = self.c2(e2, p2, self.t1(pool(c1, 5)))
        e3, p3 = self.e3(e2, pool(ecg, 5)), self.p3(p2, pool(pcg, 5))
        c3 = self.c3(e3, p3, self.t2(pool(c2, 5)))
        e4, p4 = self.e4(e3, pool(ecg, 25)), self.p4(p3, pool(pcg, 25))
        c4 = self.c4(e4, p4, self.t3(pool(c3, 5)))
        return self.cls(self.head(c4).mean(-1))


# ----------------------------------------------------------------------------- beats
def resample(x, n=L):
    t0 = np.linspace(0, 1, len(x)); t1 = np.linspace(0, 1, n)
    y = si.interp1d(t0, x, kind="cubic")(t1)
    return ((y - y.mean()) / (y.std() + 1e-6)).astype(np.float32)


def beats_of(X, i, pcg_off_ms=None, margin=300):
    """All R-to-R beats of subject i (four sites); the PCG beat is read pcg_off_ms later in the buffer."""
    E, P = [], []
    for j in range(4):
        ecg, pcg = X[i, j, 0].astype(float), X[i, j, 1].astype(float)
        q, pk = assess_ecg(ecg, FS)
        if not q.usable or len(pk) < 3:
            continue
        o = 0 if pcg_off_ms is None else int(pcg_off_ms[j])
        for a, b in zip(pk[:-1], pk[1:]):
            if not (300 <= b - a <= 2000) or a - o < 0 or b - o > len(pcg) or a < margin or b > len(ecg) - margin:
                continue
            E.append(resample(ecg[a:b])); P.append(resample(pcg[a - o:b - o]))
    return (np.stack(E), np.stack(P)) if E else (np.zeros((0, L), np.float32), np.zeros((0, L), np.float32))


def train(BE, BP, BY, VE, VP, VY, epochs, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = PACFNet().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5, min_lr=1e-6)
    n_pos, n_neg = BY.sum(), len(BY) - BY.sum()
    w = torch.tensor([len(BY) / (2 * n_neg), len(BY) / (2 * n_pos)], dtype=torch.float32, device=device)
    tt = lambda a: torch.from_numpy(a[:, None, :]).to(device)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(BY)), max(1, len(BY) // 32)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(tt(BE[b]), tt(BP[b])), torch.from_numpy(BY[b]).to(device), weight=w)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([F.cross_entropy(m(tt(VE[b]), tt(VP[b])), torch.from_numpy(VY[b]).to(device), weight=w).item()
                                for b in np.array_split(np.arange(len(VY)), max(1, len(VY) // 128))]))
        sch.step(vl)
        if vl < best - 1e-6:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    m.load_state_dict(best_state)
    return m.eval(), ep + 1


@torch.no_grad()
def subject_prob(m, E, P, device):
    if len(E) == 0:
        return np.nan
    ps = []
    for b in range(0, len(E), 256):
        ps.append(torch.softmax(m(torch.from_numpy(E[b:b + 256, None]).to(device), torch.from_numpy(P[b:b + 256, None]).to(device)), -1)[:, 1].cpu().numpy())
    return float(np.concatenate(ps).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("e12", EXP / "E12_pinet" / "run.py")
    e12 = importlib.util.module_from_spec(spec); sys.modules["e12"] = e12; spec.loader.exec_module(e12)
    d = e12.build_cache(False)
    X, y, fold = d["X"], d["y"], d["fold"]
    rng = np.random.default_rng(SEED)
    conds = {"none": np.zeros((len(y), 4), int), "random": rng.integers(-100, 101, (len(y), 4)),
             "fixed+80": np.full((len(y), 4), 80, int)}
    subs = np.arange(len(y)) if not args.smoke else np.flatnonzero(fold <= 1)[:160]
    cache = {c: {} for c in conds}
    for c, off in conds.items():
        for i in subs:
            cache[c][i] = beats_of(X, i, off[i])
    print(f"beats cached ({time.time() - t0:.0f}s); clean beats per subject median "
          f"{np.median([len(cache['none'][i][0]) for i in subs]):.0f}", flush=True)
    folds = [0] if args.smoke else range(5)
    seeds = [SEED + 101 * k for k in range(1 if args.smoke else args.seeds)]
    epochs = 2 if args.smoke else args.epochs
    P = {(s, c): np.full(len(y), np.nan) for s in seeds for c in conds}
    meta = []
    for f in folds:
        te = np.intersect1d(np.flatnonzero(fold == f), subs); tr_all = np.intersect1d(np.flatnonzero(fold != f), subs)
        frng = np.random.default_rng(SEED + f)
        va = frng.choice(tr_all, size=max(2, len(tr_all) // 10), replace=False)
        tr = np.setdiff1d(tr_all, va)
        stack = lambda ids: (np.concatenate([cache["none"][i][0] for i in ids]), np.concatenate([cache["none"][i][1] for i in ids]),
                             np.concatenate([np.full(len(cache["none"][i][0]), y[i], np.int64) for i in ids]))
        BE, BP, BY = stack(tr); VE, VP, VY = stack(va)
        for s in seeds:
            ck = OUT / "partial" / f"f{f}_s{s}{'_smoke' if args.smoke else ''}.npz"   # resume after a crash
            if ck.exists():
                z = np.load(ck)
                for c in conds:
                    P[(s, c)][te] = z[c]
                meta.append({"fold": int(f), "seed": s, "epochs_run": int(z["epochs_run"]), "train_beats": int(len(BY))})
                print(f"  fold {f} seed {s}: resumed from checkpoint", flush=True)
                continue
            m, n_ep = train(BE, BP, BY, VE, VP, VY, epochs, device, s)
            meta.append({"fold": int(f), "seed": s, "epochs_run": n_ep, "train_beats": int(len(BY))})
            for c in conds:
                for i in te:
                    P[(s, c)][i] = subject_prob(m, cache[c][i][0], cache[c][i][1], device)
            ck.parent.mkdir(exist_ok=True)
            np.savez(ck, epochs_run=n_ep, **{c: P[(s, c)][te] for c in conds})
            print(f"  fold {f} seed {s}: {n_ep} epochs ({time.time() - t0:.0f}s)", flush=True)
    k_ = np.isfinite(P[(seeds[0], "none")])
    yy = y[k_]
    nthr = int(np.ceil(0.9 * yy.sum())) - 1
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "meta": meta,
           "n_params": int(sum(p.numel() for p in PACFNet().parameters())), "eval": {}}
    for c in conds:
        rows = []
        for s in seeds:
            pc, p = P[(s, "none")][k_], P[(s, c)][k_]
            t = np.sort(pc[yy == 1])[::-1][nthr]
            se, sp = (p[yy == 1] >= t).mean(), (p[yy == 0] < t).mean()
            rows.append([roc_auc_score(yy, p), se, sp, yy.mean() * se + (1 - yy.mean()) * sp])
        rows = np.array(rows)
        res["eval"][c] = {k: [round(float(rows[:, i].mean()), 4), round(float(rows[:, i].std()), 4)]
                          for i, k in enumerate(("auroc", "sen", "spe", "acc"))}
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1))
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e74_pacfnet{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2),
                                               encoding="utf-8")
    np.savez_compressed(OUT / f"e74_scores{tag}.npz", y=yy, **{f"{s}__{c}": v[k_] for (s, c), v in P.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
