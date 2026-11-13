#!/usr/bin/env python
"""E76 -- HPA-Net (Xiao & Che, Biomed. Phys. Eng. Express 12 (2026) 045057) reproduced on the main cohort's
patient-level five folds.

    python experiments/E76_hpanet/run.py --smoke
    python experiments/E76_hpanet/run.py --seeds 3

SOURCE. No public code was found (2026-10-16); the network is RECONSTRUCTED from the paper: Table 2 (encoder
structural parameters), Table 13 (per-module parameter counts, total 210,574), Figures 2-5 and Algorithm 1 (FAMO).
Where the paper is silent, the choices below were fixed before the first training run and are not tuned:
  * encoder (per branch: ECG, PCG, EF): CAMS = two stages [MDConv K(7,11,21,31) (mixed kernels, channels split
    over the four kernels) -> BN-ReLU -> CA (depth-wise small kernel 3 then large kernel 9 / 13, point-wise, sigmoid
    gate) -> SE re8 -> MaxPool 4] with a stride-4 Conv skip each; BiLSTM U32 over the CAMS output; concat
    [CAMS 32, BiLSTM 64] -> Conv1 K1 C(96->32) (Table 13: 3,072 = 96 x 32) -> BN-ReLU
  * three SEResNeXt ResBlocks (K3, G16, width x4 bottleneck, SE re8, stride 2), each followed by ADFA (current +
    stride-matched previous feature, GAP-FC-ReLU-FC-sigmoid weights, weighted sum)
  * EF branch input = ECG and PCG concatenated along time (MDConv1 has one input channel)
  * progressive-fusion branch at the three ResBlock levels: CAP (GAP over [ECG, EF, PCG] -> conv -> ReLU -> conv
    -> sigmoid -> 3C weights; weighted features concat with the previous MFE output -> BN-ReLU-Conv) then MFE
    (four channel groups with hierarchical conv-BN-ReLU, a MaxPool path, conv + residual; stride 2 to the next level)
  * four heads (BN-ReLU-GAP-FC32-ReLU-Dropout 0.3-FC2); decision fusion: omega = omega0 [0.1,0.1,0.1,0.2] +
    softmax attention over the concatenated branch logits; final = sum omega_k p_k / sum omega_k
  * loss: focal loss (gamma 2, alpha = class weights inverse to prevalence) per branch combined by FAMO
    (weight lr 0.025, gamma_reg 0.01, eps 1e-8, L_min = 0); the fused output's focal loss is added unweighted so
    the decision-fusion attention receives gradient (the paper does not say how it is trained)
  * training as in Sec. 3.1: Adam lr 0.0012, batch 32, He init, grad-norm clip 1.0, at most 100 epochs,
    early stopping on validation loss (fused focal) with patience 20, best weights restored
ADAPTATION TO THIS DATASET (stated before running)
  * preprocessing as Sec. 2.1: PCG Butterworth band-pass 25-450 Hz (4th order), ECG high-pass 0.4 Hz, both
    resampled to 2000 Hz, each 5 s window scaled by its max |x|; windows 5 s, stride 3 s, inside a 100 ms margin
    (9 windows per site, 36 per subject); split strictly by subject (the cohort's five folds)
  * validation = 10% of the training-fold subjects (same draw as E74); test fold never used for a decision
  * subject score = mean fused HFrEF probability over all windows of all four sites
  * evaluation identical to E74: clean out-of-fold AUROC, sensitivity/specificity/accuracy at each seed's own
    90%-sensitivity threshold, mean +- SD over 3 seeds; plus random U(-100,100) ms per-site and fixed +80 ms PCG
    offsets (PCG window read at the offset position), descriptive
  * success criterion for the reconstruction (checked, reported either way): parameter count within +-25% of
    the paper's 210,574
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent
EXP = OUT.parent
FS0, FS = 1000, 2000
WIN, STRIDE, MARGIN = 5000, 3000, 100            # at 1 kHz
SEED = 20261019
C = 32
DROP = 0.3


# ----------------------------------------------------------------------------- HPA-Net (reconstruction)
class SE(nn.Module):
    def __init__(self, c, re=8):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(c, c // re), nn.ReLU(), nn.Linear(c // re, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x.mean(-1))[:, :, None]


class MDConv(nn.Module):
    """Mixed-kernel convolution: output channels split over kernels 7, 11, 21, 31."""
    def __init__(self, cin, cout, ks=(7, 11, 21, 31)):
        super().__init__()
        self.c = nn.ModuleList(nn.Conv1d(cin, cout // len(ks), k, padding=k // 2, bias=False) for k in ks)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(torch.cat([c(x) for c in self.c], 1)))


class CA(nn.Module):
    def __init__(self, c, sk, lk):
        super().__init__()
        self.s = nn.Conv1d(c, c, sk, padding=sk // 2, groups=c)
        self.l = nn.Conv1d(c, c, lk, padding=lk // 2, groups=c)
        self.p = nn.Conv1d(c, c, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.p(self.l(self.s(x))))


class CAMSStage(nn.Module):
    def __init__(self, cin, cout, lk, pool=4):
        super().__init__()
        self.md, self.ca, self.se = MDConv(cin, cout), CA(cout, 3, lk), SE(cout)
        self.pool = pool
        self.skip = nn.Conv1d(cin, cout, 1, stride=pool, bias=False)

    def forward(self, x):
        return F.max_pool1d(self.se(self.ca(self.md(x))), self.pool) + self.skip(x)[..., : x.shape[-1] // self.pool]


class ResNeXt(nn.Module):
    def __init__(self, c=C, g=16, w=4):
        super().__init__()
        m = c * w // 2
        self.b = nn.Sequential(nn.Conv1d(c, m, 1, bias=False), nn.BatchNorm1d(m), nn.ReLU(),
                               nn.Conv1d(m, m, 3, 2, 1, groups=g, bias=False), nn.BatchNorm1d(m), nn.ReLU(),
                               nn.Conv1d(m, c, 1, bias=False), nn.BatchNorm1d(c), SE(c))
        self.skip = nn.Conv1d(c, c, 1, stride=2, bias=False)

    def forward(self, x):
        return F.relu(self.b(x) + self.skip(x))


class ADFA(nn.Module):
    def __init__(self, c=C, re=8):
        super().__init__()
        self.cur, self.prev = nn.Conv1d(c, c, 1), nn.Conv1d(c, c, 1)
        self.f = nn.Sequential(nn.Linear(2 * c, c // re), nn.ReLU(), nn.Linear(c // re, 2 * c), nn.Sigmoid())

    def forward(self, cur, prev):
        prev = F.adaptive_avg_pool1d(prev, cur.shape[-1])
        a, b = self.cur(cur), self.prev(prev)
        w = self.f(torch.cat([a, b], 1).mean(-1))[:, :, None]
        return a * w[:, : a.shape[1]] + b * w[:, a.shape[1]:]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1, self.s2 = CAMSStage(1, C, 9), CAMSStage(C, C, 13)
        self.lstm = nn.LSTM(C, C, batch_first=True, bidirectional=True)
        self.c1 = nn.Sequential(nn.Conv1d(3 * C, C, 1, bias=False), nn.BatchNorm1d(C), nn.ReLU())
        self.res = nn.ModuleList(ResNeXt() for _ in range(3))
        self.adfa = nn.ModuleList(ADFA() for _ in range(3))

    def forward(self, x):
        h = self.s2(self.s1(x))
        l, _ = self.lstm(h.transpose(1, 2))
        f = self.c1(torch.cat([h, l.transpose(1, 2)], 1))
        lv = []
        for r, a in zip(self.res, self.adfa):
            f = a(r(f), f)
            lv.append(f)
        return lv


class CAP(nn.Module):
    def __init__(self, c=C, first=False):
        super().__init__()
        self.g = nn.Sequential(nn.Conv1d(3 * c, 3 * c // 8, 1), nn.ReLU(), nn.Conv1d(3 * c // 8, 3 * c, 1), nn.Sigmoid())
        cin = 3 * c if first else 4 * c
        self.out = nn.Sequential(nn.BatchNorm1d(cin), nn.ReLU(), nn.Conv1d(cin, c, 1))

    def forward(self, e, f, p, prev=None):
        cat = torch.cat([e, F.adaptive_avg_pool1d(f, e.shape[-1]), p], 1)    # EF is twice as long (time concat)
        z = cat * self.g(cat.mean(-1, keepdim=True))
        if prev is not None:
            z = torch.cat([z, F.adaptive_avg_pool1d(prev, z.shape[-1])], 1)
        return self.out(z)


class MFE(nn.Module):
    def __init__(self, c=C):
        super().__init__()
        q = c // 4
        self.cv = nn.ModuleList(nn.Sequential(nn.Conv1d(q, q, 3, padding=1, bias=False), nn.BatchNorm1d(q), nn.ReLU())
                                for _ in range(3))
        self.out = nn.Sequential(nn.Conv1d(c, c, 1, bias=False), nn.BatchNorm1d(c), nn.ReLU())

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, 1)
        y4 = self.cv[0](x4)
        y3 = self.cv[1](x3 + y4)
        y2 = self.cv[2](x2 + y3)
        y = self.out(torch.cat([x1, y2, y3, y4], 1)) + x
        return F.max_pool1d(y, 2)


def head():
    return nn.Sequential(nn.BatchNorm1d(C), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(C, 32), nn.ReLU(),
                         nn.Dropout(DROP), nn.Linear(32, 2))


class HPANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList(Encoder() for _ in range(3))           # ECG, EF, PCG
        self.cap = nn.ModuleList(CAP(first=(i == 0)) for i in range(3))
        self.mfe = nn.ModuleList(MFE() for _ in range(3))
        self.heads = nn.ModuleList(head() for _ in range(4))             # ECG, PCG, EF, PF
        self.att = nn.Sequential(nn.Linear(8, 4), nn.Softmax(-1))
        self.register_buffer("w0", torch.tensor([0.1, 0.1, 0.1, 0.2]))
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, ecg, pcg):
        le, lf, lp = self.enc[0](ecg), self.enc[1](torch.cat([ecg, pcg], -1)), self.enc[2](pcg)
        m = None
        for i in range(3):
            m = self.mfe[i](self.cap[i](le[i], lf[i], lp[i], m))
        logits = [self.heads[0](le[-1]), self.heads[1](lp[-1]), self.heads[2](lf[-1]), self.heads[3](m)]
        w = self.w0 + self.att(torch.cat(logits, 1))
        probs = torch.stack([torch.softmax(z, -1) for z in logits], 1)    # B x 4 x 2
        fused = (w[:, :, None] * probs).sum(1) / w.sum(1, keepdim=True)
        return logits, fused


def focal(logits_or_prob, y, alpha, gamma=2.0, is_prob=False):
    p = logits_or_prob if is_prob else torch.softmax(logits_or_prob, -1)
    pt = p.gather(1, y[:, None]).squeeze(1).clamp(1e-7, 1.0)
    return (-alpha[y] * (1 - pt) ** gamma * torch.log(pt)).mean()


class FAMO:
    """Liu et al. 2023 fast adaptive multitask optimisation, as in Algorithm 1 (L_min = 0)."""
    def __init__(self, k, device, lr=0.025, reg=0.01, eps=1e-8):
        self.w = torch.zeros(k, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=lr, weight_decay=reg)
        self.eps, self.prev = eps, None

    def loss(self, L):
        D = L + self.eps
        z = torch.softmax(self.w, -1).detach()
        c = (z / D.detach()).sum().detach()
        self.prev = D.detach()
        return (z / c * torch.log(D)).sum()

    def update(self, L_new):
        delta = torch.log(self.prev) - torch.log(L_new.detach() + self.eps)
        self.opt.zero_grad()
        with torch.enable_grad():
            torch.autograd.backward(torch.softmax(self.w, -1), grad_tensors=delta)
        self.opt.step()


# ----------------------------------------------------------------------------- data
def prep(X):
    """Filter at 1 kHz, keep float32; returns ECG, PCG arrays (n, 4, T)."""
    be = sps.butter(4, 0.4, "highpass", fs=FS0, output="sos")
    bp = sps.butter(4, [25, 450], "bandpass", fs=FS0, output="sos")
    return (sps.sosfiltfilt(be, X[:, :, 0], axis=-1).astype(np.float32),
            sps.sosfiltfilt(bp, X[:, :, 1], axis=-1).astype(np.float32))


def windows_of(E, P, i, off):
    """All 5 s windows of subject i at 2 kHz; the PCG window is read off[j] ms later in the buffer."""
    we, wp = [], []
    T = E.shape[-1]
    for j in range(4):
        o = int(off[j])
        for a in range(MARGIN, T - MARGIN - WIN + 1, STRIDE):
            if a - o < 0 or a - o + WIN > T:
                continue
            for src, dst, s in ((E, we, a), (P, wp, a - o)):
                v = sps.resample_poly(src[i, j, s:s + WIN], 2, 1)
                dst.append((v / (np.abs(v).max() + 1e-6)).astype(np.float16))
    return np.stack(we), np.stack(wp)


def run_batches(m, E, P, device, bs=64):
    out = []
    with torch.no_grad():
        for b in range(0, len(E), bs):
            tt = lambda a: torch.from_numpy(a[b:b + bs, None].astype(np.float32)).to(device)
            out.append(m(tt(E), tt(P))[1][:, 1].cpu().numpy())
    return np.concatenate(out)


def train(BE, BP, BY, VE, VP, VY, epochs, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = HPANet().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1.2e-3)
    famo = FAMO(4, device)
    n_pos = BY.sum()
    alpha = torch.tensor([len(BY) / (2 * (len(BY) - n_pos)), len(BY) / (2 * n_pos)], dtype=torch.float32, device=device)
    tt = lambda a, b: torch.from_numpy(a[b, None].astype(np.float32)).to(device)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(BY)), max(1, len(BY) // 32)):
            y = torch.from_numpy(BY[b]).to(device)
            logits, fused = m(tt(BE, b), tt(BP, b))
            L = torch.stack([focal(z, y, alpha) for z in logits])
            loss = famo.loss(L) + focal(fused, y, alpha, is_prob=True)
            opt.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            with torch.no_grad():
                logits, _ = m(tt(BE, b), tt(BP, b))
                famo.update(torch.stack([focal(z, y, alpha) for z in logits]))
        m.eval()
        vl = []
        with torch.no_grad():
            for b in np.array_split(np.arange(len(VY)), max(1, len(VY) // 128)):
                vl.append(focal(m(tt(VE, b), tt(VP, b))[1], torch.from_numpy(VY[b]).to(device), alpha, is_prob=True).item())
        vl = float(np.mean(vl))
        if vl < best - 1e-6:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    m.load_state_dict(best_state)
    return m.eval(), ep + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_par = sum(p.numel() for p in HPANet().parameters())
    print(f"HPA-Net reconstruction: {n_par:,} parameters (paper 210,574; criterion +-25%: "
          f"{'met' if abs(n_par / 210574 - 1) <= 0.25 else 'NOT met'})", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("e12", EXP / "E12_pinet" / "run.py")
    e12 = importlib.util.module_from_spec(spec); sys.modules["e12"] = e12; spec.loader.exec_module(e12)
    d = e12.build_cache(False)
    X, y, fold = d["X"], d["y"], d["fold"]
    E, P = prep(X)
    del X
    rng = np.random.default_rng(SEED)                                   # same offsets as E74
    conds = {"none": np.zeros((len(y), 4), int), "random": rng.integers(-100, 101, (len(y), 4)),
             "fixed+80": np.full((len(y), 4), 80, int)}
    subs = np.arange(len(y)) if not args.smoke else np.flatnonzero(fold <= 1)[:160]
    cache = {c: {i: windows_of(E, P, i, off[i]) for i in subs} for c, off in conds.items()}
    print(f"windows cached ({time.time() - t0:.0f}s); clean windows per subject median "
          f"{np.median([len(cache['none'][i][0]) for i in subs]):.0f}", flush=True)
    folds = [0] if args.smoke else range(5)
    seeds = [SEED + 101 * k for k in range(1 if args.smoke else args.seeds)]
    epochs = 2 if args.smoke else args.epochs
    PR = {(s, c): np.full(len(y), np.nan) for s in seeds for c in conds}
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
            m, n_ep = train(BE, BP, BY, VE, VP, VY, epochs, device, s)
            meta.append({"fold": int(f), "seed": s, "epochs_run": n_ep, "train_windows": int(len(BY))})
            for c in conds:
                for i in te:
                    PR[(s, c)][i] = float(run_batches(m, *cache[c][i], device).mean())
            print(f"  fold {f} seed {s}: {n_ep} epochs ({time.time() - t0:.0f}s)", flush=True)
    k_ = np.isfinite(PR[(seeds[0], "none")])
    yy = y[k_]
    nthr = int(np.ceil(0.9 * yy.sum())) - 1
    res = {"n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "meta": meta, "n_params": int(n_par),
           "n_params_paper": 210574, "eval": {}}
    for c in conds:
        rows = []
        for s in seeds:
            pc, p = PR[(s, "none")][k_], PR[(s, c)][k_]
            t = np.sort(pc[yy == 1])[::-1][nthr]
            se, sp = (p[yy == 1] >= t).mean(), (p[yy == 0] < t).mean()
            rows.append([roc_auc_score(yy, p), se, sp, yy.mean() * se + (1 - yy.mean()) * sp])
        rows = np.array(rows)
        res["eval"][c] = {k: [round(float(rows[:, i].mean()), 4), round(float(rows[:, i].std()), 4)]
                          for i, k in enumerate(("auroc", "sen", "spe", "acc"))}
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1))
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e76_hpanet{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2),
                                              encoding="utf-8")
    np.savez_compressed(OUT / f"e76_scores{tag}.npz", y=yy, **{f"{s}__{c}": v[k_] for (s, c), v in PR.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
