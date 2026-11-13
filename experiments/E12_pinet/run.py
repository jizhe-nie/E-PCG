#!/usr/bin/env python
"""E12 -- does supervising physiology help, where writing it into the architecture did not?

    python experiments/E12_pinet/run.py --smoke
    python experiments/E12_pinet/run.py ablation

The ablation is the experiment; the network is just the vehicle. Every rung is the
same code path with auxiliary loss weights changed, so a difference cannot come from
anything else:

    base            classification only          w_lvef = 0, w_sti = 0
    +LVEF           add the continuous target    w_lvef > 0
    +STI            add EMAT/LVET/S3 regression  w_sti  > 0
    +both           full PI-Net

References on the same folds: the published LGC2-Net reaches 0.960 +- 0.020, and our
training-free acoustic panel reaches 0.886 +- 0.023 on the subset where S1 and S2 are
both extractable. Both are quoted, neither is re-run here.

Splits are the authors' own five folds, so nothing depends on a partition we chose.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps
from scipy.io import wavfile
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcg.models.pinet import PINet, SITES, STI_TARGETS, loss_fn  # noqa: E402
from epcg.paths import processed_dir, raw_dir  # noqa: E402

OUT = Path(__file__).parent
FS = 1000.0          # both modalities resampled here; S1/S2/S3 all live well below
SECONDS = 30.0
HFREF, SEED = 0.40, 20260908


def build_cache(force: bool = False) -> dict:
    """Waveforms + labels + per-site STI targets, cached as one npz."""
    p = processed_dir() / f"hfref_pinet_{int(FS)}hz.npz"
    if p.exists() and not force:
        z = np.load(p, allow_pickle=True)
        return {k: z[k] for k in z.files}

    lvef = {}
    for c in sorted(glob.glob(str(raw_dir() / "hfref" / "**" / "*.csv"), recursive=True)):
        for r in csv.reader(open(c, encoding="utf-8-sig")):
            if len(r) >= 2 and r[0].strip():
                try:
                    lvef[r[0].strip().rsplit("_", 1)[0]] = float(r[1])
                except ValueError:
                    pass
    fold_of = {}
    for d in sorted(glob.glob(str(raw_dir() / "hfref" / "HFrEF_5_folds" / "fold_*"))):
        f = int(os.path.basename(d).split("_")[1])
        for w in glob.glob(os.path.join(d, "*_ECG.wav")):
            fold_of[os.path.basename(w)[:-8].rsplit("_", 1)[0]] = f

    # Per-site STI targets from the beat table -- free labels, already computed.
    sti = defaultdict(dict)
    beats = defaultdict(list)
    for r in csv.DictReader(open(processed_dir() / "hfref_beats.csv", encoding="utf-8")):
        try:
            beats[(r["subject"], r["site"])].append(
                (float(r["emat_ms"]), float(r["lvet_ms"]), int(r["s3_detected"])))
        except ValueError:
            continue
    for (s, site), v in beats.items():
        a = np.asarray(v, float)
        sti[s][site] = [float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                        float(a[:, 2].mean())]

    n = int(SECONDS * FS)
    X, Y, L, T, M, S, Fd = [], [], [], [], [], [], []
    subs = sorted(s for s in fold_of if s in lvef)
    print(f"building cache for {len(subs)} subjects ...")
    for i, s in enumerate(subs, 1):
        sig, tgt, msk, ok = [], [], [], True
        for site in SITES:
            hits = glob.glob(str(raw_dir() / "hfref" / "**" / f"{s}_{site}_ECG.wav"),
                             recursive=True)
            if not hits:
                ok = False
                break
            fe, e = wavfile.read(hits[0])
            fp, pcg = wavfile.read(hits[0][:-8] + "_PCG.wav")
            e = sps.resample_poly(np.asarray(e, float).ravel(), int(FS), int(fe))
            pcg = sps.resample_poly(np.asarray(pcg, float).ravel(), int(FS), int(fp))
            if len(e) < n or len(pcg) < n:
                ok = False
                break
            z = lambda v: ((v - v.mean()) / (v.std() or 1.0)).astype(np.float32)
            sig.append(np.stack([z(e[:n]), z(pcg[:n])]))
            v = sti.get(s, {}).get(site)
            tgt.append(v if v else [np.nan] * 3)
            msk.append(bool(v))
        if not ok:
            continue
        X.append(np.stack(sig)); Y.append(int(lvef[s] <= HFREF)); L.append(lvef[s])
        T.append(tgt); M.append(msk); S.append(s); Fd.append(fold_of[s])
        if i % 100 == 0:
            print(f"  {i}/{len(subs)}")

    out = {"X": np.stack(X).astype(np.float32), "y": np.asarray(Y),
           "lvef": np.asarray(L, np.float32), "sti": np.asarray(T, np.float32),
           "sti_mask": np.asarray(M), "subjects": np.asarray(S),
           "fold": np.asarray(Fd)}
    np.savez_compressed(p, **out)
    print(f"cached {out['X'].shape} -> {p} ({p.stat().st_size / 2**20:.0f} MB)")
    return out


def run_fold(d, tr, te, args, device, w_lvef, w_sti, sti_mu, sti_sd):
    torch.manual_seed(args.seed)
    m = PINet(dim=args.dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    X = torch.from_numpy(d["X"]); y = torch.from_numpy(d["y"]).long()
    lv = torch.from_numpy(d["lvef"]); msk = torch.from_numpy(d["sti_mask"])
    sti = torch.from_numpy((d["sti"] - sti_mu) / sti_sd).nan_to_num()
    w = torch.tensor([1.0, float((y[tr] == 0).sum()) / max(int((y[tr] == 1).sum()), 1)],
                     device=device)

    for _ep in range(args.epochs):
        m.train()
        perm = np.random.default_rng(args.seed + _ep).permutation(tr)
        for i in range(0, len(perm), args.batch_size):
            j = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                o = m(X[j].to(device))
                loss, _ = loss_fn(o, y[j].to(device), lv[j].to(device),
                                  sti[j].to(device), msk[j].to(device), w,
                                  w_lvef, w_sti)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    m.eval()
    P = np.zeros(len(te))
    with torch.no_grad():
        for i in range(0, len(te), args.batch_size):
            j = te[i:i + args.batch_size]
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                o = m(X[j].to(device))
            P[i:i + len(j)] = torch.softmax(o["logits"].float(), -1)[:, 1].cpu().numpy()
    return P, m(X[te[:1]].to(device))["site_weights"].cpu().numpy().tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="ablation")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--w-lvef", type=float, default=1.0)
    ap.add_argument("--w-sti", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    d = build_cache(args.rebuild)
    y, fold = d["y"], d["fold"]
    print(f"{len(y)} subjects, {int(y.sum())} HFrEF, folds {np.bincount(fold)}  "
          f"device={device}")
    sti_mu = np.nanmean(d["sti"].reshape(-1, 3), 0)
    sti_sd = np.nanstd(d["sti"].reshape(-1, 3), 0) + 1e-6

    rungs = {"base (classification only)": (0.0, 0.0),
             "+ LVEF regression": (args.w_lvef, 0.0),
             "+ STI regression": (0.0, args.w_sti),
             "+ both (PI-Net)": (args.w_lvef, args.w_sti)}
    if args.smoke:
        rungs = {k: rungs[k] for k in list(rungs)[:2]}

    res = {}
    for name, (wl, ws) in rungs.items():
        print(f"\n{'=' * 70}\n{name}   (w_lvef={wl}, w_sti={ws})\n{'=' * 70}")
        t0 = time.time()
        oof = np.zeros(len(y)); sw = None
        folds = [0] if args.smoke else range(5)
        for f in folds:
            te = np.flatnonzero(fold == f); tr = np.flatnonzero(fold != f)
            p, sw = run_fold(d, tr, te, args, device, wl, ws, sti_mu, sti_sd)
            oof[te] = p
            print(f"  fold {f}: n={len(te)} ({int(y[te].sum())} HFrEF)  "
                  f"AUROC {roc_auc_score(y[te], p):.3f}")
        m = fold == 0 if args.smoke else np.ones(len(y), bool)
        auc = roc_auc_score(y[m], oof[m]); apr = average_precision_score(y[m], oof[m])
        rng = np.random.default_rng(SEED)
        b = [roc_auc_score(y[m][k], oof[m][k]) for k in
             (rng.integers(0, m.sum(), m.sum()) for _ in range(2000))
             if len(np.unique(y[m][k])) > 1]
        res[name] = {"auroc": round(float(auc), 4),
                     "ci": [round(float(np.percentile(b, 2.5)), 4),
                            round(float(np.percentile(b, 97.5)), 4)],
                     "auprc": round(float(apr), 4), "site_weights": sw,
                     "w_lvef": wl, "w_sti": ws,
                     "seconds": round(time.time() - t0, 1)}
        r = res[name]
        print(f"\n  AUROC {auc:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"AUPRC {apr:.3f}  site weights "
              f"{dict(zip(SITES, [round(x, 3) for x in sw]))}  "
              f"({r['seconds']:.0f}s)")

    (OUT / "e12_results.json").write_text(json.dumps(
        {"n_subjects": int(len(y)), "n_hfref": int(y.sum()), "rungs": res,
         "reference": {"LGC2-Net (published)": 0.960,
                       "acoustic panel, authors' folds": 0.886}},
        indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 70)
    for k, v in res.items():
        print(f"  {k:28s} AUROC {v['auroc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    print("  reference: LGC2-Net 0.960 (published) | acoustic panel 0.886 (same folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
