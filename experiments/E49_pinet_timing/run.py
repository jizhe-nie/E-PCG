#!/usr/bin/env python
"""E49 -- has PINet learned the electromechanical delay, or only to depend on alignment?

    python experiments/E49_pinet_timing/run.py --smoke
    python experiments/E49_pinet_timing/run.py

PINet contains no delay operator: ECG and PCG are stacked as two channels of a strided
CNN, so alignment is GIVEN at the first layer rather than measured. E18 showed it loses
0.068 AUROC when the PCG moves 80 ms -- dependence, which is direction-free and says
nothing about whether it reads the interval. E46 showed that for LGC2-Net the same kind
of dependence turned out to be "desynchronisation looks like disease".

Two direct tests on the same trained models:

1. CLASSIFIER, SIGNED SHIFT (the E46 design). PCG moved -80/-40/+40/+80 ms relative to
   the ECG, all four sites alike, plus +-80 ms joint shifts. A model that reads a
   prolonged R->S1 as HFrEF raises its logit for +d and lowers it for -d. Specificity is
   also scored at the threshold frozen on clean out-of-fold predictions.

2. THE EMAT HEAD, TRACKING. With w_sti > 0 PINet regresses per-site EMAT (R->S1) from
   its own features. That head IS a delay estimate, so the regression tracking
   coefficient applies directly: +1 if it measures the interval, 0 if it predicts it
   from correlates. Shifts are crops of the 30 s buffer (E47), no roll seam.

Configurations: `base` (classification only) and `aux` = (w_lvef 10, w_sti 3,
standardised), the E44 grid point that trains the EMAT head. Authors' five folds; the
E44 recipe (augmentation, 3-crop TTA of 21 s windows).
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

from epcg.models.pinet import PINet, loss_fn  # noqa: E402

OUT = Path(__file__).parent
EXP = OUT.parent
SEED = 20260930
SHIFTS_MS = [-80, -40, 40, 80]
CONFIGS = {"base": (0.0, 0.0, True), "aux": (10.0, 3.0, True)}
MARGIN = 200                   # samples kept free at each end so every shift is a crop


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E44 = _load("e44_run", "E44_nested/run.py")
E13 = E44._E13


def train(X, y, LV, STI, MSK, tr, cfg, wl, ws, device):
    """E44.fit_predict's training half, returning the model and the target scalers."""
    lm, lsd = LV[tr].mean(), LV[tr].std() + 1e-6
    sm = np.array([STI[tr][..., i][MSK[tr]].mean() for i in range(3)], np.float32)
    ssd = np.array([STI[tr][..., i][MSK[tr]].std() + 1e-6 for i in range(3)], np.float32)
    lv = torch.from_numpy((LV - lm) / lsd)
    sti = torch.from_numpy((STI - sm) / ssd)
    stim = torch.from_numpy(MSK)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    torch.manual_seed(SEED)
    g = torch.Generator(device=device).manual_seed(SEED)
    cg = torch.Generator().manual_seed(SEED)
    m = PINet(dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(cfg["epochs"] *
        int(np.ceil(len(tr) / cfg["bs"])), 1), pct_start=0.25)
    wc = torch.tensor([1.0, float((yt[tr] == 0).sum()) / max(int((yt[tr] == 1).sum()), 1)],
                      device=device)
    for ep in range(cfg["epochs"]):
        m.train()
        perm = np.random.default_rng(SEED + ep).permutation(tr)
        for i in range(0, len(perm), cfg["bs"]):
            j = perm[i:i + cfg["bs"]]
            # Training crops avoid the outer MARGIN so that evaluation shifts never
            # reach signal the model could not have seen positioned that way.
            xb = E13.augment(Xt[j][..., MARGIN:-MARGIN].to(device), cfg, g, cg)
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(m(xb), yt[j].to(device), lv[j].to(device),
                              sti[j].to(device), stim[j].to(device), wc,
                              w_lvef=wl, w_sti=ws)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    return m, sm, ssd


def predict(m, X, te, cfg, device, e_off, p_off):
    """3-crop TTA of crop*T' samples; ECG crop shifted by e_off, PCG by p_off samples.
    Returns (P(HFrEF), EMAT head in standardised units, shape (n, 4))."""
    Xt = torch.from_numpy(X)
    T = X.shape[-1] - 2 * MARGIN
    n = int(T * cfg["crop"])
    P = np.zeros(len(te)); E = np.zeros((len(te), X.shape[1]))
    with torch.no_grad():
        for i in range(0, len(te), cfg["bs"]):
            j = te[i:i + cfg["bs"]]
            xb = Xt[j]
            acc, emat = 0.0, 0.0
            for t in range(cfg["tta"]):
                o = MARGIN + int((T - n) * t / max(cfg["tta"] - 1, 1))
                w = torch.stack([xb[:, :, 0, o - e_off:o - e_off + n],
                                 xb[:, :, 1, o - p_off:o - p_off + n]], 2).to(device)
                out = m(w)
                acc = acc + torch.softmax(out["logits"].float(), -1)[:, 1]
                emat = emat + out["sti"][..., 0].float()
            P[i:i + len(j)] = (acc / cfg["tta"]).cpu().numpy()
            E[i:i + len(j)] = (emat / cfg["tta"]).cpu().numpy()
    return P, E


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(E44.CFG)
    if args.smoke:
        cfg["epochs"] = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E44._E12.build_cache(False)
    X, y, fold = d["X"], d["y"], d["fold"]
    LV = (d["lvef"] * 100).astype(np.float32)
    STI = np.nan_to_num(d["sti"].astype(np.float32))
    MSK = d["sti_mask"].astype(bool)
    folds = [0] if args.smoke else range(5)
    conds = {"clean": (0, 0)}
    for s in SHIFTS_MS:
        conds[f"pcg{s:+d}"] = (0, s)           # PCG later by s ms (1 kHz: ms = samples)
    for s in (-80, 80):
        conds[f"joint{s:+d}"] = (s, s)
    res = {"n": int(len(y)), "n_pos": int(y.sum()), "configs": {}}

    for name, (wl, ws, _std) in CONFIGS.items():
        P = {c: np.full(len(y), np.nan) for c in conds}
        E = {c: np.full((len(y), 4), np.nan) for c in conds}
        for f in folds:
            te, tr = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            m, sm, ssd = train(X, y, LV, STI, MSK, tr, cfg, wl, ws, device)
            for c, (eo, po) in conds.items():
                p, e = predict(m, X, te, cfg, device, eo, po)
                P[c][te] = p
                E[c][te] = e * ssd[0] + sm[0]            # back to ms
        k = ~np.isnan(P["clean"])
        yy = y[k]
        pc = P["clean"][k]
        thr = np.sort(pc[yy == 1])[::-1][int(np.ceil(0.9 * yy.sum())) - 1]
        rng = np.random.default_rng(SEED)
        idx = [rng.integers(0, k.sum(), k.sum()) for _ in range(5000)]
        r = {"w_lvef": wl, "w_sti": ws, "clean_auroc": round(float(roc_auc_score(yy, pc)), 4),
             "clean_spe_at_thr": round(float((pc[yy == 0] < thr).mean()), 4), "shifts": {}}
        # EMAT head vs the panel target, where the target exists.
        msk = MSK[k]
        emc = E["clean"][k]
        if ws > 0:
            r["emat_head_vs_panel"] = {
                "mae_ms": round(float(np.abs(emc[msk] - STI[k][..., 0][msk]).mean()), 2),
                "pearson_r": round(float(np.corrcoef(emc[msk], STI[k][..., 0][msk])[0, 1]), 4)}
        print(f"\n=== {name} (w_lvef={wl}, w_sti={ws})  clean AUROC {r['clean_auroc']:.4f}"
              + (f"  EMAT head r={r['emat_head_vs_panel']['pearson_r']:+.3f} "
                 f"MAE {r['emat_head_vs_panel']['mae_ms']:.1f} ms" if ws > 0 else ""))
        for c, (eo, po) in conds.items():
            if c == "clean":
                continue
            p = P[c][k]
            dl = logit(p) - logit(pc)
            rel = po - eo
            row = {"relative_shift_ms": rel, "auroc": round(float(roc_auc_score(yy, p)), 4),
                   "dlogit": round(float(dl.mean()), 3),
                   "dlogit_ci": [round(float(np.percentile([dl[b].mean() for b in idx], q)), 3)
                                 for q in (2.5, 97.5)],
                   "frac_toward_hfref": round(float((dl > 0).mean()), 3),
                   "sen_at_thr": round(float((p[yy == 1] >= thr).mean()), 4),
                   "spe_at_thr": round(float((p[yy == 0] < thr).mean()), 4)}
            if ws > 0:
                # Per subject: mean over sites with a target; bootstrap over subjects.
                tk = (E[c][k] - emc) / rel if rel else (E[c][k] - emc) / eo
                per = np.array([tk[i][msk[i]].mean() for i in range(len(yy)) if msk[i].any()])
                row["emat_tracking" if rel else "emat_joint_per_ms"] = round(float(per.mean()), 4)
                row["emat_ci"] = [round(float(np.percentile(
                    [rng.choice(per, len(per)).mean() for _ in range(2000)], q)), 4)
                    for q in (2.5, 97.5)]
            r["shifts"][c] = row
            print(f"  {c:9s} rel {rel:+4d} ms  AUROC {row['auroc']:.4f}  dlogit {row['dlogit']:+.2f} "
                  f"[{row['dlogit_ci'][0]:+.2f}, {row['dlogit_ci'][1]:+.2f}]  toward HFrEF "
                  f"{100 * row['frac_toward_hfref']:.0f}%  fixed thr sen {100 * row['sen_at_thr']:.0f}% "
                  f"spe {100 * row['spe_at_thr']:.0f}%"
                  + (f"  | EMAT head {'tracking' if rel else 'joint/ms'} "
                     f"{row.get('emat_tracking', row.get('emat_joint_per_ms')):+.3f}" if ws > 0 else ""),
                  flush=True)
        res["configs"][name] = r
    (OUT / f"e49_pinet_timing{'_smoke' if args.smoke else ''}.json").write_text(
        json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)}, indent=2),
        encoding="utf-8")
    print(f"\n({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
