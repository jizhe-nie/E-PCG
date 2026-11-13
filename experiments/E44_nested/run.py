#!/usr/bin/env python
"""E44 -- 嵌套交叉验证选辅助损失权重，得到可发表的数字。

    python experiments/E44_nested/run.py --smoke
    python experiments/E44_nested/run.py

E42 和 E43 发现辅助监督能把 @90% 敏感度的特异度从 85% 抬到 90-92%，但那些权重都是
看着测试折的指标挑的，不能进论文。这里用嵌套交叉验证得到诚实的数字：

    外层   数据集作者的五折，只用于报告
    内层   在每个外层训练集内部再做四折，用它选 (w_lvef, w_sti, 是否标准化)
    然后   用内层选出的配置在整个外层训练集上重训，在外层测试折上预测

内层的选择准则是 @90% 敏感度的特异度，和最终报告的指标一致 —— 用 AUROC 选而用操作点
报告，正是 E12 当年把有效配置判成零结果的原因。

网格里同时包含标准化与未标准化两种目标尺度，因为它们不是同一个目标函数：smooth_l1
在原始尺度（标准差 13-39）上主要工作在线性区，对离群值稳健；标准化后主要在二次区。
E42/E43 里未标准化版本反而更好，所以这是一个真实的超参数而不是实现细节。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from epcg.models.pinet import PINet, loss_fn  # noqa: E402

OUT = Path(__file__).parent
SEED = 20260912
CFG = dict(dim=128, dropout=0.3, lr=3e-4, wd=1e-2, bs=16, epochs=40, crop=0.7,
           amp=0.3, site_drop=0.25, noise=0.05, smooth=0.05, tta=3)
# (w_lvef, w_sti, standardise)
GRID = [(0.0, 0.0, True), (3.0, 0.0, True), (10.0, 3.0, True),
        (0.0, 3.0, True), (1.0, 0.0, False), (3.0, 0.0, False)]


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_E12 = _load("e12_run", "E12_pinet/run.py")
_E13 = _load("e13_run", "E13_tune/run.py")


def fit_predict(X, y, LV, STI, MSK, tr, te, cfg, wl, ws, std, device):
    """在 tr 上训练，返回 te 上的概率。标准化统计量只来自 tr。"""
    if std:
        lm, lsd = LV[tr].mean(), LV[tr].std() + 1e-6
        sm = np.array([STI[tr][..., i][MSK[tr]].mean() for i in range(3)], np.float32)
        ssd = np.array([STI[tr][..., i][MSK[tr]].std() + 1e-6 for i in range(3)],
                       np.float32)
    else:
        lm, lsd, sm, ssd = 0.0, 1.0, np.zeros(3, np.float32), np.ones(3, np.float32)
    lv = torch.from_numpy((LV - lm) / lsd)
    sti = torch.from_numpy((STI - sm) / ssd)
    stim = torch.from_numpy(MSK)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y).long()

    torch.manual_seed(SEED)
    g = torch.Generator(device=device).manual_seed(SEED)
    cg = torch.Generator().manual_seed(SEED)
    m = PINet(dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=max(cfg["epochs"] *
        int(np.ceil(len(tr) / cfg["bs"])), 1), pct_start=0.25)
    wc = torch.tensor([1.0, float((yt[tr] == 0).sum())
                       / max(int((yt[tr] == 1).sum()), 1)], device=device)
    for ep in range(cfg["epochs"]):
        m.train()
        perm = np.random.default_rng(SEED + ep).permutation(tr)
        for i in range(0, len(perm), cfg["bs"]):
            j = perm[i:i + cfg["bs"]]
            xb = _E13.augment(Xt[j].to(device), cfg, g, cg)
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_fn(m(xb), yt[j].to(device), lv[j].to(device),
                              sti[j].to(device), stim[j].to(device), wc,
                              w_lvef=wl, w_sti=ws)
            loss.backward(); opt.step(); sch.step()
    m.eval()
    P = np.zeros(len(te))
    with torch.no_grad():
        for i in range(0, len(te), cfg["bs"]):
            j = te[i:i + cfg["bs"]]
            xb = Xt[j].to(device)
            T = xb.shape[-1]; n = int(T * cfg["crop"]); acc = 0.0
            for t in range(cfg["tta"]):
                o = int((T - n) * t / max(cfg["tta"] - 1, 1))
                acc = acc + torch.softmax(
                    m(xb[..., o:o + n])["logits"].float(), -1)[:, 1]
            P[i:i + len(j)] = (acc / cfg["tta"]).cpu().numpy()
    return P


def spe_at(y, p, target=0.90):
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(y, p)
    i = np.argmin(np.abs(tpr - target))
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr[i]).astype(int),
                                      labels=[0, 1]).ravel()
    return tn / max(tn + fp, 1)


def report(y, p, target=0.90):
    fpr, tpr, thr = roc_curve(y, p)
    i = np.argmin(np.abs(tpr - target))
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr[i]).astype(int),
                                      labels=[0, 1]).ravel()
    return {"auroc": round(float(roc_auc_score(y, p)), 4),
            "sen": round(tp / max(tp + fn, 1), 4),
            "spe": round(tn / max(tn + fp, 1), 4),
            "ppv": round(tp / max(tp + fp, 1), 4),
            "fp": int(fp), "fn": int(fn)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = dict(CFG)
    grid = GRID[:2] if args.smoke else GRID
    if args.smoke:
        cfg["epochs"], args.inner = 3, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    d = _E12.build_cache(False)
    y, fold = d["y"], d["fold"]
    LV = (d["lvef"] * 100).astype(np.float32)
    STI = np.nan_to_num(d["sti"].astype(np.float32))
    MSK = d["sti_mask"].astype(bool)
    X = d["X"]
    print(f"{len(y)} 名受试者，{int(y.sum())} 例心衰，device={device}")
    print(f"外层 5 折 x 内层 {args.inner} 折 x {len(grid)} 组配置\n")

    P = np.zeros(len(y))
    picks = []
    outer = [0] if args.smoke else range(5)
    for f in outer:
        te = np.flatnonzero(fold == f); tr = np.flatnonzero(fold != f)
        # 内层：只在外层训练集里划分，用它选配置
        inner = list(StratifiedKFold(args.inner, shuffle=True,
                                     random_state=SEED).split(tr, y[tr]))
        scores = []
        for wl, ws, std in grid:
            s = []
            for a_, b_ in inner:
                ia, ib = tr[a_], tr[b_]
                p = fit_predict(X, y, LV, STI, MSK, ia, ib, cfg, wl, ws, std, device)
                s.append(spe_at(y[ib], p))
            scores.append(float(np.nanmean(s)))
            print(f"  折{f} 内层  w_lvef={wl:<5g} w_sti={ws:<4g} std={str(std):5s}"
                  f"  特异度 {scores[-1]:.4f}   ({time.time() - t0:.0f}s)")
        k = int(np.argmax(scores))
        wl, ws, std = grid[k]
        picks.append({"fold": f, "w_lvef": wl, "w_sti": ws, "std": std,
                      "inner_spe": round(scores[k], 4)})
        print(f"  折{f} 选中 w_lvef={wl} w_sti={ws} std={std}\n")
        P[te] = fit_predict(X, y, LV, STI, MSK, tr, te, cfg, wl, ws, std, device)

    m = np.ones(len(y), bool) if not args.smoke else (fold == 0)
    res = {"n": int(m.sum()), "n_pos": int(y[m].sum()), "picks": picks,
           "nested": report(y[m], P[m]),
           "reference": {"baseline_no_aux": {"spe": 0.852, "ppv": 0.458},
                         "tuned_on_test_best": {"spe": 0.918, "ppv": 0.600},
                         "lgc2net": {"spe": 0.927, "ppv": 0.628}}}
    print("=" * 62)
    r = res["nested"]
    print(f"嵌套交叉验证结果（可发表）")
    print(f"  AUROC {r['auroc']:.4f}   敏感度 {100 * r['sen']:.1f}%   "
          f"特异度 {100 * r['spe']:.1f}%   PPV {100 * r['ppv']:.1f}%")
    print(f"  假阳 {r['fp']}   漏诊 {r['fn']}")
    print(f"\n  对照  无辅助基线 85.2% / 45.8%")
    print(f"        测试折调参上界 91.8% / 60.0%（乐观，不可发表）")
    print(f"        LGC2-Net      92.7% / 62.8%")
    print(f"\n  各折选中的配置：")
    for p_ in picks:
        print(f"    折{p_['fold']}  w_lvef={p_['w_lvef']} w_sti={p_['w_sti']} "
              f"std={p_['std']}  内层特异度 {p_['inner_spe']:.3f}")

    # 逐样本折外预测必须落盘：报告一个点估计而没有区间，正是这个项目反复栽跟头的地方。
    np.save(OUT / "e44_oof.npy", P)
    (OUT / "e44_nested.json").write_text(
        json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)},
                   indent=2, default=str), encoding="utf-8")
    print(f"\n写入 {OUT / 'e44_nested.json'}  ({(time.time() - t0) / 60:.1f} 分钟)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
