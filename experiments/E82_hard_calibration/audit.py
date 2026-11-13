#!/usr/bin/env python
"""E82 复核（导师实验2 表7 的后三项）：截断计数、残余误差定义、噪声是否真正进入估计器。只重做 E82 的测量部分，不重做分类。

    python experiments/E82_hard_calibration/audit.py --smoke     # 第 1 折、模拟 1–2
    python experiments/E82_hard_calibration/audit.py             # 5 折 x 20 次模拟

1 截断（按“场景 x 方法”、计数单位 = 评分患者的录音位点 (折, 模拟, 患者, 位点)，与训练种子无关，不含重复调用）
  注入截断：注入偏移 |c + J·u| > 290 ms 被截到 ±290 的位点数 / 总位点数（各方法共用同一注入偏移）。
  校正截断：送入分类器的偏移 |注入 − 校正量| > 290 ms 的位点数（A、B、B*、C、D、E、Cr、Dr、Er、H 分别计）。
2 残余误差（第二部分）
  真值一律为该录音实际注入的偏移（共同偏移 + 抖动，截断后）。总残余 = 注入偏移 − 方法的回移量：
  A 回移 0；B 回移共同偏移真值；B* 回移注入偏移本身；C/D/E 回移设备估计 ĉ；Cr/Dr/Er 回移逐录音估计
  g^-1(tau − mu_site)（测量失败退回 ĉ）；H 回移 ĉ_E + Al，Al 为 DO-PINet 对已扣除 ĉ_E 的输入再做的逐录音对齐量。
  逐录音估计（Cr/Dr/Er 与 H 的 Al）同时去掉了该患者生理间期偏离位点中位数的部分，所以总残余里混有生理差异。
  拆分：同一录音在零偏移下的逐录音估计（生理基线 b）即“生理部分”；跟踪残余 = 注入偏移 − (回移量 − b)，
  只反映对注入偏移的跟踪误差。另报生理基线本身的平均绝对值。自检：总残余 MAE 与 E82 summary.json 一致。
3 噪声（第三部分）
  对每个加噪池：全频带实际 SNR（加噪后 − 加噪前 = 噪声，功率比）、20–400 Hz 带内 SNR（C、D 包络所用频带）；
  加噪前后逐录音测量值的平均绝对变化与失败率变化（证明估计器读到的是加噪信号）；设备估计 ĉ 与 E82
  calib_noise.csv 的一致性（证明复核与原运行一致）及与无噪声 ĉ 的差。存一例加噪前后波形与包络图。
输出 experiments/E82_hard_calibration/full/audit/（audit.json、clipping.csv、residual_split.csv、noise_check.csv、noise_example.png）
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as sps
import torch

OUT = Path(__file__).parent
EXP = OUT.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E82 = _load("e82_run", "E82_hard_calibration/run.py")
E81, E52, E50 = E82.E81, E82.E52, E82.E50
FS, CLIP, DEV, JS, SNRS = E82.FS, E82.CLIP, E82.DEV, E82.JS, E82.SNRS
METH = ["A", "B", "B*", "C", "D", "E", "Cr", "Dr", "Er", "H"]


def band_power(x, lo=20.0, hi=400.0):
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    P = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    return P[..., (f >= lo) & (f <= hi)].sum(-1), P[..., 1:].sum(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = E50.E49.E44._E12.build_cache(False)
    X, y, subs = d["X"], d["y"], d["subjects"]
    L = E81.make_lists(y, d["fold"])
    folds = [0] if args.smoke else list(range(5))
    sims = [1, 2] if args.smoke else list(range(1, E82.N_SIM + 1))
    odir = E82.OUT / ("smoke" if args.smoke else "full") / "audit"
    odir.mkdir(parents=True, exist_ok=True)
    cfg81 = json.loads((E82.E81DIR / "config.json").read_text(encoding="utf-8"))
    s82 = json.loads((E82.OUT / "full" / "summary.json").read_text(encoding="utf-8"))
    noise82 = {(int(r["fold"]), int(r["sim"]), int(r["snr_db"]), int(r["K"]), int(r["subset"]), r["method"]): float(r["c_hat_ms"])
               for r in csv.DictReader(open(E82.OUT / "full" / "calib_noise.csv", encoding="utf-8"))}
    D = E82.raw_draws(len(y), sims)
    TR = E81.Traditional(X)
    clip = {}                                                 # (J, method) -> [clipped, total]
    inj = {J: [0, 0] for J in JS}
    res = {}                                                  # (J, method) -> [sum |total|, sum |tracking|, n, n_tracking]
    base_abs = {m: [0.0, 0] for m in ("Cr", "Dr", "Er", "H")}
    nrows, chk = [], {"noise_chat_vs_e82_max_diff_ms": 0.0}
    example_saved = False

    for f in folds:
        tr, pool, sc = L[f]["train"], L[f]["pool"], L[f]["score"]
        TR.prep(np.concatenate([pool, sc]))
        op = E50.DelayOp(FS, lag_min_ms=-200.0, lag_max_ms=300.0).to(device)
        op.load_state_dict(torch.load(E82.E81DIR / "weights" / f"fold{f}_delayop.pt", map_location=device))
        op.eval()
        mu_op, k_op = E50.calibrate(op, X, tr, device)
        curves = {m: E81_curve(cfg81["curves"][str(f)][m]) for m in DEV}

        def m_op(ids, offs, Xa=X):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, _ = E50.op_tau(op, Xa, ix, np.asarray(offs).reshape(-1).astype(int), device)
            return tau.reshape(-1, 4)

        def m_trad(kind, T=TR):
            def m(ids, offs):
                return np.array([[T.tau(i, j, int(offs[a, j]), kind)[0] for j in range(4)] for a, i in enumerate(ids)])
            return m

        meas = {"C": m_trad("C"), "D": m_trad("D"), "E": m_op}

        def al(ids, offs):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, _ = E50.op_tau(op, X, ix, np.asarray(offs).reshape(-1).astype(int), device)
            dh = np.clip((tau.reshape(-1, 4) - mu_op[None]) / k_op, -E52.CLIP_MS, E52.CLIP_MS)
            return np.clip(np.round(dh).astype(int), offs - CLIP, offs + CLIP)

        z = np.zeros((len(sc), 4), int)
        base = {m + "r": E82.per_rec(meas[m](sc, z), *curves[m]) for m in DEV}      # physiological baseline per recording
        base["H"] = al(sc, z).astype(float)
        for m, b in base.items():
            base_abs[m][0] += float(np.nansum(np.abs(b))); base_abs[m][1] += int(np.isfinite(b).sum())

        for k, cvec, u in D:
            c_true = float(cvec[f])
            for J in JS:
                full = c_true + np.round(u * (J / 30)).astype(int)
                offs = np.clip(full, -CLIP, CLIP).astype(int)
                inj[J][0] += int((offs[sc] != full[sc]).sum()); inj[J][1] += offs[sc].size
                o = offs[sc]
                corr, phys = {"A": np.zeros_like(o, float), "B": np.full(o.shape, c_true), "B*": o.astype(float)}, {}
                for m in DEV:
                    ch = E82.est(meas[m](pool, offs[pool]), *curves[m])[0]
                    corr[m] = np.full(o.shape, ch)
                    pr = E82.per_rec(meas[m](sc, o), *curves[m])
                    corr[m + "r"] = np.where(np.isnan(pr), ch, pr)
                    phys[m + "r"] = np.where(np.isnan(pr), np.nan, pr - base[m + "r"])     # tracking part only
                shift = o - int(round(corr["E"][0, 0]))
                A_l = al(sc, shift)
                corr["H"] = corr["E"] + A_l
                phys["H"] = corr["E"] + (A_l - base["H"])
                for m in METH:
                    fed = o - np.round(corr[m]).astype(int) if m != "H" else shift
                    c = clip.setdefault((J, m), [0, 0]); c[0] += int((np.abs(fed) > CLIP).sum()); c[1] += fed.size
                    tot = o - np.round(corr[m]).astype(int) if m != "H" else shift - A_l
                    trk = o - phys[m] if m in phys else tot
                    r = res.setdefault((J, m), [0.0, 0.0, 0, 0])
                    r[0] += float(np.abs(tot).sum()); r[2] += tot.size
                    ok = np.isfinite(trk)
                    r[1] += float(np.abs(trk[ok]).sum()); r[3] += int(ok.sum())

            # ---- noise: does it reach the estimators?
            offs_p = np.clip(c_true + np.round(u).astype(int), -CLIP, CLIP).astype(int)[pool]
            clean = {m: meas[m](pool, offs_p) for m in DEV}
            ch_clean = {m: E82.est(clean[m], *curves[m])[0] for m in DEV}
            for snr in SNRS:
                Xp = E82.noisy_pool(X, pool, f, k, snr)
                nz = Xp[:, :, 1] - X[pool][:, :, 1]
                inb_s, all_s = band_power(X[pool][:, :, 1]); inb_n, all_n = band_power(nz)
                Tn = E81.Traditional(Xp)
                for a, i in enumerate(pool):
                    for j in range(4):
                        Tn.R[(a, j)] = TR.R[(i, j)]
                        e = E81._env(Xp[a, j, 1].astype(float), FS, E81.BAND_S12)
                        Tn.ENV[(a, j)] = ((e - e.mean()) / (e.std() + 1e-9)).astype(np.float32)
                loc = np.arange(len(pool))
                noisy = {"C": m_trad("C", Tn)(loc, offs_p), "D": m_trad("D", Tn)(loc, offs_p), "E": m_op(loc, offs_p, Xp)}
                for m in DEV:
                    a_, b_ = clean[m], noisy[m]
                    both = np.isfinite(a_) & np.isfinite(b_)
                    ch = E82.est(b_, *curves[m])[0]
                    chk["noise_chat_vs_e82_max_diff_ms"] = max(chk["noise_chat_vs_e82_max_diff_ms"],
                                                               abs(ch - noise82[(f, k, snr, 20, 0, m)]))
                    nrows.append({"fold": f, "sim": k, "snr_db": snr, "method": m,
                                  "snr_full_db": round(float(10 * np.log10(all_s.sum() / all_n.sum())), 2),
                                  "snr_20_400_db": round(float(10 * np.log10(inb_s.sum() / inb_n.sum())), 2),
                                  "mean_abs_change_ms": round(float(np.abs(a_[both] - b_[both]).mean()), 3),
                                  "frac_changed_gt1ms": round(float((np.abs(a_[both] - b_[both]) > 1).mean()), 4),
                                  "fail_rate_clean": round(float(np.isnan(a_).mean()), 4),
                                  "fail_rate_noisy": round(float(np.isnan(b_).mean()), 4),
                                  "c_hat_clean": round(ch_clean[m], 3), "c_hat_noisy": round(ch, 3)})
                if not example_saved and snr == 0:
                    save_example(X[pool[0], 0, 1], {s_: E82.noisy_pool(X, pool, f, k, s_)[0, 0, 1] for s_ in (10, 0)}, odir)
                    example_saved = True
            print(f"  fold {f} sim {k} ({time.time() - t0:.0f}s)", flush=True)

    # ---- outputs
    with open(odir / "clipping.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["J_ms", "method", "injected_clipped", "fed_clipped", "total_site_recordings"])
        for J in JS:
            for m in METH:
                w.writerow([J, m, inj[J][0], clip[(J, m)][0], clip[(J, m)][1]])
    split = {}
    with open(odir / "residual_split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["J_ms", "method", "total_residual_mae_ms", "tracking_residual_mae_ms", "n", "n_tracking"])
        for J in JS:
            for m in METH:
                r = res[(J, m)]
                split[f"{J}:{m}"] = {"total": r[0] / r[2], "tracking": r[1] / r[3] if r[3] else None}
                w.writerow([J, m, round(r[0] / r[2], 3), round(r[1] / r[3], 3) if r[3] else "", r[2], r[3]])
                if not args.smoke:
                    chk["residual_vs_e82_max_diff_ms"] = max(chk.get("residual_vs_e82_max_diff_ms", 0.0),
                                                             abs(r[0] / r[2] - s82["part2"][str(J)][m]["residual_mae_ms"]))
    with open(odir / "noise_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nrows[0].keys())); w.writeheader(); w.writerows(nrows)
    noise = {}
    for snr in SNRS:
        for m in DEV:
            rr = [r for r in nrows if r["snr_db"] == snr and r["method"] == m]
            noise[f"{snr}:{m}"] = {kk: float(np.mean([r[kk] for r in rr])) for kk in
                                   ("snr_full_db", "snr_20_400_db", "mean_abs_change_ms", "frac_changed_gt1ms",
                                    "fail_rate_clean", "fail_rate_noisy")}
            noise[f"{snr}:{m}"]["mean_abs_chat_change_ms"] = float(np.mean([abs(r["c_hat_noisy"] - r["c_hat_clean"]) for r in rr]))
    A = {"folds": folds, "sims": sims, "unit": "scored patient x site recording per (fold, sim); seed independent",
         "injected_clipping": {J: inj[J] for J in JS}, "fed_clipping": {f"{J}:{m}": clip[(J, m)] for J in JS for m in METH},
         "physiological_baseline_mae_ms": {m: v[0] / v[1] for m, v in base_abs.items()},
         "residual_split": split, "noise": noise, "self_checks": chk, "elapsed_min": round((time.time() - t0) / 60, 1)}
    (odir / "audit.json").write_text(json.dumps(A, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: A[k] for k in ("injected_clipping", "physiological_baseline_mae_ms", "self_checks")}, default=float))
    for J in JS:
        print(J, {m: (round(split[f'{J}:{m}']['total'], 1), None if split[f'{J}:{m}']['tracking'] is None
                      else round(split[f'{J}:{m}']['tracking'], 1), clip[(J, m)][0]) for m in METH})
    for k_, v in noise.items():
        print(k_, {kk: round(vv, 3) for kk, vv in v.items()})
    return 0


def E81_curve(c):
    mu, gt = np.array(c["mu_site"]), np.array(c["g"])
    return mu, gt, np.concatenate([[True], np.diff(np.maximum.accumulate(gt)) > 0])


def save_example(clean, noisy, odir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = slice(10000, 12000)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(np.arange(2000), clean[t], lw=0.6, label="clean")
    for s_, x in noisy.items():
        ax[0].plot(np.arange(2000), x[t], lw=0.4, alpha=0.6, label=f"SNR {s_} dB")
    ax[0].set_ylabel("PCG (z)"); ax[0].legend(fontsize=7)
    ax[1].plot(np.arange(2000), E81._env(clean.astype(float), FS, E81.BAND_S12)[t], label="clean")
    for s_, x in noisy.items():
        ax[1].plot(np.arange(2000), E81._env(x.astype(float), FS, E81.BAND_S12)[t], lw=0.8, label=f"SNR {s_} dB")
    ax[1].set_ylabel("20-400 Hz envelope"); ax[1].set_xlabel("ms (10-12 s of the buffer)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(odir / "noise_example.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
