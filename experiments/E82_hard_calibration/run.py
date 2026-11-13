#!/usr/bin/env python
"""E82 -- 实验1补充：在更难的条件下比较时延估计器（预设方案，2026-10-31 起草，跑之前锁定）。

    python experiments/E82_hard_calibration/run.py --smoke    # 1 折、1 个种子：流程与自检
    python experiments/E82_hard_calibration/run.py            # 5 折 x 种子 0/1/2

定位。E82 是看到 E81 结果之后提出的事后补充、探索性分析。E81（实验1）的表格保持原样上交，不改动；
E82 的结果以“补充表 S1–S4”的形式附在实验1表格之后，并注明“事后补充”。论文中同样如此标注。

动机（来自 E81）。S2 中已知真实偏移的 B 的 MCC 为 0.526，C、D、E 均为 0.53 左右，已到上限，任何估计器都不可能在
该条件下显出优势。同时，传统方法在校准池中频繁测量失败（failures.log：C 涉及 302 名受试者、D 涉及 69 名，E 为 0），
20 人取中位数吸收了这些失败。E82 检验三种 E81 没有覆盖的条件：校准池很小、逐录音时延变化大、PCG 有噪声。

不改动的部分（全部沿用 E81，不重新训练、不重新选阈值）
  患者名单 lists.json（训练/验证/校准池/评分）；E81 保存的权重（每折每种子的 pinet 与 dual，每折一个时延算子）；
  E81 config.json 中锁定的阈值（A–E 及其逐录音变体用 A 的阈值，H 用 I 的阈值）；C/D 的测量器与失败规则；
  响应曲线 g 与 mu_site 的建立方法（训练患者、无标签）；越界截断与整池失败的计数规则；S2 的共同偏移抽样（种子 20261030+k）。

第一部分  校准池人数（问题：校准池很小时，哪种估计器更稳？）
  场景：E81 的 S2（20 次模拟）。K ∈ {1, 2, 3, 5, 10, 20}。K < 20 时，从每折 E81 的 20 人校准池中无放回抽取 50 个子池
  （种子 20261031，各方法、各模拟共用同一批子池，即配对比较）；K = 20 即 E81 的原校准池。估计规则与 E81 完全相同。
  主要终点（只看估计，不需推理）：|c_hat - c_true| 的平均绝对误差（MAE）与第 90 百分位数；失准率 = |误差| > 40 ms 的比例
  （40 ms 为 S1 中 80 ms 偏移的一半，取作保守容差）；整池失败率（失败即退回不校准，计入误差）；越界率。
  次要终点：K ∈ {1, 3, 20} 时 C/D/E 校准后 PINet 的敏感度、特异度、AUROC、MCC（K < 20 只用前 10 个子池）。
  判据 P1（“学习型算子在小校准池下更稳”）：K = 1 与 K = 3 时同时满足
    (a) E 的 MAE 比 C 和 D 都低至少 5 ms，且配对差的 95% 自助法置信区间不含 0
        （以“折 x 模拟”的 100 个单元为整群重抽样，B = 2000，种子 0）；
    (b) E 的整池失败率不高于 C 和 D。
  只满足 (b) 时报告为“E 不会测量失败，但精度没有优势”。下游指标只作描述，不另设判据。

第二部分  逐录音时延变化（问题：设备内每段录音的时延各不相同时，逐录音估计是否必要？哪种估计器更好？）
  场景 S3：共同偏移与 S2 相同；每个“患者 x 位点”录音另加抖动 J * u，u 由 E81 S2 同一随机流中未取整的 U(-30, 30) 抽样
  除以 30 得到（因而 J = 30 与 E81 的 S2 完全相同，各抖动幅度之间嵌套），J ∈ {30, 60, 100, 150} ms，取整到 1 ms。
  方法（除 H 外均为 E81 的 PINet 权重与 A 的阈值）：
    A 不校准；B 去掉共同偏移（参照）；B* 去掉全部注入偏移（逐录音理想上限，保留患者自身的生理时延）；
    C/D/E 设备级校准（K = 20，同 E81）；
    Cr/Dr/Er 逐录音校准：每段录音用自身测量值经 g^{-1} 反查得到的偏移校正（截断到曲线范围）；该录音测量失败时退回
    同方法的设备级估计（计数报告）；
    H 完整方法（E 的设备级估计 + 双模式 DO-PINet 开模式，同 E81）。
  已知风险（预先写明）：逐录音校正会把患者自身偏离 mu_site 的生理时延一并拉回，论文“不可分离性”一节及已做的逐录音
  校正尝试显示这必然以敏感度换特异度。B* 与 Cr/Dr/Er 的差距就是这部分代价。
  终点：每段录音的残余偏移（施加的校正与注入偏移之差）的 MAE；敏感度、特异度、AUROC、MCC。
  判据 P2a（“该场景需要逐录音校正”）：J = 100 时 MCC(B*) - MCC(B) ≥ 0.05。不满足则报告“在该抖动范围内设备级校准
    已足够”，不再据第二部分对估计器下任何结论。
  判据 P2b（仅在 P2a 成立时检验，“学习型算子逐录音跟踪优于传统方法”）：J = 100 与 J = 150 时同时满足
    (a) Er 的残余 MAE 比 Cr 和 Dr 都低至少 5 ms；
    (b) MCC(Er) - max(MCC(Cr), MCC(Dr)) ≥ 0.02，配对差的 95% 置信区间（experiments/_pairstats.py：按受试者自助重抽样，
        B = 2000，种子 0，各种子共用同一重抽样）不含 0。
  另外如实报告 Er 与 E、H 与 Er、H 与 E 的差值，不设判据。

第三部分  PCG 噪声（问题：噪声下哪种估计器的设备级估计更准？）
  场景：S2 的偏移；校准池录音的 PCG 加高斯白噪声，信噪比 SNR ∈ {无噪声, 20, 10, 5, 0} dB（相对于该段 PCG 自身功率，
  噪声种子 20261031 + 模拟号）；ECG 不加噪声；响应曲线仍用干净的训练数据建立（与部署时一致）。K ∈ {3, 20}
  （K = 3 用第一部分的前 10 个子池）。
  只评估估计：MAE、失准率、整池失败率、越界率。不评估下游分类，原因是 E79 已显示 PCG 噪声本身会使分类器特异度下降
  28–34 个百分点，此时比较校准的好坏没有意义；0 dB 只作描述（实际应由信号质量门控拒收）。
  判据 P3（“学习型算子抗噪更好”）：SNR = 10 与 5 dB、K = 3 与 20 四种组合下同时满足：E 的 MAE 比 C 和 D 都低至少
    5 ms 且置信区间（同第一部分）不含 0，E 的整池失败率不高于 C 和 D。

自检（任一不通过即停止，不报告结果）
  1. 重建的响应曲线与 E81 config.json 一致（最大差 < 0.01 ms）；
  2. 第一部分 K = 20 的 C/D/E 估计值与 E81 calibration.csv 一致（最大差 < 0.01 ms）；
  3. 第二部分 J = 30 时 A、B、C、D、E、H 的逐患者概率与 E81 predictions.csv 中 S2 的结果一致（最大差 < 1e-5）；
  4. B* 去掉全部注入偏移后输入即为干净信号，其逐患者概率须与 E81 S0 中 A 的结果一致（最大差 < 1e-5）。

报告规则
  除 P1、P2a、P2b、P3 外不作其他“优于”结论；每条判据报告“成立/不成立”及全部数字，结果不利也如实报告；
  任何一条判据不成立都不改方法、不改阈值重跑。E81 中 H 在干净数据上不如卷积基线（S0：MCC 0.540 对 0.592）
  这一发现 E82 不处理，在论文中如实写明。

实现中的补充说明（导师批准后、正式运行前写入，2026-10-31）
  1. 分类器读取 PCG 的缓冲只允许 |PCG 偏移| ≤ 300 ms；J = 150 时注入偏移最高可达 350 ms，逐录音校正在测量严重失准时
     也可能超出。处理：注入偏移与送入分类器的校正后偏移均截断到 ±290 ms（与 DO-PINet 已有的截断一致），截断次数报告。
     J ≤ 100 时注入偏移不会超出（最大 300 ms 以内），J = 30 与 E81 完全相同。
  2. 第三部分的噪声随机数种子为 (20261031 + 模拟号, 折号, 信噪比) 的组合，保证各折、各信噪比独立且可复现。
  3. 预测结果按（网络, 送入的偏移矩阵）缓存，相同输入只推理一次，不影响结果。
  4. 置信区间：第一、三部分按“折 x 模拟”整群重抽样；第二部分按受试者重抽样（_pairstats.boots），差值为
     “各种子内先对 20 次模拟取平均、再对种子取平均”的 MCC 差，与表中数字的估计量一致。

输出（experiments/E82_hard_calibration/full/）
  calib_K.csv（第一部分逐子池估计）、calib_noise.csv（第三部分）、residual.csv（第二部分逐录音残余偏移）、
  predictions.csv、confusion.csv、summary.json（含判据结论与自检）、failures.log。
  补充表由 scripts/fill_e82_supplement.py 从 summary.json 自动填入实验1表格之后。
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
import torch

OUT = Path(__file__).parent
EXP = OUT.parent
ROOT = EXP.parent
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E81 = _load("e81_run", "E81_controlled/run.py")
PS = _load("pairstats", "_pairstats.py")
E52, E50 = E81.E52, E81.E50
GC, LIST_SEED, N_SIM, FS = E81.GC, E81.LIST_SEED, E81.N_SIM, E81.FS
E81DIR = EXP / "E81_controlled" / "full"
KS = [1, 2, 3, 5, 10, 20]
N_SUB, N_SUB_DOWN, SUB_SEED = 50, 10, 20261031
K_DOWN = [1, 3]
JS = [30, 60, 100, 150]
SNRS = [20, 10, 5, 0]                     # plus the clean measurement (reported as "inf")
CLIP, TOL = 290, 40
DEV = ["C", "D", "E"]
P2_METHODS = ["A", "B", "B*", "C", "D", "E", "Cr", "Dr", "Er", "H"]


# ----------------------------------------------------------------------------- helpers
def raw_draws(n, sims):
    """E81 S2's random stream, jitter kept unrounded so that J * u / 30 nests the levels (J = 30 -> E81 S2)."""
    out = []
    for k in sims:
        r = np.random.default_rng(LIST_SEED + k)
        c = np.round(r.uniform(40, 200, 5))
        out.append((k, c, r.uniform(-30, 30, (n, 4))))
    return out


def est(t, mu, gt, keep):
    """E81's device rule on precomputed measurements t (m, 4) -> (c_hat, out_of_range, pool_failed)."""
    v = (t - mu[None]).ravel()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, False, True
    med = float(np.median(v))
    lo, hi = gt[keep].min(), gt[keep].max()
    return float(np.interp(med, gt[keep], GC[keep])), bool(med < lo or med > hi), False


def per_rec(t, mu, gt, keep):
    """per-recording offset g^-1(tau - mu_site), clamped to the curve range; nan where the measurement failed."""
    o = np.full(t.shape, np.nan)
    ok = np.isfinite(t)
    o[ok] = np.interp((t - mu[None])[ok], gt[keep], GC[keep])
    return o


def mcc_vec(pred, yy):
    tp = int((pred & (yy == 1)).sum()); fn = int((~pred & (yy == 1)).sum())
    tn = int((~pred & (yy == 0)).sum()); fp = int((pred & (yy == 0)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / den if den else 0.0


def cell_boot(a, b, B=2000, seed=0):
    """paired difference mean(a - b) over cells with a percentile bootstrap over cells."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def noisy_pool(X, pool, f, k, snr):
    """copy of the pool patients' recordings with white Gaussian noise added to the PCG at the given SNR."""
    Xp = X[pool].astype(np.float32, copy=True)
    rng = np.random.default_rng([SUB_SEED + k, f, int(snr)])
    for a in range(len(pool)):
        for j in range(4):
            x = Xp[a, j, 1]
            p = float(np.mean((x - x.mean()) ** 2))
            Xp[a, j, 1] = x + rng.normal(0.0, np.sqrt(p / 10 ** (snr / 10)), x.shape).astype(np.float32)
    return Xp


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    cfg = dict(E50.E49.E44.CFG)
    d = E50.E49.E44._E12.build_cache(False)
    X, y, subs = d["X"], d["y"], d["subjects"]
    L = E81.make_lists(y, d["fold"])
    folds = [0] if args.smoke else list(range(5))
    seeds = E81.TRAIN_SEEDS[:1] if args.smoke else E81.TRAIN_SEEDS
    sims = list(range(1, 3)) if args.smoke else list(range(1, N_SIM + 1))
    odir = OUT / ("smoke" if args.smoke else "full")
    odir.mkdir(exist_ok=True)
    W = E81DIR / "weights"
    cfg81 = json.loads((E81DIR / "config.json").read_text(encoding="utf-8"))
    # E81 references for the self-checks
    cal81 = {(int(r["fold"]), int(r["sim"]), r["method"]): float(r["c_hat_ms"])
             for r in csv.DictReader(open(E81DIR / "calibration.csv", encoding="utf-8"))
             if r["scenario"] == "S2" and r["seed"] == "0"}
    pr81 = {}
    for r in csv.DictReader(open(E81DIR / "predictions.csv", encoding="utf-8")):
        if r["scenario"] == "S2" or (r["scenario"] == "S0" and r["method"] == "A"):
            pr81[(r["patient_id"], int(r["seed"]), r["scenario"], int(r["sim"]), r["method"])] = float(r["prob"])
    D = raw_draws(len(y), sims)
    sub_rng = np.random.default_rng(SUB_SEED)
    SUBS = {f: {K: [np.sort(sub_rng.choice(E81.N_POOL, K, replace=False)) for _ in range(N_SUB)] for K in KS if K < 20}
            for f in range(5)}
    for f in range(5):
        SUBS[f][20] = [np.arange(E81.N_POOL)]
    TR = E81.Traditional(X)
    fail = {}                                         # (part, method) -> [failed, total] site measurements
    flog = open(odir / "failures.log", "w", encoding="utf-8")
    chk = {"curves_max_diff_ms": 0.0, "K20_chat_max_diff_ms": 0.0, "J30_prob_max_diff": 0.0, "Bstar_vs_S0A_max_diff": 0.0}
    rows_K, rows_noise, rows_res = [], [], []
    clip_n = {"injected": 0, "fed": 0, "total": 0}
    fallback = {m: [0, 0] for m in ("Cr", "Dr", "Er")}
    P = {}                                            # key -> (probs over all patients, thresholds)

    def store(key, sc, p, t):
        if key not in P:
            P[key] = (np.full(len(y), np.nan), np.full(len(y), np.nan))
        P[key][0][sc] = p
        P[key][1][sc] = t

    for f in folds:
        tr, pool, sc = L[f]["train"], L[f]["pool"], L[f]["score"]
        TR.prep(np.concatenate([tr, pool, sc]))
        op = E50.DelayOp(FS, lag_min_ms=-200.0, lag_max_ms=300.0).to(device)
        op.load_state_dict(torch.load(W / f"fold{f}_delayop.pt", map_location=device))
        op.eval()
        mu_op, k_op = E50.calibrate(op, X, tr, device)

        def m_op(ids, offs, Xa=X):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, _ = E50.op_tau(op, Xa, ix, np.asarray(offs).reshape(-1).astype(int), device)
            return tau.reshape(-1, 4)

        def m_trad(kind, part, T=TR):
            def m(ids, offs):
                out = np.full((len(ids), 4), np.nan)
                for a, i in enumerate(ids):
                    for j in range(4):
                        t, why = T.tau(i, j, int(offs[a, j]), kind)
                        out[a, j] = t
                        c = fail.setdefault((part, kind), [0, 0]); c[1] += 1
                        if why:
                            c[0] += 1
                            if part != "curve":
                                who = subs[i] if T is TR else subs[pool[i]]
                                flog.write(f"{part} fold {f} method {kind} subject {who} site {j} offset {int(offs[a, j])}: {why}\n")
                return out
            return m

        curves = {"C": E81.build_curve(m_trad("C", "curve"), tr), "D": E81.build_curve(m_trad("D", "curve"), tr),
                  "E": E81.build_curve(m_op, tr)}
        for m in DEV:
            ref = cfg81["curves"][str(f)][m]
            chk["curves_max_diff_ms"] = max(chk["curves_max_diff_ms"],
                                            float(np.max(np.abs(curves[m][1] - np.array(ref["g"])))),
                                            float(np.max(np.abs(curves[m][0] - np.array(ref["mu_site"])))))
        meas = {"C": lambda ids, o, part: m_trad("C", part)(ids, o), "D": lambda ids, o, part: m_trad("D", part)(ids, o),
                "E": lambda ids, o, part: m_op(ids, o)}

        # ---- measurements (seed independent)
        CH, PR, OFFS = {}, {}, {}                     # device estimates, per-recording estimates, injected offsets
        for k, cvec, u in D:
            c_true = float(cvec[f])
            for J in JS:
                full = c_true + np.round(u * (J / 30)).astype(int)
                offs = np.clip(full, -CLIP, CLIP).astype(int)
                clip_n["injected"] += int((offs[sc] != full[sc]).sum()); clip_n["total"] += offs[sc].size
                OFFS[(k, J)] = offs
                for m in DEV:
                    tp = meas[m](pool, offs[pool], "pool")
                    ch, oor, pf = est(tp, *curves[m])
                    CH[(k, J, m)] = ch
                    if J == 30:                       # part 1: pool size
                        for K in KS:
                            for r, sub in enumerate(SUBS[f][K]):
                                chs, oors, pfs = est(tp[sub], *curves[m])
                                CH[(k, "K", K, r, m)] = chs
                                rows_K.append({"fold": f, "sim": k, "K": K, "subset": r, "method": m, "c_true_ms": c_true,
                                               "c_hat_ms": round(chs, 3), "out_of_range": int(oors), "pool_failed": int(pfs)})
                        chk["K20_chat_max_diff_ms"] = max(chk["K20_chat_max_diff_ms"], abs(ch - cal81[(f, k, m)]))
                    ts = meas[m](sc, offs[sc], f"rec_J{J}")
                    o = per_rec(ts, *curves[m])
                    fallback[m + "r"][0] += int(np.isnan(o).sum()); fallback[m + "r"][1] += o.size
                    PR[(k, J, m)] = np.where(np.isnan(o), ch, o)
            # part 3: noise on the pool's PCG (S2 offsets, J = 30)
            offs_p = OFFS[(k, 30)][pool]
            for snr in SNRS:
                Xp = noisy_pool(X, pool, f, k, snr)
                Tn = E81.Traditional(Xp)
                for a, i in enumerate(pool):                  # ECG (R peaks) unchanged; envelope of the noisy PCG
                    for j in range(4):
                        Tn.R[(a, j)] = TR.R[(i, j)]
                        e = E81._env(Xp[a, j, 1].astype(float), FS, E81.BAND_S12)
                        Tn.ENV[(a, j)] = ((e - e.mean()) / (e.std() + 1e-9)).astype(np.float32)
                local = np.arange(len(pool))
                tn = {"C": m_trad("C", f"noise{snr}", Tn)(local, offs_p), "D": m_trad("D", f"noise{snr}", Tn)(local, offs_p),
                      "E": m_op(local, offs_p, Xp)}
                for m in DEV:
                    for K, subl in ((3, SUBS[f][3][:N_SUB_DOWN]), (20, SUBS[f][20])):
                        for r, sub in enumerate(subl):
                            chs, oors, pfs = est(tn[m][sub], *curves[m])
                            rows_noise.append({"fold": f, "sim": k, "snr_db": snr, "K": K, "subset": r, "method": m,
                                               "c_true_ms": c_true, "c_hat_ms": round(chs, 3), "out_of_range": int(oors),
                                               "pool_failed": int(pfs)})
            print(f"  fold {f} sim {k} measured ({time.time() - t0:.0f}s)", flush=True)

        # ---- DO-PINet timing features (as in E81)
        def measure_dh(ids, offs):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, sg = E50.op_tau(op, X, ix, np.asarray(offs).reshape(-1).astype(int), device)
            return np.clip((tau.reshape(-1, 4) - mu_op[None]) / k_op, -E52.CLIP_MS, E52.CLIP_MS), sg.reshape(-1, 4)

        dh_tr, sg_tr = measure_dh(tr, np.zeros((len(tr), 4), int))
        fm = np.array([dh_tr.mean(), sg_tr.mean()]); fsd = np.array([dh_tr.std(), sg_tr.std()]) + 1e-6
        feat = lambda dh, sg: ((np.stack([dh, sg], -1) - fm) / fsd).astype(np.float32)

        def on_inputs(ids, offs):
            dh, sg = measure_dh(ids, offs)
            return feat(dh, sg), np.clip(np.round(dh).astype(int), offs - 290, offs + 290)

        # residual offsets per recording (seed independent)
        for k, cvec, u in D:
            c_true = float(cvec[f])
            for J in JS:
                offs = OFFS[(k, J)][sc]
                corr = {"A": 0.0, "B": c_true, "B*": offs, **{m: CH[(k, J, m)] for m in DEV},
                        **{m + "r": PR[(k, J, m)] for m in DEV}}
                for m, cv in corr.items():
                    res = offs - np.round(cv).astype(int)
                    rows_res.append({"fold": f, "sim": k, "J": J, "method": m, "n": res.size,
                                     "abs_sum": float(np.abs(res).sum())})
                shift = offs - int(round(CH[(k, J, "E")]))
                _, Al = on_inputs(sc, shift)
                rows_res.append({"fold": f, "sim": k, "J": J, "method": "H", "n": shift.size,
                                 "abs_sum": float(np.abs(shift - Al).sum())})

        for s in seeds:
            nets = {}
            for nm, timing in (("pinet", False), ("dual", True)):
                net = E52.Net(timing, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
                net.load_state_dict(torch.load(W / f"fold{f}_seed{s}_{nm}.pt", map_location=device))
                nets[nm] = net.eval()
            thr = cfg81["thresholds"][f"fold{f}_seed{s}"]
            cache = {}

            def plain(offs):
                fed = np.clip(offs, -CLIP, CLIP).astype(int)
                clip_n["fed"] += int((fed != offs).sum())
                key = ("pinet", fed.tobytes())
                if key not in cache:
                    cache[key] = E52.predict(nets["pinet"], X, sc, fed, None, None, None, cfg, device)
                return cache[key]

            def on_pred(shift):
                key = ("dual", shift.tobytes())
                if key not in cache:
                    Tf, Al = on_inputs(sc, shift)
                    cache[key] = E52.predict(nets["dual"], X, sc, shift, Al, Tf, 1.0, cfg, device)
                return cache[key]

            for k, cvec, u in D:
                c_true = float(cvec[f])
                for J in JS:
                    offs = OFFS[(k, J)][sc]
                    sh = lambda c: (offs - np.round(c).astype(int)).astype(int)
                    probs = {"A": plain(offs), "B": plain(sh(c_true)), "B*": plain(sh(offs)),
                             **{m: plain(sh(CH[(k, J, m)])) for m in DEV},
                             **{m + "r": plain(sh(PR[(k, J, m)])) for m in DEV},
                             "H": on_pred(sh(CH[(k, J, "E")]))}
                    for m, p in probs.items():
                        store(("P2", s, J, k, m), sc, p, thr["H"] if m == "H" else thr["A"])
                    if J == 30:
                        for m81 in ("A", "B", "C", "D", "E", "H"):
                            ref = np.array([pr81[(str(subs[i]), s, "S2", k, m81)] for i in sc])
                            chk["J30_prob_max_diff"] = max(chk["J30_prob_max_diff"], float(np.max(np.abs(probs[m81] - ref))))
                        ref = np.array([pr81[(str(subs[i]), s, "S0", 0, "A")] for i in sc])
                        chk["Bstar_vs_S0A_max_diff"] = max(chk["Bstar_vs_S0A_max_diff"], float(np.max(np.abs(probs["B*"] - ref))))
                        for K in K_DOWN:
                            for r in range(N_SUB_DOWN):
                                for m in DEV:
                                    store(("P1", s, K, k, r, m), sc, plain(sh(CH[(k, "K", K, r, m)])), thr["A"])
            print(f"  fold {f} seed {s} scored, {len(cache)} distinct inputs ({time.time() - t0:.0f}s)", flush=True)
    flog.close()

    # ----------------------------------------------------------------------------- self-checks
    ok = {"curves": chk["curves_max_diff_ms"] < 0.01, "K20_chat": chk["K20_chat_max_diff_ms"] < 0.01,
          "J30_probs": chk["J30_prob_max_diff"] < 1e-5, "Bstar_eq_S0A": chk["Bstar_vs_S0A_max_diff"] < 1e-5}
    print("self-checks", chk, ok, flush=True)
    for name, rows in (("calib_K.csv", rows_K), ("calib_noise.csv", rows_noise), ("residual.csv", rows_res)):
        with open(odir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if not all(ok.values()):
        (odir / "summary.json").write_text(json.dumps({"self_checks": chk, "passed": ok, "STOPPED": True}, indent=2),
                                           encoding="utf-8")
        print("SELF-CHECK FAILED -- results not reported")
        return 1

    # ----------------------------------------------------------------------------- metrics
    conf, pred_rows = [], []
    for key, (p, t) in P.items():
        m_ = np.isfinite(p)
        r = E81.metrics(y[m_], p[m_], p[m_] >= t[m_])
        conf.append({"key": "|".join(map(str, key)), **r})
    cm = {c["key"]: c for c in conf}
    for key, (p, t) in P.items():
        if key[0] == "P2":
            for i in np.flatnonzero(np.isfinite(p)):
                pred_rows.append({"patient_id": str(subs[i]), "part": "P2", "seed": key[1], "J": key[2], "sim": key[3],
                                  "method": key[4], "label": int(y[i]), "prob": round(float(p[i]), 6),
                                  "threshold": round(float(t[i]), 6), "pred": int(p[i] >= t[i])})
    mean_of = lambda keys: {kk: float(np.mean([cm[k][kk] for k in keys])) for kk in ("sen", "spe", "auroc", "mcc")}

    def cal_stats(rows):
        e = np.array([r["c_hat_ms"] - r["c_true_ms"] for r in rows])
        return {"mae_ms": float(np.abs(e).mean()), "p90_ms": float(np.percentile(np.abs(e), 90)),
                "miss_rate": float((np.abs(e) > TOL).mean()),
                "pool_failed_rate": float(np.mean([r["pool_failed"] for r in rows])),
                "out_of_range_rate": float(np.mean([r["out_of_range"] for r in rows])), "n": len(rows)}

    def cell_err(rows, m):
        """mean |error| per (fold, sim) cell, in a fixed cell order."""
        acc = {}
        for r in rows:
            if r["method"] == m:
                acc.setdefault((r["fold"], r["sim"]), []).append(abs(r["c_hat_ms"] - r["c_true_ms"]))
        return [np.mean(acc[c]) for c in sorted(acc)]

    def rate(rows, m, fld):
        return float(np.mean([r[fld] for r in rows if r["method"] == m]))

    sc_all = np.flatnonzero(np.isfinite(P[("P2", seeds[0], 30, sims[0], "A")][0]))
    S = {"n_scored": int(len(sc_all)), "n_pos": int(y[sc_all].sum()), "folds": folds, "seeds": seeds, "sims": sims,
         "self_checks": chk, "passed": ok, "clipping": clip_n,
         "per_recording_fallback": {m: {"failed": v[0], "total": v[1]} for m, v in fallback.items()},
         "measurement_failures": {f"{a}:{b}": {"failed": v[0], "total": v[1]} for (a, b), v in fail.items()}}
    # part 1
    p1 = {}
    for K in KS:
        rk = [r for r in rows_K if r["K"] == K]
        p1[K] = {m: cal_stats([r for r in rk if r["method"] == m]) for m in DEV}
        for m in DEV:
            if K in K_DOWN:
                p1[K][m].update(mean_of([f"P1|{s}|{K}|{k}|{r}|{m}" for s in seeds for k in sims for r in range(N_SUB_DOWN)]))
            elif K == 20:
                p1[K][m].update(mean_of([f"P2|{s}|30|{k}|{m}" for s in seeds for k in sims]))
    S["part1"] = p1
    P1 = {}
    for K in (1, 3):
        rk = [r for r in rows_K if r["K"] == K]
        e = cell_err(rk, "E")
        res = {f"E-{b}": cell_boot(e, cell_err(rk, b)) for b in ("C", "D")}
        a = all(v["diff"] <= -5 and v["ci"][1] < 0 for v in res.values())
        bb = all(rate(rk, "E", "pool_failed") <= rate(rk, b, "pool_failed") for b in ("C", "D"))
        P1[K] = {"mae_diffs": res, "a_mae": bool(a), "b_failure": bool(bb)}
    S["P1"] = {"by_K": P1, "holds": all(v["a_mae"] and v["b_failure"] for v in P1.values()),
               "only_b": all(v["b_failure"] for v in P1.values()) and not all(v["a_mae"] for v in P1.values())}
    # part 2
    res_mae = {}
    for r in rows_res:
        a = res_mae.setdefault((r["J"], r["method"]), [0.0, 0])
        a[0] += r["abs_sum"]; a[1] += r["n"]
    p2 = {J: {m: {**mean_of([f"P2|{s}|{J}|{k}|{m}" for s in seeds for k in sims]),
                  "residual_mae_ms": res_mae[(J, m)][0] / res_mae[(J, m)][1]} for m in P2_METHODS} for J in JS}
    S["part2"] = p2
    p2a = p2[100]["B*"]["mcc"] - p2[100]["B"]["mcc"]
    ysc = y[sc_all]
    bs = PS.boots(ysc, 2000, 0)

    def mcc_diff_ci(J, a, b):
        pr = {(m, s, k): (P[("P2", s, J, k, m)][0][sc_all] >= P[("P2", s, J, k, m)][1][sc_all])
              for m in (a, b) for s in seeds for k in sims}

        def dd(idx):
            yy = ysc[idx]
            return np.mean([np.mean([mcc_vec(pr[(a, s, k)][idx], yy) - mcc_vec(pr[(b, s, k)][idx], yy) for k in sims])
                            for s in seeds])
        pt = dd(np.arange(len(ysc)))
        Dd = np.array([dd(i) for i in bs])
        return {"diff": float(pt), "ci": [float(np.percentile(Dd, 2.5)), float(np.percentile(Dd, 97.5))]}

    P2b = {}
    for J in (100, 150):
        cmp = max(("Cr", "Dr"), key=lambda m: p2[J][m]["mcc"])
        ra = p2[J]["Er"]["residual_mae_ms"] <= min(p2[J]["Cr"]["residual_mae_ms"], p2[J]["Dr"]["residual_mae_ms"]) - 5
        ci = mcc_diff_ci(J, "Er", cmp)
        P2b[J] = {"comparator": cmp, "a_residual": bool(ra), "mcc_diff": ci,
                  "b_mcc": bool(ci["diff"] >= 0.02 and ci["ci"][0] > 0)}
    S["P2a"] = {"mcc_Bstar_minus_B_J100": p2a, "holds": bool(p2a >= 0.05)}
    S["P2b"] = {"by_J": P2b, "tested": S["P2a"]["holds"],
                "holds": bool(S["P2a"]["holds"] and all(v["a_residual"] and v["b_mcc"] for v in P2b.values()))}
    S["part2_diffs_described"] = {f"J{J}:{a}-{b}": {kk: p2[J][a][kk] - p2[J][b][kk] for kk in ("sen", "spe", "mcc")}
                                  for J in JS for a, b in (("Er", "E"), ("H", "Er"), ("H", "E"))}
    # part 3
    p3 = {}
    for snr in ["inf"] + SNRS:
        p3[snr] = {}
        for K in (3, 20):
            if snr == "inf":
                rk = [r for r in rows_K if r["K"] == K and (K == 20 or r["subset"] < N_SUB_DOWN)]
            else:
                rk = [r for r in rows_noise if r["snr_db"] == snr and r["K"] == K]
            p3[snr][K] = {m: cal_stats([r for r in rk if r["method"] == m]) for m in DEV}
            if snr in (10, 5):
                p3[snr][K]["test"] = {f"E-{b}": cell_boot(cell_err(rk, "E"), cell_err(rk, b)) for b in ("C", "D")}
                p3[snr][K]["holds"] = bool(all(v["diff"] <= -5 and v["ci"][1] < 0 for v in p3[snr][K]["test"].values())
                                           and all(rate(rk, "E", "pool_failed") <= rate(rk, b, "pool_failed")
                                                   for b in ("C", "D")))
    S["part3"] = {str(k): v for k, v in p3.items()}
    S["P3"] = {"holds": bool(all(p3[snr][K]["holds"] for snr in (10, 5) for K in (3, 20)))}
    S["elapsed_min"] = round((time.time() - t0) / 60, 1)
    (odir / "summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for name, rows in (("confusion.csv", conf), ("predictions.csv", pred_rows)):
        with open(odir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: S[k] for k in ("P1", "P2a", "P2b", "P3", "clipping")}, indent=1, default=float))
    for K in KS:
        print("K", K, {m: round(p1[K][m]["mae_ms"], 1) for m in DEV}, {m: round(p1[K][m]["pool_failed_rate"], 3) for m in DEV})
    for J in JS:
        print("J", J, {m: round(p2[J][m]["mcc"], 3) for m in P2_METHODS})
    for snr in ["inf"] + SNRS:
        print("SNR", snr, {K: {m: round(p3[snr][K][m]["mae_ms"], 1) for m in DEV} for K in (3, 20)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
