#!/usr/bin/env python
"""E83 -- 实验2：社区数据验证与校准人群变化（按导师方案执行，方案文件 2026-11-02）。

    python experiments/E83_community/run.py --smoke      # 第 1 折、种子 0，只跑 S0、S1：流程检查
    python experiments/E83_community/run.py              # 5 折 x 种子 0,1,2，S0、S1、S2（20 次模拟）

问题  (1) 实验1的结论换到社区数据后是否仍成立？(2) 校准所用人群改变后，会不会把人群差异误认为设备偏差？
      只用现有数据与实验1（E81）已训练的模型，不新增采集、标注、网络或训练。社区结果此前（E66）已看过，
      论文中只能写成“对已有独立队列的补充分析”。本实验不设成立/不成立判据，只报告描述性结果与区间。

数据  社区筛查集 data/raw/hfref/HFrEF_community_scenario_test_set（120 人），预处理与主队列完全相同
      （1 kHz 重采样、前 30 s、逐位点逐模态 z 标准化、四个位点齐全，LVEF <= 0.40 为阳性）。社区记录不参与
      训练、响应曲线建立或阈值选择；与主队列 ID 必须无重叠（id_check.json）。

名单  社区 ID 按字符串排序，种子 20261102，不看标签无放回抽 20 人作校准池，其余为评分集；所有方法、模型与
      场景共用。名单锁定后才统计阳性数，不因阳性数重抽。校准患者不计入评分。

模型  E81 的 5 折 x 3 训练种子 = 15 套配置，各自预测同一社区评分集；每套配置配套对应折的训练参考值
      （mu_site 与响应曲线取自 E81 config.json，DO-PINet 的时延特征标准化量由该折主队列训练患者重算）、
      该折的时延算子（每折一个，三个种子共用）与 E81 锁定的阈值。不重训、不挑折/种子、不在社区上改阈值。

方法  A 不校准（PINet）；B 扣除实际注入的共同偏移、保留抖动（参照，唯一可用真值的方法）；
      C 峰值法 / D 互相关法 / E DelayOp 估计共同偏移后输入同一 PINet；F 晚融合，不做设备级校准；
      H = E 的设备校准量 + DO-PINet 开模式；I = H 的权重，跳过设备级校准；J 只用心电（仅作程序自检）。
      阈值：A–E 用 A 的，H、I 用 I 的，F、J 用各自的（均为 E81 锁定值）。

场景  S0 无偏移（C、D、E、H 仍校准）；S1 全体 PCG 后移 80 ms；S2 第 k 次模拟（k = 1..20）用种子 20261102 + k，
      先抽一个共同偏移 U(40, 200) ms（全体社区校准者与评分者、15 套配置共用），再抽每人每位点 U(−30, 30) ms
      抖动；均取整到毫秒（1 kHz 采样）。ECG 不动；PCG 从缓冲区更早处读取，不循环平移。偏移表存 offsets.csv。

校准  E81 规则不变：池中有效测量 (tau − mu_site) 的中位数，经响应曲线单调段线性反插得设备偏移；超出曲线范围
      截到端点并计为越界；池内无有效测量则整池失败、回退为不校准（保留该次）。校准程序不读取疾病标签或
      实际注入偏移。误差 = 估计的共同偏移 − 实际注入的共同偏移（S2 抖动不计入真值）。

校准人群变化（表5）  评分集仍为同一社区评分患者，只做 S0、S1：C、D、E、H 改用 E81 对应折原有的 20 名主队列
      校准患者（加同样的已知偏移）；报告“主队列池 − 社区池”的估计量差与分类差。只比较人群来源，不声称两
      队列由同一真实设备采集。

统计  每套配置、每次模拟先在完整评分集上算敏感度、特异度、AUROC、MCC，再取平均（S0、S1：15 次；S2：300 次）；
      不平均概率，不把同一患者的多次预测当成多人。检出数、误报数报告各次运行的最小–最大值 / 真实人数。
      区间：以患者为单位，阳性、阴性分别有放回抽样 5000 次（种子 20261103），同一患者在所有方法、模型、模拟中的
      记录一起抽取，重算平均指标与配对差值，取 2.5% 与 97.5% 分位数。模型与校准池固定，不含重训或换池的不确定性。
      逐次运行的敏感度另给 Clopper–Pearson 精确 95% 区间。评分集无阳性时，敏感度与 AUROC 记为无法计算。
      表6 主队列四行用 E81 S2 逐患者预测重算（按 E81 口径：每个种子、每次模拟合并五折评分患者），重抽样在每折内
      按阳性/阴性分别进行再合并五折（种子 20261103）。

自检  社区与主队列 ID 无重叠；校准池为 20 个不同的人且与评分集互斥；S0 中 A 与 B 逐患者概率一致；S1 中 B 与 S0 的 A
      一致；J 在所有场景下一致；主队列池在 S0、S1 的估计值与 E81 calibration.csv 一致（< 0.01 ms）；表6 主队列
      点估计与 E81 summary.json 一致；送入分类器的偏移均在 ±290 ms 内（不截断）。

运行前预估（2026-11-02 写定，跑完逐条对照；依据 E81、E66 与社区集构成，并非判据）
  1. 评分集阳性约 4 人（5 人约 40%、4 人约 41%、3 人约 16%）；社区 MCC 约 0.30–0.45，因患病率低，不能与主队列 0.53 比。
  2. S0：A = B，特异度 78–85%；C、D、E 与 A 相差 ±2 个百分点以内；F 72–80%；H、I 与 A 相近，可能多漏 1 例。
  3. S1：A 特异度 35–50%、I 45–60%（敏感度近 100%）；C、D、E、H 恢复到 S0 的 ±2 个百分点内；F 基本不变。
  4. S2：A 约 45%、I 约 55–60%；B、C、D、E、H 比 S0 低 1–3 个百分点；方法间 MCC 差在 ±0.02 以内。
  5. 估计误差：S0 平均误差 +1 至 +5 ms、MAE 2–6 ms；S2 MAE 3–6 ms；越界与整池失败为 0。S0 估计量 > 10 ms 的可能约 15%。
  6. 换池（主队列 − 社区）：估计量差 < 5 ms（方向不定），特异度差 < 2 个百分点，MCC 差 ±0.02，区间跨 0。
  7. 表6：主队列 E−C、E−D 区间窄且跨 0，H−I 明确为正；社区 E−C、E−D 跨 0；H−I 特异度 +20 至 +30 且不跨 0。
     学习型方法在社区上显著优于 C、D 的可能约 10%。

输出（experiments/E83_community/full 或 smoke）  id_check.json、lists.json、offsets.csv、calibration.csv、
  predictions.csv（逐患者，含 ID、标签、来源、角色、折、种子、模拟、场景、方法、校准池来源、概率、阈值、预测类别）、
  confusion.csv（每次运行的 TP/FN/TN/FP）、cp_sensitivity.csv、failures.log、summary.json
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
from scipy.io import wavfile
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

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
E52, E50, E77, E79 = E81.E52, E81.E50, E81.E77, E81.E79
E12 = E50.E49.E44._E12
GC, FS = E81.GC, E81.FS
E81DIR = EXP / "E81_controlled" / "full"
COMM = ROOT / "data" / "raw" / "hfref" / "HFrEF_community_scenario_test_set"
CACHE = ROOT / "data" / "processed" / "hfref_community_1000hz.npz"
LIST_SEED, BOOT_SEED = 20261102, 20261103
N_POOL, N_SIM, N_BOOT = 20, 20, 5000
CLIP = 290
METHODS = ["A", "B", "C", "D", "E", "F", "H", "I", "J"]
DEV = ["C", "D", "E"]
SWAP = ["C", "D", "E", "H"]


# ----------------------------------------------------------------------------- data, lists, offsets
def load_community():
    """same preprocessing as the main-cohort cache (E12.build_cache)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["X"], z["y"], z["subjects"], list(z["skipped"])
    lv = {}
    for r in csv.reader(open(COMM / "test_set_label.csv", encoding="utf-8-sig")):
        if len(r) >= 2 and r[0].strip():
            try:
                lv[r[0].strip()] = float(r[1])
            except ValueError:
                pass
    n = int(E12.SECONDS * FS)
    z = lambda v: ((v - v.mean()) / (v.std() or 1.0)).astype(np.float32)
    X, y, subs, skipped = [], [], [], []
    for s in sorted(lv):
        sig = []
        for site in E12.SITES:
            w = COMM / "test_set" / f"{s}_{site}_ECG.wav"
            if not w.exists():
                break
            fe, e = wavfile.read(str(w))
            fp, pcg = wavfile.read(str(w)[:-8] + "_PCG.wav")
            e = sps.resample_poly(np.asarray(e, float).ravel(), int(FS), int(fe))
            pcg = sps.resample_poly(np.asarray(pcg, float).ravel(), int(FS), int(fp))
            if len(e) < n or len(pcg) < n:
                break
            sig.append(np.stack([z(e[:n]), z(pcg[:n])]))
        if len(sig) == 4:
            X.append(np.stack(sig)); y.append(int(lv[s] <= E12.HFREF)); subs.append(s)
        else:
            skipped.append(s)
    X, y, subs = np.stack(X).astype(np.float32), np.array(y), np.array(subs)
    np.savez_compressed(CACHE, X=X, y=y, subjects=subs, skipped=np.array(skipped, dtype=object))
    return X, y, subs, skipped


def make_lists(n):
    rng = np.random.default_rng(LIST_SEED)
    pool = np.sort(rng.choice(n, size=N_POOL, replace=False))
    return pool, np.setdiff1d(np.arange(n), pool)


def make_offsets(n, scen):
    """scenario -> list of (sim id, common offset, jitter (n, 4)); one common offset per simulation for everyone."""
    S = {"S0": [(0, 0.0, np.zeros((n, 4), int))], "S1": [(0, 80.0, np.zeros((n, 4), int))], "S2": []}
    for k in range(1, N_SIM + 1):
        r = np.random.default_rng(LIST_SEED + k)
        S["S2"].append((k, float(np.round(r.uniform(40, 200))), np.round(r.uniform(-30, 30, (n, 4))).astype(int)))
    return {s: S[s] for s in scen}


def curve_from_cfg(c):
    mu, gt = np.array(c["mu_site"]), np.array(c["g"])
    keep = np.concatenate([[True], np.diff(np.maximum.accumulate(gt)) > 0])          # E81.build_curve's rule
    return mu, gt, keep


# ----------------------------------------------------------------------------- metrics and bootstrap
def run_metrics(yt, p, pred):
    tp, fn = int((pred & (yt == 1)).sum()), int((~pred & (yt == 1)).sum())
    tn, fp = int((~pred & (yt == 0)).sum()), int((pred & (yt == 0)).sum())
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    has = tp + fn > 0
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "sen": tp / (tp + fn) if has else np.nan, "spe": tn / (tn + fp),
            "auroc": float(roc_auc_score(yt, p)) if has else np.nan, "mcc": (tp * tn - fp * fn) / den if den else 0.0}


def cp_interval(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def boot_weights(groups, n, B, seed):
    """(B, n) resampling counts; each group (e.g. positives / negatives, per fold) resampled with replacement."""
    rng = np.random.default_rng(seed)
    W = np.zeros((B, n), np.float32)
    for g in groups:
        draw = g[rng.integers(0, len(g), (B, len(g)))]
        for b in range(B):
            W[b] += np.bincount(draw[b], minlength=n)
    return W


def boot_means(W, yv, preds):
    """mean over runs of sen, spe, mcc for each weight row; preds (R, n) bool -> dict of (B,) arrays."""
    pos, neg = yv == 1, yv == 0
    nP, nN = W[:, pos].sum(1, keepdims=True), W[:, neg].sum(1, keepdims=True)
    TP = W[:, pos] @ preds[:, pos].T.astype(np.float32)
    FP = W[:, neg] @ preds[:, neg].T.astype(np.float32)
    FN, TN = nP - TP, nN - FP
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = np.where(den > 0, (TP * TN - FP * FN) / np.where(den > 0, den, 1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sen": (TP / nP).mean(1), "spe": (TN / nN).mean(1), "mcc": mcc.mean(1)}


def ci(v):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    cfg = dict(E50.E49.E44.CFG)
    d = E12.build_cache(False)
    X, ym, subs_m = d["X"], d["y"], d["subjects"]
    L81 = E81.make_lists(ym, d["fold"])
    Xc, yc, subs_c, skipped = load_community()
    folds = [0] if args.smoke else list(range(5))
    seeds = E81.TRAIN_SEEDS[:1] if args.smoke else E81.TRAIN_SEEDS
    scens = ["S0", "S1"] if args.smoke else ["S0", "S1", "S2"]
    odir = OUT / ("smoke" if args.smoke else "full")
    odir.mkdir(exist_ok=True)
    W8 = E81DIR / "weights"
    cfg81 = json.loads((E81DIR / "config.json").read_text(encoding="utf-8"))
    lists81 = json.loads((E81DIR / "lists.json").read_text(encoding="utf-8"))
    cal81 = {(int(r["fold"]), r["scenario"], r["method"]): float(r["c_hat_ms"])
             for r in csv.DictReader(open(E81DIR / "calibration.csv", encoding="utf-8"))
             if r["scenario"] in ("S0", "S1") and r["seed"] == "0"}

    # ---- step 1-2: identity checks and locked lists
    overlap = sorted(set(map(str, subs_c)) & set(map(str, subs_m)))
    pool, sc = make_lists(len(yc))
    ysc = yc[sc]
    chk = {"community_main_overlap": len(overlap), "pool_unique": int(len(set(subs_c[pool]))),
           "pool_score_disjoint": bool(len(np.intersect1d(pool, sc)) == 0)}
    for f in range(5):
        assert [str(subs_m[i]) for i in L81[f]["pool"]] == lists81[str(f)]["pool"], "E81 lists differ"
    idc = {"community_dir": str(COMM.relative_to(ROOT)), "community_subjects": int(len(yc)), "community_pos": int(yc.sum()),
           "skipped_incomplete": len(skipped), "main_subjects": int(len(ym)), "overlap_ids": overlap,
           "pool": {"n": N_POOL, "pos": int(yc[pool].sum()), "neg": int((yc[pool] == 0).sum())},
           "score": {"n": int(len(sc)), "pos": int(ysc.sum()), "neg": int((ysc == 0).sum())}}
    (odir / "id_check.json").write_text(json.dumps(idc, indent=1, ensure_ascii=False), encoding="utf-8")
    (odir / "lists.json").write_text(json.dumps(
        {"community": {"seed": LIST_SEED, "pool": [str(subs_c[i]) for i in pool], "score": [str(subs_c[i]) for i in sc]},
         "main_pool_by_fold": {f: lists81[str(f)]["pool"] for f in range(5)}}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"community {len(yc)} ({yc.sum()} HFrEF), skipped {len(skipped)}, overlap {len(overlap)}; "
          f"pool {N_POOL} ({yc[pool].sum()} pos), score {len(sc)} ({ysc.sum()} pos)", flush=True)
    if overlap or not chk["pool_score_disjoint"] or chk["pool_unique"] != N_POOL:
        raise SystemExit("identity check failed")

    OFF = make_offsets(len(yc), scens)
    with open(odir / "offsets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "sim", "common_ms", "patient_id", "role"] + [f"jitter_{s}" for s in E12.SITES])
        for scen, sims in OFF.items():
            for k, c, jit in sims:
                for i in range(len(yc)):
                    w.writerow([scen, k, c, subs_c[i], "pool" if i in pool else "score", *jit[i].tolist()])

    TRc, TRm = E81.Traditional(Xc), E81.Traditional(X)
    TRc.prep(np.arange(len(yc)))
    fail = {}                                          # (source, method) -> [failed, total] site measurements
    flog = open(odir / "failures.log", "w", encoding="utf-8")
    cal_rows, pred_rows, conf_rows, cp_rows = [], [], [], []
    PR = {}                                            # (source, scen, method) -> list of (run key, prob, pred) on sc
    chk.update({"main_pool_chat_max_diff_ms": 0.0, "S0_A_eq_B_max": 0.0, "S1_B_eq_S0A_max": 0.0,
                "J_across_scen_max": 0.0, "fed_offsets_beyond_290": 0})

    for f in folds:
        tr, poolm = L81[f]["train"], L81[f]["pool"]
        TRm.prep(poolm)
        op = E50.DelayOp(FS, lag_min_ms=-200.0, lag_max_ms=300.0).to(device)
        op.load_state_dict(torch.load(W8 / f"fold{f}_delayop.pt", map_location=device))
        op.eval()
        mu_op, k_op = E50.calibrate(op, X, tr, device)
        curves = {m: curve_from_cfg(cfg81["curves"][str(f)][m]) for m in DEV}

        def m_op(Xa):
            def m(ids, offs):
                ix = [(i, j) for i in ids for j in range(4)]
                tau, _ = E50.op_tau(op, Xa, ix, np.asarray(offs).reshape(-1).astype(int), device)
                return tau.reshape(-1, 4)
            return m

        def m_trad(kind, T, src, names):
            def m(ids, offs):
                out = np.full((len(ids), 4), np.nan)
                for a, i in enumerate(ids):
                    for j in range(4):
                        t, why = T.tau(i, j, int(offs[a, j]), kind)
                        out[a, j] = t
                        c = fail.setdefault((src, kind), [0, 0]); c[1] += 1
                        if why:
                            c[0] += 1
                            flog.write(f"{src} fold {f} method {kind} subject {names[i]} site {j} offset {int(offs[a, j])}: {why}\n")
                return out
            return m

        meas = {"community": {"C": m_trad("C", TRc, "community", subs_c), "D": m_trad("D", TRc, "community", subs_c),
                              "E": m_op(Xc)},
                "main": {"C": m_trad("C", TRm, "main", subs_m), "D": m_trad("D", TRm, "main", subs_m), "E": m_op(X)}}

        # ---- device calibration (seed independent); the estimator never sees labels or the true offset
        CH = {}
        for scen, sims in OFF.items():
            for k, c, jit in sims:
                srcs = [("community", pool, (c + jit[pool]).astype(int))]
                if scen in ("S0", "S1"):
                    srcs.append(("main", poolm, np.full((len(poolm), 4), int(c))))
                for src, pids, offs_pool in srcs:
                    names = subs_c if src == "community" else subs_m
                    for m in DEV:
                        ch, oor, pf = E81.estimate(meas[src][m], pids, offs_pool, *curves[m])
                        CH[(src, scen, k, m)] = ch
                        cal_rows.append({"fold": f, "scenario": scen, "sim": k, "pool_source": src, "method": m,
                                         "pool_ids": " ".join(str(names[i]) for i in pids), "c_true_ms": c,
                                         "c_hat_ms": round(ch, 3), "error_ms": round(ch - c, 3),
                                         "out_of_range": int(oor), "pool_failed": int(pf)})
                        if src == "main":
                            chk["main_pool_chat_max_diff_ms"] = max(chk["main_pool_chat_max_diff_ms"],
                                                                    abs(ch - cal81[(f, scen, m)]))
        print(f"  fold {f} calibrated ({time.time() - t0:.0f}s)", flush=True)

        # ---- DO-PINet timing features: normalisation from this fold's main-cohort training patients
        def measure_dh(Xa, ids, offs):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, sg = E50.op_tau(op, Xa, ix, np.asarray(offs).reshape(-1).astype(int), device)
            return np.clip((tau.reshape(-1, 4) - mu_op[None]) / k_op, -E52.CLIP_MS, E52.CLIP_MS), sg.reshape(-1, 4)

        dh_tr, sg_tr = measure_dh(X, tr, np.zeros((len(tr), 4), int))
        fm = np.array([dh_tr.mean(), sg_tr.mean()]); fsd = np.array([dh_tr.std(), sg_tr.std()]) + 1e-6
        feat = lambda dh, sg: ((np.stack([dh, sg], -1) - fm) / fsd).astype(np.float32)

        for s in seeds:
            nets = {}
            for nm, cls in (("pinet", E52.Net), ("dual", E52.Net), ("late", E77.LateNet),
                            ("ecg", E79.variant(force_zero=1))):
                net = (cls(nm == "dual", dim=cfg["dim"], dropout=cfg["dropout"]) if cls is E52.Net
                       else cls(False, dim=cfg["dim"], dropout=cfg["dropout"]))
                net.load_state_dict(torch.load(W8 / f"fold{f}_seed{s}_{nm}.pt", map_location=device))
                nets[nm] = net.to(device).eval()
            thr = cfg81["thresholds"][f"fold{f}_seed{s}"]
            cache = {}

            def run(nm, offs):
                key = (nm, offs.tobytes())
                if key not in cache:
                    chk["fed_offsets_beyond_290"] += int((np.abs(offs) > CLIP).sum())
                    if nm == "dual":
                        dh, sg = measure_dh(Xc, sc, offs)
                        Al = np.clip(np.round(dh).astype(int), offs - CLIP, offs + CLIP)
                        cache[key] = E52.predict(nets["dual"], Xc, sc, offs, Al, feat(dh, sg), 1.0, cfg, device)
                    else:
                        cache[key] = E52.predict(nets[nm], Xc, sc, offs, None, None, None, cfg, device)
                return cache[key]

            for scen, sims in OFF.items():
                for k, c, jit in sims:
                    offs = (c + jit[sc]).astype(int)
                    shift = lambda v: (offs - int(round(v))).astype(int)
                    jobs = [("community", m) for m in METHODS] + ([("main", m) for m in SWAP] if scen != "S2" else [])
                    for src, m in jobs:
                        if m in "CDE":
                            p = run("pinet", shift(CH[(src, scen, k, m)]))
                        elif m == "H":
                            p = run("dual", shift(CH[(src, scen, k, "E")]))
                        else:
                            p = {"A": lambda: run("pinet", offs), "B": lambda: run("pinet", shift(c)),
                                 "F": lambda: run("late", offs), "I": lambda: run("dual", offs),
                                 "J": lambda: run("ecg", offs)}[m]()
                        t = float(thr[m])
                        pred = p >= t
                        PR.setdefault((src, scen, m), []).append(((f, s, k), p, pred))
                        r = run_metrics(ysc, p, pred)
                        conf_rows.append({"fold": f, "seed": s, "scenario": scen, "sim": k, "pool_source": src,
                                          "method": m, "n": len(sc), "n_pos": int(ysc.sum()), **r})
                        if ysc.sum():
                            lo, hi = cp_interval(r["TP"], int(ysc.sum()))
                            cp_rows.append({"fold": f, "seed": s, "scenario": scen, "sim": k, "pool_source": src,
                                            "method": m, "TP": r["TP"], "n_pos": int(ysc.sum()),
                                            "sen": r["sen"], "cp_lo": round(lo, 4), "cp_hi": round(hi, 4)})
                        for a, i in enumerate(sc):
                            pred_rows.append({"patient_id": str(subs_c[i]), "label": int(yc[i]), "source": "community",
                                              "role": "score", "fold": f, "seed": s, "sim": k, "scenario": scen,
                                              "method": m, "pool_source": src, "prob": round(float(p[a]), 6),
                                              "threshold": round(t, 6), "pred": int(pred[a])})
            # ---- per-config self-checks
            g = lambda scen, m, src="community": [x for x in PR[(src, scen, m)] if x[0][:2] == (f, s)]
            a0 = g("S0", "A")[0][1]
            chk["S0_A_eq_B_max"] = max(chk["S0_A_eq_B_max"], float(np.abs(a0 - g("S0", "B")[0][1]).max()))
            chk["S1_B_eq_S0A_max"] = max(chk["S1_B_eq_S0A_max"], float(np.abs(a0 - g("S1", "B")[0][1]).max()))
            j0 = g("S0", "J")[0][1]
            chk["J_across_scen_max"] = max(chk["J_across_scen_max"],
                                           max(float(np.abs(x[1] - j0).max()) for scen in OFF for x in g(scen, "J")))
            print(f"  fold {f} seed {s} scored ({time.time() - t0:.0f}s)", flush=True)
    flog.close()

    # ---- summaries
    nP, nN = int(ysc.sum()), int((ysc == 0).sum())
    B = 200 if args.smoke else N_BOOT
    Wc = boot_weights([np.flatnonzero(ysc == 1), np.flatnonzero(ysc == 0)], len(sc), B, BOOT_SEED)
    BM = {key: boot_means(Wc, ysc, np.stack([x[2] for x in v])) for key, v in PR.items()}
    PT = {key: boot_means(np.ones((1, len(sc)), np.float32), ysc, np.stack([x[2] for x in v])) for key, v in PR.items()}
    tables = {}
    for (src, scen, m), v in PR.items():
        rows = [r for r in conf_rows if (r["pool_source"], r["scenario"], r["method"]) == (src, scen, m)]
        e = {kk: float(np.nanmean([r[kk] for r in rows])) if nP or kk in ("spe", "mcc") else None
             for kk in ("sen", "spe", "auroc", "mcc")}
        tables.setdefault(src, {}).setdefault(scen, {})[m] = {
            **e, "runs": len(rows),
            "sen_ci": ci(BM[(src, scen, m)]["sen"]) if nP else None, "spe_ci": ci(BM[(src, scen, m)]["spe"]),
            "mcc_ci": ci(BM[(src, scen, m)]["mcc"]),
            "tp_range": [min(r["TP"] for r in rows), max(r["TP"] for r in rows)], "n_pos": nP,
            "fp_range": [min(r["FP"] for r in rows), max(r["FP"] for r in rows)], "n_neg": nN}

    def diff(ka, kb):
        a, b, pa, pb = BM[ka], BM[kb], PT[ka], PT[kb]
        out = {}
        for kk in ("sen", "spe", "mcc"):
            if kk == "sen" and not nP:
                out[kk] = None
                continue
            sc_ = 100 if kk != "mcc" else 1
            out[kk] = {"diff": float(sc_ * (pa[kk][0] - pb[kk][0])), "ci": [sc_ * x for x in ci(a[kk] - b[kk])]}
        return out

    calerr = {}
    for scen in OFF:
        for m in DEV:
            rows = [r for r in cal_rows if (r["pool_source"], r["scenario"], r["method"]) == ("community", scen, m)]
            e = np.array([r["error_ms"] for r in rows])
            calerr[f"{scen}:{m}"] = {"mean_error_ms": float(e.mean()), "mae_ms": float(np.abs(e).mean()),
                                     "out_of_range": [sum(r["out_of_range"] for r in rows), len(rows)],
                                     "pool_failed": [sum(r["pool_failed"] for r in rows), len(rows)]}
    swap = {}
    for scen in ("S0", "S1"):
        for m in SWAP:
            mm = "E" if m == "H" else m
            cm = {r["fold"]: r["c_hat_ms"] for r in cal_rows if (r["pool_source"], r["scenario"], r["method"]) == ("main", scen, mm)}
            cc = {r["fold"]: r["c_hat_ms"] for r in cal_rows if (r["pool_source"], r["scenario"], r["method"]) == ("community", scen, mm)}
            swap[f"{scen}:{m}"] = {"est_diff_ms": float(np.mean([cm[f] - cc[f] for f in folds])),
                                   "c_hat_main_by_fold": [cm[f] for f in folds], "c_hat_community_by_fold": [cc[f] for f in folds],
                                   **diff(("main", scen, m), ("community", scen, m))}
    pairs = {}
    if "S2" in OFF:
        for a, b in (("E", "C"), ("E", "D"), ("H", "E"), ("H", "I")):
            pairs[f"community {a}-{b}"] = diff(("community", "S2", a), ("community", "S2", b))
        pairs.update(main_cohort_pairs(chk, B))

    S = {"folds": folds, "seeds": seeds, "id_check": idc, "self_checks": chk,
         "passed": bool(chk["S0_A_eq_B_max"] < 1e-6 and chk["S1_B_eq_S0A_max"] < 1e-6 and chk["J_across_scen_max"] < 1e-5
                        and chk["main_pool_chat_max_diff_ms"] < 0.01 and chk["fed_offsets_beyond_290"] == 0
                        and chk.get("main_pairs_point_max_diff", 0.0) < 1e-6),
         "tables": tables, "calibration_error": calerr, "pool_swap": swap, "pairs": pairs,
         "failures": {f"{k[0]}:{k[1]}": v for k, v in fail.items()}, "bootstrap": {"B": B, "seed": BOOT_SEED},
         "elapsed_min": round((time.time() - t0) / 60, 1)}
    (odir / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    for name, rows in (("predictions.csv", pred_rows), ("confusion.csv", conf_rows), ("calibration.csv", cal_rows),
                       ("cp_sensitivity.csv", cp_rows)):
        if rows:
            with open(odir / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    print(json.dumps({"self_checks": chk, "passed": S["passed"]}, indent=1, default=float))
    for scen in OFF:
        print(scen)
        for m in METHODS:
            v = tables["community"][scen][m]
            sen = "  n/a" if v["sen"] is None else f"{100 * v['sen']:5.1f}"
            print(f"  {m}  sen {sen}  spe {100 * v['spe']:5.1f} [{100 * v['spe_ci'][0]:.1f}, {100 * v['spe_ci'][1]:.1f}]"
                  f"  AUROC {v['auroc'] if v['auroc'] is None else round(v['auroc'], 3)}  MCC {v['mcc']:.3f}"
                  f"  TP {v['tp_range']}/{nP}  FP {v['fp_range']}/{nN}")
    print("calibration", json.dumps(calerr, default=float))
    print("pool swap", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("est_diff_ms", "spe", "mcc")}
                                   for k, v in swap.items()}, default=float))
    if pairs:
        print("pairs", json.dumps(pairs, default=float))
    return 0


def main_cohort_pairs(chk, B):
    """Table 6 rows 1-4 from E81 S2 per-patient predictions: per (seed, sim) pool the five folds' scored patients (E81's
    unit), resample positives / negatives within each fold, merge folds."""
    lists81 = json.loads((E81DIR / "lists.json").read_text(encoding="utf-8"))
    summ81 = json.loads((E81DIR / "summary.json").read_text(encoding="utf-8"))["summary"]["S2"]
    ids = [p for f in range(5) for p in lists81[str(f)]["score"]]
    col = {p: n for n, p in enumerate(ids)}
    fold_of = {p: f for f in range(5) for p in lists81[str(f)]["score"]}
    lab = np.zeros(len(ids), int)
    pred = {}
    for r in csv.DictReader(open(E81DIR / "predictions.csv", encoding="utf-8")):
        if r["scenario"] != "S2" or r["method"] not in "CDEHI":
            continue
        n = col[r["patient_id"]]
        lab[n] = int(r["label"])
        pred.setdefault(r["method"], {}).setdefault((int(r["seed"]), int(r["sim"])), np.zeros(len(ids), bool))[n] = r["pred"] == "1"
    groups = [np.array([col[p] for p in ids if fold_of[p] == f and lab[col[p]] == c]) for f in range(5) for c in (1, 0)]
    W = boot_weights(groups, len(ids), B, BOOT_SEED)
    keys = sorted(pred["C"])
    BM = {m: boot_means(W, lab, np.stack([pred[m][k] for k in keys])) for m in "CDEHI"}
    PT = {m: boot_means(np.ones((1, len(ids)), np.float32), lab, np.stack([pred[m][k] for k in keys])) for m in "CDEHI"}
    out, dev = {}, 0.0
    for a, b in (("E", "C"), ("E", "D"), ("H", "E"), ("H", "I")):
        out[f"main {a}-{b}"] = {}
        for kk in ("sen", "spe", "mcc"):
            s_ = 100 if kk != "mcc" else 1
            dpt = float(PT[a][kk][0] - PT[b][kk][0])
            dev = max(dev, abs(dpt - (summ81[a][kk] - summ81[b][kk])))
            out[f"main {a}-{b}"][kk] = {"diff": s_ * dpt, "ci": [s_ * x for x in ci(BM[a][kk] - BM[b][kk])]}
    chk["main_pairs_point_max_diff"] = dev
    return out


if __name__ == "__main__":
    raise SystemExit(main())
