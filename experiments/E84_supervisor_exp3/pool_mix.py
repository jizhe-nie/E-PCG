"""Experiment 3, part 3: 0/10/20% positive ten-person calibration pools.

Run after modules.py. Reuses E81 weights, curves, thresholds, folds and offsets.
No retraining and no use of labels inside calibration.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.E81_controlled import run as E81

E50, E52 = E81.E50, E81.E52
SRC = ROOT / "experiments/E81_controlled/full"
OUT = Path(__file__).parent
METHODS = ("C", "D", "E")
PROPS = (0, 10, 20)
SEED = 20261103


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader(); w.writerows(rows)


def make_subpools(L, y, subs, folds, sims):
    """Nested pools: retain eight negatives, replacing up to two with positives."""
    rng = np.random.default_rng(SEED)
    draws, rows = {}, []
    eligibility = []
    for f in range(5):
        p = L[f]["pool"]
        pos, neg = p[y[p] == 1], p[y[p] == 0]
        ok = len(pos) >= 2 and len(neg) >= 10
        eligibility.append({"fold": f, "n_positive": int(len(pos)), "n_negative": int(len(neg)), "eligible": bool(ok)})
        if not ok or f not in folds: continue
        for sim in sims:
            neg10 = rng.choice(neg, 10, replace=False)
            pos2 = rng.choice(pos, 2, replace=False)
            for prop, chosen in ((0, neg10), (10, np.r_[neg10[:9], pos2[:1]]),
                                 (20, np.r_[neg10[:8], pos2])):
                chosen = np.sort(chosen)
                assert len(chosen) == 10 and int(y[chosen].sum()) == prop // 10
                draws[(f, sim, prop)] = chosen
                rows.append({"fold": f, "sim": sim, "pool_positive_pct": prop,
                             "patient_ids": " ".join(map(str, subs[chosen])),
                             "n_positive": prop // 10, "n_negative": 10 - prop // 10})
    return draws, rows, eligibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT / ("smoke_pool" if args.smoke else "full_pool")
    out.mkdir(parents=True, exist_ok=True)
    d = E50.E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    L = E81.make_lists(y, fold)
    lists = json.loads((SRC / "lists.json").read_text(encoding="utf-8"))
    for f in range(5):
        for role in ("train", "val", "pool", "score"):
            assert [str(subs[i]) for i in L[f][role]] == lists[str(f)][role]
    folds = [0] if args.smoke else list(range(5))
    sims = [1] if args.smoke else list(range(1, 21))
    seeds = [0] if args.smoke else [0, 1, 2]
    draws, pool_rows, eligible = make_subpools(L, y, subs, folds, sims)
    if not any(r["eligible"] for r in eligible):
        (out / "summary.json").write_text(json.dumps({"eligible": eligible, "status": "人数不足"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    save_csv(out / "subpools.csv", pool_rows)
    cfg81 = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
    orig_cal = {}
    for r in csv.DictReader((SRC / "calibration.csv").open(newline="", encoding="utf-8")):
        if r["seed"] == "0":
            orig_cal[(int(r["fold"]), r["scenario"], int(r["sim"]), r["method"])] = (float(r["c_true_ms"]), float(r["c_hat_ms"]))
    offsets = E81.make_offsets(len(y))
    cfg = dict(E50.E49.E44.CFG)
    score_ids = np.concatenate([L[f]["score"] for f in folds])
    n_score = len(score_ids)
    if not args.smoke:
        assert n_score == 400 and int(y[score_ids].sum()) == 46
    score_map = {int(v): i for i, v in enumerate(score_ids)}
    preds = defaultdict(lambda: (np.full(n_score, np.nan), np.full(n_score, np.nan)))
    cal_rows = []
    checks = {"E81_full_pool_chat_max_abs_diff_ms": 0.0, "E81_common_offset_max_abs_diff_ms": 0.0,
              "score_pool_overlap": 0, "out_of_crop_range": 0}
    pred_path = out / "predictions.csv.gz"
    with gzip.open(pred_path, "wt", newline="", encoding="utf-8") as pf:
        writer = csv.DictWriter(pf, ["patient_id", "fold", "seed", "scenario", "sim", "pool_positive_pct",
                                      "method", "label", "prob", "threshold", "pred"])
        writer.writeheader()
        for f in folds:
            tr, pool, sc = L[f]["train"], L[f]["pool"], L[f]["score"]
            checks["score_pool_overlap"] += len(np.intersect1d(pool, sc))
            trad = E81.Traditional(X)
            trad.prep(pool)
            op = E50.DelayOp(E81.FS, lag_min_ms=-200.0, lag_max_ms=300.0).to(device)
            op.load_state_dict(torch.load(SRC / "weights" / f"fold{f}_delayop.pt", map_location=device, weights_only=True))
            op.eval()
            curves = {}
            for m in METHODS:
                src = cfg81["curves"][str(f)][m]
                gt = np.asarray(src["g"], float)
                keep = np.concatenate([[True], np.diff(np.maximum.accumulate(gt)) > 0])
                curves[m] = (np.asarray(src["mu_site"], float), gt, keep)
            nets = {}
            for seed in seeds:
                net = E52.Net(False, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
                net.load_state_dict(torch.load(SRC / "weights" / f"fold{f}_seed{seed}_pinet.pt", map_location=device, weights_only=True))
                net.eval(); nets[seed] = net

            for scen in ("S0", "S2"):
                for sim, cvec, jitter in offsets[scen]:
                    if scen == "S2" and sim not in sims: continue
                    if scen == "S0" and args.smoke:
                        pass
                    true_c = float(cvec[f])
                    if scen == "S0":
                        # Twenty different pools all see the same no-offset input.
                        units = [(k, np.zeros((len(y), 4), int)) for k in sims]
                    else:
                        units = [(sim, (true_c + jitter).astype(int))]
                    for k, all_offs in units:
                        pool_offs = all_offs[pool]
                        measured = {}
                        for m in METHODS:
                            if m == "E":
                                ix = [(i, j) for i in pool for j in range(4)]
                                tau, _ = E50.op_tau(op, X, ix, pool_offs.reshape(-1), device)
                                measured[m] = tau.reshape(len(pool), 4)
                            else:
                                measured[m] = np.array([[trad.tau(i, j, int(pool_offs[a, j]), m)[0]
                                                         for j in range(4)] for a, i in enumerate(pool)])
                            old_true, old_hat = orig_cal[(f, scen, 0 if scen == "S0" else sim, m)]
                            checks["E81_common_offset_max_abs_diff_ms"] = max(checks["E81_common_offset_max_abs_diff_ms"], abs(true_c - old_true))
                            rec_hat = E81.estimate(lambda ids, off, vals=measured[m]: vals, pool, pool_offs, *curves[m])[0]
                            checks["E81_full_pool_chat_max_abs_diff_ms"] = max(checks["E81_full_pool_chat_max_abs_diff_ms"], abs(rec_hat - old_hat))
                        lookup = {int(i): a for a, i in enumerate(pool)}
                        for prop in PROPS:
                            if (f, k, prop) not in draws: continue
                            chosen = draws[(f, k, prop)]
                            loc = np.array([lookup[int(i)] for i in chosen])
                            for m in METHODS:
                                tau_subset = measured[m][loc]
                                hat, oor, failed = E81.estimate(lambda ids, off, vals=tau_subset: vals,
                                                                 chosen, all_offs[chosen], *curves[m])
                                cal_rows.append({"fold": f, "scenario": scen, "sim": k, "pool_positive_pct": prop,
                                                 "method": m, "pool_ids": " ".join(map(str, subs[chosen])),
                                                 "c_true_ms": true_c, "c_hat_ms": hat,
                                                 "error_ms": hat - true_c, "out_of_range": int(oor), "pool_failed": int(failed)})
                                corrected = all_offs[sc] - int(round(hat))
                                if np.any(np.abs(corrected) > 290):
                                    checks["out_of_crop_range"] += int((np.abs(corrected) > 290).sum())
                                    raise RuntimeError("offset outside classifier crop range")
                                for seed in seeds:
                                    p = E52.predict(nets[seed], X, sc, corrected, None, None, None, cfg, device)
                                    threshold = float(cfg81["thresholds"][f"fold{f}_seed{seed}"]["A"])
                                    key = (scen, k, prop, m, seed)
                                    pp, tt = preds[key]
                                    for i, v in zip(sc, p):
                                        a = score_map[int(i)]; pp[a] = float(v); tt[a] = threshold
                                        writer.writerow({"patient_id": str(subs[i]), "fold": f, "seed": seed,
                                                         "scenario": scen, "sim": k, "pool_positive_pct": prop,
                                                         "method": m, "label": int(y[i]), "prob": float(v),
                                                         "threshold": threshold, "pred": int(v >= threshold)})
                    if scen == "S0": break
                    print(f"fold {f} S2 sim {sim} complete ({time.time()-t0:.0f}s)", flush=True)
            print(f"fold {f} complete ({time.time()-t0:.0f}s)", flush=True)
    save_csv(out / "calibration.csv", cal_rows)
    if checks["E81_full_pool_chat_max_abs_diff_ms"] > 0.01 or checks["E81_common_offset_max_abs_diff_ms"] > 0:
        raise RuntimeError(f"E81 calibration reproduction failed: {checks}")
    if args.smoke:
        result = {"eligible": eligible, "checks": checks, "calibration_rows": len(cal_rows), "prediction_file": pred_path.name}
        (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(result)
        return
    # Classification: first merge all five folds for each (seed, sim, pool, method).
    yy = y[score_ids]
    run_metrics = {}
    for key, (p, t) in preds.items():
        assert np.isfinite(p).all() and np.isfinite(t).all()
        z = E81.metrics(yy, p, p >= t)
        run_metrics[key] = z
    run_rows = [{"scenario": scn, "sim": sim, "pool_positive_pct": prop, "method": m, "seed": seed, **v}
                for (scn, sim, prop, m, seed), v in run_metrics.items()]
    save_csv(out / "confusion.csv", run_rows)
    table = {}
    for prop in PROPS:
        for m in METHODS:
            rec = [r for r in cal_rows if r["pool_positive_pct"] == prop and r["method"] == m]
            out_m = {}
            for scen in ("S0", "S2"):
                cr = [r for r in rec if r["scenario"] == scen]
                rr = [r for r in run_rows if r["pool_positive_pct"] == prop and r["method"] == m and r["scenario"] == scen]
                out_m[scen] = {"mean_error_ms": float(np.mean([r["error_ms"] for r in cr])),
                               "mae_ms": float(np.mean([abs(r["error_ms"]) for r in cr])),
                               "out_of_range": sum(r["out_of_range"] for r in cr),
                               "pool_failed": sum(r["pool_failed"] for r in cr), "n_calibrations": len(cr),
                               **{k: float(np.mean([r[k] for r in rr])) for k in ("sen", "spe", "auroc", "mcc")}}
            table[f"{prop}:{m}"] = out_m
    d0, d20 = table["0:E"], table["20:E"]
    result = {"eligible": eligible, "n_scored": n_score, "n_pos": int(yy.sum()), "n_neg": int((yy == 0).sum()),
              "checks": checks, "table": table,
              "M5_20_minus_0": {"S0_mean_error_ms": d20["S0"]["mean_error_ms"] - d0["S0"]["mean_error_ms"],
                                "S2_sensitivity_pct_points": 100*(d20["S2"]["sen"] - d0["S2"]["sen"]),
                                "S2_specificity_pct_points": 100*(d20["S2"]["spe"] - d0["S2"]["spe"])},
              "elapsed_min": (time.time()-t0)/60}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"table": table, "M5_20_minus_0": result["M5_20_minus_0"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
