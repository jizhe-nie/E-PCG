"""Experiment 3, part 2: A1/A2/A3/A4 on the locked E81 protocol.

Run --smoke first. Full run trains only A2; A1/A3 predictions and weights are
frozen E81 outputs. A4 changes only A3's inference flag.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.E81_controlled import run as E81
from experiments.E83_community import run as E83

E50, E52 = E81.E50, E81.E52
OUT = Path(__file__).parent
SRC = ROOT / "experiments/E81_controlled/full"
METHODS = ("A1", "A2", "A3", "A4")


def csvwrite(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_reference():
    rows = list(csv.DictReader((SRC / "predictions.csv").open(newline="", encoding="utf-8")))
    ref = {}
    for r in rows:
        if r["method"] in ("E", "H"):
            ref[(int(r["fold"]), int(r["seed"]), r["scenario"], int(r["sim"]),
                 r["method"], r["patient_id"])] = float(r["prob"])
    cal = {}
    for r in csv.DictReader((SRC / "calibration.csv").open(newline="", encoding="utf-8")):
        if r["seed"] == "0" and r["method"] == "E":
            cal[(int(r["fold"]), r["scenario"], int(r["sim"]))] = float(r["c_hat_ms"])
    return ref, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT / ("smoke_modules" if args.smoke else "full_modules")
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)
    d = E50.E49.E44._E12.build_cache(False)
    X, y, fold, subs = d["X"], d["y"], d["fold"], d["subjects"]
    L = E81.make_lists(y, fold)
    lists = json.loads((SRC / "lists.json").read_text(encoding="utf-8"))
    for f in range(5):
        for role in ("train", "val", "pool", "score"):
            assert [str(subs[i]) for i in L[f][role]] == lists[str(f)][role]
    offs = E81.make_offsets(len(y))
    cfg = dict(E50.E49.E44.CFG)
    cfg81 = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
    ref, cal = load_reference()
    folds = [0] if args.smoke else range(5)
    seeds = [0] if args.smoke else (0, 1, 2)
    scens = ("S0", "S1") if args.smoke else ("S0", "S1", "S2")
    pred_rows, threshold_rows, check = [], [], {"A1_max_abs_prob_diff": 0.0, "A3_max_abs_prob_diff": 0.0}

    for f in folds:
        tr, va, sc = (L[f][r] for r in ("train", "val", "score"))
        op = E50.DelayOp(E81.FS, lag_min_ms=-200.0, lag_max_ms=300.0).to(device)
        op.load_state_dict(torch.load(SRC / "weights" / f"fold{f}_delayop.pt", map_location=device, weights_only=True))
        op.eval()
        mu, slope = E50.calibrate(op, X, tr, device)

        def measure(ids, shift):
            ix = [(i, j) for i in ids for j in range(4)]
            tau, sg = E50.op_tau(op, X, ix, np.asarray(shift).reshape(-1).astype(int), device)
            dh = np.clip((tau.reshape(-1, 4) - mu[None]) / slope, -E52.CLIP_MS, E52.CLIP_MS)
            return dh, sg.reshape(-1, 4)

        dh_tr, sg_tr = measure(tr, np.zeros((len(tr), 4), int))
        fm = np.array([dh_tr.mean(), sg_tr.mean()]); fsd = np.array([dh_tr.std(), sg_tr.std()]) + 1e-6
        feat = lambda dh, sg: ((np.stack([dh, sg], -1) - fm) / fsd).astype(np.float32)
        T2 = np.zeros((len(y), 4, 2), np.float32)
        T2[tr] = feat(dh_tr, sg_tr)
        A_al = np.zeros((len(y), 4), int)
        A_al[tr] = np.round(dh_tr).astype(int)

        def inputs(ids, shift):
            dh, sg = measure(ids, shift)
            align = np.clip(np.round(dh).astype(int), shift - 290, shift + 290)
            return align, feat(dh, sg)

        for seed in seeds:
            a2 = E52.train(X, y, tr, T2, A_al, None, cfg, seed, device)
            torch.save(a2.state_dict(), out / "weights" / f"fold{f}_seed{seed}_A2.pt")
            dual = E52.Net(True, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
            dual.load_state_dict(torch.load(SRC / "weights" / f"fold{f}_seed{seed}_dual.pt", map_location=device, weights_only=True))
            dual.eval()
            pinet = E52.Net(False, dim=cfg["dim"], dropout=cfg["dropout"]).to(device)
            pinet.load_state_dict(torch.load(SRC / "weights" / f"fold{f}_seed{seed}_pinet.pt", map_location=device, weights_only=True))
            pinet.eval()
            zv = np.zeros((len(va), 4), int)
            alv, _ = inputs(va, zv)
            val_prob = E52.predict(a2, X, va, zv, alv, None, None, cfg, device)
            thr = E81.pick_threshold(val_prob, y[va])
            locked = cfg81["thresholds"][f"fold{f}_seed{seed}"]
            threshold_rows.append({"fold": f, "seed": seed, "A1": locked["E"], "A2": thr,
                                   "A3": locked["H"], "A4": locked["H"]})
            for scen in scens:
                for sim, cvec, jit in offs[scen]:
                    injected = (float(cvec[f]) + jit[sc]).astype(int)
                    corrected = injected - int(round(cal[(f, scen, sim)]))
                    if np.any(np.abs(corrected) > 290):
                        raise ValueError("classification crop out of range")
                    al, tf = inputs(sc, corrected)
                    p2 = E52.predict(a2, X, sc, corrected, al, None, None, cfg, device)
                    p4 = E52.predict(dual, X, sc, corrected, al, tf, 0.0, cfg, device)
                    # The full run reuses exact E81 predictions. The smoke run first
                    # reproduces both of them with the saved E81 weights.
                    if args.smoke:
                        p1r = E52.predict(pinet, X, sc, corrected, None, None, None, cfg, device)
                        p3r = E52.predict(dual, X, sc, corrected, al, tf, 1.0, cfg, device)
                    for i, p2i, p4i in zip(sc, p2, p4):
                        pid = str(subs[i])
                        p1 = ref[(f, seed, scen, sim, "E", pid)]
                        p3 = ref[(f, seed, scen, sim, "H", pid)]
                        for m, p, threshold in (("A1", p1, locked["E"]), ("A2", float(p2i), thr),
                                                ("A3", p3, locked["H"]), ("A4", float(p4i), locked["H"])):
                            pred_rows.append({"patient_id": pid, "fold": f, "seed": seed, "scenario": scen,
                                              "sim": sim, "method": m, "label": int(y[i]), "prob": p,
                                              "threshold": threshold, "pred": int(p >= threshold)})
                    if args.smoke:
                        for m, p in (("A1", p1r), ("A3", p3r)):
                            source = "E" if m == "A1" else "H"
                            err = max(abs(float(v) - ref[(f, seed, scen, sim, source, str(subs[i]))])
                                      for i, v in zip(sc, p))
                            check[f"{m}_max_abs_prob_diff"] = max(check[f"{m}_max_abs_prob_diff"], err)
            print(f"fold {f} seed {seed} complete ({time.time()-t0:.0f}s)", flush=True)
    if args.smoke and max(check.values()) > 1e-5:
        raise RuntimeError(f"E81 reproduction failed: {check}")
    csvwrite(out / "predictions.csv", pred_rows)
    csvwrite(out / "thresholds.csv", threshold_rows)
    if args.smoke:
        (out / "summary.json").write_text(json.dumps({"checks": check, "n_scored": len(L[0]["score"])}, indent=2), encoding="utf-8")
        print(check)
        return

    # Group by scenario/run and pool all five folds before metrics.
    score_ids = np.concatenate([L[f]["score"] for f in range(5)])
    assert len(score_ids) == 400 and int(y[score_ids].sum()) == 46
    idx = {str(subs[i]): j for j, i in enumerate(score_ids)}
    grouped = {}
    for r in pred_rows:
        key = (r["scenario"], int(r["seed"]), int(r["sim"]), r["method"])
        if key not in grouped:
            grouped[key] = (np.full(400, np.nan), np.full(400, False))
        grouped[key][0][idx[r["patient_id"]]] = float(r["prob"])
        grouped[key][1][idx[r["patient_id"]]] = bool(r["pred"])
    yy = y[score_ids]
    tables, conf = {}, []
    for scen in scens:
        for m in METHODS:
            runs = []
            for (s, sd, sim, mm), (p, decision) in grouped.items():
                if s != scen or mm != m: continue
                assert np.isfinite(p).all()
                z = E81.metrics(yy, p, decision)
                conf.append({"scenario": scen, "seed": sd, "sim": sim, "method": m, **z})
                runs.append(z)
            tables.setdefault(scen, {})[m] = {k: float(np.mean([v[k] for v in runs]))
                                             for k in ("sen", "spe", "auroc", "mcc")}
    csvwrite(out / "confusion.csv", conf)
    from scipy.stats import scoreatpercentile
    folds_sc = fold[score_ids]
    groups = [np.flatnonzero((folds_sc == f) & (yy == cls)) for f in range(5) for cls in (0, 1)]
    W = E83.boot_weights(groups, 400, 5000, 20261103)
    pair = {}
    for a, b in (("A2", "A1"), ("A3", "A2"), ("A3", "A4")):
        runs = sorted((sd, sim) for s, sd, sim, m in grouped if s == "S2" and m == a)
        aa = np.stack([grouped[("S2", sd, sim, a)][1] for sd, sim in runs])
        bb = np.stack([grouped[("S2", sd, sim, b)][1] for sd, sim in runs])
        va, vb = E83.boot_means(W, yy, aa), E83.boot_means(W, yy, bb)
        pa, pb = E83.boot_means(np.ones((1, 400), np.float32), yy, aa), E83.boot_means(np.ones((1, 400), np.float32), yy, bb)
        pair[f"{a}-{b}"] = {k: {"diff": float((pa[k]-pb[k])[0]),
                               "ci": [float(scoreatpercentile(va[k]-vb[k], q)) for q in (2.5, 97.5)]}
                           for k in ("sen", "spe", "mcc")}
    result = {"n_scored": 400, "n_pos": 46, "n_neg": 354, "tables": tables, "pairs_S2": pair,
              "bootstrap": "5000 paired subject resamples, stratified by fold and label; fixed models, pool, thresholds",
              "A1_A3_reuse": "exact E81 full predictions; smoke check reports reproduction against saved E81 weights",
              "elapsed_min": (time.time()-t0)/60}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tables": tables, "pairs_S2": pair}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
