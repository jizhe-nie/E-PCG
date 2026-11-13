#!/usr/bin/env python
"""E78 -- device self-calibration in front of two more third-party fusion networks (PACFNet, HPA-Net).

    python experiments/E78_selfcal_thirdparty/run.py --model hpanet  [--smoke]
    python experiments/E78_selfcal_thirdparty/run.py --model pacfnet [--smoke]

WHY. The claim that the self-calibration front end is model-agnostic rests on one model (E65, LGC2-Net). E74/E76
showed that PACFNet and HPA-Net collapse under offsets just as LGC2-Net does; this asks whether the same front
end, unchanged, repairs them too.

SCENARIO. Exactly E65's: E61's uncalibrated wireless-audio scenario (device latency c ~ U(40, 200) ms per
subject's device + U(-30, 30) ms per recording), and E65's saved correction maps (experiments/E65_lgc2_selfcal/
e65_shifts.npz, operator trained per fold, c_hat from N = 20 other test-fold subjects, E63b curve inversion).
The PCG is read at (offset - correction) ms, per subject and site:
  none            clean reference (threshold frozen here)
  audio_raw       offset, no correction
  audio_band50    E50 tolerance rule on the operator's d_hat (correct only the part outside +-50 ms)
  audio_selfcal   device self-calibration c_hat (one constant per subject's device, all four sites)
  audio_calib     true constant latency c removed (oracle; jitter remains)
  none_selfcal    self-calibration applied on a device with no latency (the cost of the front end on clean input)
MODELS. Retrained exactly as E74 (PACFNet, official-code port) / E76 (HPA-Net reconstruction): same seeds,
same validation subjects, same recipe -- the E74/E76 runs did not keep weights. Weights are saved this time
(weights/, not committed). Threshold: 90% sensitivity on each seed's own clean out-of-fold scores.
Statistics: experiments/_pairstats.py (seed mean of per-seed paired differences, subject bootstrap).

PRE-SPECIFIED (stated before running; reported either way)
  PRIMARY  per model: audio_selfcal minus audio_raw, specificity; supported if the 95% CI lies above 0
  SECONDARY  audio_selfcal minus audio_band50 (sensitivity and specificity, the E65 comparison);
             audio_selfcal minus audio_calib; none_selfcal minus none (cost on clean input)
  CHECK    clean AUROC within 0.02 of E74/E76 (reproducibility of the retraining; GPU nondeterminism)
NOTE (HPA-Net) windows whose offset PCG falls outside the buffer are dropped as in E76, so at audio offsets
  > 100 ms a site contributes 8 instead of 9 windows.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent
EXP = OUT.parent
sys.path.insert(0, str(EXP))
from _pairstats import boots, paired  # noqa: E402

CONDS = ["none", "audio_raw", "audio_band50", "audio_selfcal", "audio_calib", "none_selfcal"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXP / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def shift_maps(n):
    """(n, 4) int ms per condition, rebuilt from E65's saved arrays exactly as E65's maps() did."""
    z = np.load(EXP / "E65_lgc2_selfcal" / "e65_shifts.npz")
    off, c, DH = z["off"], z["c"], z["dhat"]
    band = _load("e55_run", "E55_correction/run.py").band
    corr = {"audio_raw": np.zeros_like(DH), "audio_band50": band(DH, 50, 50),
            "audio_selfcal": z["c_hat_audio"][:, None] * np.ones((1, 4)), "audio_calib": c}
    S = {m: (off - np.round(v)).astype(int) for m, v in corr.items()}
    S["none_selfcal"] = (-np.round(z["c_hat_none"][:, None] * np.ones((1, 4)))).astype(int)
    S["none"] = np.zeros((n, 4), int)
    assert len(off) == n and list(z["subjects"]) == list(map(str, SUBS))
    return S


def main() -> int:
    global SUBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pacfnet", "hpanet"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    e12 = _load("e12", "E12_pinet/run.py")
    d = e12.build_cache(False)
    X, y, fold = d["X"], d["y"], d["fold"]
    SUBS = d["subjects"]
    S = shift_maps(len(y))
    if args.model == "pacfnet":
        M = _load("e74_run", "E74_pacfnet/run.py")
        SEED, prior = M.SEED, EXP / "E74_pacfnet" / "e74_pacfnet.json"
        unit = lambda i, off: M.beats_of(X, i, off)
        fit = lambda tr_, va_, s: M.train(*tr_, *va_, 2 if args.smoke else 100, device, s)
        score = lambda m, u: M.subject_prob(m, u[0], u[1], device)
    else:
        M = _load("e76_run", "E76_hpanet/run.py")
        SEED, prior = M.SEED, EXP / "E76_hpanet" / "e76_hpanet.json"
        E, P = M.prep(X)
        unit = lambda i, off: M.windows_of(E, P, i, off)
        fit = lambda tr_, va_, s: M.train(*tr_, *va_, 2 if args.smoke else 100, device, s)
        score = lambda m, u: float(M.run_batches(m, *u, device).mean()) if len(u[0]) else np.nan
    subs = np.arange(len(y)) if not args.smoke else np.flatnonzero(fold <= 1)[:160]
    clean = {i: unit(i, S["none"][i]) for i in subs}
    print(f"{args.model}: clean units cached ({time.time() - t0:.0f}s)", flush=True)
    seeds = [SEED + 101 * k for k in range(1 if args.smoke else args.seeds)]
    PR = {(s, args.model, c): np.full(len(y), np.nan) for s in seeds for c in CONDS}
    (OUT / "weights").mkdir(exist_ok=True); (OUT / "partial").mkdir(exist_ok=True)
    for f in ([0] if args.smoke else range(5)):
        te = np.intersect1d(np.flatnonzero(fold == f), subs); tr_all = np.intersect1d(np.flatnonzero(fold != f), subs)
        va = np.random.default_rng(SEED + f).choice(tr_all, size=max(2, len(tr_all) // 10), replace=False)
        tr = np.setdiff1d(tr_all, va)
        stack = lambda ids: (np.concatenate([clean[i][0] for i in ids]), np.concatenate([clean[i][1] for i in ids]),
                             np.concatenate([np.full(len(clean[i][0]), y[i], np.int64) for i in ids]))
        test = None
        for s in seeds:
            tag = f"{args.model}_f{f}_s{s}{'_smoke' if args.smoke else ''}"
            ck = OUT / "partial" / f"{tag}.npz"
            if ck.exists():
                z = np.load(ck)
                for c in CONDS:
                    PR[(s, args.model, c)][te] = z[c]
                print(f"  fold {f} seed {s}: resumed", flush=True)
                continue
            m, n_ep = fit(stack(tr), stack(va), s)
            torch.save(m.state_dict(), OUT / "weights" / f"{tag}.pt")
            if test is None:
                test = {c: {i: (clean[i] if c == "none" else unit(i, S[c][i])) for i in te} for c in CONDS}
            for c in CONDS:
                PR[(s, args.model, c)][te] = [score(m, test[c][i]) for i in te]
            np.savez(ck, **{c: PR[(s, args.model, c)][te] for c in CONDS})
            print(f"  fold {f} seed {s}: {n_ep} epochs ({time.time() - t0:.0f}s)", flush=True)
    k_ = np.isfinite(PR[(seeds[0], args.model, "none")])
    for c in CONDS:                                    # a subject with no unit in some condition is dropped everywhere
        k_ &= np.isfinite(PR[(seeds[0], args.model, c)])
    yy = y[k_]
    Pk = {key: v[k_] for key, v in PR.items()}
    def _m(cond, s, idx):
        """(sen, spe) of condition cond at seed s's threshold: 90% sensitivity on its own clean scores (resample idx)."""
        yb = y[k_][idx]
        pc, p = Pk[(s, args.model, "none")][idx], Pk[(s, args.model, cond)][idx]
        t = np.sort(pc[yb == 1])[::-1][int(np.ceil(0.9 * yb.sum())) - 1]
        return (p[yb == 1] >= t).mean(), (p[yb == 0] < t).mean()

    bs = boots(yy, B=200 if args.smoke else 2000, seed=SEED)
    full = np.arange(len(yy))
    res = {"model": args.model, "n": int(k_.sum()), "n_pos": int(yy.sum()), "seeds": seeds, "eval": {}, "paired": {}}
    for c in CONDS:
        ss = [_m(c, s, full) for s in seeds]
        au = [roc_auc_score(yy, Pk[(s, args.model, c)]) for s in seeds]
        res["eval"][c] = {"auroc": [round(float(np.mean(au)), 4), round(float(np.std(au)), 4)],
                          "sen": [round(float(np.mean([v[0] for v in ss])), 4), round(float(np.std([v[0] for v in ss])), 4)],
                          "spe": [round(float(np.mean([v[1] for v in ss])), 4), round(float(np.std([v[1] for v in ss])), 4)]}
    for a, b in (("audio_selfcal", "audio_raw"), ("audio_selfcal", "audio_band50"), ("audio_selfcal", "audio_calib"),
                 ("none_selfcal", "none")):
        res["paired"][f"{a}-{b}"] = paired(_m, a, b, seeds, len(yy), bs)
    res["primary_supported"] = bool(res["paired"]["audio_selfcal-audio_raw"]["spe_ci"][0] > 0)
    if prior.exists():
        pa = json.loads(prior.read_text(encoding="utf-8"))["eval"]["none"]["auroc"][0]
        res["check_clean_auroc_vs_prior"] = {"prior": pa, "now": res["eval"]["none"]["auroc"][0],
                                             "within_0.02": bool(abs(pa - res["eval"]["none"]["auroc"][0]) <= 0.02)}
    for c in CONDS:
        print(f"  {c:14s} " + "  ".join(f"{k} {100 * v[0]:.1f}" if k != "auroc" else f"AUROC {v[0]:.3f}"
                                        for k, v in res["eval"][c].items()))
    print(json.dumps(res["paired"], indent=1), res["primary_supported"], res.get("check_clean_auroc_vs_prior"))
    tag = "_smoke" if args.smoke else ""
    (OUT / f"e78_{args.model}{tag}.json").write_text(json.dumps({**res, "elapsed_min": round((time.time() - t0) / 60, 2)},
                                                               indent=2), encoding="utf-8")
    np.savez_compressed(OUT / f"e78_{args.model}_scores{tag}.npz", y=yy,
                        **{f"{s}__{c}": Pk[(s, args.model, c)] for s in seeds for c in CONDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
