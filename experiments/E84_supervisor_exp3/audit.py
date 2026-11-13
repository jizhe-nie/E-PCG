"""Experiment 3, part 1: reconcile the E47 R2 and E73 CCC source predictions.

Read-only with respect to prior experiments. Writes the audit alongside this file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def measures(ref: np.ndarray, pred: np.ndarray) -> dict:
    valid = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[valid].astype(float), pred[valid].astype(float)
    delta = pred - ref
    ccc = 2 * np.mean((pred - pred.mean()) * (ref - ref.mean())) / (
        pred.var() + ref.var() + (pred.mean() - ref.mean()) ** 2
    )
    return {
        "n_site_recordings": int(len(ref)),
        "r2_coefficient_of_determination": float(r2_score(ref, pred)),
        "ccc": float(ccc),
        "bias_prediction_minus_reference_ms": float(delta.mean()),
        "mae_ms": float(np.abs(delta).mean()),
        "pearson_r": float(np.corrcoef(pred, ref)[0, 1]),
    }


def main() -> None:
    e47 = ROOT / "experiments/E47_crop_shift"
    e73 = ROOT / "experiments/E73_bland_altman"
    with (e47 / "e47_crop_shift_oof.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref47 = np.array([float(r["emat_panel_ms"]) for r in rows])
    pred47 = np.array([float(r["emat_delay_op_ms"]) for r in rows])
    with np.load(e73 / "e73_pairs.npz", allow_pickle=False) as z:
        ref73, pred73 = z["main_ref"].copy(), z["main_op"].copy()
    assert len(rows) == len(ref73) == 1700
    assert len({r["subject"] for r in rows}) == 425
    assert np.max(np.abs(ref47 - ref73)) < 0.001
    source47 = json.loads((e47 / "e47_crop_shift.json").read_text(encoding="utf-8"))
    source73 = json.loads((e73 / "e73_bland_altman.json").read_text(encoding="utf-8"))
    result = {
        "sources": {
            "E47": "experiments/E47_crop_shift/e47_crop_shift_oof.csv",
            "E73": "experiments/E73_bland_altman/e73_pairs.npz",
        },
        "unit": "one site recording; 425 subjects x 4 sites",
        "reference_max_abs_difference_ms": float(np.max(np.abs(ref47 - ref73))),
        "E47": measures(ref47, pred47),
        "E73": measures(ref73, pred73),
        "source_rounding": {
            "E47_reported_r2": source47["arms"]["delay_op"]["r2"],
            "E73_reported_ccc": source73["main"]["ccc"],
        },
        "configuration": {
            "E47": {
                "source": "experiments/E47_crop_shift/run.py",
                "operator": "epcg.models.emat_arms.build('delay_op', 1000)",
                "lag_window_ms": [0, 250],
                "seed": 20260930,
                "training": "40 epochs, random 10 s crops, normalized SmoothL1, AdamW 1e-3, OneCycleLR; final epoch",
                "evaluation": "clean 10 s crop starting at 1 s",
                "weights_saved": False,
            },
            "E73": {
                "source": "experiments/E73_bland_altman/run.py; experiments/E50_resync/run.py",
                "operator": "epcg.models.emat_arms.DelayOp(1000, lag_min_ms=-200, lag_max_ms=300)",
                "lag_window_ms": [-200, 300],
                "seed": 20261002,
                "training": "40 epochs, random 10 s crops, normalized SmoothL1, AdamW 1e-3, OneCycleLR; final epoch",
                "evaluation": "clean 10 s crop starting at 10 s",
                "weights_saved": False,
            },
            "E81": {
                "source": "experiments/E81_controlled/run.py; experiments/E50_resync/run.py",
                "operator": "same operator construction as E73, retrained on E81 training-only lists",
                "lag_window_ms": [-200, 300],
                "seed": 20261002,
                "training": "40 epochs, random 10 s crops, normalized SmoothL1, AdamW 1e-3, OneCycleLR; final epoch",
                "evaluation": "operator 10-20 s of 30 s buffer; classifier three 70%-length crops",
                "weights_saved": True,
            },
            "calibration": "E81 g: -40..250 ms every 10 ms; strict increases of cumulative maximum; median of (tau-mu_site); linear inverse; outside clamp; all-invalid pool -> zero",
            "other": "E81 class weights: negative=1, positive=n_negative/n_positive; PINet/dual 40 epochs, AdamW 3e-4, 70% crop, 3-crop TTA; alignment clipped to +/-150 ms feature and fed offset +/-290 ms",
        },
        "finding": "The two manuscript values use the same record-level reference and sample count, but different trained operator configurations and predictions. Do not combine as one model's metrics.",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("E47", "E73", "finding")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
