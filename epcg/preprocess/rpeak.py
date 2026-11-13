"""R-peak detection and ECG signal-quality assessment.

Used in two places:
  * Gate A -- deciding how many CinC-2016 set-a records carry a usable ECG at all.
  * The preprocessing pipeline -- computing the cardiac phase encoding phi.

Design note: R-peaks are deliberately NOT used to slice the signal into beats.
They only produce a continuous phase channel, so that a detector failure degrades
one feature instead of breaking the whole pipeline (BLUEPRINT.md §2, point 4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sps

# Physiologically plausible RR bounds: 30-200 bpm.
RR_MIN_S = 0.30
RR_MAX_S = 2.00
# Two detections of the same beat by different algorithms should land this close.
MATCH_TOL_S = 0.15


@dataclass
class EcgQuality:
    """Per-record ECG quality summary. All fields are report-safe (no NaN)."""

    usable: bool
    reason: str
    n_beats: int
    bsqi: float  # agreement between two independent detectors, 0..1
    frac_plausible_rr: float
    hr_median: float
    rr_cv: float  # coefficient of variation of RR; huge values imply misdetection
    flatline_frac: float
    duration_s: float

    def as_row(self) -> dict:
        return asdict(self)


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass, clipped to a valid Nyquist range."""
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.98)
    if not (0 < lo < hi):
        return x
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def flatline_fraction(x: np.ndarray, fs: float, win_s: float = 1.0,
                      eps_ratio: float = 1e-3) -> float:
    """Fraction of 1-second windows whose spread is negligible.

    Catches dead channels and disconnected electrodes, which in this dataset are
    the dominant failure mode and would otherwise be scored as 'no beats found'
    without saying why.
    """
    w = max(int(win_s * fs), 1)
    n = len(x) // w
    if n == 0:
        return 1.0
    trimmed = x[: n * w].reshape(n, w)
    spread = trimmed.max(axis=1) - trimmed.min(axis=1)
    global_spread = np.percentile(x, 99) - np.percentile(x, 1)
    if global_spread <= 0:
        return 1.0
    return float(np.mean(spread < eps_ratio * global_spread))


def _detect(x: np.ndarray, fs: float, method: str) -> np.ndarray:
    """Return R-peak sample indices, or an empty array if the detector fails."""
    import neurokit2 as nk

    try:
        _, info = nk.ecg_peaks(x, sampling_rate=fs, method=method, correct_artifacts=False)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=float)
        peaks = peaks[np.isfinite(peaks)].astype(int)
        return peaks
    except Exception:  # noqa: BLE001 - detector failure is an outcome, not a crash
        return np.empty(0, dtype=int)


def refine_peaks(xf: np.ndarray, peaks: np.ndarray, fs: float,
                 before_s: float, after_s: float) -> np.ndarray:
    """Move each peak to the maximum of `xf` within [p - before_s, p + after_s].

    The R fiducial is defined as the positive maximum of the band-passed QRS. On leads
    where the S wave is deeper than R this is still the R peak, not the largest absolute
    deflection, which keeps the definition the same across recordings."""
    b, a = int(round(before_s * fs)), int(round(after_s * fs))
    out = []
    for p in np.asarray(peaks, int):
        lo, hi = max(p - b, 0), min(p + a + 1, len(xf))
        if hi - lo < 2:
            continue
        out.append(lo + int(np.argmax(xf[lo:hi])))
    return np.unique(np.asarray(out, dtype=int))


def bsqi(peaks_a: np.ndarray, peaks_b: np.ndarray, fs: float,
         tol_s: float = MATCH_TOL_S) -> float:
    """Jaccard agreement between two detectors' peak sets.

    The standard bSQI. Near 1.0 means two independent algorithms see the same
    beats, which is strong evidence of real ECG; near 0 means at least one is
    tracking noise.
    """
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return 0.0
    tol = tol_s * fs
    a = np.sort(peaks_a.astype(float))
    b = np.sort(peaks_b.astype(float))
    idx = np.searchsorted(b, a)
    matched = 0
    for i, ai in enumerate(a):
        best = np.inf
        for j in (idx[i] - 1, idx[i]):
            if 0 <= j < len(b):
                best = min(best, abs(b[j] - ai))
        if best <= tol:
            matched += 1
    union = len(a) + len(b) - matched
    return float(matched / union) if union > 0 else 0.0


def assess_ecg(x: np.ndarray, fs: float,
               min_beats: int = 5,
               bsqi_thresh: float = 0.80,
               rr_thresh: float = 0.90) -> tuple[EcgQuality, np.ndarray]:
    """Judge whether a channel carries usable ECG, and return its R-peaks.

    Returns (quality, rpeak_indices). ``quality.usable`` is the Gate A criterion:
    two independent detectors must agree (bSQI) AND the resulting rhythm must be
    physiologically plausible. Requiring both rejects the common failure where a
    detector locks onto periodic noise and produces confident nonsense.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    dur = len(x) / fs if fs else 0.0
    empty = np.empty(0, dtype=int)

    def fail(reason: str, **kw) -> tuple[EcgQuality, np.ndarray]:
        base = dict(usable=False, reason=reason, n_beats=0, bsqi=0.0,
                    frac_plausible_rr=0.0, hr_median=0.0, rr_cv=0.0,
                    flatline_frac=1.0, duration_s=dur)
        base.update(kw)
        return EcgQuality(**base), empty

    if len(x) < int(2 * fs) or fs <= 0:
        return fail("too_short")
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    flat = flatline_fraction(x, fs)
    if flat > 0.5:
        return fail("flatline_or_dead_channel", flatline_frac=flat)

    xf = bandpass(x, fs)
    if np.std(xf) <= 0:
        return fail("zero_variance", flatline_frac=flat)
    xf = (xf - np.mean(xf)) / np.std(xf)

    p1 = _detect(xf, fs, "neurokit")
    p2 = _detect(xf, fs, "pantompkins1985")
    agreement = bsqi(p1, p2, fs)
    # ONE fiducial for every recording (E54). Both detectors still vote on quality
    # (bSQI above), but the returned positions always come from NeuroKit, refined to the
    # local positive maximum of the band-passed ECG. The previous rule -- keep whichever
    # detector returned more peaks -- switched to Pan-Tompkins on 53% of recordings,
    # whose fiducial sits a median 32 ms after the QRS peak, so every downstream R->X
    # interval carried a detector-dependent offset. Pan-Tompkins is used only when
    # NeuroKit finds too few beats, and is then pulled back onto the peak it trails.
    if len(p1) >= min_beats:
        peaks = refine_peaks(xf, p1, fs, before_s=0.015, after_s=0.015)
    else:
        peaks = refine_peaks(xf, p2, fs, before_s=0.060, after_s=0.010)

    if len(peaks) < min_beats:
        return fail("too_few_beats", n_beats=int(len(peaks)),
                    bsqi=agreement, flatline_frac=flat)

    rr = np.diff(peaks) / fs
    plausible = float(np.mean((rr >= RR_MIN_S) & (rr <= RR_MAX_S))) if len(rr) else 0.0
    hr = float(60.0 / np.median(rr)) if len(rr) and np.median(rr) > 0 else 0.0
    cv = float(np.std(rr) / np.mean(rr)) if len(rr) and np.mean(rr) > 0 else 0.0

    ok = agreement >= bsqi_thresh and plausible >= rr_thresh
    reason = "ok" if ok else (
        "low_detector_agreement" if agreement < bsqi_thresh else "implausible_rhythm"
    )
    return EcgQuality(
        usable=ok, reason=reason, n_beats=int(len(peaks)), bsqi=round(agreement, 4),
        frac_plausible_rr=round(plausible, 4), hr_median=round(hr, 2),
        rr_cv=round(cv, 4), flatline_frac=round(flat, 4), duration_s=round(dur, 2),
    ), peaks
