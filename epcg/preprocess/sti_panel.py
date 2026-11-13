"""Systolic time intervals and S3 from a consumer electronic stethoscope.

Cardiology has measured heart failure acoustically for decades. The quantities are
established; what is not established is whether they can be extracted reliably from a
consumer device (EKO DUO) with nothing but signal processing, which is what decides
whether acoustic screening can leave the clinic.

    EMAT      R peak -> S1        isovolumic contraction; PROLONGS in HFrEF
    LVET      S1 -> S2            left ventricular ejection time; SHORTENS in HFrEF
    EMAT/LVET                     systolic dysfunction index; rises from both ends
    LVETi                         rate-corrected LVET (Weissler): LVET + 1.7 * HR
    S1 / S2 strength              S1 weakens with reduced contractility
    S3 index                      third heart sound -- the classic HF sign

S3 deserves its own treatment and gets it below. It is the quietest thing here and
the easiest to invent by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sps

# Search windows, in ms relative to their anchor. Generous on the side where the
# pathology pushes, so a prolonged or shortened interval stays measurable.
S1_AFTER_R = (0.0, 250.0)
S2_AFTER_S1 = (150.0, 500.0)
S3_AFTER_S2 = (100.0, 250.0)      # early diastole, rapid filling phase

# S1/S2 are valve-closure transients with energy well above 100 Hz. S3 is a
# low-frequency ventricular wall vibration, typically 25-70 Hz, and separating the
# bands is what stops a loud S2 tail from being scored as an S3.
BAND_S12 = (20.0, 400.0)
BAND_S3 = (25.0, 70.0)
ENV_LP = 20.0


@dataclass
class StiPanel:
    n_beats: int
    emat_ms: float
    emat_sd_ms: float
    lvet_ms: float
    lvet_sd_ms: float
    lveti_ms: float            # rate-corrected
    emat_over_lvet: float      # systolic dysfunction index
    s1_strength: float
    s2_strength: float
    s3_index: float            # S3 energy relative to the diastolic floor
    s3_prevalence: float       # fraction of beats with a detectable S3
    s3_timing_ms: float        # S2 -> S3, when present
    hr_bpm: float

    def as_row(self) -> dict:
        return asdict(self)


def _env(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    nyq = fs / 2
    sos = sps.butter(4, [band[0] / nyq, min(band[1], nyq * .98) / nyq],
                     btype="band", output="sos")
    e = np.abs(sps.hilbert(sps.sosfiltfilt(sos, x)))
    sos2 = sps.butter(2, ENV_LP / nyq, btype="low", output="sos")
    return sps.sosfiltfilt(sos2, e)


def _peak(env: np.ndarray, start: int, lo_ms: float, hi_ms: float, fs: float,
          limit: int | None = None) -> tuple[int, float] | None:
    """Interior maximum of the envelope in a window. Edge maxima are rejected: a peak
    on the boundary usually means the real one is outside, and accepting it drags the
    interval toward the window edge."""
    a = start + int(lo_ms / 1000 * fs)
    b = start + int(hi_ms / 1000 * fs)
    if limit is not None:
        b = min(b, limit)
    if a < 0 or b > len(env) or b - a < 5:
        return None
    seg = env[a:b]
    k = int(np.argmax(seg))
    if k == 0 or k == len(seg) - 1:
        return None
    return a + k, float(seg[k])


@dataclass
class BeatRow:
    """One heartbeat. The unit a multiple-instance model would treat as an instance."""

    beat: int
    rr_ms: float
    emat_ms: float
    lvet_ms: float
    emat_over_lvet: float
    s1_amp: float
    s2_amp: float
    s3_z: float          # early-diastolic low-band peak, in SDs above this beat's floor
    s3_timing_ms: float  # S2 -> S3
    s3_detected: int


def extract_beats(pcg: np.ndarray, fs: float, rpeaks_s: np.ndarray,
                  s3_z: float = 2.0) -> list[BeatRow]:
    """Per-beat measurements, before any aggregation.

    :func:`extract` reduces these to a median, which is the crudest pooling available
    and is exactly wrong for an intermittent sign. S3 appears on a minority of beats --
    measured prevalence 29.6% in HFrEF against 16.8% otherwise, the strongest single
    association in the panel -- so a median over 35 beats discards the quantity that
    carries the most information. Keeping the beats makes the bag-of-instances
    structure available, and multiplies the data seen by a representation learner by
    the number of beats per recording even though the number of labels is unchanged.
    """
    x = np.asarray(pcg, float).ravel()
    e12 = _env(x, fs, BAND_S12)
    e3 = _env(x, fs, BAND_S3)
    e12 = (e12 - e12.mean()) / (e12.std() or 1.0)
    e3 = (e3 - e3.mean()) / (e3.std() or 1.0)
    r = np.sort(np.asarray(rpeaks_s, float))
    rr = np.diff(r)
    out: list[BeatRow] = []

    for i, t in enumerate(r[:-1]):
        r0, nxt = int(t * fs), int(r[i + 1] * fs)
        if rr[i] < 0.30 or rr[i] > 2.0:
            continue
        p1 = _peak(e12, r0, *S1_AFTER_R, fs, limit=nxt)
        if p1 is None:
            continue
        i1, a1 = p1
        p2 = _peak(e12, i1, *S2_AFTER_S1, fs,
                   limit=min(nxt, r0 + int(0.60 * rr[i] * fs)))
        if p2 is None:
            continue
        i2, a2 = p2
        emat = 1000.0 * (i1 - r0) / fs
        lvet = 1000.0 * (i2 - i1) / fs

        z, tim, det = float("nan"), float("nan"), 0
        a = i2 + int(S3_AFTER_S2[0] / 1000 * fs)
        b = min(i2 + int(S3_AFTER_S2[1] / 1000 * fs), nxt)
        if b - a >= 5 and nxt - b >= 5 and nxt <= len(e3):
            seg, floor = e3[a:b], e3[b:nxt]
            fmu, fsd = float(floor.mean()), float(floor.std() or 1e-6)
            k = int(np.argmax(seg))
            z = (float(seg[k]) - fmu) / fsd
            if z >= s3_z and 0 < k < len(seg) - 1:
                det, tim = 1, 1000.0 * (a + k - i2) / fs

        out.append(BeatRow(
            beat=i, rr_ms=round(1000.0 * rr[i], 2),
            emat_ms=round(emat, 2), lvet_ms=round(lvet, 2),
            emat_over_lvet=round(emat / max(lvet, 1e-6), 5),
            s1_amp=round(a1, 4), s2_amp=round(a2, 4),
            s3_z=round(z, 4) if np.isfinite(z) else float("nan"),
            s3_timing_ms=round(tim, 2) if np.isfinite(tim) else float("nan"),
            s3_detected=det))
    return out


def extract(pcg: np.ndarray, fs: float, rpeaks_s: np.ndarray,
            s3_z: float = 2.0) -> StiPanel:
    """Beat-by-beat panel, then robust central tendency across beats.

    S3 DETECTION, AND WHY IT IS DONE THIS WAY
    -----------------------------------------
    S3 is a low-amplitude, low-frequency vibration in early diastole. Three things
    make it easy to fabricate, and each is handled explicitly:

      * A loud S2 has a decaying tail that reaches into the S3 window. Working in a
        25-70 Hz band, where S2's valve-closure energy is comparatively weak,
        separates them; the search also starts 100 ms after S2.
      * Any envelope has a maximum somewhere, so "the largest value in the S3 window"
        always exists. A beat only counts as having an S3 if that maximum stands
        `s3_z` standard deviations above the LATE-diastolic floor of the same beat --
        a per-beat, per-subject reference that needs no global threshold.
      * A single noisy beat can dominate. The index is the median over beats and is
        reported alongside the fraction of beats in which S3 was detectable, so a
        strong index resting on two beats is visible as such.

    No expert S3 annotations exist for this dataset, so this is a continuous index,
    not a diagnosis. It is not claimed to be a validated S3 detector.
    """
    x = np.asarray(pcg, float).ravel()
    e12 = _env(x, fs, BAND_S12)
    e3 = _env(x, fs, BAND_S3)
    e12 = (e12 - e12.mean()) / (e12.std() or 1.0)
    e3 = (e3 - e3.mean()) / (e3.std() or 1.0)

    r = np.sort(np.asarray(rpeaks_s, float))
    rr = np.diff(r)
    hr = float(60.0 / np.median(rr)) if len(rr) and np.median(rr) > 0 else np.nan

    emat, lvet, s1a, s2a, s3v, s3t = [], [], [], [], [], []
    n_s3 = 0
    for i, t in enumerate(r[:-1]):
        r0 = int(t * fs)
        nxt = int(r[i + 1] * fs)
        if rr[i] < 0.30 or rr[i] > 2.0:
            continue
        p1 = _peak(e12, r0, *S1_AFTER_R, fs, limit=nxt)
        if p1 is None:
            continue
        i1, a1 = p1
        # S2 must still leave room for diastole; cap the search at 60% of this RR.
        p2 = _peak(e12, i1, *S2_AFTER_S1, fs, limit=min(nxt, r0 + int(0.60 * rr[i] * fs)))
        if p2 is None:
            continue
        i2, a2 = p2
        emat.append(1000.0 * (i1 - r0) / fs)
        lvet.append(1000.0 * (i2 - i1) / fs)
        s1a.append(a1)
        s2a.append(a2)

        # --- S3: early diastole, low band, compared to this beat's own late floor ---
        a = i2 + int(S3_AFTER_S2[0] / 1000 * fs)
        b = min(i2 + int(S3_AFTER_S2[1] / 1000 * fs), nxt)
        floor_a = b
        floor_b = nxt
        if b - a >= 5 and floor_b - floor_a >= 5 and floor_b <= len(e3):
            seg = e3[a:b]
            floor = e3[floor_a:floor_b]
            fmu, fsd = float(floor.mean()), float(floor.std() or 1e-6)
            k = int(np.argmax(seg))
            z = (float(seg[k]) - fmu) / fsd
            s3v.append(z)
            if z >= s3_z and 0 < k < len(seg) - 1:
                n_s3 += 1
                s3t.append(1000.0 * (a + k - i2) / fs)

    n = len(emat)
    if n < 5:
        return StiPanel(n, *([float("nan")] * 12))
    E, L = np.asarray(emat), np.asarray(lvet)
    return StiPanel(
        n_beats=n,
        emat_ms=round(float(np.median(E)), 2),
        emat_sd_ms=round(float(E.std(ddof=1)), 2),
        lvet_ms=round(float(np.median(L)), 2),
        lvet_sd_ms=round(float(L.std(ddof=1)), 2),
        # Weissler's rate correction: LVET shortens ~1.7 ms per bpm.
        lveti_ms=round(float(np.median(L) + 1.7 * hr), 2) if np.isfinite(hr) else float("nan"),
        emat_over_lvet=round(float(np.median(E) / max(np.median(L), 1e-6)), 5),
        s1_strength=round(float(np.median(s1a)), 4),
        s2_strength=round(float(np.median(s2a)), 4),
        s3_index=round(float(np.median(s3v)), 4) if s3v else float("nan"),
        s3_prevalence=round(n_s3 / n, 4),
        s3_timing_ms=round(float(np.median(s3t)), 2) if s3t else float("nan"),
        hr_bpm=round(hr, 2) if np.isfinite(hr) else float("nan"),
    )
