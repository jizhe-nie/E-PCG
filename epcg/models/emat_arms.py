"""Four ways to estimate electromechanical delay, only one of which can be honest.

EMAT is the interval from the ECG R peak to the first heart sound. It is the cleanest
task in this project for one reason: NEITHER MODALITY CAN PRODUCE IT ALONE. The ECG
locates R and has no access to S1; the PCG locates S1 and has no access to R. Any
architecture that beats the single-modality arms must be using the correspondence
between them, which is exactly the property that HFrEF classification turned out not to
need -- there, PCG acoustics alone reach 0.878 and the timing is redundant.

So this is the positive control the whole instrument was missing. If the time-shift
probe cannot detect alignment use HERE, it cannot detect it anywhere, and every
"alignment is barely used" conclusion we have drawn is a statement about the probe
rather than about the models.

The four arms:

  ecg_only    lower bound. Cannot see S1. Whatever it scores comes from EMAT's
              correlation with things visible in the ECG alone (rate, QRS width).
  pcg_only    lower bound. Cannot see R. Whatever it scores comes from EMAT's
              correlation with S1 amplitude, rate, and other acoustic covariates.
  attention   both modalities, standard cross-modal attention into an MLP head. The
              design this literature uses. Free to solve the task any way it likes.
  delay_op    both modalities, but the ONLY route to an answer is an explicit delay.

The last one is the mechanism claim. An ECG branch produces a nonnegative trigger
envelope, a PCG branch produces a nonnegative response envelope, they are cross-
correlated over the physiological lag range, and tau comes out as a soft-argmax of that
correlation profile. No weight anywhere in the model stores a delay -- tau is the
location of a maximum, so it cannot be memorised from the label distribution the way an
MLP head can, and it cannot be routed around the way an attention bias was in E4.

THE MEASUREMENT THAT DECIDES IT. Rolling the PCG later by d does not merely perturb the
input: it makes the TRUE EMAT become EMAT + d. So a model that is really measuring the
delay must raise its prediction by d, and a model leaning on content correlates will not
move. That gives a signed, unit-free quantity,

    tracking = mean(pred_shifted - pred_clean) / d

which is 1 for an honest delay estimator and 0 for a content correlate, and it is far
more informative than watching an error metric get worse -- both kinds of model get
worse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAG_MIN_MS = 0.0
LAG_MAX_MS = 250.0          # sti_panel's S1 search window, so the arms share a prior


class Env(nn.Module):
    """Signal -> nonnegative envelope at the input rate.

    Deliberately shallow. A deep encoder here would let the delay arm smuggle content
    features into the correlation profile, which is the failure mode being tested.
    """

    def __init__(self, ch: int = 16, k: int = 31):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, 1, k, padding=k // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x.unsqueeze(1))).squeeze(1)


class Trunk(nn.Module):
    """Strided encoder used by the three arms that are allowed to learn freely."""

    def __init__(self, dim: int = 64, dropout: float = 0.2):
        super().__init__()
        c = dim // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, c, 15, stride=4, padding=7), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, 11, stride=4, padding=5), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, dim, 9, stride=4, padding=4), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Conv1d(dim, dim, 7, stride=2, padding=3), nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)

    def seq(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.net(x.unsqueeze(1))).transpose(1, 2)      # (B, L, D)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))
        return self.drop(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))   # (B, 2D)


class SingleModality(nn.Module):
    """ecg_only / pcg_only. Structurally incapable of measuring a delay."""

    def __init__(self, which: str, dim: int = 64, dropout: float = 0.2):
        super().__init__()
        assert which in ("ecg", "pcg")
        self.ch = 0 if which == "ecg" else 1
        self.trunk = Trunk(dim, dropout)
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> dict:
        return {"pred": self.head(self.trunk.pooled(x[:, self.ch])).squeeze(-1)}


class CrossAttention(nn.Module):
    """The literature's design: ECG queries PCG, pooled, then an MLP regresses EMAT."""

    def __init__(self, dim: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.e, self.p = Trunk(dim, dropout), Trunk(dim, dropout)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> dict:
        q, kv = self.e.seq(x[:, 0]), self.p.seq(x[:, 1])
        a, _ = self.attn(self.norm(q), kv, kv)
        h = torch.cat([a.mean(1), a.amax(1)], dim=-1)
        return {"pred": self.head(h).squeeze(-1)}


class DelayOp(nn.Module):
    """tau as the soft-argmax of an ECG-PCG cross-correlation profile.

    The correlation is computed by FFT in float32. Under autocast, fp16 costs the phase
    precision the whole method rests on -- the same trap that produced NaNs in the mel
    front end earlier in this project.
    """

    def __init__(self, fs: float, ch: int = 16, temp_init: float = 2.0,
                 lag_min_ms: float = LAG_MIN_MS, lag_max_ms: float = LAG_MAX_MS):
        super().__init__()
        self.fs = fs
        # 自监督训练需要更宽的 lag 窗：注入 +d 之后 tau 要能落到 tau+d，
        # 若窗口卡在 250 ms，大 tau 的样本会被截断，等变性就成了假的。
        self.lag_min_ms, self.lag_max_ms = lag_min_ms, lag_max_ms
        self.e, self.p = Env(ch), Env(ch)
        self.log_temp = nn.Parameter(torch.tensor(float(temp_init)).log())
        # A scalar affine readout on tau. Two parameters cannot encode a per-sample
        # delay, so tau still has to be right; they only absorb a systematic offset
        # between the correlation peak and the panel's S1 picking rule.
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def profile(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, enabled=False):
            e = self.e(x[:, 0].float())
            p = self.p(x[:, 1].float())
            e = e - e.mean(-1, keepdim=True)
            p = p - p.mean(-1, keepdim=True)
            n = int(2 ** torch.ceil(torch.log2(torch.tensor(2.0 * e.shape[-1]))).item())
            E = torch.fft.rfft(e, n=n, dim=-1)
            P = torch.fft.rfft(p, n=n, dim=-1)
            # PCG lags ECG, so the positive-lag half of corr(pcg, ecg) is the range
            # where S1 follows R.
            c = torch.fft.irfft(P * E.conj(), n=n, dim=-1)
            lo = int(self.lag_min_ms / 1000 * self.fs)
            hi = int(self.lag_max_ms / 1000 * self.fs)
            # 负 lag 存在 FFT 结果的尾部，需要先把两半接起来再截取。
            c = torch.roll(c, -lo, dims=-1)[..., :hi - lo + 1] if lo < 0                 else c[..., lo:hi + 1]
            return c / (c.std(-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> dict:
        c = self.profile(x)
        lags_ms = (torch.arange(c.shape[-1], device=c.device, dtype=torch.float32)
                   + int(self.lag_min_ms / 1000 * self.fs)) * 1000.0 / self.fs
        w = torch.softmax(c * self.log_temp.exp(), dim=-1)
        tau = (w * lags_ms).sum(-1)
        # THE SPREAD IS FREE, AND IT IS THE UNCERTAINTY.
        # The soft-argmax already builds a distribution over candidate lags; its mean is
        # the estimate and its standard deviation says how much the profile disagrees
        # with itself. A sharp peak means one lag explains the pairing far better than
        # the others; a flat profile means "any delay looks about as good", which is
        # exactly the state a model should report rather than answer through. No extra
        # parameters are involved -- this is the second moment of a distribution the
        # operator was computing anyway.
        #
        # That peak width relates to estimation precision is classical (the Cramer-Rao
        # bound for time-delay estimation depends on correlation curvature). What is not
        # classical is calibrating it end to end, using it to DETECT desynchronisation,
        # and checking that it carries no disease bias; those are what E28 tests.
        var = (w * (lags_ms[None, :] - tau[:, None]) ** 2).sum(-1)
        return {"pred": self.scale * tau + self.bias, "tau_ms": tau,
                "sigma_ms": torch.sqrt(var + 1e-8), "profile_peak": c.amax(-1)}


class DelayOpMixture(DelayOp):
    """把相关剖面分解成 S1 和 S2 两个成分，而不是在整条曲线上取一次加权平均。

    为什么要分解。ECG 包络近似是 R 波处的脉冲串，所以互相关剖面约等于"以 R 波对齐、
    逐拍平均后的心音包络" —— 上面本来就有 S1 和 S2 两个包。在这种双峰曲线上做一次
    soft-argmax，结果会落在两峰之间，给出一个物理上不存在的延迟。E29 见到了这个失效
    模式的极端版本：自监督训练把跟踪系数刷到 0.974，锁到的却不是 S1。

    做法是两段有序的 soft-argmax：先在生理合理的 S1 窗内定位第一个峰，再用一个以
    tau1 为参照的软窗去找第二个峰。第二步的窗是随 tau1 浮动的，不是固定区间 ——
    心率快的人 S2 来得早，固定窗会切错。

    额外的好处是 LVET（tau2 - tau1）跟着一起出来了，于是 EMAT/LVET 这个收缩功能指数
    可以由同一个算子端到端给出。它在文献里比单独的 EMAT 更贴近心功能。
    """

    def __init__(self, fs: float, ch: int = 16, temp_init: float = 2.0,
                 s1_max_ms: float = 160.0, lvet_center_ms: float = 300.0,
                 lvet_width_ms: float = 110.0):
        super().__init__(fs, ch, temp_init, lag_min_ms=0.0, lag_max_ms=600.0)
        self.s1_max_ms = s1_max_ms
        self.lvet_center_ms = lvet_center_ms
        self.lvet_width_ms = lvet_width_ms
        self.lvet_scale = nn.Parameter(torch.tensor(1.0))
        self.lvet_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict:
        c = self.profile(x)
        lags = (torch.arange(c.shape[-1], device=c.device, dtype=torch.float32)
                * 1000.0 / self.fs)
        t = self.log_temp.exp()

        # 第一步：S1。生理上 S1 必在 R 波后 160 ms 内，硬门比软门稳，
        # 因为落在窗外的成分若只是被压低而不是排除，仍会把加权平均拉偏。
        m1 = (lags <= self.s1_max_ms).float()
        w1 = torch.softmax(c * t + (m1 - 1.0) * 1e4, dim=-1)
        tau1 = (w1 * lags).sum(-1)
        v1 = (w1 * (lags[None] - tau1[:, None]) ** 2).sum(-1)

        # 第二步：S2。软窗以 tau1 为参照浮动 —— 心率快的人 S2 来得早，
        # 固定区间会切错。
        d = lags[None] - tau1[:, None] - self.lvet_center_ms
        gate = -(d / self.lvet_width_ms) ** 2
        w2 = torch.softmax(c * t + gate, dim=-1)
        tau2 = (w2 * lags).sum(-1)

        lvet = (tau2 - tau1).clamp(min=1.0)
        return {"pred": self.scale * tau1 + self.bias,
                "tau_ms": tau1, "sigma_ms": torch.sqrt(v1 + 1e-8),
                "lvet_pred": self.lvet_scale * lvet + self.lvet_bias,
                "lvet_ms": lvet, "tau2_ms": tau2,
                "ratio": tau1 / lvet}


def build(arm: str, fs: float, dim: int = 64, dropout: float = 0.2) -> nn.Module:
    if arm == "ecg_only":
        return SingleModality("ecg", dim, dropout)
    if arm == "pcg_only":
        return SingleModality("pcg", dim, dropout)
    if arm == "attention":
        return CrossAttention(dim, dropout=dropout)
    if arm == "delay_op":
        return DelayOp(fs)
    if arm == "delay_op_mix":
        return DelayOpMixture(fs)
    raise ValueError(arm)
