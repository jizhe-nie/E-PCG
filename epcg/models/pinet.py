"""PI-Net -- physiology-supervised four-channel ECG/PCG network.

WHERE THIS DESIGN COMES FROM
----------------------------
This project spent a long time on the opposite idea: writing physiology into the
architecture, as a learned electromechanical-delay bias on cross-modal attention.
Four controls killed it, and each one narrows what a successor is allowed to assume:

  * on synthetic data with a known delay swept 20-160 ms, the learned tau stayed
    pinned at 53 ms -- the parameter is not identifiable inside attention;
  * band and cardiac-phase decomposition found the bias's +6.8 dB advantage flat
    across all 11 cells, not concentrated where the coupling is;
  * under HR-matched shuffled pairing the advantage survived at +5.1 dB, so 88% of it
    never depended on the two modalities belonging to the same person;
  * the learned tau/sigma were identical to within 1 ms whether pairing was real or
    scrambled.

An architectural prior can be routed around: tau is 24 scalars against 131k attention
parameters, and nothing in the objective ever required tau to be right. Supervision
cannot be routed around in the same way -- if the network must predict EMAT, LVET and
S3 from the waveform, those quantities have to be present in the representation.

So the physiology moves from the architecture into the loss:

    main       HFrEF (LVEF <= 0.40)
    aux 1      LVEF regression -- uses the continuous gold standard from all subjects
               instead of collapsing 500 echo readings into a 49/364 split
    aux 2      STI regression: EMAT, LVET, S3 prevalence, per site
               labels come free from our own extractor, no annotation needed

The auxiliary heads also give a clinically readable intermediate. LGC2-Net reaches
0.960 as a black box; a model that additionally reports the EMAT and LVET it inferred
can be checked against a measurement, which is the difference we can actually offer.

Deliberately small: 49 positive subjects. Two earlier results in this project showed
that added capacity hurts at this scale (a gradient booster lost to logistic
regression, 0.820 vs 0.859; adding S3@APEX to the panel dropped 0.873 to 0.861).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SITES = ("APEX", "LLSB", "LUSB", "RUSB")
STI_TARGETS = ("emat_ms", "lvet_ms", "s3_prevalence")


class SiteEncoder(nn.Module):
    """Encode one auscultation site's synchronised ECG+PCG pair.

    Both channels enter at the same rate and are stacked, so the convolutions see the
    two modalities in register from the first layer -- the alignment the attention
    experiments failed to learn is simply given.
    """

    def __init__(self, dim: int = 128, in_ch: int = 2, dropout: float = 0.3):
        super().__init__()
        c = dim // 4
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, c, 15, stride=4, padding=7), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c * 2, 11, stride=4, padding=5), nn.BatchNorm1d(c * 2), nn.GELU(),
            nn.Conv1d(c * 2, dim, 9, stride=4, padding=4), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Conv1d(dim, dim, 7, stride=2, padding=3), nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                                     # (B, dim, T')
        # Average and max together: average carries the persistent shift that the beat
        # analysis showed HFrEF produces, max keeps the loudest event in the recording.
        return self.drop(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))


class PINet(nn.Module):
    """Shared site encoder, learned aggregation over sites, one main and two aux heads.

    The site encoder is shared rather than per-site: with 49 positives, four separate
    encoders would be four times the capacity for the same labels. Site identity
    enters as an embedding so the model can still tell the apex from the aortic area,
    which E7 showed differ by about 30 ms in baseline timing.
    """

    def __init__(self, dim: int = 128, n_sites: int = 4, n_sti: int = 3,
                 dropout: float = 0.3, learn_site_weights: bool = True):
        super().__init__()
        self.encoder = SiteEncoder(dim, dropout=dropout)
        self.site_emb = nn.Embedding(n_sites, 2 * dim)
        # Scalar gate per site, learned. E11 found one or two sites carry nearly all
        # the signal, so a uniform mean over four is not the obvious choice.
        self.site_gate = nn.Parameter(torch.zeros(n_sites), requires_grad=learn_site_weights)
        self.trunk = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                   nn.GELU(), nn.Dropout(dropout))
        self.head_cls = nn.Linear(dim, 2)
        self.head_lvef = nn.Linear(dim, 1)
        # STI is predicted PER SITE, from that site's own features: EMAT at the apex
        # and at the aortic area are different numbers and averaging them away would
        # make the auxiliary task teach the wrong thing.
        self.head_sti = nn.Linear(2 * dim, n_sti)

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, n_sites, 2, T) -- ECG and PCG stacked, one pair per site."""
        B, S = x.shape[:2]
        f = self.encoder(x.reshape(B * S, *x.shape[2:])).reshape(B, S, -1)
        f = f + self.site_emb(torch.arange(S, device=x.device))[None]
        sti = self.head_sti(f)                              # (B, S, n_sti)
        w = torch.softmax(self.site_gate, dim=0)[None, :, None]
        pooled = self.trunk((f * w).sum(1))
        return {"logits": self.head_cls(pooled),
                "lvef": self.head_lvef(pooled).squeeze(-1),
                "sti": sti,
                "site_weights": torch.softmax(self.site_gate, dim=0).detach()}


def loss_fn(out: dict, y: torch.Tensor, lvef: torch.Tensor, sti: torch.Tensor,
            sti_mask: torch.Tensor, w_cls: torch.Tensor,
            w_lvef: float = 0.0, w_sti: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Weighted sum of the three objectives. Auxiliary weights default to zero so the
    ablation's baseline rung is literally the same code path with the physiology
    switched off, not a separate model."""
    cls = F.cross_entropy(out["logits"], y, weight=w_cls)
    parts = {"cls": float(cls)}
    total = cls
    if w_lvef > 0:
        l = F.smooth_l1_loss(out["lvef"], lvef)
        total = total + w_lvef * l
        parts["lvef"] = float(l)
    if w_sti > 0 and sti_mask.any():
        # Standardised targets so three quantities on different scales (ms, ms,
        # fraction) contribute comparably; masked where extraction failed.
        s = F.smooth_l1_loss(out["sti"][sti_mask], sti[sti_mask])
        total = total + w_sti * s
        parts["sti"] = float(s)
    parts["total"] = float(total)
    return total, parts
