"""Multimodal transformer fusion backbone with per-modality reconstruction.

Reconstruction targets are the *pooled representations produced by the same
encoders on the complete input*. This uses paired complete observations at
TRAINING time only -- the same privilege MRCF and MMIN-style methods take, and
it must be stated in the paper. Nothing about the erased content is visible at
inference.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.masking import MODALITIES


def masked_mean(h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B,T,d) x (B,T) -> (B,d); zero vector when nothing is valid."""
    w = valid.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


class BertTextEncoder(nn.Module):
    """Fine-tuned BERT producing per-token 768-d features.

    Deliberately a drop-in replacement for the pre-extracted `text` features:
    it emits (B, T, 768) which then goes through the SAME ModalityEncoder as
    before. The only thing that changes between frozen and fine-tuned mode is
    how those 768-d vectors are produced, which is exactly the confound we are
    testing -- whether Delta ~ 0 is real or an artefact of an underfitting
    text pathway.
    """

    def __init__(self, name: str = "bert-base-uncased", unk_id: int = -1):
        super().__init__()
        from transformers import AutoModel
        # unk_id >= 0 selects the operation used in the literature: masked text
        # positions are REPLACED by [UNK] and remain visible to attention,
        # rather than being removed from the attention mask. EMT-DLFR states
        # it "replaces the original token with the [UNK] token in BERT
        # vocabulary"; our default instead makes the position invisible. The
        # two are different corruptions and we test both.
        self.unk_id = unk_id
        # Metal's fused attention does not implement training-time dropout.
        # Eager attention preserves the pretrained model's dropout probability.
        attention_options = ({"attn_implementation": "eager"}
                             if torch.backends.mps.is_available() else {})
        self.bert = AutoModel.from_pretrained(name, **attention_options)
        self.out_dim = self.bert.config.hidden_size
        # RoBERTa has type_vocab_size == 1 and does not use segment ids;
        # passing them is either ignored or an error depending on version.
        self.use_token_type = getattr(self.bert.config, "type_vocab_size", 1) > 1

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                orig_mask=None):
        if self.unk_id >= 0 and orig_mask is not None:
            # positions that masking removed, among originally-real tokens
            dropped = orig_mask.bool() & (~attention_mask.bool())
            input_ids = torch.where(dropped,
                                    torch.full_like(input_ids, self.unk_id),
                                    input_ids)
            attention_mask = orig_mask  # keep them visible, as [UNK]
        kw = {}
        if self.use_token_type and token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask.long(), **kw)
        return out.last_hidden_state


class ModalityEncoder(nn.Module):
    def __init__(self, d_in: int, d: int, seq_len: int, n_layers: int = 2,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4 * d,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        # MPS cannot do dropout inside scaled_dot_product_attention. Disable
        # ATTENTION dropout unconditionally (not just on MPS) so that MPS and
        # CUDA compute the identical function -- development happens on MPS,
        # reported numbers come from CUDA. Residual/FFN dropout is untouched.
        for lyr in self.enc.layers:
            lyr.self_attn.dropout = 0.0
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        pad = ~valid
        # A fully-padded row makes softmax produce NaN; let position 0 attend and
        # discard the result afterwards via masked_mean over the ORIGINAL mask.
        fully = pad.all(dim=1)
        if fully.any():
            pad = pad.clone()
            pad[fully, 0] = False
        h = self.enc(h, src_key_padding_mask=pad)
        return self.norm(masked_mean(h, valid))


class ReconstructionHead(nn.Module):
    """Predict modality k's pooled representation from the other modalities."""

    def __init__(self, d: int, hidden: int, n_other: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_other * d + n_other, hidden),  # +availability flags
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, others: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([others, avail], dim=-1))


class MultimodalFusion(nn.Module):
    def __init__(self, dims: Dict[str, int], seqs: Dict[str, int], d: int = 64,
                 n_layers: int = 2, nhead: int = 4, dropout: float = 0.1,
                 recon_hidden: Optional[Dict[str, int]] = None,
                 use_recon: bool = True, text_encoder: str = "frozen",
                 bert_name: str = "bert-base-uncased", unk_id: int = -1):
        super().__init__()
        self.d, self.use_recon = d, use_recon
        self.text_encoder = text_encoder
        self.bert = (BertTextEncoder(bert_name, unk_id)
                     if text_encoder == "bert" else None)
        self.encoders = nn.ModuleDict({
            k: ModalityEncoder(dims[k], d, seqs[k], n_layers, nhead, dropout)
            for k in MODALITIES
        })
        if use_recon:
            rh = recon_hidden or {k: 2 * d for k in MODALITIES}
            self.recon = nn.ModuleDict({
                k: ReconstructionHead(d, rh[k], dropout=dropout) for k in MODALITIES
            })
        self.fuse = nn.Sequential(
            nn.Linear(3 * d + 3, 2 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.GELU(),
        )
        self.head = nn.Linear(d, 1)

    # -- encoding ---------------------------------------------------------
    def encode(self, feats, valid, bert=None,
               orig_text_mask=None) -> Dict[str, torch.Tensor]:
        if self.bert is not None:
            if bert is None:
                raise ValueError("text_encoder='bert' requires bert token ids")
            # valid["t"] is the (possibly masked) attention mask -> erased
            # tokens are genuinely invisible to BERT, not just zeroed after.
            # In UNK mode they are instead substituted and remain visible.
            txt = self.bert(bert[:, 0, :], valid["t"], bert[:, 2, :],
                            orig_mask=orig_text_mask)
            feats = dict(feats); feats["t"] = txt
            if self.bert.unk_id >= 0 and orig_text_mask is not None:
                valid = dict(valid); valid["t"] = orig_text_mask
        return {k: self.encoders[k](feats[k], valid[k]) for k in MODALITIES}

    @staticmethod
    def availability(valid) -> Dict[str, torch.Tensor]:
        return {k: valid[k].any(dim=1).float() for k in MODALITIES}

    # -- forward ----------------------------------------------------------
    def forward(self, feats, valid, return_parts: bool = False, bert=None,
                orig_text_mask=None):
        z_obs = self.encode(feats, valid, bert, orig_text_mask)
        avail = self.availability(valid)
        B = next(iter(z_obs.values())).shape[0]

        z_use, z_hat = {}, {}
        for k in MODALITIES:
            others = [o for o in MODALITIES if o != k]
            if self.use_recon:
                oz = torch.cat([z_obs[o] * avail[o].unsqueeze(-1) for o in others], -1)
                oa = torch.stack([avail[o] for o in others], dim=-1)
                z_hat[k] = self.recon[k](oz, oa)
                a = avail[k].unsqueeze(-1)
                z_use[k] = a * z_obs[k] + (1.0 - a) * z_hat[k]
            else:
                z_use[k] = z_obs[k] * avail[k].unsqueeze(-1)

        flags = torch.stack([avail[k] for k in MODALITIES], dim=-1)
        z = self.fuse(torch.cat([z_use[k] for k in MODALITIES] + [flags], dim=-1))
        pred = self.head(z).squeeze(-1)

        if return_parts:
            return pred, {"z": z, "z_obs": z_obs, "z_hat": z_hat,
                          "z_use": z_use, "avail": avail}
        return pred

    # -- reconstruction loss ---------------------------------------------
    def recon_loss(self, parts, z_full: Dict[str, torch.Tensor],
                   weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """MSE between predicted and complete-input pooled representations,
        applied only where the modality was actually degraded."""
        if not self.use_recon:
            return torch.zeros((), device=parts["z"].device)
        w = weights or {k: 1.0 for k in MODALITIES}
        total, denom = 0.0, 0.0
        for k in MODALITIES:
            missing = (1.0 - parts["avail"][k])                 # (B,)
            if missing.sum() < 1:
                continue
            err = F.mse_loss(parts["z_hat"][k], z_full[k].detach(), reduction="none")
            total = total + w[k] * (err.mean(dim=-1) * missing).sum()
            denom = denom + w[k] * missing.sum()
        if isinstance(total, float):
            return torch.zeros((), device=parts["z"].device)
        return total / max(float(denom), 1.0)


@torch.no_grad()
def representation_diagnostics(Z: torch.Tensor) -> Dict[str, float]:
    """Quantify representation collapse for a matrix of embeddings (N, d).

    Tests the mechanism behind the T-frag damage directly, rather than
    inferring it from accuracy. A healthy encoder spreads inputs over many
    directions; a damaged one maps everything to a narrow cone.

      cos_offdiag : mean pairwise cosine similarity between distinct inputs.
                    Near 1.0 means every input maps to nearly the same vector.
      eff_rank    : entropy-based effective rank of the centred embeddings,
                    exp(-sum p_i log p_i) over normalised singular values.
                    Collapse drives this toward 1.
      part_ratio  : participation ratio (sum s)^2 / sum(s^2), a second,
                    differently-biased spread estimate.
    """
    Z = Z.float()
    n = Z.shape[0]
    zn = Z / Z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos = zn @ zn.T
    off = (cos.sum() - cos.diagonal().sum()) / max(n * (n - 1), 1)

    def spread(M):
        try:
            sv = torch.linalg.svdvals(M).clamp(min=0)
        except Exception:
            return float("nan"), float("nan")
        tot = sv.sum().clamp(min=1e-12)
        pr = sv / tot
        nz = pr[pr > 1e-12]
        return (float(torch.exp(-(nz * nz.log()).sum())),
                float(tot.pow(2) / sv.pow(2).sum().clamp(min=1e-12)))

    # Uncentered: detects mean-dominated collapse (everything on one ray).
    # Centered: detects loss of *variation* structure around the mean.
    # Both are needed -- a collapsed encoder can still have full-rank noise
    # once the dominant shared direction is subtracted out, so the centered
    # figure alone is blind to exactly the failure we are hunting.
    er_raw, pr_raw = spread(Z)
    er_cen, pr_cen = spread(Z - Z.mean(dim=0, keepdim=True))
    return {"cos_offdiag": float(off),
            "eff_rank_raw": er_raw, "part_ratio_raw": pr_raw,
            "eff_rank_centred": er_cen, "part_ratio_centred": pr_cen,
            "emb_std": float(Z.std())}


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
