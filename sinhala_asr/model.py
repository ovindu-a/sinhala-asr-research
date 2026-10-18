"""Trainable QUAN-style CTC reader on the EmbeddingGemma 2 audio encoder.

Parameter names match quan_transcribe.model.QuanCTC, so a saved checkpoint (config.json + units.json +
model.safetensors) loads with QUAN's inference code (`QuanTranscribe.from_pretrained(dir)`).

Adds what inference does not need: SpecAugment, dropout, dynamic-chunk attention masks, text-embedding head init,
and gradients through the reader and the top `unfreeze_top_k` encoder layers.
"""

import random
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from quan_transcribe.features import LogMel
from quan_transcribe.model import TransformerBlock
from transformers import AutoModel


@dataclass
class Config:
    encoder_id: str = "google/embeddinggemma-2"
    vocab_size: int = 0
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    layer_mix: bool = True
    unfreeze_top_k: int = 0
    head_init: str = "text"  # "text": unit rows start from Gemma text embeddings; "random"
    freeze_head: bool = False
    specaug: dict = field(default_factory=lambda: {"freq_masks": 2, "freq_width": 27, "time_masks": 10, "time_width_frac": 0.04})
    dynamic_chunk: bool = True
    chunk_frames: int = 12
    max_chunk_units: int = 4
    max_left_chunks: int = 8

    def to_dict(self):
        return asdict(self)


def load_full(encoder_id: str):
    return AutoModel.from_pretrained(encoder_id, dtype=torch.bfloat16, vision_config=None)


def spec_augment(feats: torch.Tensor, lens: torch.Tensor, cfg: dict) -> torch.Tensor:
    """feats (B, T, F) log-mel; masked bins/frames are filled with each utterance's mean. Vectorized."""
    B, T, Fdim = feats.shape
    dev = feats.device
    valid = torch.arange(T, device=dev)[None, :] < lens[:, None]  # (B, T)
    fill = (feats * valid[..., None]).sum((1, 2)) / (lens.clamp(min=1) * Fdim)  # (B,)

    def spans(n_masks, max_w, size):  # (B, n) starts/widths -> (B, size) bool mask
        w = (torch.rand(B, n_masks, device=dev) * (max_w + 1)).floor()
        s = torch.rand(B, n_masks, device=dev) * (size - w).clamp(min=0)
        pos = torch.arange(size, device=dev)[None, None, :]
        return ((pos >= s[..., None]) & (pos < (s + w)[..., None])).any(1)

    fmask = spans(cfg["freq_masks"], cfg["freq_width"], Fdim)  # (B, F)
    max_tw = max(1, int(cfg["time_width_frac"] * int(lens.max())))
    tw = (cfg["time_width_frac"] * lens.float()).clamp(min=1)  # per-utterance max width
    w = torch.rand(B, cfg["time_masks"], device=dev) * (tw[:, None] + 1)
    w = w.floor().clamp(max=max_tw)
    s = torch.rand(B, cfg["time_masks"], device=dev) * (lens[:, None].float() - w).clamp(min=0)
    pos = torch.arange(T, device=dev)[None, None, :]
    tmask = ((pos >= s[..., None]) & (pos < (s + w)[..., None])).any(1) & valid  # (B, T)
    m = (fmask[:, None, :] | tmask[:, :, None]) & valid[..., None]
    return torch.where(m, fill[:, None, None], feats)


class CTCModel(nn.Module):
    def __init__(self, cfg: Config, full_model=None, gemma_ids: list[int] | list[list[int]] | None = None):
        """gemma_ids: `units.init_ids` — one Gemma id per unit, or a list of ids per unit (grapheme/gbpe units),
        whose text embeddings are averaged to start that unit's head row."""
        super().__init__()
        self.cfg = cfg
        full = full_model if full_model is not None else load_full(cfg.encoder_id)
        self.mel = LogMel(cfg.encoder_id)
        self.encoder = full.audio_tower.float()
        n_enc = self.encoder.config.num_hidden_layers
        enc_dim = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for layer in self.encoder.layers[n_enc - cfg.unfreeze_top_k :] if cfg.unfreeze_top_k else []:
            for p in layer.parameters():
                p.requires_grad_(True)

        self._acts: list[torch.Tensor] = []
        grab = lambda mod, inp, out: self._acts.append(out[0] if isinstance(out, tuple) else out)  # noqa: E731
        self.encoder.subsample_conv_projection.register_forward_hook(grab)
        for layer in self.encoder.layers:
            layer.register_forward_hook(grab)

        d = cfg.d_model
        self.layer_logits = nn.Parameter(torch.zeros(n_enc + 1)) if cfg.layer_mix else None
        self.in_norm = nn.LayerNorm(enc_dim)
        self.in_proj = nn.Linear(enc_dim, d)
        self.pos_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)
        self.blocks = nn.ModuleList([TransformerBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers)])
        for blk in self.blocks:
            blk.ff[2] = nn.Dropout(cfg.dropout)  # Identity at inference; same Sequential indices
        self.drop = nn.Dropout(cfg.dropout)
        self.out_norm = nn.LayerNorm(d)

        emb = full.language_model.embed_tokens.weight  # (262144, 512)
        text_dim = emb.shape[1]
        self.to_text = nn.Linear(d, text_dim)
        self.unit_rows = nn.Parameter(torch.zeros(cfg.vocab_size - 2, text_dim))
        self.special_rows = nn.Parameter(torch.randn(2, text_dim) * 0.02)
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        with torch.no_grad():
            if cfg.head_init == "text":
                assert gemma_ids is not None and len(gemma_ids) == cfg.vocab_size - 2
                if gemma_ids and isinstance(gemma_ids[0], list):
                    rows = torch.stack([emb[torch.tensor(ids)].float().mean(0) for ids in gemma_ids])
                else:
                    rows = emb[torch.tensor(gemma_ids)].float()
                self.unit_rows.copy_(rows / rows.norm(dim=-1).mean())
            else:
                self.unit_rows.normal_(0, 0.02)
        if cfg.freeze_head:
            self.unit_rows.requires_grad_(False)
        del full

    # ---- encoder
    def _encoder_hidden(self, feats, feat_mask):
        self._acts.clear()
        out = self.encoder(feats, feat_mask, return_dict=True)
        acts, self._acts = self._acts, []
        if self.layer_logits is not None:
            w = torch.softmax(self.layer_logits.float(), dim=0)
            h = sum(w[i] * a.float() for i, a in enumerate(acts))
        else:
            h = acts[-1].float()
        return h, out.attention_mask

    def chunk_mask(self, T, device, chunk, left):
        c = torch.arange(T, device=device) // chunk
        allow = c[None, :] <= c[:, None]
        if left is not None:
            allow &= c[None, :] >= c[:, None] - left
        return allow

    def forward(self, wav, wav_lens, chunk: int | None = None, left: int | None = None):
        """Returns log-probs (B, T, V) float32 and output lengths (B,). In training, samples a dynamic chunk."""
        cfg = self.cfg
        with torch.no_grad():
            feats, feat_mask = self.mel(wav, wav_lens)
            if self.training and cfg.specaug:
                feats = spec_augment(feats, feat_mask.sum(1), cfg.specaug)
        h, mask = self._encoder_hidden(feats, feat_mask)

        x = self.in_proj(self.in_norm(h))
        xc = F.pad((x * mask[..., None]).transpose(1, 2), (cfg.conv_kernel - 1, 0))
        x = self.drop(x + F.gelu(self.pos_conv(xc).transpose(1, 2)))

        T = x.shape[1]
        if self.training and cfg.dynamic_chunk and chunk is None and random.random() < 0.5:
            chunk = cfg.chunk_frames * random.randint(1, cfg.max_chunk_units)
            left = random.randint(1, cfg.max_left_chunks)
        attn = mask[:, None, None, :]
        if chunk is not None:
            attn = attn & self.chunk_mask(T, x.device, chunk, left)[None, None]
        attn = attn | torch.eye(T, dtype=torch.bool, device=x.device)[None, None]  # never an empty row

        for blk in self.blocks:
            x = blk(x, attn)
        t = self.to_text(self.out_norm(x))
        rows = torch.cat([self.special_rows, self.unit_rows], dim=0)
        logits = t.float() @ rows.float().t() + self.head_bias.float()
        return logits.log_softmax(-1), mask.sum(1)

    def trainable_state_dict(self):
        """Reader + fine-tuned encoder layers; the frozen encoder comes from the base model at load time."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad or not n.startswith("encoder.")}
        return {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if k in keep}


def greedy(logp: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    best = logp.argmax(-1).cpu()
    out = []
    for b in range(best.shape[0]):
        seq = best[b, : int(lens[b])]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != 0].tolist())
    return out
