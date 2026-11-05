"""Soft-decision receiver.

Where deploy/vogeo_rx.py takes BITS, this takes LLRs, and three things follow
from that. Compare the two files side by side -- everything else is the same.

1. THE CODEWORD POSTERIOR. A 10-bit RVQ index is sent as 10 independent bits,
   so log P(index = i) = sum_j log P(bit_j = b_j(i)). One matmul against the
   1024x10 bit table gives the posterior over all 1024 codewords, exactly.

2. THE LATENT. The hard receiver looks up codebook[argmax]. This one takes the
   posterior average, sum_q E[codebook_q[index_q]] -- the MMSE estimate of the
   latent given the channel. It is the same operation: EnCodec's RVQ decode is
   a sum of per-layer embedding lookups (project_out is Identity, asserted at
   export), so a collapsed posterior reproduces quantizer.decode() bit for bit.
   Hard decoding is the special case where the posterior has collapsed.

3. THE PRIOR. The hard decoder is handed a 0/1 one-hot at the received index.
   This one is handed the normalised log-posterior, so adding it to the logits
   is Bayes' rule in log space. The trained copy_logit sits near 1.0, at which
   value untrained logits would reproduce MAP detection exactly.

The parity bits get the same treatment in miniature: instead of +-1 the decoder
sees clamp(L / llr_clip, -1, 1), a bounded confidence rather than a decision.

INPUT SCALE MATTERS. See vogeo_llr.py -- this class assumes true LLRs.
"""
import os
import json

import numpy as np
import torch
import torch.nn.functional as F

from .vogeo_frame import unpack_llrs, BITS_PER_CODE


class VogeoRx:
    def __init__(self, bundle_dir, device="cpu"):
        self.device = torch.device(device)
        with open(os.path.join(bundle_dir, "manifest.json")) as f:
            self.m = json.load(f)
        assert self.m.get("decision_mode") == "soft", (
            f"this receiver needs a SOFT bundle; manifest says "
            f"{self.m.get('decision_mode', 'hard')!r}. Use deploy/ for hard bundles.")
        self.n_q = self.m["n_q"]
        self.parity_dim = self.m["parity_dim"]
        self.lookback = self.m["decoder_lookback_frames"]
        self.run_id = self.m["run_id"]
        self.llr_clip = self.m["llr_clip"]
        self.logpost_floor = self.m["logpost_floor"]

        self.dec = torch.jit.load(os.path.join(bundle_dir, "code_decoder.ts"),
                                  map_location=self.device).eval()
        cb = np.load(os.path.join(bundle_dir, "codebooks.npy"))       # [n_q,1024,128]
        self.codebooks = torch.as_tensor(cb, device=self.device).float()

        # [1024, 10] bit pattern of every index, MSB first -- the same order
        # pack_frames() writes, so it inverts the mapping exactly
        w = 1 << torch.arange(BITS_PER_CODE - 1, -1, -1, device=self.device)
        self.code_bits = ((torch.arange(self.codebooks.shape[1],
                                        device=self.device)[:, None] & w) > 0).float()
        self.reset()

    def reset(self):
        self._in_tail = None                # carried decoder input
        self._prior_tail = None             # carried log-posterior prior

    def code_log_posterior(self, llr):
        """[..., 10] LLRs -> [..., 1024] log-likelihood per codeword.
        log P = (log s(L) - log s(-L)) @ CODE_BITS^T + sum_j log s(-L_j)."""
        lp = F.logsigmoid(llr)              # log P(bit = 1)
        ln = F.logsigmoid(-llr)             # log P(bit = 0)
        return (lp - ln) @ self.code_bits.t() + ln.sum(-1, keepdim=True)

    def soft_dequantize(self, logpost):
        """[n_q, T, 1024] log-posterior -> [1, 128, T] MMSE latent."""
        z = 0.0
        for q in range(self.n_q):
            z = z + logpost[q].softmax(dim=-1) @ self.codebooks[q]     # [T,128]
        return z.t()[None]

    @torch.no_grad()
    def decode(self, llr):
        """llr [T,40] float -> corrected RVQ indices [n_q, T] int64.

        Streaming: call it per block and the causal context is carried across
        boundaries, so a streamed run matches a one-shot run exactly."""
        code_np, par_np = unpack_llrs(llr, self.n_q, self.parity_dim)
        code_llr = torch.as_tensor(code_np, dtype=torch.float32,
                                   device=self.device)                # [n_q,T,10]

        logpost = self.code_log_posterior(code_llr)                   # [n_q,T,C]
        z_rx = self.soft_dequantize(logpost)                          # [1,128,T]
        prior = (logpost - logpost.logsumexp(dim=-1, keepdim=True))
        prior = prior.clamp(min=self.logpost_floor)
        prior = prior.permute(0, 2, 1)[None]                          # [1,n_q,C,T]

        if self.parity_dim:
            p = torch.as_tensor(par_np, dtype=torch.float32, device=self.device)[None]
            p = (p / self.llr_clip).clamp(-1.0, 1.0)   # bounded confidence, not +-1
            dec_in = torch.cat([z_rx, p], dim=1)
        else:
            dec_in = z_rx

        if self._in_tail is None:
            xin, pin, cut = dec_in, prior, 0
        else:
            xin = torch.cat([self._in_tail, dec_in], dim=-1)
            pin = torch.cat([self._prior_tail, prior], dim=-1)
            cut = self._in_tail.shape[-1]
        logits = self.dec(xin, pin)[..., cut:]                        # [1,n_q,C,T]
        self._in_tail = xin[..., -self.lookback:].clone()
        self._prior_tail = pin[..., -self.lookback:].clone()

        return logits.argmax(dim=2)[0].cpu().numpy()                  # [n_q,T]
