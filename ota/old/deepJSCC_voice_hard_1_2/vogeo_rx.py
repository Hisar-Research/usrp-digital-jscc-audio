import os
import json

import numpy as np
import torch

from deepJSCC_voice.vogeo_frame import unpack_frames


class VogeoRx:
    def __init__(self, bundle_dir, device="cpu"):
        self.device = torch.device(device)
        with open(os.path.join(bundle_dir, "manifest.json")) as f:
            self.m = json.load(f)
        self.n_q = self.m["n_q"]
        self.parity_dim = self.m["parity_dim"]
        self.lookback = self.m["decoder_lookback_frames"]
        self.run_id = self.m["run_id"]

        self.dec = torch.jit.load(os.path.join(bundle_dir, "code_decoder.ts"),
                                  map_location=self.device).eval()
        cb = np.load(os.path.join(bundle_dir, "codebooks.npy"))       # [n_q,1024,128]
        self.codebooks = torch.as_tensor(cb, device=self.device).float()
        self.reset()

    def reset(self):
        self._in_tail = None                # carried decoder input
        self._code_tail = None              # carried received indices

    def dequantize(self, codes):
        """quantizer.decode() for this model: sum of per-layer lookups.
        codes [n_q,T] long -> [1,128,T]."""
        z = torch.zeros(codes.shape[-1], self.codebooks.shape[-1], device=self.device)
        for q in range(self.n_q):
            z = z + self.codebooks[q][codes[q]]
        return z.T[None]

    @torch.no_grad()
    def decode(self, bits):
        """bits [T,40] uint8 -> corrected RVQ indices [n_q, T] int64."""
        codes_np, parity_np = unpack_frames(bits, self.n_q, self.parity_dim)
        rx_codes = torch.as_tensor(codes_np, dtype=torch.long, device=self.device)
        z_rx = self.dequantize(rx_codes)                              # [1,128,T]

        if self.parity_dim:
            p = torch.as_tensor(parity_np, dtype=torch.float32, device=self.device)[None]
            dec_in = torch.cat([z_rx, 2.0 * p - 1.0], dim=1)          # bits -> +-1
        else:
            dec_in = z_rx

        rxc = rx_codes[:, None, :]                                    # [n_q,1,T]
        if self._in_tail is None:
            xin, cin, cut = dec_in, rxc, 0
        else:
            xin = torch.cat([self._in_tail, dec_in], dim=-1)
            cin = torch.cat([self._code_tail, rxc], dim=-1)
            cut = self._in_tail.shape[-1]
        logits = self.dec(xin, cin)[..., cut:]                        # [1,n_q,C,T]
        self._in_tail = xin[..., -self.lookback:].clone()
        self._code_tail = cin[..., -self.lookback:].clone()

        return logits.argmax(dim=2)[0].cpu().numpy()                  # [n_q,T]
