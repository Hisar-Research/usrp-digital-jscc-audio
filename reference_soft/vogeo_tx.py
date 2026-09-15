"""Transmitter. Logic identical to deploy/vogeo_tx.py -- soft decoding is a
RECEIVER-ONLY change. The parity encoder still emits hard 0/1 bits, because
bits are what gets modulated.

What does differ is the BUNDLE: a soft run trains its own parity encoder
alongside its own decoder, so the weights and the run_id are not those of the
hard bundle. Do not mix them.
"""
import os
import json

import numpy as np
import torch

from .vogeo_frame import pack_frames


def _as_tensor(x, device):
    return torch.as_tensor(np.asarray(x), device=device)


class VogeoTx:
    def __init__(self, bundle_dir, device="cpu"):
        self.device = torch.device(device)
        with open(os.path.join(bundle_dir, "manifest.json")) as f:
            self.m = json.load(f)
        self.n_q = self.m["n_q"]
        self.parity_dim = self.m["parity_dim"]
        self.lookback = self.m["parity_lookback_frames"]
        self.run_id = self.m["run_id"]
        self.bandwidth = self.m["quantizer_bandwidth_kbps"]   # hand this to EnCodec

        self.pe = None
        if self.parity_dim > 0:
            self.pe = torch.jit.load(os.path.join(bundle_dir, "parity_encoder.ts"),
                                     map_location=self.device).eval()
        self.reset()

    def reset(self):
        self._z_tail = None                # carried latent context between blocks

    @torch.no_grad()
    def encode(self, z, codes):
        """z [128,T] or [1,128,T]; codes [n_q,T] or [n_q,1,T] -> bits [T,40] uint8."""
        z = _as_tensor(z, self.device).float()
        if z.dim() == 2:
            z = z[None]
        c = _as_tensor(codes, self.device).long()
        if c.dim() == 3:
            c = c[:, 0]
        assert c.shape[0] == self.n_q, f"expected {self.n_q} RVQ layers, got {c.shape[0]}"
        assert z.shape[-1] == c.shape[-1], "z and codes disagree on frame count"

        if self.pe is None:
            parity = np.zeros((0, z.shape[-1]), dtype=np.uint8)      # control config
        else:
            # prepend the carried tail so early frames see real history, then
            # drop the outputs that belong to it
            zc = z if self._z_tail is None else torch.cat([self._z_tail, z], dim=-1)
            cut = 0 if self._z_tail is None else self._z_tail.shape[-1]
            parity = self.pe(zc)[..., cut:][0].to(torch.uint8).cpu().numpy()
            self._z_tail = zc[..., -self.lookback:].clone()

        return pack_frames(c.cpu().numpy(), parity, self.n_q, self.parity_dim)
