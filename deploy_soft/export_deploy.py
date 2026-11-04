"""Turns one SOFT-decision training checkpoint into a deployable bundle:

    parity_encoder.ts   TorchScript, transmitter side (absent if parity_dim = 0)
    code_decoder.ts     TorchScript, receiver side
    manifest.json       everything both ends must agree on, incl. a run_id
    codebooks.npy       the n_q RVQ codebooks, so the receiver needs no EnCodec

Same shape as deploy/export_deploy.py. The one structural difference is the
decoder's second input: the hard model takes the received INDICES and builds a
one-hot prior inside forward(); the soft model takes the prior ready-made, as a
float [B, n_q, 1024, T] log-posterior, because the receiver is what computes it.
That makes the two .ts files mutually incompatible in a way that raises a shape
or dtype error rather than decoding silently -- which is the good outcome.

TorchScript rather than a raw state_dict on purpose: the graph travels with the
weights, so the deployment side needs no copy of the model classes.

run_id is a hash of the weights. The parity encoder and the decoder are
CO-TRAINED -- the parity bits mean whatever the decoder learned they mean -- so
mixing halves from different runs yields garbage with no error anywhere.

Usage:
    python export_deploy.py --ckpt ../outputs_v8/vogeo_v8_enc20_par20_awgn_soft.pt \
                            --out_dir ./bundle
"""

import os
import json
import argparse
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vogeo_frame import BITS_PER_CODE, TOTAL_BITS_PER_FRAME

# ---- must mirror finalVersion_v2.ipynb exactly ----
CODEBOOK_SIZE, LATENT_DIM = 1024, 128
DEC_WIDTH, DEC_DILATIONS, ENC_WIDTH = 256, (1, 2, 4), 128
LLR_CLIP, LOGPOST_FLOOR = 10.0, -30.0


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[..., :-self.pad] if self.pad > 0 else x


class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, 3, dilation=dilation)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class ParityEncoder(nn.Module):
    """Unchanged from the hard scheme: the bits it emits are what gets
    modulated, so they are hard on both sides of the link."""

    def __init__(self, parity_dim, latent_dim=LATENT_DIM, width=ENC_WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(latent_dim, width, 3), nn.GELU(),
            CausalConv1d(width, width, 3), nn.GELU(),
            CausalConv1d(width, parity_dim, 3), nn.Sigmoid(),
        )

    def forward(self, z):
        return torch.round(self.net(z))          # deployment: hard bits, no STE


class CodeDecoder(nn.Module):
    """Soft variant: code_prior arrives as a float [B, n_q, C, T] normalised
    log-posterior instead of being built from indices inside forward()."""

    def __init__(self, in_channels, n_q, width=DEC_WIDTH, dilations=DEC_DILATIONS):
        super().__init__()
        self.n_q = n_q
        # kept on the instance, not read from a module global: TorchScript
        # cannot close over globals
        self.codebook_size = CODEBOOK_SIZE
        self.in_conv = CausalConv1d(in_channels, width, 3)
        self.blocks = nn.Sequential(*[ResBlock(width, d) for d in dilations])
        self.out_conv = CausalConv1d(width, n_q * CODEBOOK_SIZE, 3)
        self.copy_logit = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, code_prior):
        h = self.out_conv(self.blocks(F.gelu(self.in_conv(x))))
        B, _, T = h.shape
        logits = h.view(B, self.n_q, self.codebook_size, T)
        return logits + self.copy_logit * code_prior


def parity_lookback():
    return 3 * 2                                          # 3 x CausalConv1d(k=3)


def decoder_lookback(dilations=DEC_DILATIONS):
    return 2 + sum(2 * d for d in dilations) + 2


def weights_hash(*state_dicts):
    h = hashlib.sha256()
    for sd in state_dicts:
        if sd is None:
            continue
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    n_q, parity_dim = ck["n_q"], ck["parity_dim"]
    mode = ck.get("decision_mode", "hard")
    assert mode == "soft", (
        f"{os.path.basename(args.ckpt)} was trained with {mode!r} decisions. "
        f"Export it with deploy/export_deploy.py instead -- the two decoders "
        f"take different second arguments and are not interchangeable.")
    assert n_q * BITS_PER_CODE + parity_dim == TOTAL_BITS_PER_FRAME, \
        f"checkpoint spends {n_q*BITS_PER_CODE+parity_dim} bits/frame, not {TOTAL_BITS_PER_FRAME}"
    print(f"{os.path.basename(args.ckpt)}: n_q={n_q} parity_dim={parity_dim} "
          f"channel={ck.get('channel')} decision={mode}")

    dec = CodeDecoder(LATENT_DIM + parity_dim, n_q).eval()
    dec.load_state_dict(ck["decoder"])
    pe = None
    if parity_dim > 0:
        pe = ParityEncoder(parity_dim).eval()
        pe.load_state_dict(ck["parity_encoder"])
    print(f"  copy_logit {float(dec.copy_logit):.3f}  "
          f"(1.0 = the prior alone reproduces MAP detection)")

    run_id = weights_hash(ck.get("parity_encoder"), ck["decoder"])

    # ---- TorchScript ----
    dec_path = os.path.join(args.out_dir, "code_decoder.ts")
    torch.jit.script(dec).save(dec_path)
    print(f"  wrote {dec_path} ({os.path.getsize(dec_path)/1e6:.1f} MB)  [receiver]")
    if pe is not None:
        pe_path = os.path.join(args.out_dir, "parity_encoder.ts")
        torch.jit.script(pe).save(pe_path)
        print(f"  wrote {pe_path} ({os.path.getsize(pe_path)/1e6:.2f} MB)  [transmitter]")

    # ---- manifest ----
    bandwidth = n_q * BITS_PER_CODE * 75 / 1000 + 1e-6
    manifest = {
        "run_id": run_id,
        "decision_mode": "soft",
        "receiver_input": "llr",       # NOT bits -- see vogeo_llr.py
        "llr_convention": "L = 2*a*y/sigma^2, positive means bit = 1",
        "llr_clip": LLR_CLIP,          # parity feature = clamp(L/llr_clip, -1, 1)
        "logpost_floor": LOGPOST_FLOOR,
        "n_q": n_q, "parity_dim": parity_dim,
        "bits_per_frame": TOTAL_BITS_PER_FRAME, "bits_per_code": BITS_PER_CODE,
        "frame_rate_hz": 75, "bitrate_bps": TOTAL_BITS_PER_FRAME * 75,
        "sample_rate": 24000, "channels": 1,
        "encodec_model": "encodec_model_24khz",
        # set_target_bandwidth() only accepts EnCodec's official list, which has
        # no entry for n_q of 1 or 3 -- pass this to quantizer() directly
        "quantizer_bandwidth_kbps": bandwidth,
        "bit_order": "MSB_first",
        "frame_layout": "n_q code words (10 bits each, layer 0 first), then parity bits",
        "parity_lookback_frames": parity_lookback(),
        "decoder_lookback_frames": decoder_lookback(),
        "trained_channel": ck.get("channel"),
        "trained_snr_range_db": [-5.0, 8.0],
        "trained_val_acc": ck.get("val_acc"),
        "torch_version": torch.__version__,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote manifest.json (run_id {run_id})")

    # ---- RVQ codebooks -------------------------------------------------
    # The soft receiver needs these for more than a lookup: the MMSE latent is
    # a posterior-weighted AVERAGE over all 1024 entries, so the full codebook
    # is on the critical path, not just the selected row.
    from encodec import EncodecModel
    ec = EncodecModel.encodec_model_24khz()
    ec.set_target_bandwidth(1.5)
    ec.eval()
    for p in ec.parameters():
        p.requires_grad = False

    layers = ec.quantizer.vq.layers[:n_q]
    for i, lay in enumerate(layers):
        assert type(lay.project_out).__name__ == "Identity", (
            f"RVQ layer {i} has a non-Identity project_out; the posterior average "
            f"would no longer commute with decode()")
    codebooks = np.stack([lay._codebook.embed.detach().cpu().numpy()
                          for lay in layers])                       # [n_q, 1024, 128]
    np.save(os.path.join(args.out_dir, "codebooks.npy"), codebooks)
    print(f"  wrote codebooks.npy {codebooks.shape} "
          f"({codebooks.nbytes/1e6:.1f} MB)  [receiver]")

    # sanity: with a collapsed posterior the MMSE average must reduce EXACTLY to
    # quantizer.decode(). That identity is what makes soft a strict
    # generalisation of hard rather than a different scheme.
    with torch.no_grad():
        probe = torch.randint(0, CODEBOOK_SIZE, (n_q, 1, 40))
        cb = torch.from_numpy(codebooks)
        onehot = F.one_hot(probe[:, 0], CODEBOOK_SIZE).float()      # [n_q,40,C]
        mmse = sum(onehot[q] @ cb[q] for q in range(n_q)).t()[None]
        assert torch.allclose(mmse, ec.quantizer.decode(probe), atol=1e-6), \
            "posterior average does not match quantizer.decode()"
    print("  collapsed posterior == quantizer.decode()  [verified]")

    print(f"\nBundle ready in {args.out_dir}. Ship parity_encoder.ts + manifest.json "
          f"+ vogeo_frame.py + vogeo_tx.py to the TX Pi,\ncode_decoder.ts + "
          f"codebooks.npy + manifest.json + vogeo_frame.py + vogeo_llr.py + "
          f"vogeo_rx.py to the RX Pi.")


if __name__ == "__main__":
    main()
