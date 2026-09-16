"""
One pass of the whole SOFT-decision chain, top to bottom, for the DeepJSCC
(EnCodec + Channel Coding Model) setup.

    audio -> EnCodec -> VogeoTx -> [ RADIO ] -> LLRs -> VogeoRx -> EnCodec -> audio

The samples are also scaled by an arbitrary gain first, because a real radio's
output is not calibrated, and the LLR estimator has to recover from that.

Run it:
    python selftest_v2.py

Writes four wav files to selftest_out/ .
"""

import math
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from encodec import EncodecModel
from encodec.utils import convert_audio
from pystoi import stoi

from vogeo_tx import VogeoTx
from vogeo_rx import VogeoRx
from vogeo_frame import hard_slice
from vogeo_llr import llr_from_samples, esn0_db

BUNDLE  = "./bundle"                    # made by export_deploy.py
SECONDS = 5.0
ESN0_DB = -3.0                          # Q(sqrt(2*Es/N0)) = 0.01 -> BER 0.01
RADIO_GAIN = 3.7                        # uncalibrated receiver, to be estimated away
OUT_DIR = "selftest_out"
WAV = next((p for p in ("./test2.wav", "../test2.wav", "../deploy/test2.wav")
            if os.path.exists(p)), "./test2.wav")


def estoi(ref24k, deg24k):
    """ESTOI is defined at 16 kHz, so both signals are resampled first."""
    r = AF.resample(torch.as_tensor(ref24k), 24000, 16000).numpy()
    d = AF.resample(torch.as_tensor(deg24k), 24000, 16000).numpy()
    n = min(len(r), len(d))
    return stoi(r[:n], d[:n], 16000, extended=True)


os.makedirs(OUT_DIR, exist_ok=True)
tx, rx = VogeoTx(BUNDLE), VogeoRx(BUNDLE)
assert tx.run_id == rx.run_id, "transmitter and receiver are from different runs"
print(f"0. bundle     n_q={tx.n_q} parity={tx.parity_dim} soft, run_id {tx.run_id}")

ec = EncodecModel.encodec_model_24khz().eval()
for p in ec.parameters():
    p.requires_grad = False


# ---- 1. audio in -----------------------------------------------------------
data, sr = sf.read(WAV)
w = torch.from_numpy(data).float()
w = w.unsqueeze(0) if w.ndim == 1 else w.T
wav = convert_audio(w, sr, 24000, 1)[..., :int(SECONDS * 24000)]
print(f"1. audio      {wav.shape[-1] / 24000:.2f} s, 24 kHz mono")


# ---- 2. EnCodec: audio -> latent + RVQ indices ------------------------------
# encoder() for the latent (encode() would give codes only, and the parity
# network needs the latent); bandwidth from the manifest because
# set_target_bandwidth() cannot express every n_q.
with torch.no_grad():
    z = ec.encoder(wav[None])                                       # [1,128,T]
    codes = ec.quantizer(z, ec.frame_rate, tx.bandwidth).codes      # [n_q,1,T]
codes = codes[:, 0].numpy()
print(f"2. EnCodec    latent {tuple(z.shape)} -> codes {codes.shape} "
      f"({codes.shape[1]} frames)")


# ---- 3. our transmitter: latent + codes -> bits -----------------------------
# Unchanged from the hard scheme: bits are what gets modulated.
bits = tx.encode(z, codes)                                          # [T,40] uint8
print(f"3. VogeoTx    {bits.shape[0]} frames x {bits.shape[1]} bits "
      f"= {bits.size} bits = {tx.m['bitrate_bps']} bit/s")


# ---- 4. the radio ----------------------------------------------------------
# Everything here is GNU Radio's: packetize, BPSK, USRP, receive, synchronise.
# The one thing it must NOT do is slice -- the soft receiver needs the matched
# filter output, not a decision.
sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (ESN0_DB / 10.0)))
rng = np.random.default_rng(0)
y = RADIO_GAIN * ((2.0 * bits.astype(np.float32) - 1.0)
                  + sigma * rng.standard_normal(bits.shape).astype(np.float32))
ber = float(((y > 0) != (bits > 0)).mean())
print(f"4. radio      Es/N0 {ESN0_DB:.2f} dB, gain x{RADIO_GAIN} -> BER {ber:.4f}")


# ---- 5. soft samples -> LLRs ------------------------------------------------
# The receiver was trained on TRUE LLRs, L = 2*a*y/sigma^2, so the arbitrary
# radio gain has to come out. Nothing here knows RADIO_GAIN or sigma: both are
# estimated from the samples themselves (vogeo_llr.py).
llr = llr_from_samples(y)
print(f"5. LLR        blind Es/N0 estimate {esn0_db(y):+.2f} dB "
      f"(true {ESN0_DB:+.2f}) | |L| median {np.median(np.abs(llr)):.2f}")


# ---- 6. our receiver: LLRs -> corrected indices -----------------------------
damaged, _ = hard_slice(llr, tx.n_q, tx.parity_dim)                 # without us
fixed = rx.decode(llr)                                              # with us
print(f"6. VogeoRx    codes wrong: {(damaged != codes).mean():.4f} in "
      f"-> {(fixed != codes).mean():.4f} out")


# ---- 7. EnCodec: indices -> audio ------------------------------------------
def to_audio(c):
    with torch.no_grad():
        return ec.decoder(ec.quantizer.decode(
            torch.as_tensor(c, dtype=torch.long)[:, None]))[0, 0].numpy()

ref     = wav[0].numpy()
ceiling = to_audio(codes)        # no channel at all: the best EnCodec can do
raw     = to_audio(damaged)      # the link without us
ours    = to_audio(fixed)        # the link with us
print(f"7. EnCodec    codes -> {len(ours) / 24000:.2f} s of audio")


# ---- 8. what we got --------------------------------------------------------
for name, sig in (("00_reference", ref), ("01_ceiling", ceiling),
                  ("02_raw", raw), ("03_ours", ours)):
    sf.write(os.path.join(OUT_DIR, f"{name}.wav"), sig, 24000)

print(f"\n8. ESTOI (1.0 = perfect intelligibility)")
print(f"     EnCodec ceiling, no channel : {estoi(ref, ceiling):.4f}")
print(f"     link without us             : {estoi(ref, raw):.4f}")
print(f"     link with us                : {estoi(ref, ours):.4f}")
print(f"\n   wavs in {OUT_DIR}/ -- 02_raw.wav vs 03_ours.wav is the comparison.")
print("   'with us' must beat 'without us'. If it does not, the checkpoint is")
print("   untrained, or the LLR scale is wrong -- check step 5 first: the blind")
print("   Es/N0 estimate should land within a few tenths of a dB of the truth.")
