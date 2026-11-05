"""
One pass of the whole chain, top to bottom, for the DeepJSCC
(Encodec + Channel Coding Model) setup.

(2 RVQ layers (20 Encodec bits) + 20 parity bits = 40 bits per frame = 3000 bit/s).


    audio -> EnCodec -> VogeoTx -> [ RADIO ] -> VogeoRx -> EnCodec -> audio

Run it:
    python selftest_v2.py

Writes three wav files to selftest_out/ .
"""

import os

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from encodec import EncodecModel
from encodec.utils import convert_audio
from pystoi import stoi

from deepJSCC_voice.vogeo_tx import VogeoTx
from deepJSCC_voice.vogeo_rx import VogeoRx
from deepJSCC_voice.vogeo_frame import unpack_frames

BUNDLE  = "./bundle"                    # made by export_deploy.py
WAV     = "./test2.wav"
SECONDS = 5.0
BER     = 0.01                          # what the radio delivers, ~0 dB Es/N0
OUT_DIR = "selftest_out"


def estoi(ref24k, deg24k):
    """ESTOI is defined at 16 kHz, so both signals are resampled first."""
    r = AF.resample(torch.as_tensor(ref24k), 24000, 16000).numpy()
    d = AF.resample(torch.as_tensor(deg24k), 24000, 16000).numpy()
    n = min(len(r), len(d))
    return stoi(r[:n], d[:n], 16000, extended=True)


os.makedirs(OUT_DIR, exist_ok=True)
tx, rx = VogeoTx(BUNDLE), VogeoRx(BUNDLE)
assert (tx.n_q, tx.parity_dim) == (2, 20), \
    f"this script is for enc20_par20 only, bundle is n_q={tx.n_q} par={tx.parity_dim}"

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
    codes = ec.quantizer(z, ec.frame_rate, tx.bandwidth).codes      # [2,1,T]
codes = codes[:, 0].numpy()
print(f"2. EnCodec    latent {tuple(z.shape)} -> codes {codes.shape} "
      f"({codes.shape[1]} frames)")


# ---- 3. our transmitter: latent + codes -> bits -----------------------------
bits = tx.encode(z, codes)                                          # [T,40] uint8
print(f"3. VogeoTx    {bits.shape[0]} frames x {bits.shape[1]} bits "
      f"= {bits.size} bits = {tx.m['bitrate_bps']} bit/s")


# ---- 4. the radio ----------------------------------------------------------
# Everything from here to the next line is GNU Radio's: packetize, BPSK,
# USRP, receive, synchronise, slice. To this code it is only "bits in, bits
# out", so one line of bit flips stands in for it.
rx_bits = (bits ^ (np.random.default_rng(0).random(bits.shape) < BER)).astype(np.uint8)
print(f"4. radio      BER {BER}: {int((rx_bits != bits).sum())} of {bits.size} bits flipped")


# ---- 5. our receiver: bits -> corrected indices -----------------------------
damaged, _ = unpack_frames(rx_bits, tx.n_q, tx.parity_dim)          # before we help
fixed = rx.decode(rx_bits)                                          # after
print(f"5. VogeoRx    codes wrong: {(damaged != codes).mean():.4f} in "
      f"-> {(fixed != codes).mean():.4f} out")


# ---- 6. EnCodec: indices -> audio ------------------------------------------
def to_audio(c):
    with torch.no_grad():
        return ec.decoder(ec.quantizer.decode(
            torch.as_tensor(c, dtype=torch.long)[:, None]))[0, 0].numpy()

ref     = wav[0].numpy()
ceiling = to_audio(codes)        # no channel at all: the best EnCodec can do
raw     = to_audio(damaged)      # the link without us
ours    = to_audio(fixed)        # the link with us
print(f"6. EnCodec    codes -> {len(ours) / 24000:.2f} s of audio")


# ---- 7. what we got --------------------------------------------------------
for name, sig in (("00_reference", ref), ("01_ceiling", ceiling),
                  ("02_raw", raw), ("03_ours", ours)):
    sf.write(os.path.join(OUT_DIR, f"{name}.wav"), sig, 24000)

print(f"\n7. ESTOI (1.0 = perfect intelligibility)")
print(f"     EnCodec ceiling, no channel : {estoi(ref, ceiling):.4f}")
print(f"     link without us             : {estoi(ref, raw):.4f}")
print(f"     link with us                : {estoi(ref, ours):.4f}")
print(f"\n   wavs in {OUT_DIR}/ -- 02_raw.wav vs 03_ours.wav is the comparison.")
print("   'with us' must beat 'without us'. If it does not, the checkpoint is")
print("   untrained or the wrong one for this bundle.")
