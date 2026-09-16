#!/usr/bin/env python3
"""
Offline processing-latency benchmark for three 3-kbps speech transmission methods:

1) Uncoded EnCodec 3.0 kbps
2) EnCodec 2.25 kbps + rate-3/4 punctured convolutional coding
3) Proposed 30 source bits + 10 learned protection bits/frame (SOFT VOGEO)

This script does NOT use SDRs, GNU Radio, sockets, PESQ, or ESTOI.
It measures computation only on the local machine.

Outputs
-------
latency_per_chunk.csv
    One row per audio chunk and method.

latency_per_audio.csv
    Per-utterance averages across chunks.

latency_summary.csv
    Mean, median, std, 95% CI, p95 across utterances for each method/stage.

Recommended paper usage
-----------------------
Report:
  - TX processing latency
  - RX protection/decoding latency
  - RX EnCodec synthesis latency
  - RX total processing latency

The 95% confidence intervals in latency_summary.csv are computed across
utterance-level means, not across individual chunks.

Example
-------
python benchmark_processing_latency.py \
    --num-audio 100 \
    --warmup 20 \
    --threads 1 \
    --snr-db 4 \
    --seed 1234

Notes
-----
* Model-loading time and file I/O are excluded from timed regions.
* CPU-only benchmarking is used by default.
* The synthetic BPSK/AWGN metrics are used only to exercise the same RX
  algorithms offline. They are NOT used as paper channel-performance results.
* The proposed RX includes the payload-only two-Gaussian EM LLR calibration,
  because that is part of the actual implemented receiver.
"""

import os
import sys
import glob
import math
import time
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import t as student_t

from encodec import EncodecModel
from encodec.utils import convert_audio


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Offline CPU latency benchmark for uncoded, Conv(3/4), and proposed SOFT VOGEO."
)
parser.add_argument("--num-audio", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20,
                    help="Number of untimed warm-up iterations.")
parser.add_argument("--threads", type=int, default=1,
                    help="PyTorch intra-op CPU threads.")
parser.add_argument("--snr-db", type=float, default=4.0,
                    help="Synthetic BPSK/AWGN Es/N0 used only to create offline RX metrics.")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min-duration", type=float, default=8.0)
parser.add_argument("--max-duration", type=float, default=15.0)
parser.add_argument("--vogeo-llr-clip", type=float, default=50.0)
parser.add_argument("--output-prefix", default="latency")
parser.add_argument(
    "--dataset-dir",
    default=None,
    help="Optional LibriTTS test-clean directory. If omitted, search below this project."
)
args = parser.parse_args()

if args.num_audio <= 0:
    raise ValueError("--num-audio must be positive")
if args.warmup < 0:
    raise ValueError("--warmup must be >= 0")
if args.threads <= 0:
    raise ValueError("--threads must be positive")
if args.max_duration <= args.min_duration:
    raise ValueError("--max-duration must be greater than --min-duration")
if args.vogeo_llr_clip <= 0:
    raise ValueError("--vogeo-llr-clip must be positive")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.set_num_threads(args.threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# PROJECT + VOGEO PACKAGE
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SOFT_REF_DIR = os.path.join(PROJECT_DIR, "reference_soft")

required_py = (
    "vogeo_tx.py",
    "vogeo_rx.py",
    "vogeo_frame.py",
)

missing_py = [
    name for name in required_py
    if not os.path.isfile(os.path.join(SOFT_REF_DIR, name))
]
if missing_py:
    raise RuntimeError(
        f"Missing VOGEO reference files under {SOFT_REF_DIR}: "
        + ", ".join(missing_py)
    )

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from reference_soft.vogeo_tx import VogeoTx
from reference_soft.vogeo_rx import VogeoRx


def _read_manifest(bundle_dir):
    path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _find_30plus10_soft_bundle():
    preferred_roots = [
        os.path.join(SOFT_REF_DIR, "bundle_3_4"),
        os.path.join(SOFT_REF_DIR, "bundle_3_out_4"),
        os.path.join(SOFT_REF_DIR, "bundle_30_10"),
    ]

    candidates = []
    seen = set()

    def add_candidate(path):
        path = os.path.abspath(path)
        if path in seen:
            return
        seen.add(path)

        manifest = _read_manifest(path)
        if manifest is None:
            return

        def as_int(*names):
            for name in names:
                if name in manifest:
                    try:
                        return int(manifest[name])
                    except Exception:
                        pass
            return None

        def as_float(*names):
            for name in names:
                if name in manifest:
                    try:
                        return float(manifest[name])
                    except Exception:
                        pass
            return None

        mode = manifest.get("decision_mode")
        n_q = as_int("n_q", "num_codebooks", "num_quantizers")
        parity = as_int("parity_dim", "parity_bits", "parity_bits_per_frame")
        bw = as_float("bandwidth", "bandwidth_kbps", "encodec_bandwidth")

        if mode is not None and str(mode).lower() != "soft":
            return
        if n_q is not None and n_q != 3:
            return
        if parity is not None and parity != 10:
            return

        score = 0
        low = path.lower()
        if "bundle_3_4" in low:
            score += 100
        if "bundle_3_out_4" in low:
            score += 95
        if "bundle_30_10" in low:
            score += 90
        if n_q == 3:
            score += 20
        if parity == 10:
            score += 20
        if bw is not None and abs(bw - 2.25) < 1e-3:
            score += 10
        if mode is not None and str(mode).lower() == "soft":
            score += 10

        candidates.append((score, path))

    for root in preferred_roots:
        if not os.path.isdir(root):
            continue
        if os.path.isfile(os.path.join(root, "manifest.json")):
            add_candidate(root)
        for walk_root, _dirs, files in os.walk(root):
            if "manifest.json" in files:
                add_candidate(walk_root)

    if not candidates and os.path.isdir(SOFT_REF_DIR):
        for walk_root, _dirs, files in os.walk(SOFT_REF_DIR):
            if "manifest.json" in files:
                add_candidate(walk_root)

    if not candidates:
        raise RuntimeError(
            f"Could not find a compatible 30+10 SOFT VOGEO bundle below {SOFT_REF_DIR}"
        )

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


VOGEO_BUNDLE_DIR = _find_30plus10_soft_bundle()
DEVICE = "cpu"

vogeo_tx = VogeoTx(VOGEO_BUNDLE_DIR, device=DEVICE)
vogeo_rx = VogeoRx(VOGEO_BUNDLE_DIR, device=DEVICE)

if int(vogeo_tx.n_q) != 3 or int(vogeo_rx.n_q) != 3:
    raise RuntimeError("Expected VOGEO n_q=3")
if int(vogeo_tx.parity_dim) != 10 or int(vogeo_rx.parity_dim) != 10:
    raise RuntimeError("Expected VOGEO parity_dim=10")

PROTECTED_BANDWIDTH_KBPS = float(vogeo_tx.bandwidth)
PROTECTED_N_Q = 3
PLAIN_BANDWIDTH_KBPS = 3.0
PLAIN_N_Q = 4

FRAMES_PER_CHUNK = 75
BITS_PER_CODE = 10
VOGEO_BITS_PER_FRAME = 40
CHUNK_SAMPLES = 24000
VOGEO_LLR_CLIP = float(args.vogeo_llr_clip)


# ============================================================
# ENCODEC
# ============================================================

model = EncodecModel.encodec_model_24khz()
model.eval()

if int(model.sample_rate) != 24000:
    raise RuntimeError(f"Expected EnCodec 24 kHz, got {model.sample_rate}")


# ============================================================
# DATASET
# ============================================================

def find_librtts_test_clean():
    if args.dataset_dir is not None:
        path = os.path.abspath(args.dataset_dir)
        if not os.path.isdir(path):
            raise RuntimeError(f"--dataset-dir does not exist: {path}")
        return path

    preferred = [
        os.path.join(PROJECT_DIR, "datasets", "LibriTTS", "test-clean"),
        os.path.join(PROJECT_DIR, "datasets", "LibriTTS", "LibriTTS", "test-clean"),
        os.path.join(PROJECT_DIR, "datasets", "test-clean"),
    ]

    for path in preferred:
        if os.path.isdir(path) and glob.glob(os.path.join(path, "**", "*.wav"), recursive=True):
            return os.path.abspath(path)

    datasets_root = os.path.join(PROJECT_DIR, "datasets")
    if os.path.isdir(datasets_root):
        for root, _dirs, _files in os.walk(datasets_root):
            if os.path.basename(root).lower() == "test-clean":
                if glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
                    return os.path.abspath(root)

    raise RuntimeError(
        "Could not locate LibriTTS test-clean. "
        "Use --dataset-dir /path/to/LibriTTS/test-clean"
    )


DATASET_DIR = find_librtts_test_clean()

all_wavs = sorted(glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True))

wav_files = []
for path in all_wavs:
    try:
        info = sf.info(path)
        dur = float(info.frames) / float(info.samplerate)
    except Exception:
        continue
    if args.min_duration <= dur < args.max_duration:
        wav_files.append(path)

wav_files = wav_files[: args.num_audio]

if not wav_files:
    raise RuntimeError("No audio files matched the requested duration range")


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav.T.copy())
    audio = convert_audio(audio, int(sr), model.sample_rate, model.channels)
    total_chunks = math.ceil(audio.shape[-1] / CHUNK_SAMPLES)
    return audio.contiguous(), total_chunks


def get_chunk(audio, chunk_id):
    start = chunk_id * CHUNK_SAMPLES
    end = min(start + CHUNK_SAMPLES, audio.shape[-1])
    chunk = audio[:, start:end]
    valid_samples = int(chunk.shape[-1])

    if valid_samples <= 0:
        raise RuntimeError("Empty chunk")

    if valid_samples < CHUNK_SAMPLES:
        chunk = torch.nn.functional.pad(
            chunk,
            (0, CHUNK_SAMPLES - valid_samples)
        )

    return chunk.contiguous(), valid_samples


# ============================================================
# BIT UTILITIES
# ============================================================

def indices_to_bits_frame_major(codes_np):
    """
    codes_np: [n_q, 75]
    return: [75, n_q*10]
    """
    codes_np = np.asarray(codes_np, dtype=np.uint16)
    n_q = codes_np.shape[0]
    shifts = np.arange(BITS_PER_CODE - 1, -1, -1, dtype=np.uint16)
    bits = ((codes_np.T[:, :, None] >> shifts) & 1).astype(np.uint8)
    return bits.reshape(FRAMES_PER_CHUNK, n_q * BITS_PER_CODE)


def bits_to_indices_frame_major(bits, n_q):
    """
    bits: flat or [75,n_q*10]
    return [n_q,75]
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(
        FRAMES_PER_CHUNK, n_q, BITS_PER_CODE
    )
    weights = (1 << np.arange(
        BITS_PER_CODE - 1, -1, -1, dtype=np.uint16
    ))
    codes = (
        bits.astype(np.uint16) * weights[None, None, :]
    ).sum(axis=2, dtype=np.uint16)
    return codes.T


def hard_slice(x):
    return (np.asarray(x, dtype=np.float32) > 0.0).astype(np.uint8)


def bpsk_awgn(bits, snr_db, rng):
    """
    bit 0 -> -1
    bit 1 -> +1
    Es/N0 convention: sigma^2 = 1/(2*gamma)
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    symbols = 2.0 * bits.astype(np.float32) - 1.0
    gamma = 10.0 ** (float(snr_db) / 10.0)
    sigma = math.sqrt(1.0 / (2.0 * gamma))
    noise = rng.normal(0.0, sigma, size=symbols.shape).astype(np.float32)
    return symbols + noise


# ============================================================
# RATE-3/4 CONVOLUTIONAL CODE
# ============================================================

K = 7
N_STATES = 1 << (K - 1)
G_POLY = (0o133, 0o171)
TAIL_BITS = K - 1
PUNCTURE_PATTERN = np.array([1, 1, 1, 0, 0, 1], dtype=np.uint8)

CONV_INFO_BITS = PROTECTED_N_Q * FRAMES_PER_CHUNK * BITS_PER_CODE  # 2250
CONV_TRELLIS_INPUT_BITS = CONV_INFO_BITS + TAIL_BITS               # 2256
CONV_MOTHER_BITS = 2 * CONV_TRELLIS_INPUT_BITS                     # 4512

if CONV_MOTHER_BITS % PUNCTURE_PATTERN.size != 0:
    raise RuntimeError("Convolutional mother-code length is not puncture aligned")

CONV_PUNCTURE_MASK = np.tile(
    PUNCTURE_PATTERN.astype(bool),
    CONV_MOTHER_BITS // PUNCTURE_PATTERN.size,
)
CONV_USEFUL_BITS = int(CONV_PUNCTURE_MASK.sum())                    # 3008


def _parity(x):
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def _build_trellis():
    nxt = np.zeros((N_STATES, 2), dtype=np.int64)
    out = np.zeros((N_STATES, 2, 2), dtype=np.uint8)

    for state in range(N_STATES):
        for bit in (0, 1):
            reg = (bit << (K - 1)) | state
            for j, g in enumerate(G_POLY):
                out[state, bit, j] = _parity(reg & g)
            nxt[state, bit] = reg >> 1
    return nxt, out


NEXT_STATE, OUT_BITS = _build_trellis()


def conv_encode(info_bits):
    info_bits = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
    if info_bits.size != CONV_INFO_BITS:
        raise ValueError(f"Expected {CONV_INFO_BITS} conv info bits")

    trellis_input = np.concatenate([
        info_bits,
        np.zeros(TAIL_BITS, dtype=np.uint8)
    ])

    mother = np.empty((CONV_TRELLIS_INPUT_BITS, 2), dtype=np.uint8)
    state = 0
    for i, bit in enumerate(trellis_input):
        bit = int(bit)
        mother[i] = OUT_BITS[state, bit]
        state = NEXT_STATE[state, bit]

    if state != 0:
        raise RuntimeError("Convolutional encoder did not terminate at zero state")

    return mother.reshape(-1)[CONV_PUNCTURE_MASK]


def depuncture_soft(punctured_soft):
    punctured_soft = np.asarray(punctured_soft, dtype=np.float32).reshape(-1)
    if punctured_soft.size != CONV_USEFUL_BITS:
        raise ValueError(
            f"Expected {CONV_USEFUL_BITS} punctured soft metrics, got {punctured_soft.size}"
        )
    mother = np.zeros(CONV_MOTHER_BITS, dtype=np.float32)
    mother[CONV_PUNCTURE_MASK] = punctured_soft
    return mother


def viterbi_decode_soft(mother_soft):
    mother_soft = np.asarray(mother_soft, dtype=np.float32).reshape(-1)
    rx_soft = mother_soft.reshape(CONV_TRELLIS_INPUT_BITS, 2)
    expected_sym = 2.0 * OUT_BITS.astype(np.float32) - 1.0

    INF = np.float64(1e300)
    metric = np.full(N_STATES, INF, dtype=np.float64)
    metric[0] = 0.0

    prev_state = np.zeros((CONV_TRELLIS_INPUT_BITS, N_STATES), dtype=np.int16)
    prev_bit = np.zeros((CONV_TRELLIS_INPUT_BITS, N_STATES), dtype=np.uint8)

    for ti in range(CONV_TRELLIS_INPUT_BITS):
        branch = -(expected_sym * rx_soft[ti][None, None, :]).sum(axis=2)
        new_metric = np.full(N_STATES, INF, dtype=np.float64)

        for state in range(N_STATES):
            if not np.isfinite(metric[state]):
                continue
            for bit in (0, 1):
                ns = int(NEXT_STATE[state, bit])
                candidate = metric[state] + float(branch[state, bit])
                if candidate < new_metric[ns]:
                    new_metric[ns] = candidate
                    prev_state[ti, ns] = state
                    prev_bit[ti, ns] = bit

        m = new_metric.min()
        if np.isfinite(m):
            new_metric -= m
        metric = new_metric

    state = 0
    decoded = np.zeros(CONV_TRELLIS_INPUT_BITS, dtype=np.uint8)
    for ti in range(CONV_TRELLIS_INPUT_BITS - 1, -1, -1):
        decoded[ti] = prev_bit[ti, state]
        state = int(prev_state[ti, state])

    return decoded[:CONV_INFO_BITS]


# ============================================================
# VOGEO LLR ESTIMATION
# Same payload-only two-Gaussian EM idea used in the SDR RX.
# ============================================================

def estimate_vogeo_llr(raw_metric):
    x = np.asarray(raw_metric, dtype=np.float32)
    flat = x.reshape(-1).astype(np.float64)

    if flat.size == 0:
        raise ValueError("Empty VOGEO metric array")
    if not np.all(np.isfinite(flat)):
        raise ValueError("NaN/Inf in VOGEO metrics")

    if flat.size >= 100:
        lo, hi = np.percentile(flat, [0.5, 99.5])
        fit = flat[(flat >= lo) & (flat <= hi)]
        if fit.size < max(100, flat.size // 2):
            fit = flat
    else:
        fit = flat

    neg = fit[fit < 0.0]
    pos = fit[fit >= 0.0]

    if neg.size >= 10 and pos.size >= 10:
        mu0 = float(np.median(neg))
        mu1 = float(np.median(pos))
    else:
        q25, q75 = np.percentile(fit, [25.0, 75.0])
        mu0 = float(q25)
        mu1 = float(q75)

    if mu0 > mu1:
        mu0, mu1 = mu1, mu0

    sep = mu1 - mu0
    if (not np.isfinite(sep)) or sep <= 1e-9:
        center = float(np.median(fit))
        amp = float(np.median(np.abs(fit - center)))
        amp = max(amp, 1e-3)
        mu0 = center - amp
        mu1 = center + amp

    d0 = (fit - mu0) ** 2
    d1 = (fit - mu1) ** 2
    nearest_resid = np.where(d0 <= d1, fit - mu0, fit - mu1)
    sigma2 = float(np.mean(nearest_resid ** 2))

    overall_var = float(np.var(fit))
    sep = max(mu1 - mu0, 1e-6)
    variance_floor = max((1e-4 * sep) ** 2, 1e-10)

    if (not np.isfinite(sigma2)) or sigma2 <= variance_floor:
        sigma2 = max(0.05 * overall_var, variance_floor)
    sigma2 = max(sigma2, variance_floor)

    for _ in range(30):
        old_mu0, old_mu1, old_sigma2 = mu0, mu1, sigma2

        logp0 = -0.5 * ((fit - mu0) ** 2) / sigma2
        logp1 = -0.5 * ((fit - mu1) ** 2) / sigma2

        mx = np.maximum(logp0, logp1)
        p0 = np.exp(logp0 - mx)
        p1 = np.exp(logp1 - mx)
        den = p0 + p1

        r0 = p0 / np.maximum(den, 1e-300)
        r1 = p1 / np.maximum(den, 1e-300)

        n0 = float(np.sum(r0))
        n1 = float(np.sum(r1))
        if n0 < 1.0 or n1 < 1.0:
            break

        mu0 = float(np.sum(r0 * fit) / n0)
        mu1 = float(np.sum(r1 * fit) / n1)

        if mu0 > mu1:
            mu0, mu1 = mu1, mu0
            r0, r1 = r1, r0
            n0, n1 = n1, n0

        sigma2 = float(
            (
                np.sum(r0 * (fit - mu0) ** 2)
                + np.sum(r1 * (fit - mu1) ** 2)
            )
            / max(n0 + n1, 1.0)
        )

        sep = max(mu1 - mu0, 1e-6)
        variance_floor = max((1e-4 * sep) ** 2, 1e-10)
        sigma2 = max(sigma2, variance_floor)

        delta = max(
            abs(mu0 - old_mu0),
            abs(mu1 - old_mu1),
            abs(sigma2 - old_sigma2),
        )
        if delta < 1e-8:
            break

    llr_flat = (
        (flat - mu0) ** 2 - (flat - mu1) ** 2
    ) / (2.0 * sigma2)

    llr_flat = np.clip(
        llr_flat,
        -VOGEO_LLR_CLIP,
        VOGEO_LLR_CLIP,
    )

    return llr_flat.reshape(x.shape).astype(np.float32)


# ============================================================
# ENCODEC DECODE
# ============================================================

@torch.no_grad()
def decode_encodec_codes(codes_np, valid_samples):
    codes_np = np.asarray(codes_np, dtype=np.int64)
    codes = torch.from_numpy(codes_np).unsqueeze(0)
    audio = model.decode([(codes, None)])
    return audio[0, :, :valid_samples]


# ============================================================
# TX BRANCHES
# We time each complete method independently from audio chunk
# to transmitted bits. This intentionally repeats the common
# EnCodec encoder so each number is a true standalone TX time.
# ============================================================

@torch.no_grad()
def tx_plain(chunk):
    z = model.encoder(chunk.unsqueeze(0))
    q = model.quantizer(
        z, model.frame_rate, float(PLAIN_BANDWIDTH_KBPS)
    )
    codes_q = q.codes
    codes_np = (
        codes_q[:, 0].detach().cpu().numpy().astype(np.uint16)
    )
    bits = indices_to_bits_frame_major(codes_np).reshape(-1)
    if bits.size != 3000:
        raise RuntimeError(f"Plain TX produced {bits.size} bits, expected 3000")
    return bits


@torch.no_grad()
def tx_conv(chunk):
    z = model.encoder(chunk.unsqueeze(0))
    q = model.quantizer(
        z, model.frame_rate, float(PROTECTED_BANDWIDTH_KBPS)
    )
    codes_q = q.codes
    codes_np = (
        codes_q[:, 0].detach().cpu().numpy().astype(np.uint16)
    )
    info_bits = indices_to_bits_frame_major(codes_np).reshape(-1)
    coded = conv_encode(info_bits)
    if coded.size != CONV_USEFUL_BITS:
        raise RuntimeError("Unexpected conv coded length")
    return coded


@torch.no_grad()
def tx_proposed(chunk):
    z = model.encoder(chunk.unsqueeze(0))
    q = model.quantizer(
        z, model.frame_rate, float(PROTECTED_BANDWIDTH_KBPS)
    )
    codes_q = q.codes

    # VOGEO encoder returns [75, 40] hard bits in your current implementation.
    bits = vogeo_tx.encode(z, codes_q)
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    if bits.size != 3000:
        raise RuntimeError(f"Proposed TX produced {bits.size} bits, expected 3000")
    return bits


# ============================================================
# RX BRANCHES
# Each returns:
#   protection_ms
#   synthesis_ms
#   total_ms
# ============================================================

def rx_plain(soft_metrics, valid_samples):
    total_start = time.perf_counter_ns()

    p0 = time.perf_counter_ns()
    hard = hard_slice(soft_metrics[:3000])
    codes = bits_to_indices_frame_major(hard, PLAIN_N_Q)
    p1 = time.perf_counter_ns()

    s0 = time.perf_counter_ns()
    _audio = decode_encodec_codes(codes, valid_samples)
    s1 = time.perf_counter_ns()

    return (
        (p1 - p0) / 1e6,
        (s1 - s0) / 1e6,
        (s1 - total_start) / 1e6,
    )


def rx_conv(soft_metrics, valid_samples):
    total_start = time.perf_counter_ns()

    p0 = time.perf_counter_ns()
    mother_soft = depuncture_soft(soft_metrics)
    info_bits = viterbi_decode_soft(mother_soft)
    codes = bits_to_indices_frame_major(info_bits, PROTECTED_N_Q)
    p1 = time.perf_counter_ns()

    s0 = time.perf_counter_ns()
    _audio = decode_encodec_codes(codes, valid_samples)
    s1 = time.perf_counter_ns()

    return (
        (p1 - p0) / 1e6,
        (s1 - s0) / 1e6,
        (s1 - total_start) / 1e6,
    )


def rx_proposed(soft_metrics, valid_samples):
    total_start = time.perf_counter_ns()

    p0 = time.perf_counter_ns()
    raw = np.asarray(
        soft_metrics[:3000], dtype=np.float32
    ).reshape(FRAMES_PER_CHUNK, VOGEO_BITS_PER_FRAME)
    llr = estimate_vogeo_llr(raw)
    corrected_codes = vogeo_rx.decode(llr)
    corrected_codes = np.asarray(corrected_codes, dtype=np.int64)
    p1 = time.perf_counter_ns()

    s0 = time.perf_counter_ns()
    _audio = decode_encodec_codes(corrected_codes, valid_samples)
    s1 = time.perf_counter_ns()

    return (
        (p1 - p0) / 1e6,
        (s1 - s0) / 1e6,
        (s1 - total_start) / 1e6,
    )


# ============================================================
# TIMING HELPERS
# ============================================================

def time_tx(fn, chunk):
    t0 = time.perf_counter_ns()
    bits = fn(chunk)
    t1 = time.perf_counter_ns()
    return bits, (t1 - t0) / 1e6


def make_rng(audio_id, chunk_id, method_index):
    # Deterministic but different stream per method/chunk.
    seed = (
        int(args.seed)
        + 1_000_003 * int(audio_id)
        + 10_007 * int(chunk_id)
        + 101 * int(method_index)
    ) & 0xFFFFFFFF
    return np.random.default_rng(seed)


# ============================================================
# WARM-UP
# ============================================================

print("============================================================")
print("OFFLINE PROCESSING-LATENCY BENCHMARK")
print("============================================================")
print("Dataset:", DATASET_DIR)
print("Audio files:", len(wav_files))
print("CPU threads:", args.threads)
print("Synthetic RX Es/N0:", args.snr_db, "dB")
print("VOGEO bundle:", VOGEO_BUNDLE_DIR)
print("Warm-up iterations:", args.warmup)
print()

warm_audio, _ = load_audio(wav_files[0])
warm_chunk, warm_valid = get_chunk(warm_audio, 0)

# Warm-up should not contaminate state used for real measurements.
for _ in range(args.warmup):
    vogeo_tx.reset()
    vogeo_rx.reset()

    p_bits = tx_plain(warm_chunk)
    c_bits = tx_conv(warm_chunk)
    v_bits = tx_proposed(warm_chunk)

    _ = rx_plain(
        bpsk_awgn(p_bits, args.snr_db, np.random.default_rng(11)),
        warm_valid,
    )
    _ = rx_conv(
        bpsk_awgn(c_bits, args.snr_db, np.random.default_rng(22)),
        warm_valid,
    )
    _ = rx_proposed(
        bpsk_awgn(v_bits, args.snr_db, np.random.default_rng(33)),
        warm_valid,
    )

vogeo_tx.reset()
vogeo_rx.reset()


# ============================================================
# BENCHMARK
# ============================================================

rows = []

for audio_id, audio_path in enumerate(wav_files):
    audio, total_chunks = load_audio(audio_path)

    # Preserve causal/stateful behavior across chunks within an utterance.
    vogeo_tx.reset()
    vogeo_rx.reset()

    print(
        f"[{audio_id + 1:3d}/{len(wav_files):3d}] "
        f"{os.path.basename(audio_path)} | chunks={total_chunks}"
    )

    for chunk_id in range(total_chunks):
        chunk, valid_samples = get_chunk(audio, chunk_id)

        # ---------------- TX ----------------
        plain_bits, plain_tx_ms = time_tx(tx_plain, chunk)
        conv_bits, conv_tx_ms = time_tx(tx_conv, chunk)
        prop_bits, prop_tx_ms = time_tx(tx_proposed, chunk)

        # ---------------- synthetic offline channel ----------------
        plain_soft = bpsk_awgn(
            plain_bits,
            args.snr_db,
            make_rng(audio_id, chunk_id, 0),
        )
        conv_soft = bpsk_awgn(
            conv_bits,
            args.snr_db,
            make_rng(audio_id, chunk_id, 1),
        )
        prop_soft = bpsk_awgn(
            prop_bits,
            args.snr_db,
            make_rng(audio_id, chunk_id, 2),
        )

        # ---------------- RX ----------------
        plain_prot_ms, plain_syn_ms, plain_rx_ms = rx_plain(
            plain_soft, valid_samples
        )
        conv_prot_ms, conv_syn_ms, conv_rx_ms = rx_conv(
            conv_soft, valid_samples
        )
        prop_prot_ms, prop_syn_ms, prop_rx_ms = rx_proposed(
            prop_soft, valid_samples
        )

        rows.extend([
            {
                "audio_id": audio_id,
                "chunk_id": chunk_id,
                "method": "uncoded",
                "tx_ms": plain_tx_ms,
                "rx_protection_ms": plain_prot_ms,
                "rx_synthesis_ms": plain_syn_ms,
                "rx_total_ms": plain_rx_ms,
                "tx_plus_rx_ms": plain_tx_ms + plain_rx_ms,
                "valid_samples": valid_samples,
                "snr_db_for_offline_metrics": args.snr_db,
            },
            {
                "audio_id": audio_id,
                "chunk_id": chunk_id,
                "method": "conv-coded",
                "tx_ms": conv_tx_ms,
                "rx_protection_ms": conv_prot_ms,
                "rx_synthesis_ms": conv_syn_ms,
                "rx_total_ms": conv_rx_ms,
                "tx_plus_rx_ms": conv_tx_ms + conv_rx_ms,
                "valid_samples": valid_samples,
                "snr_db_for_offline_metrics": args.snr_db,
            },
            {
                "audio_id": audio_id,
                "chunk_id": chunk_id,
                "method": "proposed",
                "tx_ms": prop_tx_ms,
                "rx_protection_ms": prop_prot_ms,
                "rx_synthesis_ms": prop_syn_ms,
                "rx_total_ms": prop_rx_ms,
                "tx_plus_rx_ms": prop_tx_ms + prop_rx_ms,
                "valid_samples": valid_samples,
                "snr_db_for_offline_metrics": args.snr_db,
            },
        ])


# ============================================================
# SAVE RAW + PER-AUDIO
# ============================================================

chunk_csv = f"{args.output_prefix}_per_chunk.csv"
audio_csv = f"{args.output_prefix}_per_audio.csv"
summary_csv = f"{args.output_prefix}_summary.csv"

df = pd.DataFrame(rows)
df.to_csv(chunk_csv, index=False)

metric_cols = [
    "tx_ms",
    "rx_protection_ms",
    "rx_synthesis_ms",
    "rx_total_ms",
    "tx_plus_rx_ms",
]

per_audio = (
    df.groupby(["audio_id", "method"], as_index=False)[metric_cols]
      .mean()
)
per_audio.to_csv(audio_csv, index=False)


# ============================================================
# SUMMARY WITH 95% CI ACROSS UTTERANCE MEANS
# ============================================================

summary_rows = []

for method in ["uncoded", "conv-coded", "proposed"]:
    sub = per_audio[per_audio["method"] == method]

    for metric in metric_cols:
        x = sub[metric].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        n = len(x)
        mean = float(np.mean(x)) if n else float("nan")
        median = float(np.median(x)) if n else float("nan")
        std = float(np.std(x, ddof=1)) if n > 1 else float("nan")
        p95 = float(np.percentile(x, 95)) if n else float("nan")

        if n > 1:
            sem = std / math.sqrt(n)
            crit = float(student_t.ppf(0.975, df=n - 1))
            ci95 = crit * sem
        else:
            ci95 = float("nan")

        summary_rows.append({
            "method": method,
            "metric": metric,
            "n_utterances": n,
            "mean_ms": mean,
            "median_ms": median,
            "std_ms": std,
            "ci95_halfwidth_ms": ci95,
            "ci95_low_ms": mean - ci95 if np.isfinite(ci95) else float("nan"),
            "ci95_high_ms": mean + ci95 if np.isfinite(ci95) else float("nan"),
            "p95_ms": p95,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(summary_csv, index=False)


# ============================================================
# PRINT PAPER-FRIENDLY TABLE
# ============================================================

def get_summary(method, metric):
    row = summary[
        (summary["method"] == method)
        & (summary["metric"] == metric)
    ].iloc[0]
    return float(row["mean_ms"]), float(row["ci95_halfwidth_ms"])


print()
print("============================================================")
print("PAPER-FRIENDLY LATENCY SUMMARY")
print("mean ± 95% CI across utterance-level mean latencies")
print("============================================================")
print(
    f"{'Method':<14} "
    f"{'TX (ms)':>18} "
    f"{'RX protection (ms)':>24} "
    f"{'RX total (ms)':>20} "
    f"{'TX+RX (ms)':>20}"
)

for method in ["uncoded", "conv-coded", "proposed"]:
    tx_m, tx_c = get_summary(method, "tx_ms")
    pr_m, pr_c = get_summary(method, "rx_protection_ms")
    rx_m, rx_c = get_summary(method, "rx_total_ms")
    ee_m, ee_c = get_summary(method, "tx_plus_rx_ms")

    print(
        f"{method:<14} "
        f"{tx_m:8.3f} ± {tx_c:6.3f} "
        f"{pr_m:10.3f} ± {pr_c:6.3f} "
        f"{rx_m:8.3f} ± {rx_c:6.3f} "
        f"{ee_m:8.3f} ± {ee_c:6.3f}"
    )

print()
print("Saved:")
print(" ", os.path.abspath(chunk_csv))
print(" ", os.path.abspath(audio_csv))
print(" ", os.path.abspath(summary_csv))
print()
print("IMPORTANT:")
print("These are offline computational processing times.")
print("Do not label them as OTA/end-to-end communication latency.")
