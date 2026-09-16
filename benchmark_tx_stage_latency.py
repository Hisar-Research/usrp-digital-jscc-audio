#!/usr/bin/env python3
"""
TX stage-by-stage latency benchmark for the current 30+10 SOFT VOGEO experiment.

Measures, on CPU and without SDR/GNU Radio:

  1) EnCodec waveform encoder
  2) EnCodec 4-codebook quantizer (uncoded 3 kbps)
  3) EnCodec 3-codebook quantizer (Conv / Proposed 2.25 kbps source)
  4) Plain bit packing
  5) Rate-3/4 punctured convolutional encoder + packing
  6) VOGEO learned protection encoder + packing

It also reconstructs standalone method TX times as:

  Uncoded  = EnCodec encoder + 4-codebook quantizer + plain packing
  Conv     = EnCodec encoder + 3-codebook quantizer + Conv encoder/packing
  Proposed = EnCodec encoder + 3-codebook quantizer + VOGEO encoder/packing

Important:
- This is processing time, NOT OTA/end-to-end latency.
- Model loading and disk I/O are excluded.
- VOGEO state is reset once per utterance, preserving temporal operation across chunks.
- 95% confidence intervals are computed across utterance-level mean latencies.

Example:
  python3 benchmark_tx_stage_latency.py \
      --num-audio 100 \
      --warmup 20 \
      --threads 1 \
      --dataset-dir "/path/to/LibriTTS/test-clean"
"""

import os
import sys
import glob
import math
import time
import json
import platform
import argparse

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
    description="Stage-by-stage TX processing latency benchmark."
)
parser.add_argument("--num-audio", type=int, default=100)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--min-duration", type=float, default=8.0)
parser.add_argument("--max-duration", type=float, default=15.0)
parser.add_argument("--dataset-dir", default=None)
parser.add_argument("--output-prefix", default="tx_stage_latency")
args = parser.parse_args()

if args.num_audio <= 0:
    raise ValueError("--num-audio must be positive")
if args.warmup < 0:
    raise ValueError("--warmup must be >= 0")
if args.threads <= 0:
    raise ValueError("--threads must be positive")

torch.set_num_threads(args.threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# PROJECT / VOGEO
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SOFT_REF_DIR = os.path.join(PROJECT_DIR, "reference_soft")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

required_py = ("vogeo_tx.py", "vogeo_frame.py")
missing = [
    name for name in required_py
    if not os.path.isfile(os.path.join(SOFT_REF_DIR, name))
]
if missing:
    raise RuntimeError(
        f"Missing files in {SOFT_REF_DIR}: " + ", ".join(missing)
    )

from reference_soft.vogeo_tx import VogeoTx


def _read_manifest(bundle_dir):
    p = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
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
            f"Could not find compatible 30+10 SOFT bundle under {SOFT_REF_DIR}"
        )

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


VOGEO_BUNDLE_DIR = _find_30plus10_soft_bundle()
vogeo = VogeoTx(VOGEO_BUNDLE_DIR, device="cpu")

if int(vogeo.n_q) != 3:
    raise RuntimeError(f"Expected VOGEO n_q=3, got {vogeo.n_q}")
if int(vogeo.parity_dim) != 10:
    raise RuntimeError(f"Expected parity_dim=10, got {vogeo.parity_dim}")

PROTECTED_BANDWIDTH_KBPS = float(vogeo.bandwidth)
PLAIN_BANDWIDTH_KBPS = 3.0

PROTECTED_N_Q = 3
PLAIN_N_Q = 4
FRAMES_PER_CHUNK = 75
BITS_PER_CODE = 10
CHUNK_SAMPLES = 24000


# ============================================================
# ENCODEC
# ============================================================

model = EncodecModel.encodec_model_24khz()
model.eval()

if int(model.sample_rate) != 24000:
    raise RuntimeError(f"Expected 24-kHz EnCodec, got {model.sample_rate}")


# ============================================================
# DATASET
# ============================================================

def find_dataset():
    if args.dataset_dir is not None:
        p = os.path.abspath(args.dataset_dir)
        if not os.path.isdir(p):
            raise RuntimeError(f"--dataset-dir not found: {p}")
        return p

    preferred = [
        os.path.join(PROJECT_DIR, "datasets", "LibriTTS", "test-clean"),
        os.path.join(PROJECT_DIR, "datasets", "LibriTTS", "LibriTTS", "test-clean"),
        os.path.join(PROJECT_DIR, "datasets", "test-clean"),
    ]

    for p in preferred:
        if os.path.isdir(p) and glob.glob(
            os.path.join(p, "**", "*.wav"), recursive=True
        ):
            return os.path.abspath(p)

    raise RuntimeError(
        "Could not locate LibriTTS test-clean. "
        "Use --dataset-dir /path/to/LibriTTS/test-clean"
    )


DATASET_DIR = find_dataset()

all_wavs = sorted(
    glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
)

wav_files = []
for p in all_wavs:
    try:
        info = sf.info(p)
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        continue

    if args.min_duration <= duration < args.max_duration:
        wav_files.append(p)

wav_files = wav_files[:args.num_audio]

if not wav_files:
    raise RuntimeError("No WAV files matched requested duration range")


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav.T.copy())
    audio = convert_audio(
        audio,
        int(sr),
        model.sample_rate,
        model.channels,
    )
    total_chunks = math.ceil(audio.shape[-1] / CHUNK_SAMPLES)
    return audio.contiguous(), total_chunks


def get_chunk(audio, chunk_id):
    start = chunk_id * CHUNK_SAMPLES
    end = min(start + CHUNK_SAMPLES, audio.shape[-1])
    chunk = audio[:, start:end]

    if chunk.shape[-1] < CHUNK_SAMPLES:
        chunk = torch.nn.functional.pad(
            chunk,
            (0, CHUNK_SAMPLES - chunk.shape[-1]),
        )

    return chunk.contiguous()


# ============================================================
# CONVOLUTIONAL ENCODER
# Same K=7, (133,171)_8, rate-3/4 puncturing as your experiment.
# ============================================================

K = 7
N_STATES = 1 << (K - 1)
G_POLY = (0o133, 0o171)
TAIL_BITS = K - 1
PUNCTURE_PATTERN = np.array([1, 1, 1, 0, 0, 1], dtype=np.uint8)

CONV_INFO_BITS = PROTECTED_N_Q * FRAMES_PER_CHUNK * BITS_PER_CODE
CONV_TRELLIS_INPUT_BITS = CONV_INFO_BITS + TAIL_BITS
CONV_MOTHER_BITS = 2 * CONV_TRELLIS_INPUT_BITS

CONV_PUNCTURE_MASK = np.tile(
    PUNCTURE_PATTERN.astype(bool),
    CONV_MOTHER_BITS // PUNCTURE_PATTERN.size,
)
CONV_USEFUL_BITS = int(CONV_PUNCTURE_MASK.sum())


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


def codes_to_frame_major_bits(codes_np):
    codes_np = np.asarray(codes_np, dtype=np.uint16)
    shifts = np.arange(
        BITS_PER_CODE - 1, -1, -1, dtype=np.uint16
    )
    bits = (
        (codes_np.T[:, :, None] >> shifts) & 1
    ).astype(np.uint8)
    return bits.reshape(-1)


def conv_encode(info_bits):
    info_bits = np.asarray(info_bits, dtype=np.uint8).reshape(-1)

    trellis_input = np.concatenate([
        info_bits,
        np.zeros(TAIL_BITS, dtype=np.uint8),
    ])

    mother = np.empty(
        (CONV_TRELLIS_INPUT_BITS, 2),
        dtype=np.uint8,
    )

    state = 0
    for i, bit in enumerate(trellis_input):
        bit = int(bit)
        mother[i] = OUT_BITS[state, bit]
        state = NEXT_STATE[state, bit]

    return mother.reshape(-1)[CONV_PUNCTURE_MASK]


def pack_plain(codes_np):
    bits = codes_to_frame_major_bits(codes_np)
    return np.packbits(bits, bitorder="big").tobytes()


def encode_and_pack_conv(codes_np):
    info_bits = codes_to_frame_major_bits(codes_np)
    coded_bits = conv_encode(info_bits)
    return np.packbits(coded_bits, bitorder="big").tobytes()


def pack_vogeo(vogeo_bits):
    b = np.asarray(vogeo_bits, dtype=np.uint8).reshape(-1)
    return np.packbits(b, bitorder="big").tobytes()


# ============================================================
# TIMED STAGES
# ============================================================

def ms(t0, t1):
    return (t1 - t0) / 1e6


@torch.no_grad()
def benchmark_chunk(chunk):
    """
    All timings are CPU synchronous because this benchmark runs on CPU.

    Returns component timings and reconstructed standalone TX totals.
    """

    # ---------------- shared EnCodec waveform encoder ----------------
    t0 = time.perf_counter_ns()
    z = model.encoder(chunk.unsqueeze(0))
    t1 = time.perf_counter_ns()
    encodec_encoder_ms = ms(t0, t1)

    # ---------------- 3-codebook quantizer ----------------
    t0 = time.perf_counter_ns()
    q_protected = model.quantizer(
        z,
        model.frame_rate,
        float(PROTECTED_BANDWIDTH_KBPS),
    )
    t1 = time.perf_counter_ns()
    quantizer_3cb_ms = ms(t0, t1)

    codes_protected_q = q_protected.codes
    codes_protected_np = (
        codes_protected_q[:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint16)
    )

    # ---------------- 4-codebook quantizer ----------------
    t0 = time.perf_counter_ns()
    q_plain = model.quantizer(
        z,
        model.frame_rate,
        float(PLAIN_BANDWIDTH_KBPS),
    )
    t1 = time.perf_counter_ns()
    quantizer_4cb_ms = ms(t0, t1)

    codes_plain_np = (
        q_plain.codes[:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint16)
    )

    # ---------------- uncoded packing ----------------
    t0 = time.perf_counter_ns()
    _plain_payload = pack_plain(codes_plain_np)
    t1 = time.perf_counter_ns()
    plain_pack_ms = ms(t0, t1)

    # ---------------- conv encode + packing ----------------
    t0 = time.perf_counter_ns()
    _conv_payload = encode_and_pack_conv(codes_protected_np)
    t1 = time.perf_counter_ns()
    conv_encode_pack_ms = ms(t0, t1)

    # ---------------- VOGEO learned protection ONLY ----------------
    t0 = time.perf_counter_ns()
    vogeo_bits = vogeo.encode(
        z,
        codes_protected_q,
    )
    t1 = time.perf_counter_ns()
    vogeo_encode_only_ms = ms(t0, t1)

    # ---------------- VOGEO packing ONLY ----------------
    t0 = time.perf_counter_ns()
    _vogeo_payload = pack_vogeo(vogeo_bits)
    t1 = time.perf_counter_ns()
    vogeo_pack_ms = ms(t0, t1)

    # Standalone method totals reconstructed from the measured stages.
    uncoded_total_ms = (
        encodec_encoder_ms
        + quantizer_4cb_ms
        + plain_pack_ms
    )

    conv_total_ms = (
        encodec_encoder_ms
        + quantizer_3cb_ms
        + conv_encode_pack_ms
    )

    proposed_total_ms = (
        encodec_encoder_ms
        + quantizer_3cb_ms
        + vogeo_encode_only_ms
        + vogeo_pack_ms
    )

    return {
        "encodec_encoder_ms": encodec_encoder_ms,
        "quantizer_3cb_ms": quantizer_3cb_ms,
        "quantizer_4cb_ms": quantizer_4cb_ms,
        "plain_pack_ms": plain_pack_ms,
        "conv_encode_pack_ms": conv_encode_pack_ms,
        "vogeo_encode_only_ms": vogeo_encode_only_ms,
        "vogeo_pack_ms": vogeo_pack_ms,
        "uncoded_total_ms": uncoded_total_ms,
        "conv_total_ms": conv_total_ms,
        "proposed_total_ms": proposed_total_ms,
    }


# ============================================================
# WARM-UP
# ============================================================

print("============================================================")
print("TX STAGE-BY-STAGE LATENCY BENCHMARK")
print("============================================================")
print("Dataset:", DATASET_DIR)
print("Audio files:", len(wav_files))
print("Threads:", args.threads)
print("Bundle:", VOGEO_BUNDLE_DIR)
print("PyTorch:", torch.__version__)
print("Python:", platform.python_version())
print("Machine:", platform.machine())
print("Processor:", platform.processor() or "not reported")
print()

warm_audio, _ = load_audio(wav_files[0])
warm_chunk = get_chunk(warm_audio, 0)

vogeo.reset()
for _ in range(args.warmup):
    _ = benchmark_chunk(warm_chunk)

vogeo.reset()


# ============================================================
# MAIN BENCHMARK
# ============================================================

rows = []

for audio_id, audio_path in enumerate(wav_files):
    audio, total_chunks = load_audio(audio_path)

    # Preserve temporal VOGEO state within one utterance.
    vogeo.reset()

    print(
        f"[{audio_id + 1:3d}/{len(wav_files):3d}] "
        f"{os.path.basename(audio_path)} | chunks={total_chunks}"
    )

    for chunk_id in range(total_chunks):
        chunk = get_chunk(audio, chunk_id)
        result = benchmark_chunk(chunk)

        row = {
            "audio_id": audio_id,
            "chunk_id": chunk_id,
            **result,
        }
        rows.append(row)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

df = pd.DataFrame(rows)

chunk_csv = f"{args.output_prefix}_per_chunk.csv"
audio_csv = f"{args.output_prefix}_per_audio.csv"
summary_csv = f"{args.output_prefix}_summary.csv"

df.to_csv(chunk_csv, index=False)

stage_cols = [
    "encodec_encoder_ms",
    "quantizer_3cb_ms",
    "quantizer_4cb_ms",
    "plain_pack_ms",
    "conv_encode_pack_ms",
    "vogeo_encode_only_ms",
    "vogeo_pack_ms",
    "uncoded_total_ms",
    "conv_total_ms",
    "proposed_total_ms",
]

per_audio = (
    df.groupby("audio_id", as_index=False)[stage_cols]
      .mean()
)
per_audio.to_csv(audio_csv, index=False)


# ============================================================
# 95% CI ACROSS UTTERANCE-LEVEL MEANS
# ============================================================

summary_rows = []

for metric in stage_cols:
    x = per_audio[metric].to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1)) if n > 1 else float("nan")

    if n > 1:
        sem = std / math.sqrt(n)
        crit = float(student_t.ppf(0.975, df=n - 1))
        ci95 = crit * sem
    else:
        ci95 = float("nan")

    summary_rows.append({
        "metric": metric,
        "n_utterances": n,
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": std,
        "ci95_halfwidth_ms": ci95,
        "ci95_low_ms": mean - ci95 if np.isfinite(ci95) else float("nan"),
        "ci95_high_ms": mean + ci95 if np.isfinite(ci95) else float("nan"),
        "p95_ms": float(np.percentile(x, 95)),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(summary_csv, index=False)


# ============================================================
# PRINT RESULT
# ============================================================

def result(metric):
    r = summary[summary["metric"] == metric].iloc[0]
    return float(r["mean_ms"]), float(r["ci95_halfwidth_ms"])


print()
print("============================================================")
print("STAGE LATENCY: mean ± 95% CI across utterances")
print("============================================================")

labels = [
    ("EnCodec encoder", "encodec_encoder_ms"),
    ("3-codebook quantizer", "quantizer_3cb_ms"),
    ("4-codebook quantizer", "quantizer_4cb_ms"),
    ("Plain packing", "plain_pack_ms"),
    ("Conv encode + packing", "conv_encode_pack_ms"),
    ("VOGEO encoder ONLY", "vogeo_encode_only_ms"),
    ("VOGEO packing", "vogeo_pack_ms"),
]

for label, metric in labels:
    mean, ci = result(metric)
    print(f"{label:<28} {mean:9.3f} ± {ci:7.3f} ms")

print()
print("Standalone reconstructed TX totals")
print("-----------------------------------")

for label, metric in [
    ("Uncoded", "uncoded_total_ms"),
    ("Conv-coded", "conv_total_ms"),
    ("Proposed", "proposed_total_ms"),
]:
    mean, ci = result(metric)
    print(f"{label:<14} {mean:9.3f} ± {ci:7.3f} ms")

v_mean, v_ci = result("vogeo_encode_only_ms")
e_mean, _ = result("encodec_encoder_ms")

print()
print(
    f"VOGEO encoder / EnCodec encoder = "
    f"{100.0 * v_mean / max(e_mean, 1e-12):.2f}%"
)
print()
print("Saved:")
print(" ", os.path.abspath(chunk_csv))
print(" ", os.path.abspath(audio_csv))
print(" ", os.path.abspath(summary_csv))
print()
print("Use 'VOGEO encoder ONLY' to support claims about learned TX overhead.")
print("Do not call these measurements OTA/end-to-end latency.")
