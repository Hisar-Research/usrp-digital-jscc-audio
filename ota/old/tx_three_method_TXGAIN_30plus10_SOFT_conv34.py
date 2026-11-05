import os
import sys
import glob
import math
import socket
import struct
import time
import json

import numpy as np
import soundfile as sf
import torch

from encodec import EncodecModel
from encodec.utils import convert_audio


# ============================================================
# SOFT VOGEO PACKAGE
# ============================================================

# Exact project/reference paths on your machine.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# reference_soft is inside the project directory
SOFT_REF_DIR = os.path.join(PROJECT_DIR, "reference_soft")

# Import reference_soft as a package because its modules use relative imports.
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

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
        f"SOFT reference path is {SOFT_REF_DIR}, but these files are missing: "
        + ", ".join(missing_py)
    )

from reference_soft.vogeo_tx import VogeoTx


# ============================================================
# NETWORK
# ============================================================

GRC_HOST, GRC_PORT = "127.0.0.1", 5000
RX_HOST, RX_PORT = "127.0.0.1", 5002

rf_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
header_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ============================================================
# DATASET
# ============================================================

def _find_librtts_test_clean():
    """Locate LibriTTS test-clean relative to this project."""
    preferred = [
        os.path.join(PROJECT_DIR, "datasets", "LibriTTS", "test-clean"),
        os.path.join(PROJECT_DIR, "datasets", "LibriTTS", "LibriTTS", "test-clean"),
        os.path.join(PROJECT_DIR, "datasets", "test-clean"),
    ]

    def has_wav(path):
        if not os.path.isdir(path):
            return False
        return bool(glob.glob(os.path.join(path, "**", "*.wav"), recursive=True))

    for path in preferred:
        if has_wav(path):
            return os.path.abspath(path)

    datasets_root = os.path.join(PROJECT_DIR, "datasets")
    if os.path.isdir(datasets_root):
        for root, dirs, _files in os.walk(datasets_root):
            if os.path.basename(root).lower() == "test-clean" and has_wav(root):
                return os.path.abspath(root)

    raise RuntimeError(
        "Could not locate LibriTTS test-clean under the project.\n"
        f"Searched below: {datasets_root}\n"
        "Expected something like datasets/LibriTTS/test-clean or "
        "datasets/LibriTTS/LibriTTS/test-clean."
    )


DATASET_DIR = _find_librtts_test_clean()

all_wav_files = sorted(
    glob.glob(
        os.path.join(DATASET_DIR, "**", "*.wav"),
        recursive=True,
    )
)

if not all_wav_files:
    raise RuntimeError(
        f"No WAV files found under {DATASET_DIR}"
    )

# Keep only utterances whose ORIGINAL duration satisfies:
#     8.0 seconds <= duration < 15.0 seconds
MIN_AUDIO_DURATION_S = 8.0
MAX_AUDIO_DURATION_S = 15.0

wav_files = []

for wav_path in all_wav_files:
    try:
        info = sf.info(wav_path)
        duration_s = float(info.frames) / float(info.samplerate)
    except Exception as exc:
        print(
            f"[TX] Skipping unreadable WAV: {wav_path} | {exc}"
        )
        continue

    if (
        MIN_AUDIO_DURATION_S
        <= duration_s
        < MAX_AUDIO_DURATION_S
    ):
        wav_files.append(wav_path)

if not wav_files:
    raise RuntimeError(
        f"No WAV files found with duration "
        f"{MIN_AUDIO_DURATION_S:.1f} <= duration < "
        f"{MAX_AUDIO_DURATION_S:.1f} seconds under {DATASET_DIR}"
    )

NUM_AUDIO = 100
wav_files = wav_files[:NUM_AUDIO]

print(f"[TX] Dataset             = {DATASET_DIR}")
print(
    f"[TX] Duration filter     = "
    f"{MIN_AUDIO_DURATION_S:.1f} <= duration < "
    f"{MAX_AUDIO_DURATION_S:.1f} s"
)
print(f"[TX] WAV files selected  = {len(wav_files)}")


# ============================================================
# ENCODEC + SOFT VOGEO TX BUNDLE
# ============================================================

def _bundle_decision_mode(bundle_dir):
    """
    Return (decision_mode, manifest_path) for a VOGEO bundle.
    Only JSON files at the bundle root or one level below are inspected.
    """
    json_candidates = []

    try:
        for root, dirs, files in os.walk(bundle_dir):
            rel = os.path.relpath(root, bundle_dir)
            depth = 0 if rel == "." else rel.count(os.sep) + 1

            if depth > 1:
                dirs[:] = []
                continue

            for name in files:
                if name.lower().endswith(".json"):
                    json_candidates.append(os.path.join(root, name))
    except OSError:
        return None, None

    # Prefer manifest-looking names.
    json_candidates.sort(
        key=lambda p: (
            0 if "manifest" in os.path.basename(p).lower() else 1,
            p,
        )
    )

    for path in json_candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        if isinstance(obj, dict) and "decision_mode" in obj:
            return str(obj.get("decision_mode")).lower(), path

    return None, None


def _read_manifest(bundle_dir):
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _find_30plus10_soft_bundle():
    """Find a compatible 3-codebook + 10-parity SOFT VOGEO bundle."""
    preferred_roots = [
        os.path.join(SOFT_REF_DIR, "bundle_3_4"),
        os.path.join(SOFT_REF_DIR, "bundle_3_out_4"),
        os.path.join(SOFT_REF_DIR, "bundle_30_10"),
    ]

    candidates = []
    seen = set()

    def add_manifest_dir(path):
        path = os.path.abspath(path)
        if path in seen:
            return
        seen.add(path)

        manifest = _read_manifest(path)
        if manifest is None:
            return

        mode = manifest.get("decision_mode")
        if mode is not None and str(mode).lower() != "soft":
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

        n_q = as_int("n_q", "num_codebooks", "num_quantizers")
        parity = as_int("parity_dim", "parity_bits", "parity_bits_per_frame")
        bw = as_float("bandwidth", "bandwidth_kbps", "encodec_bandwidth")

        if n_q is not None and n_q != 3:
            return
        if parity is not None and parity != 10:
            return

        score = 0
        lower = path.lower()
        if "bundle_3_4" in lower:
            score += 100
        if "bundle_3_out_4" in lower:
            score += 95
        if "bundle_30_10" in lower:
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
            add_manifest_dir(root)
        for walk_root, _dirs, files in os.walk(root):
            if "manifest.json" in files:
                add_manifest_dir(walk_root)

    if not candidates and os.path.isdir(SOFT_REF_DIR):
        for walk_root, _dirs, files in os.walk(SOFT_REF_DIR):
            if "manifest.json" in files:
                add_manifest_dir(walk_root)

    if not candidates:
        raise RuntimeError(
            "Could not find a compatible 30+10 SOFT VOGEO bundle under:\n"
            f"  {SOFT_REF_DIR}\n"
            "Expected a directory containing manifest.json."
        )

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


VOGEO_BUNDLE_DIR = _find_30plus10_soft_bundle()
BUNDLE_NAME = os.path.basename(VOGEO_BUNDLE_DIR)

print(f"[TX] SOFT reference code = {SOFT_REF_DIR}")
print(f"[TX] VOGEO bundle        = {VOGEO_BUNDLE_DIR}")

DEVICE = "cpu"

vogeo = VogeoTx(
    VOGEO_BUNDLE_DIR,
    device=DEVICE,
)

# Equal-channel-budget experiment:
#   PLAIN baseline: EnCodec 3.0 kbps = 4 RVQ codebooks = 3000 useful bits/s
#   CONV baseline : EnCodec 2.25 kbps = 3 RVQ codebooks, then punctured rate-3/4 FEC
#   PROPOSED      : SOFT VOGEO = 3 EnCodec codebooks (30 bits/frame)
#                   + 10 learned bits/frame = 40 bits/frame
PLAIN_BANDWIDTH_KBPS = 3.0
PLAIN_N_Q = 4

PROTECTED_BANDWIDTH_KBPS = float(vogeo.bandwidth)   # expected 2.25
PROTECTED_N_Q = int(vogeo.n_q)                      # expected 3

FRAMES_PER_CHUNK = 75
BITS_PER_CODE = 10

model = EncodecModel.encodec_model_24khz()
model.eval()

if int(model.sample_rate) != 24000:
    raise RuntimeError(
        f"Expected 24-kHz EnCodec model, "
        f"got sample_rate={model.sample_rate}"
    )

CHUNK_SAMPLES = 24000

if int(vogeo.n_q) != PROTECTED_N_Q:
    raise RuntimeError(
        f"Soft VOGEO bundle n_q={vogeo.n_q}, "
        f"expected {PROTECTED_N_Q}"
    )

if PROTECTED_N_Q != 3:
    raise RuntimeError(
        f"This experiment expects the VOGEO/Conv source to use 3 RVQ codebooks, "
        f"got {PROTECTED_N_Q}"
    )

if PLAIN_N_Q != 4:
    raise RuntimeError(
        f"This experiment expects the plain 3-kbps baseline to use 4 RVQ codebooks, "
        f"got {PLAIN_N_Q}"
    )

if int(vogeo.parity_dim) != 10:
    raise RuntimeError(
        f"Soft VOGEO parity_dim={vogeo.parity_dim}, "
        f"expected 10"
    )


# ============================================================
# PLAIN ENCODEC 3.0 kbps
#
# 4 codebooks * 75 frames * 10 bits = 3000 bits = 375 bytes.
# This is the uncoded 3-kbps baseline.
# ============================================================

PLAIN_INFO_BITS = (
    PLAIN_N_Q
    * FRAMES_PER_CHUNK
    * BITS_PER_CODE
)                                                   # 3000

PLAIN_BYTES = math.ceil(
    PLAIN_INFO_BITS / 8
)                                                   # 375

PLAIN_PHYSICAL_BITS = (
    PLAIN_BYTES * 8
)                                                   # 3000

PLAIN_PAD_BITS = (
    PLAIN_PHYSICAL_BITS
    - PLAIN_INFO_BITS
)                                                   # 0


# ============================================================
# RATE-3/4 PUNCTURED CONVOLUTIONAL CODE
#
# Mother code: K=7, rate 1/2, generators (133,171) octal.
# Puncture pattern over the serialized mother-code bits:
#     [1, 1, 1, 0, 0, 1]
# This keeps 4 of every 6 mother-code bits -> effective rate 3/4.
#
# ONE terminated codeword per 1-second chunk:
#   3 codebooks * 75 frames * 10 bits = 2250 information bits
#   + 6 zero tail bits = 2256 trellis inputs
#   -> 4512 mother-code bits
#   -> 3008 transmitted punctured bits = 376 bytes
#
# The extra 8 bits over the ideal 3000-bit budget are only the
# termination overhead, consistent with the terminated baseline used before.
# ============================================================

K = 7
N_STATES = 1 << (K - 1)
G_POLY = (0o133, 0o171)
TAIL_BITS = K - 1

# IEEE-802.11-style rate-3/4 puncturing for the K=7 (133,171) mother code.
PUNCTURE_PATTERN = np.array(
    [1, 1, 1, 0, 0, 1],
    dtype=np.uint8,
)

CONV_INFO_BITS = (
    PROTECTED_N_Q
    * FRAMES_PER_CHUNK
    * BITS_PER_CODE
)                                                   # 2250

CONV_TRELLIS_INPUT_BITS = (
    CONV_INFO_BITS
    + TAIL_BITS
)                                                   # 2256

CONV_MOTHER_BITS = (
    2
    * CONV_TRELLIS_INPUT_BITS
)                                                   # 4512

if CONV_MOTHER_BITS % PUNCTURE_PATTERN.size != 0:
    raise RuntimeError(
        "Mother-code length is not aligned to the puncture period"
    )

CONV_PUNCTURE_MASK = np.tile(
    PUNCTURE_PATTERN.astype(bool),
    CONV_MOTHER_BITS // PUNCTURE_PATTERN.size,
)

CONV_USEFUL_BITS = int(CONV_PUNCTURE_MASK.sum())    # 3008

CONV_BYTES = math.ceil(
    CONV_USEFUL_BITS / 8
)                                                   # 376

CONV_PHYSICAL_BITS = (
    CONV_BYTES * 8
)                                                   # 3008

CONV_PAD_BITS = (
    CONV_PHYSICAL_BITS
    - CONV_USEFUL_BITS
)                                                   # 0


def _parity(x):
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def _build_trellis():
    nxt = np.zeros(
        (N_STATES, 2),
        dtype=np.int64,
    )

    out = np.zeros(
        (N_STATES, 2, 2),
        dtype=np.uint8,
    )

    for state in range(N_STATES):
        for bit in (0, 1):
            reg = (
                (bit << (K - 1))
                | state
            )

            for j, g in enumerate(G_POLY):
                out[state, bit, j] = (
                    _parity(reg & g)
                )

            nxt[state, bit] = (
                reg >> 1
            )

    return nxt, out


NEXT_STATE, OUT_BITS = _build_trellis()


# ============================================================
# SOFT VOGEO TX
#
# SOFT VOGEO bundle transmitter format:
#   3 EnCodec RVQ codes/frame = 30 bits/frame
#   + 10 learned parity bits/frame
#   = 40 bits/frame
#
# 75 frames * 40 bits = 3000 bits = 375 bytes exactly.
# ============================================================

VOGEO_BITS_PER_FRAME = (
    int(vogeo.n_q) * BITS_PER_CODE
    + int(vogeo.parity_dim)
)                                                   # 40

VOGEO_USEFUL_BITS = (
    FRAMES_PER_CHUNK
    * VOGEO_BITS_PER_FRAME
)                                                   # 3000

VOGEO_BYTES = math.ceil(
    VOGEO_USEFUL_BITS / 8
)                                                   # 375

VOGEO_PHYSICAL_BITS = (
    VOGEO_BYTES * 8
)                                                   # 3000

VOGEO_PAD_BITS = (
    VOGEO_PHYSICAL_BITS
    - VOGEO_USEFUL_BITS
)                                                   # 0


# ============================================================
# ONE SUPER-PACKET
#
# plain 3 kbps = 375 B
# conv(3/4)      = 376 B
# soft VOGEO     = 375 B
#
# total = 1126 B = 9008 RF bits
# ============================================================

ORDERS = (
    ("plain", "conv", "vogeo"),
    ("conv", "vogeo", "plain"),
    ("vogeo", "plain", "conv"),
)

SEGMENT_BYTES = {
    "plain": PLAIN_BYTES,
    "conv": CONV_BYTES,
    "vogeo": VOGEO_BYTES,
}

SEGMENT_PHYSICAL_BITS = {
    "plain": PLAIN_PHYSICAL_BITS,
    "conv": CONV_PHYSICAL_BITS,
    "vogeo": VOGEO_PHYSICAL_BITS,
}

SUPER_PACKET_BYTES = (
    PLAIN_BYTES
    + CONV_BYTES
    + VOGEO_BYTES
)                                                   # 1126

SUPER_PACKET_BITS = (
    SUPER_PACKET_BYTES * 8
)                                                   # 9008


# ============================================================
# DIRECT METADATA HEADER
# ============================================================

MAGIC = b"VQ"
TYPE_SUPER_SOFT = 5

HEADER_FORMAT = "!2sBIHHH"
HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)                                                   # 13 bytes


# ============================================================
# TIMING / OUTPUT
# ============================================================

CHUNK_INTERVAL_S = 0.25
AUDIO_INTERVAL_S = 5

ORIGINAL_DIR = "original_audio"
os.makedirs(
    ORIGINAL_DIR,
    exist_ok=True,
)


# ============================================================
# PACKING
# ============================================================

def codes_to_plain_bits(codes_np):
    """
    [2,75] RVQ indices
        -> [2,75,10]
        -> 1500 bits.

    This is codebook-major and is used only by the PLAIN branch.
    """
    codes_np = np.asarray(
        codes_np,
        dtype=np.uint16,
    )

    expected = (
        PLAIN_N_Q,
        FRAMES_PER_CHUNK,
    )

    if codes_np.shape != expected:
        raise ValueError(
            f"Plain codes shape={codes_np.shape}, "
            f"expected {expected}"
        )

    if np.any(codes_np > 1023):
        raise ValueError(
            "EnCodec code outside 0..1023"
        )

    shifts = np.arange(
        BITS_PER_CODE - 1,
        -1,
        -1,
        dtype=np.uint16,
    )

    bits = (
        (
            codes_np[:, :, None]
            >> shifts
        )
        & 1
    ).astype(
        np.uint8
    ).reshape(-1)

    if bits.size != PLAIN_INFO_BITS:
        raise RuntimeError(
            f"Plain bits={bits.size}, "
            f"expected {PLAIN_INFO_BITS}"
        )

    return bits


def pack_plain(codes_np):
    bits = codes_to_plain_bits(
        codes_np
    )

    payload = np.packbits(
        bits,
        bitorder="big",
    ).tobytes()

    if len(payload) != PLAIN_BYTES:
        raise RuntimeError(
            f"Plain payload={len(payload)} B, "
            f"expected {PLAIN_BYTES} B"
        )

    return payload


def codes_to_conv_info_bits(codes_np):
    """
    Conv branch is frame-major:

        [3,75]
          -> transpose [75,3]
          -> [75,3,10]
          -> 2250 bits.
    """
    codes_np = np.asarray(
        codes_np,
        dtype=np.uint16,
    )

    expected = (
        PROTECTED_N_Q,
        FRAMES_PER_CHUNK,
    )

    if codes_np.shape != expected:
        raise ValueError(
            f"Conv codes shape={codes_np.shape}, "
            f"expected {expected}"
        )

    shifts = np.arange(
        BITS_PER_CODE - 1,
        -1,
        -1,
        dtype=np.uint16,
    )

    bits = (
        (
            codes_np.T[:, :, None]
            >> shifts
        )
        & 1
    ).astype(np.uint8)

    info_bits = bits.reshape(-1)

    if info_bits.size != CONV_INFO_BITS:
        raise RuntimeError(
            f"Conv information bits="
            f"{info_bits.size}, "
            f"expected {CONV_INFO_BITS}"
        )

    return info_bits


def conv_encode(info_bits):
    """
    K=7 rate-1/2 mother convolutional encoder, terminated with
    six zero tail bits, followed by rate-3/4 puncturing.
    """
    info_bits = np.asarray(
        info_bits,
        dtype=np.uint8,
    ).reshape(-1)

    if info_bits.size != CONV_INFO_BITS:
        raise ValueError(
            f"Expected {CONV_INFO_BITS} "
            f"information bits, "
            f"got {info_bits.size}"
        )

    trellis_input = np.concatenate(
        (
            info_bits,
            np.zeros(
                TAIL_BITS,
                dtype=np.uint8,
            ),
        )
    )

    mother = np.empty(
        (
            CONV_TRELLIS_INPUT_BITS,
            2,
        ),
        dtype=np.uint8,
    )

    state = 0

    for i, bit in enumerate(trellis_input):
        bit = int(bit)
        mother[i] = OUT_BITS[state, bit]
        state = NEXT_STATE[state, bit]

    if state != 0:
        raise RuntimeError(
            f"Conv encoder final state={state}, expected 0"
        )

    mother = mother.reshape(-1)

    if mother.size != CONV_MOTHER_BITS:
        raise RuntimeError(
            f"Mother-code bits={mother.size}, expected {CONV_MOTHER_BITS}"
        )

    punctured = mother[CONV_PUNCTURE_MASK]

    if punctured.size != CONV_USEFUL_BITS:
        raise RuntimeError(
            f"Punctured coded bits={punctured.size}, "
            f"expected {CONV_USEFUL_BITS}"
        )

    return punctured


def pack_conv(codes_np):
    info_bits = (
        codes_to_conv_info_bits(
            codes_np
        )
    )

    coded_bits = conv_encode(
        info_bits
    )

    payload = np.packbits(
        coded_bits,
        bitorder="big",
    ).tobytes()

    if len(payload) != CONV_BYTES:
        raise RuntimeError(
            f"Conv payload="
            f"{len(payload)} B, "
            f"expected {CONV_BYTES} B"
        )

    return payload


def pack_vogeo(vogeo_bits):
    """
    Hard VOGEO output is already a hard uint8 bit matrix:
        [75,40].
    """
    vogeo_bits = np.asarray(
        vogeo_bits,
        dtype=np.uint8,
    )

    expected = (
        FRAMES_PER_CHUNK,
        VOGEO_BITS_PER_FRAME,
    )

    if vogeo_bits.shape != expected:
        raise ValueError(
            f"VOGEO hard bits shape="
            f"{vogeo_bits.shape}, "
            f"expected {expected}"
        )

    if np.any(
        (vogeo_bits != 0)
        & (vogeo_bits != 1)
    ):
        raise ValueError(
            "VOGEO output is not hard binary"
        )

    payload = np.packbits(
        vogeo_bits.reshape(-1),
        bitorder="big",
    ).tobytes()

    if len(payload) != VOGEO_BYTES:
        raise RuntimeError(
            f"VOGEO payload="
            f"{len(payload)} B, "
            f"expected {VOGEO_BYTES} B"
        )

    return payload


def make_header(
    audio_id,
    chunk_id,
    total_chunks,
    valid_samples,
):
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        TYPE_SUPER_SOFT,
        audio_id & 0xFFFFFFFF,
        chunk_id,
        total_chunks,
        valid_samples,
    )


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):
    wav, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    audio = torch.from_numpy(
        wav.T.copy()
    )

    audio = convert_audio(
        audio,
        int(sr),
        model.sample_rate,
        model.channels,
    )

    total_chunks = math.ceil(
        audio.shape[-1]
        / CHUNK_SAMPLES
    )

    return audio, total_chunks


def get_chunk(
    audio,
    chunk_id,
):
    start = (
        chunk_id
        * CHUNK_SAMPLES
    )

    end = min(
        start + CHUNK_SAMPLES,
        audio.shape[-1],
    )

    chunk = audio[
        :,
        start:end,
    ]

    valid_samples = int(
        chunk.shape[-1]
    )

    if valid_samples <= 0:
        raise RuntimeError(
            f"Empty chunk {chunk_id}"
        )

    if (
        valid_samples
        < CHUNK_SAMPLES
    ):
        chunk = (
            torch.nn.functional.pad(
                chunk,
                (
                    0,
                    CHUNK_SAMPLES
                    - valid_samples,
                ),
            )
        )

    if (
        chunk.shape[-1]
        != CHUNK_SAMPLES
    ):
        raise RuntimeError(
            f"Chunk has "
            f"{chunk.shape[-1]} samples, "
            f"expected {CHUNK_SAMPLES}"
        )

    return (
        chunk.contiguous(),
        valid_samples,
    )


# ============================================================
# ONE ENCODEC PASS -> THREE METHODS
# ============================================================

@torch.no_grad()
def encode_all_three(chunk):
    """
    ONE EnCodec encoder pass, then TWO quantizer operating points.

    PLAIN:
        z -> EnCodec quantizer at 3.0 kbps -> 4 codebooks -> 3000 bits.

    CONV:
        z -> EnCodec quantizer at the VOGEO bundle bandwidth (2.25 kbps)
          -> 3 codebooks -> 2250 information bits
          -> terminated K=7 mother code + rate-3/4 puncturing
          -> 3008 transmitted coded bits.

    PROPOSED / SOFT VOGEO:
        uses the SAME original continuous latent z and the SAME 2.25-kbps
        3-codebook codes as the Conv source, then adds 10 learned bits/frame
        -> 3000 transmitted bits.

    This gives an approximately equal 3-kbit/s channel budget:
        plain=3000, conv=3008 physical, proposed=3000 bits/chunk.
    """
    if chunk.ndim != 2:
        raise ValueError(
            f"Expected [channels,samples], "
            f"got {tuple(chunk.shape)}"
        )

    if chunk.shape[-1] != CHUNK_SAMPLES:
        raise ValueError(
            f"Expected {CHUNK_SAMPLES} samples, "
            f"got {chunk.shape[-1]}"
        )

    start = time.perf_counter()

    # Shared continuous EnCodec latent.
    z = model.encoder(
        chunk.unsqueeze(0)
    )

    # --------------------------------------------------------
    # 2.25-kbps / 3-codebook representation:
    # used by Conv and SOFT VOGEO.
    # --------------------------------------------------------
    q_protected = model.quantizer(
        z,
        model.frame_rate,
        float(PROTECTED_BANDWIDTH_KBPS),
    )

    codes_protected_q = q_protected.codes

    protected_expected = (
        PROTECTED_N_Q,
        1,
        FRAMES_PER_CHUNK,
    )

    if tuple(codes_protected_q.shape) != protected_expected:
        raise ValueError(
            f"Expected protected/2.25-kbps RVQ codes {protected_expected}, "
            f"got {tuple(codes_protected_q.shape)}. "
            f"bundle bandwidth={PROTECTED_BANDWIDTH_KBPS}"
        )

    codes_protected_np = (
        codes_protected_q[:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint16)
    )

    # --------------------------------------------------------
    # 3.0-kbps / 4-codebook representation:
    # used ONLY by the uncoded plain baseline.
    # --------------------------------------------------------
    q_plain = model.quantizer(
        z,
        model.frame_rate,
        float(PLAIN_BANDWIDTH_KBPS),
    )

    codes_plain_q = q_plain.codes

    plain_expected = (
        PLAIN_N_Q,
        1,
        FRAMES_PER_CHUNK,
    )

    if tuple(codes_plain_q.shape) != plain_expected:
        raise ValueError(
            f"Expected plain 3-kbps RVQ codes {plain_expected}, "
            f"got {tuple(codes_plain_q.shape)}"
        )

    codes_plain_np = (
        codes_plain_q[:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint16)
    )

    plain_payload = pack_plain(
        codes_plain_np
    )

    conv_payload = pack_conv(
        codes_protected_np
    )

    # SOFT VOGEO TX keeps the original z + 3-codebook input.
    vogeo_bits = vogeo.encode(
        z,
        codes_protected_q,
    )

    vogeo_payload = pack_vogeo(
        vogeo_bits
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "plain": plain_payload,
        "conv": conv_payload,
        "vogeo": vogeo_payload,
    }, elapsed


def build_super_packet(
    payloads,
    chunk_id,
):
    order = ORDERS[
        chunk_id
        % len(ORDERS)
    ]

    packet = b"".join(
        payloads[name]
        for name in order
    )

    if (
        len(packet)
        != SUPER_PACKET_BYTES
    ):
        raise RuntimeError(
            f"Super-packet="
            f"{len(packet)} B, "
            f"expected "
            f"{SUPER_PACKET_BYTES} B"
        )

    return packet, order


# ============================================================
# STARTUP
# ============================================================

print(
    "============================================================"
)
print(
    "THREE-METHOD TX-GAIN SOFT SUPER-PACKET TX | 30+10 + CONV 3/4"
)
print(
    "============================================================"
)
print(
    "Soft VOGEO run ID      =",
    vogeo.run_id,
)
print(
    "Bundle quantizer BW    =",
    vogeo.bandwidth,
    "kbps",
)
print(
    "Plain EnCodec BW       =",
    PLAIN_BANDWIDTH_KBPS,
    "kbps",
)
print(
    "Plain EnCodec n_q      =",
    PLAIN_N_Q,
)
print(
    "Conv/VOGEO source BW   =",
    PROTECTED_BANDWIDTH_KBPS,
    "kbps",
)
print(
    "Conv/VOGEO source n_q  =",
    PROTECTED_N_Q,
)
print(
    "Sample rate            =",
    model.sample_rate,
    "Hz",
)
print(
    "Chunk samples          =",
    CHUNK_SAMPLES,
)
print(
    "Frames/chunk           =",
    FRAMES_PER_CHUNK,
)
print()
print("PLAIN ENCODEC 3 KBPS")
print(
    "  useful bits          =",
    PLAIN_INFO_BITS,
)
print(
    "  bytes                =",
    PLAIN_BYTES,
)
print(
    "  physical bits        =",
    PLAIN_PHYSICAL_BITS,
)
print()
print("CONV: ENCODEC 2.25 KBPS + RATE-3/4 PUNCTURED")
print(
    "  useful coded bits    =",
    CONV_USEFUL_BITS,
)
print(
    "  bytes                =",
    CONV_BYTES,
)
print(
    "  physical bits        =",
    CONV_PHYSICAL_BITS,
)
print()
print("PROPOSED SOFT VOGEO")
print(
    "  parity/frame         =",
    vogeo.parity_dim,
)
print(
    "  bits/frame           =",
    VOGEO_BITS_PER_FRAME,
)
print(
    "  bits                 =",
    VOGEO_USEFUL_BITS,
)
print(
    "  bytes                =",
    VOGEO_BYTES,
)
print()
print("SUPER PACKET")
print(
    "  bytes                =",
    SUPER_PACKET_BYTES,
)
print(
    "  RF bits              =",
    SUPER_PACKET_BITS,
)
print(
    "  GNU Radio soft output=",
    SUPER_PACKET_BITS,
    "float32",
)
print(
    "  RX UDP bytes         =",
    SUPER_PACKET_BITS * 4,
)
print(
    "  direct header        =",
    HEADER_SIZE,
    "bytes",
)
print(
    "============================================================"
)


# ============================================================
# MAIN
# ============================================================

try:
    for (
        audio_id,
        audio_path,
    ) in enumerate(
        wav_files
    ):
        (
            audio,
            total_chunks,
        ) = load_audio(
            audio_path
        )

        original_path = os.path.join(
            ORIGINAL_DIR,
            (
                f"original_audio_"
                f"{audio_id}.wav"
            ),
        )

        sf.write(
            original_path,
            audio.detach()
            .cpu()
            .numpy()
            .T,
            model.sample_rate,
            subtype="PCM_16",
        )

        # SOFT VOGEO TX parity encoder has temporal context.
        vogeo.reset()

        print(
            f"\nAUDIO {audio_id} | "
            f"chunks={total_chunks} | "
            f"{audio_path}"
        )

        for chunk_id in range(
            total_chunks
        ):
            (
                chunk,
                valid_samples,
            ) = get_chunk(
                audio,
                chunk_id,
            )

            (
                payloads,
                encode_time,
            ) = encode_all_three(
                chunk
            )

            (
                super_packet,
                order,
            ) = build_super_packet(
                payloads,
                chunk_id,
            )

            header = make_header(
                audio_id,
                chunk_id,
                total_chunks,
                valid_samples,
            )

            # Header bypasses RF.
            header_sock.sendto(
                header,
                (
                    RX_HOST,
                    RX_PORT,
                ),
            )

            # ONE RF super-packet.
            rf_sock.sendto(
                super_packet,
                (
                    GRC_HOST,
                    GRC_PORT,
                ),
            )

            print(
                f"TX | "
                f"audio={audio_id} | "
                f"chunk="
                f"{chunk_id + 1}/"
                f"{total_chunks} | "
                f"order="
                f"{'->'.join(order)} | "
                f"{len(super_packet)} B | "
                f"{SUPER_PACKET_BITS} RF bits | "
                f"encode="
                f"{encode_time:.4f}s"
            )

            if (
                chunk_id
                < total_chunks - 1
            ):
                time.sleep(
                    CHUNK_INTERVAL_S
                )

        if (
            audio_id
            < len(wav_files) - 1
        ):
            time.sleep(
                AUDIO_INTERVAL_S
            )

except KeyboardInterrupt:
    print(
        "\nTX stopped"
    )

finally:
    rf_sock.close()
    header_sock.close()
