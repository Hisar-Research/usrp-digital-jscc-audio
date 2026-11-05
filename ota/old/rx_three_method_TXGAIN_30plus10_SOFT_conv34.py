import os
import sys
import math
import socket
import struct
import argparse
import json

import numpy as np
import soundfile as sf
import torch

from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi
from encodec import EncodecModel


# ============================================================
# SOFT VOGEO PACKAGE
# ============================================================

# Exact paths on your machine.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# reference_soft is inside the project directory
SOFT_REF_DIR = os.path.join(PROJECT_DIR, "reference_soft")

# The Python reference files are expected here.
required_py = (
    "vogeo_rx.py",
    "vogeo_llr.py",
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

# Import reference_soft as a package because vogeo_rx.py uses relative imports
# such as "from .vogeo_frame import ...".
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from reference_soft.vogeo_rx import VogeoRx


# ============================================================
# NETWORK
# ============================================================

HOST, PORT = (
    "127.0.0.1",
    5002,
)

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM,
)

sock.bind(
    (
        HOST,
        PORT,
    )
)

sock.settimeout(
    0.25
)


# ============================================================
# EXPERIMENT
# ============================================================

parser = argparse.ArgumentParser(
    description="Three-method equal-budget SOFT receiver for a USRP TX-gain sweep"
)
parser.add_argument(
    "--tx-gain",
    type=float,
    required=True,
    help=(
        "USRP transmitter gain used for this run, in dB. "
        "This value is only an experimental label; it is NOT treated as Es/N0. "
        "Example: --tx-gain 30"
    ),
)
parser.add_argument(
    "--vogeo-llr-clip",
    type=float,
    default=50.0,
    help=(
        "Symmetric clipping magnitude for hardware-estimated VOGEO LLRs. "
        "Default: 50"
    ),
)
args = parser.parse_args()
TX_GAIN_DB = float(args.tx_gain)
VOGEO_LLR_CLIP = float(args.vogeo_llr_clip)
if VOGEO_LLR_CLIP <= 0.0:
    raise ValueError("--vogeo-llr-clip must be positive")

# TX_GAIN_DB is an experimental control/label only. Never convert it to gamma.

# Internal method keys; CSV labels are proposed, conv, 3.
METHODS = (
    "vogeo",
    "conv",
    "plain",
)

DISPLAY_NAMES = {
    "vogeo":
        "PROPOSED SOFT VOGEO",
    "conv":
        "EnCodec 2.25 kbps + Conv(3/4) SOFT Viterbi",
    "plain":
        "EnCodec 3 kbps uncoded (hard BPSK decisions)",
}

# Exact labels written in the CSV "method" column.
CSV_METHOD_NAMES = {
    "vogeo": "proposed",
    "conv": "conv",
    "plain": "3",
}


# ============================================================
# ENCODEC + SOFT VOGEO
# ============================================================

# Locate the trained SOFT VOGEO bundle.
# Do NOT simply take the first directory named "bundle": your project contains
# both hard and soft bundles. We inspect JSON manifest files and only accept
# a bundle whose manifest declares decision_mode == "soft".

def _bundle_decision_mode(bundle_dir):
    """
    Return (mode, manifest_path).

    Searches JSON files directly inside the bundle and one level below.
    The VOGEO package records decision_mode in its manifest.
    """
    json_candidates = []

    try:
        for root, dirs, files in os.walk(bundle_dir):
            rel = os.path.relpath(root, bundle_dir)

            # Keep this bounded; manifests should be at/near the bundle root.
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 1:
                dirs[:] = []
                continue

            for name in files:
                if name.lower().endswith(".json"):
                    json_candidates.append(os.path.join(root, name))
    except OSError:
        return None, None

    # Prefer files that look like manifests.
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


BUNDLE_NAME = "bundle_3_out_4"
VOGEO_BUNDLE_DIR = os.path.join(SOFT_REF_DIR, BUNDLE_NAME)

if not os.path.isdir(VOGEO_BUNDLE_DIR):
    raise RuntimeError(
        f"New 30+10 VOGEO bundle not found: {VOGEO_BUNDLE_DIR}\n"
        "Place the trained soft bundle there or change BUNDLE_NAME."
    )

mode, manifest = _bundle_decision_mode(VOGEO_BUNDLE_DIR)
if mode is not None and mode != "soft":
    raise RuntimeError(
        f"Bundle decision_mode={mode!r}; expected 'soft'. "
        f"manifest={manifest}"
    )

print(f"[RX] SOFT reference code = {SOFT_REF_DIR}")
print(f"[RX] VOGEO bundle        = {VOGEO_BUNDLE_DIR}")

DEVICE = "cpu"

vogeo = VogeoRx(
    VOGEO_BUNDLE_DIR,
    device=DEVICE,
)

PLAIN_N_Q = 4
PROTECTED_N_Q = 3

PLAIN_BANDWIDTH_KBPS = 3.0

FRAMES_PER_CHUNK = 75
BITS_PER_CODE = 10

model = (
    EncodecModel
    .encodec_model_24khz()
)

model.eval()

if int(model.sample_rate) != 24000:
    raise RuntimeError(
        f"Expected 24-kHz EnCodec, "
        f"got {model.sample_rate}"
    )

CHUNK_SAMPLES = 24000

if int(vogeo.n_q) != PROTECTED_N_Q:
    raise RuntimeError(
        f"Soft VOGEO n_q="
        f"{vogeo.n_q}, "
        f"expected {PROTECTED_N_Q}"
    )

if int(vogeo.parity_dim) != 10:
    raise RuntimeError(
        f"Soft VOGEO parity_dim="
        f"{vogeo.parity_dim}, "
        f"expected 10"
    )


# ============================================================
# PLAIN
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
# RATE-3/4 PUNCTURED CONVOLUTIONAL DECODER
#
# Mother code: K=7, rate 1/2, generators (133,171) octal.
# Puncture pattern [1,1,1,0,0,1] keeps 4 of every 6 mother bits,
# giving effective rate 3/4. The encoder remains terminated by 6 zeros.
#
# 2250 source bits + 6 tail = 2256 trellis inputs
# -> 4512 mother bits -> 3008 punctured transmitted bits = 376 B.
# ============================================================

K = 7
N_STATES = 1 << (K - 1)
G_POLY = (0o133, 0o171)
TAIL_BITS = K - 1

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

    for state in range(
        N_STATES
    ):
        for bit in (0, 1):
            reg = (
                (bit << (K - 1))
                | state
            )

            for j, g in enumerate(
                G_POLY
            ):
                out[
                    state,
                    bit,
                    j,
                ] = _parity(
                    reg & g
                )

            nxt[
                state,
                bit,
            ] = (
                reg >> 1
            )

    return nxt, out


NEXT_STATE, OUT_BITS = (
    _build_trellis()
)


# ============================================================
# HARD VOGEO
# ============================================================

VOGEO_BITS_PER_FRAME = (
    int(vogeo.n_q)
    * BITS_PER_CODE
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
# SUPER-PACKET
# ============================================================

ORDERS = (
    ("plain", "conv", "vogeo"),
    ("conv", "vogeo", "plain"),
    ("vogeo", "plain", "conv"),
)

SEGMENT_PHYSICAL_BITS = {
    "plain":
        PLAIN_PHYSICAL_BITS,
    "conv":
        CONV_PHYSICAL_BITS,
    "vogeo":
        VOGEO_PHYSICAL_BITS,
}

SUPER_PACKET_BYTES = (
    PLAIN_BYTES
    + CONV_BYTES
    + VOGEO_BYTES
)                                                   # 1127

SUPER_PACKET_BITS = (
    SUPER_PACKET_BYTES * 8
)                                                   # 9016

# GNU Radio can KEEP sending one float32 soft metric/bit.
# Soft metrics are preserved for Conv and VOGEO; Plain uses a sign decision.
SOFT_DTYPE = np.dtype(
    "<f4"
)

SOFT_SUPER_PACKET_BYTES = (
    SUPER_PACKET_BITS
    * SOFT_DTYPE.itemsize
)                                                   # 36064


# ============================================================
# HEADER
# ============================================================

MAGIC = b"VQ"
TYPE_SUPER_SOFT = 5

HEADER_FORMAT = "!2sBIHHH"

HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)                                                   # 13


# ============================================================
# OUTPUT
# ============================================================

ORIGINAL_DIR = os.path.abspath(
    "original_audio"
)

# ------------------------------------------------------------
# Received-audio folder organization
#
# Example for --tx-gain 12:
#
# received_audio_superpacket_soft/
#   audio_gain_12/
#     proposed/
#     conv/
#     3/
# ------------------------------------------------------------
SAVE_ROOT_DIR = os.path.abspath(
    "received_audio_superpacket_soft"
)

# :g keeps integer gains clean ("12" instead of "12.0") while
# still supporting non-integer values such as "12.5".
TX_GAIN_FOLDER_LABEL = f"{TX_GAIN_DB:g}"

SAVE_DIR = os.path.join(
    SAVE_ROOT_DIR,
    f"audio_gain_{TX_GAIN_FOLDER_LABEL}",
)

for method in METHODS:
    method_folder = CSV_METHOD_NAMES[method]
    os.makedirs(
        os.path.join(
            SAVE_DIR,
            method_folder,
        ),
        exist_ok=True,
    )

PESQ_SAMPLE_RATE = 16000

PESQ_CSV = os.path.abspath(
    "pesq_results_superpacket_soft_txgain.csv"
)

ESTOI_CSV = os.path.abspath(
    "estoi_results_superpacket_soft_txgain.csv"
)

if not os.path.exists(
    PESQ_CSV
):
    with open(
        PESQ_CSV,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "method,audio_id,pesq,tx_gain_db\n"
        )

if not os.path.exists(
    ESTOI_CSV
):
    with open(
        ESTOI_CSV,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "method,audio_id,estoi,tx_gain_db\n"
        )


# ============================================================
# STATE
# ============================================================

states = {}
completed = set()

metric_scores = {
    method: {
        "pesq": [],
        "estoi": [],
    }
    for method in METHODS
}

pending_header = None

vogeo_audio_id = None
vogeo_expected_chunk = 0


# ============================================================
# HEADER
# ============================================================

def parse_header(raw):
    if len(raw) != HEADER_SIZE:
        return None, (
            f"header={len(raw)} B, "
            f"expected {HEADER_SIZE}"
        )

    try:
        (
            magic,
            packet_type,
            audio_id,
            chunk_id,
            total_chunks,
            valid_samples,
        ) = struct.unpack(
            HEADER_FORMAT,
            raw,
        )

    except struct.error as exc:
        return None, str(exc)

    if magic != MAGIC:
        return None, (
            f"wrong magic "
            f"{magic!r}"
        )

    if (
        packet_type
        != TYPE_SUPER_SOFT
    ):
        return None, (
            f"wrong packet type "
            f"{packet_type}; "
            f"expected "
            f"{TYPE_SUPER_SOFT}"
        )

    if total_chunks == 0:
        return None, (
            "total_chunks cannot "
            "be zero"
        )

    if (
        chunk_id
        >= total_chunks
    ):
        return None, (
            f"invalid chunk_id="
            f"{chunk_id}"
        )

    if not (
        1
        <= valid_samples
        <= CHUNK_SAMPLES
    ):
        return None, (
            f"invalid valid_samples="
            f"{valid_samples}"
        )

    return {
        "audio_id":
            int(audio_id),
        "chunk_id":
            int(chunk_id),
        "total_chunks":
            int(total_chunks),
        "valid_samples":
            int(valid_samples),
    }, None


# ============================================================
# SUPER-PACKET SPLIT
# ============================================================

def unpack_super_soft(
    payload,
    chunk_id,
):
    if (
        len(payload)
        != SOFT_SUPER_PACKET_BYTES
    ):
        raise ValueError(
            f"Soft super-packet="
            f"{len(payload)} B, "
            f"expected "
            f"{SOFT_SUPER_PACKET_BYTES} B"
        )

    soft = np.frombuffer(
        payload,
        dtype=SOFT_DTYPE,
    )

    if (
        soft.size
        != SUPER_PACKET_BITS
    ):
        raise ValueError(
            f"Soft count={soft.size}, "
            f"expected "
            f"{SUPER_PACKET_BITS}"
        )

    if not np.all(
        np.isfinite(soft)
    ):
        raise ValueError(
            "NaN/Inf in soft "
            "super-packet"
        )

    order = ORDERS[
        chunk_id
        % len(ORDERS)
    ]

    parts = {}
    pos = 0

    for method in order:
        n = SEGMENT_PHYSICAL_BITS[
            method
        ]

        parts[method] = (
            soft[
                pos:
                pos + n
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        pos += n

    if pos != SUPER_PACKET_BITS:
        raise RuntimeError(
            f"Split ended at "
            f"{pos}, expected "
            f"{SUPER_PACKET_BITS}"
        )

    return parts, order


def hard_slice(x):
    """
    GNU Radio convention used in your current flowgraph:
        negative -> bit 0
        positive -> bit 1

    ZERO is treated as bit 0.
    """
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    return (
        x > 0.0
    ).astype(
        np.uint8
    )


# ============================================================
# COMMON ENCODEC DECODER
# ============================================================

@torch.no_grad()
def decode_encodec_codes(
    codes_np,
    valid_samples,
    expected_n_q,
):
    codes_np = np.asarray(
        codes_np,
        dtype=np.int64,
    )

    expected = (
        int(expected_n_q),
        FRAMES_PER_CHUNK,
    )

    if codes_np.shape != expected:
        raise ValueError(
            f"EnCodec codes shape={codes_np.shape}, "
            f"expected {expected}"
        )

    if np.any(
        (codes_np < 0)
        | (codes_np > 1023)
    ):
        raise ValueError(
            "Decoded EnCodec index outside 0..1023"
        )

    codes = (
        torch
        .from_numpy(codes_np)
        .unsqueeze(0)
    )

    audio = model.decode(
        [
            (
                codes,
                None,
            )
        ]
    )

    audio = (
        audio[0]
        .detach()
        .cpu()
    )

    if valid_samples > audio.shape[-1]:
        raise ValueError(
            f"valid_samples={valid_samples} > "
            f"decoded={audio.shape[-1]}"
        )

    return audio[
        :,
        :valid_samples,
    ]


# ============================================================
# PLAIN HARD DECODER
# ============================================================

def decode_plain_hard(
    soft_physical,
    valid_samples,
):
    if (
        soft_physical.size
        != PLAIN_PHYSICAL_BITS
    ):
        raise ValueError(
            f"Plain physical="
            f"{soft_physical.size}, "
            f"expected "
            f"{PLAIN_PHYSICAL_BITS}"
        )

    # Hard slice and discard 4 byte-padding bits.
    bits = hard_slice(
        soft_physical[
            :PLAIN_INFO_BITS
        ]
    )

    b = bits.reshape(
        PLAIN_N_Q,
        FRAMES_PER_CHUNK,
        BITS_PER_CODE,
    )

    weights = (
        1
        << np.arange(
            BITS_PER_CODE - 1,
            -1,
            -1,
            dtype=np.uint16,
        )
    )

    codes = (
        b.astype(np.uint16)
        * weights[
            None,
            None,
            :,
        ]
    ).sum(
        axis=2,
        dtype=np.uint16,
    )

    return decode_encodec_codes(
        codes,
        valid_samples,
        PLAIN_N_Q,
    )


# ============================================================
# SOFT-DECISION VITERBI
# ============================================================

def viterbi_decode_soft(mother_soft):
    """
    Soft Viterbi on the reconstructed rate-1/2 mother-code metrics.

    Punctured positions are inserted as 0.0, i.e. neutral reliability,
    so they contribute no preference to either branch.
    """
    mother_soft = np.asarray(
        mother_soft,
        dtype=np.float32,
    ).reshape(-1)

    if mother_soft.size != CONV_MOTHER_BITS:
        raise ValueError(
            f"Conv mother soft values={mother_soft.size}, "
            f"expected {CONV_MOTHER_BITS}"
        )

    rx_soft = mother_soft.reshape(
        CONV_TRELLIS_INPUT_BITS,
        2,
    )

    expected_sym = (
        2.0 * OUT_BITS.astype(np.float32) - 1.0
    )

    INF = np.float64(1e300)
    metric = np.full(N_STATES, INF, dtype=np.float64)
    metric[0] = 0.0

    prev_state = np.zeros(
        (CONV_TRELLIS_INPUT_BITS, N_STATES),
        dtype=np.int16,
    )
    prev_bit = np.zeros(
        (CONV_TRELLIS_INPUT_BITS, N_STATES),
        dtype=np.uint8,
    )

    for t in range(CONV_TRELLIS_INPUT_BITS):
        branch = -(
            expected_sym * rx_soft[t][None, None, :]
        ).sum(axis=2)

        new_metric = np.full(
            N_STATES, INF, dtype=np.float64
        )

        for state in range(N_STATES):
            if not np.isfinite(metric[state]):
                continue

            for bit in (0, 1):
                ns = int(NEXT_STATE[state, bit])
                candidate = metric[state] + float(
                    branch[state, bit]
                )

                if candidate < new_metric[ns]:
                    new_metric[ns] = candidate
                    prev_state[t, ns] = state
                    prev_bit[t, ns] = bit

        m = new_metric.min()
        if np.isfinite(m):
            new_metric -= m
        metric = new_metric

    # Encoder is terminated to state zero.
    state = 0
    decoded = np.zeros(
        CONV_TRELLIS_INPUT_BITS, dtype=np.uint8
    )

    for t in range(CONV_TRELLIS_INPUT_BITS - 1, -1, -1):
        decoded[t] = prev_bit[t, state]
        state = int(prev_state[t, state])

    return decoded[:CONV_INFO_BITS]


def depuncture_soft(punctured_soft):
    punctured_soft = np.asarray(
        punctured_soft,
        dtype=np.float32,
    ).reshape(-1)

    if punctured_soft.size != CONV_USEFUL_BITS:
        raise ValueError(
            f"Punctured soft values={punctured_soft.size}, "
            f"expected {CONV_USEFUL_BITS}"
        )

    mother_soft = np.zeros(
        CONV_MOTHER_BITS,
        dtype=np.float32,
    )
    mother_soft[CONV_PUNCTURE_MASK] = punctured_soft
    return mother_soft


def decode_conv_soft(soft_physical, valid_samples):
    if soft_physical.size != CONV_PHYSICAL_BITS:
        raise ValueError(
            f"Conv physical={soft_physical.size}, "
            f"expected {CONV_PHYSICAL_BITS}"
        )

    punctured_soft = soft_physical[:CONV_USEFUL_BITS]
    mother_soft = depuncture_soft(punctured_soft)
    info_bits = viterbi_decode_soft(mother_soft)

    bits = info_bits.reshape(
        FRAMES_PER_CHUNK,
        PROTECTED_N_Q,
        BITS_PER_CODE,
    )

    weights = (
        1 << np.arange(
            BITS_PER_CODE - 1, -1, -1,
            dtype=np.uint16,
        )
    )

    codes = (
        bits.astype(np.uint16)
        * weights[None, None, :]
    ).sum(axis=2, dtype=np.uint16).T

    return decode_encodec_codes(
        codes, valid_samples, PROTECTED_N_Q
    )


# ============================================================
# SOFT VOGEO DECODER
# ============================================================

def estimate_vogeo_llr_from_hardware(raw_metric):
    """
    Blind payload-only LLR calibration for the UHD BPSK soft metrics.

    No preamble information and no TX-gain-to-SNR conversion are used.

    We fit the received payload metric with a two-component Gaussian mixture
    having a shared variance:

        m | b=0 ~ N(mu0, sigma^2)
        m | b=1 ~ N(mu1, sigma^2),   mu0 < mu1

    The two clouds are fitted by EM directly from the 3000 received VOGEO
    payload metrics.  We intentionally use equal bit priors (0.5 / 0.5), so
    source-bit imbalance does not become artificial channel reliability.

    After fitting, the soft decoder input is the Gaussian likelihood-ratio:

        L(m) = ((m-mu0)^2 - (m-mu1)^2) / (2*sigma^2)

    Positive LLR means bit 1; negative LLR means bit 0.

    This is gain/offset adaptive and, unlike the old |m|-MAD estimator, it
    explicitly estimates the two received BPSK clouds.  At a clean/high-gain
    packet the fitted variance becomes small and the LLR naturally approaches
    the +/- VOGEO_LLR_CLIP condition that worked in the zero-error test.  At a
    noisy/low-gain packet the clouds overlap more and the LLR magnitude falls.
    """
    x = np.asarray(raw_metric, dtype=np.float32)
    flat = x.reshape(-1).astype(np.float64)

    if flat.size == 0:
        raise ValueError("Empty VOGEO soft-metric array")
    if not np.all(np.isfinite(flat)):
        raise ValueError("NaN/Inf in VOGEO soft metrics")

    # ---------------- robust fitting copy ----------------
    # Ignore only extreme outliers while estimating the two clouds.  The final
    # LLR is still evaluated for every original payload sample.
    if flat.size >= 100:
        lo, hi = np.percentile(flat, [0.5, 99.5])
        fit = flat[(flat >= lo) & (flat <= hi)]
        if fit.size < max(100, flat.size // 2):
            fit = flat
    else:
        fit = flat

    # Robust initialization from the negative/positive halves when both exist.
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

    # If initialization collapses, create a symmetric pair around the median.
    sep = mu1 - mu0
    if (not np.isfinite(sep)) or sep <= 1e-9:
        center = float(np.median(fit))
        amp = float(np.median(np.abs(fit - center)))
        amp = max(amp, 1e-3)
        mu0 = center - amp
        mu1 = center + amp

    # Initial shared variance from distance to the closest initial cloud.
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

    # ---------------- equal-prior two-Gaussian EM ----------------
    # Fixed equal priors are deliberate: the model estimates the channel, not
    # the source probability of zeros and ones.
    for _ in range(30):
        old_mu0, old_mu1, old_sigma2 = mu0, mu1, sigma2

        logp0 = -0.5 * ((fit - mu0) ** 2) / sigma2
        logp1 = -0.5 * ((fit - mu1) ** 2) / sigma2

        # Stable two-class softmax responsibilities.
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

    sigma = math.sqrt(sigma2)
    separation = mu1 - mu0

    # Exact equal-variance Gaussian LLR for the fitted two BPSK clouds.
    llr_flat = (
        (flat - mu0) ** 2 - (flat - mu1) ** 2
    ) / (2.0 * sigma2)

    llr_flat = np.clip(
        llr_flat,
        -VOGEO_LLR_CLIP,
        VOGEO_LLR_CLIP,
    )
    llr = llr_flat.reshape(x.shape).astype(np.float32)

    # Diagnostics only.  These are NOT BER measurements.
    midpoint = 0.5 * (mu0 + mu1)
    scale = max(0.5 * abs(separation), 1e-9)
    near_boundary_fraction = float(
        np.mean(np.abs(flat - midpoint) < 0.25 * scale)
    )

    stats = {
        "mu0": float(mu0),
        "mu1": float(mu1),
        "sigma_metric": float(sigma),
        "separation": float(separation),
        "separation_over_sigma": float(separation / max(sigma, 1e-12)),
        "median_abs_raw": float(np.median(np.abs(flat))),
        "median_abs_llr": float(np.median(np.abs(llr_flat))),
        "mean_abs_llr": float(np.mean(np.abs(llr_flat))),
        "near_boundary_fraction": near_boundary_fraction,
    }
    return llr, stats

def decode_vogeo_soft(
    soft_physical,
    valid_samples,
    audio_id,
    chunk_id,
):
    """
    Decode VOGEO using soft information measured from the USRP receive path.

    This TX-gain experiment intentionally does NOT use
        gamma = 10**(TX_GAIN_DB/10)
    because USRP TX gain is not the received Es/N0.

    Instead, packet-wise reliability is estimated blindly from the received
    payload metrics with a two-Gaussian EM model and converted to an LLR for
    the learned VOGEO decoder.  No preamble information is used.
    """
    global vogeo_audio_id
    global vogeo_expected_chunk

    if soft_physical.size != VOGEO_PHYSICAL_BITS:
        raise ValueError(
            f"VOGEO physical={soft_physical.size}, "
            f"expected {VOGEO_PHYSICAL_BITS}"
        )

    if vogeo_audio_id != audio_id:
        vogeo.reset()
        vogeo_audio_id = audio_id
        vogeo_expected_chunk = 0

    if chunk_id != vogeo_expected_chunk:
        print(
            f"  SOFT VOGEO CONTEXT RESET | "
            f"expected chunk {vogeo_expected_chunk + 1}, "
            f"got {chunk_id + 1}"
        )
        vogeo.reset()

    raw_metric = (
        soft_physical[:VOGEO_USEFUL_BITS]
        .astype(np.float32, copy=True)
        .reshape(FRAMES_PER_CHUNK, VOGEO_BITS_PER_FRAME)
    )

    llr, llr_stats = estimate_vogeo_llr_from_hardware(raw_metric)

    print(
        "  VOGEO SOFT EM | "
        f"TX gain={TX_GAIN_DB:+.2f} dB | "
        f"metric|x|50={llr_stats['median_abs_raw']:.4g} | "
        f"mu0={llr_stats['mu0']:.4g} | "
        f"mu1={llr_stats['mu1']:.4g} | "
        f"sigma={llr_stats['sigma_metric']:.4g} | "
        f"sep/sigma={llr_stats['separation_over_sigma']:.3g} | "
        f"LLR|x|50={llr_stats['median_abs_llr']:.4g} | "
        f"near_boundary={100.0 * llr_stats['near_boundary_fraction']:.1f}%"
    )

    corrected_codes = vogeo.decode(llr)

    audio = decode_encodec_codes(
        corrected_codes,
        valid_samples,
        PROTECTED_N_Q,
    )

    vogeo_expected_chunk = chunk_id + 1
    return audio


# ============================================================
# AUDIO STATE
# ============================================================

def get_state(
    audio_id,
    total_chunks,
):
    if audio_id not in states:
        states[audio_id] = {
            "total_chunks":
                total_chunks,
            "last_valid_samples":
                None,
            "chunks": {
                method: {}
                for method
                in METHODS
            },
        }

    elif (
        states[audio_id][
            "total_chunks"
        ]
        != total_chunks
    ):
        raise ValueError(
            f"audio={audio_id}: "
            f"total_chunks changed"
        )

    return states[
        audio_id
    ]


def make_silence(n):
    return torch.zeros(
        (
            model.channels,
            n,
        ),
        dtype=torch.float32,
    )


# ============================================================
# METRICS
# ============================================================

def read_mono(path):
    x, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    mono = (
        x.mean(axis=1)
        .astype(np.float32)
    )

    return mono, int(sr)


def resample_np(
    x,
    old_sr,
    new_sr,
):
    if old_sr == new_sr:
        return np.asarray(
            x,
            dtype=np.float32,
        )

    g = math.gcd(
        int(old_sr),
        int(new_sr),
    )

    return resample_poly(
        x,
        int(new_sr) // g,
        int(old_sr) // g,
    ).astype(np.float32)


def calculate_metrics(
    method,
    audio_id,
    received_path,
):
    original_path = os.path.join(
        ORIGINAL_DIR,
        (
            f"original_audio_"
            f"{audio_id}.wav"
        ),
    )

    if not os.path.exists(
        original_path
    ):
        print(
            "    Metrics skipped: "
            "original missing"
        )
        return

    try:
        reference, ref_sr = (
            read_mono(
                original_path
            )
        )

        received, rec_sr = (
            read_mono(
                received_path
            )
        )

        if (
            not np.all(
                np.isfinite(reference)
            )
            or not np.all(
                np.isfinite(received)
            )
        ):
            print(
                "    Metrics skipped: "
                "NaN/Inf in audio"
            )
            return

        # ---------------- ESTOI ----------------
        rec_estoi = resample_np(
            received,
            rec_sr,
            ref_sr,
        )

        n = min(
            len(reference),
            len(rec_estoi),
        )

        if n > 0:
            try:
                estoi_score = float(
                    stoi(
                        reference[
                            :n
                        ].astype(
                            np.float64
                        ),
                        rec_estoi[
                            :n
                        ].astype(
                            np.float64
                        ),
                        ref_sr,
                        extended=True,
                    )
                )

                if np.isfinite(
                    estoi_score
                ):
                    with open(
                        ESTOI_CSV,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            f"{CSV_METHOD_NAMES[method]},"
                            f"{audio_id},"
                            f"{estoi_score:.6f},"
                            f"{TX_GAIN_DB:.2f}\n"
                        )

                    metric_scores[
                        method
                    ]["estoi"].append(
                        (
                            audio_id,
                            estoi_score,
                        )
                    )

                    print(
                        f"    ESTOI = "
                        f"{estoi_score:.6f}"
                    )

                else:
                    print(
                        "    ESTOI skipped: "
                        "non-finite result"
                    )

            except Exception as exc:
                print(
                    f"    ESTOI ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        # ---------------- PESQ ----------------
        ref_pesq = resample_np(
            reference,
            ref_sr,
            PESQ_SAMPLE_RATE,
        )

        rec_pesq = resample_np(
            received,
            rec_sr,
            PESQ_SAMPLE_RATE,
        )

        n = min(
            len(ref_pesq),
            len(rec_pesq),
        )

        if n > 0:
            try:
                pesq_score = float(
                    pesq(
                        PESQ_SAMPLE_RATE,
                        ref_pesq[:n],
                        rec_pesq[:n],
                        "wb",
                    )
                )

                if np.isfinite(
                    pesq_score
                ):
                    with open(
                        PESQ_CSV,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            f"{CSV_METHOD_NAMES[method]},"
                            f"{audio_id},"
                            f"{pesq_score:.6f},"
                            f"{TX_GAIN_DB:.2f}\n"
                        )

                    metric_scores[
                        method
                    ]["pesq"].append(
                        (
                            audio_id,
                            pesq_score,
                        )
                    )

                    print(
                        f"    PESQ  = "
                        f"{pesq_score:.6f}"
                    )

                else:
                    print(
                        "    PESQ skipped: "
                        "non-finite result"
                    )

            except Exception as exc:
                print(
                    f"    PESQ ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    except Exception as exc:
        print(
            f"    METRIC SETUP ERROR | "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


# ============================================================
# FINALIZE AUDIO
# ============================================================

def finalize_audio(
    audio_id,
    reason,
):
    if (
        audio_id in completed
        or audio_id
        not in states
    ):
        return

    state = states[
        audio_id
    ]

    total_chunks = state[
        "total_chunks"
    ]

    last_valid = (
        state[
            "last_valid_samples"
        ]
        or CHUNK_SAMPLES
    )

    print(
        f"\nFINALIZE AUDIO "
        f"{audio_id} | "
        f"{reason}"
    )

    for method in METHODS:
        chunks = []
        missing = []

        method_chunks = state[
            "chunks"
        ][method]

        for chunk_id in range(
            total_chunks
        ):
            if (
                chunk_id
                in method_chunks
            ):
                chunks.append(
                    method_chunks[
                        chunk_id
                    ]
                )

            else:
                missing.append(
                    chunk_id
                )

                n = (
                    last_valid
                    if (
                        chunk_id
                        == total_chunks - 1
                    )
                    else CHUNK_SAMPLES
                )

                chunks.append(
                    make_silence(n)
                )

        audio = torch.cat(
            chunks,
            dim=-1,
        )

        out_path = os.path.join(
            SAVE_DIR,
            CSV_METHOD_NAMES[method],
            (
                f"received_audio_"
                f"{audio_id}_"
                f"{CSV_METHOD_NAMES[method]}.wav"
            ),
        )

        sf.write(
            out_path,
            audio.detach()
            .cpu()
            .numpy()
            .T,
            model.sample_rate,
            subtype="PCM_16",
        )

        print(
            f"  {method.upper()} | "
            f"missing={missing} | "
            f"{out_path}"
        )

        calculate_metrics(
            method,
            audio_id,
            out_path,
        )

    completed.add(
        audio_id
    )

    del states[
        audio_id
    ]


# ============================================================
# MISSING PACKET
# ============================================================

def mark_missing_super(
    header,
    reason,
):
    audio_id = header[
        "audio_id"
    ]

    chunk_id = header[
        "chunk_id"
    ]

    total_chunks = header[
        "total_chunks"
    ]

    valid_samples = header[
        "valid_samples"
    ]

    if (
        audio_id
        in completed
    ):
        return

    state = get_state(
        audio_id,
        total_chunks,
    )

    print(
        f"MISSING SUPER PACKET | "
        f"audio={audio_id} | "
        f"chunk="
        f"{chunk_id + 1}/"
        f"{total_chunks} | "
        f"{reason}"
    )

    if (
        chunk_id
        == total_chunks - 1
    ):
        state[
            "last_valid_samples"
        ] = valid_samples

        finalize_audio(
            audio_id,
            "final super-packet missing",
        )


# ============================================================
# PROCESS ONE SUPER-PACKET
# ============================================================

def process_super_packet(
    header,
    payload,
):
    audio_id = header[
        "audio_id"
    ]

    chunk_id = header[
        "chunk_id"
    ]

    total_chunks = header[
        "total_chunks"
    ]

    valid_samples = header[
        "valid_samples"
    ]

    if (
        audio_id
        in completed
    ):
        return

    state = get_state(
        audio_id,
        total_chunks,
    )

    (
        parts,
        order,
    ) = unpack_super_soft(
        payload,
        chunk_id,
    )

    print(
        f"SUPER ACCEPTED | "
        f"audio={audio_id} | "
        f"chunk="
        f"{chunk_id + 1}/"
        f"{total_chunks} | "
        f"order="
        f"{'->'.join(order)}"
    )

    # ---------------- PLAIN (hard symbol decisions) ----------------
    try:
        audio = decode_plain_hard(
            parts["plain"],
            valid_samples,
        )

        state[
            "chunks"
        ]["plain"][
            chunk_id
        ] = audio

        print(
            f"  PLAIN 3KBPS OK | "
            f"{PLAIN_PHYSICAL_BITS} RX metrics "
            f"-> {PLAIN_INFO_BITS} hard bits"
        )

    except Exception as exc:
        print(
            f"  PLAIN 3KBPS ERROR | "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    # ---------------- CONV SOFT ----------------
    try:
        audio = decode_conv_soft(
            parts["conv"],
            valid_samples,
        )

        state[
            "chunks"
        ]["conv"][
            chunk_id
        ] = audio

        print(
            f"  CONV SOFT OK | "
            f"{CONV_PHYSICAL_BITS} RX metrics "
            f"-> {CONV_USEFUL_BITS} soft coded metrics "
            f"-> SOFT Viterbi "
            f"-> {CONV_INFO_BITS} info bits"
        )

    except Exception as exc:
        print(
            f"  CONV SOFT ERROR | "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    # ---------------- VOGEO SOFT ----------------
    try:
        audio = decode_vogeo_soft(
            parts["vogeo"],
            valid_samples,
            audio_id,
            chunk_id,
        )

        state[
            "chunks"
        ]["vogeo"][
            chunk_id
        ] = audio

        print(
            f"  VOGEO SOFT OK | "
            f"{VOGEO_PHYSICAL_BITS} RX metrics "
            f"-> {VOGEO_USEFUL_BITS} soft metrics "
            f"-> blind EM LLR -> soft learned decoder"
        )

    except Exception as exc:
        print(
            f"  VOGEO SOFT ERROR | "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    if (
        chunk_id
        == total_chunks - 1
    ):
        state[
            "last_valid_samples"
        ] = valid_samples

        finalize_audio(
            audio_id,
            "final soft super-packet received",
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary():
    print(
        "\n================ SOFT SUMMARY ================"
    )

    for method in METHODS:
        print(
            f"\n"
            f"{DISPLAY_NAMES[method]}"
        )

        for metric in (
            "estoi",
            "pesq",
        ):
            values = [
                score
                for (
                    _audio_id,
                    score,
                )
                in metric_scores[
                    method
                ][metric]
            ]

            if values:
                values = np.asarray(
                    values,
                    dtype=np.float64,
                )

                print(
                    f"  {metric.upper():5s} "
                    f"n={len(values)} "
                    f"avg={values.mean():.4f} "
                    f"min={values.min():.4f} "
                    f"max={values.max():.4f}"
                )

            else:
                print(
                    f"  {metric.upper():5s} "
                    f"n=0"
                )

    print(
        "\n=============================================="
    )


# ============================================================
# STARTUP
# ============================================================

print("============================================================")
print("THREE-METHOD TX-GAIN SOFT SUPER-PACKET RX | 30+10 + CONV 3/4")
print("============================================================")
print("Soft VOGEO folder       =", SOFT_REF_DIR)
print("Soft VOGEO run ID       =", vogeo.run_id)
print("Header packet type      =", TYPE_SUPER_SOFT)
print("USRP TX gain             =", TX_GAIN_DB, "dB")
print()
print("CSV method labels       = proposed, conv, 3")
print("Plain 3 kbps            = hard BPSK slice, 3000 bits")
print("Conv 2.25 + rate-3/4    = punctured SOFT Viterbi, 3008 physical bits")
print("Proposed                = SOFT VOGEO blind two-Gaussian EM LLR decoder, 3000 bits")
print()
print("SUPER packet            =", SUPER_PACKET_BYTES, "B /", SUPER_PACKET_BITS, "bits")
print("Expected GNU Radio UDP  =", SOFT_SUPER_PACKET_BYTES, "bytes")
print("PESQ CSV                =", PESQ_CSV)
print("ESTOI CSV               =", ESTOI_CSV)
print("Received audio folder   =", SAVE_DIR)
print("VOGEO EM LLR clip        =", VOGEO_LLR_CLIP)
print("============================================================")


# ============================================================
# MAIN RX LOOP
# ============================================================

try:
    while True:
        try:
            raw, _ = sock.recvfrom(
                65535
            )

        except socket.timeout:
            continue

        # ---------------- DIRECT HEADER ----------------
        if (
            len(raw)
            == HEADER_SIZE
        ):
            (
                header,
                error,
            ) = parse_header(
                raw
            )

            if (
                error
                is not None
            ):
                print(
                    "DROP HEADER |",
                    error,
                )
                continue

            if (
                pending_header
                is not None
            ):
                mark_missing_super(
                    pending_header,
                    (
                        "next direct header "
                        "arrived before RF payload"
                    ),
                )

            pending_header = (
                header
            )

            print(
                f"HEADER | "
                f"audio="
                f"{header['audio_id']} | "
                f"chunk="
                f"{header['chunk_id'] + 1}/"
                f"{header['total_chunks']} | "
                f"valid_samples="
                f"{header['valid_samples']}"
            )

            continue

        # ---------------- SUPER SOFT PDU -> HARD HERE ----------------
        if (
            len(raw)
            == SOFT_SUPER_PACKET_BYTES
        ):
            if (
                pending_header
                is None
            ):
                print(
                    f"ORPHAN SUPER PACKET | "
                    f"{len(raw)} B | "
                    f"no pending header"
                )
                continue

            header = (
                pending_header
            )

            pending_header = None

            try:
                process_super_packet(
                    header,
                    bytes(raw),
                )

            except Exception as exc:
                print(
                    f"SUPER PACKET ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                mark_missing_super(
                    header,
                    (
                        "hard super-packet "
                        "processing failed"
                    ),
                )

            continue

        print(
            f"DROP | UDP="
            f"{len(raw)} B | "
            f"expected "
            f"{HEADER_SIZE} B header "
            f"or "
            f"{SOFT_SUPER_PACKET_BYTES} B "
            f"super-packet"
        )

except KeyboardInterrupt:
    print(
        "\nRX stopped"
    )

    if (
        pending_header
        is not None
    ):
        mark_missing_super(
            pending_header,
            (
                "RX stopped before "
                "RF payload"
            ),
        )

        pending_header = None

    for audio_id in list(
        states.keys()
    ):
        finalize_audio(
            audio_id,
            "RX stopped",
        )

    print_summary()

finally:
    sock.close()
