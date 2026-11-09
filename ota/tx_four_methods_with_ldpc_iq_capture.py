import os
import sys
import glob
import math
import socket
import struct
import time
import json
import csv
import datetime

import numpy as np
import soundfile as sf
import torch

from encodec import EncodecModel
from encodec.utils import convert_audio


# ============================================================
# SHARED CODE FROM sim/
#
# The proposed method (M1) uses the simulation's transmitter and trained
# bundle, and M3.1 uses the simulation's Golay code, so OTA and simulation run
# the same methods.
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))        # ota/
REPO_DIR = os.path.dirname(PROJECT_DIR)
SIM_DIR = os.path.join(REPO_DIR, "sim")

for name in ("vogeo_tx.py", "vogeo_frame.py", os.path.join("bundle", "manifest.json")):
    if not os.path.isfile(os.path.join(SIM_DIR, name)):
        raise RuntimeError(f"Missing {os.path.join(SIM_DIR, name)}")

for path in (SIM_DIR, PROJECT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from vogeo_tx import VogeoTx
from golay_sbsd import GolaySBSD, BITS_PER_FRAME as GOLAY_BITS_PER_FRAME


# ============================================================
# NETWORK
# ============================================================

GRC_HOST, GRC_PORT = "127.0.0.1", 5000
RX_HOST, RX_PORT = "127.0.0.1", 5002

rf_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
header_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Expected RF packets for offline loss accounting / ID matching. This is a
# side-channel *transmit* log; it does not prove an RF detection.
TX_CAPTURE_DIR = os.path.join(PROJECT_DIR, "ota_captures")
os.makedirs(TX_CAPTURE_DIR, exist_ok=True)
TX_MANIFEST = os.path.join(
    TX_CAPTURE_DIR,
    f"tx_expected_packets_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.csv"
)
_tx_manifest_file = open(TX_MANIFEST, "w", newline="", encoding="utf-8", buffering=1)
_tx_manifest = csv.writer(_tx_manifest_file)
_tx_manifest.writerow(["audio_id", "chunk_id_zero_based", "total_chunks",
                       "valid_samples", "tx_unix_ns", "audio_source"])
print(f"TX expected-packet CSV: {TX_MANIFEST}")


# ============================================================
# DATASET
# ============================================================

def _find_librtts_test_clean():
    """Locate LibriTTS test-clean under datasets/ at the repository root
    (or, for older layouts, under ota/datasets/)."""
    preferred = [
        os.path.join(base, "datasets", *sub)
        for base in (REPO_DIR, PROJECT_DIR)
        for sub in (("LibriTTS", "test-clean"),
                    ("LibriTTS", "LibriTTS", "test-clean"),
                    ("test-clean",))
    ]

    def has_wav(path):
        if not os.path.isdir(path):
            return False
        return bool(glob.glob(os.path.join(path, "**", "*.wav"), recursive=True))

    for path in preferred:
        if has_wav(path):
            return os.path.abspath(path)

    datasets_root = os.path.join(REPO_DIR, "datasets")
    for base in (REPO_DIR, PROJECT_DIR):
        root_dir = os.path.join(base, "datasets")
        if not os.path.isdir(root_dir):
            continue
        for root, dirs, _files in os.walk(root_dir):
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

def _read_manifest(bundle_dir):
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_soft_bundle(bundle_dir):
    """The OTA model must be the soft 3-layer + 10-parity bundle."""
    m = _read_manifest(bundle_dir)
    if (str(m.get("decision_mode")).lower() != "soft"
            or int(m.get("n_q", -1)) != 3 or int(m.get("parity_dim", -1)) != 10):
        raise RuntimeError(
            f"{bundle_dir} is not a soft n_q=3, parity_dim=10 bundle: "
            f"decision_mode={m.get('decision_mode')}, n_q={m.get('n_q')}, "
            f"parity_dim={m.get('parity_dim')}"
        )
    return bundle_dir


VOGEO_BUNDLE_DIR = _check_soft_bundle(os.path.join(SIM_DIR, "bundle"))
BUNDLE_NAME = os.path.basename(VOGEO_BUNDLE_DIR)

print(f"[TX] Simulation code     = {SIM_DIR}")
print(f"[TX] VOGEO bundle        = {VOGEO_BUNDLE_DIR}")

DEVICE = "cpu"

vogeo = VogeoTx(
    VOGEO_BUNDLE_DIR,
    device=DEVICE,
)

# Equal-channel-budget experiment:
#   PLAIN baseline (M2)  : EnCodec 3.0 kbps = 4 RVQ codebooks = 3000 bits/s
#   GOLAY baseline (M3.1): EnCodec 2.25 kbps = 3 RVQ codebooks, (20,10,6) Golay
#                          code on layer 1 = 40 bits/frame = 3000 bits/s
#   LDPC baseline (M3.2) : EnCodec 2.25 kbps = 3 RVQ codebooks, then rate-3/4 LDPC
#   PROPOSED (M1)        : SOFT VOGEO = 3 EnCodec codebooks (30 bits/frame)
#                          + 10 learned bits/frame = 40 bits/frame
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
        f"This experiment expects the VOGEO/Golay/LDPC source to use 3 RVQ codebooks, "
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
# GOLAY + SBSD (M3.1), as in sim/selftest_sim_v3.py --blk_parity 10 0 0
#
# Per EnCodec frame, 40 bits, frame-major, MSB first:
#   [layer 1: 10 index bits + 10 Golay parity bits | layer 2: 10 | layer 3: 10]
# The receiver decodes it with softbit source decoding (golay_sbsd.py).
#
# 75 frames * 40 bits = 3000 bits. One zero byte is appended so the segment is
# 376 B, the size of the convolutional segment it replaces: the super-packet
# stays 1502 B and the GNU Radio flowgraph needs no change. The receiver
# ignores the 8 fill bits.
# ============================================================

golay = GolaySBSD()                                 # encoder only; no prior needed

PROTECTED_INFO_BITS = (
    PROTECTED_N_Q
    * FRAMES_PER_CHUNK
    * BITS_PER_CODE
)                                                   # 2250

GOLAY_USEFUL_BITS = (
    FRAMES_PER_CHUNK
    * GOLAY_BITS_PER_FRAME
)                                                   # 3000

GOLAY_FILL_BITS = 8
GOLAY_PHYSICAL_BITS = GOLAY_USEFUL_BITS + GOLAY_FILL_BITS   # 3008
GOLAY_BYTES = GOLAY_PHYSICAL_BITS // 8                      # 376


# ============================================================
# RATE-3/4 SYSTEMATIC LDPC
#
# Same 2250-bit, 3-codebook source representation as Golay + SBSD and the
# proposed method.
#
#   EnCodec source bits                  = 2250
#   + 6 known zero padding bits          = 2256 LDPC information bits
#   rate 3/4                             -> 3008 transmitted bits
#   payload                              = 376 bytes
#
# H = [A | B], where:
#   - A connects every information variable to 3 parity checks.
#   - Every parity check has exactly 9 A-edges.
#   - B is a dual-diagonal accumulator matrix.
#
# This is a deterministic custom LDPC matrix, NOT an IEEE/5G
# standardized LDPC matrix. The receiver must use the exact same
# construction below.
# ============================================================

LDPC_SOURCE_BITS = PROTECTED_INFO_BITS                # 2250
LDPC_ZERO_PAD_BITS = 6
LDPC_K = LDPC_SOURCE_BITS + LDPC_ZERO_PAD_BITS       # 2256
LDPC_N = LDPC_K * 4 // 3                              # 3008
LDPC_M = LDPC_N - LDPC_K                              # 752
LDPC_RATE = LDPC_K / LDPC_N                           # 0.75
LDPC_INFO_DEGREE = 3

if LDPC_K * 4 != LDPC_N * 3:
    raise RuntimeError(
        f"LDPC dimensions are not exactly rate 3/4: "
        f"k={LDPC_K}, n={LDPC_N}"
    )



def _build_ldpc_info_checks():
    """
    Return [LDPC_K, 3] check-node indices for the systematic LDPC code.

    The mapping is deterministic and regular:
      - each information variable has degree 3
      - each check receives exactly 9 information-variable edges
      - no information variable has duplicate check connections
    """
    j = np.arange(
        LDPC_K,
        dtype=np.int64,
    )

    group = j // LDPC_M
    r = j % LDPC_M

    c0 = r
    c1 = (
        87 * r
        + 103
        + 128 * group
    ) % LDPC_M
    c2 = (
        75 * r
        + 561
        + 448 * group
    ) % LDPC_M

    checks = np.stack(
        (c0, c1, c2),
        axis=1,
    ).astype(np.int32)

    if np.any(checks[:, 0] == checks[:, 1]) or \
       np.any(checks[:, 0] == checks[:, 2]) or \
       np.any(checks[:, 1] == checks[:, 2]):
        raise RuntimeError(
            "LDPC construction produced duplicate checks for an information bit"
        )

    check_degrees = np.bincount(
        checks.reshape(-1),
        minlength=LDPC_M,
    )

    if not np.all(check_degrees == 9):
        raise RuntimeError(
            "LDPC information-side check degrees are not all 9"
        )

    return checks


LDPC_INFO_CHECKS = _build_ldpc_info_checks()


def ldpc_encode(source_bits):
    """
    Systematic rate-3/4 LDPC encoder.

    Input:
        2250 source bits from the same 3-codebook EnCodec representation
        used by the Golay and proposed branches.

    Output:
        3008 coded bits:
            [2250 source bits | 6 known zeros | 752 parity bits]

    Parity equations use a dual-diagonal accumulator:
        A_i u XOR p_i XOR p_{i-1} = 0
    with p_{-1}=0.
    """
    source_bits = np.asarray(
        source_bits,
        dtype=np.uint8,
    ).reshape(-1)

    if source_bits.size != LDPC_SOURCE_BITS:
        raise ValueError(
            f"Expected {LDPC_SOURCE_BITS} LDPC source bits, "
            f"got {source_bits.size}"
        )

    if np.any(
        (source_bits != 0)
        & (source_bits != 1)
    ):
        raise ValueError(
            "LDPC source is not binary"
        )

    u = np.concatenate(
        (
            source_bits,
            np.zeros(
                LDPC_ZERO_PAD_BITS,
                dtype=np.uint8,
            ),
        )
    )

    if u.size != LDPC_K:
        raise RuntimeError(
            f"LDPC information vector has {u.size} bits, "
            f"expected {LDPC_K}"
        )

    # syndrome = A @ u (mod 2)
    syndrome = np.zeros(
        LDPC_M,
        dtype=np.uint8,
    )

    for edge in range(LDPC_INFO_DEGREE):
        np.bitwise_xor.at(
            syndrome,
            LDPC_INFO_CHECKS[:, edge],
            u,
        )

    # Solve B p = syndrome over GF(2) for the dual-diagonal B.
    parity = np.bitwise_xor.accumulate(
        syndrome
    ).astype(np.uint8)

    coded = np.concatenate(
        (
            u,
            parity,
        )
    )

    if coded.size != LDPC_N:
        raise RuntimeError(
            f"LDPC coded bits={coded.size}, expected {LDPC_N}"
        )

    # Internal parity-check verification.
    check = syndrome.copy()
    check ^= parity
    check[1:] ^= parity[:-1]

    if np.any(check):
        raise RuntimeError(
            "LDPC encoder parity-check verification failed"
        )

    return coded


LDPC_USEFUL_BITS = LDPC_N                            # 3008
LDPC_BYTES = math.ceil(
    LDPC_USEFUL_BITS / 8
)                                                    # 376
LDPC_PHYSICAL_BITS = LDPC_BYTES * 8                 # 3008
LDPC_PAD_BITS = LDPC_PHYSICAL_BITS - LDPC_USEFUL_BITS  # 0

if LDPC_BYTES != GOLAY_BYTES:
    raise RuntimeError(
        f"LDPC payload={LDPC_BYTES} B but Golay payload={GOLAY_BYTES} B"
    )


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
# golay + SBSD = 376 B (375 B + 1 fill byte)
# ldpc(3/4)    = 376 B
# soft VOGEO   = 375 B
#
# total = 1502 B = 12016 RF bits
# ============================================================

# Four cyclic rotations keep each method from always occupying the same
# position inside the RF super-packet.
ORDERS = (
    ("plain", "golay", "ldpc", "vogeo"),
    ("golay", "ldpc", "vogeo", "plain"),
    ("ldpc", "vogeo", "plain", "golay"),
    ("vogeo", "plain", "golay", "ldpc"),
)

SEGMENT_BYTES = {
    "plain": PLAIN_BYTES,
    "golay": GOLAY_BYTES,
    "ldpc": LDPC_BYTES,
    "vogeo": VOGEO_BYTES,
}

SEGMENT_PHYSICAL_BITS = {
    "plain": PLAIN_PHYSICAL_BITS,
    "golay": GOLAY_PHYSICAL_BITS,
    "ldpc": LDPC_PHYSICAL_BITS,
    "vogeo": VOGEO_PHYSICAL_BITS,
}

SUPER_PACKET_BYTES = (
    PLAIN_BYTES
    + GOLAY_BYTES
    + LDPC_BYTES
    + VOGEO_BYTES
)                                                   # 1502

SUPER_PACKET_BITS = (
    SUPER_PACKET_BYTES * 8
)                                                   # 12016


# ============================================================
# DIRECT METADATA HEADER
# ============================================================

MAGIC = b"VQ"
# 6 = Golay + SBSD super-packet with frame-major M2 bits. Type 5 was the
# earlier convolutional layout; the receiver rejects it.
TYPE_SUPER_SOFT = 6

HEADER_FORMAT = "!2sBIHHH"
HEADER_SIZE = struct.calcsize(
    HEADER_FORMAT
)                                                   # 13 bytes


# ============================================================
# TIMING / OUTPUT
# ============================================================

CHUNK_INTERVAL_S = 1
AUDIO_INTERVAL_S = 3

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
    [4,75] RVQ indices
        -> transpose [75,4]
        -> [75,4,10]
        -> 3000 bits.

    Frame-major and MSB first, as in the simulation's uncoded baseline (M2).
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
            codes_np.T[:, :, None]
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


def codes_to_protected_info_bits(codes_np):
    """
    LDPC source bits, frame-major:

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
            f"Protected codes shape={codes_np.shape}, "
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

    if info_bits.size != PROTECTED_INFO_BITS:
        raise RuntimeError(
            f"Protected information bits="
            f"{info_bits.size}, "
            f"expected {PROTECTED_INFO_BITS}"
        )

    return info_bits


def pack_golay(codes_np):
    """
    [3,75] RVQ indices -> Golay frames [75,40] -> 3000 bits + 8 fill bits
    -> 376 bytes.
    """
    codes_np = np.asarray(codes_np, dtype=np.int64)

    if codes_np.shape != (PROTECTED_N_Q, FRAMES_PER_CHUNK):
        raise ValueError(
            f"Golay codes shape={codes_np.shape}, "
            f"expected {(PROTECTED_N_Q, FRAMES_PER_CHUNK)}"
        )

    bits = golay.encode(codes_np).reshape(-1)

    if bits.size != GOLAY_USEFUL_BITS:
        raise RuntimeError(
            f"Golay bits={bits.size}, expected {GOLAY_USEFUL_BITS}"
        )

    bits = np.concatenate(
        (bits, np.zeros(GOLAY_FILL_BITS, dtype=np.uint8))
    )

    payload = np.packbits(
        bits,
        bitorder="big",
    ).tobytes()

    if len(payload) != GOLAY_BYTES:
        raise RuntimeError(
            f"Golay payload={len(payload)} B, expected {GOLAY_BYTES} B"
        )

    return payload


def pack_ldpc(codes_np):
    """
    LDPC uses frame-major 2250 source bits, then appends six known zeros
    internally and encodes them at rate 3/4.
    """
    info_bits = codes_to_protected_info_bits(
        codes_np
    )

    coded_bits = ldpc_encode(
        info_bits
    )

    payload = np.packbits(
        coded_bits,
        bitorder="big",
    ).tobytes()

    if len(payload) != LDPC_BYTES:
        raise RuntimeError(
            f"LDPC payload="
            f"{len(payload)} B, "
            f"expected {LDPC_BYTES} B"
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
# ONE ENCODEC PASS -> FOUR METHODS
# ============================================================

@torch.no_grad()
def encode_all_four(chunk):
    """
    ONE EnCodec encoder pass, then TWO quantizer operating points.

    PLAIN:
        z -> EnCodec quantizer at 3.0 kbps -> 4 codebooks -> 3000 bits.

    GOLAY + SBSD:
        z -> EnCodec quantizer at the VOGEO bundle bandwidth (2.25 kbps)
          -> 3 codebooks -> 30 bits/frame
          -> (20,10,6) Golay code on layer 1 -> 40 bits/frame
          -> 3000 bits + 8 fill bits = 3008 transmitted bits.

    LDPC:
        uses the SAME 2.25-kbps / 3-codebook source as Golay
          -> 2250 information bits
          -> 6 known zero padding bits
          -> systematic rate-3/4 LDPC
          -> 3008 transmitted coded bits.

    PROPOSED / SOFT VOGEO:
        uses the SAME original continuous latent z and the SAME 2.25-kbps
        3-codebook codes as Golay/LDPC, then adds 10 learned bits/frame
        -> 3000 transmitted bits.

    Channel budgets per 1-s chunk:
        plain=3000, golay=3000 (+8 fill), ldpc=3008, proposed=3000 bits.
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
    # used by Golay, LDPC, and SOFT VOGEO.
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

    golay_payload = pack_golay(
        codes_protected_np
    )

    ldpc_payload = pack_ldpc(
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
        "golay": golay_payload,
        "ldpc": ldpc_payload,
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
    "FOUR-METHOD TX-GAIN SOFT SUPER-PACKET TX | 30+10 + GOLAY+SBSD + LDPC 3/4"
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
    "Golay/LDPC/VOGEO BW    =",
    PROTECTED_BANDWIDTH_KBPS,
    "kbps",
)
print(
    "Golay/LDPC/VOGEO n_q   =",
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
print("GOLAY + SBSD: ENCODEC 2.25 KBPS + (20,10,6) GOLAY ON LAYER 1")
print(
    "  useful bits          =",
    GOLAY_USEFUL_BITS,
)
print(
    "  fill bits            =",
    GOLAY_FILL_BITS,
)
print(
    "  bytes                =",
    GOLAY_BYTES,
)
print()
print("LDPC: ENCODEC 2.25 KBPS + RATE-3/4 SYSTEMATIC")
print(
    "  source bits          =",
    LDPC_SOURCE_BITS,
)
print(
    "  zero padding bits    =",
    LDPC_ZERO_PAD_BITS,
)
print(
    "  LDPC k / n           =",
    f"{LDPC_K} / {LDPC_N}",
)
print(
    "  code rate            =",
    f"{LDPC_RATE:.6f}",
)
print(
    "  bytes                =",
    LDPC_BYTES,
)
print(
    "  physical bits        =",
    LDPC_PHYSICAL_BITS,
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
            ) = encode_all_four(
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
            # Record the *transmitted* packet ID. Use with RX detection CSV
            # after the run; no assumption is made that the RF packet arrived.
            _tx_manifest.writerow([audio_id, chunk_id, total_chunks,
                                   valid_samples, time.time_ns(), audio_path])

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
    _tx_manifest_file.close()
