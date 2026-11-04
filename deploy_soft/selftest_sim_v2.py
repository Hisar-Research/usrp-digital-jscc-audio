"""
SNR sweep for the deployed SOFT bundle, against THREE baselines, all over AWGN.

selftest_sim.py plus an LDPC baseline. Where selftest_v2.py runs the chain once
at a single operating point, this runs it across -5:1:8 dB and reports ESTOI at
every step. It drives the same deployment classes -- VogeoTx, VogeoRx,
vogeo_llr -- so what it measures is the bundle you would actually ship, not the
training-time model.

Four methods, ALL spending the same 40 bits/frame = 3.0 kbps:

  M1   Proposed   : the bundle. EnCodec's RVQ indices sent uncoded plus the
                    learned parity side channel, soft-decoded: 1024-way codeword
                    posterior, MMSE latent, log-posterior prior.
  M2   Uncoded    : EnCodec at 3.0 kbps (4 RVQ layers), no protection at all.
                    The pure 3 kbps benchmark.
  M3.1 Conv 3/4   : EnCodec at 30 bits/frame (3 RVQ layers) + the CCSDS
                    131.0-B-3 rate-1/2 K=7 code (C1 = 171, C2 = 133 inverted),
                    punctured to rate 3/4 per CCSDS Table 3-2: of every 3 pairs
                    send C1(1) C2(1) C2(2) C1(3). dfree 5, checked at startup.
                    Soft-decision streaming Viterbi.
  M3.2 LDPC 3/4   : EnCodec at 30 bits/frame (3 RVQ layers) + a (3008, 2256)
                    quasi-cyclic LDPC code, rate exactly 3/4, decoded with
                    soft NORMALIZED min-sum (Chen & Fossorier 2002).

The rate fixes the split: 3/4 of 40 bits/frame is 30 info bits, i.e. n_q = 3.
That is the same split the bundle uses, so M1, M3.1 and M3.2 are three ways of
spending one identical budget -- 30 bits of EnCodec plus 10 bits of protection
-- and the comparison is exactly "learned parity vs. the two standard codes".

THE LDPC CODE, AND WHAT IT IS NOT. It is NOT a DVB-RCS2 code: ETSI EN 301 545-2
has no LDPC at all -- its return-link FEC is a 16-state duo-binary turbo code
(Turbo-phi, s7.3.5.1) for linear modulation and a convolutional code for CPM.
LDPC in DVB is the forward link, DVB-S2/S2X, at n = 16200 / 64800, which at
3000 sym/s would mean 5.4 s / 21.6 s of decode latency -- unusable for voice.
So this is a DVB-S2-CLASS code -- same decoder family, same rate -- built at a
block length that fits a one-second latency budget. (3008, 2256) factors as
64*47 and 48*47, so it is a quasi-cyclic lift of a 16 x 64 base matrix with
circulant size Z = 47: (3, 12)-regular, girth >= 6 (4-cycles removed at the
base-matrix level), full rank so k = 2256 exactly. Encoding uses the dense
systematic form from a GF(2) row reduction of H; decoding uses the sparse H.
Do not confuse it with the H = [A | I] shortcut in evaluate_models_v3.py,
whose parity columns all have weight 1 -- that flatters M1.

Normalized min-sum scales every check-to-variable message by alpha = 0.8,
repairing min-sum's overestimate of the extrinsic magnitude. Measured on this
code at +1.5 dB: alpha in 0.7..0.9 decodes clean in ~12 iterations, plain
min-sum (alpha = 1) leaves BER 3.7e-3 after 22. The self-test at startup
re-checks girth, rank, and a noiseless round trip.

TWO FAIRNESS NOTES, both in M1's favour, both unavoidable:

 * M3.1 gets SOFT-decision Viterbi, not hard. Hard decisions would cost it
   roughly 2 dB and the comparison would be rigged. The branch metric
   -sum_j (2c_j - 1) * L_j is exact ML for the trellis, not an approximation:
   it differs from -sum_j log sigmoid((2c_j-1)L_j) only by a term that depends
   on |L| alone, which is common to every branch.
 * M2 has no decoder, so soft information has nowhere to go -- it slices and
   dequantizes. The nearest soft analogue, MMSE dequantization of the codeword
   posterior, is part of M1's receiver and is not a baseline.

Latency is NOT equal and the table says so. M1 is causal: one frame, 13.3 ms.
M3.1's truncated traceback adds tb_depth/(30*75) seconds on top. M3.2 must
receive a whole 3008-bit codeword before it can decode anything: 1003 ms at
3000 sym/s. That second is inherent to a block code of this length and is the
price of its extra coding gain -- report both numbers, not just the ESTOI.

Interleaving is off: AWGN has no burst structure to break up.

Common random numbers: the noise for a given (method, trial) is drawn ONCE and
reused at every SNR, scaled by sigma. Without that, each point draws fresh noise
and the curves wander non-monotonically for reasons that are pure Monte-Carlo.

Usage:
    python selftest_sim_v2.py                    # the bundle in ./bundle
    python selftest_sim_v2.py --trials 5 --seconds 8
    python selftest_sim_v2.py --true-llr         # skip the blind LLR estimator
    python selftest_sim_v2.py --ldpc_alpha 0.75 --ldpc_iters 100
    python selftest_sim_v2.py --test_dir test_data --seconds 0 --trials 1 \
                              --true-llr --workers 12     # the paper run
    python selftest_sim_v2.py --pesq_mode wb     # P.862.2 instead of P.862 NB
    python ../results_plot_v2.py --json vogeo_sim_soft_awgn_v2.json --out_dir figs

Writes vogeo_sim_soft_awgn_v2.json (the format results_plot*.py read) and
sim_estoi_awgn_v2.png.
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import time
import zlib

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encodec import EncodecModel
from encodec.utils import convert_audio
from pystoi import stoi as stoi_metric

try:
    from pesq import pesq as pesq_metric
except ImportError:                       # PESQ is a nice-to-have, ESTOI is not
    pesq_metric = None

from vogeo_tx import VogeoTx
from vogeo_rx import VogeoRx
from vogeo_llr import llr_from_samples
from vogeo_frame import BITS_PER_CODE, TOTAL_BITS_PER_FRAME

SNR_GRID_DB = list(range(-5, 9))          # -5:1:8
METRIC_SR   = 16000
# ITU-T P.862 narrowband PESQ (P.862.1 MOS-LQO). The pesq package runs it at
# 16 kHz too -- IRS receive filter, 300-3400 Hz -- so the input is the same as
# ESTOI's. "wb" (P.862.2) is still available through --pesq_mode.
PESQ_MODE   = "nb"
N_Q_UNCODED = 4                           # M2: 4 x 10 = 40 bits/frame
M3_CODE_RATE = "3/4"
TB_DEPTH    = 35                          # 5*K, the standard traceback rule
LDPC_N, LDPC_K = 3008, 2256               # rate 2256/3008 = 3/4 exactly
LDPC_Z         = 47                       # 3008 = 64*47, 2256 = 48*47
LDPC_COL_W     = 3                        # (3, 12)-regular
LDPC_ALPHA     = 0.8                      # normalized min-sum scale
LDPC_MAX_ITERS = 50
FRAME_RATE  = 75
BW_PER_LAYER_KBPS = math.log2(1024) * FRAME_RATE / 1000      # 0.75
SEED = 20260914


# =========================================================
# CCSDS 131.0-B-3 s3 CONVOLUTIONAL CODE: rate 1/2, K = 7, punctured to 3/4
# =========================================================
# Encoder (CCSDS Fig. 3-1). Leftmost tap = the current input bit.
#     G1 = 1111001 = 171 octal  ->  C1
#     G2 = 1011011 = 133 octal  ->  C2, INVERTED
# Every info bit emits the pair (C1, C2), in that order.
#
# Rate 3/4 (CCSDS Table 3-2, period 3), puncture matrix in (C1, C2) columns:
#     C1: 1 0 1
#     C2: 1 1 0        ->  transmitted  C1(1) C2(1) C2(2) C1(3)
# i.e. the 133 output survives at t1,t2 and the 171 output at t1,t3 -- the
# same code as IEEE 802.11a's rate 3/4 (A = 133: 110, B = 171: 101).
#
# The column ORDER is the whole point. The first version of this file had
# G = (133, 171) against this same table, which keeps 171 at t1,t2 instead:
# a different punctured code with dfree 4 instead of 5, 1.1-5x the BER at
# 1-4 dB. The self-test in _conv_selftest() now pins dfree for both rates.
#
# C2's inversion is a coset shift: it changes no distance and no AWGN error
# rate. It is kept because it is in the standard (it guarantees transitions in
# the channel stream), and the self-test checks the decoder handles it.
K_CONST = 7
N_STATES = 1 << (K_CONST - 1)
CCSDS_G = (0o171, 0o133)                  # (G1 -> C1, G2 -> C2)
CCSDS_INV = np.array([0, 1], dtype=np.uint8)
PUNCTURE_3_4 = np.array([[1, 1],          # t1: C1 C2
                         [0, 1],          # t2:    C2
                         [1, 0]],         # t3: C1
                        dtype=bool)
DFREE_EXPECTED = {"1/2": 10, "3/4": 5}    # Odenwalder; Yasuda/Cain et al.


def _taps(g):
    """Octal generator -> 0/1 tap vector, index i = delay i (0 = current bit)."""
    return np.array([(g >> (K_CONST - 1 - i)) & 1 for i in range(K_CONST)], dtype=np.uint8)


TAPS = np.stack([_taps(g) for g in CCSDS_G], axis=1)          # [7, 2]


def _build_trellis():
    """State s = the last 6 inputs, bit 5 = delay 1 ... bit 0 = delay 6.
    Returns NEXT_STATE[s, b], OUT_LIN[s, b, :] (linear part, no inversion) and
    PRED[ns, j] -- the two states that lead to ns. Input b into s gives
    ns = (b << 5) | (s >> 1), so ns's predecessors are ((ns & 31) << 1) | j
    and the input that reached ns is ns >> 5."""
    nxt = np.zeros((N_STATES, 2), dtype=np.int64)
    out = np.zeros((N_STATES, 2, 2), dtype=np.uint8)
    for s in range(N_STATES):
        for b in (0, 1):
            reg = np.array([b] + [(s >> (K_CONST - 2 - i)) & 1
                                  for i in range(K_CONST - 1)], dtype=np.uint8)
            out[s, b] = (reg @ TAPS) & 1
            nxt[s, b] = (b << (K_CONST - 2)) | (s >> 1)
    ns = np.arange(N_STATES)
    pred = np.stack([((ns & (N_STATES // 2 - 1)) << 1) | j for j in (0, 1)], axis=1)
    return nxt, out, pred


NEXT_STATE, OUT_LIN, PRED = _build_trellis()
OUT_BITS = OUT_LIN ^ CCSDS_INV                               # what is on the wire
# branch into ns from PRED[ns, j], as +/-1 symbols: [64, 2 (j), 2 (C1, C2)]
BRANCH_PM = (2.0 * OUT_BITS[PRED, (np.arange(N_STATES) >> (K_CONST - 2))[:, None]]
             - 1.0)


def conv_encode_pairs(info_bits):
    """1-D uint8 -> [N + 6, 2] (C1, C2) pairs, zero-tail terminated.

    Written straight from the shift-register picture, not from the trellis
    tables, so _conv_selftest() can check each against the other."""
    u = np.concatenate([np.zeros(K_CONST - 1, dtype=np.uint8),
                        np.asarray(info_bits, dtype=np.uint8),
                        np.zeros(K_CONST - 1, dtype=np.uint8)])
    reg = np.lib.stride_tricks.sliding_window_view(u, K_CONST)[:, ::-1]   # col i = delay i
    return (((reg.astype(np.int64) @ TAPS) & 1).astype(np.uint8)) ^ CCSDS_INV


def _mask(n_pairs):
    return np.tile(PUNCTURE_3_4, (n_pairs // len(PUNCTURE_3_4) + 1, 1))[:n_pairs]


def apply_rate(pairs, code_rate):
    """[N,2] rate-1/2 output -> the bit stream actually transmitted."""
    if code_rate == "1/2":
        return pairs.reshape(-1)
    if code_rate == "3/4":
        return pairs[_mask(len(pairs))]
    raise ValueError(f"unsupported code_rate {code_rate!r}")


def undo_rate_llr(llr_flat, n_pairs, code_rate):
    """Inverse of apply_rate, in the LLR domain, back into [n_pairs, 2].

    A punctured position gets LLR 0 -- literally no information, which is what
    puncturing means, and which contributes 0 to every branch metric regardless
    of the hypothesised bit. That is cleaner than the hard-decision version's
    0.5 erasure fill and needs no special case in the trellis loop."""
    if code_rate == "1/2":
        return llr_flat.reshape(n_pairs, 2)
    if code_rate == "3/4":
        out = np.zeros((n_pairs, 2), dtype=np.float64)
        out[_mask(n_pairs)] = llr_flat
        return out
    raise ValueError(f"unsupported code_rate {code_rate!r}")


def viterbi_decode_soft(llr_pairs, n_info, tb_depth=TB_DEPTH):
    """Sliding-window soft-decision Viterbi with truncated traceback.

    Information bit t is emitted once step t + tb_depth has been processed, so
    the decoder never waits for the end of the utterance. Branch metric is
    -sum_j (2c_j - 1) * L_j, exact ML for BPSK (see the module header); a
    punctured position has L = 0 and drops out on its own.

    Add-compare-select runs over the butterfly: every state ns has exactly two
    predecessors PRED[ns, 0/1], and the input that reached ns is ns >> 5, so a
    survivor is one bit (which predecessor) and the decoded bit comes free
    with the state. All ACS decisions are made first, then the tracebacks;
    traceback t only reads decisions up to step t, so the output is bit-for-bit
    what a streaming decoder emits with tb_depth steps of delay."""
    n_steps = n_info + K_CONST - 1
    rx = np.asarray(llr_pairs, dtype=np.float64).reshape(n_steps, 2)
    in_shift = K_CONST - 2

    pm = np.full(N_STATES, 1e18); pm[0] = 0.0          # known start: state 0
    dec = np.empty((n_steps, N_STATES), dtype=np.uint8)
    best = np.empty(n_steps, dtype=np.int64)
    for t in range(n_steps):
        cand = pm[PRED] - BRANCH_PM @ rx[t]            # [64, 2]
        d = cand[:, 1] < cand[:, 0]
        pm = np.where(d, cand[:, 1], cand[:, 0])
        pm -= pm.min()                                 # renormalise, no drift
        dec[t] = d
        best[t] = np.argmin(pm)

    dec_l, pred_l = dec.tolist(), PRED.tolist()       # plain lists: fast scalar reads
    out = np.zeros(n_info, dtype=np.uint8)
    for t in range(tb_depth, n_steps):                 # window full: emit step t - tb_depth
        if t - tb_depth >= n_info:
            break
        s = int(best[t])
        for k in range(t, t - tb_depth, -1):           # state after step t-tb_depth
            s = pred_l[s][dec_l[k][s]]
        out[t - tb_depth] = s >> in_shift

    s = 0                                              # flush the zero-terminated tail
    for t in range(n_steps - 1, max(n_steps - tb_depth, 0) - 1, -1):
        if t < n_info:
            out[t] = s >> in_shift
        s = pred_l[s][dec_l[t][s]]
    return out


def free_distance(mask):
    """dfree of the (punctured) code: the lightest path that leaves state 0
    and first remerges with it, over every puncture phase. Dijkstra over
    (state, phase); weights use the LINEAR outputs, since the inversion of C2
    is common to every codeword and cancels out of every distance."""
    import heapq
    P, best_w = len(mask), 10 ** 9
    for ph in range(P):
        s1 = int(NEXT_STATE[0, 1])
        heap = [(int((OUT_LIN[0, 1] & mask[ph]).sum()), s1, (ph + 1) % P)]
        seen = set()
        while heap:
            w, s, p = heapq.heappop(heap)
            if s == 0:
                best_w = min(best_w, w); break
            if (s, p) in seen:
                continue
            seen.add((s, p))
            for b in (0, 1):
                heapq.heappush(heap, (w + int((OUT_LIN[s, b] & mask[p]).sum()),
                                      int(NEXT_STATE[s, b]), (p + 1) % P))
    return best_w


def _conv_selftest(rng):
    """Encoder == trellis, dfree as published, noiseless round trip through
    puncturing, and a correction check. Raises on any mismatch."""
    u = rng.integers(0, 2, 500, dtype=np.uint8)
    pairs = conv_encode_pairs(u)
    s, walk = 0, []
    for b in np.concatenate([u, np.zeros(K_CONST - 1, dtype=np.uint8)]):
        walk.append(OUT_BITS[s, b]); s = NEXT_STATE[s, b]
    assert s == 0 and np.array_equal(pairs, np.array(walk)), "encoder != trellis"
    # all-zero info must give C1 = 0, C2 = 1 (the inverted output)
    assert np.array_equal(conv_encode_pairs(np.zeros(8, np.uint8))[0], [0, 1])
    df = {"1/2": free_distance(np.ones((1, 2), dtype=bool)),
          "3/4": free_distance(PUNCTURE_3_4)}
    assert df == DFREE_EXPECTED, f"dfree {df}, expected {DFREE_EXPECTED}"
    for rate in ("1/2", "3/4"):
        sym = 2.0 * apply_rate(pairs, rate) - 1.0
        llr = 8.0 * sym
        llr[::97] *= -1.0                  # a few isolated errors to correct
        dec = viterbi_decode_soft(undo_rate_llr(llr, len(pairs), rate), len(u))
        assert np.array_equal(dec, u), f"Viterbi round trip failed at rate {rate}"
    return df


# =========================================================
# (3008, 2256) QUASI-CYCLIC LDPC + NORMALIZED MIN-SUM
# =========================================================
def _base_graph(mb, nb, col_w, row_w, rng):
    """Regular bipartite base graph: column j -> col_w DISTINCT rows, every
    row ends at exactly row_w. Greedy with capacity, restart on a dead end."""
    for _ in range(1000):
        cap = np.full(mb, row_w); cols = []; ok = True
        for _j in range(nb):
            avail = np.nonzero(cap > 0)[0]
            if len(avail) < col_w:
                ok = False; break
            pick = rng.choice(avail, size=col_w, replace=False,
                              p=cap[avail] / cap[avail].sum())
            cap[pick] -= 1; cols.append(pick)
        if ok and (cap == 0).all():
            return cols
    raise RuntimeError("no simple regular base graph found")


def build_qc_ldpc(n, k, Z, col_w, rng):
    """Lift a regular base matrix by circulant permutations of size Z. Any
    4-cycle in the lifted graph shows up at the base level as a 2x2 block whose
    shifts satisfy s11 - s12 + s22 - s21 = 0 (mod Z); those are resampled."""
    nb, mb = n // Z, (n - k) // Z
    assert nb * Z == n and mb * Z == n - k, f"({n},{k}) not multiples of Z={Z}"
    row_w = col_w * nb // mb
    assert row_w * mb == col_w * nb, "regular degrees do not balance"
    cols = _base_graph(mb, nb, col_w, row_w, rng)
    base = -np.ones((mb, nb), dtype=np.int64)
    for j in range(nb):
        for i in cols[j]:
            base[i, j] = rng.integers(Z)

    def find_4cycle():
        for j1 in range(nb):
            r1 = base[:, j1] >= 0
            for j2 in range(j1 + 1, nb):
                rows = np.nonzero(r1 & (base[:, j2] >= 0))[0]
                for a in range(len(rows)):
                    for b in range(a + 1, len(rows)):
                        i1, i2 = rows[a], rows[b]
                        if (base[i1, j1] - base[i1, j2] + base[i2, j2] - base[i2, j1]) % Z == 0:
                            return i1, j1
        return None

    fixes = 0
    while (c := find_4cycle()) is not None:
        base[c] = rng.integers(Z); fixes += 1
        assert fixes < 100000, "could not clear 4-cycles"

    H = np.zeros((mb * Z, nb * Z), dtype=np.uint8)
    eye = np.eye(Z, dtype=np.uint8)
    for i in range(mb):
        for j in range(nb):
            if base[i, j] >= 0:
                H[i*Z:(i+1)*Z, j*Z:(j+1)*Z] = np.roll(eye, base[i, j], axis=1)
    return H, base, fixes


def gf2_rref(H):
    """Row-reduce over GF(2). Returns (reduced rows, pivot columns)."""
    R = H.copy(); m, n = R.shape; piv = []; r = 0
    for c in range(n):
        if r >= m:
            break
        nz = np.nonzero(R[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            R[[r, p]] = R[[p, r]]
        mask = R[:, c].astype(bool); mask[r] = False
        R[mask] ^= R[r]
        piv.append(c); r += 1
    return R[:r], piv


def build_ldpc(n, k, Z, col_w, seed):
    """Returns (check_vars, A, info). Decode on the sparse H[:, perm], encode
    with the dense A. Column order perm puts the k info positions first and the
    m pivot (parity) positions last, so codeword = [u | A @ u mod 2]. The seed
    is retried if H is rank-deficient, which would make k_true > k."""
    m = n - k
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt)
        H, base, fixes = build_qc_ldpc(n, k, Z, col_w, rng)
        R, piv = gf2_rref(H)
        if len(piv) == m:
            break
    else:
        raise RuntimeError(f"no full-rank ({n},{k}) code in 20 seeds")
    nonpiv = [c for c in range(n) if c not in set(piv)]
    perm = np.array(nonpiv + piv)
    A = R[:, nonpiv]                                     # [m, k] dense
    Hp = H[:, perm]
    rows = [np.nonzero(Hp[i])[0] for i in range(m)]
    assert len({len(r) for r in rows}) == 1, "rows not regular"
    check_vars = np.stack(rows)                          # [m, row_w]
    ov = Hp.astype(np.int32) @ Hp.T.astype(np.int32); np.fill_diagonal(ov, 0)
    info = {"n": n, "k": k, "Z": Z, "base_shape": list(base.shape),
            "col_w": col_w, "row_w": int(check_vars.shape[1]),
            "girth_ge_6": bool(ov.max() <= 1), "rank": len(piv),
            "cycle_fixes": fixes, "seed": seed + attempt,
            "base_crc32": int(zlib.crc32(base.tobytes()) & 0xFFFFFFFF)}
    return check_vars, A, info


def ldpc_encode(A, u):
    return np.concatenate([u, (A.astype(np.int32) @ u.astype(np.int32)) % 2]).astype(np.uint8)


def nms_decode(llr, check_vars, alpha=LDPC_ALPHA, max_iters=LDPC_MAX_ITERS):
    """Normalized min-sum, flooding schedule, early stop on a clean syndrome.

    Input convention is this file's: L > 0 means bit 1. The textbook algorithm
    takes positive = 0, so the input is negated on entry rather than adapting
    the check-node rule: the sign product runs over d-1 incoming messages and
    flips with the parity of d-1, so a global sign change is NOT harmless.

    Check-to-variable: alpha * (product of the other signs) * (min of the other
    magnitudes), the min via the two-smallest trick so every row is one
    vectorised pass. alpha < 1 corrects min-sum's overestimate of the extrinsic
    magnitude; alpha = 1 is plain min-sum."""
    lam = -np.asarray(llr, dtype=np.float64)
    m, dc = check_vars.shape
    n = len(lam)
    v2c = lam[check_vars]                                          # [m, dc]
    ar = np.arange(m)
    x = (lam < 0).astype(np.uint8)
    for it in range(1, max_iters + 1):
        s = np.where(v2c < 0, -1.0, 1.0)
        mag = np.abs(v2c)
        tot = np.prod(s, axis=1, keepdims=True)
        order = np.argsort(mag, axis=1)
        m1, m2 = mag[ar, order[:, 0]], mag[ar, order[:, 1]]
        mn = np.broadcast_to(m1[:, None], mag.shape).copy()
        mn[ar, order[:, 0]] = m2
        c2v = alpha * (tot * s) * mn
        tot_in = np.zeros(n); np.add.at(tot_in, check_vars.ravel(), c2v.ravel())
        L = lam + tot_in
        x = (L < 0).astype(np.uint8)
        if not np.bitwise_xor.reduce(x[check_vars], axis=1).any():
            return x, it, True
        v2c = L[check_vars] - c2v
    return x, max_iters, False


def ldpc_encode_stream(info_bits, A, k):
    n_blocks = int(math.ceil(len(info_bits) / k))
    pad = n_blocks * k - len(info_bits)
    padded = np.concatenate([info_bits, np.zeros(pad, dtype=np.uint8)])
    return (np.concatenate([ldpc_encode(A, padded[b * k:(b + 1) * k])
                            for b in range(n_blocks)]), n_blocks, pad)


def ldpc_decode_stream(llr_stream, check_vars, k, n, n_info, alpha, max_iters):
    out, its, fails = [], 0, 0
    n_blocks = len(llr_stream) // n
    for b in range(n_blocks):
        x, it, ok = nms_decode(llr_stream[b * n:(b + 1) * n], check_vars, alpha, max_iters)
        out.append(x[:k]); its += it; fails += (not ok)
    return np.concatenate(out)[:n_info], its / max(n_blocks, 1), fails


# =========================================================
# BITS  (frame-major, matching what pack_frames puts on the wire)
# =========================================================
SHIFTS = np.arange(BITS_PER_CODE - 1, -1, -1)


def codes_to_bitstream(codes):
    """[n_q, T] -> flat uint8, frame by frame, MSB first within each index."""
    b = ((codes[:, :, None] >> SHIFTS) & 1).astype(np.uint8)      # [n_q,T,10]
    return b.transpose(1, 0, 2).reshape(-1)                       # [T*n_q*10]


def bitstream_to_codes(bits, n_q, T):
    b = bits.reshape(T, n_q, BITS_PER_CODE).transpose(1, 0, 2)
    return (b.astype(np.int64) << SHIFTS).sum(axis=-1)            # [n_q,T]


# =========================================================
# CHANNEL / METRICS
# =========================================================
def snr_to_sigma(snr_db):
    """BPSK, unit average symbol energy, real AWGN: Es/N0 = snr_db."""
    return math.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))


def crn_noise(n, trial, method):
    """Common random numbers: one draw per (method, trial), reused at every SNR.
    crc32 rather than hash(), whose string hashing is salted per process."""
    seed = zlib.crc32(f"{SEED}|{trial}|{method}".encode()) & 0x7FFFFFFF
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def to_metric_rate(wav24k):
    w = torch.as_tensor(wav24k).squeeze().detach().cpu()
    return AF.resample(w, 24000, METRIC_SR).numpy().astype(np.float64)


def compute_metrics(ref16, deg16, pesq_mode=PESQ_MODE):
    n = min(len(ref16), len(deg16))
    ref, deg = ref16[:n], deg16[:n]
    out = {}
    try:
        out["estoi"] = float(stoi_metric(ref, deg, METRIC_SR, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    if pesq_metric is not None:
        try:
            out["pesq"] = float(pesq_metric(METRIC_SR, ref, deg, pesq_mode))
        except Exception:
            out["pesq"] = float("nan")     # PESQ throws on silent/degenerate input
    r = ref - ref.mean(); d = deg - deg.mean()
    tgt = (np.dot(d, r) / (np.dot(r, r) + 1e-9)) * r
    noi = d - tgt
    out["sisdr"] = float(10.0 * np.log10((np.dot(tgt, tgt) + 1e-9)
                                         / (np.dot(noi, noi) + 1e-9)))
    return out


def n_q_to_bandwidth(n_q):
    """set_target_bandwidth() only accepts EnCodec's official list, which rules
    out n_q of 1 or 3; quantizer() itself does not validate, so the bandwidth is
    passed directly. Epsilon guards floor() underflow."""
    return n_q * BW_PER_LAYER_KBPS + 1e-6


# =========================================================
# ONE FILE AT EVERY SNR -- the unit of work, one per worker process
# =========================================================
_CTX = {}


def _init_worker(cfg):
    """Everything a file needs, built once per process. With --workers N each
    process is pinned to one torch thread, so N workers occupy N cores."""
    if cfg["torch_threads"]:
        torch.set_num_threads(cfg["torch_threads"])
    tx, rx = VogeoTx(cfg["bundle"]), VogeoRx(cfg["bundle"])
    check_vars, A, _ = build_ldpc(LDPC_N, LDPC_K, LDPC_Z, LDPC_COL_W, SEED)
    ec = EncodecModel.encodec_model_24khz().eval()
    ec.set_target_bandwidth(1.5)                      # arbitrary; always overridden
    for p in ec.parameters():
        p.requires_grad = False
    _CTX.clear()
    _CTX.update(cfg, tx=tx, rx=rx, ec=ec, check_vars=check_vars, A=A)


def _run_file(job):
    """-> (fidx, {method: {snr: [row per trial]}}, {ceiling: [row]}, summary)"""
    fidx, fpath = job
    c = _CTX
    tx, rx, ec, pm = c["tx"], c["rx"], c["ec"], c["pesq_mode"]
    grid, m_keys, n_qs, scenario = c["grid"], c["m_keys"], c["n_qs"], c["scenario"]
    n_q1, n_q3 = c["n_q1"], c["n_q3"]
    acc = {m: {s: [] for s in grid} for m in m_keys}
    ceil_acc = {f"ceiling_nq{q}": [] for q in n_qs}

    data, sr = sf.read(fpath)
    w = torch.from_numpy(data).float()
    w = w.unsqueeze(0) if w.ndim == 1 else w.T
    wav = convert_audio(w, sr, 24000, 1)
    if c['seconds'] > 0:
        wav = wav[..., :int(c['seconds'] * 24000)]
    wav = wav[None]                                              # [1,1,N]
    ref16 = to_metric_rate(wav)

    # ---- everything that does not depend on SNR or trial, computed once ----
    with torch.no_grad():
        z = ec.encoder(wav)
        codes = {q: ec.quantizer(z, FRAME_RATE, n_q_to_bandwidth(q)).codes
                 for q in n_qs}                                  # [n_q,1,T]
        for q in n_qs:
            ceil = ec.decoder(ec.quantizer.decode(codes[q]))
            ceil_acc[f"ceiling_nq{q}"].append(
                compute_metrics(ref16, to_metric_rate(ceil), pm))
    T = codes[n_q1].shape[-1]
    c1 = codes[n_q1][:, 0].numpy()
    c2 = codes[N_Q_UNCODED][:, 0].numpy()
    c3 = codes[n_q3][:, 0].numpy()

    tx.reset()
    bits1 = tx.encode(z, c1).astype(np.float32)                  # [T,40]
    bits2 = codes_to_bitstream(c2).astype(np.float32)            # T*40
    info3 = codes_to_bitstream(c3)                               # T*30
    pairs3 = conv_encode_pairs(info3)
    coded3 = apply_rate(pairs3, M3_CODE_RATE).astype(np.float32)
    coded4, n_blk, pad4 = ldpc_encode_stream(info3, c['A'], LDPC_K)   # same info bits
    coded4 = coded4.astype(np.float32)
    summary = (f"\n{T} frames | M1 {bits1.size} bits | M2 {bits2.size} bits | "
               f"M3.1 {info3.size} info -> {coded3.size} coded "
               f"(+{100*(coded3.size/(T*TOTAL_BITS_PER_FRAME)-1):.2f}% from the "
               f"{K_CONST-1}-bit zero tail) | M3.2 {info3.size} info -> {coded4.size} "
               f"coded in {n_blk} block(s) (+{100*(coded4.size/(T*TOTAL_BITS_PER_FRAME)-1):.2f}% "
               f"from {pad4} pad bits in the last block)\n")

    tx_sym = {f"M1_{scenario}": 2.0 * bits1.reshape(-1) - 1.0,
              "M2_uncoded":     2.0 * bits2 - 1.0,
              f"M3.1_{scenario}": 2.0 * coded3 - 1.0,
              f"M3.2_{scenario}": 2.0 * coded4 - 1.0}
    noise = {m: {t: crn_noise(len(tx_sym[m]), t, f"{fidx}|{m}")
                 for t in range(c['trials'])} for m in m_keys}

    for snr in grid:
        sigma = snr_to_sigma(snr)
        for t in range(c['trials']):
            with torch.no_grad():
                # ---- M1: the deployed soft receiver ----
                m = f"M1_{scenario}"
                y = (tx_sym[m] + sigma * noise[m][t]).reshape(T, TOTAL_BITS_PER_FRAME)
                llr = (2.0 * y / sigma ** 2).astype(np.float32) if c['true_llr'] \
                    else llr_from_samples(y)
                rx.reset()
                pred1 = rx.decode(llr)                            # [n_q,T]
                out1 = ec.decoder(ec.quantizer.decode(
                    torch.as_tensor(pred1, dtype=torch.long)[:, None]))
                # what the link would have delivered with no model: slice
                # the RVQ columns of the frame at zero
                sliced = (llr[:, :n_q1 * BITS_PER_CODE].reshape(-1) > 0).astype(np.uint8)
                acc[m][snr].append({
                    **compute_metrics(ref16, to_metric_rate(out1), pm),
                    "code_acc": float((pred1 == c1).mean()),
                    "hard_acc": float((bitstream_to_codes(sliced, n_q1, T) == c1).mean())})

                # ---- M2: no protection, slice and dequantize ----
                m = "M2_uncoded"
                y = tx_sym[m] + sigma * noise[m][t]
                pred2 = bitstream_to_codes((y > 0).astype(np.uint8), N_Q_UNCODED, T)
                out2 = ec.decoder(ec.quantizer.decode(
                    torch.as_tensor(pred2, dtype=torch.long)[:, None]))
                a2 = float((pred2 == c2).mean())
                acc[m][snr].append({**compute_metrics(ref16, to_metric_rate(out2), pm),
                                    "code_acc": a2, "hard_acc": a2})

                # ---- M3.1: CCSDS rate-3/4, soft Viterbi ----
                m = f"M3.1_{scenario}"
                y = tx_sym[m] + sigma * noise[m][t]
                llr3 = 2.0 * y.astype(np.float64) / sigma ** 2
                dec3 = viterbi_decode_soft(
                    undo_rate_llr(llr3, len(pairs3), M3_CODE_RATE).reshape(-1),
                    len(info3), tb_depth=c['tb_depth'])
                pred3 = bitstream_to_codes(dec3, n_q3, T)
                out3 = ec.decoder(ec.quantizer.decode(
                    torch.as_tensor(pred3, dtype=torch.long)[:, None]))
                acc[m][snr].append({
                    **compute_metrics(ref16, to_metric_rate(out3), pm),
                    "code_acc": float((pred3 == c3).mean()),
                    "raw_ber": float(((y > 0) != (coded3 > 0)).mean()),
                    "post_ber": float((dec3 != info3).mean())})

                # ---- M3.2: QC-LDPC rate-3/4, normalized min-sum ----
                m = f"M3.2_{scenario}"
                y = tx_sym[m] + sigma * noise[m][t]
                llr4 = 2.0 * y.astype(np.float64) / sigma ** 2
                dec4, avg_it, n_fail = ldpc_decode_stream(
                    llr4, c['check_vars'], LDPC_K, LDPC_N, len(info3),
                    c['ldpc_alpha'], c['ldpc_iters'])
                pred4 = bitstream_to_codes(dec4, n_q3, T)
                out4 = ec.decoder(ec.quantizer.decode(
                    torch.as_tensor(pred4, dtype=torch.long)[:, None]))
                acc[m][snr].append({
                    **compute_metrics(ref16, to_metric_rate(out4), pm),
                    "code_acc": float((pred4 == c3).mean()),
                    "raw_ber": float(((y > 0) != (coded4 > 0)).mean()),
                    "post_ber": float((dec4 != info3).mean()),
                    "avg_bp_iters": float(avg_it),
                    "block_fail_rate": float(n_fail / max(n_blk, 1))})

    return fidx, acc, ceil_acc, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./bundle")
    ap.add_argument("--wav", default=None, help="single file; default: test2.wav nearby")
    ap.add_argument("--test_dir", default=None, help="average over every wav/flac here")
    ap.add_argument("--seconds", type=float, default=5.0, help="0 = whole file")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tb_depth", type=int, default=TB_DEPTH)
    ap.add_argument("--ldpc_alpha", type=float, default=LDPC_ALPHA,
                    help="normalized min-sum scale; 1.0 = plain min-sum")
    ap.add_argument("--ldpc_iters", type=int, default=LDPC_MAX_ITERS)
    ap.add_argument("--snrs", type=float, nargs="*", default=None,
                    help=f"override the sweep (default {SNR_GRID_DB[0]}:1:{SNR_GRID_DB[-1]})")
    ap.add_argument("--true-llr", dest="true_llr", action="store_true",
                    help="form L = 2y/sigma^2 from the known sigma instead of "
                         "estimating it blind, to separate the scheme from the "
                         "estimator")
    ap.add_argument("--pesq_mode", choices=("nb", "wb"), default=PESQ_MODE,
                    help="P.862 narrowband (default) or P.862.2 wideband")
    ap.add_argument("--workers", type=int, default=1,
                    help="files decoded in parallel, one torch thread each")
    ap.add_argument("--json", default="vogeo_sim_soft_awgn_v2.json")
    ap.add_argument("--plot", default="sim_estoi_awgn_v2.png")
    args = ap.parse_args()
    grid = [int(v) if float(v).is_integer() else float(v)
            for v in (args.snrs if args.snrs else SNR_GRID_DB)]

    # ---- files ----
    if args.test_dir:
        files = sorted(glob.glob(os.path.join(args.test_dir, "**", "*.wav"), recursive=True)
                       + glob.glob(os.path.join(args.test_dir, "**", "*.flac"), recursive=True))
    else:
        w = args.wav or next((p for p in ("./test2.wav", "../test2.wav",
                                          "../deploy/test2.wav") if os.path.exists(p)), None)
        files = [w] if w else []
    if not files:
        raise SystemExit("no audio found -- pass --wav or --test_dir")

    # ---- bundle ----
    tx, rx = VogeoTx(args.bundle), VogeoRx(args.bundle)
    assert tx.run_id == rx.run_id, "transmitter and receiver are from different runs"
    n_q1, par1 = tx.n_q, tx.parity_dim
    scenario = f"enc{n_q1 * BITS_PER_CODE}_par{par1}"

    # ---- M3.1: the requested code rate fixes the split ----
    num, den = (int(v) for v in M3_CODE_RATE.split("/"))
    m3_info_bits = TOTAL_BITS_PER_FRAME * num // den
    assert TOTAL_BITS_PER_FRAME * num % den == 0 and m3_info_bits % BITS_PER_CODE == 0, \
        f"rate {M3_CODE_RATE} of {TOTAL_BITS_PER_FRAME} bits is not a whole number of code words"
    n_q3 = m3_info_bits // BITS_PER_CODE

    # ---- M3.2: same split, same rate, block code ----
    assert LDPC_K / LDPC_N == num / den, \
        f"LDPC ({LDPC_N},{LDPC_K}) is not rate {M3_CODE_RATE}"
    t0 = time.time()
    check_vars, A_ldpc, ldpc_info = build_ldpc(LDPC_N, LDPC_K, LDPC_Z, LDPC_COL_W, SEED)
    assert ldpc_info["girth_ge_6"] and ldpc_info["rank"] == LDPC_N - LDPC_K
    rng0 = np.random.default_rng(SEED)
    u0 = rng0.integers(0, 2, LDPC_K, dtype=np.uint8)
    x0, it0, ok0 = nms_decode(np.where(ldpc_encode(A_ldpc, u0) == 1, 30.0, -30.0),
                              check_vars, args.ldpc_alpha, args.ldpc_iters)
    assert ok0 and np.array_equal(x0[:LDPC_K], u0), "LDPC noiseless round trip failed"
    ldpc_lat = 1000.0 * LDPC_N / (TOTAL_BITS_PER_FRAME * FRAME_RATE)   # whole block first

    t0 = time.time()
    dfree = _conv_selftest(np.random.default_rng(SEED))
    conv_test_s = time.time() - t0

    m1_lat = 1000.0 / FRAME_RATE
    m3_lat = 1000.0 * args.tb_depth / (n_q3 * BITS_PER_CODE * FRAME_RATE)
    print(f"bundle    {scenario} soft, run_id {tx.run_id}, trained on "
          f"{tx.m.get('trained_channel')} (val acc {tx.m.get('trained_val_acc', float('nan')):.4f})")
    print(f"M1        n_q={n_q1} + {par1} learned parity bits   latency {m1_lat:7.1f} ms (causal)")
    print(f"M2        n_q={N_Q_UNCODED}, unprotected            latency {m1_lat:7.1f} ms")
    print(f"M3.1      n_q={n_q3} + CCSDS K=7 rate {M3_CODE_RATE}   latency "
          f"{m1_lat + m3_lat:7.1f} ms (tb={args.tb_depth}) | G = (171, 133-bar), keep "
          f"C1(1) C2(1) C2(2) C1(3), dfree {dfree['3/4']} (mother {dfree['1/2']}) | "
          f"self-test OK in {conv_test_s:.1f}s")
    print(f"M3.2      n_q={n_q3} + QC-LDPC ({LDPC_N},{LDPC_K})  latency "
          f"{m1_lat + ldpc_lat:7.1f} ms (one codeword) | ({ldpc_info['col_w']},"
          f"{ldpc_info['row_w']})-regular, Z={LDPC_Z}, girth>=6, rank {ldpc_info['rank']}, "
          f"NMS alpha={args.ldpc_alpha} <= {args.ldpc_iters} it | built in "
          f"{time.time()-t0:.1f}s, self-test OK")
    print(f"channel   AWGN, {grid} dB | {args.trials} trial(s) | {len(files)} file(s) | "
          f"LLR {'true' if args.true_llr else 'blind-estimated'} | "
          f"PESQ {args.pesq_mode.upper()} | {args.workers} worker(s)")

    m_keys = [f"M1_{scenario}", "M2_uncoded", f"M3.1_{scenario}", f"M3.2_{scenario}"]
    n_qs = sorted({n_q1, n_q3, N_Q_UNCODED})
    acc = {m: {s: [] for s in grid} for m in m_keys}
    ceil_acc = {f"ceiling_nq{q}": [] for q in n_qs}
    t_start = time.time()

    cfg = {"bundle": args.bundle, "torch_threads": 1 if args.workers > 1 else 0,
           "seconds": args.seconds, "trials": args.trials, "true_llr": args.true_llr,
           "tb_depth": args.tb_depth, "ldpc_alpha": args.ldpc_alpha,
           "ldpc_iters": args.ldpc_iters, "pesq_mode": args.pesq_mode, "grid": grid,
           "m_keys": m_keys, "n_qs": n_qs, "scenario": scenario,
           "n_q1": n_q1, "n_q3": n_q3}
    jobs = list(enumerate(files))

    def collect(res, n_done):
        fidx, acc_f, ceil_f, summary = res
        for m in m_keys:
            for snr in grid:
                acc[m][snr].extend(acc_f[m][snr])
        for k in ceil_acc:
            ceil_acc[k].extend(ceil_f[k])
        if n_done == 1:
            print(summary)
        el = time.time() - t_start
        mid = min(grid, key=lambda v: abs(v - 2))
        print(f"  [{n_done:3d}/{len(files)}] {os.path.basename(files[fidx]):28s} ESTOI @ "
              f"{mid:+g} dB  " + "  ".join(f"{m.split('_')[0]} {acc_f[m][mid][-1]['estoi']:.3f}"
                                            for m in m_keys)
              + f"   [{el:6.0f} s, ETA {el / n_done * (len(files) - n_done):6.0f} s]",
              flush=True)

    if args.workers > 1:
        # spawn, not fork: forking a parent that has touched torch can deadlock
        # its OpenMP pool. Each worker loads the models once in _init_worker.
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                          initargs=(cfg,)) as pool:
            for n_done, res in enumerate(pool.imap_unordered(_run_file, jobs), 1):
                collect(res, n_done)
    else:
        _init_worker(cfg)
        for n_done, job in enumerate(jobs, 1):
            collect(_run_file(job), n_done)

    # ---- aggregate ----
    def mean_of(rows):
        keys = set().union(*(r.keys() for r in rows))
        return {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}

    results = {m: {str(s): mean_of(acc[m][s]) for s in grid} for m in m_keys}
    for k, rows in ceil_acc.items():
        results[k] = mean_of(rows)
    results["_meta"] = {
        "channel": "awgn", "decision_mode": "soft", "snr_grid": grid,
        "trials": args.trials, "n_test_files": len(files), "seconds": args.seconds,
        "interleaver": "off (AWGN has no bursts)",
        "pesq": {"nb": "ITU-T P.862 narrowband, P.862.1 MOS-LQO",
                 "wb": "ITU-T P.862.2 wideband MOS-LQO"}[args.pesq_mode],
        "metric_sr": METRIC_SR,
        "scenario": scenario, "run_id": tx.run_id,
        "m1": {"n_q": n_q1, "parity_dim": par1, "latency_ms": m1_lat,
               "llr": "true" if args.true_llr else "blind"},
        "m2": {"n_q": N_Q_UNCODED, "latency_ms": m1_lat},
        "m3_1": {"n_q": n_q3, "code_rate": M3_CODE_RATE, "K": K_CONST,
                 "standard": "CCSDS 131.0-B-3 s3 (= IEEE 802.11a rate 3/4)",
                 "poly_octal": ["0o171 (C1)", "0o133 (C2, inverted)"],
                 "puncture": "C1(1) C2(1) C2(2) C1(3)", "dfree": dfree["3/4"],
                 "decision": "soft",
                 "tb_depth": args.tb_depth, "latency_ms": m1_lat + m3_lat},
        "m3_2": {"n_q": n_q3, "code_rate": M3_CODE_RATE, "code": "QC-LDPC",
                 "standard": "none -- DVB-S2-class decoding at a voice-latency "
                             "block length; DVB-RCS2 has no LDPC (its FEC is a "
                             "duo-binary turbo code)",
                 "decoder": "normalized min-sum", "alpha": args.ldpc_alpha,
                 "max_iters": args.ldpc_iters, "latency_ms": m1_lat + ldpc_lat,
                 **ldpc_info},
    }
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- table ----
    print(f"\nESTOI, mean over {args.trials} trial(s) x {len(files)} file(s)")
    print(f"{'SNR':>6s} | {'M1 proposed':>11s} {'M2 uncoded':>11s} "
          f"{'M3.1 conv 3/4':>13s} {'M3.2 LDPC 3/4':>13s} | "
          f"{'M1-M2':>7s} {'M1-M3.1':>8s} {'M1-M3.2':>8s}")
    print("-" * 96)
    for s in grid:
        a, b, c, d = (results[m][str(s)]["estoi"] for m in m_keys)
        print(f"{s:+5d}  | {a:11.4f} {b:11.4f} {c:13.4f} {d:13.4f} | "
              f"{a-b:+7.4f} {a-c:+8.4f} {a-d:+8.4f}")
    print("-" * 96)
    print(f"  latency: M1 {m1_lat:.0f} ms | M2 {m1_lat:.0f} ms | M3.1 {m1_lat+m3_lat:.0f} ms "
          f"| M3.2 {m1_lat+ldpc_lat:.0f} ms")
    for q in n_qs:
        print(f"  ceiling n_q={q} (no channel): ESTOI "
              f"{results[f'ceiling_nq{q}']['estoi']:.4f}")

    # ---- plot ----
    styles = {m_keys[0]: ("Proposed (M1), soft", "o", "#1f77b4", "-"),
              m_keys[1]: ("Uncoded 3.0 kbps (M2)", "s", "#7f7f7f", "--"),
              m_keys[2]: (f"Conv {M3_CODE_RATE} CCSDS K=7 (M3.1), soft", "^", "#2ca02c", "-."),
              m_keys[3]: (f"LDPC ({LDPC_N},{LDPC_K}) NMS (M3.2), soft", "D", "#9467bd",
                          (0, (3, 1, 1, 1)))}
    plt.figure(figsize=(9, 5.5))
    for m in m_keys:
        lbl, mk, col, ls = styles[m]
        plt.plot(grid, [results[m][str(s)]["estoi"] for s in grid],
                 label=lbl, marker=mk, color=col, linestyle=ls, markersize=6)
    for q, c in zip(n_qs, ("#333333", "#777777", "#aaaaaa")):
        plt.axhline(results[f"ceiling_nq{q}"]["estoi"], color=c, linestyle=":",
                    linewidth=1.5, label=f"Ceiling, n_q={q} (no channel)")
    plt.xlabel("Es/N0 (dB)"); plt.ylabel("ESTOI")
    plt.title(f"AWGN, soft decision, 3.0 kbps -- {scenario}")
    plt.xticks(grid); plt.grid(alpha=0.4, linestyle="--")
    plt.legend(loc="best", fontsize=9); plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.json} and {args.plot}")
    print(f"paper figures: python ../results_plot_v2.py --json {args.json} --out_dir figs")


if __name__ == "__main__":
    main()
