"""
SNR sweep of the deployed bundle against three equal-rate baselines, over AWGN
and block-Rayleigh fading, with matched and mismatched noise-variance knowledge.

Every method spends 40 bits/frame = 3.0 kbps (M3.2: 3008 bits per 75 frames, 0.3%
more, as over the air), takes SOFT channel input (LLRs) and makes a HARD index
decision per frame, so the comparison is like-for-like with the proposed receiver:

  M1   Proposed       : the bundle. 30 RVQ index bits sent as is + 10 learned
                        parity bits; causal soft receiver, argmax output.
  M3.1 Hamming + SBSD : the same 30 index bits, each 10-bit index protected by
                        its own systematic block code INSIDE the frame (default
                        (14,10) shortened Hamming on layer 1 and (13,10) on
                        layers 2 and 3: 10 parity bits in total). Receiver:
                        classical softbit source decoding (Fingscheidt & Vary
                        2001) -- codeword likelihoods from the LLRs, combined
                        with a first-order Markov index prior and filtered
                        forward in time (causal, no added delay); the MAP index
                        of the filtered posterior is decoded. Hard output: this
                        is SBSD, not SBSD-S (no MMSE latent).
  M3.2 LDPC 3/4       : the (3008, 2256) IRA-LDPC code of the over-the-air
                        experiments, normalized min-sum. One codeword per 75
                        frames: 2250 source bits, 6 known zeros and 752 parity
                        bits, ~1 s of buffering.
  M2   Uncoded        : EnCodec at 3.0 kbps (n_q = 4), slicer.

The convolutional baseline of selftest_sim_v2.py is gone.

THE BLOCK CODES. A 10-bit index with 4 parity bits fits a shortened (15,11)
Hamming code: (14,10), dmin 3. With 3 parity bits no dmin-3 code of dimension
10 exists (a 3 x 13 parity-check matrix has only 7 distinct nonzero columns),
so the best (13,10) code has dmin 2; design_block_code() picks the one with the
fewest weight-2 codewords. The search is exhaustive for these sizes and
deterministic. --blk_parity 4 4 2 puts Hamming codes on layers 1 AND 2 instead.

THE MARKOV PRIOR. Estimated once from --prior_dir (EnCodec indices of speech
that must NOT overlap the test set -- the training folder is the natural
choice) and cached in --prior_cache. Per layer, transitions are smoothed toward
the unigram with a Dirichlet weight alpha, T(i,j) = (N(i,j) + alpha u(j)) /
(N(i) + alpha), and alpha is picked by held-out likelihood on every 10th file.
The first frame of each utterance uses the unigram, just as the M1 receiver
starts each utterance from a reset state.

CHANNELS (--channels):
  awgn           y = x + n.
  rayleigh       y = |h| x + n, |h| constant over one 13.33-ms frame (40
                 symbols) and correlated across frames: AR(1) complex Gaussian,
                 rho = exp(-1/coherence_frames), E|h|^2 = 1, so Es/N0 is the
                 AVERAGE SNR. Coherent detection; the LLR uses the true |h|.
  rayleigh_nocsi same fading, but the LLR uses E|h| = sqrt(pi)/2 (no CSI).
No interleaver anywhere: the per-frame schemes cannot use one without adding
delay, and the LDPC block already spans ~75 frames of fading.

SNR MISMATCH (--mismatch_db). The receiver forms its LLRs from
sigma_hat^2 = sigma^2 * 10^(-m/10): it believes Es/N0 is m dB higher than it
is. m > 0 gives over-confident LLRs, m < 0 under-confident ones. The slicer
(M2) is invariant to a common LLR scale, and so is min-sum (M3.2) except for
the fixed floor on its 6 known zeros, which has no practical effect; M1 and SBSD
are not, because both weigh the channel against prior knowledge. The mismatch
is applied identically to all methods.

COMMON RANDOM NUMBERS. Noise is drawn once per (file, trial, method) and
fading once per (file, trial); both are reused at every SNR, every mismatch
and on both channels, so every curve and every difference between scenarios
is paired.

OUTPUTS (--out_prefix P):
  P_rows.csv      one row per file x trial x scenario x method x SNR
  P_ceil.csv      error-free codec ceilings per file and n_q
  P_summary.json  means per scenario / method / SNR, plus all settings
  P_config.json   the run configuration (checked on --resume)
Rows are appended file by file, so an interrupted run continues with --resume.

Usage:
    python selftest_sim_v3.py --selftest                     # code-level checks only
    python selftest_sim_v3.py --test_dir test_data --seconds 0 --trials 1 \\
        --prior_dir train_data --workers 12                  # the full run
    python selftest_sim_v3.py ... --channels awgn rayleigh rayleigh_nocsi \\
        --mismatch_db 0 3 -3 --coherence_frames 6 --blk_parity 4 3 3
    python plot_sim_v3.py --prefix vogeo_sim_v3 --out_dir figs
    python selftest_sim_v3.py --test_dir test_data --seconds 0 --trials 1 \\
        --channels rayleigh --mismatch_db 0 --snrs 0 --max_files 5 \\
        --save_audio audio_rayleigh_0dB --out_prefix listen  # clips, same noise as the sweep

--blk_parity 10 0 0 puts all 10 parity bits on layer 1 with the optimal
(20,10,6) code (shortened/punctured extended Golay), layers 2-3 uncoded.
"""

import argparse
import csv
import glob
import itertools
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
from encodec import EncodecModel
from encodec.utils import convert_audio
from pystoi import stoi as stoi_metric

try:
    from pesq import pesq as pesq_metric
except ImportError:                       # PESQ is a nice-to-have, ESTOI is not
    pesq_metric = None

from vogeo_tx import VogeoTx
from vogeo_rx import VogeoRx
from vogeo_frame import BITS_PER_CODE, TOTAL_BITS_PER_FRAME

SNR_GRID_DB = list(range(-5, 9))          # -5:1:8
METRIC_SR   = 16000
PESQ_MODE   = "nb"                        # as in selftest_sim_v2.py; --pesq_mode wb for P.862.2
N_Q_UNCODED = 4                           # M2: 4 x 10 = 40 bits/frame
N_Q_CODED   = 3                           # M1, M3.1, M3.2: 30 index bits + 10 protection
CODEBOOK    = 1 << BITS_PER_CODE          # 1024
LDPC_N, LDPC_K = 3008, 2256               # rate 2256/3008 = 3/4 exactly
LDPC_SOURCE_BITS = 2250                   # 75 frames x 3 layers x 10 bits
LDPC_PAD_BITS    = LDPC_K - LDPC_SOURCE_BITS  # 6 known zeros
LDPC_CHECK_INFO_DEG = 9                   # information bits per check
LDPC_ALPHA     = 0.8                      # normalized min-sum scale
LDPC_MAX_ITERS = 40
LDPC_KNOWN_ZERO_LLR = 50.0                # floor on the known zeros
FRAME_RATE  = 75
BW_PER_LAYER_KBPS = math.log2(CODEBOOK) * FRAME_RATE / 1000      # 0.75
SEED = 20260914

METHODS  = ("M1", "M3.1", "M3.2", "M2")
CHANNELS = ("awgn", "rayleigh", "rayleigh_nocsi")
H_MEAN   = math.sqrt(math.pi) / 2.0       # E|h| for unit-power Rayleigh
PRIOR_ALPHA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)

ROW_FIELDS = ["scenario", "channel", "mismatch_db", "method", "snr", "file", "trial",
              "estoi", "pesq", "sisdr", "code_acc", "acc_l1", "acc_l2", "acc_l3",
              "raw_ber", "hard_acc", "ml_acc", "block_fail_rate"]
CEIL_FIELDS = ["file", "n_q", "estoi", "pesq", "sisdr"]


def scenario_id(channel, mismatch_db):
    return f"{channel}_mis{mismatch_db:+g}dB"


# =========================================================
# BITS  (frame-major, MSB first within each index)
# =========================================================
SHIFTS = np.arange(BITS_PER_CODE - 1, -1, -1)
MSG_BITS = ((np.arange(CODEBOOK)[:, None] >> SHIFTS) & 1).astype(np.uint8)   # [1024, 10]


def codes_to_bitstream(codes):
    """[n_q, T] -> flat uint8, frame by frame, MSB first within each index."""
    b = ((codes[:, :, None] >> SHIFTS) & 1).astype(np.uint8)      # [n_q,T,10]
    return b.transpose(1, 0, 2).reshape(-1)                       # [T*n_q*10]


def bitstream_to_codes(bits, n_q, T):
    b = np.asarray(bits).reshape(T, n_q, BITS_PER_CODE).transpose(1, 0, 2)
    return (b.astype(np.int64) << SHIFTS).sum(axis=-1)            # [n_q,T]


# =========================================================
# M3.1: PER-INDEX SYSTEMATIC BLOCK CODES
# =========================================================
def _weight_spectrum(A):
    """(A_1, ..., A_n) of the systematic code [u | u A mod 2], zero word excluded."""
    par = (MSG_BITS.astype(np.int64) @ A.astype(np.int64)) & 1
    w = MSG_BITS.astype(np.int64).sum(axis=1) + par.sum(axis=1)
    return np.bincount(w[1:], minlength=A.shape[0] + A.shape[1] + 1)[1:]


def golay_20_10():
    """(20,10,6) code from the extended Golay (24,12,8): shorten 2 information
    positions -> (22,10,8), then puncture the 2 parity positions that leave the
    best weight spectrum -> dmin 6, the best possible for (20,10). A random
    search over (20,10) codes only reaches dmin 5."""
    row = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=np.uint8)
    B = np.zeros((12, 12), dtype=np.uint8)
    B[0, 1:] = 1
    B[1:, 0] = 1
    for i in range(11):
        B[1 + i, 1:] = np.roll(row, i)
    Bs = B[2:]                                            # shortened: info bits 0, 1 fixed to 0
    best_A, best_key = None, None
    for drop in itertools.combinations(range(12), 2):
        A = np.delete(Bs, drop, axis=1)
        key = tuple(int(v) for v in _weight_spectrum(A))
        if best_key is None or key < best_key:
            best_A, best_key = A, key
    return best_A, best_key


def design_block_code(r, k=BITS_PER_CODE, max_eval=20000, seed=SEED):
    """Systematic (k+r, k) code, codeword [u | u A mod 2], A in {0,1}^{k x r}.

    The rows of A are the parity-check columns of the information bits. A zero
    row or a row equal to a unit vector makes a codeword of weight <= 2, so
    rows of weight >= 2 are preferred: k DISTINCT such rows give dmin = 3 (a
    shortened Hamming code). When fewer than k exist (r = 3: only 4), rows may
    repeat and dmin = 2. Among the candidates, the weight spectrum is
    minimised lexicographically: largest dmin first, then fewest
    minimum-weight codewords, and so on. Exhaustive when the candidate set is
    small (all cases with r <= 4), otherwise a seeded random sample."""
    if r == 0:
        return np.zeros((k, 0), dtype=np.uint8), ()
    if r == 10 and k == 10:
        return golay_20_10()
    vecs = list(range(1, 1 << r))
    heavy = [v for v in vecs if bin(v).count("1") >= 2]
    distinct = len(heavy) >= k
    pool = heavy if distinct else vecs
    n_cand = math.comb(len(pool), k) if distinct else math.comb(len(pool) + k - 1, k)
    if n_cand <= max_eval:
        cands = (itertools.combinations(pool, k) if distinct
                 else itertools.combinations_with_replacement(pool, k))
    else:
        rng = np.random.default_rng(seed + r)
        cands = (tuple(sorted(rng.choice(pool, size=k, replace=not distinct)))
                 for _ in range(max_eval))
    col_shift = np.arange(r - 1, -1, -1)
    best_A, best_key = None, None
    for rows in cands:
        A = ((np.asarray(rows, dtype=np.int64)[:, None] >> col_shift) & 1).astype(np.uint8)
        key = tuple(int(v) for v in _weight_spectrum(A))
        if best_key is None or key < best_key:
            best_A, best_key = A, key
    return best_A, best_key


class BlockCodes:
    """One systematic code per RVQ layer, codewords back to back in the frame:
    [layer 1: 10 + r1 | layer 2: 10 + r2 | layer 3: 10 + r3]."""

    def __init__(self, parity):
        self.layers = []
        off = 0
        for r in parity:
            A, spec = design_block_code(r)
            n = BITS_PER_CODE + r
            cw = np.concatenate([MSG_BITS, (MSG_BITS.astype(np.int64) @ A) & 1],
                                axis=1).astype(np.uint8) if r else MSG_BITS.copy()
            dmin = next((i + 1 for i, a in enumerate(spec) if a), 1) if r else 1
            a_dmin = int(spec[dmin - 1]) if r else BITS_PER_CODE
            self.layers.append({"n": n, "k": BITS_PER_CODE, "r": r, "off": off, "A": A,
                                "cw": cw, "x": (2.0 * cw - 1.0).astype(np.float32),
                                "dmin": dmin, "a_dmin": a_dmin,
                                "hamming": r > 0 and dmin == 3,
                                "golay": r == 10 and dmin == 6})
            off += n
        self.n_bits = off

    def describe(self):
        return [f"({L['n']},{L['k']}) dmin {L['dmin']} with {L['a_dmin']} min-weight "
                f"codewords{' [shortened Hamming]' if L['hamming'] else ''}"
                f"{' [shortened/punctured Golay]' if L['golay'] else ''}"
                for L in self.layers]

    def encode(self, codes):
        """[n_q, T] indices -> [T, n_bits] uint8."""
        return np.concatenate([L["cw"][codes[q]] for q, L in enumerate(self.layers)], axis=1)

    def loglik(self, llr, q):
        """llr [..., n_bits] -> channel log-likelihood of every codeword of
        layer q, [..., 1024], up to a constant: sum_l c_l L_l = 0.5 sum_l
        (2c_l - 1) L_l + const (L > 0 means bit 1)."""
        L = self.layers[q]
        seg = llr[..., L["off"]:L["off"] + L["n"]].astype(np.float32)
        return 0.5 * (seg @ L["x"].T)


def sbsd_filter(ll, trans, log_u):
    """Causal softbit source decoding of one layer, batched over SNRs.

    ll    [S, T, K] channel log-likelihoods of every index
    trans [K, K]    T(i, j) = P(i_t = j | i_{t-1} = i), rows sum to 1
    log_u [K]       log unigram, the prior of the first frame
    Forward recursion a_t ∝ p(y_t | j) * sum_i a_{t-1}(i) T(i, j): the posterior
    of frame t given y_1..y_t only, so no delay is added. Returns the MAP index
    of every frame, [S, T] -- a hard decision, like M1's argmax."""
    S, n_t, _ = ll.shape
    out = np.empty((S, n_t), dtype=np.int64)
    a = None
    for t in range(n_t):
        lp = log_u if a is None else np.log(a @ trans + 1e-30)
        l = ll[:, t] + lp
        l -= l.max(axis=1, keepdims=True)
        out[:, t] = l.argmax(axis=1)
        a = np.exp(l)
        a /= a.sum(axis=1, keepdims=True)
    return out


# =========================================================
# MARKOV INDEX PRIOR (estimated on data disjoint from the test set)
# =========================================================
def transition_matrix(N, u, alpha):
    """Dirichlet smoothing toward the unigram u: (N(i,j) + alpha u(j)) / (N(i) + alpha)."""
    return (N + alpha * u[None, :]) / (N.sum(axis=1, keepdims=True) + alpha)


def bigram_counts(code_list, n_q):
    N = np.zeros((n_q, CODEBOOK, CODEBOOK), dtype=np.int64)
    U = np.zeros((n_q, CODEBOOK), dtype=np.int64)
    for c in code_list:
        c = c.astype(np.int64)
        for q in range(n_q):
            U[q] += np.bincount(c[q], minlength=CODEBOOK)
            if c.shape[1] > 1:
                N[q] += np.bincount(c[q, :-1] * CODEBOOK + c[q, 1:],
                                    minlength=CODEBOOK * CODEBOOK).reshape(CODEBOOK, CODEBOOK)
    return N, U


def fit_prior(codes_per_file, n_q):
    """Choose alpha per layer on every 10th file, then refit on all files."""
    held = [c for i, c in enumerate(codes_per_file) if i % 10 == 9]
    train = [c for i, c in enumerate(codes_per_file) if i % 10 != 9] or codes_per_file
    Ntr, Utr = bigram_counts(train, n_q)
    Nho, _ = bigram_counts(held, n_q) if held else (None, None)
    alphas, xent = [], []
    for q in range(n_q):
        u = (Utr[q] + 1.0) / (Utr[q].sum() + CODEBOOK)
        if Nho is None or Nho[q].sum() == 0:
            alphas.append(8.0); xent.append(float("nan")); continue
        nz = Nho[q] > 0
        best = None
        for a in PRIOR_ALPHA_GRID:
            T = transition_matrix(Ntr[q].astype(np.float64), u, a)
            bits = -(Nho[q][nz] * np.log2(T[nz])).sum() / Nho[q].sum()
            if best is None or bits < best[1]:
                best = (a, bits)
        alphas.append(best[0]); xent.append(float(best[1]))
    N, U = bigram_counts(codes_per_file, n_q)
    trans = np.empty((n_q, CODEBOOK, CODEBOOK), dtype=np.float32)
    uni = np.empty((n_q, CODEBOOK), dtype=np.float64)
    for q in range(n_q):
        uni[q] = (U[q] + 1.0) / (U[q].sum() + CODEBOOK)
        trans[q] = transition_matrix(N[q].astype(np.float64), uni[q], alphas[q])
    return {"trans": trans, "unigram": uni, "alpha": np.array(alphas),
            "heldout_bits_per_index": np.array(xent),
            "n_frames": int(U[0].sum()), "n_files": len(codes_per_file)}


_PCTX = {}


def _prior_init(cfg):
    if cfg["torch_threads"]:
        torch.set_num_threads(cfg["torch_threads"])
    ec = EncodecModel.encodec_model_24khz().eval()
    for p in ec.parameters():
        p.requires_grad = False
    _PCTX.update(ec=ec, n_q=cfg["n_q"])


def _prior_codes(path):
    ec, n_q = _PCTX["ec"], _PCTX["n_q"]
    data, sr = sf.read(path)
    w = torch.from_numpy(data).float()
    w = w.unsqueeze(0) if w.ndim == 1 else w.T
    wav = convert_audio(w, sr, 24000, 1)[None]
    with torch.no_grad():
        codes = ec.quantizer(ec.encoder(wav), FRAME_RATE, n_q_to_bandwidth(n_q)).codes
    return codes[:, 0].numpy().astype(np.int16)


def list_audio(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
                  + glob.glob(os.path.join(folder, "**", "*.flac"), recursive=True))


def load_or_build_prior(args, n_q, test_files):
    path = args.prior_cache or f"sbsd_prior_nq{n_q}.npz"
    if os.path.exists(path) and not args.rebuild_prior:
        z = np.load(path)
        if int(z["n_q"]) == n_q and z["trans"].shape[1] == CODEBOOK:
            print(f"prior     loaded {path}: {int(z['n_files'])} files, "
                  f"{int(z['n_frames'])} frames, alpha {z['alpha'].tolist()}")
            return {k: z[k] for k in z.files}, path
        print(f"prior     {path} is for n_q={int(z['n_q'])}, rebuilding")
    if not args.prior_dir:
        raise SystemExit("M3.1 needs a Markov index prior: pass --prior_dir with speech "
                         "that is NOT in the test set (e.g. the training folder), or an "
                         "existing --prior_cache")
    files = list_audio(args.prior_dir)
    if args.prior_max_files > 0:
        files = files[:args.prior_max_files]
    if not files:
        raise SystemExit(f"no audio under --prior_dir {args.prior_dir}")
    test_real = {os.path.realpath(f) for f in test_files}
    test_base = {os.path.basename(f) for f in test_files}
    clash = [f for f in files if os.path.realpath(f) in test_real
             or os.path.basename(f) in test_base]
    if clash:
        raise SystemExit(f"--prior_dir overlaps the test set ({len(clash)} files, e.g. "
                         f"{clash[0]}): the prior must come from disjoint speech")
    print(f"prior     estimating from {len(files)} files under {args.prior_dir} ...", flush=True)
    t0 = time.time()
    cfg = {"torch_threads": 1 if args.workers > 1 else 0, "n_q": n_q}
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers, initializer=_prior_init,
                                          initargs=(cfg,)) as pool:
            codes = pool.map(_prior_codes, files, chunksize=4)
    else:
        _prior_init(cfg)
        codes = [_prior_codes(f) for f in files]
    pr = fit_prior(codes, n_q)
    pr["n_q"] = np.array(n_q)
    pr["files_crc32"] = np.array(zlib.crc32("|".join(os.path.basename(f) for f in files)
                                            .encode()) & 0xFFFFFFFF)
    np.savez_compressed(path, **pr)
    print(f"prior     {pr['n_files']} files, {pr['n_frames']} frames in "
          f"{time.time() - t0:.0f} s -> {path} | alpha {pr['alpha'].tolist()} | held-out "
          f"cross-entropy {np.round(pr['heldout_bits_per_index'], 2).tolist()} bits/index "
          f"(10 without the prior)")
    return pr, path


# =========================================================
# (3008, 2256) IRA-LDPC + NORMALIZED MIN-SUM  (the over-the-air code)
# =========================================================
def build_ira_ldpc():
    """The deterministic IRA code of the OTA experiments, H = [A | B], as in
    tx_four_methods_with_ldpc.py / rx_four_methods_soft_ldpc.py of
    Hisar-Research/usrp-digital-jscc-audio. Not a standardized code.

    A: every information bit joins 3 checks, every check takes 9 information
       bits. B: dual-diagonal accumulator, check i holds p_i and p_{i-1}.

    Returns (check_vars, info_checks, info). check_vars is [m, 11] for
    nms_decode. Check 0 has only 10 edges, so its last entry is index n: a
    virtual variable fixed to 0, which is exactly the accumulator's p_{-1} = 0.
    info_checks is the [k, 3] check list of each information bit, for encoding."""
    n, k, m = LDPC_N, LDPC_K, LDPC_N - LDPC_K
    j = np.arange(k, dtype=np.int64)
    group, r = j // m, j % m
    info_checks = np.stack((r, (87 * r + 103 + 128 * group) % m,
                            (75 * r + 561 + 448 * group) % m), axis=1)
    assert (info_checks[:, 0] != info_checks[:, 1]).all() \
        and (info_checks[:, 0] != info_checks[:, 2]).all() \
        and (info_checks[:, 1] != info_checks[:, 2]).all(), "repeated check on a bit"
    assert (np.bincount(info_checks.ravel(), minlength=m) == LDPC_CHECK_INFO_DEG).all(), \
        "check degrees of A are not all 9"
    rows = [[] for _ in range(m)]
    for var, checks in enumerate(info_checks):
        for chk in checks:
            rows[chk].append(var)
    for chk in range(m):
        rows[chk] += [k + chk - 1 if chk else n, k + chk]
    check_vars = np.array(rows, dtype=np.int64)

    Hd = np.zeros((m, n + 1), dtype=np.int32)
    Hd[np.arange(m)[:, None], check_vars] = 1
    ov = Hd[:, :n] @ Hd[:, :n].T
    np.fill_diagonal(ov, 0)
    info = {"n": n, "k": k, "source_bits": LDPC_SOURCE_BITS, "known_zero_bits": LDPC_PAD_BITS,
            "structure": "IRA, H = [A | B], B dual-diagonal",
            "info_degree": int(info_checks.shape[1]), "check_info_degree": LDPC_CHECK_INFO_DEG,
            "four_cycles": int((ov[np.triu_indices(m, 1)] >= 2).sum()),
            "graph_crc32": int(zlib.crc32(check_vars.tobytes()) & 0xFFFFFFFF)}
    return check_vars, info_checks, info


def ldpc_encode(info_checks, source):
    """2250 source bits -> 3008 bits, [source | 6 known zeros | 752 parity]."""
    u = np.concatenate([source, np.zeros(LDPC_PAD_BITS, dtype=np.uint8)])
    syndrome = np.zeros(LDPC_N - LDPC_K, dtype=np.uint8)
    for e in range(info_checks.shape[1]):
        np.bitwise_xor.at(syndrome, info_checks[:, e], u)
    return np.concatenate([u, np.bitwise_xor.accumulate(syndrome)]).astype(np.uint8)


def nms_decode(llr, check_vars, alpha=LDPC_ALPHA, max_iters=LDPC_MAX_ITERS):
    """Normalized min-sum, flooding schedule, early stop on a clean syndrome.
    llr [3008], L > 0 means bit 1; negated on entry, so L > 0 means bit 0 below.
    The 6 known zeros are held at L >= LDPC_KNOWN_ZERO_LLR, whatever arrived."""
    lam = -np.asarray(llr, dtype=np.float64)
    lam[LDPC_SOURCE_BITS:LDPC_K] = np.maximum(lam[LDPC_SOURCE_BITS:LDPC_K],
                                              LDPC_KNOWN_ZERO_LLR)
    lam = np.append(lam, np.inf)                                   # p_{-1} = 0
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
            return x[:-1], it, True
        v2c = L[check_vars] - c2v
    return x[:-1], max_iters, False


def ldpc_encode_stream(info_bits, info_checks):
    """One codeword per 75-frame chunk, as over the air. A short last chunk is
    filled with zeros that the receiver is not told about."""
    n_blocks = int(math.ceil(len(info_bits) / LDPC_SOURCE_BITS))
    pad = n_blocks * LDPC_SOURCE_BITS - len(info_bits)
    padded = np.concatenate([info_bits, np.zeros(pad, dtype=np.uint8)])
    return (np.concatenate([ldpc_encode(info_checks,
                                        padded[b * LDPC_SOURCE_BITS:(b + 1) * LDPC_SOURCE_BITS])
                            for b in range(n_blocks)]), n_blocks, pad)


def ldpc_decode_stream(llr_stream, check_vars, n_info, alpha, max_iters):
    out, its, fails = [], 0, 0
    n_blocks = len(llr_stream) // LDPC_N
    for b in range(n_blocks):
        x, it, ok = nms_decode(llr_stream[b * LDPC_N:(b + 1) * LDPC_N], check_vars,
                               alpha, max_iters)
        out.append(x[:LDPC_SOURCE_BITS]); its += it; fails += (not ok)
    return np.concatenate(out)[:n_info], its / max(n_blocks, 1), fails


# =========================================================
# CHANNEL
# =========================================================
def snr_to_sigma(snr_db):
    """BPSK, unit average symbol energy, real AWGN: Es/N0 = snr_db."""
    return math.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))


def _seed(*parts):
    """crc32 rather than hash(), whose string hashing is salted per process."""
    return zlib.crc32("|".join(str(p) for p in (SEED,) + parts).encode()) & 0x7FFFFFFF


def crn_noise(n, file_key, trial, method):
    """Common random numbers: one draw per (file, trial, method), reused at
    every SNR, every mismatch and on every channel."""
    return np.random.default_rng(_seed("noise", file_key, trial, method)) \
        .standard_normal(n).astype(np.float32)


def fading_amplitudes(n_frames, coherence_frames, file_key, trial):
    """|h| per frame, E|h|^2 = 1, shared by every method and scenario.
    AR(1) complex Gaussian started in its stationary law, so it is stationary
    from frame 0; coherence_frames <= 0 gives independent frames."""
    rng = np.random.default_rng(_seed("fading", file_key, trial))
    w = (rng.standard_normal(n_frames) + 1j * rng.standard_normal(n_frames)) / math.sqrt(2.0)
    if coherence_frames <= 0:
        return np.abs(w).astype(np.float32)
    rho = math.exp(-1.0 / coherence_frames)
    g = np.empty(n_frames, dtype=np.complex128)
    g[0] = w[0]
    c = math.sqrt(1.0 - rho * rho)
    for i in range(1, n_frames):
        g[i] = rho * g[i - 1] + c * w[i]
    return np.abs(g).astype(np.float32)


def channel_llr(x, noise, h_frame, sigmas, channel, mismatch_db):
    """BPSK symbols x [N] through the channel at every SNR -> LLRs [S, N].

    Symbol k sits in frame k // 40 at the common 3000 symbol/s rate, so every
    method sees the same fade at the same time. The receiver uses
    sigma_hat^2 = sigma^2 10^(-mismatch/10) (and |h| or E|h|, see the header)."""
    sig = np.asarray(sigmas, dtype=np.float64)[:, None]
    if channel == "awgn":
        h = np.ones(len(x), dtype=np.float32)
        h_rx = 1.0
    else:
        h = h_frame[np.arange(len(x)) // TOTAL_BITS_PER_FRAME]
        h_rx = h if channel == "rayleigh" else H_MEAN
    y = (h * x)[None, :] + sig * noise[None, :]
    s2_hat = sig ** 2 * 10.0 ** (-mismatch_db / 10.0)
    return 2.0 * h_rx * y / s2_hat


# =========================================================
# METRICS
# =========================================================
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
    out["pesq"] = float("nan")
    if pesq_metric is not None:
        try:
            out["pesq"] = float(pesq_metric(METRIC_SR, ref, deg, pesq_mode))
        except Exception:
            pass                           # silent/degenerate output; see the plot script
    r = ref - ref.mean(); d = deg - deg.mean()
    rr = float(np.dot(r, r)) + 1e-12
    if float(np.dot(d, d)) < 1e-8 * rr:    # silent output: v2 returned 0 dB here
        out["sisdr"] = -60.0
    else:
        tgt = (np.dot(d, r) / rr) * r
        noi = d - tgt
        out["sisdr"] = float(max(-60.0, 10.0 * np.log10((np.dot(tgt, tgt) + 1e-12)
                                                         / (np.dot(noi, noi) + 1e-12))))
    return out


def n_q_to_bandwidth(n_q):
    """set_target_bandwidth() only accepts EnCodec's official list, which rules
    out n_q of 1 or 3; quantizer() itself does not validate, so the bandwidth is
    passed directly. Epsilon guards floor() underflow."""
    return n_q * BW_PER_LAYER_KBPS + 1e-6


# =========================================================
# ONE FILE, EVERY SCENARIO AND SNR -- the unit of work, one per worker
# =========================================================
_CTX = {}


def _init_worker(cfg):
    """Everything a file needs, built once per process. With --workers N each
    process uses one torch thread and one BLAS thread, so N workers occupy N
    cores."""
    if cfg["torch_threads"]:
        torch.set_num_threads(cfg["torch_threads"])
        try:
            from threadpoolctl import threadpool_limits
            _CTX["_blas_limit"] = threadpool_limits(1)
        except ImportError:
            pass
    tx, rx = VogeoTx(cfg["bundle"]), VogeoRx(cfg["bundle"])
    check_vars, info_checks, _ = build_ira_ldpc()
    ec = EncodecModel.encodec_model_24khz().eval()
    ec.set_target_bandwidth(1.5)                      # arbitrary; always overridden
    for p in ec.parameters():
        p.requires_grad = False
    z = np.load(cfg["prior_cache"])
    _CTX.update(cfg, tx=tx, rx=rx, ec=ec, check_vars=check_vars, info_checks=info_checks,
                blk=BlockCodes(cfg["blk_parity"]),
                trans=[z["trans"][q].astype(np.float32) for q in range(N_Q_CODED)],
                log_u=[np.log(z["unigram"][q]).astype(np.float32) for q in range(N_Q_CODED)])


def _layer_acc(pred, ref):
    acc = (pred == ref).mean(axis=1)
    return {f"acc_l{q + 1}": float(acc[q]) for q in range(min(3, len(acc)))}


def _run_file(job):
    """-> (file_key, [row dicts], [ceiling dicts])"""
    fidx, fpath, file_key = job
    c = _CTX
    tx, rx, ec, pm, blk = c["tx"], c["rx"], c["ec"], c["pesq_mode"], c["blk"]
    grid = c["grid"]
    sigmas = [snr_to_sigma(s) for s in grid]
    S = len(grid)

    data, sr = sf.read(fpath)
    w = torch.from_numpy(data).float()
    w = w.unsqueeze(0) if w.ndim == 1 else w.T
    wav = convert_audio(w, sr, 24000, 1)
    if c["seconds"] > 0:
        wav = wav[..., :int(c["seconds"] * 24000)]
    wav = wav[None]                                              # [1,1,N]
    ref16 = to_metric_rate(wav)

    def synth(codes):
        with torch.no_grad():
            return ec.decoder(ec.quantizer.decode(
                torch.as_tensor(codes, dtype=torch.long)[:, None]))

    # --save_audio: same noise and fading as the sweep, so a clip is exactly
    # the realization behind the plotted point
    save_dir = c["save_audio"] if c["save_audio"] and fidx < c["save_audio_files"] else None
    stem = os.path.splitext(file_key)[0].replace(os.sep, "_").replace("/", "_")

    def write_wav(path, wav24):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        x = torch.as_tensor(wav24).squeeze().detach().cpu().numpy()
        sf.write(path, np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), 24000,
                 subtype="PCM_16")

    # metrics memo keyed by the decoded index array: exact recoveries reuse the
    # ceiling, and the scale-invariant decoders (M2, M3.2) reuse their result
    # across mismatch scenarios -- no waveform is decoded twice
    memo = {}

    def metrics_for(codes):
        key = (codes.shape[0], np.ascontiguousarray(codes, dtype=np.int64).tobytes())
        if key not in memo:
            memo[key] = compute_metrics(ref16, to_metric_rate(synth(codes)), pm)
        return memo[key]

    with torch.no_grad():
        z = ec.encoder(wav)
        codes = {q: ec.quantizer(z, FRAME_RATE, n_q_to_bandwidth(q)).codes[:, 0].numpy()
                 .astype(np.int64) for q in (N_Q_CODED, N_Q_UNCODED)}
    ceil_rows = []
    for q, cq in codes.items():
        m = metrics_for(cq)
        ceil_rows.append({"file": file_key, "n_q": q, **m})
    if save_dir:
        write_wav(os.path.join(save_dir, "_ref", f"{stem}__reference.wav"), wav)
        for q, cq in codes.items():
            write_wav(os.path.join(save_dir, "_ref", f"{stem}__ceiling_nq{q}.wav"), synth(cq))
    c3, c4 = codes[N_Q_CODED], codes[N_Q_UNCODED]
    T = c3.shape[1]
    assert c4.shape[1] == T

    # ---- transmitted bits, all at 40 bits/frame ----
    tx.reset()
    bits = {"M1": tx.encode(z, c3).astype(np.uint8).reshape(-1),         # [T*40]
            "M3.1": blk.encode(c3).reshape(-1),                          # [T*40]
            "M2": codes_to_bitstream(c4)}                                # [T*40]
    info3 = codes_to_bitstream(c3)
    coded4, n_blk, _ = ldpc_encode_stream(info3, c["info_checks"])
    bits["M3.2"] = coded4
    for m in ("M1", "M3.1", "M2"):
        assert bits[m].size == T * TOTAL_BITS_PER_FRAME, f"{m}: {bits[m].size} bits"
    sym = {m: 2.0 * b.astype(np.float32) - 1.0 for m, b in bits.items()}
    n_slots = int(math.ceil(max(b.size for b in bits.values()) / TOTAL_BITS_PER_FRAME))

    rows = []

    def emit(base, method, si, pred, ref, extra):
        m = metrics_for(pred)
        rows.append({**base, "method": method, "snr": grid[si], **m,
                     "code_acc": float((pred == ref).mean()), **_layer_acc(pred, ref),
                     **extra})
        if save_dir and base["trial"] == 0:
            write_wav(os.path.join(save_dir, base["scenario"], f"snr{grid[si]:+g}dB",
                                   f"{stem}__{method}.wav"), synth(pred))

    for t in range(c["trials"]):
        noise = {m: crn_noise(sym[m].size, file_key, t, m) for m in METHODS}
        h_frame = fading_amplitudes(n_slots, c["coherence_frames"], file_key, t)
        for ch in c["channels"]:
            for mis in c["mismatch_db"]:
                base = {"scenario": scenario_id(ch, mis), "channel": ch,
                        "mismatch_db": mis, "file": file_key, "trial": t}
                llr = {m: channel_llr(sym[m], noise[m], h_frame, sigmas, ch, mis)
                       for m in METHODS}
                raw = {m: ((llr[m] > 0) != bits[m][None, :].astype(bool)).mean(axis=1)
                       for m in METHODS}

                # ---- M1: the deployed receiver, argmax output ----
                L1 = llr["M1"].reshape(S, T, TOTAL_BITS_PER_FRAME).astype(np.float32)
                for si in range(S):
                    with torch.no_grad():
                        rx.reset()
                        pred = np.asarray(rx.decode(L1[si]), dtype=np.int64)
                    sliced = bitstream_to_codes(
                        (L1[si, :, :N_Q_CODED * BITS_PER_CODE] > 0).astype(np.uint8),
                        N_Q_CODED, T)
                    emit(base, "M1", si, pred, c3,
                         {"raw_ber": float(raw["M1"][si]),
                          "hard_acc": float((sliced == c3).mean())})

                # ---- M3.1: per-index block codes, SBSD, MAP output ----
                L3 = llr["M3.1"].reshape(S, T, TOTAL_BITS_PER_FRAME)
                pred3 = np.empty((S, N_Q_CODED, T), dtype=np.int64)
                ml3 = np.empty((S, N_Q_CODED, T), dtype=np.int64)
                for q in range(N_Q_CODED):
                    ll = blk.loglik(L3, q)                               # [S,T,1024]
                    ml3[:, q] = ll.argmax(axis=2)
                    pred3[:, q] = sbsd_filter(ll, c["trans"][q], c["log_u"][q])
                for si in range(S):
                    emit(base, "M3.1", si, pred3[si], c3,
                         {"raw_ber": float(raw["M3.1"][si]),
                          "ml_acc": float((ml3[si] == c3).mean())})

                # ---- M3.2: IRA-LDPC, normalized min-sum ----
                for si in range(S):
                    dec, _, n_fail = ldpc_decode_stream(
                        llr["M3.2"][si], c["check_vars"], info3.size,
                        c["ldpc_alpha"], c["ldpc_iters"])
                    emit(base, "M3.2", si, bitstream_to_codes(dec, N_Q_CODED, T), c3,
                         {"raw_ber": float(raw["M3.2"][si]),
                          "block_fail_rate": float(n_fail / max(n_blk, 1))})

                # ---- M2: uncoded n_q = 4, slicer ----
                for si in range(S):
                    pred = bitstream_to_codes((llr["M2"][si] > 0).astype(np.uint8),
                                              N_Q_UNCODED, T)
                    emit(base, "M2", si, pred, c4, {"raw_ber": float(raw["M2"][si])})
    return file_key, rows, ceil_rows


# =========================================================
# SELF-TEST (code level, no audio)
# =========================================================
def selftest(blk_parity, check_vars=None, info_checks=None):
    """Raises on any failure; returns a short report."""
    rng = np.random.default_rng(SEED)
    rep = []
    blk = BlockCodes(blk_parity)
    assert blk.n_bits == TOTAL_BITS_PER_FRAME, f"block codes use {blk.n_bits} bits"
    for q, L in enumerate(blk.layers):
        if L["r"] == 4:
            assert L["dmin"] == 3, "(14,10) must be a shortened Hamming code"
        if L["r"] in (1, 2, 3):
            assert L["dmin"] == 2
    g_A, g_spec = golay_20_10()
    assert next(i + 1 for i, a in enumerate(g_spec) if a) == 6, "(20,10) Golay code must have dmin 6"
    # noiseless round trip, and a single bit error corrected where dmin >= 3
    c = rng.integers(0, CODEBOOK, (N_Q_CODED, 400))
    x = 2.0 * blk.encode(c) - 1.0
    for q in range(N_Q_CODED):
        assert np.array_equal(blk.loglik(8.0 * x, q).argmax(-1), c[q])
    for q, L in enumerate(blk.layers):
        if L["dmin"] >= 3:
            xe = x.copy()
            pos = L["off"] + rng.integers(0, L["n"], len(xe))
            xe[np.arange(len(xe)), pos] *= -1.0
            assert np.array_equal(blk.loglik(8.0 * xe, q).argmax(-1), c[q])
    rep.append("block codes: " + "; ".join(blk.describe()))
    # SBSD on a synthetic sticky Markov source must beat the prior-free decision
    K = CODEBOOK
    stay = 0.9
    trans = np.full((K, K), (1 - stay) / (K - 1), dtype=np.float32)
    np.fill_diagonal(trans, stay)
    log_u = np.full(K, -math.log(K), dtype=np.float32)
    seq = [int(rng.integers(K))]
    for _ in range(299):
        seq.append(seq[-1] if rng.random() < stay else int(rng.integers(K)))
    seq = np.array(seq)
    cw = blk.layers[0]["x"][seq]                                         # [T, 14]
    sig = snr_to_sigma(-2.0)
    y = cw + sig * rng.standard_normal(cw.shape).astype(np.float32)
    ll = 0.5 * ((2 * y / sig ** 2) @ blk.layers[0]["x"].T)[None]
    acc_ml = (ll[0].argmax(-1) == seq).mean()
    acc_sb = (sbsd_filter(ll, trans, log_u)[0] == seq).mean()
    assert acc_sb > acc_ml + 0.1, f"SBSD {acc_sb:.3f} vs ML {acc_ml:.3f}"
    rep.append(f"SBSD on a sticky Markov source at -2 dB: {acc_sb:.3f} vs ML {acc_ml:.3f}")
    # fading: unit power, and the LLR reduces to 2y/sigma^2 on AWGN without mismatch
    h = fading_amplitudes(200000, 6.0, "selftest", 0)
    assert abs(float(np.mean(h ** 2)) - 1.0) < 0.05, "E|h|^2 != 1"
    xs = np.where(rng.random(80) < 0.5, -1.0, 1.0).astype(np.float32)
    nz = rng.standard_normal(80).astype(np.float32)
    L = channel_llr(xs, nz, None, [snr_to_sigma(0.0)], "awgn", 0.0)[0]
    s0 = snr_to_sigma(0.0)
    assert np.allclose(L, 2 * (xs + s0 * nz) / s0 ** 2, rtol=1e-5)
    L3 = channel_llr(xs, nz, None, [s0], "awgn", 3.0)[0]
    assert np.allclose(L3 / L, 10 ** 0.3, rtol=1e-5), "mismatch must scale the LLR"
    rep.append(f"fading E|h|^2 = {np.mean(h ** 2):.3f}; LLR and mismatch scaling OK")
    if check_vars is not None:
        u0 = rng.integers(0, 2, LDPC_SOURCE_BITS, dtype=np.uint8)
        c0 = ldpc_encode(info_checks, u0)
        assert not c0[LDPC_SOURCE_BITS:LDPC_K].any(), "LDPC pad bits must be zero"
        assert not np.bitwise_xor.reduce(np.append(c0, 0)[check_vars], axis=1).any(), \
            "LDPC codeword fails H c = 0"
        l0 = np.where(c0 == 1, 30.0, -30.0)
        x0, _, ok0 = nms_decode(l0, check_vars)
        assert ok0 and np.array_equal(x0, c0), "LDPC noiseless round trip failed"
        l0[LDPC_SOURCE_BITS:LDPC_K] = 30.0                  # all 6 pad bits arrive wrong
        x1, it1, ok1 = nms_decode(l0, check_vars)
        assert ok1 and it1 == 1 and np.array_equal(x1, c0), "known zeros were not enforced"
        rep.append("IRA-LDPC H c = 0, noiseless round trip and known zeros OK")
    return rep


# =========================================================
# OUTPUT
# =========================================================
def _append_csv(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_summary(rows_csv, ceil_csv, meta):
    rows, ceil = _read_csv(rows_csv), _read_csv(ceil_csv)
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["method"], _num(r["snr"])), []).append(r)
    num_fields = [f for f in ROW_FIELDS if f not in
                  ("scenario", "channel", "mismatch_db", "method", "snr", "file", "trial")]
    res = {}
    for (scen, meth, snr), rs in sorted(groups.items()):
        out = {"n": len(rs)}
        for k in num_fields:
            v = np.array([_num(r.get(k)) for r in rs])
            if np.isfinite(v).any():
                out[k] = float(np.nanmean(v))
        out["pesq_fail_frac"] = float(np.mean([not np.isfinite(_num(r.get("pesq")))
                                               for r in rs]))
        res.setdefault(scen, {}).setdefault(meth, {})[f"{snr:g}"] = out
    ceilings = {}
    for q in sorted({int(_num(r["n_q"])) for r in ceil}):
        rs = [r for r in ceil if int(_num(r["n_q"])) == q]
        ceilings[f"ceiling_nq{q}"] = {k: float(np.nanmean([_num(r[k]) for r in rs]))
                                      for k in ("estoi", "pesq", "sisdr")}
        ceilings[f"ceiling_nq{q}"]["n"] = len(rs)
    return {"results": res, **ceilings, "_meta": meta}


def write_listening_page(root):
    """root/index.html: one table per scenario and SNR, one row per utterance,
    reference, ceilings and every method side by side. Open it in a browser."""
    ref_dir = os.path.join(root, "_ref")
    stems = sorted({f.split("__")[0] for f in os.listdir(ref_dir) if f.endswith(".wav")})
    cols = [("Reference", "_ref", "reference"), ("Ceiling n_q=3", "_ref", "ceiling_nq3"),
            ("Ceiling n_q=4", "_ref", "ceiling_nq4")]
    names = {"M1": "M1 proposed", "M3.1": "M3.1 Hamming + SBSD", "M3.2": "M3.2 LDPC",
             "M2": "M2 uncoded"}
    parts = ["<!doctype html><meta charset='utf-8'><title>Listening</title>",
             "<style>body{font:14px system-ui,sans-serif;margin:24px;color:#0b0b0b;"
             "background:#fcfcfb}table{border-collapse:collapse;margin-bottom:28px}"
             "td,th{border:1px solid #d9d8d4;padding:4px 6px;text-align:left;"
             "vertical-align:top}th{background:#f1f0ec}audio{width:190px}"
             "h2{margin-top:32px}</style>",
             "<h1>Listening samples</h1><p>Same noise and fading as the plotted sweep "
             "(trial 0). 24 kHz, clipped to &plusmn;1.</p>"]
    scen_dirs = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)) and d != "_ref")
    for scen in scen_dirs:
        snr_dirs = sorted(os.listdir(os.path.join(root, scen)), key=lambda s: float(s[3:-2]))
        for sd in snr_dirs:
            parts.append(f"<h2>{scen}, E<sub>s</sub>/N<sub>0</sub> = {sd[3:-2]} dB</h2>"
                         "<table><tr><th>Utterance</th>"
                         + "".join(f"<th>{c[0]}</th>" for c in cols)
                         + "".join(f"<th>{names[m]}</th>" for m in METHODS) + "</tr>")
            for st in stems:
                cells = [f"<td>{st}</td>"]
                for _, d, tag in cols:
                    rel = f"{d}/{st}__{tag}.wav"
                    cells.append(f"<td><audio controls preload='none' src='{rel}'></audio></td>"
                                 if os.path.exists(os.path.join(root, rel)) else "<td></td>")
                for m in METHODS:
                    rel = f"{scen}/{sd}/{st}__{m}.wav"
                    cells.append(f"<td><audio controls preload='none' src='{rel}'></audio></td>"
                                 if os.path.exists(os.path.join(root, rel)) else "<td></td>")
                parts.append("<tr>" + "".join(cells) + "</tr>")
            parts.append("</table>")
    path = os.path.join(root, "index.html")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return path


def print_table(summary, grid, scenarios, metric="estoi"):
    for scen in scenarios:
        block = summary["results"].get(scen, {})
        if not block:
            continue
        print(f"\n{metric.upper()} -- {scen}")
        print(f"{'SNR':>6s} | " + " ".join(f"{m:>8s}" for m in METHODS))
        for s in grid:
            vals = [block.get(m, {}).get(f"{float(s):g}", {}).get(metric, float("nan"))
                    for m in METHODS]
            print(f"{s:+6g} | " + " ".join(f"{v:8.4f}" for v in vals))


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", default="./bundle")
    ap.add_argument("--wav", default=None, help="single file; default: test2.wav nearby")
    ap.add_argument("--test_dir", default=None, help="average over every wav/flac here")
    ap.add_argument("--seconds", type=float, default=5.0, help="0 = whole file")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--snrs", type=float, nargs="*", default=None,
                    help=f"override the sweep (default {SNR_GRID_DB[0]}:1:{SNR_GRID_DB[-1]})")
    ap.add_argument("--channels", nargs="+", choices=CHANNELS, default=["awgn", "rayleigh"])
    ap.add_argument("--mismatch_db", type=float, nargs="+", default=[0.0, 3.0, -3.0],
                    help="assumed minus true Es/N0 at the receiver, dB (0 = perfect)")
    ap.add_argument("--coherence_frames", type=float, default=6.0,
                    help="Rayleigh AR(1) coherence in 13.33-ms frames; 0 = independent "
                         "frames (6 ~ 80 ms, pedestrian at 868 MHz is ~10)")
    ap.add_argument("--blk_parity", type=int, nargs=N_Q_CODED, default=[4, 3, 3],
                    help="M3.1 parity bits per RVQ layer; must sum to 10")
    ap.add_argument("--prior_dir", default=None,
                    help="speech NOT in the test set for the M3.1 Markov prior")
    ap.add_argument("--prior_cache", default=None,
                    help=f"default sbsd_prior_nq{N_Q_CODED}.npz")
    ap.add_argument("--prior_max_files", type=int, default=0, help="0 = all")
    ap.add_argument("--rebuild_prior", action="store_true")
    ap.add_argument("--ldpc_alpha", type=float, default=LDPC_ALPHA)
    ap.add_argument("--ldpc_iters", type=int, default=LDPC_MAX_ITERS)
    ap.add_argument("--pesq_mode", choices=("nb", "wb"), default=PESQ_MODE)
    ap.add_argument("--workers", type=int, default=1,
                    help="files decoded in parallel, one torch/BLAS thread each")
    ap.add_argument("--out_prefix", default="vogeo_sim_v3")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already in OUT_prefix_rows.csv")
    ap.add_argument("--max_files", type=int, default=0,
                    help="use only the first N test files (sorted); 0 = all")
    ap.add_argument("--save_audio", default=None,
                    help="write decoded WAVs (trial 0) and an index.html here")
    ap.add_argument("--save_audio_files", type=int, default=5,
                    help="how many of the first test files to save audio for")
    ap.add_argument("--selftest", action="store_true", help="code-level checks, then exit")
    args = ap.parse_args()
    grid = [int(v) if float(v).is_integer() else float(v)
            for v in (args.snrs if args.snrs else SNR_GRID_DB)]
    mismatches = [int(v) if float(v).is_integer() else float(v) for v in args.mismatch_db]
    if sum(args.blk_parity) + N_Q_CODED * BITS_PER_CODE != TOTAL_BITS_PER_FRAME:
        raise SystemExit(f"--blk_parity must sum to "
                         f"{TOTAL_BITS_PER_FRAME - N_Q_CODED * BITS_PER_CODE}")

    t0 = time.time()
    check_vars, info_checks, ldpc_info = build_ira_ldpc()
    rep = selftest(args.blk_parity, check_vars, info_checks)
    print(f"self-test OK in {time.time() - t0:.1f}s")
    for line in rep:
        print(f"  {line}")
    if args.selftest:
        return

    # ---- files ----
    if args.test_dir:
        files = list_audio(args.test_dir)
        keys = [os.path.relpath(f, args.test_dir) for f in files]
    else:
        w = args.wav or next((p for p in ("./test2.wav", "../test2.wav",
                                          "../deploy/test2.wav") if os.path.exists(p)), None)
        files = [w] if w else []
        keys = [os.path.basename(f) for f in files]
    if not files:
        raise SystemExit("no audio found -- pass --wav or --test_dir")
    if args.max_files > 0:
        files, keys = files[:args.max_files], keys[:args.max_files]

    # ---- bundle ----
    tx, rx = VogeoTx(args.bundle), VogeoRx(args.bundle)
    assert tx.run_id == rx.run_id, "transmitter and receiver are from different runs"
    if (tx.n_q, tx.parity_dim) != (N_Q_CODED, TOTAL_BITS_PER_FRAME - N_Q_CODED * BITS_PER_CODE):
        raise SystemExit(f"bundle is n_q={tx.n_q}, P={tx.parity_dim}; this script "
                         f"compares at n_q={N_Q_CODED} + 10 protection bits")
    del tx, rx

    prior, prior_path = load_or_build_prior(args, N_Q_CODED, files)
    blk = BlockCodes(args.blk_parity)
    m1_lat = 1000.0 / FRAME_RATE
    ldpc_lat = 1000.0 * LDPC_N / (TOTAL_BITS_PER_FRAME * FRAME_RATE)
    scenarios = [scenario_id(ch, m) for ch in args.channels for m in mismatches]

    print(f"M1        n_q={N_Q_CODED} + 10 learned parity bits, causal CNN, argmax  "
          f"latency {m1_lat:6.1f} ms")
    print(f"M3.1      n_q={N_Q_CODED} + per-index codes {', '.join(blk.describe())}; "
          f"SBSD (Markov prior, forward filtering), MAP output  latency {m1_lat:6.1f} ms")
    print(f"M3.2      n_q={N_Q_CODED} + IRA-LDPC ({LDPC_N},{LDPC_K}), NMS alpha="
          f"{args.ldpc_alpha} <= {args.ldpc_iters} it  latency {m1_lat + ldpc_lat:6.1f} ms")
    print(f"M2        n_q={N_Q_UNCODED}, slicer  latency {m1_lat:6.1f} ms")
    print(f"channels  {args.channels} x mismatch {mismatches} dB -> {len(scenarios)} "
          f"scenarios | Rayleigh coherence {args.coherence_frames:g} frames | SNR {grid} | "
          f"{args.trials} trial(s) | {len(files)} file(s) | PESQ {args.pesq_mode.upper()} | "
          f"{args.workers} worker(s)")

    meta = {
        "snr_grid": grid, "channels": args.channels, "mismatch_db": mismatches,
        "scenarios": scenarios, "methods": list(METHODS),
        "coherence_frames": args.coherence_frames, "trials": args.trials,
        "n_test_files": len(files), "seconds": args.seconds, "test_dir": args.test_dir,
        "pesq": {"nb": "ITU-T P.862 narrowband, P.862.1 MOS-LQO",
                 "wb": "ITU-T P.862.2 wideband MOS-LQO"}[args.pesq_mode],
        "metric_sr": METRIC_SR, "interleaver": "none",
        "llr": "2 h_rx y / sigma_hat^2, sigma_hat^2 = sigma^2 10^(-mismatch/10)",
        "m1": {"n_q": N_Q_CODED, "parity_bits": 10, "decision": "argmax (hard)",
               "latency_ms": m1_lat},
        "m3_1": {"n_q": N_Q_CODED, "codes": blk.describe(), "parity": args.blk_parity,
                 "A": [L["A"].tolist() for L in blk.layers],
                 "decoder": "softbit source decoding, first-order Markov prior, "
                            "forward filtering, MAP (hard) output",
                 "prior_cache": prior_path, "prior_alpha": prior["alpha"].tolist(),
                 "prior_heldout_bits_per_index": prior["heldout_bits_per_index"].tolist(),
                 "prior_n_files": int(prior["n_files"]), "prior_n_frames": int(prior["n_frames"]),
                 "latency_ms": m1_lat},
        "m3_2": {"n_q": N_Q_CODED, "code_rate": "3/4", "code": "IRA-LDPC",
                 "decoder": "normalized min-sum", "alpha": args.ldpc_alpha,
                 "max_iters": args.ldpc_iters, "known_zero_llr": LDPC_KNOWN_ZERO_LLR,
                 "latency_ms": m1_lat + ldpc_lat, **ldpc_info},
        "m2": {"n_q": N_Q_UNCODED, "decision": "slicer", "latency_ms": m1_lat},
    }

    rows_csv, ceil_csv = f"{args.out_prefix}_rows.csv", f"{args.out_prefix}_ceil.csv"
    cfg_json = f"{args.out_prefix}_config.json"
    run_cfg = {k: meta[k] for k in ("snr_grid", "channels", "mismatch_db", "coherence_frames",
                                    "trials", "seconds", "test_dir")}
    run_cfg.update(blk_parity=args.blk_parity, pesq_mode=args.pesq_mode,
                   prior_crc=int(prior.get("files_crc32", 0)),
                   ldpc_crc=ldpc_info["graph_crc32"], ldpc_iters=args.ldpc_iters)
    done = set()
    if args.resume and os.path.exists(cfg_json):
        old = json.load(open(cfg_json))
        if old != json.loads(json.dumps(run_cfg)):
            raise SystemExit(f"--resume: {cfg_json} differs from this run's settings")
        done = {r["file"] for r in _read_csv(rows_csv)}
        print(f"resume    {len(done)} file(s) already in {rows_csv}")
    else:
        for p in (rows_csv, ceil_csv):
            if os.path.exists(p):
                os.remove(p)
    json.dump(run_cfg, open(cfg_json, "w"), indent=2)

    cfg = {"bundle": args.bundle, "torch_threads": 1 if args.workers > 1 else 0,
           "seconds": args.seconds, "trials": args.trials, "grid": grid,
           "channels": args.channels, "mismatch_db": mismatches,
           "coherence_frames": args.coherence_frames, "blk_parity": args.blk_parity,
           "prior_cache": prior_path, "ldpc_alpha": args.ldpc_alpha,
           "ldpc_iters": args.ldpc_iters, "pesq_mode": args.pesq_mode,
           "save_audio": args.save_audio, "save_audio_files": args.save_audio_files}
    jobs = [(i, f, k) for i, (f, k) in enumerate(zip(files, keys)) if k not in done]
    t_start = time.time()
    mid = min(grid, key=lambda v: abs(v))

    def collect(res, n_done):
        key, rows, ceil_rows = res
        _append_csv(rows_csv, ROW_FIELDS, rows)
        _append_csv(ceil_csv, CEIL_FIELDS, ceil_rows)
        parts = []
        for ch in args.channels:
            sc = scenario_id(ch, mismatches[0])
            v = {r["method"]: r["estoi"] for r in rows
                 if r["scenario"] == sc and r["snr"] == mid and r["trial"] == 0}
            parts.append(f"{ch}: " + " ".join(f"{m} {v.get(m, float('nan')):.2f}"
                                               for m in METHODS))
        el = time.time() - t_start
        print(f"  [{n_done:3d}/{len(jobs)}] {key[-28:]:28s} ESTOI @ {mid:+g} dB, "
              f"mis {mismatches[0]:+g}  " + " | ".join(parts)
              + f"   [{el:6.0f} s, ETA {el / n_done * (len(jobs) - n_done):6.0f} s]",
              flush=True)

    if args.workers > 1 and len(jobs) > 1:
        # spawn, not fork: forking a parent that has touched torch can deadlock
        # its OpenMP pool. Each worker loads the models once in _init_worker.
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                          initargs=(cfg,)) as pool:
            for n_done, res in enumerate(pool.imap_unordered(_run_file, jobs), 1):
                collect(res, n_done)
    elif jobs:
        _init_worker(cfg)
        for n_done, job in enumerate(jobs, 1):
            collect(_run_file(job), n_done)

    summary = build_summary(rows_csv, ceil_csv, meta)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_table(summary, grid, scenarios, "estoi")
    for k in sorted(x for x in summary if x.startswith("ceiling_nq")):
        print(f"  {k} (no channel): ESTOI {summary[k]['estoi']:.4f}  "
              f"PESQ {summary[k]['pesq']:.3f}")
    print(f"\nwrote {rows_csv}, {ceil_csv}, {args.out_prefix}_summary.json")
    if args.save_audio and os.path.isdir(os.path.join(args.save_audio, "_ref")):
        print(f"listening page: {write_listening_page(args.save_audio)}")
    print(f"figures: python plot_sim_v3.py --prefix {args.out_prefix} --out_dir figs")


if __name__ == "__main__":
    main()