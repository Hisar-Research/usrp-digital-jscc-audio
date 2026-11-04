"""
SNR sweep for the deployed SOFT bundle, against two baselines, all over AWGN.

Where selftest_v2.py runs the chain once at a single operating point, this runs
it across -5:1:8 dB and reports ESTOI at every step. It drives the same
deployment classes -- VogeoTx, VogeoRx, vogeo_llr -- so what it measures is the
bundle you would actually ship, not the training-time model.

Three methods, ALL spending the same 40 bits/frame = 3.0 kbps:

  M1   Proposed   : the bundle. EnCodec's RVQ indices sent uncoded plus the
                    learned parity side channel, soft-decoded: 1024-way codeword
                    posterior, MMSE latent, log-posterior prior.
  M2   Uncoded    : EnCodec at 3.0 kbps (4 RVQ layers), no protection at all.
                    The pure 3 kbps benchmark.
  M3.1 Conv 3/4   : EnCodec at 30 bits/frame (3 RVQ layers) + the CCSDS
                    131.0-B-3 / IESS-308 rate-1/2 K=7 (133,171) code, punctured
                    to rate 3/4 per CCSDS Table 3 (Cain/Clark/Geist 1979: keep
                    X1,Y1,Y2,X3 of every 3 pairs, drop X2,Y3). Soft-decision
                    streaming Viterbi.

The rate fixes the split: 3/4 of 40 bits/frame is 30 info bits, i.e. n_q = 3.
That is the same split the bundle uses, so M1 and M3.1 are two ways of spending
one identical budget -- 30 bits of EnCodec plus 10 bits of protection -- and the
comparison is exactly "learned parity vs. a standard convolutional code".

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
M3.1's truncated traceback adds tb_depth/(30*75) seconds on top.

Interleaving is off: AWGN has no burst structure to break up.

Common random numbers: the noise for a given (method, trial) is drawn ONCE and
reused at every SNR, scaled by sigma. Without that, each point draws fresh noise
and the curves wander non-monotonically for reasons that are pure Monte-Carlo.

Usage:
    python selftest_sim.py                       # the bundle in ./bundle
    python selftest_sim.py --trials 5 --seconds 8
    python selftest_sim.py --true-llr            # skip the blind LLR estimator
    python results_plot.py --json vogeo_sim_soft_awgn.json --out_dir plots

Writes vogeo_sim_soft_awgn.json (the format results_plot.py reads) and
sim_estoi_awgn.png.
"""

import argparse
import glob
import json
import math
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
N_Q_UNCODED = 4                           # M2: 4 x 10 = 40 bits/frame
M3_CODE_RATE = "3/4"
TB_DEPTH    = 35                          # 5*K, the standard traceback rule
FRAME_RATE  = 75
BW_PER_LAYER_KBPS = math.log2(1024) * FRAME_RATE / 1000      # 0.75
SEED = 20260914


# =========================================================
# RATE-1/2 K=7 CONVOLUTIONAL CODE  (CCSDS / IESS-308, G = [171,133] octal)
# =========================================================
K_CONST = 7
N_STATES = 1 << (K_CONST - 1)
G_POLY = (0o171, 0o133)   # CCSDS order: column 0 = C1 = 171, column 1 = C2 = 133.
                          # With (133, 171) the puncture table below kept the wrong
                          # outputs (dfree 4, not 5). C2's inversion is omitted: a
                          # coset shift, no effect on distance or AWGN error rate.


def _parity(x):
    x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1
    return x & 1


def _build_trellis():
    nxt = np.zeros((N_STATES, 2), dtype=np.int64)
    out = np.zeros((N_STATES, 2, 2), dtype=np.int64)
    for s in range(N_STATES):
        for b in (0, 1):
            reg = (b << (K_CONST - 1)) | s          # newest bit at the MSB
            for j, g in enumerate(G_POLY):
                out[s, b, j] = _parity(reg & g)
            nxt[s, b] = reg >> 1
    return nxt, out


NEXT_STATE, OUT_BITS = _build_trellis()
SYM_PM = (2 * OUT_BITS - 1).astype(np.float64)      # [64, 2, 2] in {-1, +1}
# CCSDS 131.0-B-3 Table 3-2: of every 3 pairs send C1(1) C2(1) C2(2) C1(3)
PUNCTURE_3_4 = np.array([(True, True), (False, True), (True, False)], dtype=bool)


def conv_encode_pairs(info_bits):
    """1-D uint8 -> [N,2] rate-1/2 output pairs, zero-tail terminated."""
    padded = np.concatenate([info_bits, np.zeros(K_CONST - 1, dtype=np.uint8)])
    out = np.empty((len(padded), 2), dtype=np.uint8)
    state = 0
    for i, b in enumerate(padded):
        out[i] = OUT_BITS[state, b]
        state = NEXT_STATE[state, b]
    return out


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
    -sum_j (2c_j - 1) * L_j, exact ML for BPSK (see the module header)."""
    n_steps = n_info + K_CONST - 1
    rx = np.asarray(llr_pairs, dtype=np.float64).reshape(n_steps, 2)

    INF = 1e18
    pm = np.full(N_STATES, INF); pm[0] = 0.0
    surv = np.zeros((tb_depth + 1, N_STATES), dtype=np.int8)      # ring buffers:
    prev = np.zeros((tb_depth + 1, N_STATES), dtype=np.int64)     # the only state kept
    out = np.zeros(n_info, dtype=np.uint8)
    src = np.arange(N_STATES)

    for t in range(n_steps):
        bm = -(SYM_PM * rx[t]).sum(axis=2)          # [64, 2]
        cand = pm[:, None] + bm
        new_pm = np.full(N_STATES, INF)
        new_prev = np.zeros(N_STATES, dtype=np.int64)
        new_bit = np.zeros(N_STATES, dtype=np.int8)
        for b in (0, 1):
            tgt = NEXT_STATE[:, b]
            order = np.argsort(cand[:, b])
            tgt_o, val_o, src_o = tgt[order], cand[order, b], src[order]
            _, first = np.unique(tgt_o, return_index=True)
            mask = np.zeros(len(tgt_o), dtype=bool); mask[first] = True
            sel = mask & (val_o < new_pm[tgt_o])
            new_pm[tgt_o[sel]] = val_o[sel]
            new_prev[tgt_o[sel]] = src_o[sel]
            new_bit[tgt_o[sel]] = b

        pm = new_pm - new_pm.min()          # renormalise so metrics do not drift
        slot = t % (tb_depth + 1)
        surv[slot] = new_bit; prev[slot] = new_prev

        if t >= tb_depth:                    # window full: emit tb_depth steps back
            state = int(np.argmin(pm))
            for k in range(tb_depth + 1):
                s = (t - k) % (tb_depth + 1)
                bit = surv[s, state]; state = prev[s, state]
            if t - tb_depth < n_info:
                out[t - tb_depth] = bit

    state = 0                                # flush the zero-terminated tail
    for t in range(n_steps - 1, max(n_steps - tb_depth, 0) - 1, -1):
        s = t % (tb_depth + 1)
        bit = surv[s, state]; state = prev[s, state]
        if t < n_info:
            out[t] = bit
    return out


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


def compute_metrics(ref16, deg16):
    n = min(len(ref16), len(deg16))
    ref, deg = ref16[:n], deg16[:n]
    out = {}
    try:
        out["estoi"] = float(stoi_metric(ref, deg, METRIC_SR, extended=True))
    except Exception:
        out["estoi"] = float("nan")
    if pesq_metric is not None:
        try:
            out["pesq"] = float(pesq_metric(METRIC_SR, ref, deg, "wb"))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./bundle")
    ap.add_argument("--wav", default=None, help="single file; default: test2.wav nearby")
    ap.add_argument("--test_dir", default=None, help="average over every wav/flac here")
    ap.add_argument("--seconds", type=float, default=5.0, help="0 = whole file")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tb_depth", type=int, default=TB_DEPTH)
    ap.add_argument("--snrs", type=float, nargs="*", default=None,
                    help=f"override the sweep (default {SNR_GRID_DB[0]}:1:{SNR_GRID_DB[-1]})")
    ap.add_argument("--true-llr", dest="true_llr", action="store_true",
                    help="form L = 2y/sigma^2 from the known sigma instead of "
                         "estimating it blind, to separate the scheme from the "
                         "estimator")
    ap.add_argument("--json", default="vogeo_sim_soft_awgn.json")
    ap.add_argument("--plot", default="sim_estoi_awgn.png")
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

    m1_lat = 1000.0 / FRAME_RATE
    m3_lat = 1000.0 * args.tb_depth / (n_q3 * BITS_PER_CODE * FRAME_RATE)
    print(f"bundle    {scenario} soft, run_id {tx.run_id}, trained on "
          f"{tx.m.get('trained_channel')} (val acc {tx.m.get('trained_val_acc', float('nan')):.4f})")
    print(f"M1        n_q={n_q1} + {par1} learned parity bits   latency {m1_lat:.1f} ms (causal)")
    print(f"M2        n_q={N_Q_UNCODED}, unprotected            latency {m1_lat:.1f} ms")
    print(f"M3.1      n_q={n_q3} + CCSDS K=7 rate {M3_CODE_RATE}   latency "
          f"{m1_lat + m3_lat:.1f} ms (tb={args.tb_depth})")
    print(f"channel   AWGN, {grid} dB | {args.trials} trial(s) | {len(files)} file(s) | "
          f"LLR {'true' if args.true_llr else 'blind-estimated'}")

    ec = EncodecModel.encodec_model_24khz().eval()
    ec.set_target_bandwidth(1.5)                      # arbitrary; always overridden
    for p in ec.parameters():
        p.requires_grad = False

    m_keys = [f"M1_{scenario}", "M2_uncoded", f"M3.1_{scenario}"]
    n_qs = sorted({n_q1, n_q3, N_Q_UNCODED})
    acc = {m: {s: [] for s in grid} for m in m_keys}
    ceil_acc = {f"ceiling_nq{q}": [] for q in n_qs}
    t_start = time.time()

    for fidx, fpath in enumerate(files):
        data, sr = sf.read(fpath)
        w = torch.from_numpy(data).float()
        w = w.unsqueeze(0) if w.ndim == 1 else w.T
        wav = convert_audio(w, sr, 24000, 1)
        if args.seconds > 0:
            wav = wav[..., :int(args.seconds * 24000)]
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
                    compute_metrics(ref16, to_metric_rate(ceil)))
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
        if fidx == 0:
            print(f"\n{T} frames | M1 {bits1.size} bits | M2 {bits2.size} bits | "
                  f"M3.1 {info3.size} info -> {coded3.size} coded "
                  f"(+{100*(coded3.size/(T*TOTAL_BITS_PER_FRAME)-1):.2f}% from the "
                  f"{K_CONST-1}-bit zero tail)\n")

        tx_sym = {f"M1_{scenario}": 2.0 * bits1.reshape(-1) - 1.0,
                  "M2_uncoded":     2.0 * bits2 - 1.0,
                  f"M3.1_{scenario}": 2.0 * coded3 - 1.0}
        noise = {m: {t: crn_noise(len(tx_sym[m]), t, f"{fidx}|{m}")
                     for t in range(args.trials)} for m in m_keys}

        for snr in grid:
            sigma = snr_to_sigma(snr)
            for t in range(args.trials):
                with torch.no_grad():
                    # ---- M1: the deployed soft receiver ----
                    m = f"M1_{scenario}"
                    y = (tx_sym[m] + sigma * noise[m][t]).reshape(T, TOTAL_BITS_PER_FRAME)
                    llr = (2.0 * y / sigma ** 2).astype(np.float32) if args.true_llr \
                        else llr_from_samples(y)
                    rx.reset()
                    pred1 = rx.decode(llr)                            # [n_q,T]
                    out1 = ec.decoder(ec.quantizer.decode(
                        torch.as_tensor(pred1, dtype=torch.long)[:, None]))
                    # what the link would have delivered with no model: slice
                    # the RVQ columns of the frame at zero
                    sliced = (llr[:, :n_q1 * BITS_PER_CODE].reshape(-1) > 0).astype(np.uint8)
                    acc[m][snr].append({
                        **compute_metrics(ref16, to_metric_rate(out1)),
                        "code_acc": float((pred1 == c1).mean()),
                        "hard_acc": float((bitstream_to_codes(sliced, n_q1, T) == c1).mean())})

                    # ---- M2: no protection, slice and dequantize ----
                    m = "M2_uncoded"
                    y = tx_sym[m] + sigma * noise[m][t]
                    pred2 = bitstream_to_codes((y > 0).astype(np.uint8), N_Q_UNCODED, T)
                    out2 = ec.decoder(ec.quantizer.decode(
                        torch.as_tensor(pred2, dtype=torch.long)[:, None]))
                    a2 = float((pred2 == c2).mean())
                    acc[m][snr].append({**compute_metrics(ref16, to_metric_rate(out2)),
                                        "code_acc": a2, "hard_acc": a2})

                    # ---- M3.1: CCSDS rate-3/4, soft Viterbi ----
                    m = f"M3.1_{scenario}"
                    y = tx_sym[m] + sigma * noise[m][t]
                    llr3 = 2.0 * y.astype(np.float64) / sigma ** 2
                    dec3 = viterbi_decode_soft(
                        undo_rate_llr(llr3, len(pairs3), M3_CODE_RATE).reshape(-1),
                        len(info3), tb_depth=args.tb_depth)
                    pred3 = bitstream_to_codes(dec3, n_q3, T)
                    out3 = ec.decoder(ec.quantizer.decode(
                        torch.as_tensor(pred3, dtype=torch.long)[:, None]))
                    acc[m][snr].append({
                        **compute_metrics(ref16, to_metric_rate(out3)),
                        "code_acc": float((pred3 == c3).mean()),
                        "raw_ber": float(((y > 0) != (coded3 > 0)).mean()),
                        "post_ber": float((dec3 != info3).mean())})

            done = [f"{acc[m][snr][-1]['estoi']:.4f}" for m in m_keys]
            print(f"  {snr:+3d} dB  ESTOI  M1 {done[0]}  M2 {done[1]}  M3.1 {done[2]}"
                  f"   [{time.time()-t_start:5.0f} s]")

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
        "scenario": scenario, "run_id": tx.run_id,
        "m1": {"n_q": n_q1, "parity_dim": par1, "latency_ms": m1_lat,
               "llr": "true" if args.true_llr else "blind"},
        "m2": {"n_q": N_Q_UNCODED, "latency_ms": m1_lat},
        "m3_1": {"n_q": n_q3, "code_rate": M3_CODE_RATE, "K": K_CONST,
                 "poly_octal": ["0o171", "0o133"], "decision": "soft",
                 "tb_depth": args.tb_depth, "latency_ms": m1_lat + m3_lat},
    }
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- table ----
    print(f"\nESTOI, mean over {args.trials} trial(s) x {len(files)} file(s)")
    print(f"{'SNR':>6s} | {'M1 proposed':>11s} {'M2 uncoded':>11s} "
          f"{'M3.1 conv 3/4':>13s} | {'M1-M2':>7s} {'M1-M3.1':>8s}")
    print("-" * 70)
    for s in grid:
        a, b, c = (results[m][str(s)]["estoi"] for m in m_keys)
        print(f"{s:+5d}  | {a:11.4f} {b:11.4f} {c:13.4f} | {a-b:+7.4f} {a-c:+8.4f}")
    print("-" * 70)
    for q in n_qs:
        print(f"  ceiling n_q={q} (no channel): ESTOI "
              f"{results[f'ceiling_nq{q}']['estoi']:.4f}")

    # ---- plot ----
    styles = {m_keys[0]: ("Proposed (M1), soft", "o", "#1f77b4", "-"),
              m_keys[1]: ("Uncoded 3.0 kbps (M2)", "s", "#7f7f7f", "--"),
              m_keys[2]: (f"Conv {M3_CODE_RATE} CCSDS K=7 (M3.1), soft", "^", "#2ca02c", "-.")}
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
    print(f"full 4-panel version: python ../results_plot.py --json {args.json} "
          f"--out_dir plots")


if __name__ == "__main__":
    main()
