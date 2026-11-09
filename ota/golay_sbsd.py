"""Golay + SBSD (M3.1) for the OTA experiment.

This is the M3.1 baseline of the simulation, sim/selftest_sim_v3.py run with
--blk_parity 10 0 0, packaged for 1-second OTA chunks. The code construction
and the SBSD recursion are copied from the simulation; check_against_sim()
compares the two.

Frame: 40 bits per EnCodec frame, frame-major, MSB first:

    [layer 1: 10 index bits + 10 parity bits | layer 2: 10 bits | layer 3: 10 bits]

Layer 1 uses a systematic (20,10,6) code: the extended Golay (24,12,8) code
shortened on 2 information positions and punctured on the 2 parity positions
that leave the best weight spectrum. Layers 2 and 3 are sent uncoded.

Receiver: softbit source decoding (SBSD; Fingscheidt and Vary, 2001).
1. Codeword log-likelihoods come from the LLRs (L > 0 means bit 1).
2. They are combined with a first-order Markov index prior per layer, read
   from sim/sbsd_prior_nq3.npz.
3. The posterior is filtered forward in time, and the MAP index is decided
   (hard output).

The filter state is carried from chunk to chunk, so decoding one utterance
chunk by chunk gives the same indices as decoding it in one pass. Call reset()
at the start of each utterance, and after a lost chunk.
"""
import itertools

import numpy as np

BITS_PER_CODE = 10
CODEBOOK = 1 << BITS_PER_CODE                       # 1024
N_Q = 3
PARITY = (10, 0, 0)                                 # --blk_parity 10 0 0
BITS_PER_FRAME = N_Q * BITS_PER_CODE + sum(PARITY)  # 40

SHIFTS = np.arange(BITS_PER_CODE - 1, -1, -1)
MSG_BITS = ((np.arange(CODEBOOK)[:, None] >> SHIFTS) & 1).astype(np.uint8)   # [1024, 10]


def _weight_spectrum(A):
    """(A_1, ..., A_n) of the systematic code [u | u A mod 2], zero word excluded."""
    par = (MSG_BITS.astype(np.int64) @ A.astype(np.int64)) & 1
    w = MSG_BITS.astype(np.int64).sum(axis=1) + par.sum(axis=1)
    return np.bincount(w[1:], minlength=A.shape[0] + A.shape[1] + 1)[1:]


def golay_20_10():
    """(20,10,6) code from the extended Golay (24,12,8): shorten 2 information
    positions -> (22,10,8), then puncture the 2 parity positions that leave the
    best weight spectrum -> dmin 6, the best possible for (20,10)."""
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


class GolaySBSD:
    """Encoder and streaming SBSD decoder. prior_path is needed only to decode."""

    def __init__(self, prior_path=None):
        A, spec = golay_20_10()
        dmin = next(i + 1 for i, a in enumerate(spec) if a)
        if dmin != 6:
            raise RuntimeError(f"(20,10) Golay code has dmin {dmin}, expected 6")
        self.A = A
        cw1 = np.concatenate([MSG_BITS, (MSG_BITS.astype(np.int64) @ A) & 1],
                             axis=1).astype(np.uint8)                  # [1024, 20]
        self.layers = []
        off = 0
        for r, cw in zip(PARITY, (cw1, MSG_BITS, MSG_BITS)):
            n = BITS_PER_CODE + r
            self.layers.append({"off": off, "n": n, "cw": cw,
                                "x": (2.0 * cw - 1.0).astype(np.float32)})
            off += n
        assert off == BITS_PER_FRAME

        self.trans = self.log_u = None
        if prior_path is not None:
            z = np.load(prior_path)
            if int(z["n_q"]) != N_Q or z["trans"].shape != (N_Q, CODEBOOK, CODEBOOK):
                raise RuntimeError(f"{prior_path} is not an n_q={N_Q} index prior")
            self.trans = [z["trans"][q].astype(np.float32) for q in range(N_Q)]
            self.log_u = [np.log(z["unigram"][q]).astype(np.float32) for q in range(N_Q)]
            self.prior_crc = int(z["files_crc32"]) if "files_crc32" in z.files else None
        self.reset()

    def reset(self):
        """Start a new utterance: the next frame uses the unigram prior."""
        self._post = [None] * N_Q

    def encode(self, codes):
        """[3, T] RVQ indices -> [T, 40] uint8 bits."""
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape[0] != N_Q:
            raise ValueError(f"expected {N_Q} RVQ layers, got {codes.shape[0]}")
        return np.concatenate([L["cw"][codes[q]] for q, L in enumerate(self.layers)], axis=1)

    def loglik(self, llr, q):
        """llr [..., 40] -> log-likelihood of every codeword of layer q, [..., 1024],
        up to a constant: sum_l c_l L_l = 0.5 sum_l (2c_l - 1) L_l + const."""
        L = self.layers[q]
        seg = llr[..., L["off"]:L["off"] + L["n"]].astype(np.float32)
        return 0.5 * (seg @ L["x"].T)

    def decode(self, llr):
        """llr [T, 40], L > 0 means bit 1 -> MAP indices [3, T] int64.
        Continues the forward filter from the previous call."""
        if self.trans is None:
            raise RuntimeError("GolaySBSD was built without a prior; it cannot decode")
        llr = np.asarray(llr, dtype=np.float32)
        if llr.ndim != 2 or llr.shape[1] != BITS_PER_FRAME:
            raise ValueError(f"expected LLRs [T, {BITS_PER_FRAME}], got {llr.shape}")
        out = np.empty((N_Q, llr.shape[0]), dtype=np.int64)
        for q in range(N_Q):
            ll = self.loglik(llr[None], q)                            # [1, T, 1024]
            idx, self._post[q] = sbsd_filter(ll, self.trans[q], self.log_u[q],
                                             self._post[q])
            out[q] = idx[0]
        return out


def sbsd_filter(ll, trans, log_u, a=None):
    """Causal softbit source decoding of one layer (from the simulation).

    ll    [S, T, K] channel log-likelihoods of every index
    trans [K, K]    T(i, j) = P(i_t = j | i_{t-1} = i), rows sum to 1
    log_u [K]       log unigram, the prior of the first frame
    a     [S, K]    posterior of the previous frame, or None at the start

    Returns the MAP index of every frame [S, T] and the last posterior."""
    S, n_t, _ = ll.shape
    out = np.empty((S, n_t), dtype=np.int64)
    for t in range(n_t):
        lp = log_u if a is None else np.log(a @ trans + 1e-30)
        l = ll[:, t] + lp
        l -= l.max(axis=1, keepdims=True)
        out[:, t] = l.argmax(axis=1)
        a = np.exp(l)
        a /= a.sum(axis=1, keepdims=True)
    return out, a


def check_against_sim(sim_dir, prior_path, n_frames=600, chunk=75, seed=0):
    """Compare with sim/selftest_sim_v3.py: same codewords, and the same
    indices when one noisy utterance is decoded chunk by chunk here and in one
    pass in the simulation. Returns a short report; raises on a mismatch."""
    import sys
    sys.path.insert(0, sim_dir)
    import selftest_sim_v3 as sim

    rng = np.random.default_rng(seed)
    blk = sim.BlockCodes(list(PARITY))
    ours = GolaySBSD(prior_path)
    assert np.array_equal(ours.A, blk.layers[0]["A"]), "Golay matrix differs from the simulation"
    codes = rng.integers(0, CODEBOOK, (N_Q, n_frames))
    assert np.array_equal(ours.encode(codes), blk.encode(codes)), "codewords differ"

    z = np.load(prior_path)
    trans = [z["trans"][q].astype(np.float32) for q in range(N_Q)]
    log_u = [np.log(z["unigram"][q]).astype(np.float32) for q in range(N_Q)]
    report = []
    for snr in (-3.0, 0.0, 3.0):
        sigma = 10 ** (-snr / 20)
        y = 2.0 * ours.encode(codes) - 1.0 + sigma * rng.standard_normal((n_frames, BITS_PER_FRAME))
        llr = (2.0 * y / sigma ** 2).astype(np.float32)
        ref = np.stack([sim.sbsd_filter(blk.loglik(llr[None], q), trans[q], log_u[q])[0]
                        for q in range(N_Q)])
        ours.reset()
        got = np.concatenate([ours.decode(llr[t:t + chunk])
                              for t in range(0, n_frames, chunk)], axis=1)
        same = float((got == ref).mean())
        report.append(f"{snr:+.0f} dB: {same:.4f} of indices identical, "
                      f"accuracy {float((got == codes).mean()):.3f}")
        assert same == 1.0, f"SBSD differs from the simulation at {snr} dB"
    return report


if __name__ == "__main__":
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_dir = os.path.join(repo, "sim")
    for line in check_against_sim(sim_dir, os.path.join(sim_dir, "sbsd_prior_nq3.npz")):
        print(line)
    print("Golay + SBSD matches the simulation")
