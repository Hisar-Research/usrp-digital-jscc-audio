"""Wire format. Identical to deploy/vogeo_frame.py plus unpack_llrs().

The frame layout is the same in both schemes -- soft decoding changes only what
the RECEIVER does with what arrives, never what the transmitter sends.
"""
import numpy as np

BITS_PER_CODE = 10
TOTAL_BITS_PER_FRAME = 40


def pack_frames(codes, parity, n_q, parity_dim):
    """codes [n_q, T] int, parity [parity_dim, T] uint8 -> bits [T, 40] uint8."""
    T = codes.shape[1]
    bits = np.zeros((T, TOTAL_BITS_PER_FRAME), dtype=np.uint8)
    for q in range(n_q):
        for b in range(BITS_PER_CODE):
            bits[:, q * BITS_PER_CODE + b] = (codes[q] >> (BITS_PER_CODE - 1 - b)) & 1
    if parity_dim:
        bits[:, n_q * BITS_PER_CODE:] = parity.T
    return bits


def unpack_frames(bits, n_q, parity_dim):
    """bits [T, 40] uint8 -> (codes [n_q, T] int64, parity [parity_dim, T] uint8).

    Still here because the hard-decision indices are the reference metric even
    in the soft receiver: slice the LLRs at zero and unpack, and you have what
    the link would have delivered with no model at all.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    T = bits.shape[0]
    codes = np.zeros((n_q, T), dtype=np.int64)
    for q in range(n_q):
        for b in range(BITS_PER_CODE):
            codes[q] = (codes[q] << 1) | bits[:, q * BITS_PER_CODE + b]
    parity = (bits[:, n_q * BITS_PER_CODE:].T.copy() if parity_dim
              else np.zeros((0, T), dtype=np.uint8))
    return codes, parity


def unpack_llrs(llr, n_q, parity_dim):
    """llr [T, 40] float -> (code_llr [n_q, T, 10], parity_llr [parity_dim, T]).

    Same field order as pack_frames: n_q code words MSB first, then the parity
    bits. SIGN CONVENTION: llr > 0 means the bit is more likely to be 1. Get
    this backwards and the receiver decodes confident nonsense with no error
    anywhere -- see check [1] in selftest_v2.py.
    """
    llr = np.asarray(llr, dtype=np.float32)
    T = llr.shape[0]
    code = llr[:, :n_q * BITS_PER_CODE].reshape(T, n_q, BITS_PER_CODE)
    code = np.ascontiguousarray(np.transpose(code, (1, 0, 2)))    # [n_q, T, 10]
    parity = (np.ascontiguousarray(llr[:, n_q * BITS_PER_CODE:].T) if parity_dim
              else np.zeros((0, T), dtype=np.float32))
    return code, parity


def hard_slice(llr, n_q, parity_dim):
    """The no-model reference: threshold the LLRs at zero and unpack."""
    return unpack_frames((np.asarray(llr) > 0).astype(np.uint8), n_q, parity_dim)
