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
    """bits [T, 40] uint8 -> (codes [n_q, T] int64, parity [parity_dim, T] uint8)."""
    bits = np.asarray(bits, dtype=np.uint8)
    T = bits.shape[0]
    codes = np.zeros((n_q, T), dtype=np.int64)
    for q in range(n_q):
        for b in range(BITS_PER_CODE):
            codes[q] = (codes[q] << 1) | bits[:, q * BITS_PER_CODE + b]
    parity = (bits[:, n_q * BITS_PER_CODE:].T.copy() if parity_dim
              else np.zeros((0, T), dtype=np.uint8))
    return codes, parity
