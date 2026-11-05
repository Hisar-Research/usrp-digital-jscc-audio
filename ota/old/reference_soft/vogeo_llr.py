"""The radio boundary: soft samples -> LLRs.

This file has no counterpart in deploy/. The hard receiver takes bits, and a
bit is a bit; the soft receiver takes LOG-LIKELIHOOD RATIOS, and an LLR is only
meaningful at the right SCALE. Everything the model does with them -- the
codeword posterior, the MMSE latent, the clipped parity feature -- assumes the
LLR is the true one:

    L = 2 * a * y / sigma^2

for a matched-filter output y = a*x + n, x = +-1, n ~ N(0, sigma^2). Hand it
an LLR that is 10x too large and the receiver believes every bit; 10x too small
and it believes none. Neither raises an error. This is the single most likely
way to integrate soft decoding incorrectly.

GNU Radio's demodulator output is NOT calibrated: AGC, filter gain and
normalisation all scale it arbitrarily. So either pass a measured amplitude and
noise variance, or let estimate_bpsk_params() recover them from the samples.
"""
import numpy as np


def estimate_bpsk_params(y):
    """Blind (a, sigma^2) from BPSK matched-filter samples, by moments.

    With y = a*x + n, x = +-1 and real Gaussian n, the signal has kurtosis 1
    (x^2 is always 1) and the noise 3, so with S = a^2 and N = sigma^2:

        m2 = S + N
        m4 = S^2 + 6*S*N + 3*N^2   ->   m4 = 3*m2^2 - 2*S^2

    which inverts in closed form to S = sqrt((3*m2^2 - m4)/2), N = m2 - S.
    This is the real-BPSK case of the M2M4 estimator (Pauluzzi & Beaulieu,
    IEEE Trans. Commun. 48(10), 2000); it needs no pilots and no decisions.

    A few thousand samples are plenty. Returns (a, sigma^2).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    m2 = np.mean(y ** 2)
    m4 = np.mean(y ** 4)
    S = np.sqrt(max(1.5 * m2 ** 2 - 0.5 * m4, 0.0))     # (3*m2^2 - m4)/2
    N = max(m2 - S, 1e-12)
    return float(np.sqrt(S)), float(N)


def esn0_db(y):
    """Es/N0 in dB from the same moments -- worth logging next to the LLRs.
    The model was trained over -5..+8 dB; outside that it is extrapolating.

    Note the 2: for real baseband BPSK the noise variance is N0/2, so
    Es/N0 = a^2 / (2*sigma^2). Dropping it reports 3.01 dB too much, and since
    the LLR itself is unaffected the error shows up only here -- as an estimator
    that looks broken when it is fine.
    """
    a, s2 = estimate_bpsk_params(y)
    return 10.0 * np.log10(max(a * a, 1e-30) / (2.0 * s2))


def llr_from_samples(y, amplitude=None, noise_var=None):
    """Soft samples [T, 40] -> LLRs [T, 40], positive meaning bit = 1.

    Pass amplitude and noise_var when the radio can measure them (from a pilot
    or a known preamble); otherwise they are estimated blind from y itself.
    Estimate over a long enough window that the statistics are stable -- a
    whole utterance is fine, a single frame is not.
    """
    y = np.asarray(y, dtype=np.float32)
    if amplitude is None or noise_var is None:
        a_hat, n_hat = estimate_bpsk_params(y)
        amplitude = a_hat if amplitude is None else amplitude
        noise_var = n_hat if noise_var is None else noise_var
    return (2.0 * amplitude * y / noise_var).astype(np.float32)
