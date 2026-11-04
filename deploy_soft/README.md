# VoGEO deployment bundle — SOFT decision

The soft-decoding counterpart of `../deploy/`. Same link, same wire format, same
3000 bit/s; the receiver keeps the analog channel output instead of a sliced
bit, so it learns *how much* to trust each one.

    export_deploy.py    soft checkpoint -> bundle (run once, off-line)
    vogeo_frame.py      the wire format. Both ends must agree on this file.
    vogeo_llr.py        the radio boundary: soft samples -> LLRs   <-- NEW
    vogeo_tx.py         audio -> 40-bit frames        (transmitter Pi)
    vogeo_rx.py         LLRs -> corrected indices     (receiver Pi)
    selftest_v2.py      START HERE: one pass of the chain

## What differs from `../deploy/`

| | hard (`deploy/`) | soft (`deploy_soft/`) |
|---|---|---|
| transmitter | bits out | **identical** |
| radio delivers | sliced bits `[T,40] uint8` | soft samples, then LLRs `[T,40] float32` |
| RVQ index | `codebook[argmax]` | posterior average over all 1024 entries |
| decoder's 2nd input | one-hot at the received index | normalised log-posterior |
| parity feature | ±1 | `clamp(L/llr_clip, -1, 1)` |
| `copy_logit` at init | 4.0 | 1.0 |
| extra file | — | `vogeo_llr.py` |

Only the receiver changes. `vogeo_tx.py` is byte-identical in logic — bits are
what gets modulated either way.

Hard decoding is the **special case of soft where the posterior has collapsed**:
feed a confident LLR and the posterior average reduces to the codebook lookup
exactly (verified at export, and again as check [1] below, max error 0.00e+00).
That is why soft can only match or beat hard at the same Es/N0.

## Build a bundle

    python export_deploy.py --ckpt ../outputs_v8/vogeo_v8_enc20_par20_awgn_soft.pt \
                            --out_dir ./bundle

Produces `parity_encoder.ts`, `code_decoder.ts`, `codebooks.npy`, `manifest.json`.
It refuses a hard-trained checkpoint, because the two decoders take different
second arguments and would otherwise be silently interchangeable.

Ship to the **TX Pi**: `parity_encoder.ts`, `manifest.json`, `vogeo_frame.py`,
`vogeo_tx.py`.
Ship to the **RX Pi**: `code_decoder.ts`, `codebooks.npy`, `manifest.json`,
`vogeo_frame.py`, `vogeo_llr.py`, `vogeo_rx.py`.

## Test it before the radio

    python selftest_v2.py

It runs the chain once at Es/N0 = 4.32 dB — the point where BPSK gives BER =
0.01, the same operating point `../deploy/selftest_v2.py` uses, so the two
scripts' numbers compare directly. It also multiplies the received samples by
an arbitrary gain first, because a real radio's output is not calibrated, and
then recovers the scale blind.

"with us" must beat "without us". If it does not, check step 5 first: the blind
Es/N0 estimate should land within a few tenths of a dB of the truth.

## Where the radio goes

    [TX Pi]  audio -> EnCodec encoder -> z, codes -> VogeoTx -> bits [T,40]
                                                     |
                                          ZMQ -> GNU Radio: packetize,
                                                 BPSK, USRP
                                                     |
    [RX Pi]  soft samples [T,40] <- ZMQ <- GNU Radio: sync, matched filter
             -> vogeo_llr -> LLRs -> VogeoRx -> corrected codes
             -> EnCodec decoder -> audio

**Do not slice.** The one change the radio team has to make versus the hard
scheme is to send the matched-filter output across ZMQ as `float32` instead of
packed bits. That is 32× the bytes on the ZMQ hop — 480 kB/s at 3000 bit/s,
which is nothing over loopback or Ethernet, and it never touches the air.

## The four things that silently break it

1. **LLR scale.** The new one, and the most likely. Everything the model does
   assumes the *true* LLR, `L = 2*a*y/sigma^2`. GNU Radio's output is not
   calibrated — AGC, filter gain and normalisation all scale it arbitrarily.
   10× too large and the receiver believes every bit; 10× too small and it
   believes none. Neither raises an error. Measured on this bundle, a 10× scale
   error changes ~9% of the decoded indices.
   Use `vogeo_llr.llr_from_samples(y)`: it recovers amplitude and noise
   variance blind, by moments (real-BPSK M2M4), needing no pilots and no
   decisions. Measured recovery error over −5…+8 dB and gains 0.05…3.7: under
   6% on the LLR, under 0.05 dB on Es/N0. Pass a measured amplitude and noise
   variance instead if the radio can supply them.
2. **Bit order.** Code words are MSB first, and `llr > 0` means the bit is 1.
   Reversed, either way, it decodes to unrelated codebook entries: no error,
   just wrong audio.
3. **Streaming context.** Both networks are causal with a lookback (see the
   manifest). Calling them per block without carrying the tail loses history and
   quietly degrades quality. `VogeoRx` carries it for you — verified that
   50-frame blocks give bit-identical output to one shot.
4. **Mismatched halves.** The parity encoder and decoder are co-trained; the
   parity bits mean only what that decoder learned. Both files carry the same
   `run_id`. A soft bundle's `run_id` is *not* the hard bundle's, even for the
   same config — they are different training runs.

## Frame sync is the radio team's hardest job

There is no FEC on this stream by design. A one-bit slip in frame alignment
desynchronises the 10-bit code words and destroys the audio — far worse than a
high BER. Put a sync word on a superframe and protect that separately.

Target Es/N0 inside **−5 to +8 dB**, the range the model was trained over
(`trained_snr_range_db` in the manifest). Outside it, behaviour is
extrapolation. `vogeo_llr.esn0_db(y)` gives you the operating point in one call
— worth logging.

## Cost

The soft receiver does strictly more work than the hard one: a 1024-way
posterior per RVQ layer per frame, then a `[T,1024] @ [1024,128]` matmul for the
MMSE latent, where hard does a single row lookup. Measured here with 4 threads:

    enc20_par20, 1 s blocks : 3.8 ms per second of audio  (266x real time)
    enc30_par10, 1 s blocks : 3.1 ms per second of audio  (326x real time)

A Raspberry Pi is perhaps 10–30× slower than this box, which still leaves an
order of magnitude of headroom. Measure it on the actual Pi before committing.

## This package never runs EnCodec

`VogeoTx.encode(z, codes)` takes what your EnCodec already produces;
`VogeoRx.decode(llr)` gives back corrected RVQ indices for you to feed to
`quantizer.decode()` + `decoder()`. The bundle ships the RVQ codebooks
(`codebooks.npy`, ~1 MB) so the receiver can dequantize without importing
EnCodec at all — and the soft receiver needs the *full* codebook, not just the
selected row, since the MMSE latent is an average over all 1024 entries.

Two EnCodec details that will trip you up:

* `model.encode()` returns **codes only**. The parity encoder needs the
  continuous latent, which only `model.encoder(wav)` gives you.
* Pass `manifest["quantizer_bandwidth_kbps"]` straight to
  `model.quantizer(z, frame_rate, bw)`. `set_target_bandwidth()` accepts only
  EnCodec's official list, which has no entry for n_q of 1 or 3 — and
  `enc30_par10` is n_q = 3.

## EnCodec is not block-invariant (measured)

EnCodec 24 kHz contains an **LSTM in both encoder and decoder** (`SLSTM`). Its
state runs the length of the utterance, so chunked processing never matches
one-shot no matter how much overlap you add: measured at 1 s blocks, codes
agreed with one-shot only **88.7%** of the time, and adding up to 64 frames of
audio context did not help at all.

That is a property of EnCodec, not of this package — our two networks are plain
causal convnets and their context carry is exact. But it means **whoever chunks
the audio owns this problem**: either carry the LSTM state across blocks, or use
long enough blocks that the boundary cost is acceptable, and measure it.
