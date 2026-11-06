# DeepJSCC: channel-robust neural speech coding at 3 kbps

Simulation code for a learned channel-protection scheme built on the
[EnCodec](https://github.com/facebookresearch/encodec) 24 kHz neural codec. The
scheme sends 3 RVQ layers (30 index bits per frame) plus 10 learned parity bits,
and a causal soft-input neural receiver corrects the indices before EnCodec
decodes them.

`selftest_sim_v3.py` compares this scheme with three baselines at 3.0 kbps, over
AWGN and block-Rayleigh fading, with matched and mismatched SNR knowledge at the
receiver. M1, M3.1 and M2 use exactly 40 bits/frame. The LDPC baseline uses the
same block format as our over-the-air experiments, which is 3008 bits per 75
frames, or 0.3% more.

The accompanying paper is not yet published. A citation will be added here when
it is.

## Compared methods

Every method takes soft channel input (LLRs) and makes a hard index decision per
frame.

| ID | Method | Payload per frame | Decoder | Latency |
|---|---|---|---|---|
| M1 | **Proposed** | 30 index bits + 10 learned parity bits | Causal CNN receiver, argmax | 13.3 ms |
| M3.1 | Golay + SBSD | 30 index bits + 10 parity bits: (20,10,6) shortened and punctured Golay code on layer 1, layers 2 and 3 uncoded | Softbit source decoding with a first-order Markov index prior, forward filtering, MAP | 13.3 ms |
| M3.2 | IRA-LDPC, rate 3/4 | 30 index bits, (3008,2256) code, one codeword per 75 frames | Normalized min-sum, alpha 0.8, up to 40 iterations | 1016 ms |
| M2 | Uncoded | 40 index bits (4 RVQ layers) | Slicer | 13.3 ms |

No method uses an interleaver. Frame synchronisation is assumed to be perfect.

M3.1 is called Golay + SBSD with `--blk_parity 10 0 0`, and Hamming + SBSD with
any other setting. The script default is `4 3 3`: a (14,10) shortened Hamming
code on layer 1 and (13,10) codes on layers 2 and 3. Golay + SBSD is the baseline
reported here, so the commands below pass `--blk_parity 10 0 0` explicitly. The
SBSD receiver is the same for both.

## Results

These results use all 100 test clips, 1 trial per clip, perfect SNR knowledge and
a Rayleigh coherence time of 6 frames (about 80 ms). M3.1 is Golay + SBSD, and
PESQ is wideband (P.862.2).

**ESTOI.** The 90% bootstrap CI half-widths are 0.012 or less.

| Es/N0 (dB) | AWGN M1 | AWGN M3.1 | AWGN M3.2 | AWGN M2 | Rayleigh M1 | Rayleigh M3.1 | Rayleigh M3.2 | Rayleigh M2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | **0.430** | 0.389 | 0.040 | 0.045 | **0.373** | 0.317 | 0.054 | 0.057 |
| -3 | **0.520** | 0.498 | 0.079 | 0.088 | **0.448** | 0.404 | 0.095 | 0.096 |
| 0 | **0.659** | 0.636 | 0.272 | 0.237 | **0.556** | 0.527 | 0.195 | 0.198 |
| +2 | 0.729 | 0.704 | **0.785** | 0.415 | **0.619** | 0.597 | 0.312 | 0.288 |
| +5 | 0.776 | 0.770 | **0.785** | 0.717 | **0.690** | 0.677 | 0.630 | 0.432 |
| +8 | 0.785 | 0.785 | 0.785 | **0.816** | 0.735 | 0.728 | **0.781** | 0.565 |

**Wideband PESQ.** The 90% bootstrap CI half-widths are 0.052 or less.

| Es/N0 (dB) | AWGN M1 | AWGN M3.1 | AWGN M3.2 | AWGN M2 | Rayleigh M1 | Rayleigh M3.1 | Rayleigh M3.2 | Rayleigh M2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -5 | **1.10** | 1.09 | 1.08 | 1.05 | **1.08** | 1.07 | 1.05 | 1.06 |
| -3 | **1.16** | 1.15 | 1.04 | 1.04 | **1.11** | 1.10 | 1.05 | 1.05 |
| 0 | **1.37** | 1.34 | 1.04 | 1.03 | **1.18** | 1.16 | 1.04 | 1.04 |
| +2 | 1.62 | 1.53 | **2.04** | 1.05 | **1.25** | 1.24 | 1.05 | 1.04 |
| +5 | 1.95 | 1.89 | **2.04** | 1.27 | **1.41** | 1.39 | 1.22 | 1.06 |
| +8 | 2.04 | 2.04 | 2.04 | **2.25** | 1.60 | 1.57 | **1.97** | 1.10 |

Error-free EnCodec ceilings over the same clips:

| Layers | Rate | ESTOI | PESQ (WB) |
|---|---|---:|---:|
| n_q = 3 (M1, M3.1, M3.2) | 2.25 kbps | 0.785 | 2.04 |
| n_q = 4 (M2) | 3.0 kbps | 0.819 | 2.34 |

## Repository contents

| Path | Purpose |
|---|---|
| `selftest_sim_v3.py` | Main simulation: all methods, channels, metrics and outputs |
| `plot_sim_v3.py` | Figures and summary tables from a finished run |
| `vogeo_tx.py` | Transmitter: EnCodec latent and indices to 40-bit frames |
| `vogeo_rx.py` | Soft receiver: LLRs to corrected indices |
| `vogeo_frame.py` | Frame format shared by both ends |
| `bundle/` | Trained model: `parity_encoder.ts`, `code_decoder.ts`, `codebooks.npy`, `manifest.json` |
| `sbsd_prior_nq3.npz` | Cached Markov index prior for M3.1 |
| `test_data/` | The 100 test clips |

Training code and checkpoints are not included. `bundle/` holds the exported
model used for the published numbers (`run_id` `fa03d421aa9837a8`), trained on
AWGN over -5 to +8 dB.

## Installation

The code was run on CPU with Python 3.12. No GPU is needed.

```bash
python -m venv venv
source venv/bin/activate
pip install torch==2.12.1 torchaudio==2.11.0 numpy==2.5.0 soundfile==0.14.0 \
            encodec==0.1.1 pystoi==0.4.1 pesq==0.0.4 matplotlib==3.11.0
```

`pesq` compiles a C extension, so it needs a C compiler. If it is missing, the
simulation still runs and leaves PESQ empty. `threadpoolctl` is optional; when
installed, it limits each worker to one BLAS thread.

The first run downloads the EnCodec weights through Torch Hub, so it needs
internet access.

## Usage

Run every command from the repository root, because the default paths are
relative.

### 1. Check the code

```bash
python selftest_sim_v3.py --selftest
```

This takes a few seconds and needs no audio. It checks the block codes, SBSD,
the fading statistics, the LLR scaling and an LDPC round trip.

### 2. Quick run

```bash
python selftest_sim_v3.py --test_dir test_data --seconds 0 --trials 1 --max_files 2 \
    --snrs 0 3 --channels awgn rayleigh --mismatch_db 0 --blk_parity 10 0 0 \
    --out_prefix smoke
python plot_sim_v3.py --prefix smoke --out_dir figs_smoke --fmt png
```

This took under a minute on our machine. At 0 dB the printed AWGN ESTOI values
should be close to M1 0.67, M3.1 0.64, M3.2 0.25 and M2 0.23.

### 3. Full sweep

```bash
python selftest_sim_v3.py --test_dir test_data --seconds 0 --trials 1 \
    --mismatch_db 0 --blk_parity 10 0 0 --pesq_mode wb --workers 12 \
    --out_prefix final_golay
python plot_sim_v3.py --prefix final_golay --out_dir figs_golay
```

This is the run behind the main results above: all 100 clips over AWGN and
Rayleigh, with Es/N0 from -5 to +8 dB in 1 dB steps. It took under 4 hours on our
machine. Set `--workers` to suit your CPU; each worker uses one core.

To add SNR mismatch of +3 and -3 dB, leave out `--mismatch_db`, which takes about
three times as long. For Hamming + SBSD, leave out `--blk_parity`. Use a different
`--out_prefix` for each variant. `plot_sim_v3.py` names M3.1 in the figure legends
from the setting used in the run.

An interrupted run continues from the last finished file if you add `--resume`
with the same settings.

Noise and fading come from fixed seeds, but EnCodec's encoder output can depend
on the thread count. `--workers 1` uses all cores, while larger values use one
thread per worker. In a check of 3 clips, moving from one thread to several
changed one index in 2 of them. On a 10-clip subset this moved the tabulated means
by about 0.001 in ESTOI and 0.005 in narrowband PESQ. To repeat a run exactly, use
the pinned versions and the same kind of `--workers` setting.

### 4. Listening samples

```bash
python selftest_sim_v3.py --test_dir test_data --seconds 0 --trials 1 \
    --channels rayleigh --mismatch_db 0 --snrs 0 --max_files 5 --blk_parity 10 0 0 \
    --save_audio audio_rayleigh_0dB --out_prefix listen
```

This writes decoded WAV files and an `index.html` page for side-by-side
listening. The clips use the same noise and fading as the sweep.

## Main options

| Option | Default | Meaning |
|---|---|---|
| `--test_dir` | none | Folder of `.wav` or `.flac` speech, searched recursively |
| `--wav` | none | Single file, as an alternative to `--test_dir` |
| `--seconds` | `5.0` | Audio used per file; `0` uses the whole file |
| `--trials` | `3` | Noise realisations per file |
| `--snrs` | `-5` to `8` | Es/N0 grid in dB |
| `--channels` | `awgn rayleigh` | Also accepts `rayleigh_nocsi` |
| `--mismatch_db` | `0 3 -3` | Assumed minus true Es/N0 at the receiver |
| `--coherence_frames` | `6` | Rayleigh AR(1) coherence in frames; `0` gives independent frames |
| `--blk_parity` | `4 3 3` | M3.1 parity bits per layer; must sum to 10. `10 0 0` gives Golay + SBSD, and anything else gives Hamming + SBSD |
| `--pesq_mode` | `nb` | `nb` (P.862) or `wb` (P.862.2); the main results use `wb` |
| `--workers` | `1` | Files processed in parallel |
| `--out_prefix` | `vogeo_sim_v3` | Prefix of the output files |
| `--max_files` | `0` | Use only the first N sorted files; `0` uses all |

`python selftest_sim_v3.py --help` lists every option. Note that `--seconds` and
`--trials` default to values different from the full sweep, so pass them
explicitly as shown above.

## Outputs

With `--out_prefix P`:

| File | Content |
|---|---|
| `P_rows.csv` | One row per file, trial, scenario, method and SNR |
| `P_ceil.csv` | Error-free codec ceilings per file and n_q |
| `P_summary.json` | Means per scenario, method and SNR, plus all settings |
| `P_config.json` | Run configuration, checked by `--resume` |
| `P_table.txt` | Summary tables, written by `plot_sim_v3.py` |

`plot_sim_v3.py` writes these figures as PDF and PNG:

- `fig_<scenario>`: PESQ, ESTOI, SI-SDR and codeword accuracy against Es/N0
- `overview_<metric>`: all methods, one panel per channel
- `mismatch_delta_<metric>`: paired change caused by SNR mismatch
- `fading_penalty_<metric>`: paired change from AWGN to Rayleigh

## Simulation details

**Frame format.** EnCodec runs at 75 frames/s, and each frame carries 40 bits,
giving 3000 bit/s. M1 frames hold the n_q index words, MSB first and layer 1
first, followed by the parity bits. An LLR above 0 means the bit is more likely
to be 1.

**Channels.** All methods use BPSK with unit symbol energy.

- `awgn`: y = x + n
- `rayleigh`: y = |h| x + n, with |h| constant over a frame of 40 symbols and
  correlated across frames by a complex Gaussian AR(1) process. E|h|^2 = 1, so
  Es/N0 is the average SNR. The LLR uses the true |h|.
- `rayleigh_nocsi`: the same fading, but the LLR uses E|h|. It is available as
  an option and is not part of the results above.

**SNR mismatch.** The receiver forms LLRs with
sigma_hat^2 = sigma^2 x 10^(-m/10), so m > 0 means it believes the SNR is m dB
higher than it really is. The slicer (M2) does not depend on a common LLR scale.
Neither does the min-sum decoder (M3.2), apart from the fixed floor on its known
zeros; scaling by +3 and -3 dB changed no decoded bits in 160 test blocks. M1 and
M3.1 do depend on it.

**LDPC baseline.** M3.2 uses the rate-3/4 IRA-LDPC code from our over-the-air
experiments
([usrp-digital-jscc-audio](https://github.com/Hisar-Research/usrp-digital-jscc-audio)).
It is a deterministic custom code, not a standardized one.

- H = [A | B]. In A, each information bit joins 3 checks and each check takes 9
  information bits. B is a dual-diagonal accumulator.
- Each codeword covers 75 frames: 2250 source bits, 6 known zeros and 752 parity
  bits, or 3008 bits in total. A short last chunk is filled with zeros that the
  receiver is not told about.
- The decoder holds the known zeros at an LLR of at least 50 and stops on a clean
  syndrome.
- The graph has 24 pairs of checks sharing two variables (4-cycles).
- On Rayleigh at +8 dB, 12.1% of blocks ended without a clean syndrome, while
  codeword accuracy was 0.997.

The simulation's graph, encoder and decoder were checked bit for bit against the
over-the-air scripts. The over-the-air system encodes each 1 s chunk with EnCodec
separately, while the simulation encodes each utterance in one pass.

**Paired comparisons.** Noise is drawn once per file, trial and method, and
fading once per file and trial. Both are reused across SNRs, mismatches and
channels, so differences between scenarios are paired.

**Metrics.** Audio is resampled to 16 kHz for ESTOI, PESQ and SI-SDR. PESQ is
narrowband by default, and the main results use wideband. In the figures and tables, a failed PESQ call counts as
1.0 instead of being dropped; none occurred in the runs reported above.
`plot_sim_v3.py` averages over trials within an utterance, then over utterances,
and reports utterance-level percentile bootstrap intervals (90% by default).

**Markov prior for M3.1.** `sbsd_prior_nq3.npz` was estimated from 2272
utterances (1,214,759 frames) of the LibriSpeech dev-clean training split, with
Dirichlet smoothing chosen on held-out files. Because the file is included, the
LibriSpeech audio is not needed. Rebuilding it requires your own speech
corpus through `--prior_dir` and `--rebuild_prior`.

## Data

`test_data/` contains 100 speech clips, 18.1 minutes in total. They are 24 kHz
mono 16-bit WAV files of 8.0 to 14.8 s, randomly selected from LibriTTS and
renamed `original_audio_<n>.wav`.

LibriTTS is derived from LibriSpeech, so the clips were checked against the
LibriSpeech dev-clean training and validation audio. Two 3 s snippets of each
clip were cross-correlated with all 2524 utterances at 4 kHz. The highest
normalised correlation was 0.34, against 1.00 for a positive control, so no
shared recording was found. This check covers recordings only; it does not rule
out speakers appearing in both sets.

## Licence

This code has no licence yet, and one will be chosen when the paper is
published. Until then, no permission to reuse or redistribute it is granted.

Third-party material has its own terms:

- **EnCodec:** this project depends on EnCodec, and `bundle/codebooks.npy`
  contains codebooks from its 24 kHz model. The PyPI package `encodec` 0.1.1
  used here declares CC BY-NC 4.0. The upstream repository's README states MIT
  for the code and gives no separate terms for the pretrained weights. Check the
  current terms before any commercial use.
- **LibriTTS:** the clips in `test_data/` are from a corpus released under
  CC BY 4.0.
- **LibriSpeech:** the training audio and the prior's source are from a corpus
  released under CC BY 4.0.

## References

If you use this code, please cite EnCodec:

```bibtex
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```

Other sources:

- H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen and Y. Wu,
  "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech," Interspeech
  2019. [arXiv:1904.02882](https://arxiv.org/abs/1904.02882)
- V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: An ASR corpus
  based on public domain audio books," ICASSP 2015.
- T. Fingscheidt and P. Vary, "Softbit speech decoding: a new approach to error
  concealment," IEEE Trans. Speech and Audio Processing, vol. 9, no. 3,
  pp. 240-251, Mar. 2001.
  [doi:10.1109/89.905998](https://doi.org/10.1109/89.905998)
