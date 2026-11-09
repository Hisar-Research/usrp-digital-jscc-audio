# Four-Method USRP Over-the-Air Audio Experiment

This directory implements the **OTA validation** of neural audio transmission using **USRP B210** radios, **GNU Radio**, Meta **EnCodec 24 kHz**, and a soft-input learned protection model (**VOGEO**). It includes a four-method BPSK super-packet transmitter, a receiver with conventional/learned decoders, a GNU Radio flowgraph, measurement CSVs, figures, and an **IQ + soft-metric + LLR capture path**.

[Back to the project overview](../README.md) · [Software simulation and exported weights](../sim/README.md)

> These are experimental implementations. The OTA scripts run the same four methods as the software simulation in `sim/`, and load the trained model, the learned TX/RX code and the SBSD prior from there. `sim/` has its own documentation; **do not edit or replace that README as part of an OTA documentation update.**

## 1. End-to-end signal path

```text
LibriTTS test-clean  -> resample/convert to mono 24 kHz -> 1-second chunks
                                          |
                                one EnCodec pass
                                          |
           +----------------+-------------+-------------+----------------+
           |                |             |             |                |
    Uncoded 3 kbps     Golay + SBSD   LDPC rate 3/4  Learned 30+10
     4 codebooks        3 codebooks    3 codebooks    3 codebooks
       375 B           + Golay(20,10)    376 B        + 10 bits/frame
                       375 B + 1 fill                    375 B
           +----------------+-------------+-------------+
                                          |
                             rotate segment order
                                          |
                            1,502-byte super-packet
                                          |
                              UDP 5000 -> GNU Radio
                                          |
                       BPSK / framing -> USRP TX -> RF -> USRP RX
                                          |
                         sync, timing + carrier recovery
                                          |
                            post-Costas complex symbols
                                          |
                       +------------------+-------------------+
                       |                                      |
                  UDP 5002                                  UDP 5003
            SPK1 + detection ID +                         IQF1 fragments
              12,016 soft float32                          complex64 symbols
                       |                                      |
                       +------------------+-------------------+
                                          |
                           Python RX: correlate by detection ID
                           + capture IQ / soft / estimated LLR
                           + split and decode 4 methods
                           + reconstruct WAV / PESQ / ESTOI
```

The transmitter also sends a **13-byte metadata header directly over localhost UDP 5002**, bypassing RF. This is an experiment-control side channel, not a fully standalone air-interface protocol.

## 2. Methods and payload budget

| Label in CSV/WAV | Simulation ID | Source representation | Protection/decoding | Transmitted payload |
|---|:---:|---|---|---:|
| `proposed` | M1 | EnCodec, 3 RVQ layers = 30 bits/frame | Learned 10-bit/frame parity; Gaussian-EM LLRs + soft VOGEO decoder | **3,000 bits = 375 B** |
| `golay` | M3.1 | EnCodec, 3 RVQ layers = 30 bits/frame | (20,10,6) Golay code on layer 1, layers 2 and 3 uncoded (40 bits/frame); Gaussian-EM LLRs + softbit source decoding (SBSD) | **3,000 bits + 8 fill bits = 376 B** |
| `ldpc` | M3.2 | EnCodec, 3 layers = 2,250 source bits/s | Custom systematic (3008,2256) IRA LDPC, 6 known zero padding bits; soft normalized min-sum | **3,008 bits = 376 B** |
| `3` | M2 | EnCodec, 4 RVQ layers = 40 bits/frame | No FEC; hard slicing | **3,000 bits = 375 B** |

The exact 1-second chunk assumptions are **24,000 samples, 75 EnCodec frames and 10 bits per RVQ index**. All methods use approximately the same *per-method* channel-bit budget. Including four variants in one experimental RF super-packet gives **1,502 B = 12,016 bits** per chunk before physical-layer preamble/framing overhead. This aggregate must not be confused with a single method's nominal 3-kbps budget.

All four methods match the simulation (`sim/selftest_sim_v3.py`):

- **Proposed (M1):** the OTA scripts import `vogeo_tx.py` and `vogeo_rx.py` from `../sim/` and load `../sim/bundle/` (`n_q=3`, `parity_dim=10`, `decision_mode=soft`). The `../reference_soft/bundle_soft/` directory is a different 2-layer + 20-parity model and is **not interchangeable**.
- **Golay + SBSD (M3.1):** [`golay_sbsd.py`](golay_sbsd.py) builds the same code as the simulation's `--blk_parity 10 0 0` and uses its Markov index prior, `../sim/sbsd_prior_nq3.npz`. `python golay_sbsd.py` checks it against the simulation. One zero fill byte keeps the segment at 376 B, the size of the convolutional segment it replaced, so the super-packet and the GNU Radio flowgraph are unchanged.
- **LDPC (M3.2):** the graph, encoder and decoder were checked bit for bit against the simulation. The construction is **custom/deterministic**, not a standardized 5G LDPC code, and both ends must use exactly the same parity-check graph. The RX uses normalized min-sum (maximum 40 iterations, normalization factor 0.8).
- **Uncoded (M2):** the 3,000 bits are sent frame by frame (75 frames × 4 layers × 10 bits, MSB first), the same order as the simulation.

One difference remains, for every method: the OTA chain encodes each 1-second chunk with EnCodec separately, while the simulation encodes each utterance in one pass.

> **Earlier measurements.** Runs before this change used a convolutional code as M3.1 (`conv`: terminated K=7, (133,171) octal, punctured to rate 3/4, soft Viterbi) and sent the uncoded bits codebook by codebook. The CSVs in `measurement_30 oct/` and `measurement_before_30 oct/` and the audio demo come from that version, and `old/` keeps a snapshot of its code. The current receiver rejects packets from that transmitter: the metadata header type changed from 5 to 6.

## 3. Relevant files

| File | Role |
|---|---|
| [`tx_four_methods_with_ldpc_iq_capture.py`](tx_four_methods_with_ldpc_iq_capture.py) | Prepare four bitstreams, send metadata and RF super-packets; save a CSV of **expected transmitted** packet IDs |
| [`rx_four_methods_soft_ldpc_iq_llr_capture.py`](rx_four_methods_soft_ldpc_iq_llr_capture.py) | Decode all four methods; record detected-packet complex symbols, original soft metrics and estimated LLRs; reconstruct WAV and calculate PESQ/ESTOI |
| [`golay_sbsd.py`](golay_sbsd.py) | Golay + SBSD (M3.1) encoder and streaming decoder, shared by TX and RX; run it to check it against the simulation |
| [`bpsk_jscc_RX_iq_capture.grc`](bpsk_jscc_RX_iq_capture.grc) | GNU Radio transmit/receive flowgraph, synchronization, soft output and IQ-fragment output |
| [`calculate_sisdr.py`](calculate_sisdr.py) | SI-SDR of every received WAV against the original, after a sweep |
| [`graphs.py`](graphs.py) | Plot the new sweep's PESQ/ESTOI/SI-SDR against TX gain, with the gain=89 run as ceilings |
| `measurement_30 oct/` | Archived measurement CSVs, made with the convolutional M3.1 (`conv`); not plotted |
| `measurement_before_30 oct/` | Older archived CSVs; not plotted |
| [`../sim/`](../sim/) | Learned soft TX/RX code, trained bundle and SBSD prior, shared with the simulation |
| [`../docs/`](../docs/) | Experiment image, architecture and audio listening examples |

### Paths

Both scripts find the repository root from their own location. They load the model, the learned TX/RX code and the SBSD prior from `../sim/`, and the transmitter looks for LibriTTS under `datasets/` at the repository root (or, for older layouts, `ota/datasets/`). No links or path edits are needed. `../reference_soft/` is no longer used by the OTA scripts.

## 4. Hardware and software prerequisites

- **Hardware:** two configured USRP B210 devices with suitable antennas, cabling, clocking and power; a host computer capable of running GNU Radio/UHD and the Python DSP/model pipeline.
- **GNU Radio + UHD:** install compatible versions on the host and verify the devices using `uhd_find_devices` or `uhd_usrp_probe`. Open `bpsk_jscc_RX_iq_capture.grc` in GNU Radio Companion and verify device addresses and network endpoints.
- **Python:** an environment with `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `torchaudio`, `soundfile`, `encodec`, `pystoi`, and `pesq` as relevant to scripts you run. The model exports depend on compatible PyTorch/TorchScript support. GNU Radio Python modules are normally provided by the GNU Radio installation.
- **Audio:** LibriTTS **`test-clean`** from [OpenSLR 60](https://www.openslr.org/60/). The transmitter selects up to the first **100** valid WAVs (sorted order), with **8.0 ≤ original duration < 15.0 seconds**, and converts them to the EnCodec 24-kHz format.
- **Model and SBSD prior:** `sim/bundle/` and `sim/sbsd_prior_nq3.npz`, as supplied in the repository.

For example, keep the dataset under this root-relative layout:

```text
repo/
├── datasets/
│   └── LibriTTS/
│       └── test-clean/
│           └── ... .wav
├── sim/
│   ├── bundle/
│   └── sbsd_prior_nq3.npz
└── ota/
```

The scripts also recognize certain nested `LibriTTS/LibriTTS/test-clean/` layouts. The dataset is not part of the Git checkout. The EnCodec model may require a weight download on first execution.

**Radio settings:** the provided `.grc` snapshot uses **868 MHz** as the default center frequency and **1 MS/s** sample rate, with RX AGC enabled. These are example hardware settings, not universally correct or authorized operating conditions. Check the local RF regulations, antennas, actual gain controls, sample-rate compatibility, and device selection before enabling transmission. The TX gain is configured in the GNU Radio flowgraph, while `--tx-gain` below labels the receiver outputs.

## 5. Running an OTA measurement

The following assumes the correct model has loaded, GNU Radio sees the radios, and commands are run with `ota/` as the working directory (important because result paths are relative to the shell's working directory).

**Terminal 1 — start the Python receiver first:**

```bash
cd ota
python3 rx_four_methods_soft_ldpc_iq_llr_capture.py --tx-gain 31
```

Optional hardware-derived learned-decoder LLR clipping:

```bash
python3 rx_four_methods_soft_ldpc_iq_llr_capture.py \
    --tx-gain 31 --vogeo-llr-clip 50
```

Here `31` is an **example gain label**, which must match the actual **USRP TX gain** selected in the flowgraph; it is not an assumed channel SNR. The default LLR clip is ±50, and it applies to both the proposed and the Golay + SBSD decoders.

**Terminal 2 — start the GNU Radio flowgraph:**

Open `bpsk_jscc_RX_iq_capture.grc`, verify B210 device addresses, RF settings, packet length and UDP ports, then generate/start the flowgraph in GNU Radio Companion. Verify receiver synchronization before sending audio.

**Terminal 3 — start the Python transmitter:**

```bash
cd ota
python3 tx_four_methods_with_ldpc_iq_capture.py
```

The TX script has no `--tx-gain` CLI argument; set the hardware transmit gain in GNU Radio. The current timing constants in the transmitter are **1 second between chunks** and **3 seconds between audio files**. They are source-code constants, not command-line flags.

The receiver should be running before transmitting so that both the direct metadata and RF-detected packets can be received. Use `Ctrl+C` to stop the run; stopping RX also attempts to finalize the audio state it knows about. A detector log is **not** proof that every expected packet was delivered; compare against the TX manifest after capture.

### A full TX-gain sweep, from zero

A sweep is **7 TX gains plus one reference run at 89 dB**, the maximum B210 TX gain. Each run sends the same 100 utterances. One run takes about 25 minutes (≈11 s of speech per utterance, sent at one chunk per second, plus 3 s between utterances), so a sweep takes about 3.5 hours.

1. **Start from an empty `ota/`.** The receiver *appends* to its CSVs, so stale files mix into the new sweep. Move the outputs of any earlier run out of `ota/`:

   ```bash
   cd ota
   mkdir -p ../ota_runs_old
   mv original_audio received_audio_superpacket_soft ota_captures \
      pesq_results_superpacket_soft_txgain.csv estoi_results_superpacket_soft_txgain.csv \
      sisdr_results_superpacket_soft_txgain.csv figures ../ota_runs_old/ 2>/dev/null
   ```

   The archived tables in `measurement_30 oct/` and `measurement_before_30 oct/` can stay; nothing reads them.

2. **Choose the 7 gains.** The earlier sweep used 27, 29, …, 39 dB, but the right range depends on the antennas, distance and cabling. Keep the setup fixed for the whole sweep. A quick check: run one gain with the transmitter stopped after a few utterances (`Ctrl+C`). Aim for the lowest gain to be near the bottom of the curves, where uncoded ESTOI is about 0.1, and the highest gain close to the ceiling.

3. **For each gain** (for example 27, 29, 31, 33, 35, 37, 39, then 89):
   1. Set the TX gain in the flowgraph and start it.
   2. Start the receiver with the same label: `python3 rx_four_methods_soft_ldpc_iq_llr_capture.py --tx-gain 27`.
   3. Start the transmitter: `python3 tx_four_methods_with_ldpc_iq_capture.py`.
   4. When the transmitter exits after utterance 99, wait a few seconds, then stop the receiver with `Ctrl+C` so the last utterance is written.
   5. Check that the PESQ and ESTOI CSVs gained 400 rows (100 utterances × 4 methods) for this gain.

   **Run each gain once.** If a run goes wrong, delete its rows (the `tx_gain_db` column) from both CSVs and its `received_audio_superpacket_soft/audio_gain_<g>/` folder before repeating it; `graphs.py` refuses to plot duplicated measurements.

4. **Score and plot** (§9):

   ```bash
   python3 calculate_sisdr.py
   python3 graphs.py
   ```

## 6. RF payload and UDP protocol

Each chunk is represented by **one super-packet** with the four method segments in a cyclic order based on `chunk_id % 4`:

```text
chunk 0: plain -> golay -> ldpc  -> vogeo
chunk 1: golay -> ldpc  -> vogeo -> plain
chunk 2: ldpc  -> vogeo -> plain -> golay
chunk 3: vogeo -> plain -> golay -> ldpc
```

The receiver reconstructs the order from the associated metadata, not from an extra RF tag. The total payload is **12,016 RF data bits**; GNU Radio preamble and framing are additional.

| Connection | UDP destination | Message |
|---|---|---|
| Python TX → GNU Radio | `127.0.0.1:5000` | 1,502-byte super-packet |
| Python TX → Python RX | `127.0.0.1:5002` | 13-byte direct bookkeeping header, **not transmitted over RF** |
| GNU Radio → Python RX | `127.0.0.1:5002` | `SPK1` + network-order `uint32` detection ID + 12,016 little-endian `float32` demodulator metrics (48,072 bytes total) |
| GNU Radio → Python RX | `127.0.0.1:5003` | Fragmented `IQF1` messages with the same detection ID and post-Costas complex64 symbol data |

Direct metadata uses `struct` format **`!2sBIHHH`** (13 bytes): `VQ` magic, type `6` (type `5` was the earlier convolutional-code layout and is rejected), `audio_id` (uint32), `chunk_id` (uint16, zero-based), `total_chunks` (uint16), and `valid_samples` (uint16). Chunk IDs and audio IDs originate from the transmitter.

The **two RX UDP sockets** serve different streams: soft metrics/headers on 5002 and IQ fragments on 5003. They are joined using the **GNU Radio detection ID**. The Python receiver does not require a third socket for LLRs: it computes LLRs locally from the soft metrics.

## 7. Soft decisions and saved signal representations

- **Uncoded (`3`):** received metric sign → EnCodec bits, frame by frame; no FEC.
- **Golay + SBSD (`golay`):** the same blind Gaussian-EM LLR estimate and clip as the learned decoder, fitted on the segment's 3,000 payload metrics (the 8 fill bits are ignored). SBSD then combines the codeword likelihoods with the Markov index prior and filters forward in time. The filter state carries over from chunk to chunk within an utterance and restarts at a new utterance or after a missing chunk, as the learned decoder's context does.
- **LDPC (`ldpc`):** soft normalized min-sum decoding using the matching custom matrix and known-zero padding bits.
- **Learned (`proposed`):** blind two-component, equal-variance Gaussian estimation on the received BPSK metrics; estimate per-bit LLRs and feed the soft-input neural decoder. The LLR sign convention is **positive → bit 1; negative → bit 0**. LLR magnitudes are clipped before model decoding.

The estimator uses the measured **demodulator soft values**, rather than converting a USRP gain setting into an artificial \(E_s/N_0\) or fixed noise power. The learned model also carries its own manifest-specific LLR convention: verify any re-decoding pipeline against that manifest and the OTA script.

### Capture files

The TX script writes a transmit-side expectation log:

```text
ota/ota_captures/tx_expected_packets_<timestamp>_<pid>.csv
```

The RX creates a separate per-run directory, for example:

```text
ota/ota_captures/audio_gain_31/run_<timestamp>_<pid>/
├── rx_detected_packets.csv
├── rx_missing_headers.csv
├── voice_000/
│   ├── packet_0001.bin
│   ├── packet_0001_soft.bin
│   ├── packet_0001_llr.bin
│   └── ...
├── voice_001/
│   └── ...
└── unmatched/
    └── detection_000123*.bin  # when no TX metadata is associated
```

(The exact run-directory name is generated by the receiver; inspect its startup message. If a filename already exists, an additional detection suffix is used to avoid overwriting it.)

| Capture suffix | Representation | Elements per complete detected packet | Bytes |
|---|---|---:|---:|
| `.bin` | **Post-Costas, post-timing-recovery** complex symbols, little-endian `complex64` | 12,016 | 96,128 |
| `_soft.bin` | Original signed BPSK demodulator metrics, little-endian `float32` | 12,016 | 48,064 |
| `_llr.bin` | Gaussian-EM estimated LLRs, little-endian `float32` | 12,016 | 48,064 |

**These `.bin` captures are not the raw ADC-rate USRP IQ stream.** They are payload-aligned symbols after receiver processing. Files are written only when matching soft and **all** IQ fragments for a detected packet have arrived; a failed LLR estimator can leave the IQ/soft files without a valid LLR file. Partial IQ fragments expire rather than being presented as complete samples.

The detection CSV includes `det_id`, tentative `audio_id` / zero-based `chunk_id`, relative paths to the `.bin` files, capture time, association status and LLR-estimation status. The TX expected-packet CSV lists `audio_id`, `chunk_id_zero_based` and the transmit timestamp. **An associated TX metadata header is tentative, not an RF-decoded packet identity:** if a header and RF packet are mismatched because of losses or timing, check detection order, transmitted manifest and original audio alignment before accepting the pairing.

**No RF detection means no received packet IQ or LLR to save.** The `rx_missing_headers.csv` records certain detected bookkeeping gaps, but the definitive loss audit should also compare the full TX expected-packet manifest with the RX detection table. Do not replace absent measurements with zeros and call them received samples.

To load one valid capture after a run:

```python
import numpy as np

iq = np.fromfile("packet_0001.bin", dtype="<c8")        # complex64
soft = np.fromfile("packet_0001_soft.bin", dtype="<f4") # float32
llr = np.fromfile("packet_0001_llr.bin", dtype="<f4")   # float32
assert iq.size == soft.size == llr.size == 12016
```

## 8. Reconstructed audio and metric CSVs

When running from `ota/`, the transmitter saves reference WAVs to `ota/original_audio/`. The receiver saves complete decoded utterances under:

```text
ota/received_audio_superpacket_soft/
└── audio_gain_31/
    ├── proposed/received_audio_0_proposed.wav
    ├── golay/received_audio_0_golay.wav
    ├── ldpc/received_audio_0_ldpc.wav
    └── 3/received_audio_0_3.wav
```

A missing chunk within a known utterance is represented by **silence of the expected duration** so that time alignment is maintained. This does not imply successful decoding of that chunk.

The current RX directly calculates **ESTOI** (extended STOI) and **wideband PESQ at 16 kHz**, appending to:

```text
ota/estoi_results_superpacket_soft_txgain.csv
ota/pesq_results_superpacket_soft_txgain.csv
```

Both have columns `method,audio_id,<metric>,tx_gain_db`. The RX script **does not calculate SI-SDR**. After the sweep, `calculate_sisdr.py` compares every WAV under `received_audio_superpacket_soft/audio_gain_*/` with `original_audio/` and writes `ota/sisdr_results_superpacket_soft_txgain.csv` (it rewrites the file each time).

## 9. Plotting the saved measurements

From the `ota/` directory:

```bash
python3 graphs.py
```

`graphs.py` reads the three CSVs of the new sweep from the current directory (`--results-dir` to change it) and writes `pesq_…`, `estoi_…` and `sisdr_vs_tx_gain_95ci_ceiling` as EPS, PDF and PNG into `figures/` (`--out-dir`). It does **not** read the archived tables in `measurement_30 oct/` or `measurement_before_30 oct/`. If the SI-SDR table is missing, it skips that figure.

For each figure it:

- Prints the number of utterances per gain and method, notes if there are not 7 regular gains, and stops if a (method, utterance, gain) appears twice.
- Plots one curve per method: Proposed (M1), Golay + SBSD (M3.1), IRA-LDPC (M3.2) and Uncoded (M2), with the colours of the simulation figures. Each point is the mean over utterances with a **95% Student-t confidence interval** (`t.ppf(0.975, n-1)`).
- Leaves the **gain=89** run off the x-axis and draws its means as horizontal ceilings: proposed for the 3-layer methods (M1/M3.1/M3.2), uncoded for M2. All three metrics, PESQ included, take their ceilings from this run.
- Fixes the ESTOI axis to −0.02…0.86 and fits the PESQ and SI-SDR axes to the data.

## 10. Reproducibility and troubleshooting

| Symptom | Check |
|---|---|
| `Missing .../sim/...` at startup | Run from a full checkout: the scripts load the model, learned TX/RX code and SBSD prior from `../sim/` (§3) |
| `DROP HEADER \| wrong packet type 5` | The transmitter is the earlier convolutional-code version; run the current `tx_four_methods_with_ldpc_iq_capture.py` |
| LibriTTS not found | Confirm `datasets/LibriTTS/test-clean/` contains `.wav` files and matches the OTA script's searched directory |
| No detected packets | UHD/B210 selection, sample rate, frequency, antenna/cable link, AGC/gain, GNU Radio sync and UDP 5000/5002/5003 |
| IQ capture missing for a detected packet | Inspect IQF1 fragment expiry/drop logs; complete IQ and soft packets are required to write a capture |
| RF/metadata mismatch | Match TX expectation and RX detection CSVs by timeline and detection order; side-channel metadata is not over-the-air verified |
| Output CSV entries missing | Metric libraries, original reference WAV paths, RX completion, invalid reconstructed audio and receiver logs |
| SI-SDR looks unusually poor | Verify sample alignment, original/received matching, scoring implementation and valid decoding; plotting cannot repair incorrect scores |
| `graphs.py` stops on duplicated measurements | A gain was run twice into the same CSV; delete the rows of the bad run (§5) |

**Version control:** keep new measurements and code changes on a separate branch, open a pull request for review, and **do not merge directly**. The simulation directory `sim/` and its contributor-maintained README should remain untouched unless its owner requests changes.

## License

The repository-level [Apache 2.0 license](../LICENSE) covers project code as stated there; third-party code, weights and speech datasets retain their applicable licenses. Consult the [main README](../README.md) for attribution and upstream links.
