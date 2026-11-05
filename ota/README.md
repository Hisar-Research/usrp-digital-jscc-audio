# Four-Method USRP Over-the-Air Audio Experiment

This directory implements the **OTA validation** of neural audio transmission using **USRP B210** radios, **GNU Radio**, Meta **EnCodec 24 kHz**, and a soft-input learned protection model (**VOGEO**). It includes a four-method BPSK super-packet transmitter, a receiver with conventional/learned decoders, a GNU Radio flowgraph, measurement CSVs, figures, and an **IQ + soft-metric + LLR capture path**.

[Back to the project overview](../README.md) · [Software simulation and exported weights](../deploy_soft/README.md)

> These are experimental implementations. Software simulation in `deploy_soft/` has its own comparison baselines and documentation; **do not edit or replace that README as part of an OTA documentation update.**

## 1. End-to-end signal path

```text
LibriTTS test-clean  -> resample/convert to mono 24 kHz -> 1-second chunks
                                          |
                                one EnCodec pass
                                          |
           +----------------+-------------+-------------+----------------+
           |                |             |             |                |
    Uncoded 3 kbps     Conv rate 3/4  LDPC rate 3/4  Learned 30+10    
     4 codebooks        3 codebooks    3 codebooks    3 codebooks
       375 B              376 B          376 B        + 10 bits/frame
                                                         375 B
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

| Label in CSV/WAV | Source representation | Protection/decoding | Transmitted payload |
|---|---|---|---:|
| `proposed` | EnCodec, 3 RVQ layers = 30 bits/frame | Learned 10-bit/frame parity; Gaussian-EM LLRs + soft VOGEO decoder | **3,000 bits = 375 B** |
| `conv` | EnCodec, 3 layers = 2,250 source bits/s | Terminated K=7 convolutional (133,171 octal), rate-1/2 mother, puncture `[1,1,1,0,0,1]`; soft Viterbi | **3,008 bits = 376 B** |
| `ldpc` | EnCodec, 3 layers = 2,250 source bits/s | Custom systematic (3008,2256) LDPC, 6 known zero padding bits; soft normalized min-sum | **3,008 bits = 376 B** |
| `3` | EnCodec, 4 RVQ layers = 40 bits/frame | No FEC; hard slicing | **3,000 bits = 375 B** |

The exact 1-second chunk assumptions are **24,000 samples, 75 EnCodec frames and 10 bits per RVQ index**. All methods use approximately the same *per-method* channel-bit budget. Including four variants in one experimental RF super-packet gives **1,502 B = 12,016 bits** per chunk before physical-layer preamble/framing overhead. This aggregate must not be confused with a single method's nominal 3-kbps budget.

The learned bundle must be the **soft, 3-layer + 10-parity configuration**: `../reference_soft/bundle_3_out_4/manifest.json` (`n_q=3`, `parity_dim=10`, `decision_mode=soft`). The `bundle_soft/` directory is a different 2-layer + 20-parity model and is **not interchangeable**.

The LDPC construction is **custom/deterministic**, not a standardized 5G LDPC code. Both ends must use exactly the same parity-check graph. The RX uses normalized min-sum (default maximum 40 iterations, normalization factor 0.8).

## 3. Relevant files

| File | Role |
|---|---|
| [`tx_four_methods_with_ldpc_iq_capture.py`](tx_four_methods_with_ldpc_iq_capture.py) | Prepare four bitstreams, send metadata and RF super-packets; save a CSV of **expected transmitted** packet IDs |
| [`rx_four_methods_soft_ldpc_iq_llr_capture.py`](rx_four_methods_soft_ldpc_iq_llr_capture.py) | Decode all four methods; record detected-packet complex symbols, original soft metrics and estimated LLRs; reconstruct WAV and calculate PESQ/ESTOI |
| [`bpsk_jscc_RX_iq_capture.grc`](bpsk_jscc_RX_iq_capture.grc) | GNU Radio transmit/receive flowgraph, synchronization, soft output and IQ-fragment output |
| [`graphs.py`](graphs.py) | Plot PESQ/ESTOI/SI-SDR CSVs against TX gain, including the separate gain=89 reference |
| `measurement_30 oct/` | Available later measurement CSVs (including four-method data) |
| `measurement_before_30 oct/` | Earlier saved CSVs for comparison/history |
| [`../reference_soft/`](../reference_soft/) | Learned soft TX/RX code and exported models |
| [`../docs/`](../docs/) | Experiment image, architecture and audio listening examples |

### Path-layout issue in this snapshot

**Read this before running.** Both OTA Python scripts currently contain:

```python
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SOFT_REF_DIR = os.path.join(PROJECT_DIR, "reference_soft")
```

Because the scripts live in `ota/` but the tracked `reference_soft/` directory is in the **repository root**, these paths will not resolve in an unchanged checkout. The transmitter similarly searches for `datasets/` relative to `ota/`. The scripts must be made consistent with the directory layout.

**Non-invasive local workaround, from the repository root:**

```bash
# First place LibriTTS under datasets/ (see below).
ln -s ../reference_soft ota/reference_soft
ln -s ../datasets ota/datasets
```

Run these only if the target paths do not already exist. This leaves the existing Python files unchanged and makes their current path assumptions work on Unix-like systems. **For a permanent code fix**, update both scripts' project-root resolution consistently and test their input/output paths before merging the PR. The README does not silently claim that the supplied code is plug-and-play without this adjustment.

## 4. Hardware and software prerequisites

- **Hardware:** two configured USRP B210 devices with suitable antennas, cabling, clocking and power; a host computer capable of running GNU Radio/UHD and the Python DSP/model pipeline.
- **GNU Radio + UHD:** install compatible versions on the host and verify the devices using `uhd_find_devices` or `uhd_usrp_probe`. Open `bpsk_jscc_RX_iq_capture.grc` in GNU Radio Companion and verify device addresses and network endpoints.
- **Python:** an environment with `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `torchaudio`, `soundfile`, `encodec`, `pystoi`, and `pesq` as relevant to scripts you run. The model exports depend on compatible PyTorch/TorchScript support. GNU Radio Python modules are normally provided by the GNU Radio installation.
- **Audio:** LibriTTS **`test-clean`** from [OpenSLR 60](https://www.openslr.org/60/). The transmitter selects up to the first **100** valid WAVs (sorted order), with **8.0 ≤ original duration < 15.0 seconds**, and converts them to the EnCodec 24-kHz format.
- **Model:** the `reference_soft/bundle_3_out_4/` soft model directory and its manifest/weights, as supplied in the repository.

For example, keep the dataset under this root-relative layout:

```text
repo/
├── datasets/
│   └── LibriTTS/
│       └── test-clean/
│           └── ... .wav
├── reference_soft/
│   └── bundle_3_out_4/
└── ota/
```

The scripts also recognize certain nested `LibriTTS/LibriTTS/test-clean/` layouts. The dataset is not part of the Git checkout. The EnCodec model may require a weight download on first execution.

**Radio settings:** the provided `.grc` snapshot uses **868 MHz** as the default center frequency and **1 MS/s** sample rate, with RX AGC enabled. These are example hardware settings, not universally correct or authorized operating conditions. Check the local RF regulations, antennas, actual gain controls, sample-rate compatibility, and device selection before enabling transmission. The TX gain is configured in the GNU Radio flowgraph, while `--tx-gain` below labels the receiver outputs.

## 5. Running an OTA measurement

The following assumes the layout workaround above is in place, the correct model has loaded, GNU Radio sees the radios, and commands are run with `ota/` as the working directory (important because result paths are relative to the shell's working directory).

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

Here `31` is an **example gain label**, which must match the actual **USRP TX gain** selected in the flowgraph; it is not an assumed channel SNR. The default LLR clip is ±50.

**Terminal 2 — start the GNU Radio flowgraph:**

Open `bpsk_jscc_RX_iq_capture.grc`, verify B210 device addresses, RF settings, packet length and UDP ports, then generate/start the flowgraph in GNU Radio Companion. Verify receiver synchronization before sending audio.

**Terminal 3 — start the Python transmitter:**

```bash
cd ota
python3 tx_four_methods_with_ldpc_iq_capture.py
```

The TX script has no `--tx-gain` CLI argument; set the hardware transmit gain in GNU Radio. The current timing constants in the transmitter are **1 second between chunks** and **3 seconds between audio files**. They are source-code constants, not command-line flags.

The receiver should be running before transmitting so that both the direct metadata and RF-detected packets can be received. Use `Ctrl+C` to stop the run; stopping RX also attempts to finalize the audio state it knows about. A detector log is **not** proof that every expected packet was delivered; compare against the TX manifest after capture.

## 6. RF payload and UDP protocol

Each chunk is represented by **one super-packet** with the four method segments in a cyclic order based on `chunk_id % 4`:

```text
chunk 0: plain -> conv -> ldpc -> vogeo
chunk 1: conv  -> ldpc -> vogeo -> plain
chunk 2: ldpc  -> vogeo -> plain -> conv
chunk 3: vogeo -> plain -> conv  -> ldpc
```

The receiver reconstructs the order from the associated metadata, not from an extra RF tag. The total payload is **12,016 RF data bits**; GNU Radio preamble and framing are additional.

| Connection | UDP destination | Message |
|---|---|---|
| Python TX → GNU Radio | `127.0.0.1:5000` | 1,502-byte super-packet |
| Python TX → Python RX | `127.0.0.1:5002` | 13-byte direct bookkeeping header, **not transmitted over RF** |
| GNU Radio → Python RX | `127.0.0.1:5002` | `SPK1` + network-order `uint32` detection ID + 12,016 little-endian `float32` demodulator metrics (48,072 bytes total) |
| GNU Radio → Python RX | `127.0.0.1:5003` | Fragmented `IQF1` messages with the same detection ID and post-Costas complex64 symbol data |

Direct metadata uses `struct` format **`!2sBIHHH`** (13 bytes): `VQ` magic, type `5`, `audio_id` (uint32), `chunk_id` (uint16, zero-based), `total_chunks` (uint16), and `valid_samples` (uint16). Chunk IDs and audio IDs originate from the transmitter.

The **two RX UDP sockets** serve different streams: soft metrics/headers on 5002 and IQ fragments on 5003. They are joined using the **GNU Radio detection ID**. The Python receiver does not require a third socket for LLRs: it computes LLRs locally from the soft metrics.

## 7. Soft decisions and saved signal representations

- **Uncoded (`3`):** received metric sign → EnCodec bits; no FEC.
- **Convolutional (`conv`):** soft-input Viterbi decoding, with erased punctured positions assigned zero reliability.
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
    ├── conv/received_audio_0_conv.wav
    ├── ldpc/received_audio_0_ldpc.wav
    └── 3/received_audio_0_3.wav
```

A missing chunk within a known utterance is represented by **silence of the expected duration** so that time alignment is maintained. This does not imply successful decoding of that chunk.

The current RX directly calculates **ESTOI** (extended STOI) and **wideband PESQ at 16 kHz**, appending to:

```text
ota/estoi_results_superpacket_soft_txgain.csv
ota/pesq_results_superpacket_soft_txgain.csv
```

Both have columns `method,audio_id,<metric>,tx_gain_db`. The RX script **does not calculate SI-SDR**; the supplied `measurement_30 oct/sisdr_results_superpacket_soft_txgain.csv` is a separately prepared result table. Before combining tables, verify that all metrics refer to the same audio IDs, run conditions, detected-packet handling and gain labels.

## 9. Plotting the saved measurements

From the `ota/` directory:

```bash
python3 graphs.py
```

At the top of `graphs.py`, `PESQ_CSV`, `ESTOI_CSV` and `SISDR_CSV` currently point to the three CSVs in `measurement_30 oct/`. Change those paths if you are using later runs or newly computed SI-SDR. The script writes EPS/PDF/PNG figures in the **current working directory**.

The plotting script:

- Shows curves for `proposed`, `3`, `conv` and `ldpc`.
- Excludes **gain=89** (and historical sentinel 100) from the ordinary TX-gain x-axis sweep; measured gain=89 data are used for separate horizontal ESTOI/SI-SDR references.
- Currently uses **95% Student-t confidence intervals** for regular curve means (`t.ppf(0.975, n-1)`), and output filenames contain `95ci`. **If your paper specifies 90% confidence intervals, update the quantile to `0.95`, variable names and figure filenames before publishing.** Do not label the present figures as 90% CI.
- Has **hard-coded PESQ ceiling constants** in `PESQ_CEILINGS`, rather than reading them automatically from gain=89 PESQ observations. Recheck these against the measurement dataset before submitting revised figures.

Treat the saved `measurement_30 oct/` files as a particular measurement snapshot; the directory structure in this archive does not establish that every subsequent (e.g. October 31) run or revised manuscript value has been committed.

## 10. Reproducibility and troubleshooting

| Symptom | Check |
|---|---|
| `reference_soft` files missing | Apply the root/`ota` path fix or temporary symlink described in §3 |
| LibriTTS not found | Confirm `datasets/LibriTTS/test-clean/` contains `.wav` files and matches the OTA script's searched directory |
| No detected packets | UHD/B210 selection, sample rate, frequency, antenna/cable link, AGC/gain, GNU Radio sync and UDP 5000/5002/5003 |
| IQ capture missing for a detected packet | Inspect IQF1 fragment expiry/drop logs; complete IQ and soft packets are required to write a capture |
| RF/metadata mismatch | Match TX expectation and RX detection CSVs by timeline and detection order; side-channel metadata is not over-the-air verified |
| Output CSV entries missing | Metric libraries, original reference WAV paths, RX completion, invalid reconstructed audio and receiver logs |
| SI-SDR looks unusually poor | Verify sample alignment, original/received matching, scoring implementation and valid decoding; plotting cannot repair incorrect scores |
| Plots disagree with paper | Confirm the correct measurement folder, reference gain=89 handling, PESQ ceiling constants and CI level |

**Version control:** keep new measurements and code changes on a separate branch, open a pull request for review, and **do not merge directly**. The simulation directory `deploy_soft/` and its contributor-maintained README should remain untouched unless its owner requests changes.

## License

The repository-level [Apache 2.0 license](../LICENSE) covers project code as stated there; third-party code, weights and speech datasets retain their applicable licenses. Consult the [main README](../README.md) for attribution and upstream links.
