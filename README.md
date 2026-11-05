# Robust Semantic Audio Communication: Simulation and SDR Testbed

Research code for **channel-robust neural speech transmission** using Meta's [EnCodec](https://github.com/facebookresearch/encodec) audio codec, learned protection bits, conventional forward-error correction (FEC), and a USRP/GNU Radio over-the-air (OTA) testbed.

The project has two complementary parts:

1. **Software evaluation:** controlled AWGN and fading-channel experiments with exported learned models, conventional baselines, and speech-quality metrics. The simulation implementation and its detailed instructions are maintained in [`deploy_soft/README.md`](deploy_soft/README.md).
2. **Over-the-air evaluation:** a digital BPSK link using USRP B210 radios and GNU Radio. The current experiment compares four methods in the same RF super-packet and can save detected-packet IQ, demodulator soft metrics, and estimated log-likelihood ratios (LLRs). See **[`ota/README.md`](ota/README.md)** for the setup and reproduction workflow.

> **Scope:** This is a research prototype, not a real-time voice-call application. The simulation and OTA experiments have different baselines and assumptions; their results should not be compared as though they use identical channels, synchronization, or receiver knowledge.

## System at a glance

```text
                         Speech / LibriTTS
                                |
                          EnCodec 24 kHz
                                |
             +------------------+------------------+
             |                                     |
       Software simulation                      OTA testbed
       deploy_soft/                             ota/
             |                                     |
      AWGN / Rayleigh                    Four bitstreams per chunk
      + channel models                   Uncoded | Conv | LDPC | Learned
             |                                     |
      Soft-input decoding                     RF super-packet
             |                                     |
      PESQ / ESTOI / ...                   GNU Radio BPSK TX
                                                   |
                                                USRP B210
                                                   |
                                                RF channel
                                                   |
                                                USRP B210
                                                   |
                                          GNU Radio soft RX
                                                   |
                                      Soft decoding and reconstruction
                                                   |
                                        PESQ / ESTOI; IQ/LLR capture
```

[OTA testbed block diagram](docs/OTA-jscc-testbed.drawio.png) · [Testbed photo](docs/setup_image.jpeg) · [Audio demonstration](docs/index.html)

## Current OTA transmission methods

All four methods operate on **1-second, 24-kHz audio chunks**, with 75 EnCodec frames per chunk and 10 bits per RVQ index. The methods are transmitted together for experimental comparison, not as four simultaneous independent user streams.

| CSV label | Method | EnCodec source | Protection | Payload per 1-s chunk |
|---|---|---|---|---:|
| `proposed` | Learned soft-input protection (VOGEO) | 3 RVQ codebooks, 2.25 kbps | 10 learned parity bits/frame | **3,000 bits (375 B)** |
| `conv` | Punctured convolutional FEC | 3 codebooks, 2.25 kbps | K=7, (133,171) octal, nominal rate 3/4; terminated | **3,008 bits (376 B)** |
| `ldpc` | Systematic LDPC FEC | 3 codebooks, 2.25 kbps | Custom (3008,2256) code; soft normalized min-sum decoder | **3,008 bits (376 B)** |
| `3` | Uncoded EnCodec | 4 codebooks, 3.0 kbps | None; hard decision on received soft values | **3,000 bits (375 B)** |

The **combined four-method super-packet** is **1,502 bytes / 12,016 payload bits** before GNU Radio framing and synchronization overhead. Method order rotates across chunks. The learned, convolutional, and LDPC receivers use soft information; the uncoded branch slices received values into bits.

The OTA learned model is the **30 source bits + 10 learned bits per frame** variant, using the compatible soft bundle in `reference_soft/bundle_3_out_4/`. Other bundles in this repository are separate model variants and must not be interchanged.

**Important:** USRP transmit gain is an **experimental gain setting**, **not** measured SNR or \(E_s/N_0\). The OTA plots use transmit gain on the x-axis. The simulation uses modeled \(E_s/N_0\) and must be labeled accordingly.

## Repository layout

```text
.
├── README.md                         # Overview and navigation (this file)
├── deploy_soft/                      # Current software simulation, assets, weights
│   ├── README.md                     # Maintained separately by its contributor
│   ├── selftest_sim_v3.py
│   ├── plot_sim_v3.py
│   ├── bundle/                       # Exported soft learned model
│   └── test_data/                    # Included example/evaluation speech clips
├── ota/
│   ├── README.md                     # Detailed USRP OTA setup and capture guide
│   ├── tx_four_methods_with_ldpc_iq_capture.py
│   ├── rx_four_methods_soft_ldpc_iq_llr_capture.py
│   ├── bpsk_jscc_RX_iq_capture.grc
│   ├── graphs.py
│   ├── measurement_30 oct/           # Saved metric tables
│   └── measurement_before_30 oct/    # Earlier metric tables
├── reference_soft/                  # Soft VOGEO transmitter/receiver and bundles
│   ├── bundle_3_out_4/              # 3 codebooks + 10 parity bits (current OTA)
│   └── bundle_soft/                 # Different model configuration (not current OTA)
├── deepJSCC_voice_hard_1_2/        # Earlier hard-decision implementation
├── benchmark_processing_latency.py # Offline three-method processing benchmark
├── benchmark_tx_stage_latency.py   # Offline TX stage benchmark
├── docs/                            # Diagrams, demo and audio examples
└── LICENSE
```

The root-level latency scripts benchmark **three** methods from an earlier comparison; they should not be presented as four-method OTA latency measurements. The `deploy_soft/` simulation uses its own baseline definitions and has its own results and instructions.

## Getting started

### A. Explore the software simulation

The simulation can be run on a computer without USRP hardware. Follow the environment, version and experiment instructions in [`deploy_soft/README.md`](deploy_soft/README.md). For example, after installing its listed dependencies:

```bash
cd deploy_soft
python3 selftest_sim_v3.py --selftest
python3 selftest_sim_v3.py --test_dir test_data --seconds 0 --trials 1 \
    --max_files 2 --snrs 0 3 --channels awgn rayleigh \
    --mismatch_db 0 --out_prefix smoke
python3 plot_sim_v3.py --prefix smoke --out_dir figs_smoke --fmt png
```

The exported bundle and test clips are included under `deploy_soft/`. The detailed simulation README is the authoritative reference for its method definitions, dependencies, parameters and reported software results.

### B. Reproduce the OTA measurements

The OTA test requires **two compatible USRP radios (tested configuration: B210)**, GNU Radio/UHD, the 24-kHz EnCodec model, and LibriTTS `test-clean` speech. The receiver writes decoded audio and metrics, and can capture post-Costas complex symbols, demodulator metrics and LLR arrays for later processing.

**Start with [`ota/README.md`](ota/README.md).** It documents the required folder-layout adjustment for the scripts in this archive, dependencies, UDP ports, start order, RF packet format, capture files, and plotting. Do not attempt to reproduce the OTA experiment from the simulation quick-start commands above.

## Evaluation and reproducibility

- **Speech quality/intelligibility:** reconstructed audio is evaluated with PESQ and ESTOI; saved OTA data also include SI-SDR tables used for plotting. The current OTA receiver computes PESQ and ESTOI directly; SI-SDR is not calculated by that RX script.
- **Simulation:** AWGN/Rayleigh experiments and their evaluation options are documented in `deploy_soft/README.md`.
- **OTA:** saved tables under `ota/measurement_30 oct/` contain the four method labels. `ota/graphs.py` plots PESQ, ESTOI and SI-SDR against TX gain. A separate gain=89 reference is handled as a horizontal reference rather than a regular sweep point.
- **Missing RF detections:** a non-detected packet has no genuine received IQ/LLR recording. The OTA code logs expected transmit packets and observed detections separately; reconstructed speech uses silence for missing chunks where the corresponding audio state is known.

For any paper figure, record the dataset subset, TX/RX gain settings, receiver configuration, bundle `run_id`, packet detection counts, and metric/CI calculation. The repository contains scripts and saved measurements, but the exact performance reported in a manuscript depends on the selected measurement run.

## Model and data notes

The OTA code searches for LibriTTS `test-clean` in a local `datasets/` directory. The full LibriTTS dataset is **not shipped here**; obtain it from [OpenSLR 60](https://www.openslr.org/60/). The `deploy_soft/test_data/` directory holds a separate set of example/evaluation clips for the simulation workflow.

Trained model exports are included for inference/evaluation; model-training code and training checkpoints are **not** included with `deploy_soft/` as documented in its README. EnCodec model weights may be fetched on first use depending on the local environment.

## License and attribution

Repository code is licensed under the [Apache License 2.0](LICENSE) (Copyright 2026 Hisar Research), subject to the included license text. **Third-party code, pretrained weights, datasets and audio retain their own terms**, including EnCodec and LibriTTS/LibriSpeech. Check the relevant upstream licenses before redistribution or commercial use.

For method-specific questions, first consult [`deploy_soft/README.md`](deploy_soft/README.md) for the simulation or [`ota/README.md`](ota/README.md) for the RF implementation. A paper reference can be added once a citable version is available.
