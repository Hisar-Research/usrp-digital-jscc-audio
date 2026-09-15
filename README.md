# Experimental SDR Testbed for Robust Semantic Audio Transmission

This repository contains the SDR implementation used to experimentally evaluate three digital audio transmission schemes over a real wireless link using two USRP B210 software-defined radios and GNU Radio.

The evaluated methods are:

* **Proposed** – learned robust semantic audio transmission
* **Conv** – conventional convolutionally coded transmission
* **Plain / 3 kbps** – uncoded EnCodec transmission

The system uses **EnCodec** for speech representation and compares the three methods under approximately equal transmitted-bit budgets.

---

## System Overview

The complete experimental chain is:

```text
LibriTTS Audio
      |
      v
   EnCodec
      |
      v
+------------------------------+
| Plain | Conv | Proposed      |
+------------------------------+
      |
      v
 Python TX
      |
      v
 GNU Radio TX
      |
      v
 USRP B210
      |
   Wireless Link
      |
      v
 USRP B210
      |
      v
 GNU Radio RX
      |
      v
 Python RX
      |
      v
Reconstructed Audio
      |
      v
PESQ / ESTOI / SI-SDR
```

---

## Dataset

The experiments use the **LibriTTS `test-clean`** dataset.

The dataset is **not included in this repository and must be downloaded separately**.

Download LibriTTS from the official OpenSLR repository:

```text
https://www.openslr.org/60/
```

The required archive is:

```text
test-clean.tar.gz
```

It can also be downloaded directly from the terminal:

```bash
wget https://www.openslr.org/resources/60/test-clean.tar.gz
```

Extract it using:

```bash
tar -xzf test-clean.tar.gz
```

The expected dataset directory is:

```text
datasets/
└── LibriTTS/
    └── test-clean/
```

The experiment selects:

```text
Duration >= 8.0 s
Duration < 15.0 s
```

and uses the **first 100 valid audio files** after filtering.

All speech signals are converted to:

```text
Sample rate: 24 kHz
Mono
```

before EnCodec processing.

---

## Audio Processing

The experiment uses the pretrained **24-kHz EnCodec model**.

Each speech signal is encoded into discrete neural audio representations before transmission.

The three transmission methods use different ways of spending approximately the same communication budget.

```text
Plain       : 3000 transmitted bits
Convolution : 3008 transmitted bits
Proposed    : 3000 transmitted bits
```

This allows comparison of learned protection against conventional channel coding under approximately equal transmission cost.

---

## Transmission Methods

### 1. Plain

The plain baseline transmits the EnCodec representation directly without channel coding.

```text
Method label: 3
Bits:         3000
```

At the receiver, the received BPSK metrics are converted directly to hard bits.

```text
metric >= 0 -> 1
metric <  0 -> 0
```

No error-correcting code is applied.

---

### 2. Convolutional Coding

The conventional baseline protects the EnCodec representation using convolutional coding.

```text
Method label: conv
Coded bits:  3008
```

The receiver uses the received soft BPSK metrics for **soft-input Viterbi decoding**.

This provides a conventional digital channel-coding baseline for comparison with the learned method.

---

### 3. Proposed Method

The proposed method introduces learned redundancy into the EnCodec representation.

```text
Method label: proposed
Bits:         3000
```

Unlike the plain method, the received BPSK values are not immediately converted into hard bits.

The receiver instead derives soft reliability information and passes it to the learned decoder.

---

## Soft VOGEO Receiver

The proposed SDR receiver does **not** assume that a given USRP transmit gain corresponds to a specific SNR or \(E_s/N_0\).

Instead, reliability is estimated directly from the received BPSK payload metrics.

A two-component Gaussian model is fitted:

```text
m | bit=0 ~ N(mu0, sigma²)
m | bit=1 ~ N(mu1, sigma²)
```

using a blind EM procedure.

The estimated parameters are converted into bit LLRs:

```text
             (m - mu0)^2 - (m - mu1)^2
LLR(m) =    -----------------------------
                       2 sigma²
```

where:

```text
positive LLR -> bit 1
negative LLR -> bit 0
```

The magnitude of the LLR represents the estimated reliability.

Before being passed to the learned decoder, the LLRs are clipped to:

```text
[-50, +50]
```

Therefore, the proposed decoder receives **soft values**, not only hard binary decisions.

This allows the decoder to adapt to the actual received metric distribution instead of assuming that USRP TX gain is equivalent to SNR.

---

## Super-Packet Design

The three methods are transmitted together as one experimental super-packet.

The RF payload is:

```text
1126 bytes
9008 bits
```

The super-packet contains the representations required for:

```text
Proposed
Plain
Convolutional
```

The transmission order may be varied between packets to avoid systematically favoring one method because of its location inside the RF payload.

The receiver identifies the order from the corresponding metadata.

---

## Metadata Header

Metadata is transmitted separately from the RF payload.

It contains information required to associate received packets with the original speech signal and audio chunk.

Typical fields include:

```text
audio_id
chunk_id
total_chunks
valid_samples
method order
TX gain
```

This allows the receiver to reconstruct each audio file correctly even when packets are received in different method orders.

---

## UDP Interfaces

Python and GNU Radio communicate using UDP sockets.

The transmitter sends prepared RF payloads to GNU Radio.

GNU Radio performs:

```text
packet framing
BPSK modulation
pulse shaping
USRP transmission
```

At the receiver, GNU Radio performs:

```text
USRP reception
timing recovery
carrier recovery
packet synchronization
payload extraction
```

and forwards the recovered payload metrics to the Python receiver.

The Python receiver then performs the appropriate decoding operation for each method.

---

## Required Project Structure

A typical setup is:

```text
project/
│
├── tx/
├── rx/
├── models/
├── results/
│
├── datasets/
│   └── LibriTTS/
│       └── test-clean/
│
├── original_audio/
├── received_audio/
│
├── *.py
├── *.grc
└── README.md
```

The LibriTTS dataset itself is not stored in the Git repository.

---

## Python Dependencies

The main Python dependencies include:

```text
numpy
scipy
pandas
torch
torchaudio
encodec
soundfile
librosa
pesq
pystoi
```

Install the required packages in the Python environment used by the experiment.

GNU Radio and UHD must also be installed for the SDR experiment.

Check that the B210 devices are detected using:

```bash
uhd_find_devices
```

or:

```bash
uhd_usrp_probe
```

---

## Running the Experiment

### 1. Download the dataset

Download and extract LibriTTS `test-clean` as described in the **Dataset** section.

The expected location is:

```text
datasets/LibriTTS/test-clean/
```

---

### 2. Connect the USRPs

Connect the transmitter and receiver USRP B210 devices.

Verify that both devices are detected.

---

### 3. Start GNU Radio

Start the GNU Radio transmitter and receiver flowgraphs.

The receiver should be running before starting the Python transmission process.

---

### 4. Start the Python receiver

Run the three-method receiver.

For example:

```bash
python3 rx_three_method_TXGAIN_30plus10_SOFT_conv34_BLIND_EM.py --tx-gain <GAIN>
```

where `<GAIN>` is the USRP transmit-gain value corresponding to the current experiment.

---

### 5. Start the transmitter

Run the transmitter script using the same experimental configuration.

The selected LibriTTS files are encoded and transmitted through the SDR testbed.

---

### 6. Repeat for each TX gain

Repeat the experiment for the desired USRP transmit-gain operating points.

The TX gain values are treated as **experimental RF operating conditions** and are not converted into assumed SNR values.

---

## Output

The experiment stores reconstructed audio for all three methods.

Typical filenames are:

```text
received_audio_<id>_proposed.wav
received_audio_<id>_conv.wav
received_audio_<id>_3.wav
```

The original reference audio is stored separately.

Metric results are saved as CSV files.

Typical method labels are:

```text
proposed
conv
3
```

---

## Saved Experimental Results

The saved experimental results used in this project are available here:

```text
https://drive.google.com/drive/folders/1ObEJoAW_-GTYUfZdp0jjQLMnjmCX27nh?usp=sharing
```

The Google Drive folder contains the saved outputs from the experiments, including the result files used for evaluation and figure generation.

This allows the experimental results to be inspected without rerunning the complete over-the-air SDR experiment.

---

## Evaluation Metrics

The reconstructed speech is evaluated using three objective metrics.

### ESTOI

Extended Short-Time Objective Intelligibility measures speech intelligibility.

Higher values indicate better intelligibility.

---

### PESQ

Perceptual Evaluation of Speech Quality measures perceptual speech quality.

Higher values indicate better quality.

---

### SI-SDR

Scale-Invariant Signal-to-Distortion Ratio measures waveform reconstruction quality.

It is reported in dB.

Higher values indicate better reconstruction.

---

## Experimental Fairness

The three transmission schemes are designed to use approximately the same RF transmission budget.

```text
Proposed : 3000 bits
Plain    : 3000 bits
Conv     : 3008 bits
```

Therefore, improvements from the proposed approach are not obtained by simply transmitting substantially more physical bits.

The comparison investigates how the available redundancy is used:

```text
Plain
    -> no explicit redundancy

Conv
    -> conventional channel-coding redundancy

Proposed
    -> learned redundancy
```

---

## Important Experimental Note

The SDR experiment does **not** interpret USRP transmit gain as SNR.

In particular:

```text
TX gain != SNR
TX gain != Es/N0
```

USRP transmit gain controls the RF transmit-chain operating point.

The actual received signal is affected by:

```text
path loss
antenna characteristics
receiver gain
hardware response
noise
frequency offset
timing errors
synchronization
multipath
other RF impairments
```

For this reason, SDR results are reported against the actual experimental transmit-gain setting.

For the proposed method, channel reliability is estimated directly from the received BPSK soft metrics using the blind Gaussian model described above.

---

## Purpose

The purpose of this repository is to provide the implementation used to experimentally investigate whether learned protection of neural audio representations can offer improved robustness over conventional digital transmission methods.

The testbed complements software simulations by evaluating the methods over a real wireless link using USRP B210 software-defined radios.

The experiment therefore compares:

```text
learned semantic protection
vs.
conventional channel coding
vs.
uncoded transmission
```

under practical RF impairments and approximately equal transmitted-bit budgets.
