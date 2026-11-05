# Experimental SDR Testbed for Robust Semantic Audio Transmission

This repository implements an over-the-air SDR testbed for evaluating a learned joint source-channel coding approach for robust speech transmission.

The system combines **Meta EnCodec**, a learned VOGEO protection model, GNU Radio, and USRP radios. Three transmission strategies are evaluated under approximately the same channel-bit budget:

| Method       | Source Representation              | Channel Protection                    | Transmitted Bits / 1-s Chunk |
| ------------ | ---------------------------------- | ------------------------------------- | ---------------------------: |
| **Proposed** | EnCodec 2.25 kbps, 3 RVQ codebooks | 10 learned protection bits/frame      |                         3000 |
| **Conv**     | EnCodec 2.25 kbps, 3 RVQ codebooks | Rate-3/4 punctured convolutional code |                         3008 |
| **Plain**    | EnCodec 3 kbps, 4 RVQ codebooks    | None                                  |                         3000 |

The objective is to compare the robustness of learned semantic/channel protection against conventional channel coding and uncoded transmission in a real SDR link.

## System Overview

The experimental chain is:

```text
LibriTTS speech
      |
      v
   EnCodec
      |
      +-------------------+--------------------+
      |                   |                    |
      v                   v                    v
 Plain 3 kbps        Conv baseline       Proposed VOGEO
 4 codebooks         3 codebooks         3 codebooks
 3000 bits           2250 source bits    2250 source bits
      |                   |              + 750 learned bits
      |              Conv K=7                  |
      |              Rate 3/4                  |
      |                   |                    |
      +--------- 3000 / 3008 / 3000 bits ------+
                          |
                    Super-packet
                          |
                    GNU Radio TX
                          |
                       USRP TX
                          |
                    Wireless Channel
                          |
                       USRP RX
                          |
                    GNU Radio RX
                          |
                  Soft BPSK metrics
                          |
          +---------------+---------------+
          |               |               |
     Hard slice       Soft Viterbi    Blind EM LLR
       Plain              Conv          + VOGEO
          |               |               |
          +---------------+---------------+
                          |
                    EnCodec Decode
                          |
                 Reconstructed Audio
                          |
                    PESQ / ESTOI
```

## Dataset

The transmitter uses the **LibriTTS `test-clean`** dataset.

The script automatically searches for the dataset under locations such as:

```text
datasets/LibriTTS/test-clean/
datasets/LibriTTS/LibriTTS/test-clean/
datasets/test-clean/
```

Only utterances satisfying

```text
8.0 s <= duration < 15.0 s
```

are selected.

The first **100 valid audio files** are used for the experiment.

All audio is converted to the 24-kHz format required by EnCodec.

## Audio Processing

Audio is divided into **1-second chunks**:

```text
Sampling rate      = 24 kHz
Samples/chunk      = 24000
EnCodec frames     = 75 frames/chunk
Bits/RVQ index     = 10
```

The last chunk is zero-padded before encoding while its original number of valid samples is stored in the metadata header.

## Transmission Methods

### 1. Plain EnCodec Baseline

The uncoded baseline uses:

```text
EnCodec bandwidth = 3.0 kbps
RVQ codebooks     = 4
Bits/frame        = 40
Frames/chunk      = 75

Total = 4 × 75 × 10
      = 3000 bits/chunk
```

At the receiver, the GNU Radio soft BPSK metrics are converted directly into hard bit decisions.

No channel code is applied.

### 2. Conventional Coding Baseline

The conventional baseline uses:

```text
EnCodec bandwidth = 2.25 kbps
RVQ codebooks     = 3

Source bits = 3 × 75 × 10
            = 2250 bits/chunk
```

The source bits are protected using a terminated convolutional code:

```text
Constraint length K = 7
Generators          = (133, 171) octal
Mother rate         = 1/2
Punctured rate      = 3/4
Puncture pattern    = [1, 1, 1, 0, 0, 1]
Tail bits           = 6
```

After termination and puncturing:

```text
2250 source bits
+ 6 termination bits
-> 4512 mother-code bits
-> 3008 transmitted bits
```

The receiver performs **soft-decision Viterbi decoding** using the received BPSK soft metrics.

Punctured positions are inserted with zero reliability.

### 3. Proposed Learned VOGEO Scheme

The proposed method uses the same 2.25-kbps EnCodec source representation as the convolutionally coded baseline:

```text
3 RVQ codebooks × 10 bits
= 30 source bits/frame
```

The learned protection encoder generates:

```text
10 learned protection bits/frame
```

giving:

```text
30 source bits
+ 10 learned bits
= 40 bits/frame

40 × 75
= 3000 transmitted bits/chunk
```

Therefore, the proposed system and the uncoded 3-kbps baseline use exactly 3000 payload bits per one-second chunk, while the convolutional baseline uses 3008 bits because of trellis termination.

## Soft VOGEO Receiver

The SDR experiment does **not** convert the configured USRP transmit gain into an assumed SNR or \(E_s/N_0\).

Instead, the reliability of each received VOGEO packet is estimated directly from the received BPSK payload metrics.

A two-component Gaussian model is fitted:

```text
m | bit=0 ~ N(mu0, sigma²)
m | bit=1 ~ N(mu1, sigma²)
```

using a blind EM procedure.

The estimated parameters are converted into bit LLRs:

```text
LLR(m) =
((m - mu0)^2 - (m - mu1)^2)
--------------------------------
            2 sigma²
```

where:

```text
positive LLR -> bit 1
negative LLR -> bit 0
```

The LLRs are clipped before being passed to the learned soft VOGEO decoder.

The default clipping magnitude is:

```text
±50
```

This makes the learned decoder adaptive to the actual received metric amplitude rather than assuming that USRP TX gain corresponds to channel SNR.

## Super-Packet Design

For every audio chunk, all three methods are placed into a single RF super-packet.

```text
Plain       = 375 bytes
Conv        = 376 bytes
Proposed    = 375 bytes
-------------------------
Total       = 1126 bytes
            = 9008 RF bits
```

The order of the three methods is rotated across chunks:

```text
Chunk 0: plain -> conv  -> proposed
Chunk 1: conv  -> proposed -> plain
Chunk 2: proposed -> plain -> conv
```

and then repeats.

This prevents one method from always occupying the same temporal position inside the RF transmission.

GNU Radio returns one `float32` soft metric for each received bit:

```text
9008 float32 values
= 36032 UDP bytes
```

The Plain branch uses only the sign of these values, whereas the convolutional and proposed methods preserve their soft information.

## Metadata Header

A small metadata header is transmitted directly from the Python transmitter to the Python receiver and therefore bypasses the RF channel.

Format:

```text
!2sBIHHH
```

Fields:

```text
Magic         : "VQ"
Packet type   : 5
Audio ID      : uint32
Chunk ID      : uint16
Total chunks  : uint16
Valid samples : uint16
```

Header size:

```text
13 bytes
```

This header is used for experiment bookkeeping and reconstruction and is not part of the RF payload comparison.

## UDP Interfaces

The transmitter communicates with GNU Radio using:

```text
TX Python -> GNU Radio
127.0.0.1:5000
```

Metadata and the recovered GNU Radio payload are handled through:

```text
Python/GNU Radio -> RX Python
127.0.0.1:5002
```

The transmitter sends the 13-byte metadata header directly to port `5002` before sending the corresponding RF super-packet to GNU Radio.

## Required Project Structure

A typical project directory is:

```text
project/
├── tx_three_method_TXGAIN_30plus10_SOFT_conv34.py
├── rx_three_method_TXGAIN_30plus10_SOFT_conv34.py
│
├── reference_soft/
│   ├── vogeo_tx.py
│   ├── vogeo_rx.py
│   ├── vogeo_llr.py
│   ├── vogeo_frame.py
│   └── bundle_3_out_4/
│       └── ...
│
├── datasets/
│   └── LibriTTS/
│       └── test-clean/
│
└── GNU Radio flowgraph
```

The VOGEO bundle must correspond to the **soft 30+10 configuration**:

```text
n_q        = 3
parity_dim = 10
decision   = soft
```

The same trained bundle must be used at both the transmitter and receiver.

## Python Dependencies

The Python scripts require packages including:

```bash
numpy
scipy
torch
soundfile
encodec
pesq
pystoi
```

The SDR portion additionally requires a working GNU Radio/UHD installation and the appropriate USRP hardware configuration.

## Running the Experiment

### 1. Start the receiver

The transmit-gain value is required by the receiver as an **experimental label**:

```bash
python3 rx_three_method_TXGAIN_30plus10_SOFT_conv34.py \
    --tx-gain 30
```

Optional VOGEO LLR clipping:

```bash
python3 rx_three_method_TXGAIN_30plus10_SOFT_conv34.py \
    --tx-gain 30 \
    --vogeo-llr-clip 50
```

Importantly, `--tx-gain 30` does **not** tell the receiver that the channel SNR is 30 dB.

It records the actual USRP transmitter-gain operating point used for the experiment.

### 2. Start the GNU Radio flowgraph

Start the SDR TX/RX flowgraph and verify that:

```text
Python TX UDP input  = port 5000
Python RX UDP output = port 5002
```

The GNU Radio receiver must output one `float32` soft BPSK metric for each recovered payload bit.

### 3. Start the transmitter

```bash
python3 tx_three_method_TXGAIN_30plus10_SOFT_conv34.py
```

For every selected audio file, the transmitter:

1. loads and resamples the speech;
2. splits it into one-second chunks;
3. generates all three representations;
4. constructs one super-packet;
5. sends the metadata header directly to the receiver;
6. sends the super-packet through GNU Radio and the SDR link.

The current transmission timing is:

```text
Chunk interval = 0.25 s
Audio interval = 5 s
```

## Output

Original audio files are stored under:

```text
original_audio/
```

Received files are organized by TX-gain operating point:

```text
received_audio_superpacket_soft/
└── audio_gain_30/
    ├── proposed/
    ├── conv/
    └── 3/
```

Example:

```text
received_audio_superpacket_soft/
└── audio_gain_30/
    └── proposed/
        └── received_audio_0_proposed.wav
```

If an RF chunk is missing, the receiver inserts silence for that chunk so that the reconstructed waveform maintains its original time alignment.

## Evaluation Metrics

Two objective speech metrics are computed automatically.

### ESTOI

Extended Short-Time Objective Intelligibility is calculated on the reconstructed complete utterance.

Results are stored in:

```text
estoi_results_superpacket_soft_txgain.csv
```

Format:

```text
method,audio_id,estoi,tx_gain_db
```

### PESQ

Wideband PESQ is calculated after resampling both reference and received speech to 16 kHz.

Results are stored in:

```text
pesq_results_superpacket_soft_txgain.csv
```

Format:

```text
method,audio_id,pesq,tx_gain_db
```

Method labels are:

```text
proposed = learned VOGEO
conv     = rate-3/4 convolutional baseline
3        = uncoded EnCodec 3-kbps baseline
```

## Experimental Fairness

Several design choices are used to make the comparison fair:

* The three methods use approximately the same RF bit budget.
* Proposed and convolutional methods start from the same 3-codebook EnCodec source representation.
* All methods are transmitted inside the same super-packet.
* Their positions within the packet are rotated between chunks.
* Conv and VOGEO use the same received soft-demodulator output.
* TX gain is reported as TX gain rather than incorrectly relabeled as SNR.
* Missing packets are represented by silence rather than removing time from the reconstructed signal.



## Purpose

This implementation is intended to experimentally evaluate whether learned channel-aware redundancy can provide improved speech quality and intelligibility under challenging wireless conditions compared with conventional channel coding at a comparable transmission budget.

