#!/usr/bin/env python3

import os
import re
import csv
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

# ============================================================
# ABSOLUTE PROJECT PATHS
# ============================================================



PROJECT_ROOT = Path.cwd()

# Original clean audio:
# /home/amrtaweel/Desktop/tubitak_research/original_audio/
ORIGINAL_DIR = os.path.join(
    PROJECT_ROOT,
    "original_audio"
)

# Received/reconstructed audio:
#
# received_audio_superpacket_soft_gain/
# ├── audio_gain_27/
# │   ├── proposed/
# │   ├── conv/
# │   ├── ldpc/
# │   └── 3/
# ├── audio_gain_29/
# ├── audio_gain_31/
# ├── audio_gain_33/
# ├── audio_gain_35/
# ├── audio_gain_37/
# └── audio_gain_39/
#
RECEIVED_ROOT = os.path.join(
    PROJECT_ROOT,
    "received_audio_superpacket_soft"
)

# Output CSV
OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "sisdr_results_superpacket_soft_txgain.csv"
)


# ============================================================
# METHODS
# ============================================================

METHODS = (
    "proposed",
    "conv",
    "ldpc",
    "3",
)


# ============================================================
# AUDIO READING
# ============================================================

def read_mono(path):
    """
    Read WAV file as float64 and convert it to mono.
    """

    x, sr = sf.read(
        path,
        dtype="float64",
        always_2d=True
    )

    x = np.mean(
        x,
        axis=1
    )

    if not np.all(
        np.isfinite(x)
    ):
        raise ValueError(
            f"NaN/Inf found in {path}"
        )

    return (
        x.astype(np.float64),
        int(sr)
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_audio(
    x,
    old_sr,
    new_sr
):
    """
    Resample received audio when its sample rate differs
    from the original signal.
    """

    if old_sr == new_sr:
        return np.asarray(
            x,
            dtype=np.float64
        )

    g = math.gcd(
        int(old_sr),
        int(new_sr)
    )

    y = resample_poly(
        x,
        int(new_sr) // g,
        int(old_sr) // g,
    )

    return np.asarray(
        y,
        dtype=np.float64
    )


# ============================================================
# SI-SDR
# ============================================================

def si_sdr(
    reference,
    estimate,
    zero_mean=True,
    eps=1e-12
):
    """
    Scale-Invariant Signal-to-Distortion Ratio.

    SI-SDR = 10 log10(
        ||s_target||^2 /
        ||e_noise||^2
    )

    where

        alpha =
            <estimate, reference>
            ---------------------
              ||reference||^2

        s_target = alpha * reference

        e_noise = estimate - s_target
    """

    reference = np.asarray(
        reference,
        dtype=np.float64
    ).reshape(-1)

    estimate = np.asarray(
        estimate,
        dtype=np.float64
    ).reshape(-1)

    # Compare common duration only
    n = min(
        reference.size,
        estimate.size
    )

    if n == 0:
        raise ValueError(
            "Empty signal"
        )

    reference = (
        reference[:n]
        .copy()
    )

    estimate = (
        estimate[:n]
        .copy()
    )

    # Remove DC component
    if zero_mean:

        reference -= np.mean(
            reference
        )

        estimate -= np.mean(
            estimate
        )

    ref_energy = np.sum(
        reference ** 2
    )

    if ref_energy <= eps:
        raise ValueError(
            "Reference audio has near-zero energy"
        )

    alpha = (
        np.dot(
            estimate,
            reference
        )
        /
        (
            ref_energy
            + eps
        )
    )

    target = (
        alpha
        * reference
    )

    residual = (
        estimate
        - target
    )

    target_energy = np.sum(
        target ** 2
    )

    residual_energy = np.sum(
        residual ** 2
    )

    score = (
        10.0
        * np.log10(
            (target_energy + eps)
            /
            (residual_energy + eps)
        )
    )

    return float(score)


# ============================================================
# FILENAME PARSING
# ============================================================

def parse_audio_id(
    filename,
    method
):
    """
    Expected names:

    received_audio_0_proposed.wav
    received_audio_0_conv.wav
    received_audio_0_ldpc.wav
    received_audio_0_3.wav
    """

    pattern = (
        rf"^received_audio_(\d+)_"
        rf"{re.escape(method)}"
        rf"\.wav$"
    )

    match = re.match(
        pattern,
        filename
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


# ============================================================
# GAIN FOLDER PARSING
# ============================================================

def gain_from_folder(
    folder_name
):
    """
    Example:

        audio_gain_27 -> 27.0
    """

    prefix = "audio_gain_"

    if not folder_name.startswith(
        prefix
    ):
        return None

    text = folder_name[
        len(prefix):
    ]

    try:
        return float(text)

    except ValueError:
        return None


# ============================================================
# EVALUATE ONE AUDIO FILE
# ============================================================

def evaluate_one_file(
    original_path,
    received_path
):
    """
    Calculate SI-SDR for one original/received pair.
    """

    reference, ref_sr = read_mono(
        original_path
    )

    received, rec_sr = read_mono(
        received_path
    )

    # Match sample rates
    if rec_sr != ref_sr:

        received = resample_audio(
            received,
            rec_sr,
            ref_sr
        )

    # Common available duration
    n = min(
        len(reference),
        len(received)
    )

    if n == 0:
        raise ValueError(
            "No common audio samples"
        )

    reference = reference[:n]

    received = received[:n]

    score = si_sdr(
        reference,
        received,
        zero_mean=True
    )

    return (
        float(score),
        n,
        ref_sr
    )


# ============================================================
# MAIN
# ============================================================

def main():

    original_dir = os.path.abspath(
        ORIGINAL_DIR
    )

    received_root = os.path.abspath(
        RECEIVED_ROOT
    )

    output_csv = os.path.abspath(
        OUTPUT_CSV
    )


    # ========================================================
    # VERIFY DIRECTORIES
    # ========================================================

    if not os.path.isdir(
        original_dir
    ):
        raise RuntimeError(
            "Original-audio directory not found:\n"
            f"{original_dir}"
        )

    if not os.path.isdir(
        received_root
    ):
        raise RuntimeError(
            "Received-audio root not found:\n"
            f"{received_root}"
        )


    # ========================================================
    # FIND ALL TX-GAIN FOLDERS
    # ========================================================

    gain_folders = []

    for name in os.listdir(
        received_root
    ):

        path = os.path.join(
            received_root,
            name
        )

        if not os.path.isdir(
            path
        ):
            continue

        gain = gain_from_folder(
            name
        )

        if gain is not None:

            gain_folders.append(
                (
                    gain,
                    path
                )
            )

    gain_folders.sort(
        key=lambda item: item[0]
    )

    if not gain_folders:

        raise RuntimeError(
            "No audio_gain_<gain> folders found inside:\n"
            f"{received_root}"
        )


    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "SI-SDR EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        "Project root    :",
        PROJECT_ROOT
    )

    print(
        "Original audio  :",
        original_dir
    )

    print(
        "Received audio  :",
        received_root
    )

    print(
        "Output CSV      :",
        output_csv
    )

    print(
        "Methods         :",
        METHODS
    )

    print()

    print(
        "TX gains found  :",
        [
            gain
            for gain, _ in gain_folders
        ]
    )

    print()


    # ========================================================
    # CALCULATE SI-SDR
    # ========================================================

    rows = []

    for (
        tx_gain_db,
        gain_dir
    ) in gain_folders:

        print()

        print(
            "=" * 80
        )

        print(
            f"TX GAIN = {tx_gain_db:g} dB"
        )

        print(
            "=" * 80
        )

        for method in METHODS:

            method_dir = os.path.join(
                gain_dir,
                method
            )

            if not os.path.isdir(
                method_dir
            ):

                print(
                    f"{method:10s} | "
                    f"FOLDER MISSING | "
                    f"{method_dir}"
                )

                continue

            filenames = sorted(
                os.listdir(
                    method_dir
                )
            )

            method_scores = []

            for filename in filenames:

                audio_id = parse_audio_id(
                    filename,
                    method
                )

                if audio_id is None:
                    continue

                received_path = os.path.join(
                    method_dir,
                    filename
                )

                original_path = os.path.join(
                    original_dir,
                    f"original_audio_{audio_id}.wav"
                )

                if not os.path.isfile(
                    original_path
                ):

                    print(
                        f"{method:10s} | "
                        f"audio={audio_id:3d} | "
                        "SKIP: original missing"
                    )

                    continue

                try:

                    (
                        score,
                        num_samples,
                        sample_rate
                    ) = evaluate_one_file(
                        original_path,
                        received_path
                    )

                    rows.append(
                        {
                            "method": method,
                            "audio_id": audio_id,
                            "si_sdr_db": score,
                            "tx_gain_db": tx_gain_db,
                        }
                    )

                    method_scores.append(
                        score
                    )

                    print(
                        f"{method:10s} | "
                        f"audio={audio_id:3d} | "
                        f"SI-SDR={score:9.3f} dB | "
                        f"n={num_samples} | "
                        f"sr={sample_rate}"
                    )

                except Exception as exc:

                    print(
                        f"{method:10s} | "
                        f"audio={audio_id:3d} | "
                        f"ERROR: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )


            # =================================================
            # METHOD SUMMARY
            # =================================================

            if method_scores:

                values = np.asarray(
                    method_scores,
                    dtype=np.float64
                )

                print()

                print(
                    f">>> {method:10s} | "
                    f"n={len(values):3d} | "
                    f"AVG={values.mean():9.3f} dB | "
                    f"STD={values.std(ddof=0):9.3f} | "
                    f"MIN={values.min():9.3f} | "
                    f"MAX={values.max():9.3f}"
                )

                print()

            else:

                print(
                    f">>> {method:10s} | n=0"
                )


    # ========================================================
    # SORT RESULTS
    # ========================================================

    rows.sort(
        key=lambda r: (
            float(
                r["tx_gain_db"]
            ),
            str(
                r["method"]
            ),
            int(
                r["audio_id"]
            ),
        )
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "audio_id",
                "si_sdr_db",
                "tx_gain_db",
            ],
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    "method":
                        row["method"],

                    "audio_id":
                        row["audio_id"],

                    "si_sdr_db":
                        f"{row['si_sdr_db']:.6f}",

                    "tx_gain_db":
                        f"{row['tx_gain_db']:.2f}",
                }
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL SI-SDR SUMMARY"
    )

    print(
        "=" * 80
    )

    all_gains = sorted(
        set(
            float(
                row["tx_gain_db"]
            )
            for row in rows
        )
    )

    for gain in all_gains:

        print()

        print(
            f"TX gain = {gain:g} dB"
        )

        for method in METHODS:

            values = np.asarray(
                [
                    row["si_sdr_db"]

                    for row in rows

                    if (
                        float(
                            row["tx_gain_db"]
                        )
                        == gain

                        and

                        row["method"]
                        == method
                    )
                ],
                dtype=np.float64,
            )

            if values.size:

                print(
                    f"  {method:10s} | "
                    f"n={values.size:3d} | "
                    f"avg={values.mean():9.3f} dB | "
                    f"std={values.std(ddof=0):9.3f} dB"
                )

            else:

                print(
                    f"  {method:10s} | n=0"
                )


    print()

    print(
        "=" * 80
    )

    print(
        "CSV saved to:"
    )

    print(
        output_csv
    )

    print(
        "=" * 80
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()