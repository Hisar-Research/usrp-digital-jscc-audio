"""Plot PESQ, ESTOI and SI-SDR against USRP TX gain for a new OTA sweep.

Reads the CSVs that the current scripts write into the results directory
(default: the current directory, i.e. ota/):

    pesq_results_superpacket_soft_txgain.csv    written by the RX script
    estoi_results_superpacket_soft_txgain.csv   written by the RX script
    sisdr_results_superpacket_soft_txgain.csv   written by calculate_sisdr.py (optional)

The archived measurements in measurement_30 oct/ and measurement_before_30 oct/
are not read.

Rows with tx_gain_db == 89 are the reference run at maximum TX gain. They are
not plotted as a point on the gain axis; their mean gives the horizontal
ceiling lines (coded methods: proposed; uncoded: 3).

    python3 graphs.py                         # CSVs in ., figures in ./figures
    python3 graphs.py --results-dir run_dec --out-dir run_dec/figures
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import t

REFERENCE_GAIN = 89          # maximum B210 TX gain, used only for the ceilings
EXPECTED_GAINS = 7           # regular TX-gain points per figure

CSV_NAMES = {
    "pesq": "pesq_results_superpacket_soft_txgain.csv",
    "estoi": "estoi_results_superpacket_soft_txgain.csv",
    "si_sdr_db": "sisdr_results_superpacket_soft_txgain.csv",
}

# CSV method label -> curve style. Same colours as the simulation plots.
METHODS = ("proposed", "golay", "ldpc", "3")
STYLES = {
    "proposed": {"label": "Proposed (M1)",      "marker": "o", "color": "#1f77b4", "linestyle": "-"},
    "golay":    {"label": "Golay + SBSD (M3.1)", "marker": "^", "color": "#2ca02c", "linestyle": "-."},
    "ldpc":     {"label": "IRA-LDPC (M3.2)",    "marker": "D", "color": "#9467bd", "linestyle": (0, (3, 1, 1, 1))},
    "3":        {"label": "Uncoded (M2)",       "marker": "s", "color": "#7f7f7f", "linestyle": "--"},
}
CEILINGS = (   # (key, reference method, label, colour, linestyle)
    ("coded", "proposed", f"Ceiling @ {REFERENCE_GAIN} dB, M1/M3.1/M3.2", "#000000", (0, (1, 1))),
    ("uncoded", "3", f"Ceiling @ {REFERENCE_GAIN} dB, M2", "#555555", (0, (4, 2))),
)

# Same physical canvas as the reference EPS
FIGSIZE = (228.195625 / 72.0, 170.024 / 72.0)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 5.0,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})


def load(path, metric):
    """Read one metric CSV, tolerating lines that two writers interleaved."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = [c.strip() for c in f.readline().strip().split(",")]
        for line in f:
            s = line.strip()
            if not s:
                continue
            found = re.findall(r"(proposed|golay|ldpc|3),(\d+),([-+0-9.eEinfa]+),([-+0-9.eE]+)", s)
            if found:
                rows.extend(found)
            elif len(s.split(",")) == len(header):
                rows.append(s.split(","))
    df = pd.DataFrame(rows, columns=header)
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    for c in ("audio_id", metric, "tx_gain_db"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["audio_id", "tx_gain_db"])
    other = sorted(set(df["method"]) - set(METHODS))
    if other:
        print(f"  {os.path.basename(path)}: ignoring methods {other}")
    return df[df["method"].isin(METHODS)]


def check(df, metric):
    """Print what is in the table and stop on duplicated measurements."""
    dup = df.duplicated(["method", "audio_id", "tx_gain_db"], keep=False)
    if dup.any():
        bad = df[dup].groupby("tx_gain_db")["audio_id"].nunique().to_dict()
        raise SystemExit(
            f"{metric}: the same (method, audio_id, tx_gain_db) appears more than once "
            f"(clips per gain: {bad}). A gain was probably run twice and the RX appended "
            f"to the CSV; delete the rows of the bad run before plotting.")
    counts = df.pivot_table(index="tx_gain_db", columns="method", values="audio_id",
                            aggfunc="nunique").reindex(columns=list(METHODS))
    print(f"  clips per gain and method ({metric}):")
    print("    " + counts.fillna(0).astype(int).to_string().replace("\n", "\n    "))
    regular = [g for g in counts.index if g != REFERENCE_GAIN]
    if len(regular) != EXPECTED_GAINS:
        print(f"  Note: {len(regular)} regular TX gains found, {EXPECTED_GAINS} expected.")
    if REFERENCE_GAIN not in counts.index:
        print(f"  Note: no gain={REFERENCE_GAIN} reference run; no ceiling lines.")


def summary_95ci(df, metric):
    d = df[df["tx_gain_db"] != REFERENCE_GAIN].copy()
    bad = ~np.isfinite(d[metric].to_numpy(dtype=float))
    if bad.any():
        print(f"  Warning: {metric}: skipping {int(bad.sum())} non-finite measurements.")
        d = d.loc[~bad]
    g = d.groupby(["method", "tx_gain_db"])[metric].agg(["mean", "std", "count"]).reset_index()
    g["ci95"] = [t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else 0.0
                 for s, n in zip(g["std"], g["count"])]
    return g


def reference_ceilings(df, metric):
    ref = df[(df["tx_gain_db"] == REFERENCE_GAIN) & np.isfinite(df[metric].to_numpy(dtype=float))]
    means = ref.groupby("method")[metric].mean()
    return {key: float(means[m]) for key, m, *_ in CEILINGS if m in means}


def make_plot(df, metric, ylabel, output_name, ylim, legend_loc):
    g = summary_95ci(df, metric)
    if g.empty:
        raise SystemExit(f"No regular TX-gain data for {metric}.")
    ceilings = reference_ceilings(df, metric)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = []
    for method in METHODS:
        st = STYLES[method]
        s = g[g["method"] == method].sort_values("tx_gain_db")
        if s.empty:
            print(f"  Warning: {metric}: no data for {method!r}; omitting curve.")
            continue
        ax.errorbar(s["tx_gain_db"], s["mean"], yerr=s["ci95"],
                    color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                    linewidth=1.05, markersize=3.1, markeredgewidth=0.45,
                    elinewidth=0.72, capsize=1.8, capthick=0.72, zorder=3)
        handles.append(Line2D([0], [0], color=st["color"], linestyle=st["linestyle"],
                              marker=st["marker"], linewidth=1.05, markersize=3.1,
                              label=st["label"]))
    for key, _, label, color, linestyle in CEILINGS:
        if key in ceilings:
            ax.axhline(ceilings[key], color=color, linestyle=linestyle, linewidth=0.9, zorder=2)
            handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                                  linewidth=0.9, label=label))

    xs = sorted(g["tx_gain_db"].unique())
    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_xlabel("USRP TX gain (dB)")
    ax.set_ylabel(ylabel)

    if ylim is None:   # fit the means, CIs and ceilings
        v = np.concatenate([(g["mean"] - g["ci95"]).to_numpy(float),
                            (g["mean"] + g["ci95"]).to_numpy(float),
                            np.asarray(list(ceilings.values()), float)])
        v = v[np.isfinite(v)]
        lo, hi = float(v.min()), float(v.max())
        pad = 0.07 * (hi - lo) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    else:
        ax.set_ylim(*ylim)

    ax.grid(True, linestyle=":", linewidth=0.45, color="#d0d0d0")
    ax.set_axisbelow(True)
    leg = ax.legend(handles=handles, loc=legend_loc, frameon=True, fancybox=False,
                    framealpha=1.0, borderpad=0.18, labelspacing=0.12, handlelength=1.65,
                    handletextpad=0.32, borderaxespad=0.32)
    leg.get_frame().set_linewidth(0.55)
    leg.get_frame().set_edgecolor("#777777")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.985)

    for ext in ("eps", "pdf", "png"):
        fig.savefig(f"{output_name}.{ext}", **({"dpi": 600} if ext == "png" else {}))
    plt.close(fig)
    print(f"  wrote {output_name}.eps/.pdf/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=".",
                    help="folder with the metric CSVs of the new sweep (default: .)")
    ap.add_argument("--out-dir", default="figures", help="where to write the figures (default: figures)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    plots = (   # metric, y label, file name, y limits (None = fit), legend position
        ("pesq", "PESQ (wideband)", "pesq_vs_tx_gain_95ci_ceiling", None, "upper left"),
        ("estoi", "ESTOI", "estoi_vs_tx_gain_95ci_ceiling", (-0.02, 0.86), "lower right"),
        ("si_sdr_db", "SI-SDR (dB)", "sisdr_vs_tx_gain_95ci_ceiling", None, "lower right"),
    )
    for metric, ylabel, name, ylim, loc in plots:
        path = os.path.join(a.results_dir, CSV_NAMES[metric])
        print(f"{metric}: {path}")
        if not os.path.isfile(path):
            hint = " (run calculate_sisdr.py first)" if metric == "si_sdr_db" else ""
            print(f"  missing, skipped{hint}")
            continue
        df = load(path, metric)
        check(df, metric)
        make_plot(df, metric, ylabel, os.path.join(a.out_dir, name), ylim, loc)


if __name__ == "__main__":
    main()
