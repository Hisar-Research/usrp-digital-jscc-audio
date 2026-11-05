import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t
from matplotlib.ticker import MaxNLocator

PESQ_CSV  = "pesq_results_superpacket_soft_txgain.csv"
ESTOI_CSV = "estoi_results_superpacket_soft_txgain.csv"
SISDR_CSV = "sisdr_results_superpacket_soft_txgain.csv"

ROLE_STYLES = {
    "M1": {
        "label": "Proposed",
        "marker": "o",
        "color": "#1f77b4",
        "linestyle": "-"
    },
    "M2": {
        "label": "Uncoded",
        "marker": "s",
        "color": "#7f7f7f",
        "linestyle": "--"
    },
    "M3.1": {
        "label": "Conv-coded",
        "marker": "^",
        "color": "#2ca02c",
        "linestyle": "-."
    },
    "M3.2": {
        "label": "LDPC-coded",
        "marker": "D",
        "color": "#9467bd",
        "linestyle": (0, (3, 1, 1, 1))
    },
    "M3.1_hard": {
        "label": "Conv-coded, hard",
        "marker": "v",
        "color": "#8c564b",
        "linestyle": ":"
    },
    "M3.1_soft": {
        "label": "Conv-coded, soft",
        "marker": "v",
        "color": "#d62728",
        "linestyle": ":"
    },
}

CEILING_STYLES = [
    ("#000000", (0, (1, 1))),
    ("#555555", (0, (4, 2))),
    ("#999999", (0, (2, 2))),
]

METHOD_TO_ROLE = {
    "proposed": "M1",
    "3": "M2",
    "conv": "M3.1",
    "ldpc": "M3.2",
}

# Same physical canvas as reference EPS
FIGSIZE = (
    228.195625 / 72.0,
    170.024 / 72.0
)

# PESQ ceilings from maximum-TX-gain experiment
PESQ_CEILINGS = {
    "coded": 1.9706,     # Proposed / Conv-coded
    "uncoded": 2.2551,   # Uncoded
}

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


def read_estoi_repaired(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")

        for line in f:
            s = line.strip()

            if not s:
                continue

            found = re.findall(
                r"(proposed|conv|ldpc|3),(\d+),([-+0-9.eE]+),([-+0-9.eE]+)",
                s,
            )

            if found:
                rows.extend(found)

            else:
                p = s.split(",")

                if len(p) == 4:
                    rows.append(p)

    df = pd.DataFrame(
        rows,
        columns=header
    )

    for c in header[1:]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    return df.dropna()


def summary_95ci(df, metric):

    # Keep the 89-dB reference measurement out of the regular TX-gain curves.
    # It is used only to draw horizontal ceiling lines.
    # 89 is the real reference gain; 100 was the old sentinel label.
    # Neither is a regular data point on the gain axis.
    d = df[~df["tx_gain_db"].isin([89, 100])].copy()

    # Metrics may contain NaN or infinities if a decoded waveform was invalid.
    # Do not turn invalid numbers into misleading plot points.
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    bad = ~np.isfinite(d[metric].to_numpy(dtype=float))
    if bad.any():
        print(f"Warning: {metric}: skipping {int(bad.sum())} non-finite measurements.")
        d = d.loc[~bad]

    g = (
        d.groupby(
            ["method", "tx_gain_db"]
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    g["ci95"] = g.apply(
        lambda r:
            (
                t.ppf(
                    0.975,
                    int(r["count"]) - 1
                )
                * r["std"]
                / np.sqrt(r["count"])
            )
            if r["count"] > 1
            else 0.0,
        axis=1,
    )

    return g


def sentinel_ceilings(df, metric):
    """Return available reference ceilings, without inventing absent data.

    Gain=89 is the separate reference experiment.
    Its measurements are used for horizontal ceilings, not TX-gain points.
    """
    sentinel = df.loc[df["tx_gain_db"] == 89].copy()
    sentinel[metric] = pd.to_numeric(sentinel[metric], errors="coerce")
    sentinel = sentinel.loc[np.isfinite(sentinel[metric].to_numpy(dtype=float))]
    c = sentinel.groupby("method")[metric].mean()

    ceilings = {}
    for key, method in (("coded", "proposed"), ("uncoded", "3")):
        value = c.get(method, np.nan)
        if pd.notna(value) and np.isfinite(value):
            ceilings[key] = float(value)
        else:
            print(
                f"Warning: {metric}: no gain=89 reference measurement "
                f"for method {method!r}; skipping that ceiling."
            )

    if sentinel.empty:
        print(f"  Available gains in {metric}: {sorted(df['tx_gain_db'].dropna().unique())}")
    else:
        print(f"  Methods recorded at gain=89 for {metric}: {list(c.index)}")
    return ceilings


def make_plot(
    df,
    metric,
    ylabel,
    output_name,
    ylim,
    ceilings,
    legend_loc="lower right",
):

    g = summary_95ci(
        df,
        metric
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    method_handles = []

    # --------------------------------------------------
    # Main methods
    # --------------------------------------------------

    for method in [
        "proposed",
        "3",
        "conv",
        "ldpc"
    ]:

        st = ROLE_STYLES[
            METHOD_TO_ROLE[method]
        ]

        s = (
            g[
                g["method"] == method
            ]
            .sort_values("tx_gain_db")
        )

        if s.empty:
            print(f"Warning: {metric}: no data for method {method!r}; omitting curve.")
            continue

        ax.errorbar(
            s["tx_gain_db"],
            s["mean"],
            yerr=s["ci95"],

            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],

            linewidth=1.05,
            markersize=3.1,
            markeredgewidth=0.45,

            elinewidth=0.72,
            capsize=1.8,
            capthick=0.72,

            zorder=3,
        )

        method_handles.append(
            Line2D(
                [0], [0],

                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],

                linewidth=1.05,
                markersize=3.1,

                label=st["label"],
            )
        )

    # --------------------------------------------------
    # Error-free / maximum-gain ceilings
    # --------------------------------------------------

    ceiling_handles = []
    for key, idx, label in (
        ("coded", 0, "Ceiling @ 89 dB, M1/M3.1/M3.2"),
        ("uncoded", 1, "Ceiling @ 89 dB, M2"),
    ):
        if key not in ceilings:
            continue

        color, linestyle = CEILING_STYLES[idx]
        ax.axhline(
            ceilings[key],
            color=color,
            linestyle=linestyle,
            linewidth=0.9,
            zorder=2,
        )
        ceiling_handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle=linestyle,
                linewidth=0.9,
                label=label,
            )
        )

    # --------------------------------------------------
    # Axes
    # --------------------------------------------------

    ax.set_xlabel(
        "USRP TX gain (dB)"
    )

    ax.set_ylabel(
        ylabel
    )

    xs = sorted(
        g["tx_gain_db"].unique()
    )
    if not len(xs):
        raise ValueError(f"No regular TX-gain data found for {metric}; check the CSV.")

    ax.set_xticks(xs)

    ax.set_xlim(
        min(xs) - 0.5,
        max(xs) + 0.5
    )

    if ylim is None:
        # Fit the y-axis to the visible means, confidence intervals and
        # the 89-dB reference ceilings. No hard-coded [-50, 4.5] clipping.
        lows = (g["mean"] - g["ci95"]).to_numpy(dtype=float)
        highs = (g["mean"] + g["ci95"]).to_numpy(dtype=float)
        limits = np.concatenate([
            lows, highs,
            np.asarray(list(ceilings.values()), dtype=float)
        ])
        limits = limits[np.isfinite(limits)]
        if limits.size:
            lo, hi = float(limits.min()), float(limits.max())
            pad = max(1.0, 0.07 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    else:
        ax.set_ylim(*ylim)

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.45,
        color="#d0d0d0",
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    leg = ax.legend(
        handles=(
            method_handles
            + ceiling_handles
        ),

        loc=legend_loc,

        frameon=True,
        fancybox=False,
        framealpha=1.0,

        borderpad=0.18,
        labelspacing=0.12,

        handlelength=1.65,
        handletextpad=0.32,

        borderaxespad=0.32,
    )

    leg.get_frame().set_linewidth(
        0.55
    )

    leg.get_frame().set_edgecolor(
        "#777777"
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.20,
        top=0.985,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    fig.savefig(
        output_name + ".eps",
        format="eps"
    )

    fig.savefig(
        output_name + ".pdf"
    )

    fig.savefig(
        output_name + ".png",
        dpi=600
    )

    plt.close(fig)


# ======================================================
# Load data
# ======================================================

pesq = pd.read_csv(
    PESQ_CSV
)

estoi = read_estoi_repaired(
    ESTOI_CSV
)

sisdr = pd.read_csv(
    SISDR_CSV
)

for dataset in (pesq, estoi, sisdr):
    dataset.columns = dataset.columns.str.strip()
    dataset["method"] = dataset["method"].astype(str).str.strip().str.lower()
    dataset["tx_gain_db"] = pd.to_numeric(dataset["tx_gain_db"], errors="coerce")

# Data diagnostics: the plotting code cannot repair incorrectly aligned audio.
print("Methods in SI-SDR CSV:", sorted(sisdr["method"].unique()))
print("TX gains in SI-SDR CSV:", sorted(sisdr["tx_gain_db"].dropna().unique()))
regular_sisdr = pd.to_numeric(
    sisdr.loc[sisdr["tx_gain_db"] != 89, "si_sdr_db"], errors="coerce"
)
if (regular_sisdr < -60).any():
    print("Note: SI-SDR contains values below -60 dB. "
          "Verify audio alignment and the SI-SDR calculation; "
          "a plot cannot fix incorrect metric values.")


# ======================================================
# PESQ
# Compact legend in the upper left, matching the three-panel figure
# ======================================================

make_plot(
    pesq,

    metric="pesq",
    ylabel="PESQ",

    output_name="pesq_vs_tx_gain_95ci_ceiling",

    ylim=(0.98, 2.38),

    ceilings=PESQ_CEILINGS,

    legend_loc="upper left",
)


# ======================================================
# ESTOI
# Keep legend lower right
# ======================================================

make_plot(
    estoi,

    metric="estoi",
    ylabel="ESTOI",

    output_name="estoi_vs_tx_gain_95ci_ceiling",

    ylim=(-0.02, 0.86),

    ceilings=sentinel_ceilings(
        estoi,
        "estoi"
    ),

    legend_loc="lower right",
)


# ======================================================
# SI-SDR
# Compact legend lower right
# ======================================================

make_plot(
    sisdr,

    metric="si_sdr_db",
    ylabel="SI-SDR (dB)",

    output_name="sisdr_vs_tx_gain_95ci_ceiling",

    ylim=None,  # SI-SDR can extend below -50 dB; fit real values instead.

    ceilings=sentinel_ceilings(
        sisdr,
        "si_sdr_db"
    ),

    legend_loc="lower right",
)