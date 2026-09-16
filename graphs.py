import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t

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
    "conv": "M3.1"
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
    "legend.fontsize": 6.2,

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
                r"(proposed|conv|3),(\d+),([-+0-9.eE]+),([-+0-9.eE]+)",
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

    # tx_gain_db=100 is only a sentinel
    # for the maximum-gain ceiling experiment
    d = df[
        df["tx_gain_db"] < 100
    ].copy()

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

    c = (
        df[
            df["tx_gain_db"] == 100
        ]
        .groupby("method")[metric]
        .mean()
    )

    return {
        "coded": float(c["proposed"]),
        "uncoded": float(c["3"]),
    }


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
        "conv"
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

    ax.axhline(
        ceilings["coded"],
        color=CEILING_STYLES[0][0],
        linestyle=CEILING_STYLES[0][1],
        linewidth=0.9,
        zorder=2,
    )

    ax.axhline(
        ceilings["uncoded"],
        color=CEILING_STYLES[1][0],
        linestyle=CEILING_STYLES[1][1],
        linewidth=0.9,
        zorder=2,
    )

    ceiling_handles = [

        Line2D(
            [0], [0],
            color=CEILING_STYLES[0][0],
            linestyle=CEILING_STYLES[0][1],
            linewidth=0.9,
            label="Ceiling @ 89 dB, M1/M3.1",
        ),

        Line2D(
            [0], [0],
            color=CEILING_STYLES[1][0],
            linestyle=CEILING_STYLES[1][1],
            linewidth=0.9,
            label="Ceiling @ 89 dB, M2",
        ),
    ]

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

    ax.set_xticks(xs)

    ax.set_xlim(
        min(xs) - 0.5,
        max(xs) + 0.5
    )

    ax.set_ylim(
        *ylim
    )

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

        borderpad=0.32,
        labelspacing=0.20,

        handlelength=2.25,
        handletextpad=0.42,

        borderaxespad=0.42,
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


# ======================================================
# PESQ
# Legend moved to UPPER LEFT
# ======================================================

make_plot(
    pesq,

    metric="pesq",
    ylabel="PESQ",

    output_name="pesq_vs_tx_gain_95ci_ceiling",

    ylim=(0.98, 2.38),

    ceilings=PESQ_CEILINGS,

    legend_loc="center left",
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
# Keep legend lower right
# ======================================================

make_plot(
    sisdr,

    metric="si_sdr_db",
    ylabel="SI-SDR (dB)",

    output_name="sisdr_vs_tx_gain_95ci_ceiling",

    ylim=(-50, 4.5),

    ceilings=sentinel_ceilings(
        sisdr,
        "si_sdr_db"
    ),

    legend_loc="lower right",
)