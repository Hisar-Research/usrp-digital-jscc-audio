"""
Figures and tables for selftest_sim_v3.py.

Reads P_rows.csv, P_ceil.csv and (optionally) P_summary.json written with
--out_prefix P, and makes:

  fig_<scenario>.pdf/png        one per scenario: PESQ, ESTOI, SI-SDR and
                                codeword accuracy vs Es/N0, all methods, with
                                bootstrap confidence bands and codec ceilings
  overview_<metric>.pdf/png     one panel per channel; colour = method, line
                                style = SNR mismatch (solid = perfect)
  mismatch_delta_<metric>       paired change caused by the SNR mismatch,
                                metric(mismatched) - metric(perfect), per method
  fading_penalty_<metric>       paired change from AWGN to Rayleigh at the same
                                average Es/N0 (perfect SNR knowledge)
  P_table.txt                   ESTOI / PESQ / accuracy at selected SNRs

Every value is first averaged over trials within an utterance, then over
utterances; bands are utterance-level percentile bootstraps of the mean. The
differences are paired per utterance, which the common random numbers of the
simulation make exact (same noise and fading in both scenarios).

A PESQ call that failed (silent or degenerate output) is stored as an empty
cell; it is counted as --pesq_fail (default 1.0, the bottom of the MOS-LQO
scale) instead of being dropped, which would average only over the
utterances that still produced speech and flatter the failing method.

Usage:
    python plot_sim_v3.py --prefix vogeo_sim_v3 --out_dir figs
    python plot_sim_v3.py --prefix vogeo_sim_v3 --out_dir figs --fmt pdf --ci 95 \\
        --summary_metrics estoi pesq code_acc --methods M1 M3.1 M3.2
"""

import argparse
import csv
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ("M1", "M3.1", "M3.2", "M2")
# categorical slots 1-4 of the reference palette, validated for adjacent-pair
# CVD separation; identity is also carried by the marker, so it is never
# colour alone
STYLE = {
    "M1":   {"label": "Proposed (M1)",         "color": "#2a78d6", "marker": "o"},
    "M3.1": {"label": "Hamming + SBSD (M3.1)", "color": "#eb6834", "marker": "^"},
    "M3.2": {"label": "LDPC 3/4 (M3.2)",       "color": "#1baf7a", "marker": "D"},
    "M2":   {"label": "Uncoded (M2)",          "color": "#eda100", "marker": "s"},
}
METRICS = {"pesq": "PESQ", "estoi": "ESTOI", "sisdr": "SI-SDR (dB)",
           "code_acc": "Codeword accuracy"}
CHANNEL_NAME = {"awgn": "AWGN", "rayleigh": "Rayleigh, perfect CSI",
                "rayleigh_nocsi": "Rayleigh, no CSI"}
CHANNEL_ORDER = ("awgn", "rayleigh", "rayleigh_nocsi")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CEIL_COLOR = {3: "#52514e", 4: "#a3a29c"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def mismatch_style(m):
    if m == 0:
        return "-"
    return (0, (5, 2)) if m > 0 else (0, (1, 1.4))


def mismatch_label(m):
    return "perfect SNR" if m == 0 else f"assumed SNR {m:+g} dB"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# =========================================================
# DATA
# =========================================================
class Results:
    def __init__(self, prefix, pesq_fail, ci, n_boot=2000, seed=0):
        self.ci, self.n_boot = ci, n_boot
        self.rng = np.random.default_rng(seed)
        acc = {}                          # (scen, method, snr, metric) -> {file: [values]}
        self.scen = {}                    # scenario -> (channel, mismatch)
        with open(f"{prefix}_rows.csv", newline="") as f:
            for r in csv.DictReader(f):
                s = r["scenario"]
                self.scen[s] = (r["channel"], _num(r["mismatch_db"]))
                snr = _num(r["snr"])
                for k in list(METRICS) + ["ml_acc"]:
                    v = _num(r.get(k))
                    if k == "pesq" and not math.isfinite(v) and pesq_fail is not None:
                        v = pesq_fail
                    if k == "ml_acc" and not math.isfinite(v):
                        continue
                    acc.setdefault((s, r["method"], snr, k), {}) \
                       .setdefault(r["file"], []).append(v)
        self.val = {key: {f: float(np.mean(v)) for f, v in d.items()}
                    for key, d in acc.items()}
        self.snrs = sorted({k[2] for k in self.val})
        self.methods = [m for m in METHODS if any(k[1] == m for k in self.val)]
        self.ceil = {}
        path = f"{prefix}_ceil.csv"
        if os.path.exists(path):
            per = {}
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    for k in ("pesq", "estoi", "sisdr"):
                        v = _num(r[k])
                        if k == "pesq" and not math.isfinite(v) and pesq_fail is not None:
                            v = pesq_fail
                        per.setdefault((int(_num(r["n_q"])), k), []).append(v)
            self.ceil = {key: float(np.nanmean(v)) for key, v in per.items()}
        self.meta = {}
        if os.path.exists(f"{prefix}_summary.json"):
            self.meta = json.load(open(f"{prefix}_summary.json")).get("_meta", {})

    def scenarios(self):
        def key(s):
            ch, m = self.scen[s]
            o = CHANNEL_ORDER.index(ch) if ch in CHANNEL_ORDER else 99
            return (o, m != 0, -m if m > 0 else 100 - m)
        return sorted(self.scen, key=key)

    def find(self, channel, mismatch):
        return next((s for s, v in self.scen.items() if v == (channel, mismatch)), None)

    def _boot(self, v):
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return (float("nan"),) * 3
        m = float(v.mean())
        if len(v) < 2:
            return m, m, m
        bm = v[self.rng.integers(0, len(v), (self.n_boot, len(v)))].mean(axis=1)
        a = (100.0 - self.ci) / 2.0
        lo, hi = np.percentile(bm, [a, 100.0 - a])
        return m, float(lo), float(hi)

    def curve(self, scen, method, metric):
        out = []
        for s in self.snrs:
            d = self.val.get((scen, method, s, metric))
            out.append(self._boot(np.array(list(d.values()))) if d else (float("nan"),) * 3)
        return np.array(self.snrs), np.array(out).T

    def delta(self, scen_a, scen_b, method, metric):
        """metric(scen_a) - metric(scen_b), paired per utterance."""
        out = []
        for s in self.snrs:
            a = self.val.get((scen_a, method, s, metric), {})
            b = self.val.get((scen_b, method, s, metric), {})
            common = sorted(set(a) & set(b))
            out.append(self._boot(np.array([a[f] - b[f] for f in common]))
                       if common else (float("nan"),) * 3)
        return np.array(self.snrs), np.array(out).T


# =========================================================
# DRAWING
# =========================================================
def _line(ax, x, y, method, ls="-", band=None, label=None, lw=None, marker=True):
    st = STYLE[method]
    ok = np.isfinite(y)
    if band is not None:
        lo, hi = band
        okb = ok & np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(x[okb], lo[okb], hi[okb], color=st["color"], alpha=0.16, lw=0)
    ax.plot(x[ok], y[ok], color=st["color"], ls=ls, lw=lw or plt.rcParams["lines.linewidth"],
            marker=st["marker"] if marker else None, label=label,
            markeredgecolor="white", markeredgewidth=0.5)


def _xaxis(ax, snrs, channel):
    ax.set_xlabel("Average $E_s/N_0$ (dB)" if channel != "awgn" else "$E_s/N_0$ (dB)")
    step = 1 if len(snrs) <= 8 else 2
    ticks = [s for s in snrs if float(s).is_integer() and int(s) % step == 0] or snrs
    ax.set_xticks(ticks)


def _metric_limits(ax, metric):
    if metric == "code_acc":
        ax.set_ylim(0, 1.02)
    elif metric == "estoi":
        ax.set_ylim(bottom=0)
    elif metric == "pesq":
        ax.set_ylim(bottom=1.0)


def _save(fig, out_dir, name, fmts):
    for fmt in fmts:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"), dpi=200)
    plt.close(fig)
    return name


def fig_scenario(R, scen, methods, out_dir, fmts, show_ml=True):
    ch, mis = R.scen[scen]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))
    for ax, metric in zip(axes.ravel(), ("pesq", "estoi", "sisdr", "code_acc")):
        for m in methods:
            x, (y, lo, hi) = R.curve(scen, m, metric)
            _line(ax, x, y, m, band=(lo, hi), label=STYLE[m]["label"])
        if metric == "code_acc" and show_ml and "M3.1" in methods:
            x, (y, _, _) = R.curve(scen, "M3.1", "ml_acc")
            if np.isfinite(y).any():
                _line(ax, x, y, "M3.1", ls=(0, (4, 2)), lw=0.9, marker=False,
                      label="Hamming, no prior (ML)")
        for q in (3, 4):
            if (q, metric) in R.ceil:
                ax.axhline(R.ceil[(q, metric)], color=CEIL_COLOR[q], ls=":", lw=1.0,
                           label=f"Ceiling, $n_q={q}$")
        ax.set_ylabel(METRICS[metric])
        _metric_limits(ax, metric)
        _xaxis(ax, R.snrs, ch)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.tight_layout(h_pad=1.2, w_pad=1.5, rect=(0, 0, 1, 0.87))
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.875))
    fig.text(0.5, 0.995, f"{CHANNEL_NAME.get(ch, ch)}, {mismatch_label(mis)}",
             ha="center", va="top", fontsize=9, color=INK)
    return _save(fig, out_dir, f"fig_{scen}", fmts)


def _channels(R):
    return [c for c in CHANNEL_ORDER if any(v[0] == c for v in R.scen.values())]


def _mismatches(R, ch):
    ms = sorted({v[1] for v in R.scen.values() if v[0] == ch},
                key=lambda m: (m != 0, -m if m > 0 else 100 - m))
    return ms


def _legends(fig, methods, mismatches, n_panels):
    mh = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"],
                 markeredgecolor="white", label=STYLE[m]["label"]) for m in methods]
    sh = [Line2D([], [], color=INK2, ls=mismatch_style(m), label=mismatch_label(m))
          for m in mismatches]
    narrow = n_panels == 1
    two = len(sh) > 1
    y_methods = (1.0 + (0.16 if narrow else 0.07)) if two else 1.0
    l1 = fig.legend(handles=mh, loc="lower center", bbox_to_anchor=(0.5, y_methods),
                    ncol=2 if narrow else len(mh))
    fig.add_artist(l1)
    if two:
        fig.legend(handles=sh, loc="lower center", ncol=1 if narrow else len(sh),
                   bbox_to_anchor=(0.5, 1.0))


def fig_overview(R, metric, methods, out_dir, fmts):
    chans = _channels(R)
    fig, axes = plt.subplots(1, len(chans), figsize=(3.5 * len(chans) + 0.2, 2.9),
                             sharey=True, squeeze=False)
    all_mis = []
    for ax, ch in zip(axes[0], chans):
        mis = _mismatches(R, ch)
        all_mis += [m for m in mis if m not in all_mis]
        for m in methods:
            for mm in mis:
                scen = R.find(ch, mm)
                x, (y, lo, hi) = R.curve(scen, m, metric)
                _line(ax, x, y, m, ls=mismatch_style(mm),
                      band=(lo, hi) if mm == 0 else None, lw=1.3 if mm == 0 else 1.0)
        for q in (3, 4):
            if (q, metric) in R.ceil:
                ax.axhline(R.ceil[(q, metric)], color=CEIL_COLOR[q], ls=":", lw=0.9)
        ax.set_title(CHANNEL_NAME.get(ch, ch), color=INK)
        _metric_limits(ax, metric)
        _xaxis(ax, R.snrs, ch)
    axes[0, 0].set_ylabel(METRICS[metric])
    fig.tight_layout(w_pad=1.5)
    _legends(fig, methods, sorted(all_mis, key=lambda m: (m != 0, -m if m > 0 else 100 - m)),
             len(chans))
    return _save(fig, out_dir, f"overview_{metric}", fmts)


def fig_mismatch_delta(R, metric, methods, out_dir, fmts):
    chans = [c for c in _channels(R) if R.find(c, 0.0) and len(_mismatches(R, c)) > 1]
    if not chans:
        return None
    fig, axes = plt.subplots(1, len(chans), figsize=(3.5 * len(chans) + 0.2, 2.9),
                             sharey=True, squeeze=False)
    all_mis = []
    for ax, ch in zip(axes[0], chans):
        ref = R.find(ch, 0.0)
        for mm in _mismatches(R, ch):
            if mm == 0:
                continue
            all_mis.append(mm) if mm not in all_mis else None
            for m in methods:
                x, (y, lo, hi) = R.delta(R.find(ch, mm), ref, m, metric)
                _line(ax, x, y, m, ls=mismatch_style(mm), band=(lo, hi), lw=1.1)
        ax.axhline(0.0, color=INK2, lw=0.7)
        ax.set_title(CHANNEL_NAME.get(ch, ch), color=INK)
        _xaxis(ax, R.snrs, ch)
    axes[0, 0].set_ylabel(f"$\\Delta$ {METRICS[metric]}, mismatched $-$ perfect")
    fig.tight_layout(w_pad=1.5)
    _legends(fig, methods, sorted(all_mis, key=lambda m: -m), len(chans))
    return _save(fig, out_dir, f"mismatch_delta_{metric}", fmts)


def fig_fading_penalty(R, metric, methods, out_dir, fmts):
    ref = R.find("awgn", 0.0)
    chans = [c for c in _channels(R) if c != "awgn" and R.find(c, 0.0)]
    if ref is None or not chans:
        return None
    fig, axes = plt.subplots(1, len(chans), figsize=(3.5 * len(chans) + 0.2, 2.9),
                             sharey=True, squeeze=False)
    for ax, ch in zip(axes[0], chans):
        for m in methods:
            x, (y, lo, hi) = R.delta(R.find(ch, 0.0), ref, m, metric)
            _line(ax, x, y, m, band=(lo, hi))
        ax.axhline(0.0, color=INK2, lw=0.7)
        ax.set_title(f"{CHANNEL_NAME.get(ch, ch)} $-$ AWGN", color=INK)
        _xaxis(ax, R.snrs, ch)
    axes[0, 0].set_ylabel(f"$\\Delta$ {METRICS[metric]}")
    fig.tight_layout(w_pad=1.5)
    _legends(fig, methods, [0.0], len(chans))
    return _save(fig, out_dir, f"fading_penalty_{metric}", fmts)


# =========================================================
# TABLE
# =========================================================
def table(R, methods, snrs, metrics=("estoi", "pesq", "code_acc")):
    lines = []
    for scen in R.scenarios():
        ch, mis = R.scen[scen]
        lines.append(f"\n{CHANNEL_NAME.get(ch, ch)}, {mismatch_label(mis)}  [{scen}]")
        for metric in metrics:
            lines.append(f"  {METRICS[metric]} (mean, {R.ci:g}% CI half-width)")
            lines.append("  " + f"{'Es/N0':>7s}" + "".join(f"{m:>16s}" for m in methods))
            for s in snrs:
                cells = []
                for m in methods:
                    d = R.val.get((scen, m, s, metric))
                    if not d:
                        cells.append(f"{'-':>16s}"); continue
                    mean, lo, hi = R._boot(np.array(list(d.values())))
                    cells.append(f"{mean:>9.3f} ±{(hi - lo) / 2:<5.3f}")
                lines.append("  " + f"{s:>+7g}" + "".join(cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefix", default="vogeo_sim_v3")
    ap.add_argument("--out_dir", default="figs")
    ap.add_argument("--fmt", nargs="+", default=["pdf", "png"])
    ap.add_argument("--ci", type=float, default=90.0, help="bootstrap interval, percent")
    ap.add_argument("--pesq_fail", default="1.0",
                    help="value for failed PESQ calls, or 'nan' to drop them")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--summary_metrics", nargs="+", default=["estoi", "pesq"],
                    choices=list(METRICS))
    ap.add_argument("--table_snrs", type=float, nargs="+", default=[-5, -3, 0, 2, 5, 8])
    ap.add_argument("--no_ml", action="store_true",
                    help="hide the prior-free (ML) accuracy of the M3.1 block codes")
    args = ap.parse_args()

    pesq_fail = None if args.pesq_fail.lower() == "nan" else float(args.pesq_fail)
    R = Results(args.prefix, pesq_fail, args.ci)
    methods = [m for m in args.methods if m in R.methods]
    os.makedirs(args.out_dir, exist_ok=True)

    made = [fig_scenario(R, s, methods, args.out_dir, args.fmt, show_ml=not args.no_ml)
            for s in R.scenarios()]
    for metric in args.summary_metrics:
        made.append(fig_overview(R, metric, methods, args.out_dir, args.fmt))
        made.append(fig_mismatch_delta(R, metric, methods, args.out_dir, args.fmt))
        made.append(fig_fading_penalty(R, metric, methods, args.out_dir, args.fmt))

    snrs = [s for s in args.table_snrs if s in R.snrs] or R.snrs
    txt = table(R, methods, snrs)
    with open(f"{args.prefix}_table.txt", "w") as f:
        f.write(txt.lstrip("\n") + "\n")
    print(txt)
    made = [m for m in made if m]
    print(f"\n{len(made)} figure(s) in {args.out_dir}/ ({', '.join(args.fmt)}): "
          + ", ".join(made))
    print(f"table: {args.prefix}_table.txt")


if __name__ == "__main__":
    main()
