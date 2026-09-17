"""Generate all paper figures from the committed result JSONs.

Style choices are for print: Okabe-Ito colourblind-safe palette, distinct
markers and line styles so every series survives greyscale printing, serif
type to match elsarticle, and no decorative ink.
"""
from __future__ import annotations
import json, re
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

# Okabe-Ito: safe for deuteranopia/protanopia, and separable in greyscale.
C_IMM, C_FMM = "#D55E00", "#0072B2"
C_ALT = "#009E73"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# Figures draw on the SAME named studies as verify_claims.py, so a figure can
# never disagree with the prose. See paper/studies.py for why pooling across
# studies is forbidden.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from studies import load as load_study, select, encoder_of  # noqa: E402


def load(patterns):
    """Deprecated path-based loader; kept only for figures not yet migrated."""
    out = []
    for pat in patterns:
        for f in glob(str(ROOT / pat)):
            try:
                out.append(json.load(open(f)))
            except Exception:
                pass
    return out


def clean_acc(r):
    return r.get("cells", {}).get("NONE", {}).get("acc2")


# ---------------------------------------------------------------- Fig 1: dose
def fig_dose():
    recs = load_study("sweep")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, ds, title in zip(axes, ["MOSI", "SIMS"], ["CMU-MOSI", "CH-SIMS"]):
        series = {}
        for r in recs:
            if r["config"]["dataset"] != ds:
                continue
            m = re.match(r"T@([0-9.]+)$", r["config"]["train_regime"])
            a = clean_acc(r)
            if not m or a is None:
                continue
            series.setdefault(r["config"]["protocol"], {}) \
                  .setdefault(float(m.group(1)), []).append(a)
        for proto, colour, marker, ls in (("FMM", C_FMM, "o", "-"),
                                          ("IMM", C_IMM, "s", "--")):
            if proto not in series:
                continue
            xs = sorted(series[proto])
            mu = np.array([np.mean(series[proto][x]) for x in xs])
            sd = np.array([np.std(series[proto][x]) for x in xs])
            lbl = ("modality-level (MOD)" if proto == "FMM"
                   else "token-level (TOK)")
            ax.plot(xs, mu, marker=marker, ls=ls, color=colour, label=lbl)
            ax.fill_between(xs, mu - sd, mu + sd, color=colour, alpha=0.15, lw=0)
        onset = 0.5 if ds == "MOSI" else 0.6
        ax.axvline(onset, color="0.45", lw=0.8, ls=":")
        # The caption defines the dotted line; keep the plotting area uncluttered.
        ax.set_title(title)
        ax.set_xlabel("text erasure rate during training")
        ax.set_xlim(-0.02, 0.92)
    axes[0].set_ylabel("clean-input Acc-2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.16), handlelength=2.2)
    fig.savefig(FIGS / "dose_response.pdf")
    plt.close(fig)
    print("  dose_response.pdf")


# ---------------------------------------------------------------- Fig 2: size
def fig_size():
    # size-response from study "size"; the full-data point from study "main"
    recs = [r for r in load_study("size")
            if r["config"]["protocol"] == "IMM"] + \
           [r for r in load_study("main")
            if r["config"]["dataset"] == "MOSEI"
            and r["config"]["protocol"] == "IMM"
            and not r["config"].get("subsample_train")]
    rows = {}
    for r in recs:
        a = clean_acc(r)
        if a is None:
            continue
        n = r.get("n_train") if r["config"].get("subsample_train") else 16326
        rows.setdefault(r["config"]["train_regime"], {}).setdefault(n, []).append(a)
    ns = sorted(set(rows.get("U-lo", {})) & set(rows.get("T-frag", {})))
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    for reg, colour, marker, ls, lbl in (
            ("U-lo", C_FMM, "o", "-", "benign (10-30% erasure)"),
            ("T-frag", C_IMM, "s", "--", "text-scarce (60-80%)")):
        mu = np.array([np.mean(rows[reg][n]) for n in ns])
        sd = np.array([np.std(rows[reg][n]) for n in ns])
        ax.plot(ns, mu, marker=marker, ls=ls, color=colour, label=lbl)
        ax.fill_between(ns, mu - sd, mu + sd, color=colour, alpha=0.15, lw=0)
    ax.set_xscale("log")
    # Matplotlib's log locator adds minor ticks (2x10^3, 4x10^3 ...) that
    # overprint the explicit labels below; remove them.
    from matplotlib.ticker import NullLocator, NullFormatter
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n/1000:.1f}k".replace(".0k", "k") for n in ns])
    ax.set_xlabel("CMU-MOSEI training examples")
    ax.set_ylabel("clean-input Acc-2")
    ax.legend(frameon=False, loc="lower right", fontsize=10, handlelength=2.2)
    fig.savefig(FIGS / "size_response.pdf")
    plt.close(fig)
    print("  size_response.pdf")


# ------------------------------------------------------------- Fig 3: encoders
def fig_encoders():
    groups = [("CMU-MOSI", "MOSI", "full"), ("CMU-MOSEI\n($n{=}1284$)", "MOSEI", 1284)]
    # bert-base MOSI-full from "main"; bert-base MOSEI@1284 from "size";
    # roberta from "roberta". Named explicitly so the figure is traceable.
    encs = [("bert-base", None), ("roberta-base", None)]
    pools = {"bert-base": load_study("main") + load_study("size"),
             "roberta-base": load_study("roberta")}
    dmg = {}
    for enc, _ in encs:
        for r in pools[enc]:
            c = r["config"]
            name = c.get("bert_name", "auto")
            name = "bert-base" if name == "auto" or name.startswith("bert-base") else name
            if name != enc:
                continue
            a = clean_acc(r)
            if a is None:
                continue
            n = r.get("n_train") if c.get("subsample_train") else "full"
            dmg.setdefault((enc, c["dataset"], n, c["protocol"]), {}) \
               .setdefault(c["train_regime"], {})[c["seed"]] = a

    def damage(enc, ds, n, proto):
        v = dmg.get((enc, ds, n, proto), {})
        if "U-lo" not in v or "T-frag" not in v:
            return None
        seeds = sorted(set(v["T-frag"]) & set(v["U-lo"]))
        delta = [100 * (v["T-frag"][s] - v["U-lo"][s]) for s in seeds]
        return float(np.mean(delta)), float(np.std(delta))

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    w, xs = 0.19, np.arange(len(groups))
    bars = [("bert-base", "IMM", C_IMM, "", "BERT, token-level"),
            ("roberta-base", "IMM", C_IMM, "//", "RoBERTa, token-level"),
            ("bert-base", "FMM", C_FMM, "", "BERT, modality-level"),
            ("roberta-base", "FMM", C_FMM, "//", "RoBERTa, modality-level")]
    for i, (enc, proto, colour, hatch, lbl) in enumerate(bars):
        pos = xs + (i - 1.5) * w
        first = True
        for x, (_, ds, n) in zip(pos, groups):
            v = damage(enc, ds, n, proto)
            if v is None:
                # Never render a missing measurement as a zero bar: that would
                # assert "no damage" where nothing was measured.
                ax.text(x, -1.0, "n/a", ha="center", va="top", fontsize=10,
                        color="0.55", rotation=90)
                continue
            mean, sd = v
            ax.bar(x, mean, w, yerr=sd, capsize=2, color=colour,
                   hatch=hatch, edgecolor="white", error_kw={"elinewidth": 0.8},
                   linewidth=0.5, label=lbl if first else None)
            first = False
            label_y = min(mean - sd - 1.2, -5.0) if proto == "FMM" else mean - sd - 1.2
            ax.text(x, label_y, f"{mean:.1f}", ha="center", va="top",
                    fontsize=10, color="0.15")
        if first:  # nothing plotted for this series; keep it in the legend
            ax.bar([np.nan], [0], w, color=colour, hatch=hatch,
                   edgecolor="white", linewidth=0.5, label=lbl)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("clean-input Acc-2 change (pts)")
    ax.set_ylim(-35, 4)
    # Legend below the axes: inside the panel it overprinted the MOSI bars.
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), handlelength=1.6, columnspacing=1.2)
    fig.savefig(FIGS / "encoders.pdf")
    plt.close(fig)
    print("  encoders.pdf")


if __name__ == "__main__":
    print("generating figures:")
    fig_dose(); fig_size(); fig_encoders()
