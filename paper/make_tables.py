"""Emit every LaTeX table from the study registry.

Tables are generated, never hand-typed, for the same reason figures are: a
number that exists in two places will eventually disagree in one of them.
Output goes to paper/tables/*.tex and is \\input{} by main.tex.

Style follows the journal conventions: booktabs rules only, no vertical lines,
metric direction in the header, consistent precision within a column.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studies import clean_acc, clean_mae, encoder_of, load, select  # noqa: E402

TAB = Path(__file__).resolve().parent / "tables"
TAB.mkdir(exist_ok=True)
REG = ["U-lo", "U-hi", "T-frag", "NV-frag"]
PRETTY = {"MOSI": "CMU-MOSI", "SIMS": "CH-SIMS", "MOSEI": "CMU-MOSEI"}


def write(name: str, body: str):
    (TAB / name).write_text(body.rstrip() + "\n")
    print(f"  tables/{name}")


def ms(vals):
    return float(np.mean(vals)), float(np.std(vals))


def degradation(recs, **eq):
    u_recs = [r for r in select(recs, train_regime="U-lo", **eq)
              if clean_acc(r) is not None]
    t_recs = [r for r in select(recs, train_regime="T-frag", **eq)
              if clean_acc(r) is not None]
    u_by_seed = {r["config"]["seed"]: clean_acc(r) for r in u_recs}
    t_by_seed = {r["config"]["seed"]: clean_acc(r) for r in t_recs}
    seeds = sorted(set(u_by_seed) & set(t_by_seed))
    if not seeds:
        return None
    u = [u_by_seed[s] for s in seeds]
    t = [t_by_seed[s] for s in seeds]
    delta = [100 * (t_by_seed[s] - u_by_seed[s]) for s in seeds]
    return dict(u=ms(u), t=ms(t), d=ms(delta), n=len(seeds))


# ---------------------------------------------------- T1 backbone competitive
def t1():
    recs = load("main")
    rows = []
    for ds in ["MOSI", "MOSEI", "SIMS"]:
        for proto, label in [("IMM", r"\tokenlevel{}"), ("FMM", r"\modlevel{}")]:
            sel = select(recs, dataset=ds, train_regime="U-lo",
                         protocol=proto, encoder="bert-base", n_train="full")
            a = [clean_acc(r) for r in sel if clean_acc(r) is not None]
            m = [clean_mae(r) for r in sel if clean_mae(r) is not None]
            if not a:
                continue
            rows.append(f"{PRETTY[ds]} & {label} & {np.mean(a):.3f} $\\pm$ {np.std(a):.3f} & "
                        f"{np.mean(m):.3f} $\\pm$ {np.std(m):.3f} \\\\")
    write("backbone.tex", r"""
\begin{tabular}{llcc}
\toprule
Dataset & Training protocol & Acc-2 $\uparrow$ & MAE $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


# ------------------------------------------------------------ T2 degradation
def t2():
    main, sz, rb = load("main"), load("size"), load("roberta")
    lines = []
    for enc, label in [("bert-base", "BERT"), ("roberta-base", "RoBERTa")]:
        src_main = main if enc == "bert-base" else rb
        first = True
        for ds, n, src in [("MOSI", "full", src_main),
                           ("MOSEI", 1284, sz if enc == "bert-base" else rb),
                           ("SIMS", "full", src_main),
                           ("MOSEI", "full", src_main)]:
            kw = dict(dataset=ds, protocol=None)
            tok = degradation(src, dataset=ds, protocol="IMM", n_train=n) \
                if enc == "bert-base" or src is rb else None
            mod = degradation(src, dataset=ds, protocol="FMM", n_train=n) \
                if enc == "bert-base" or src is rb else None
            if enc == "bert-base" and src is not main and src is not sz:
                continue
            if tok is None and mod is None:
                continue
            tag = PRETTY[ds] + ("" if n == "full" else f" ($n{{=}}{n}$)")
            cell = label if first else ""
            first = False
            def fmt(x):
                return "--" if x is None else f"{x['d'][0]:+.1f} $\\pm$ {x['d'][1]:.1f}"
            def acc(x):
                return "--" if x is None else f"{x['u'][0]:.3f} $\\pm$ {x['u'][1]:.3f}"
            lines.append(f"{cell} & {tag} & {acc(tok)} & {fmt(tok)} & {fmt(mod)} \\\\")
        lines.append(r"\addlinespace")
    body = "\n".join(lines).rstrip("\\addlinespace").rstrip()
    write("degradation.tex", r"""
\begin{tabular}{llccc}
\toprule
 & & benign & \multicolumn{2}{c}{clean-input Acc-2 change (pts)} \\
\cmidrule(lr){4-5}
Text encoder & Dataset & Acc-2 & \tokenlevel{} & \modlevel{} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}""")


# -------------------------------------------------------------- T3 shift gap
def t3():
    def gap(recs, ds, proto, exclude=()):
        cells = {}
        for r in select(recs, dataset=ds, protocol=proto, n_train="full"):
            if encoder_of(r) != "bert-base":
                continue
            tr = r["config"]["train_regime"]
            for te, cell in r.get("cells", {}).items():
                if te != "NONE":
                    cells.setdefault((tr, te), []).append(cell["acc2"])
        rows = [x for x in REG if x not in exclude]
        per = []
        for te in rows:
            if (te, te) not in cells:
                continue
            oth = [np.mean(cells[(tr, te)]) for tr in rows
                   if tr != te and (tr, te) in cells]
            if oth:
                per.append(np.mean(cells[(te, te)]) - np.mean(oth))
        return float(np.mean(per)) if per else None

    main, fz = load("main"), load("frozen")
    lines = []
    for ds in ["MOSI", "MOSEI", "SIMS"]:
        vals = []
        for proto in ["IMM", "FMM"]:
            g = gap(main, ds, proto)
            gx = gap(main, ds, proto, ("T-frag",))
            vals += [f"{100*g:+.2f}" if g is not None else "--",
                     f"{100*gx:+.2f}" if gx is not None else "--"]
        lines.append(f"{PRETTY[ds]} & " + " & ".join(vals) + r" \\")
    lines.append(r"\addlinespace")
    gi, gf = gap(fz, "MOSI", "IMM"), gap(fz, "MOSI", "FMM")
    lines.append(f"CMU-MOSI, frozen enc. & "
                 f"{100*gi:+.2f} & -- & {100*gf:+.2f} & -- \\\\")
    write("shift_gap.tex", r"""
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{\tokenlevel{}} & \multicolumn{2}{c}{\modlevel{}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Dataset & all rows & excl. degraded & all rows & excl. degraded \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ------------------------------------------------------------ T4 mitigations
def t4():
    main, mt = load("main"), load("mitig")
    lines = []
    for ds in ["MOSI", "SIMS"]:
        b = degradation(main, dataset=ds, protocol="IMM",
                        encoder="bert-base", n_train="full")
        rows = [("\\tokenlevel{} baseline", b)]
        for label, d, proto in [
                (r"M1: \modlevel{} training", "results_mitig_crossproto", "FMM"),
                (r"M2: freeze embed.\ $+$ 6 blk.", "results_mitig_freeze", "IMM"),
                (r"M3: encoder LR $5\times10^{-6}$", "results_mitig_lowlr", "IMM")]:
            sub = [r for r in mt if d in r["_file"]]
            rows.append((label, degradation(sub, dataset=ds, protocol=proto)))
        first = True
        for label, r in rows:
            if r is None:
                continue
            cell = PRETTY[ds] if first else ""
            first = False
            lines.append(f"{cell} & {label} & {r['u'][0]:.3f} $\\pm$ {r['u'][1]:.3f} & "
                         f"{r['t'][0]:.3f} $\\pm$ {r['t'][1]:.3f} & "
                         f"{r['d'][0]:+.1f} $\\pm$ {r['d'][1]:.1f} \\\\")
        lines.append(r"\addlinespace")
    write("mitigations.tex", r"""
\begin{tabular}{llccc}
\toprule
 & & \multicolumn{2}{c}{clean-input Acc-2 $\uparrow$} & \\
\cmidrule(lr){3-4}
Dataset & Training scheme & benign & text-scarce & change (pts) \\
\midrule
""" + "\n".join(lines).rstrip("\\addlinespace").rstrip() + r"""
\bottomrule
\end{tabular}""")


# ----------------------------------------------------------- T5 diagnostics
def t5():
    me = load("mech")
    lines = []
    for proto, plabel in [("IMM", r"\tokenlevel{}"), ("FMM", r"\modlevel{}")]:
        first = True
        for reg in REG:
            vals = [(r["diagnostics"]["text_repr"]["cos_offdiag"],
                     r["diagnostics"]["text_repr"]["eff_rank_raw"],
                     clean_acc(r))
                    for r in select(me, dataset="MOSI", protocol=proto,
                                    train_regime=reg)
                    if r.get("diagnostics", {}).get("text_repr")]
            if not vals:
                continue
            a = np.array(vals, float)
            cell = plabel if first else ""
            first = False
            lines.append(f"{cell} & {reg} & {np.nanmean(a[:,0]):.3f} & "
                         f"{np.nanmean(a[:,1]):.1f} & {np.nanmean(a[:,2]):.3f} \\\\")
        lines.append(r"\addlinespace")
    write("diagnostics.tex", r"""
\begin{tabular}{llccc}
\toprule
Protocol & Train regime & $\overline{\cos}$ & eff.\ rank & clean Acc-2 \\
\midrule
""" + "\n".join(lines).rstrip("\\addlinespace").rstrip() + r"""
\bottomrule
\end{tabular}""")


def t6():
    """Paired clean-input regression changes in the full-data BERT study."""
    recs = load("main")
    rows = []
    for ds in ["MOSI", "MOSEI", "SIMS"]:
        for proto, label in [("IMM", r"\tokenlevel{}"), ("FMM", r"\modlevel{}")]:
            by_regime = {}
            for regime in ["U-lo", "T-frag"]:
                selected = select(recs, dataset=ds, protocol=proto,
                                  train_regime=regime, encoder="bert-base",
                                  n_train="full")
                by_regime[regime] = {r["config"]["seed"]: clean_mae(r)
                                     for r in selected if clean_mae(r) is not None}
            u, t = by_regime["U-lo"], by_regime["T-frag"]
            assert set(u) == set(t) and len(u) == 5, (ds, proto)
            seeds = sorted(u)
            delta = [t[s] - u[s] for s in seeds]
            rows.append(f"{PRETTY[ds]} & {label} & {np.mean(list(u.values())):.3f} & "
                        f"{np.mean(list(t.values())):.3f} & "
                        f"{np.mean(delta):+.3f} $\\pm$ {np.std(delta):.3f}" + r" \\")
    write("mae_change.tex", r"""
\begin{tabular}{llccc}
\toprule
Dataset & Protocol & Benign & Text-scarce & MAE change $\downarrow$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")


def t7():
    record = json.loads((TAB.parent / "data/masking_exposure.json").read_text())
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Regime & Protocol & Intact text (\%) & Absent text (\%) & Rec. pairs \\",
             r"\midrule"]
    for row in record["rows"]:
        label = r"\tokenlevel{}" if row["protocol"] == "IMM" else r"\modlevel{}"
        lines.append(f"{row['regime']} & {label} & {100*row['text_intact']:.2f} & "
                     f"{100*row['text_absent']:.2f} & "
                     f"{row['reconstruction_pairs_per_example']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("masking_exposure.tex", "\n".join(lines))


def t8():
    result = json.loads((TAB.parent / "data/control_pilot.json").read_text())
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"Rec. coefficient & Clean Acc-2 $\uparrow$ & Clean MAE $\downarrow$ & Selected epoch \\",
             r"\midrule"]
    for row in result["rows"]:
        lines.append(f"{row['coefficient']:.1f} & {row['acc2']:.3f} & {row['mae']:.3f} & "
                     f"{row['best_epoch'] + 1}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("control_pilot.tex", "\n".join(lines))


def t9():
    result = json.loads((TAB.parent / "data/control_common_eval.json").read_text())
    lines = [r"\begin{tabular}{rlrr}", r"\toprule",
             r"Rec. coefficient & Test operator & Acc-2 $\uparrow$ & MAE $\downarrow$ \\",
             r"\midrule"]
    for row in result["rows"]:
        label = r"\tokenlevel{}" if row["protocol"] == "IMM" else r"\modlevel{}"
        lines.append(f"{row['coefficient']:.1f} & {label} & {row['acc2']:.3f} & {row['mae']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("control_common_eval.tex", "\n".join(lines))


def t10(study="control_matrix"):
    """Complete factorial control; never substitute a partial matrix."""
    if study not in ("control_matrix", "control_matrix_cuda"):
        raise ValueError("Unknown control study")
    path = TAB.parent / f"data/{study}.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    assert result["n_runs"] == 40 and result["seeds"] == list(range(5))
    lines = [r"\begin{tabular}{lrccc}", r"\toprule",
             r"Training & $\lambda_{\rm rec}$ & Benign Acc-2 $\uparrow$ & Text-scarce Acc-2 $\uparrow$ & $\Delta_{\rm clean}$ (pp) \\",
             r"\midrule"]
    mae_lines = [r"\begin{tabular}{lrccc}", r"\toprule",
                 r"Training & $\lambda_{\rm rec}$ & Benign MAE $\downarrow$ & Text-scarce MAE $\downarrow$ & MAE change $\downarrow$ \\",
                 r"\midrule"]
    for row in result["rows"]:
        label = r"\tokenlevel{}" if row["protocol"] == "IMM" else r"\modlevel{}"
        prefix = f"{label} & {row['coefficient']:.1f} & "
        for metric, target in (("acc2", lines), ("mae", mae_lines)):
            cells = []
            for condition in ("benign", "text_scarce", "change"):
                value = row[metric][condition]
                precision = 2 if metric == "acc2" and condition == "change" else 3
                mean = (f"{value['mean']:+.{precision}f}" if condition == "change"
                        else f"{value['mean']:.{precision}f}")
                cells.append(mean + f" $\\pm$ {value['sd']:.{precision}f}")
            target.append(prefix + " & ".join(cells) + r" \\")
    for target, filename in ((lines, f"{study}.tex"), (mae_lines, f"{study}_mae.tex")):
        target += [r"\bottomrule", r"\end{tabular}"]
        write(filename, "\n".join(target))


def t11():
    result = json.loads((TAB.parent / "data/tokenization_audit.json").read_text())
    records = {(r["partition"], r["model"], r["lowercase"]): r for r in result["rows"]}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Training} & \multicolumn{2}{c}{Test} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Tokenizer / input & Length & Trunc. (\%) & Length & Trunc. (\%) \\",
             r"\midrule"]
    for model, lower, label in (("bert-base-uncased", False, "BERT-uncased"),
                                ("roberta-base", False, "RoBERTa / original"),
                                ("roberta-base", True, "RoBERTa / lowercase")):
        cells = []
        for split in ("train", "test"):
            row = records[(split, model, lower)]
            cells += [f"{row['mean_capped_length']:.2f}",
                      f"{100*row['truncation_fraction']:.2f}"]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tokenization_audit.tex", "\n".join(lines))


if __name__ == "__main__":
    print("generating tables:")
    t1(); t2(); t3(); t4(); t5(); t6(); t7(); t8(); t9(); t10(); t11()
    t10("control_matrix_cuda")
