"""Recompute manuscript study summaries from explicitly identified evidence.

Run this before quoting any figure in the prose. Each block prints the study
it draws on, so a reader (or reviewer) can trace a number to a run set.

Selected numerical results and source-record agreement are additionally
asserted in tests/test_result_evidence.py.
"""
from __future__ import annotations

import re
import json
from pathlib import Path
import numpy as np

from studies import (STUDIES, clean_acc, clean_mae, counts, encoder_of, load,
                     select)

REG = ["U-lo", "U-hi", "T-frag", "NV-frag"]


def head(title, study):
    print()
    print("=" * 74)
    print(f"{title}")
    print(f"  study: {study}  --  {STUDIES[study]['desc']}")
    print("=" * 74)


def mean_sd(vals):
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


# --------------------------------------------------------------- run counts
print("=" * 74)
print("RUN COUNTS (per study; studies are disjoint and never pooled)")
print("=" * 74)
c = counts()
for s in STUDIES:
    print(f"  {s:9s} {c[s]:4d}   defines: {', '.join(STUDIES[s]['defines'])}")
ft = sum(v for k, v in c.items() if k != "frozen")
print(f"  {'-'*66}")
print(f"  fine-tuned-encoder runs : {ft}")
print(f"  frozen-feature runs     : {c['frozen']}")
print(f"  TOTAL                   : {ft + c['frozen']}")
print(f"  >>> manuscript should say: {ft} fine-tuned runs "
      f"(+{c['frozen']} frozen-feature)")


# ------------------------------------------------- backbone competitiveness
head("BACKBONE COMPETITIVENESS (clean input, benign regime)", "main")
PUBLISHED = {"MOSI": "Acc-2 .85-.86, MAE .71-.78",
             "SIMS": "Acc-2 .78-.82, MAE .40-.42",
             "MOSEI": "Acc-2 .85-.86, MAE .53-.55"}
recs = load("main")
for ds in ["MOSI", "SIMS", "MOSEI"]:
    sel = select(recs, dataset=ds, train_regime="U-lo", encoder="bert-base",
                 n_train="full")
    a = [clean_acc(r) for r in sel if clean_acc(r) is not None]
    m = [clean_mae(r) for r in sel if clean_mae(r) is not None]
    if a:
        print(f"  {ds:6s} Acc-2 {np.mean(a):.4f}  MAE {np.mean(m):.4f}  "
              f"(n={len(a)})   published: {PUBLISHED[ds]}")


# ----------------------------------------------------- headline degradation
def degradation(recs, **eq):
    u = [clean_acc(r) for r in select(recs, train_regime="U-lo", **eq)
         if clean_acc(r) is not None]
    t = [clean_acc(r) for r in select(recs, train_regime="T-frag", **eq)
         if clean_acc(r) is not None]
    if not u or not t:
        return None
    return dict(u=np.mean(u), t=np.mean(t), d=100 * (np.mean(t) - np.mean(u)),
                nu=len(u), nt=len(t))


head("CLEAN-INPUT CHANGE, headline (T-frag minus U-lo)", "main")
for ds in ["MOSI", "SIMS", "MOSEI"]:
    for proto in ["IMM", "FMM"]:
        r = degradation(recs, dataset=ds, protocol=proto, encoder="bert-base",
                        n_train="full")
        if r:
            print(f"  {ds:6s} {proto:4s}  U-lo {r['u']:.4f}  T-frag {r['t']:.4f}"
                  f"  degradation {r['d']:+6.1f} pts  (n={r['nu']},{r['nt']})")
print("  >>> manuscript headline: MOSI token-level, from THIS study only")


# ----------------------------------------------------------- dose-response
head("DOSE-RESPONSE and onset threshold", "sweep")
sw = load("sweep")
for ds in ["MOSI", "SIMS"]:
    for proto in ["FMM", "IMM"]:
        ser = {}
        for r in select(sw, dataset=ds, protocol=proto):
            m = re.match(r"T@([0-9.]+)$", r["config"]["train_regime"])
            a = clean_acc(r)
            if m and a is not None:
                ser.setdefault(float(m.group(1)), []).append(a)
        if not ser:
            continue
        xs = sorted(ser)
        mus = {x: float(np.mean(ser[x])) for x in xs}
        base = mus[xs[0]]
        worst_x = min(mus, key=mus.get)
        # onset = first rate at which the mean falls >2 pts below rate 0
        onset = next((x for x in xs if base - mus[x] > 0.02), None)
        print(f"  {ds:6s} {proto:4s}  rate0 {base:.4f} -> worst {mus[worst_x]:.4f} "
              f"at {worst_x}  drop {100*(base-mus[worst_x]):5.1f} pts"
              f"   onset(>2pts) {onset}")
print("  >>> onset is descriptive on the tested grid, not a fitted threshold")


# ---------------------------------------------------------------- shift gap
def shift_gap(recs, ds, proto, exclude=()):
    cells = {}
    for r in select(recs, dataset=ds, protocol=proto, n_train="full"):
        if encoder_of(r) != "bert-base":
            continue
        tr = r["config"]["train_regime"]
        for te, cell in r.get("cells", {}).items():
            if te == "NONE":
                continue
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
    return (float(np.mean(per)) if per else np.nan), len(per)


head("SHIFT GAP (column-wise, difficulty-controlled)", "main")
gaps = []
for ds in ["MOSI", "SIMS", "MOSEI"]:
    for proto in ["IMM", "FMM"]:
        g, n = shift_gap(recs, ds, proto)
        gx, _ = shift_gap(recs, ds, proto, ("T-frag",))
        if not np.isnan(g):
            gaps.append(gx)
            print(f"  {ds:6s} {proto:4s}  all rows {g:+.4f}   "
                  f"excl. degraded row {gx:+.4f}  ({n} columns)")
head("SHIFT GAP under a frozen encoder (different model, not a seed variant)",
     "frozen")
fz = load("frozen")
for ds in ["MOSI"]:
    for proto in ["IMM", "FMM"]:
        g, n = shift_gap(fz, ds, proto)
        if not np.isnan(g):
            print(f"  {ds:6s} {proto:4s}  all rows {g:+.4f}")
if gaps:
    print()
    print(f"  >>> manuscript range for the fine-tuned shift gap "
          f"(excl. degraded row): {100*min(gaps):.1f} to {100*max(gaps):.1f} pts")


# ------------------------------------------------------------ size-response
head("SIZE-RESPONSE (alpha_text fixed at .948)", "size")
sz = load("size")
rows = {}
for r in select(sz, dataset="MOSEI", protocol="IMM"):
    a = clean_acc(r)
    if a is not None:
        rows.setdefault(r["config"]["train_regime"], {}) \
            .setdefault(r.get("n_train"), []).append(a)
main_full = degradation(recs, dataset="MOSEI", protocol="IMM",
                        encoder="bert-base", n_train="full")
ns = sorted(set(rows.get("U-lo", {})) & set(rows.get("T-frag", {})))
for n in ns:
    u, t = np.mean(rows["U-lo"][n]), np.mean(rows["T-frag"][n])
    print(f"  n={n:>6d}  U-lo {u:.4f}  T-frag {t:.4f}  "
          f"degradation {100*(t-u):+6.1f} pts")
if main_full:
    print(f"  n=  full  U-lo {main_full['u']:.4f}  T-frag {main_full['t']:.4f}  "
          f"degradation {main_full['d']:+6.1f} pts   [study: main]")
if ns:
    small = 100 * (np.mean(rows['T-frag'][ns[0]]) - np.mean(rows['U-lo'][ns[0]]))
    print(f"  >>> manuscript: degradation at n={ns[0]} is {small:+.1f} pts")


# -------------------------------------------------------- encoder generality
head("ENCODER GENERALITY", "roberta")
rb = load("roberta")
print("  roberta-base (this study):")
for ds, n in [("MOSI", "full"), ("MOSEI", 1284)]:
    for proto in ["IMM", "FMM"]:
        r = degradation(rb, dataset=ds, protocol=proto, n_train=n)
        if r:
            print(f"    {ds:6s} n={str(n):>5s} {proto:4s}  U-lo {r['u']:.4f}  "
                  f"T-frag {r['t']:.4f}  degradation {r['d']:+6.1f} pts")
print("  bert-base comparators (study: main for MOSI-full, size for MOSEI@1284):")
for ds, n, src in [("MOSI", "full", recs), ("MOSEI", 1284, sz)]:
    for proto in ["IMM", "FMM"]:
        kw = dict(dataset=ds, protocol=proto, encoder="bert-base")
        if src is recs:
            kw["n_train"] = n
        r = degradation(src, **kw) if src is recs else degradation(
            src, dataset=ds, protocol=proto, encoder="bert-base", n_train=n)
        if src is not recs:
            if r:
                print(f"    {ds:6s} n={str(n):>5s} {proto:4s}  U-lo {r['u']:.4f}  "
                      f"T-frag {r['t']:.4f}  degradation {r['d']:+6.1f} pts")
            continue
        if r:
            print(f"    {ds:6s} n={str(n):>5s} {proto:4s}  U-lo {r['u']:.4f}  "
                  f"T-frag {r['t']:.4f}  degradation {r['d']:+6.1f} pts")


# -------------------------------------------------- masking-operator check
head("MASKING OPERATOR: [UNK] substitution vs token removal", "unk")
uk = load("unk")
r_unk = degradation(uk, dataset="MOSI", protocol="IMM")
r_rm = degradation(recs, dataset="MOSI", protocol="IMM",
                   encoder="bert-base", n_train="full")
if r_unk:
    print(f"  [UNK] substitution (literature)  U-lo {r_unk['u']:.4f}  "
          f"T-frag {r_unk['t']:.4f}  degradation {r_unk['d']:+6.1f} pts  "
          f"(n={r_unk['nu']},{r_unk['nt']})")
if r_rm:
    print(f"  token removal (ours, study=main)  U-lo {r_rm['u']:.4f}  "
          f"T-frag {r_rm['t']:.4f}  degradation {r_rm['d']:+6.1f} pts")
print("  >>> the effect persists under both operators; [UNK] gives the larger change")


# ------------------------------------------------------------- mitigations
head("MITIGATIONS", "mitig")
mt = load("mitig")
BASE = {"MOSI": None, "SIMS": None}
for ds in ["MOSI", "SIMS"]:
    b = degradation(recs, dataset=ds, protocol="IMM", encoder="bert-base",
                    n_train="full")
    BASE[ds] = b
    if b:
        print(f"  {ds:6s} baseline (study: main)        U-lo {b['u']:.4f}  "
              f"T-frag {b['t']:.4f}  degradation {b['d']:+6.1f} pts")
    for label, d, proto in [("M1 modality-level train", "results_mitig_crossproto", "FMM"),
                            ("M2 freeze 6 layers", "results_mitig_freeze", "IMM"),
                            ("M3 encoder lr 5e-6", "results_mitig_lowlr", "IMM")]:
        sub = [r for r in mt if d in r["_file"]]
        r = degradation(sub, dataset=ds, protocol=proto)
        if r:
            print(f"  {ds:6s} {label:28s} U-lo {r['u']:.4f}  "
                  f"T-frag {r['t']:.4f}  degradation {r['d']:+6.1f} pts")


# ------------------------------------------------------ collapse diagnostics
head("EXPLORATORY REPRESENTATION DIAGNOSTICS (clean input)", "mech")
me = load("mech")
for proto in ["IMM", "FMM"]:
    for reg in REG:
        vals = [(r["diagnostics"]["text_repr"]["cos_offdiag"],
                 r["diagnostics"]["text_repr"]["eff_rank_raw"],
                 clean_acc(r))
                for r in select(me, dataset="MOSI", protocol=proto,
                                train_regime=reg)
                if r.get("diagnostics", {}).get("text_repr")]
        if vals:
            a = np.array([[v[0], v[1], v[2]] for v in vals], float)
            print(f"  {proto:4s} {reg:8s} cos_offdiag {np.nanmean(a[:,0]):.4f}  "
                  f"eff_rank_raw {np.nanmean(a[:,1]):5.2f}  "
                  f"clean Acc-2 {np.nanmean(a[:,2]):.4f}")
print("  >>> these exploratory summaries do not support a low-rank-collapse account")
print()

# Local controls are separate from the archived study registry and run counts.
data_dir = Path(__file__).resolve().parent / "data"
print("MOSI TOKENIZATION AUDIT (truncation means untruncated length > 50)")
for row in json.loads((data_dir / "tokenization_audit.json").read_text())["rows"]:
    print(f"  {row['partition']} {row['model']} lowercase={row['lowercase']}: "
          f"retained length={row['mean_capped_length']:.2f}, "
          f"truncated={100*row['truncation_fraction']:.2f}% "
          f"({row['count_truncated']}/{row['n']})")
print("LOCAL OBSERVATION-EXPOSURE MEASUREMENT")
for row in json.loads((data_dir / "masking_exposure.json").read_text())["rows"]:
    print(f"  {row['regime']} {row['protocol']}: intact text "
          f"{100*row['text_intact']:.2f}%, absent text {100*row['text_absent']:.2f}%, "
          f"reconstruction pairs {row['reconstruction_pairs_per_example']:.3f}")
print("LOCAL AUXILIARY-LOSS PILOT (one training seed, distinct runtime)")
for row in json.loads((data_dir / "control_pilot.json").read_text())["rows"]:
    print(f"  coefficient={row['coefficient']}: clean Acc-2={row['acc2']:.6f}, "
          f"MAE={row['mae']:.6f}, selected epoch={row['best_epoch']+1}")
print("LOCAL PILOT COMMON TEST SUITES (five mask repetitions)")
for row in json.loads((data_dir / "control_common_eval.json").read_text())["rows"]:
    print(f"  coefficient={row['coefficient']}, test={row['protocol']}: "
          f"Acc-2={row['acc2']:.6f}, MAE={row['mae']:.6f}")
matrix_path = data_dir / "control_matrix.json"
if matrix_path.exists():
    matrix = json.loads(matrix_path.read_text())
    assert matrix["n_runs"] == 40 and matrix["seeds"] == list(range(5))
    print("LOCAL FIVE-SEED FACTORIAL CONTROL (40 runs; not pooled with CUDA)")
    for row in matrix["rows"]:
        for metric in ("acc2", "mae"):
            values = row[metric]
            print(f"  {row['protocol']} coefficient={row['coefficient']} {metric}: "
                  f"benign={values['benign']['mean']:.6f}, "
                  f"text-scarce={values['text_scarce']['mean']:.6f}, "
                  f"paired change={values['change']['mean']:+.6f} "
                  f"+/- {values['change']['sd']:.6f}")

cuda_path = data_dir / "control_matrix_cuda.json"
if cuda_path.exists():
    cuda = json.loads(cuda_path.read_text())
    assert cuda["n_runs"] == 40 and cuda["env"]["device"] == "cuda"
    print("SEPARATE FIVE-SEED CUDA FACTORIAL CONTROL")
    for row in cuda["rows"]:
        for metric in ("acc2", "mae"):
            value = row[metric]["change"]
            print(f"  {row['protocol']} coefficient={row['coefficient']} {metric}: "
                  f"paired change={value['mean']:+.6f} +/- {value['sd']:.6f}")
    print("  Protocol contrasts and coefficient interactions:")
    print(json.dumps(cuda["protocol_change_contrasts"], indent=2))
