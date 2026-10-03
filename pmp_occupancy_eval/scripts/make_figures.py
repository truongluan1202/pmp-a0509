#!/usr/bin/env python3
"""Figures 5 and 6 of the ACRA paper from the analysis CSVs.

    python3 scripts/make_figures.py --run paper_replay/ --out figs/

fig5_doseresponse.pdf  free-space error and occupied fraction against
                       realised dropout, both arms, both modes, k = k_phys,
                       mean over the six condition recordings
fig6_ksweep.pdf        proposed occupied fraction against k by condition,
                       rho in {0, 0.3, 0.6, 0.9}, independent mode, static
                       and dynamic pooled; dotted line at half the grid
"""
import argparse
import collections
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times",
                                            "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "pdf.fonttype": 42,
})
COND = {"B1": "Bare", "B2": "Sheet", "B4": "Pad"}


def rows(path):
    return list(csv.DictReader(open(path)))


def fig5(run, out):
    q1 = rows(os.path.join(run, "q1_doseresponse.csv"))
    g = collections.defaultdict(list)
    for r in q1:
        g[(r["arm"], r["inj_mode"], float(r["target_rho"]))].append(r)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.3, 3.6), sharex=True)
    style = {("hold", "independent"): dict(color="0.0", ls="-", marker="o"),
             ("hold", "structured"): dict(color="0.0", ls="--", marker="o",
                                          mfc="white"),
             ("proposed", "independent"): dict(color="0.5", ls="-",
                                               marker="s"),
             ("proposed", "structured"): dict(color="0.5", ls="--",
                                              marker="s", mfc="white")}
    for (arm, mode), st in style.items():
        rhos = sorted({k[2] for k in g if k[0] == arm and k[1] == mode})
        x = [np.mean([float(r["realised_rho"]) for r in g[(arm, mode, p)]])
             for p in rhos]
        fse = [100 * np.mean([float(r["free_space_error_agreed_given_ref"])
                              for r in g[(arm, mode, p)]]) for p in rhos]
        occ = [100 * np.mean([float(r["occupied_fraction"])
                              for r in g[(arm, mode, p)]]) for p in rhos]
        lab = f"{'Hold' if arm == 'hold' else 'Proposed'}, {mode}"
        a1.plot(x, fse, ms=3, label=lab, **st)
        a2.plot(x, occ, ms=3, **st)
    a1.set_ylabel("Free-space error (%)")
    a2.set_ylabel("Occupied fraction (%)")
    a2.set_xlabel(r"Realised injected dropout rate $\rho$")
    a2.set_ylim(0, 100)
    for a in (a1, a2):
        a.grid(True, lw=0.3, color="0.85")
    a1.legend(frameon=False, loc="upper left")
    fig.tight_layout(h_pad=0.4)
    fig.savefig(os.path.join(out, "fig5_doseresponse.pdf"))
    fig.savefig(os.path.join(out, "fig5_doseresponse.png"), dpi=200)


def fig6(run, out):
    q2 = rows(os.path.join(run, "q2_ksweep.csv"))
    g = collections.defaultdict(list)
    for r in q2:
        if r["arm"] == "proposed" and r["inj_mode"] == "independent":
            c = r["bag"].split("_")[1]
            g[(c, float(r["target_rho"]), float(r["k"]))].append(
                float(r["occupied_fraction"]))
    ks = sorted({k[2] for k in g})
    rhos = sorted({k[1] for k in g})
    greys = ["0.0", "0.3", "0.5", "0.7"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    for ax, c in zip(axes, COND):
        for rho, gr in zip(rhos, greys):
            y = [100 * np.mean(g[(c, rho, k)]) for k in ks]
            ax.plot(ks, y, color=gr, marker="o", ms=2.5,
                    label=rf"$\rho = {rho:.1f}$")
        ax.axhline(50, color="0.4", ls=":", lw=0.8)
        ax.set_title(COND[c], fontsize=8)
        ax.set_xlabel(r"Growth rate $k$ (m s$^{-1}$)")
        ax.set_ylim(0, 100)
        ax.grid(True, lw=0.3, color="0.85")
    axes[0].set_ylabel("Occupied fraction (%)")
    axes[2].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(out, "fig6_ksweep.pdf"))
    fig.savefig(os.path.join(out, "fig6_ksweep.png"), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="paper_replay/")
    ap.add_argument("--out", default="figs/")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig5(a.run, a.out)
    fig6(a.run, a.out)
    print(f"wrote {a.out}fig5_doseresponse.pdf, {a.out}fig6_ksweep.pdf")


if __name__ == "__main__":
    main()
