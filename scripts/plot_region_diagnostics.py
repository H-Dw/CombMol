"""Standalone scientific plots of the measured region-selection diagnostics."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    options = parser.parse_args()
    folder = Path(options.folder)
    with (folder / "terminal_geometry.csv").open(encoding="utf-8-sig") as stream:
        terminals = [row for row in csv.DictReader(stream) if row.get("chemistry_retained") == "True"]
    with (folder / "region_geometry.csv").open(encoding="utf-8-sig") as stream:
        groups = list(csv.DictReader(stream))
    values = [float(row["pic50_delta_parent_mean"]) for row in terminals]
    norm = TwoSlopeNorm(vmin=min(values), vcenter=0., vmax=max(values))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, target in enumerate(["6pyy_H7S_A_402", "3w54_RNB_A_504"]):
        rows = [row for row in terminals if row["target"] == target]
        frame = [float(row["frame_max_rmsd_A"]) for row in rows]
        fitted = [float(row["joint_fit_max_side_rmsd_A"]) for row in rows]
        scatter = axes[0, column].scatter(frame, fitted, c=[float(row["pic50_delta_parent_mean"]) for row in rows],
            cmap="coolwarm", norm=norm, s=19, alpha=.8, linewidths=0)
        limit = max(frame + fitted) * 1.04
        axes[0, column].plot([0, limit], [0, limit], color="0.7", linewidth=.8)
        axes[0, column].axvline(.5, color="0.35", linestyle="--", linewidth=1)
        axes[0, column].axhline(.5, color="0.35", linestyle="--", linewidth=1)
        axes[0, column].set(xlim=(0, limit), ylim=(0, limit), xlabel="Protein-frame max per-region RMSD (Å)",
            ylabel="Shared-fit max per-region RMSD (Å)", title=f"{target[:4]}: chemistry retained (n={len(rows)})")
        subset = [row for row in groups if row["target"] == target]
        axes[1, column].scatter([float(row["nearest_port_distance_A"]) for row in subset],
            [100 * float(row["qualified_rate"]) for row in subset],
            c=[float(row["mean_pic50_delta"]) for row in subset], cmap="coolwarm", norm=norm,
            s=[5 * float(row["fixed_atoms"]) for row in subset], edgecolors="0.2", linewidths=.5)
        for row in subset:
            if row["fusion"] in ("fusion-01", "fusion-03", "fusion-11", "fusion-13"):
                axes[1, column].annotate(row["fusion"], (float(row["nearest_port_distance_A"]), 100 * float(row["qualified_rate"])),
                    xytext=(4, 6), textcoords="offset points", fontsize=8)
        axes[1, column].set(ylim=(-5, 72), xlabel="Initial distance between source attachment atoms (Å)",
            ylabel="Original-protocol qualified fraction (%)", title=f"{target[:4]}: {len(subset)} frozen proposals")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.colorbar(scatter, ax=axes, shrink=.7, label="Predicted pIC50 minus mean of the two source parents")
    fig.suptitle("Input separation, final pose drift and fusion outcomes", fontsize=14)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / ("region_diagnostics." + extension), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
