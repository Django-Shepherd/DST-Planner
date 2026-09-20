#!/usr/bin/env python3
"""Plot actual observed floor cells and flown trajectories from three completed runs."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--runs", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
fig.subplots_adjust(left=0.045, right=0.985, bottom=0.16, top=0.82, wspace=0.18)
for ax, kind in zip(axes, ["forest", "partition", "dungeon"]):
    root = Path(a.runs) / kind
    grid = np.load(root / "coverage.npz")
    r = float(grid["resolution"])
    labels = np.zeros_like(grid["observed"], dtype=int)
    labels[grid["occupied"]] = 1
    labels[grid["reachable"]] = 2
    labels[grid["reachable"] & grid["observed"]] = 3
    ax.imshow(
        labels,
        origin="lower",
        extent=[0, labels.shape[1] * r, 0, labels.shape[0] * r],
        cmap=ListedColormap(["#e7eaee", "#525966", "#ef9c3a", "#b9e1c9"]),
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    events = [json.loads(line) for line in (root / "events.jsonl").open()]
    positions = np.array([e["position"] for e in events if e["event"] == "odometry"])
    ax.plot(positions[:, 0], positions[:, 1], color="#1469b4", linewidth=0.9, alpha=0.8)
    ax.scatter(*positions[0, :2], s=45, color="#135633", edgecolors="white", zorder=4)
    ax.scatter(
        *positions[-1, :2],
        s=95,
        marker="*",
        color="#971b37",
        edgecolors="white",
        zorder=4,
    )
    summary = json.loads((root / "summary.json").read_text())
    trigger = next(e["wall_time"] for e in events if e["event"] == "trigger")
    finish = next(e["wall_time"] for e in events if e.get("state") == "FINISH")
    ax.set_title(
        f"{kind.title()} | FINISH in {finish - trigger:.1f} s\n{summary['distance_m']:.1f} m flown | {summary['coverage']['observed_fraction']:.2%} floor observed",
        fontsize=11,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
fig.suptitle(
    "DST-planner / EPIC: three generated maps, three naturally completed episodes",
    fontsize=15,
    y=0.97,
)
fig.legend(
    handles=[
        Patch(color="#525966", label="Obstacle"),
        Patch(color="#b9e1c9", label="Observed reachable floor"),
        Patch(color="#ef9c3a", label="Unobserved reachable floor"),
        Patch(color="#e7eaee", label="Outside reachable mask"),
        Patch(color="#1469b4", label="Actual trajectory"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=5,
    frameon=False,
)
fig.savefig(a.output, dpi=180)
