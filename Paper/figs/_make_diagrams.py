"""
Generate the two diagram figures for the paper:
- fig_08_leakage_diagram.pdf  -- the forward-fill lag trap mechanism
- fig_09_stop_lag_broadcast.pdf -- the v2 stop-level lag broadcast fix

Both figures use a small synthetic example so the mechanism is visually clear.
Run from paper/figs/ with: python _make_diagrams.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Synthetic example: 200 seconds of a bus trip with 3 stop events
# Stop events at seconds 20, 110, 180 with passenger counts 10, 25, 15
N = 200
t = np.arange(N)
stop_times = np.array([20, 110, 180])
stop_counts = np.array([10, 25, 15])

# Raw sparse series (NaN between stops)
raw = np.full(N, np.nan)
for st, sc in zip(stop_times, stop_counts):
    raw[st] = sc

# Forward-filled dense series
dense = np.full(N, np.nan)
current = np.nan
for i in range(N):
    if not np.isnan(raw[i]):
        current = raw[i]
    dense[i] = current

# 60-second lag on dense series
lag60 = np.full(N, np.nan)
lag60[60:] = dense[:-60]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 200,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


def make_leakage_figure():
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.5), sharex=True)

    # Top: raw sparse events
    ax = axes[0]
    ax.plot(t, dense, color="lightgrey", linewidth=0.8, alpha=0.5, label="(reference)")
    ax.scatter(stop_times, stop_counts, color="#2196F3", s=60, zorder=5,
               label="stop events")
    ax.set_ylabel("passenger count")
    ax.set_title("(a) Raw sparse events (only recorded at stops)")
    ax.set_ylim(0, 35)
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.3)

    # Middle: forward-filled dense series (the staircase)
    ax = axes[1]
    ax.plot(t, dense, color="#2196F3", linewidth=1.8, label="forward-fill p(t)")
    ax.scatter(stop_times, stop_counts, color="#2196F3", s=50, zorder=5,
               edgecolor="black", linewidth=0.5)
    ax.set_ylabel("passenger count")
    ax.set_title("(b) Forward-filled 1 Hz series (staircase)")
    ax.set_ylim(0, 35)
    ax.grid(True, alpha=0.3)

    # Bottom: dense series + lag(60) overlay, showing the overlap problem
    ax = axes[2]
    ax.plot(t, dense, color="#2196F3", linewidth=1.8, label="p(t)")
    ax.plot(t, lag60, color="#E91E63", linewidth=1.4, linestyle="--",
            label="lag(60s) = p(t-60)")

    # Highlight the leakage region where lag equals current
    overlap = (dense == lag60) & ~np.isnan(dense) & ~np.isnan(lag60)
    for i in range(N):
        if overlap[i]:
            ax.axvspan(i - 0.5, i + 0.5, color="#FFC107", alpha=0.15)

    ax.set_xlabel("time (seconds within mission)")
    ax.set_ylabel("passenger count")
    ax.set_title("(c) Overlay: lag(60s) equals current value in shaded rows "
                 "(the leakage region)")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_08_leakage_diagram.pdf", bbox_inches="tight")
    plt.savefig("fig_08_leakage_diagram.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig_08_leakage_diagram.pdf / .png")


def make_stoplag_figure():
    # Compute stoplag_1 at stop events only
    stoplag1_at_stops = np.full(len(stop_times), np.nan)
    stoplag1_at_stops[1:] = stop_counts[:-1]  # previous stop's count

    # Broadcast stoplag_1 to all rows (forward-fill from each stop event forward)
    stoplag1_dense = np.full(N, np.nan)
    current_lag = np.nan
    for i in range(N):
        if i in stop_times:
            idx = np.where(stop_times == i)[0][0]
            current_lag = stoplag1_at_stops[idx]
        stoplag1_dense[i] = current_lag

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.5), sharex=True)

    # Top: raw sparse events
    ax = axes[0]
    ax.plot(t, dense, color="lightgrey", linewidth=0.8, alpha=0.5)
    ax.scatter(stop_times, stop_counts, color="#2196F3", s=60, zorder=5,
               label="stop events p(t_i)")
    for st, sc in zip(stop_times, stop_counts):
        ax.annotate(f"{int(sc)}", (st, sc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("passenger count")
    ax.set_title("(a) Stop events with passenger counts")
    ax.set_ylim(0, 35)
    ax.set_xlim(0, N)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Middle: lag values assigned only at stop events
    ax = axes[1]
    mask = ~np.isnan(stoplag1_at_stops)
    ax.scatter(stop_times[mask], stoplag1_at_stops[mask],
               color="#4CAF50", s=60, zorder=5,
               label=r"$\mathrm{stoplag}_1(t_i) = p(t_{i-1})$")
    # NaN at first stop - show with open marker
    ax.scatter([stop_times[0]], [2], color="white", s=60, zorder=5,
               edgecolor="#4CAF50", linewidth=1.5, marker="o",
               label="NaN (first stop)")
    for st, lag in zip(stop_times[mask], stoplag1_at_stops[mask]):
        ax.annotate(f"{int(lag)}", (st, lag), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("lag value")
    ax.set_title(r"(b) $\mathrm{stoplag}_1$ assigned at each stop event "
                 "(from strictly earlier stop)")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Bottom: forward-fill the lag values to the dense grid
    ax = axes[2]
    ax.plot(t, stoplag1_dense, color="#4CAF50", linewidth=1.8,
            label=r"$\mathrm{stoplag}_1$ forward-filled")
    ax.plot(t, dense, color="#2196F3", linewidth=1.2, alpha=0.4,
            linestyle=":", label="p(t) (for reference)")
    ax.scatter(stop_times[mask], stoplag1_at_stops[mask],
               color="#4CAF50", s=50, zorder=5, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("time (seconds within mission)")
    ax.set_ylabel("passenger count")
    ax.set_title(r"(c) $\mathrm{stoplag}_1$ forward-filled to the 1 Hz grid "
                 "(no overlap with p(t))")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_09_stop_lag_broadcast.pdf", bbox_inches="tight")
    plt.savefig("fig_09_stop_lag_broadcast.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved fig_09_stop_lag_broadcast.pdf / .png")


if __name__ == "__main__":
    make_leakage_figure()
    make_stoplag_figure()
