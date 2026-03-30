"""
Time series EDA for ZTBus ridership dataset.
Assesses temporal structure: stationarity, autocorrelation, partial autocorrelation.
Operates at two granularities: stop-level (within-mission) and hourly aggregate (across-missions).
All figures saved to figures/timeseries/.
"""

import sys
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FIGURES_DIR, RANDOM_SEED, DATA_DIR
from src.data_loading import load_metadata, stratified_sample_missions, load_mission_csv
from src.timeseries import extract_stop_events, run_stationarity_tests, build_hourly_aggregate

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import pacf, acf
except ImportError:
    print("ERROR: statsmodels is required. Install with: .venv/bin/pip install statsmodels")
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

TS_FIGURES_DIR = os.path.join(FIGURES_DIR, "timeseries")
N_REPRESENTATIVE = 5
STOP_ACF_LAGS = 40
HOURLY_ACF_LAGS = 48
STOP_COLS = ["time_iso", "itcs_numberOfPassengers", "itcs_stopName"]

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
})

os.makedirs(TS_FIGURES_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(TS_FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# [1/7] Load metadata + select representative missions
# -----------------------------------------------------------------------

print("=" * 60)
print("ZTBus Time Series EDA")
print("=" * 60)

print(f"\n[1/7] Loading metadata...")
meta = load_metadata()
print(f"  Total missions: {len(meta)}")

meta_repr = stratified_sample_missions(meta, n=N_REPRESENTATIVE, seed=RANDOM_SEED)
print(f"\n  Representative missions ({N_REPRESENTATIVE}):")
for _, row in meta_repr.iterrows():
    dur = (row["endTime_unix"] - row["startTime_unix"]) / 3600
    print(f"    {row['name']} | Route {row['busRoute']} | {dur:.1f}h")

# -----------------------------------------------------------------------
# [2/7] Extract stop-level series (per-mission)
# -----------------------------------------------------------------------

print(f"\n[2/7] Extracting stop-level passenger series...")

mission_stop_series = {}
mission_routes = {}

for _, row in meta_repr.iterrows():
    name = row["name"]
    df = load_mission_csv(name, usecols=STOP_COLS)
    if df is None:
        print(f"  WARNING: Could not load {name}, skipping")
        continue
    stops = extract_stop_events(df)
    if len(stops) == 0:
        print(f"  WARNING: {name} has no stop events, skipping")
        continue
    stops["time_iso"] = pd.to_datetime(stops["time_iso"])
    stops = stops.reset_index(drop=True)
    stops["stop_index"] = range(len(stops))
    mission_stop_series[name] = stops
    mission_routes[name] = row["busRoute"]
    pax = stops["itcs_numberOfPassengers"]
    print(f"  {name}: {len(stops)} stops, pax [{pax.min():.0f}, {pax.max():.0f}]")

# -----------------------------------------------------------------------
# [3/7] Build hourly aggregate series (all missions)
# -----------------------------------------------------------------------

print(f"\n[3/7] Building hourly aggregate ridership (all {len(meta)} missions)...")
hourly_agg = build_hourly_aggregate(meta, DATA_DIR, STOP_COLS)
print(f"  Hourly series: {len(hourly_agg)} data points")
print(f"  Date range: {hourly_agg.index.min()} to {hourly_agg.index.max()}")
print(f"  Mean: {hourly_agg.mean():.2f}, Std: {hourly_agg.std():.2f}")

time_diffs = hourly_agg.index.to_series().diff().dropna()
print(f"  Gap check: median={time_diffs.median()}, max={time_diffs.max()}")

# -----------------------------------------------------------------------
# [4/7] Per-mission stop-level visualization
# -----------------------------------------------------------------------

print(f"\n[4/7] Plotting per-mission stop-level passenger series...")

n_missions = len(mission_stop_series)
fig, axes = plt.subplots(n_missions, 1, figsize=(12, 3 * n_missions), sharex=False)
if n_missions == 1:
    axes = [axes]

for ax, (name, stops) in zip(axes, mission_stop_series.items()):
    route = mission_routes[name]
    pax = stops["itcs_numberOfPassengers"].values
    x = stops["stop_index"].values
    ax.plot(x, pax, "o-", markersize=3, linewidth=1, color="#4C72B0")
    ax.set_ylabel("Passengers")
    ax.set_title(f"Route {route}: {name} ({len(stops)} stops)")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Stop Event Index (sequential within mission)")
fig.suptitle("Per-Mission Stop-Level Passenger Count", fontsize=14, y=1.01)
fig.tight_layout()
save_fig(fig, "01_stop_level_series.png")

# -----------------------------------------------------------------------
# [5/7] Stationarity tests (ADF + KPSS)
# -----------------------------------------------------------------------

print(f"\n[5/7] Running stationarity tests (ADF + KPSS)...")

all_results = []

print("\n  --- Per-Mission Stop-Level ---")
for name, stops in mission_stop_series.items():
    series = stops["itcs_numberOfPassengers"].values
    if len(series) < 20:
        print(f"  {name}: too few stops ({len(series)}), skipping")
        continue
    res = run_stationarity_tests(series, f"stop:{name}")
    all_results.append(res)

print("\n  --- Hourly Aggregate ---")
hourly_vals = hourly_agg.values
res_hourly = run_stationarity_tests(hourly_vals, "hourly_aggregate")
all_results.append(res_hourly)

print(f"\n  {'Series':<55} {'ADF p':>8} {'ADF':>12} {'KPSS p':>8} {'KPSS':>12} {'Verdict':>20}")
print("  " + "-" * 120)
for r in all_results:
    adf_dec = "REJECT H0" if r["adf_reject"] else "fail reject"
    kpss_dec = "REJECT H0" if r["kpss_reject"] else "fail reject"
    print(f"  {r['name']:<55} {r['adf_p']:>8.4f} {adf_dec:>12} {r['kpss_p']:>8.4f} {kpss_dec:>12} {r['verdict']:>20}")

# First differencing for non-stationary series
non_stationary = [r for r in all_results if "NON-STATIONARY" in r["verdict"] or "TREND" in r["verdict"]]
diff_results = []

if non_stationary:
    print("\n  --- First Differencing Re-Tests ---")
    for r in non_stationary:
        if r["name"] == "hourly_aggregate":
            diff_series = np.diff(hourly_vals)
            res_diff = run_stationarity_tests(diff_series, "hourly_aggregate (d=1)")
        else:
            mission_name = r["name"].replace("stop:", "")
            series = mission_stop_series[mission_name]["itcs_numberOfPassengers"].values
            diff_series = np.diff(series)
            res_diff = run_stationarity_tests(diff_series, f"stop:{mission_name} (d=1)")
        diff_results.append(res_diff)

    print(f"\n  {'Series':<55} {'ADF p':>8} {'ADF':>12} {'KPSS p':>8} {'KPSS':>12} {'Verdict':>20}")
    print("  " + "-" * 120)
    for r in diff_results:
        adf_dec = "REJECT H0" if r["adf_reject"] else "fail reject"
        kpss_dec = "REJECT H0" if r["kpss_reject"] else "fail reject"
        print(f"  {r['name']:<55} {r['adf_p']:>8.4f} {adf_dec:>12} {r['kpss_p']:>8.4f} {kpss_dec:>12} {r['verdict']:>20}")

# -----------------------------------------------------------------------
# [6/7] ACF plots
# -----------------------------------------------------------------------

print(f"\n[6/7] Plotting ACF...")

# A) Per-mission stop-level ACF
fig, axes = plt.subplots(n_missions, 1, figsize=(12, 3 * n_missions))
if n_missions == 1:
    axes = [axes]

for ax, (name, stops) in zip(axes, mission_stop_series.items()):
    route = mission_routes[name]
    series = stops["itcs_numberOfPassengers"].values
    nlags = min(STOP_ACF_LAGS, len(series) // 2 - 1)
    if nlags < 5:
        ax.text(0.5, 0.5, f"Too few stops ({len(series)})", ha="center", va="center")
        continue
    plot_acf(series, lags=nlags, ax=ax, title=f"ACF - Route {route}: {name}")

fig.suptitle("Stop-Level Autocorrelation (Per-Mission)", fontsize=14, y=1.01)
fig.tight_layout()
save_fig(fig, "02_acf_stop_level.png")

# B) Hourly aggregate ACF
fig, ax = plt.subplots(figsize=(12, 5))
nlags_h = min(HOURLY_ACF_LAGS, len(hourly_agg) // 2 - 1)
plot_acf(hourly_agg.values, lags=nlags_h, ax=ax,
         title="ACF - Hourly Aggregate Ridership")
fig.tight_layout()
save_fig(fig, "03_acf_hourly.png")

# -----------------------------------------------------------------------
# [7/7] PACF plots + verdict
# -----------------------------------------------------------------------

print(f"\n[7/7] Plotting PACF...")

# A) Per-mission stop-level PACF
fig, axes = plt.subplots(n_missions, 1, figsize=(12, 3 * n_missions))
if n_missions == 1:
    axes = [axes]

for ax, (name, stops) in zip(axes, mission_stop_series.items()):
    route = mission_routes[name]
    series = stops["itcs_numberOfPassengers"].values
    nlags = min(STOP_ACF_LAGS, len(series) // 2 - 1)
    if nlags < 5:
        ax.text(0.5, 0.5, f"Too few stops ({len(series)})", ha="center", va="center")
        continue
    plot_pacf(series, lags=nlags, ax=ax,
              title=f"PACF - Route {route}: {name}", method="ywm")

fig.suptitle("Stop-Level Partial Autocorrelation (Per-Mission)", fontsize=14, y=1.01)
fig.tight_layout()
save_fig(fig, "04_pacf_stop_level.png")

# B) Hourly aggregate PACF
fig, ax = plt.subplots(figsize=(12, 5))
plot_pacf(hourly_agg.values, lags=nlags_h, ax=ax,
          title="PACF - Hourly Aggregate Ridership", method="ywm")
fig.tight_layout()
save_fig(fig, "05_pacf_hourly.png")

# PACF cutoff analysis
pacf_vals = pacf(hourly_agg.values, nlags=nlags_h, method="ywm")
sig_bound = 1.96 / np.sqrt(len(hourly_agg))
significant_pacf_lags = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > sig_bound]

if significant_pacf_lags:
    print(f"  Hourly PACF: significant at lags {significant_pacf_lags}")
    print(f"  Suggested AR order: p = {significant_pacf_lags[-1]}")
else:
    print(f"  Hourly PACF: no significant lags (white noise)")

# -----------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIME SERIES EDA VERDICT")
print("=" * 60)

n_stationary = sum(1 for r in all_results if r["verdict"] == "STATIONARY")
n_nonstationary = sum(1 for r in all_results if "NON-STATIONARY" in r["verdict"] or "TREND" in r["verdict"])
n_inconclusive = sum(1 for r in all_results if r["verdict"] == "INCONCLUSIVE")

print(f"\nStationarity ({len(all_results)} series tested):")
print(f"  Stationary:     {n_stationary}")
print(f"  Non-stationary: {n_nonstationary}")
print(f"  Inconclusive:   {n_inconclusive}")

acf_vals = acf(hourly_agg.values, nlags=nlags_h)
acf_sig_bound = 1.96 / np.sqrt(len(hourly_agg))
hourly_sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > acf_sig_bound]

print(f"\nAutocorrelation (hourly aggregate):")
print(f"  Significant ACF lags: {len(hourly_sig_acf)} / {nlags_h}")
if hourly_sig_acf:
    print(f"  Lags: {hourly_sig_acf[:15]}{'...' if len(hourly_sig_acf) > 15 else ''}")

has_autocorrelation = len(hourly_sig_acf) > 3
has_nonstationarity = n_nonstationary > 0

if has_autocorrelation:
    print(f"\n>>> VERDICT: Temporal structure EXISTS in ridership data.")
    print(f"    Evidence: {len(hourly_sig_acf)} significant ACF lags in hourly aggregate.")
    if has_nonstationarity:
        print(f"    Note: {n_nonstationary} series non-stationary -- first differencing may be needed for ARIMA.")
    print(f"    Implication: ARIMA / temporal models justified for Task 2.")
else:
    print(f"\n>>> VERDICT: No strong temporal structure detected.")
    print(f"    Evidence: Only {len(hourly_sig_acf)} significant ACF lags (likely noise).")
    print(f"    Implication: Pure temporal models (ARIMA) may not add value over feature-based ML.")

print(f"\nFigures saved to: {TS_FIGURES_DIR}/")
