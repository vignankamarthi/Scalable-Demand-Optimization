"""
EDA script for ZTBus ridership dataset.
Composes tested src/ primitives to produce exploratory visualizations.
All figures saved to figures/.
"""

import sys
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FIGURES_DIR, EDA_SAMPLE_SIZE, RANDOM_SEED, EDA_USE_COLS,
)
from src.data_loading import (
    load_metadata, stratified_sample_missions, load_sample_missions,
)
from src.preprocessing import (
    apply_unit_conversions, extract_temporal_features, detect_missing_values,
    forward_fill_stop_columns,
)
from src.target import compute_tercile_boundaries, assign_demand_class

warnings.filterwarnings("ignore", category=FutureWarning)

# Plot style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
})
sns.set_style("whitegrid")

os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# 1. Load and sample data
# -----------------------------------------------------------------------

print("=" * 60)
print("ZTBus EDA Pipeline")
print("=" * 60)

print(f"\n[1/3] Loading metadata...")
meta = load_metadata()
print(f"  Missions: {len(meta)}")
print(f"  Routes: {meta['busRoute'].nunique()} ({', '.join(meta['busRoute'].unique())})")
print(f"  Date range: {meta['startTime_iso'].min().date()} to {meta['startTime_iso'].max().date()}")

n_sample = min(EDA_SAMPLE_SIZE, len(meta))
print(f"\n[2/3] Stratified sampling {n_sample} missions (seed={RANDOM_SEED})...")
meta_sample = stratified_sample_missions(meta, n=n_sample, seed=RANDOM_SEED)
print(f"  Sampled route distribution:")
for route, count in meta_sample["busRoute"].value_counts().items():
    print(f"    Route {route}: {count} missions")

print(f"\n[3/3] Loading mission CSVs (columns: {len(EDA_USE_COLS)})...")
df = load_sample_missions(meta_sample, usecols=EDA_USE_COLS)
print(f"  Total observations: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

# Preprocessing
print("\nForward-filling ITCS columns (passenger count, route, stop name)...")
print(f"  Before forward-fill: {df['itcs_numberOfPassengers'].notna().sum():,} rows with passenger data "
      f"({df['itcs_numberOfPassengers'].notna().mean()*100:.1f}%)")
df = forward_fill_stop_columns(df)
# Drop rows before first stop (no passenger data to propagate from)
df_filled = df[df["itcs_numberOfPassengers"].notna()].copy()
print(f"  After forward-fill: {len(df_filled):,} rows with passenger data "
      f"({len(df_filled)/len(df)*100:.1f}%)")
print(f"  Dropped {len(df) - len(df_filled):,} pre-first-stop rows")
df = df_filled

print("Applying unit conversions...")
df = apply_unit_conversions(df)

print("Extracting temporal features...")
df = extract_temporal_features(df)


# -----------------------------------------------------------------------
# 2. Passenger count distribution (Figure 01)
# -----------------------------------------------------------------------

print("\n--- Figure 01: Passenger count distribution ---")
pax = df["itcs_numberOfPassengers"].dropna()
q1, q2 = compute_tercile_boundaries(pax)
print(f"  Tercile boundaries: q1={q1:.1f}, q2={q2:.1f}")
print(f"  Min={pax.min():.0f}, Mean={pax.mean():.1f}, Median={pax.median():.0f}, Max={pax.max():.0f}")
print(f"  Skewness={pax.skew():.2f}")

fig, ax = plt.subplots()
ax.hist(pax, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
ax.axvline(q1, color="orange", linestyle="--", linewidth=2, label=f"33rd pctl ({q1:.1f})")
ax.axvline(q2, color="red", linestyle="--", linewidth=2, label=f"67th pctl ({q2:.1f})")
ax.set_xlabel("Passenger Count")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Passenger Counts (Sampled Missions)")
ax.legend()
save_fig(fig, "01_passenger_distribution.png")


# -----------------------------------------------------------------------
# 3. Temporal distributions (Figure 02)
# -----------------------------------------------------------------------

print("\n--- Figure 02: Temporal distributions ---")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Observations by hour
hour_counts = df["hour"].value_counts().sort_index()
axes[0].bar(hour_counts.index, hour_counts.values, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Observation Count")
axes[0].set_title("Observations by Hour")
axes[0].set_xticks(range(0, 24, 3))

# Observations by day of week
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_counts = df["dayofweek"].value_counts().sort_index()
axes[1].bar(dow_counts.index, dow_counts.values, color="#55A868", edgecolor="white")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Observation Count")
axes[1].set_title("Observations by Day of Week")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_labels)

# Observations by month
month_counts = df["month"].value_counts().sort_index()
axes[2].bar(month_counts.index, month_counts.values, color="#C44E52", edgecolor="white")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Observation Count")
axes[2].set_title("Observations by Month")
axes[2].set_xticks(range(1, 13))

fig.suptitle("Temporal Distribution of Observations", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "02_temporal_distributions.png")


# -----------------------------------------------------------------------
# 4. Ridership by time of day (Figure 03)
# -----------------------------------------------------------------------

print("\n--- Figure 03: Ridership by time of day ---")
fig, ax = plt.subplots()

weekday_pax = df[~df["is_weekend"]].groupby("hour")["itcs_numberOfPassengers"].mean()
weekend_pax = df[df["is_weekend"]].groupby("hour")["itcs_numberOfPassengers"].mean()

ax.plot(weekday_pax.index, weekday_pax.values, "o-", color="#4C72B0", linewidth=2, label="Weekday")
if len(weekend_pax) > 0:
    ax.plot(weekend_pax.index, weekend_pax.values, "s--", color="#C44E52", linewidth=2, label="Weekend")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean Passenger Count")
ax.set_title("Mean Ridership by Hour (Weekday vs Weekend)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, "03_ridership_by_hour.png")

if len(weekday_pax) > 0:
    peak_hour = weekday_pax.idxmax()
    print(f"  Weekday peak: hour {peak_hour} ({weekday_pax.max():.1f} avg passengers)")
if len(weekend_pax) > 0:
    print(f"  Weekend peak: hour {weekend_pax.idxmax()} ({weekend_pax.max():.1f} avg passengers)")


# -----------------------------------------------------------------------
# 5. Seasonal trends (Figure 04)
# -----------------------------------------------------------------------

print("\n--- Figure 04: Seasonal trends ---")
df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
monthly_pax = df.groupby("year_month")["itcs_numberOfPassengers"].mean().sort_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(monthly_pax)), monthly_pax.values, "o-", color="#4C72B0", linewidth=2)
ax.set_xticks(range(len(monthly_pax)))
ax.set_xticklabels(monthly_pax.index, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Year-Month")
ax.set_ylabel("Mean Passenger Count")
ax.set_title("Monthly Mean Ridership (Seasonal Trends)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "04_seasonal_trends.png")


# -----------------------------------------------------------------------
# 6. Stop-level demand (Figure 05)
# -----------------------------------------------------------------------

print("\n--- Figure 05: Stop-level demand ---")
stop_pax = (
    df[df["itcs_stopName"] != "-"]
    .groupby("itcs_stopName")["itcs_numberOfPassengers"]
    .mean()
    .sort_values(ascending=False)
)

# Show top 25 stops
top_n = min(25, len(stop_pax))
top_stops = stop_pax.head(top_n)

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(top_n), top_stops.values, color="#4C72B0", edgecolor="white")
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_stops.index, fontsize=9)
ax.set_xlabel("Mean Passenger Count")
ax.set_title(f"Top {top_n} Stops by Mean Ridership")
ax.invert_yaxis()
fig.tight_layout()
save_fig(fig, "05_stop_level_demand.png")

print(f"  Unique stops: {len(stop_pax)}")
print(f"  Top stop: {stop_pax.index[0]} ({stop_pax.iloc[0]:.1f} avg)")


# -----------------------------------------------------------------------
# 7. Route comparison (Figure 06)
# -----------------------------------------------------------------------

print("\n--- Figure 06: Route comparison ---")
routes = df["itcs_busRoute"].unique()
routes = [r for r in routes if r != "-"]

fig, ax = plt.subplots()
data_by_route = [df[df["itcs_busRoute"] == r]["itcs_numberOfPassengers"].dropna() for r in routes]
labels = [f"Route {r}" for r in routes]

parts = ax.violinplot(data_by_route, showmeans=True, showmedians=True)
ax.set_xticks(range(1, len(routes) + 1))
ax.set_xticklabels(labels)
ax.set_ylabel("Passenger Count")
ax.set_title("Passenger Count Distribution by Route")

for route, data in zip(routes, data_by_route):
    print(f"  Route {route}: n={len(data):,}, mean={data.mean():.1f}, median={data.median():.0f}")

save_fig(fig, "06_route_comparison.png")


# -----------------------------------------------------------------------
# 8. Correlation matrix (Figure 07)
# -----------------------------------------------------------------------

print("\n--- Figure 07: Correlation matrix ---")
numeric_cols = [
    "itcs_numberOfPassengers", "speed_kmh", "temp_C",
    "electric_powerDemand", "traction_tractionForce", "traction_brakePressure",
    "lat_deg", "lon_deg", "gnss_altitude",
]
available_cols = [c for c in numeric_cols if c in df.columns]
corr = df[available_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, ax=ax, square=True,
    linewidths=0.5, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix")
fig.tight_layout()
save_fig(fig, "07_correlation_matrix.png")

# Report strongest correlations with passenger count
pax_corr = corr["itcs_numberOfPassengers"].drop("itcs_numberOfPassengers").abs().sort_values(ascending=False)
print("  Strongest correlations with passenger count:")
for feat, val in pax_corr.head(5).items():
    sign = "+" if corr.loc["itcs_numberOfPassengers", feat] > 0 else "-"
    print(f"    {feat}: {sign}{val:.3f}")


# -----------------------------------------------------------------------
# 9. Temperature vs. ridership (Figure 08)
# -----------------------------------------------------------------------

print("\n--- Figure 08: Temperature vs. ridership ---")
fig, ax = plt.subplots()

sample_idx = df.dropna(subset=["temp_C", "itcs_numberOfPassengers"]).sample(
    n=min(5000, len(df)), random_state=RANDOM_SEED
).index

ax.scatter(
    df.loc[sample_idx, "temp_C"],
    df.loc[sample_idx, "itcs_numberOfPassengers"],
    alpha=0.15, s=8, color="#4C72B0",
)
ax.set_xlabel("Temperature (C)")
ax.set_ylabel("Passenger Count")
ax.set_title("Temperature vs. Ridership (5k sample)")
ax.grid(True, alpha=0.3)
save_fig(fig, "08_temp_vs_ridership.png")

temp_pax_corr = df[["temp_C", "itcs_numberOfPassengers"]].corr().iloc[0, 1]
print(f"  Pearson correlation: {temp_pax_corr:.3f}")


# -----------------------------------------------------------------------
# 10. Door events analysis (Figure 09)
# -----------------------------------------------------------------------

print("\n--- Figure 09: Door events and boarding ---")
if "status_doorIsOpen" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Coerce to bool for consistent grouping (real CSVs use 0/1 int)
    door_bool = df["status_doorIsOpen"].astype(bool)

    # Door open vs passenger count
    door_pax = df.groupby(door_bool)["itcs_numberOfPassengers"].mean()
    axes[0].bar(
        ["Door Closed", "Door Open"],
        [door_pax.get(False, 0), door_pax.get(True, 0)],
        color=["#4C72B0", "#C44E52"], edgecolor="white",
    )
    axes[0].set_ylabel("Mean Passenger Count")
    axes[0].set_title("Mean Passengers by Door State")

    # Speed when doors open vs closed
    door_speed = df.groupby(door_bool)["speed_kmh"].mean()
    axes[1].bar(
        ["Door Closed", "Door Open"],
        [door_speed.get(False, 0), door_speed.get(True, 0)],
        color=["#4C72B0", "#C44E52"], edgecolor="white",
    )
    axes[1].set_ylabel("Mean Speed (km/h)")
    axes[1].set_title("Mean Speed by Door State")

    fig.suptitle("Door Events Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "09_door_events.png")

    n_open = door_bool.sum()
    print(f"  Door open observations: {n_open:,} ({n_open/len(df)*100:.1f}%)")
    print(f"  Mean pax (door open): {door_pax.get(True, 0):.1f}")
    print(f"  Mean pax (door closed): {door_pax.get(False, 0):.1f}")


# -----------------------------------------------------------------------
# 11. Missing data assessment (Figure 10)
# -----------------------------------------------------------------------

print("\n--- Figure 10: Missing data assessment ---")
missing_stats = detect_missing_values(df)
missing_cols = missing_stats[missing_stats["nan_count"] > 0].sort_values("nan_pct", ascending=False)

if len(missing_cols) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_cols) * 0.4)))
    ax.barh(
        range(len(missing_cols)),
        missing_cols["nan_pct"].values,
        color="#C44E52", edgecolor="white",
    )
    ax.set_yticks(range(len(missing_cols)))
    ax.set_yticklabels(missing_cols["column"].values, fontsize=9)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "10_missing_data.png")
else:
    print("  No missing values found in sampled data!")
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center", fontsize=20)
    ax.set_axis_off()
    save_fig(fig, "10_missing_data.png")

# Also check dash placeholders
dash_cols = missing_stats[missing_stats["dash_count"] > 0].sort_values("dash_pct", ascending=False)
if len(dash_cols) > 0:
    print("  Dash placeholders:")
    for _, row in dash_cols.iterrows():
        print(f"    {row['column']}: {row['dash_count']:,} ({row['dash_pct']:.1f}%)")

# Print full missing data summary
print("\n  Full missing data summary:")
for _, row in missing_stats.iterrows():
    if row["nan_count"] > 0 or row["dash_count"] > 0:
        print(f"    {row['column']}: {row['nan_count']:,} NaN ({row['nan_pct']:.1f}%), {row['dash_count']:,} dash ({row['dash_pct']:.1f}%)")


# -----------------------------------------------------------------------
# 12. Target variable distribution (Figure 11 -- bonus)
# -----------------------------------------------------------------------

print("\n--- Figure 11: Target variable class distribution ---")
demand_labels = assign_demand_class(pax, q1, q2)
class_counts = demand_labels.value_counts()

fig, ax = plt.subplots()
colors = {"low": "#55A868", "medium": "#4C72B0", "high": "#C44E52"}
bars = ax.bar(
    class_counts.index,
    class_counts.values,
    color=[colors.get(c, "#999999") for c in class_counts.index],
    edgecolor="white",
)
ax.set_xlabel("Demand Class")
ax.set_ylabel("Observation Count")
ax.set_title(f"Target Variable Distribution (q1={q1:.1f}, q2={q2:.1f})")
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(pax)*0.005,
            f"{count:,}\n({count/len(pax)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
save_fig(fig, "11_target_distribution.png")

for label in ["low", "medium", "high"]:
    cnt = class_counts.get(label, 0)
    print(f"  {label}: {cnt:,} ({cnt/len(pax)*100:.1f}%)")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"Figures saved to: {FIGURES_DIR}/")
print(f"Total figures: {len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])}")
print("\nGATE 7: HUMAN INTERVENTION REQUIRED")
print("Action needed: Review all figures in figures/")
print("Context: Visual confirmation needed before proceeding to feature engineering")
