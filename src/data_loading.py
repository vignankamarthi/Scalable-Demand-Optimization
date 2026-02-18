"""
Data loading utilities for ZTBus dataset.
"""
from __future__ import annotations

import os
import random
import pandas as pd
from src.config import DATA_DIR, METADATA_PATH, EDA_USE_COLS, RANDOM_SEED


def load_metadata(path: str = METADATA_PATH) -> pd.DataFrame:
    """
    Load and parse the metaData.csv file.
    Parses ISO timestamps and computes derived columns.
    """
    meta = pd.read_csv(path)

    meta["startTime_iso"] = pd.to_datetime(meta["startTime_iso"])
    meta["endTime_iso"] = pd.to_datetime(meta["endTime_iso"])
    meta["duration_hours"] = (meta["endTime_unix"] - meta["startTime_unix"]) / 3600

    if "busRoute" in meta.columns:
        meta["busRoute"] = meta["busRoute"].astype(str)

    return meta


def stratified_sample_missions(
    meta: pd.DataFrame,
    n: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sample n missions from metadata, stratified by route.
    Returns a subset of the metadata DataFrame.
    """
    rng = random.Random(seed)

    route_counts = meta["busRoute"].value_counts()
    sample_indices = []

    for route, count in route_counts.items():
        route_missions = meta[meta["busRoute"] == route].index.tolist()
        n_sample = max(1, int(round(n * count / len(meta))))
        sampled = rng.sample(route_missions, min(n_sample, len(route_missions)))
        sample_indices.extend(sampled)

    rng.shuffle(sample_indices)
    sample_indices = sample_indices[:n]

    return meta.loc[sample_indices]


def load_mission_csv(
    mission_name: str,
    data_dir: str = DATA_DIR,
    usecols: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Load a single mission CSV by name (without .csv extension).
    Returns DataFrame or None if file not found or load fails.
    """
    fpath = os.path.join(data_dir, mission_name + ".csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, usecols=usecols)
        return df
    except Exception:
        return None


def load_sample_missions(
    meta_sample: pd.DataFrame,
    data_dir: str = DATA_DIR,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load all mission CSVs referenced in a metadata sample.
    Concatenates into a single DataFrame with mission_name and busNumber columns.
    """
    dfs = []
    for _, row in meta_sample.iterrows():
        df = load_mission_csv(row["name"], data_dir=data_dir, usecols=usecols)
        if df is not None:
            df["mission_name"] = row["name"]
            df["busNumber"] = row["busNumber"]
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
