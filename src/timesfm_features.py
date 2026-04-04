"""
TimesFM embedding loading and integration as features.
Embeddings are pre-computed on the cluster and cached as NPY files.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


EMBEDDING_DIM = 32


def load_mission_embeddings(
    mission_name: str,
    cache_dir: str,
) -> np.ndarray | None:
    """
    Load pre-computed PCA-reduced embedding for a mission.

    Returns 1D numpy array of shape (EMBEDDING_DIM,), or None if not cached.
    """
    path = os.path.join(cache_dir, f"{mission_name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def attach_embeddings(
    df: pd.DataFrame,
    cache_dir: str,
) -> pd.DataFrame:
    """
    For each mission in df, load its cached embedding and broadcast
    the EMBEDDING_DIM-dimensional vector to all rows of that mission.

    Adds columns: tfm_emb_0, tfm_emb_1, ..., tfm_emb_{EMBEDDING_DIM-1}.
    Missions without cached embeddings get NaN in all embedding columns.
    Does not modify input DataFrame.
    """
    emb_cols = [f"tfm_emb_{i}" for i in range(EMBEDDING_DIM)]

    # Build a small lookup table: mission_name -> 32 embedding values
    records = []
    for mission_name in df["mission_name"].unique():
        emb = load_mission_embeddings(mission_name, cache_dir)
        if emb is not None:
            row = {"mission_name": mission_name}
            for i, col in enumerate(emb_cols):
                row[col] = emb[i]
            records.append(row)

    if not records:
        # No embeddings found -- add NaN columns
        result = df.copy()
        for col in emb_cols:
            result[col] = np.nan
        return result

    # Merge: one hash join instead of 45K .loc operations
    emb_df = pd.DataFrame(records)
    result = df.merge(emb_df, on="mission_name", how="left")

    return result
