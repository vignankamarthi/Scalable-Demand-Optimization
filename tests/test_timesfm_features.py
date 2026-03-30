"""
Tests for src/timesfm_features.py -- TimesFM embedding loading and integration.
All tests use synthetic NPY files (no model download, no GPU needed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.timesfm_features import load_mission_embeddings, attach_embeddings


EMBEDDING_DIM = 32


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temp cache directory with synthetic embeddings."""
    cache = tmp_path / "embeddings"
    cache.mkdir()

    # Mission A: 32-dim embedding (all 1s)
    np.save(cache / "mission_A.npy", np.ones(EMBEDDING_DIM))
    # Mission B: 32-dim embedding (all 2s)
    np.save(cache / "mission_B.npy", np.full(EMBEDDING_DIM, 2.0))

    return str(cache)


@pytest.fixture
def two_mission_df():
    """DataFrame with two missions, 5 rows each."""
    return pd.DataFrame({
        "mission_name": ["mission_A"] * 5 + ["mission_B"] * 5,
        "hour": [10] * 5 + [14] * 5,
        "speed_kmh": [30.0] * 10,
    })


# -----------------------------------------------------------------------
# load_mission_embeddings
# -----------------------------------------------------------------------

class TestLoadMissionEmbeddings:
    """Tests for loading pre-computed embeddings from cache."""

    def test_loads_existing_embedding(self, cache_dir):
        """Should load a cached NPY file and return the array."""
        emb = load_mission_embeddings("mission_A", cache_dir)
        assert emb is not None
        assert len(emb) == EMBEDDING_DIM
        assert np.allclose(emb, 1.0)

    def test_returns_none_for_missing(self, cache_dir):
        """Should return None for missions without cached embeddings."""
        emb = load_mission_embeddings("nonexistent_mission", cache_dir)
        assert emb is None

    def test_different_missions_different_values(self, cache_dir):
        """Different missions should have different embeddings."""
        emb_a = load_mission_embeddings("mission_A", cache_dir)
        emb_b = load_mission_embeddings("mission_B", cache_dir)
        assert not np.allclose(emb_a, emb_b)

    def test_returns_numpy_array(self, cache_dir):
        """Return type should be numpy array."""
        emb = load_mission_embeddings("mission_A", cache_dir)
        assert isinstance(emb, np.ndarray)


# -----------------------------------------------------------------------
# attach_embeddings
# -----------------------------------------------------------------------

class TestAttachEmbeddings:
    """Tests for broadcasting embeddings to all rows of each mission."""

    def test_adds_embedding_columns(self, two_mission_df, cache_dir):
        """Should add tfm_emb_0 through tfm_emb_31 columns."""
        result = attach_embeddings(two_mission_df, cache_dir)
        emb_cols = [c for c in result.columns if c.startswith("tfm_emb_")]
        assert len(emb_cols) == EMBEDDING_DIM

    def test_broadcasts_to_all_rows(self, two_mission_df, cache_dir):
        """All rows of a mission should have the same embedding values."""
        result = attach_embeddings(two_mission_df, cache_dir)
        mission_a = result[result["mission_name"] == "mission_A"]
        # mission_A embedding is all 1s
        for col in [f"tfm_emb_{i}" for i in range(EMBEDDING_DIM)]:
            assert (mission_a[col] == 1.0).all()

    def test_different_missions_get_different_embeddings(self, two_mission_df, cache_dir):
        """Mission A gets 1s, Mission B gets 2s."""
        result = attach_embeddings(two_mission_df, cache_dir)
        a_val = result.loc[result["mission_name"] == "mission_A", "tfm_emb_0"].iloc[0]
        b_val = result.loc[result["mission_name"] == "mission_B", "tfm_emb_0"].iloc[0]
        assert a_val == 1.0
        assert b_val == 2.0

    def test_missing_mission_gets_nan(self, cache_dir):
        """Missions without cached embeddings should get NaN."""
        df = pd.DataFrame({
            "mission_name": ["mission_C"] * 3,
            "hour": [10] * 3,
        })
        result = attach_embeddings(df, cache_dir)
        emb_cols = [c for c in result.columns if c.startswith("tfm_emb_")]
        assert len(emb_cols) == EMBEDDING_DIM
        assert result[emb_cols].isna().all().all()

    def test_preserves_existing_columns(self, two_mission_df, cache_dir):
        """Original columns should be untouched."""
        result = attach_embeddings(two_mission_df, cache_dir)
        assert "hour" in result.columns
        assert "speed_kmh" in result.columns
        assert list(result["hour"]) == [10] * 5 + [14] * 5

    def test_preserves_row_count(self, two_mission_df, cache_dir):
        """Row count should not change."""
        result = attach_embeddings(two_mission_df, cache_dir)
        assert len(result) == len(two_mission_df)

    def test_does_not_modify_input(self, two_mission_df, cache_dir):
        """Input DataFrame should not be mutated."""
        original_cols = set(two_mission_df.columns)
        _ = attach_embeddings(two_mission_df, cache_dir)
        assert set(two_mission_df.columns) == original_cols

    def test_mixed_cached_and_missing(self, cache_dir):
        """Mix of cached and uncached missions."""
        df = pd.DataFrame({
            "mission_name": ["mission_A"] * 3 + ["mission_C"] * 3,
            "hour": [10] * 6,
        })
        result = attach_embeddings(df, cache_dir)
        # mission_A: should have values
        a_rows = result[result["mission_name"] == "mission_A"]
        assert a_rows["tfm_emb_0"].notna().all()
        # mission_C: should be NaN
        c_rows = result[result["mission_name"] == "mission_C"]
        assert c_rows["tfm_emb_0"].isna().all()
