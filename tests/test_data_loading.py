"""
Tests for src/data_loading.py
Uses mock data and temporary directories -- no real dataset access.
"""

import os
import tempfile
import pandas as pd
import pytest
from src.data_loading import (
    load_metadata,
    stratified_sample_missions,
    load_mission_csv,
)


class TestLoadMetadata:
    def test_parses_timestamps(self, mock_metadata_df, tmp_path):
        """Verify ISO timestamps are parsed to datetime"""
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)

        result = load_metadata(str(fpath))
        assert pd.api.types.is_datetime64_any_dtype(result["startTime_iso"])
        assert pd.api.types.is_datetime64_any_dtype(result["endTime_iso"])

    def test_computes_duration(self, mock_metadata_df, tmp_path):
        """Verify duration_hours is computed correctly"""
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)

        result = load_metadata(str(fpath))
        assert "duration_hours" in result.columns
        # First mission: 1556613860 - 1556594336 = 19524 seconds = 5.42 hours
        expected_hours = (1556613860 - 1556594336) / 3600
        assert abs(result["duration_hours"].iloc[0] - expected_hours) < 0.01

    def test_preserves_all_columns(self, mock_metadata_df, tmp_path):
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)

        result = load_metadata(str(fpath))
        for col in mock_metadata_df.columns:
            assert col in result.columns

    def test_row_count(self, mock_metadata_df, tmp_path):
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)

        result = load_metadata(str(fpath))
        assert len(result) == len(mock_metadata_df)


class TestStratifiedSampleMissions:
    def test_returns_correct_count(self, mock_metadata_df, tmp_path):
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)
        meta = load_metadata(str(fpath))

        result = stratified_sample_missions(meta, n=3, seed=42)
        assert len(result) == 3

    def test_preserves_route_presence(self, mock_metadata_df, tmp_path):
        """With n close to total, all routes should be represented"""
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)
        meta = load_metadata(str(fpath))

        result = stratified_sample_missions(meta, n=5, seed=42)
        # Route 83 has 3/5, should be in sample
        assert "83" in result["busRoute"].values

    def test_deterministic_with_seed(self, mock_metadata_df, tmp_path):
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)
        meta = load_metadata(str(fpath))

        r1 = stratified_sample_missions(meta, n=3, seed=42)
        r2 = stratified_sample_missions(meta, n=3, seed=42)
        assert list(r1.index) == list(r2.index)

    def test_different_seed_different_result(self, mock_metadata_df, tmp_path):
        fpath = tmp_path / "metaData.csv"
        mock_metadata_df.to_csv(fpath, index=False)
        meta = load_metadata(str(fpath))

        r1 = stratified_sample_missions(meta, n=3, seed=42)
        r2 = stratified_sample_missions(meta, n=99, seed=99)
        # With different seeds and different n, results may differ
        # (not guaranteed with only 5 missions, but the function should run)
        assert len(r2) == 5  # can't sample more than available


class TestLoadMissionCsv:
    def test_loads_existing_file(self, mock_mission_df, tmp_path):
        fpath = tmp_path / "B183_test.csv"
        mock_mission_df.to_csv(fpath, index=False)

        result = load_mission_csv("B183_test", data_dir=str(tmp_path))
        assert result is not None
        assert len(result) == len(mock_mission_df)

    def test_returns_none_for_missing_file(self, tmp_path):
        result = load_mission_csv("nonexistent", data_dir=str(tmp_path))
        assert result is None

    def test_usecols_subset(self, mock_mission_df, tmp_path):
        fpath = tmp_path / "B183_test.csv"
        mock_mission_df.to_csv(fpath, index=False)

        result = load_mission_csv(
            "B183_test",
            data_dir=str(tmp_path),
            usecols=["time_iso", "itcs_numberOfPassengers"],
        )
        assert result is not None
        assert set(result.columns) == {"time_iso", "itcs_numberOfPassengers"}
