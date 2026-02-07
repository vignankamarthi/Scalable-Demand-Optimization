# Test Validation Strategy

## Hard Rule

ALL source code MUST have tests written FIRST. No source module runs in production until its test suite passes with 100% deterministic, mock-based validation. This is non-negotiable.

## Architecture

```
src/                          tests/
  config.py                     (no tests -- constants only)
  data_loading.py               test_data_loading.py
  preprocessing.py              test_preprocessing.py
  target.py                     test_target.py
  [future modules]              [mirrored test files]
```

Every `src/*.py` module (except `config.py`) has a corresponding `tests/test_*.py` file. New modules added to `src/` MUST have a test file created BEFORE any implementation code is written.

## Shared Fixtures (`conftest.py`)

All test data is synthetic, deterministic, and fast. No real dataset access in any test.

| Fixture | Shape | Purpose |
|---------|-------|---------|
| `mock_mission_df` | 20 rows, 16 cols | Standard mission with known values (Tuesday 2019-04-30, Zurich coords, mixed passengers) |
| `mock_mission_with_nans` | 20 rows | Same as above with GNSS gaps (rows 3-7) and 1 passenger NaN |
| `mock_mission_with_dashes` | 20 rows | Same as above with "-" placeholders in string columns |
| `mock_metadata_df` | 5 rows, 16 cols | Synthetic metadata (3 routes, 2 bus numbers, realistic timestamps) |
| `mock_passenger_series` | 30 values | Known distribution: 10 low (0-5), 10 medium (10-15), 10 high (25-35) |
| `weekend_mission_df` | 10 rows | Saturday mission for weekend detection |
| `rush_hour_mission_df` | 4 rows | Exact rush hour boundaries (6:59, 7:00, 15:59, 16:00 on Wednesday) |

## Test Coverage by Module

### `test_data_loading.py` (11 tests)

- **TestLoadMetadata**: Timestamp parsing (ISO -> datetime), duration computation (unix delta -> hours), column preservation, row count
- **TestStratifiedSampleMissions**: Correct sample count, route representation, seed determinism, different seed produces different results
- **TestLoadMissionCsv**: File loading, missing file returns None, column subset selection

### `test_preprocessing.py` (30 tests)

- **TestKelvinToCelsius**: Freezing/boiling points, Zurich temp range, NaN preservation, empty series
- **TestRadiansToDegrees**: Zero, pi, Zurich lat/lon coordinates, NaN preservation
- **TestMpsToKmh**: Zero, known conversion (10 m/s = 36 km/h), bus speed range, NaN preservation
- **TestExtractTemporalFeatures**: Hour/dayofweek/month/year extraction, weekend detection (weekday vs Saturday), rush hour boundaries (AM/PM on weekday, never on weekend), input immutability, output column completeness
- **TestDetectMissingValues**: Clean data, NaN detection with counts, dash detection in string columns, numeric columns never have dashes, percentage calculation, empty DataFrame
- **TestForwardFillWithinMission**: Basic fill, leading NaN stays, no NaNs passthrough, all NaNs
- **TestApplyUnitConversions**: Each derived column (temp_C, lat_deg, lon_deg, speed_kmh), input immutability, graceful handling of missing source columns

### `test_target.py` (14 tests)

- **TestComputeTercileBoundaries**: Uniform distribution, known values with expected ranges, return types, q1 < q2 ordering, NaN handling, degenerate single-value input
- **TestAssignDemandClass**: All boundary conditions (below, at low, between, at high, above), NaN input -> NaN output, multiple values in one call, index preservation
- **TestComputeClassDistribution**: Balanced/imbalanced distributions, missing class -> 0 count, all three keys always present

## Test Execution

```bash
# From repo root with venv activated
python3 -m pytest tests/ -v

# Single module
python3 -m pytest tests/test_preprocessing.py -v

# Single test class
python3 -m pytest tests/test_preprocessing.py::TestExtractTemporalFeatures -v
```

## Pandas 3.0 Compatibility Notes

This project uses pandas 3.0.0, which introduces breaking changes relevant to testing:

1. **String columns use `StringDtype` (`str`) not `object`**: Check string dtype with `pd.api.types.is_string_dtype()`, never `dtype == object`
2. **CSV round-trip type inference**: Numeric-looking strings (e.g., route "83") become integers after `to_csv` -> `read_csv`. Explicitly cast categorical identifiers with `.astype(str)`
3. **NumPy 2.x strict type promotion**: `np.select` cannot mix string choices with float defaults. Use a string sentinel and replace afterward

## What Is NOT Tested (And Why)

- **Visualization functions**: Tested via human visual review at GATE 7 checkpoints
- **EDA orchestration scripts**: These compose tested primitives -- integration testing at visual review gates
- **config.py**: Constants only, no logic to test
- **ML model training loops**: HARD GATE -- designed collaboratively before implementation, tested as a separate phase
