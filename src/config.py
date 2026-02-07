"""
Centralized configuration for the project.
All paths, constants, and hyperparameters in one place.
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METADATA_PATH = os.path.join(DATA_DIR, "metaData.csv")
# Note: data lives at data/processed/raw/ (from compressed extraction)

# Reproducibility
RANDOM_SEED = 42

# EDA sampling
EDA_SAMPLE_SIZE = 150

# Feature engineering
RUSH_HOUR_AM = (7, 9)       # 7:00 - 8:59
RUSH_HOUR_PM = (16, 19)     # 16:00 - 18:59
KELVIN_OFFSET = 273.15
ROLLING_WINDOWS = [60, 300]  # seconds

# Target variable
N_DEMAND_CLASSES = 3
DEMAND_LABELS = ["low", "medium", "high"]

# Columns from the dataset we use for EDA / modeling
TIME_COLS = ["time_iso", "time_unix"]
GNSS_COLS = ["gnss_latitude", "gnss_longitude", "gnss_altitude", "gnss_course"]
ITCS_COLS = ["itcs_busRoute", "itcs_numberOfPassengers", "itcs_stopName"]
ODOMETRY_COLS = ["odometry_vehicleSpeed"]
STATUS_COLS = [
    "status_doorIsOpen", "status_gridIsAvailable",
    "status_haltBrakeIsActive", "status_parkBrakeIsActive",
]
SENSOR_COLS = ["temperature_ambient", "electric_powerDemand",
               "traction_tractionForce", "traction_brakePressure"]

EDA_USE_COLS = TIME_COLS + GNSS_COLS[:3] + ITCS_COLS + ODOMETRY_COLS + STATUS_COLS + SENSOR_COLS
