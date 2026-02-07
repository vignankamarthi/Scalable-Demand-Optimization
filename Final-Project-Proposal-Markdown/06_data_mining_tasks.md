# Data Mining Tasks

The raw ZTBus data requires several preprocessing and transformation steps before it can be used for classification. This section outlines the data mining tasks that form the pipeline from raw sensor streams to model-ready features.

## Missing Data Imputation

The dataset contains `NaN` values, most notably in GNSS columns (latitude, longitude, altitude, course) which were intentionally left unfiltered by the dataset authors. Our imputation strategy will be context-dependent:

- **GNSS fields**: Forward-fill interpolation within each mission, as GPS coordinates change smoothly along a bus route. For missions with large GNSS gaps, we will use stop name as a spatial proxy since `itcs_stopName` provides discrete location information independent of GPS availability.
- **Passenger count**: Any missing values in `itcs_numberOfPassengers` will be forward-filled within a mission under the assumption that passenger count changes occur at stops, not between them.
- **Remaining numerical fields**: Linear interpolation for short gaps; rows with extensive missing data across multiple columns will be flagged for potential exclusion.

## Feature Engineering

The raw second-resolution data must be transformed into observation-level features suitable for classification:

- **Temporal features**: Parse `time_iso` to extract cyclical and categorical time variables. Hour of day captures intra-day demand variation (morning commute, midday lull, evening peak). Day of week distinguishes weekday commuter patterns from weekend leisure ridership. Month encodes seasonal effects, particularly relevant given that the dataset spans 2019--2022 and includes both winter low-ridership periods and summer peaks. Binary rush hour indicators (7--9 AM, 4--7 PM on weekdays) flag the high-demand windows that are most operationally significant for transit scheduling. Together, these features encode the dominant temporal rhythms that drive ridership variation on any urban bus network.
- **Spatial features**: Encode `itcs_stopName` as a categorical variable. Derive distance-along-route from cumulative GPS displacement.
- **Operational features**: Compute door open duration per stop event, mean vehicle speed in the preceding time window, and binary indicators for halt/park brake activation.
- **Lagged ridership features**: Calculate rolling mean and rolling maximum of `itcs_numberOfPassengers` over preceding time windows (e.g., 60s, 300s) to capture recent demand trends within a mission.
- **Environmental features**: Convert `temperature_ambient` from Kelvin to Celsius for interpretability.

## Data Transformation

- **Unit conversions**: GNSS coordinates from radians to degrees, temperature from Kelvin to Celsius, speed from m/s to km/h where appropriate for interpretability.
- **Aggregation**: Collapse second-resolution data to stop-level observations by aggregating features within each stop event (defined by door open/close boundaries), producing one observation per stop per mission.
- **Normalization**: Scale continuous features (speed, temperature, power demand) to comparable ranges for distance-based algorithms (k-NN) and neural networks (MLP).

## Target Variable Construction

The classification target is derived from `itcs_numberOfPassengers` by discretizing observed counts into three ordinal demand levels:

- **Low**: Passenger count in the lower tercile of the dataset distribution.
- **Medium**: Passenger count in the middle tercile.
- **High**: Passenger count in the upper tercile.

Percentile-based binning ensures that class boundaries reflect the empirical distribution of demand rather than arbitrary fixed thresholds. The exact cutoff values will be determined after data exploration.

## Classification

With the engineered features and constructed target variable, the task becomes a supervised multiclass classification problem: given a feature vector describing a stop-time observation, predict the demand level (low, medium, or high). The specific models and evaluation criteria are described in the following sections.
