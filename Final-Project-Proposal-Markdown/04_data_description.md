# Data Description

The ZTBus dataset contains 1,409 complete driving missions recorded at one-second resolution from two electric trolleybuses (vehicles 183 and 208) operating on the Zurich VBZ transit network between April 2019 and December 2022. Both vehicles are HESS lighTram 19 DC single-articulated buses with approximately 160 passenger capacity. Each mission is stored as an individual CSV file, and a separate metadata file aggregates mission-level summary statistics.

## Time-Resolved Mission Data

Each mission CSV contains the following 26 columns, sampled at 1 Hz:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `time_iso` | datetime | -- | Absolute UTC time (ISO 8601) |
| `time_unix` | integer | s | Unix timestamp |
| `electric_powerDemand` | float | W | Overall electric power demand (traction motors + auxiliary consumers) |
| `gnss_altitude` | float | m | Altitude above sea level |
| `gnss_course` | float | rad | Heading direction from GNSS sensor |
| `gnss_latitude` | float | rad | WGS 84 latitude |
| `gnss_longitude` | float | rad | WGS 84 longitude |
| `itcs_busRoute` | string | -- | VBZ bus route name, provided by ITCS |
| `itcs_numberOfPassengers` | float | -- | Estimated number of passengers, measured via infrared counting |
| `itcs_stopName` | string | -- | Name of the bus stop served, provided by ITCS |
| `odometry_articulationAngle` | float | rad | Angle of the pivoting joint (articulation) |
| `odometry_steeringAngle` | float | rad | Angle of front wheels relative to vehicle body |
| `odometry_vehicleSpeed` | float | m/s | Vehicle speed from drive shaft rotational velocity |
| `odometry_wheelSpeed_fl` | float | m/s | Front left wheel speed |
| `odometry_wheelSpeed_fr` | float | m/s | Front right wheel speed |
| `odometry_wheelSpeed_ml` | float | m/s | Middle left wheel speed |
| `odometry_wheelSpeed_mr` | float | m/s | Middle right wheel speed |
| `odometry_wheelSpeed_rl` | float | m/s | Rear left wheel speed |
| `odometry_wheelSpeed_rr` | float | m/s | Rear right wheel speed |
| `status_doorIsOpen` | boolean | -- | Whether at least one door is open |
| `status_gridIsAvailable` | boolean | -- | Whether current collector is connected to overhead grid |
| `status_haltBrakeIsActive` | boolean | -- | Whether halt brake is active (auto-activated at standstill) |
| `status_parkBrakeIsActive` | boolean | -- | Whether park brake is active (manual fail-safe brake) |
| `temperature_ambient` | float | K | Ambient temperature (1 K resolution) |
| `traction_brakePressure` | float | Pa | Mean pressure in friction braking lines |
| `traction_tractionForce` | float | N | Estimated overall traction force from two traction motors |

## Metadata File

The `metaData.csv` file contains one row per mission (1,409 rows) with 16 columns summarizing each trip:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `name` | string | -- | Name of the corresponding data file |
| `busNumber` | integer | -- | Vehicle ID (183 or 208) |
| `startTime_iso` | datetime | -- | Mission start time (ISO 8601) |
| `startTime_unix` | integer | s | Mission start time (Unix timestamp) |
| `endTime_iso` | datetime | -- | Mission end time (ISO 8601) |
| `endTime_unix` | integer | s | Mission end time (Unix timestamp) |
| `drivenDistance` | float | m | Total distance covered during mission |
| `busRoute` | string | -- | Most frequently occurring route in the mission |
| `energyConsumption` | float | J | Total electric energy consumption during mission |
| `itcs_numberOfPassengers_mean` | float | -- | Mean passenger count (NaN omitted) |
| `itcs_numberOfPassengers_min` | float | -- | Minimum passenger count (NaN omitted) |
| `itcs_numberOfPassengers_max` | float | -- | Maximum passenger count (NaN omitted) |
| `status_gridIsAvailable_mean` | float | -- | Fraction of time connected to overhead grid |
| `temperature_ambient_mean` | float | K | Mean ambient temperature (NaN omitted) |
| `temperature_ambient_min` | float | K | Minimum ambient temperature (NaN omitted) |
| `temperature_ambient_max` | float | K | Maximum ambient temperature (NaN omitted) |

## Data Characteristics

- **File naming**: `B[BusNumber]_[StartTime]_[EndTime].csv` (e.g., `B183_2019-10-16_02-52-43_2019-10-16_07-10-12.csv`)
- **Missing values**: Numerical columns use `NaN`; string columns (`itcs_busRoute`, `itcs_stopName`) use `"-"` for unavailable data.
- **Wheel speed limitation**: Wheel encoder measurements are not reliable below approximately 1.5 m/s.
- **Scale**: At one-second resolution across 1,409 missions, the dataset contains on the order of millions of time-indexed observations.
- **Target variable**: The `itcs_numberOfPassengers` column, derived from an infrared-based onboard counting system, serves as the basis for constructing the three-class demand label.
