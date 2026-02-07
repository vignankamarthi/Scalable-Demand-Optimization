# Data Exploration

Prior to model development, we propose the following exploratory analyses to understand the structure, distributions, and relationships within the dataset.

## Distribution Analysis

- **Passenger count distribution**: Histogram of `itcs_numberOfPassengers` across all observations to understand the overall demand landscape and inform the selection of thresholds for the three-class target variable (low, medium, high). We expect a right-skewed distribution, with the majority of observations falling in the low-to-moderate range and fewer instances of high-demand conditions.
- **Temporal distributions**: Frequency plots of missions and observations by hour of day, day of week, and month to identify data coverage gaps and temporal sampling biases that could affect model training.

## Temporal Patterns

- **Ridership by time of day**: Line plots of mean passenger count by hour, segmented by weekday versus weekend, to visualize rush hour peaks and off-peak valleys. These patterns are central to the temporal feature set.
- **Seasonal trends**: Monthly aggregation of mean ridership to assess whether demand shifts across seasons, particularly given the dataset's 3.5-year span that includes pre- and post-pandemic periods (2019--2022).

## Spatial Analysis

- **Stop-level demand heatmap**: Aggregated mean passenger counts by `itcs_stopName`, visualized as a bar chart or geographic heatmap using GPS coordinates, to identify high-demand stops and spatial clustering of ridership.
- **Route comparison**: Side-by-side distributions of demand on the two trolleybus routes to determine whether route-specific models or a unified model is more appropriate.

## Feature Relationships

- **Correlation matrix**: Pairwise correlations among numerical features (speed, temperature, power demand, passenger count) to identify collinear variables and potential predictive signals. We anticipate that vehicle speed and door open status will show meaningful association with passenger count, as stops with high boarding activity involve longer dwell times and lower speeds.
- **Temperature vs. ridership**: Scatter plot of `temperature_ambient` against `itcs_numberOfPassengers` to explore whether weather conditions influence demand levels.
- **Door events and boarding**: Analysis of the relationship between `status_doorIsOpen` duration at each stop and the change in passenger count, which could serve as a proxy for boarding intensity.

## Missing Data Assessment

- **NaN prevalence by column**: Column-wise counts of missing values to quantify the extent of data gaps, particularly in GNSS fields which the dataset authors left intentionally unfiltered. This assessment will guide imputation strategy during preprocessing.
