# Problem Definition

The objective of this project is to build a classification model that predicts the expected ridership demand level at a given time and location along a bus route. Rather than predicting an exact passenger count, demand is discretized into three ordinal categories -- low, medium, and high -- based on the distribution of observed onboard passenger counts across the dataset. The thresholds defining these categories will be determined empirically from the data, using percentile-based binning to ensure balanced or interpretable class boundaries.

For each observation, the model receives as input a set of features reconstructed from onboard sensor data and contextual information:

- **Temporal features**: time of day, day of the week, month, and binary indicators for rush hour windows.
- **Spatial features**: stop identity, GPS coordinates, and route designation.
- **Operational features**: door open/close events, vehicle speed, halt brake activation, and overhead grid connection status.
- **Environmental features**: ambient temperature at the time of observation.
- **Lagged ridership features**: recent passenger count history from preceding stops or time windows along the same mission.

The classification task is framed to be bus-system agnostic. While the model is trained and evaluated on Zurich trolleybus data, the feature set is deliberately constructed from signals that exist on any modern transit vehicle equipped with standard onboard sensors. Time-of-day patterns, stop-level boarding variation, and operational state indicators are universal to bus networks in cities such as New York, Boston, Los Angeles, or any system with automatic passenger counting infrastructure. The goal is to demonstrate that a demand classification pipeline built on high-resolution sensor data can serve as a transferable template for transit agencies operating in different urban contexts.

The project seeks to answer the following questions through data analytics:

1. Can onboard sensor data, combined with temporal and spatial context, reliably classify stop-level demand into discrete categories?
2. Which feature groups (temporal, spatial, operational, environmental) contribute most to prediction accuracy?
3. How do different classification algorithms compare in performance on this structured, time-indexed transit dataset?
