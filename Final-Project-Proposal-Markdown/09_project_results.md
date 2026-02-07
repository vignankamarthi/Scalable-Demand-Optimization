# Project Results

Upon completion, this project will deliver the following results and artifacts:

**Trained Classification Models**: A set of trained classifiers (Decision Tree, Random Forest, MLP variants, k-NN) capable of predicting demand level at a given stop-time observation. Each model will be evaluated under identical conditions to enable direct comparison.

**Model Comparison Summary**: A performance comparison table reporting macro F1, balanced accuracy, and per-class precision/recall for each model configuration. This summary will identify which algorithm best captures the demand classification task and whether increased model complexity (e.g., larger MLP) translates to meaningful performance gains over simpler approaches.

**Feature Importance Analysis**: For tree-based models, feature importance rankings derived from split-based or permutation importance. This analysis will reveal whether temporal features (time of day, day of week), spatial features (stop identity), or operational features (door events, speed) are the dominant predictors of demand, addressing the project's second research question.

**Exploratory Visualizations**: A set of plots and maps characterizing ridership patterns in the dataset, including temporal demand curves, stop-level heatmaps, and feature correlation structures. These visualizations contextualize the model results and provide standalone descriptive value for understanding transit demand dynamics.

**Preprocessing Pipeline**: A documented and reproducible data pipeline that transforms raw second-resolution sensor data into stop-level, model-ready observations with engineered features and a discretized target variable. This pipeline is designed to be applicable to any transit dataset with comparable sensor instrumentation.
