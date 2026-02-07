# Data Mining Models/Methods

We propose evaluating four classification algorithms that span different modeling paradigms, from interpretable tree-based methods to nonlinear neural architectures. Each model is selected for specific properties that align with the structure of the transit demand prediction task.

## Decision Tree

A single decision tree serves as the interpretable baseline. Decision trees partition the feature space through recursive binary splits, producing a model whose predictions can be traced through a sequence of human-readable rules. For transit demand classification, this interpretability is operationally valuable: a transit planner can inspect the tree to understand which features (e.g., time of day, stop identity) drive demand predictions at specific locations. Decision trees also handle mixed feature types (numerical and categorical) natively, which suits the heterogeneous feature set in this project. However, single trees are prone to overfitting on high-dimensional data, motivating the ensemble approach below.

## Random Forest

Random Forest constructs an ensemble of decorrelated decision trees, each trained on a bootstrap sample of the data with a random subset of features considered at each split. The ensemble averaging reduces variance and improves generalization compared to a single tree. For this dataset, Random Forest is well-suited because it handles the mix of continuous sensor readings (speed, temperature, power demand) and categorical variables (stop name, route) without requiring extensive feature scaling. It also provides built-in feature importance scores, which directly address the project's second research question regarding which feature groups contribute most to prediction accuracy.

## Multilayer Perceptron (MLP)

To explore whether nonlinear feature interactions improve classification, we propose training multilayer perceptron networks at three capacity levels:

- **Small**: 2 hidden layers (64, 32 neurons) -- tests whether a lightweight network can capture the core demand signal.
- **Medium**: 3 hidden layers (128, 64, 32 neurons) -- provides additional capacity for learning intermediate feature representations.
- **Large**: 4 hidden layers (256, 128, 64, 32 neurons) -- evaluates whether deeper architectures extract further predictive value from the feature set, or whether the additional parameters lead to overfitting.

MLPs are appropriate here because the feature space, after engineering, consists of fixed-length numerical and encoded categorical inputs. The three capacity levels allow us to assess the complexity-performance tradeoff: whether the structured, tabular nature of transit data is better served by simpler models or benefits from the representational flexibility of deeper networks. All MLP variants will use ReLU activations, softmax output, and be regularized with dropout and early stopping.

## k-Nearest Neighbors (k-NN)

k-NN classifies each observation by majority vote among its k closest training examples in feature space. It is a non-parametric method that makes no assumptions about the underlying data distribution, which is valuable when the decision boundary between demand levels may be irregular or locally varying. In the transit context, k-NN captures the intuition that observations with similar temporal, spatial, and operational characteristics should have similar demand levels. The primary hyperparameter, k, will be tuned via cross-validation. Because k-NN is sensitive to feature scaling and dimensionality, it will operate on the normalized feature set described in the Data Mining Tasks section.

## Model Selection Strategy

All four models (five configurations, including the three MLP variants) will be trained and evaluated under identical data splits and preprocessing pipelines. The best-performing model will be selected based on the evaluation metrics described in the Performance Evaluation section, balancing predictive accuracy with model complexity and interpretability.
