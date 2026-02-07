# Performance Evaluation

Evaluating a multiclass classification model requires metrics that account for per-class performance and potential class imbalance. Standard accuracy alone can be misleading when demand levels are unevenly distributed, which is likely given that low-demand observations may dominate the dataset. We propose the following evaluation framework.

## Primary Metrics

**Macro-Averaged F1 Score**: The F1 score is the harmonic mean of precision and recall, computed per class and then averaged across all three demand levels (low, medium, high) with equal weight. Macro-averaging ensures that the model's ability to correctly classify each demand level is weighted equally, regardless of class frequency. This is critical in the transit context because correctly identifying high-demand periods -- even if they are less frequent -- is operationally more valuable than achieving high accuracy on the dominant low-demand class.

**Balanced Accuracy**: Defined as the average of per-class recall values. Like macro F1, balanced accuracy corrects for class imbalance by giving equal weight to each class's true positive rate. It provides a single scalar summary that reflects whether the model performs well across all demand levels rather than just the majority class.

## Diagnostic Tools

**Confusion Matrix**: A 3x3 confusion matrix for each model to visualize the pattern of misclassifications. In the demand prediction context, certain errors are more costly than others -- confusing high demand with low demand is a larger operational failure than confusing medium with high. The confusion matrix allows us to inspect whether errors are concentrated in adjacent classes (medium-high boundary) or span the full range.

**Per-Class Precision and Recall**: Reported individually for each demand level to identify whether specific classes are systematically under- or over-predicted. If the high-demand class has low recall, for example, the model may need rebalancing through oversampling or class-weighted loss functions.

## Evaluation Protocol

All models will be evaluated using stratified k-fold cross-validation to preserve the class distribution across folds. The train-test split will be constructed at the mission level rather than the observation level to prevent data leakage -- observations from the same mission share temporal and spatial context, so mixing them across train and test sets would inflate performance estimates. Hyperparameters for each model will be tuned on a validation set held out from the training folds.
