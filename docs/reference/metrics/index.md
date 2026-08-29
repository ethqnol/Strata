# `strata.metrics`

Regression metrics (MSE, RMSE, MAE, R²), classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, Log Loss, ROC AUC), and clustering metrics (Silhouette Score).

---

## Functions

| Function | Description |
| :--- | :--- |
| [`mean_squared_error`](mean_squared_error.md) | Compute Mean Squared Error (MSE) regression loss. |
| [`root_mean_squared_error`](root_mean_squared_error.md) | Compute Root Mean Squared Error (RMSE) regression loss. |
| [`mean_absolute_error`](mean_absolute_error.md) | Compute Mean Absolute Error (MAE) regression loss. |
| [`r2_score`](r2_score.md) | Compute $R^2$ (coefficient of determination) regression score function. |
| [`unique_labels`](unique_labels.md) | Sorted list of the distinct labels appearing in y_true or y_pred. |
| [`accuracy_score`](accuracy_score.md) | Fraction (or count, if normalize is False) of correctly classified samples. |
| [`confusion_matrix`](confusion_matrix.md) | Confusion matrix C where C[i, j] counts samples of label i predicted as label j. |
| [`precision_score`](precision_score.md) | Compute classification precision score. |
| [`recall_score`](recall_score.md) | Compute classification recall (sensitivity) score. |
| [`f1_score`](f1_score.md) | Compute classification F1 score (harmonic mean of precision and recall). |
| [`log_loss`](log_loss.md) | Compute log loss (cross-entropy loss), the negative log-likelihood of true labels. |
| [`roc_auc_score`](roc_auc_score.md) | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC). |
| [`silhouette_score`](silhouette_score.md) | Compute the mean Silhouette Coefficient of all samples. |
