# `Lasso`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/linear_model/lasso.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/lasso.mojo)

```mojo
struct Lasso[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.linear_model import Lasso
```

**Lasso linear model with L1 regularization.**

Minimizes the penalized least-squares objective function using coordinate descent:
$$
\min_{w, b} \frac{1}{2N} \|y - (Xw + b)\|_2^2 + \alpha \|w\|_1
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`alpha`** | Regularization strength ($\alpha \ge 0$). Higher values encourage sparser solutions. Default 1.0. |
| **`fit_intercept`** | Whether to calculate the independent intercept bias term. Default True. |
| **`max_iter`** | Maximum number of coordinate descent iterations. Default 1000. |
| **`tol`** | Convergence tolerance threshold for maximum coefficient update. Default 1e-4. |
| **`positive`** | When set to True, forces coefficients to be non-negative. Default False. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`coef_`** | Weight vector coefficients of length $D$. |
| **`intercept_`** | Independent bias intercept term. |
| **`n_iter_`** | Actual number of coordinate descent iterations run. |
| **`is_fitted`** | Boolean flag indicating if the model has been fitted. |
| **`n_features_in_`** | Number of input features observed during fitting. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`Lasso.fit()`](#fit) | Fit the Lasso linear model via coordinate descent. |
| [`Lasso.predict()`](#predict) | Predict continuous target values using the fitted Lasso model. |

---

## Method Details

### `Lasso.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the Lasso linear model via coordinate descent.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `Lasso.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Predict continuous target values using the fitted Lasso model.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target vector of length $N$.
---

## Example

```mojo
from strata.linear_model import Lasso
from strata.core import Matrix

var reg = Lasso[DType.float64](alpha=0.1, max_iter=1000)
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
