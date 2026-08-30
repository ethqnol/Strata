# `ElasticNet`

**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  
**Source**: [`strata/linear_model/elastic_net.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/elastic_net.mojo)

```mojo
struct ElasticNet[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)
```

```mojo
from strata.linear_model import ElasticNet
```

**Linear regression with combined L1 and L2 regularization (ElasticNet).**

Minimizes the penalized least-squares objective function using coordinate descent:
$$
\min_{w, b} \frac{1}{2N} \|y - (Xw + b)\|_2^2 + \alpha \cdot \rho \|w\|_1 + \frac{\alpha (1 - \rho)}{2} \|w\|_2^2
$$
where $\rho \in [0, 1]$ corresponds to `l1_ratio`.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`alpha`** | Regularization constant multiplier ($\alpha \ge 0$). Default 1.0. |
| **`l1_ratio`** | ElasticNet mixing parameter $\rho \in [0, 1]$. For `l1_ratio = 1.0` the penalty is L1 (Lasso); for `l1_ratio = 0.0` the penalty is L2 (Ridge). Default 0.5. |
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
| [`ElasticNet.fit()`](#fit) | Fit the ElasticNet linear model via coordinate descent. |
| [`ElasticNet.predict()`](#predict) | Predict continuous target values using the fitted ElasticNet model. |

---

## Method Details

### `ElasticNet.fit()`

```mojo
def fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the ElasticNet linear model via coordinate descent.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |

---

### `ElasticNet.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Predict continuous target values using the fitted ElasticNet model.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted target vector of length $N$.
---

## Example

```mojo
from strata.linear_model import ElasticNet
from strata.core import Matrix

var reg = ElasticNet[DType.float64](alpha=0.1, l1_ratio=0.7)
reg.fit(X_train, y_train)
var preds = reg.predict(X_test)
```
