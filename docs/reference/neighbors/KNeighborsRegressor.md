# `KNeighborsRegressor`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Regressor, Copyable, Movable`  
**Source**: [`strata/neighbors/regression.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/regression.mojo)

```mojo
struct KNeighborsRegressor[compute_dtype: DType = DType.float64](Regressor, Copyable, Movable)
```

```mojo
from strata.neighbors import KNeighborsRegressor
```

**Regression based on k-nearest neighbors.**

Predicts the target value for query points by local interpolation:
- **Uniform weights**:
$$
\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i
$$
- **Distance weights**:
$$
w_i = \frac{1}{d(x, x_i)}, \quad \hat{y}(x) = \frac{\sum_{i \in N_K(x)} w_i y_i}{\sum_{i \in N_K(x)} w_i}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision for distance arithmetic. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`n_neighbors`** | Number of neighbors to use for prediction. Default 5. |
| **`weights`** | Weight function used in prediction ('uniform', 'distance'). Default 'uniform'. |
| **`algorithm`** | Neighbor search algorithm ('auto', 'brute'). Default 'auto'. |
| **`metric`** | Distance metric to use ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'. |
| **`p`** | Power parameter for the Minkowski metric. Default 2.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`n_samples_fit_`** | Number of samples in the fitted data. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KNeighborsRegressor.fit()`](#fit) | Fit the k-nearest neighbors regressor from the training dataset. |
| [`KNeighborsRegressor.predict()`](#predict) | Predict the continuous target values for the provided data. |

---

## Method Details

### `KNeighborsRegressor.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the k-nearest neighbors regressor from the training dataset.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

---

### `KNeighborsRegressor.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]
```

Predict the continuous target values for the provided data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `List[Scalar[feat_dtype]]` — List[Scalar[feat_dtype]]: Predicted regression values.
---

## Example

```mojo
from strata.neighbors import KNeighborsRegressor
from strata.core import Matrix

var reg = KNeighborsRegressor[DType.float64](n_neighbors=3, weights="distance")
reg.fit(X_train, y_train)
var y_pred = reg.predict(X_test)
```
