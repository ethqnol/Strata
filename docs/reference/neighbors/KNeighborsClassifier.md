# `KNeighborsClassifier`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  
**Source**: [`strata/neighbors/classification.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/classification.mojo)

```mojo
struct KNeighborsClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)
```

```mojo
from strata.neighbors import KNeighborsClassifier
```

**Classifier implementing the k-nearest neighbors vote.**

Supports uniform voting and inverse-distance weighted voting:
- **Uniform weights**:
$$
P(y = c \mid x) = \frac{1}{K} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)
$$
- **Distance weights**:
$$
w_i = \frac{1}{d(x, x_i)}, \quad P(y = c \mid x) = \frac{\sum_{i \in N_K(x)} w_i \mathbb{I}(y_i = c)}{\sum_{i \in N_K(x)} w_i}
$$

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision used for internal distance calculations. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`n_neighbors`** | Number of neighbors to use for queries. Default 5. |
| **`weights`** | Weight function used in prediction ('uniform', 'distance'). Default 'uniform'. |
| **`algorithm`** | Algorithm used to compute nearest neighbors ('auto', 'brute'). Default 'auto'. |
| **`metric`** | Distance metric to use. Default 'euclidean'. |
| **`p`** | Power parameter for the Minkowski metric. Default 2.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`classes_`** | Distinct class labels matrix of shape (n_classes,). |
| **`n_classes_`** | Number of distinct classes seen during fit. |
| **`n_samples_fit_`** | Number of samples in the fitted data. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`KNeighborsClassifier.fit()`](#fit) | Fit the k-nearest neighbors classifier from the training dataset. |
| [`KNeighborsClassifier.predict_proba()`](#predict_proba) | Return probability estimates for the test data X. |
| [`KNeighborsClassifier.predict()`](#predict) | Predict the class labels for the provided data. |

---

## Method Details

### `KNeighborsClassifier.fit()`

```mojo
def fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the k-nearest neighbors classifier from the training dataset.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

---

### `KNeighborsClassifier.predict_proba()`

```mojo
def predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]
def predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[ feat_dtype ]
```

Return probability estimates for the test data X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `Matrix[feat_dtype]` — Matrix[feat_dtype]: Probabilities of shape (n_queries, n_classes).

---

### `KNeighborsClassifier.predict()`

```mojo
def predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]
def predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]
```

Predict the class labels for the provided data.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |

**Returns**: `List[Int]` — List[Int]: Predicted class label for each sample.
---

## Example

```mojo
from strata.neighbors import KNeighborsClassifier
from strata.core import Matrix

var clf = KNeighborsClassifier[DType.float64](n_neighbors=3, weights="distance")
clf.fit(X_train, y_train)
var y_pred = clf.predict(X_test)
var proba = clf.predict_proba(X_test)
```
