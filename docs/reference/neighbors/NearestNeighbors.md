# `NearestNeighbors`

**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  
**Source**: [`strata/neighbors/base.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/base.mojo)

```mojo
struct NearestNeighbors[compute_dtype: DType = DType.float64](Copyable, Movable)
```

```mojo
from strata.neighbors import NearestNeighbors
```

**Unsupervised learner for implementing neighbor searches.**

Finds the $k$-nearest neighbors or all neighbors within a given radius
using brute-force or index-backed spatial distance metrics.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Precision used for distance computations. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`n_neighbors`** | Number of neighbors to use by default for `kneighbors` queries. Default 5. |
| **`radius`** | Range of parameter space to use by default for `radius_neighbors` queries. Default 1.0. |
| **`algorithm`** | Algorithm used to compute nearest neighbors ('auto', 'brute'). Default 'auto'. |
| **`metric`** | Distance metric to use ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'. |
| **`p`** | Parameter for the Minkowski metric. Default 2.0. |

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
| [`NearestNeighbors.fit()`](#fit) | Fit the nearest neighbors estimator from the training dataset. |
| [`NearestNeighbors.kneighbors()`](#kneighbors) | Find the K-neighbors of points in X. |
| [`NearestNeighbors.radius_neighbors()`](#radius_neighbors) | Find the neighbors within a given radius of points in X. |

---

## Method Details

### `NearestNeighbors.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
```

Fit the nearest neighbors estimator from the training dataset.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

---

### `NearestNeighbors.kneighbors()`

```mojo
def kneighbors[in_dtype: DType](self, X: Matrix[in_dtype], n_neighbors: Int = -1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]
def kneighbors[in_dtype: DType = DType.float64](self, n_neighbors: Int = -1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]
```

Find the K-neighbors of points in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`n_neighbors`** | `Int` | — |

**Returns**: `Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]` — Tuple of: - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, n_neighbors). - Matrix[DType.int32]: Indices of neighbors in the training dataset with shape (n_queries, n_neighbors).

---

### `NearestNeighbors.radius_neighbors()`

```mojo
def radius_neighbors[in_dtype: DType](self, X: Matrix[in_dtype], radius: Float64 = -1.0) -> Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]
```

Find the neighbors within a given radius of points in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`radius`** | `Float64` | — |

**Returns**: `Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]` — Tuple of: - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius. - List[List[Int]]: Indices of neighbors in the training dataset.
---

## Example

```mojo
from strata.neighbors import NearestNeighbors
from strata.core import Matrix

var nn = NearestNeighbors[DType.float64](n_neighbors=2)
nn.fit(X_train)
var res = nn.kneighbors(X_test)
var distances = res[0]
var indices = res[1]
```
