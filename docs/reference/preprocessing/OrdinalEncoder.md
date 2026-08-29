# `OrdinalEncoder`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/encoders.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/encoders.mojo)

```mojo
struct OrdinalEncoder[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import OrdinalEncoder
```

**Encode categorical features as an integer array.**

The input to this transformer should be a 2D matrix of integer or float
categorical features. The features are converted to ordinal integers
($0, 1, \dots, K_c - 1$) corresponding to the sorted order of unique
categories per feature.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`handle_unknown`** | When set to 'error' an error is raised on unknown categories. When set to 'use_encoded_value', unknown categories are set to unknown_value. Default 'error'. |
| **`unknown_value`** | Value used when handle_unknown='use_encoded_value'. Default -1.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`categories_`** | Categories of each feature determined during fitting. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`OrdinalEncoder.fit()`](#fit) | Fit the OrdinalEncoder on feature matrix X. |
| [`OrdinalEncoder.transform()`](#transform) | Transform X to ordinal integer codes. |
| [`OrdinalEncoder.fit_transform()`](#fit_transform) | Fit to data, then transform it. |
| [`OrdinalEncoder.inverse_transform()`](#inverse_transform) | Convert the ordinal codes back to original category values. |

---

## Method Details

### `OrdinalEncoder.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the OrdinalEncoder on feature matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

---

### `OrdinalEncoder.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```

Transform X to ordinal integer codes.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Encoded ordinal matrix.

---

### `OrdinalEncoder.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```

Fit to data, then transform it.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Encoded ordinal matrix.

---

### `OrdinalEncoder.inverse_transform()`

```mojo
def inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
```

Convert the ordinal codes back to original category values.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Decoded matrix containing reconstructed categories.
---

## Example

```mojo
from strata.preprocessing import OrdinalEncoder
from strata.core import Matrix

var encoder = OrdinalEncoder[DType.float64]()
encoder.fit(X_cat)
var X_ord = encoder.transform(X_cat)
```
