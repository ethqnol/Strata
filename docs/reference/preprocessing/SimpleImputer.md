# `SimpleImputer`

**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  
**Source**: [`strata/preprocessing/imputer.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/imputer.mojo)

```mojo
struct SimpleImputer[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)
```

```mojo
from strata.preprocessing import SimpleImputer
```

**Univariate imputer for completing missing values with simple statistics.**

Replaces missing values (`NaN` or a specified sentinel value) using a
chosen statistical strategy along each column.
Strategies:
- `"mean"`: Replace missing values using the mean along each column.
- `"median"`: Replace missing values using the median along each column.
- `"most_frequent"`: Replace missing values using the most frequent value (mode) along each column.
- `"constant"`: Replace missing values with `fill_value`.

---

## Parameters (Compile-Time)

| Parameter | Description |
| :--- | :--- |
| **`compute_dtype`** | Computational precision data type. Default DType.float64. |

---

## Arguments (Runtime)

| Argument | Description |
| :--- | :--- |
| **`missing_values`** | The placeholder for missing values. Can be `NaN` or a scalar value. Default `NaN`. |
| **`strategy`** | The imputation strategy ('mean', 'median', 'most_frequent', 'constant'). Default 'mean'. |
| **`fill_value`** | Value used when strategy='constant'. Default 0.0. |

---

## Attributes

| Attribute | Description |
| :--- | :--- |
| **`statistics_`** | The imputation fill value for each feature column calculated during fit. |
| **`n_features_in_`** | Number of features seen during fit. |
| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |

---

## Methods Overview

| Method | Description |
| :--- | :--- |
| [`SimpleImputer.fit()`](#fit) | Fit the imputer on feature matrix X. |
| [`SimpleImputer.transform()`](#transform) | Impute all missing values in X. |
| [`SimpleImputer.fit_transform()`](#fit_transform) | Fit to data, then transform it. |

---

## Method Details

### `SimpleImputer.fit()`

```mojo
def fit[in_dtype: DType](mut self, X: Matrix[in_dtype])
def fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])
```

Fit the imputer on feature matrix X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |

---

### `SimpleImputer.transform()`

```mojo
def transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```

Impute all missing values in X.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Matrix with missing values imputed.

---

### `SimpleImputer.fit_transform()`

```mojo
def fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]
def fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]
```

Fit to data, then transform it.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`X`** | `Matrix[in_dtype]` | Feature matrix. |
| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |

**Returns**: `Matrix[in_dtype]` — Matrix[in_dtype]: Matrix with missing values imputed.
---

## Example

```mojo
from strata.preprocessing import SimpleImputer
from strata.core import Matrix

var imputer = SimpleImputer[DType.float64](strategy="mean")
imputer.fit(X_train)
var X_imputed = imputer.transform(X_train)
```
