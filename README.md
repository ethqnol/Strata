[![CodeQL](https://github.com/ethqnol/Strata/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ethqnol/Strata/actions/workflows/github-code-scanning/codeql) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Strata

Strata is a native machine learning and linear algebra library written in [Mojo](https://docs.modular.com/mojo/). It provides scikit-learn-compatible estimators, dense and sparse matrix containers, and composable pipelines with native SIMD and LAPACK acceleration.

---

## Installation

Strata uses [Pixi](https://pixi.sh/) for environment and dependency management.

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

---

## Quick Start

### Linear Algebra & Matrix Operations

```mojo
from strata import Matrix, gemm, svd, eigh, solve

def main() raises:
    # Dense matrices
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 4.0; A[0, 1] = 2.0
    A[1, 0] = 2.0; A[1, 1] = 3.0

    var B = Matrix[DType.float64].eye(2)

    # Matrix multiplication
    var C = gemm(A, B)

    # Factorizations (LAPACK)
    var svd_res = svd(A)
    var eig_res = eigh(A)

    # Linear system solve: A * x = b
    var b: List[Scalar[DType.float64]] = [8.0, 7.0]
    var x = solve(A, b)
```

### Pipelines & Estimators

```mojo
from strata import (
    Matrix,
    StandardScaler,
    PCA,
    LinearRegression,
    PipelineTransformer,
    PipelineRegressor,
)

def main() raises:
    var X = Matrix[DType.float64](5, 3, 0)
    for r in range(5):
        X[r, 0] = Float64(r + 1)
        X[r, 1] = Float64((r + 1) * 2)
        X[r, 2] = Float64((r + 1) * 3)

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0]

    # Build a StandardScaler -> PCA -> LinearRegression pipeline
    var scaler = StandardScaler()
    var pca = PCA(n_components=2)
    var prep = PipelineTransformer((scaler^, pca^))
    var reg = LinearRegression(solver="cholesky")
    var pipe = PipelineRegressor(prep^, reg^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
```

### Clustering

```mojo
from strata import Matrix, KMeans, MiniBatchKMeans

def main() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    # Cluster samples into K groups using Lloyd's algorithm
    var kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42)
    kmeans.fit(X)

    var labels = kmeans.predict(X)
    var dists = kmeans.transform(X)

    # Online streaming mini-batch clustering
    var mbk = MiniBatchKMeans(n_clusters=2, batch_size=32, random_state=42)
    mbk.partial_fit(X)
```

### Model Persistence & Serialization

```mojo
from strata import dump, load, LinearRegression, Matrix

def main() raises:
    var reg = LinearRegression(solver="cholesky")
    reg.fit(X_train, y_train)

    # Save fitted model to disk
    dump(reg, "model.strata")

    # Load model back with exact type inference and fitted state intact
    var loaded_reg = load[LinearRegression]("model.strata")
    var preds = loaded_reg.predict(X_test)
```

---

## Implemented Modules

- **`strata.core`**:
  - `Matrix[dtype]`: Dense 2D row-major matrix.
  - `MatrixView[dtype, origin]`: Zero-copy strided 2D view.
  - `CSRMatrix[dtype]`, `CSCMatrix[dtype]`: Compressed sparse row/column matrices with `spmv`, `spmm`, `spgemm`, `sddmm`.
  - `linalg`: SIMD `gemm`, `dense_dot_vec`, and LAPACK bindings (`svd`, `eigh`, `qr`, `cholesky`, `lstsq`, `solve`, `inv`).
  - `dataset`: `Dataset` container for features, targets, and metadata.
  - `interop`: NumPy and SciPy sparse conversions.
- **`strata.linear_model`**: `LinearRegression`, `Ridge`, `Lasso` (coordinate descent), `ElasticNet`, `LogisticRegression` (binary and multinomial), `SGDRegressor`, `SGDClassifier`.
- **`strata.tree`**: `DecisionTreeClassifier`, `DecisionTreeRegressor` (with Gini, Entropy, MSE, MAE criteria).
- **`strata.ensemble`**: `RandomForestClassifier`, `RandomForestRegressor` (with OOB evaluation and soft voting), `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` (with UInt8 binning, histogram subtraction, and early stopping).
- **`strata.compose`**: `ColumnTransformer` (heterogeneous feature transformers applied to designated subsets of columns with configurable remainder routing).
- **`strata.io`**: `BufferWriter`, `BufferReader`, `Serializable`, `dump`, `load`, `dumps`, `loads` (zero-copy, endian-safe binary model persistence).
- **`strata.neighbors`**: `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor`, `KDTree`, and distance metrics (`euclidean`, `manhattan`, `chebyshev`, `minkowski`, `cosine`, `pairwise_distances`).
- **`strata.decomposition`**: `PCA` (with whitening and sign-flip), `TruncatedSVD` (dense and sparse CSR via SpMM).
- **`strata.cluster`**: `KMeans` (k-means++, Lloyd's algorithm, distance-space transforms), `MiniBatchKMeans` (streaming online updates, EWMA inertia smoothing, `partial_fit`).
- **`strata.preprocessing`**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Normalizer`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `SimpleImputer`, `Binarizer`, `PolynomialFeatures`.
- **`strata.model_selection`**: `train_test_split`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `ShuffleSplit`, `StratifiedShuffleSplit`, `cross_val_score`, `cross_val_predict`, `cross_validate`, `GridSearchRegressor`, `GridSearchClassifier`, `RandomizedSearchRegressor`, `RandomizedSearchClassifier`.
- **`strata.metrics`**:
  - Regression: `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`.
  - Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_auc_score`, `log_loss`.
  - Clustering: `silhouette_score`.
- **`strata.base`**: `Transformer`, `Regressor`, `Classifier`, `Clusterer`, and composable `Pipeline` structs.

---

## Development

```bash
# Run test suite
pixi run test-all

# Format code
pixi run format

# Compile package
pixi run build
```

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for development setup and codebase conventions.  
See [ROADMAP.md](./ROADMAP.md) for planned features and progress.

