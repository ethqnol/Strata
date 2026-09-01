# Getting Started with Strata

Strata is a native machine learning and linear algebra library for Mojo. It provides scikit-learn compatible estimator APIs, dense and sparse matrix formats, and SIMD and LAPACK acceleration.

---

## 1. Installation

### Adding to an Existing Pixi Project

Install Strata from the official Modular Community channel on Prefix.dev:

```bash
pixi add strata --channel https://repo.prefix.dev/modular-community
```

Or add it directly to `pixi.toml`:

```toml
[workspace]
channels = [
    "https://repo.prefix.dev/modular-community",
    "https://conda.modular.com/max",
    "conda-forge"
]

[dependencies]
strata = ">=0.1.0"
mojo = ">=1.0.0"

# Task shortcut to automatically link LAPACK shared libraries
[tasks]
start = "mojo run -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack main.mojo"
```

You can also develop Strata locally:
```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

> [!IMPORTANT]
> **Linking LAPACK**: `pixi add strata` automatically installs `liblapack` into `$CONDA_PREFIX/lib/`. When calling LAPACK-backed routines (`svd`, `qr`, `inv`, `eigh`, `PCA`, or `LinearRegression` with Cholesky/QR solvers), pass `-Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack` to `mojo run`, or define a task in `pixi.toml` as shown (i.e. `pixi run start`...). See the section underneath for more details




---

## 2. Execution and LAPACK Configuration

Strata routines fall into two execution profiles:

1. **Pure Mojo Routines**: Decision Trees, Random Forests, Gradient Boosting, SGD, Coordinate Descent (Lasso/ElasticNet), Scalers, Encoders, ColumnTransformer, and Serialization. Run directly with `mojo run main.mojo`.
2. **LAPACK-Backed Solvers**: Linear algebraic decompositions (`svd`, `qr`, `inv`, `eigh`, `solve`, `lstsq`), `PCA`, and Cholesky/QR regression solvers. Require linker flags pointing to the Pixi-installed `liblapack.so`.

### Recommended: Set Up a Pixi Task

To avoid passing linker flags manually on every run, add a `start` task to your project's `pixi.toml`:

```toml
[tasks]
start = "mojo run -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack main.mojo"
```

Run your application with:

```bash
pixi run start
```

### Algorithm Compatibility and Execution Matrix

| Category | Algorithm / Routine | Requires Linker Flags | Command |
| :--- | :--- | :---: | :--- |
| **Ensemble & Trees** | `RandomForestClassifier`, `RandomForestRegressor` | No | `mojo run main.mojo` |
| | `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` | No | `mojo run main.mojo` |
| | `DecisionTreeClassifier`, `DecisionTreeRegressor` | No | `mojo run main.mojo` |
| **Linear Models** | `LinearRegression(solver="normal")` | No | `mojo run main.mojo` |
| | `Lasso`, `ElasticNet` (Coordinate Descent) | No | `mojo run main.mojo` |
| | `SGDClassifier`, `SGDRegressor` | No | `mojo run main.mojo` |
| | `LogisticRegression` (L-BFGS / GD) | No | `mojo run main.mojo` |
| | `LinearRegression(solver="cholesky" / "qr" / "svd")` | **Yes** | `pixi run start` |
| | `Ridge(solver="cholesky" / "qr" / "svd")` | **Yes** | `pixi run start` |
| **Decomposition** | `PCA`, `TruncatedSVD` | **Yes** | `pixi run start` |
| **Clustering & Neighbors** | `KMeans`, `MiniBatchKMeans` | No | `mojo run main.mojo` |
| | `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` | No | `mojo run main.mojo` |
| **Preprocessing** | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `Binarizer` | No | `mojo run main.mojo` |
| | `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `SimpleImputer` | No | `mojo run main.mojo` |
| | `PolynomialFeatures`, `ColumnTransformer`, `Pipeline` | No | `mojo run main.mojo` |
| **Model Selection** | `KFold`, `StratifiedKFold`, `ShuffleSplit`, `GridSearchCV` | No | `mojo run main.mojo` |
| **Persistence** | `dump`, `load`, `dumps`, `loads` | No | `mojo run main.mojo` |
| **Core Linear Algebra** | Matrix Addition, GEMM, Dot Products | No | `mojo run main.mojo` |
| | `svd`, `qr`, `inv`, `eigh`, `cholesky`, `solve`, `lstsq` | **Yes** | `pixi run start` |

---

## 3. Code Examples

### Example A: Tree-Based Model (Pure Mojo)

Create `rf_example.mojo`:

```mojo
from strata import Matrix, RandomForestClassifier, accuracy_score

def main() raises:
    var X = Matrix[DType.float64](8, 2)
    X[0, 0] = -3.0; X[0, 1] = -2.0
    X[1, 0] = -2.0; X[1, 1] = -3.0
    X[2, 0] = -4.0; X[2, 1] = -2.5
    X[3, 0] = -2.5; X[3, 1] = -4.0
    X[4, 0] =  3.0; X[4, 1] =  2.0
    X[5, 0] =  2.0; X[5, 1] =  3.0
    X[6, 0] =  4.0; X[6, 1] =  2.5
    X[7, 0] =  2.5; X[7, 1] =  4.0

    var y: List[Scalar[DType.int32]] = [0, 0, 0, 0, 1, 1, 1, 1]

    var rf = RandomForestClassifier[DType.float64](n_estimators=20, max_depth=4, random_state=42)
    rf.fit(X, y)

    var preds = rf.predict(X)
    print("Training Accuracy:", accuracy_score(y, preds))
```

Run directly:
```bash
mojo run rf_example.mojo
```

---

### Example B: Linear Model & PCA (LAPACK Solvers)

Create `lapack_example.mojo`:

```mojo
from strata import Matrix, LinearRegression, PCA, dump, load

def main() raises:
    var X = Matrix[DType.float64](4, 2)
    X[0, 0] = 1.0; X[0, 1] = 1.0
    X[1, 0] = 1.0; X[1, 1] = 2.0
    X[2, 0] = 2.0; X[2, 1] = 2.0
    X[3, 0] = 2.0; X[3, 1] = 3.0
    var y: List[Scalar[DType.float64]] = [6.0, 8.0, 9.0, 11.0]

    # 1. Cholesky-accelerated Linear Regression
    var reg = LinearRegression(solver="cholesky")
    reg.fit(X, y)
    print("Fitted Intercept:", reg.intercept_)
    print("Fitted Coefficients:", reg.coef_[0], reg.coef_[1])

    # 2. SVD-accelerated Principal Component Analysis
    var pca = PCA(n_components=1)
    var X_trans = pca.fit_transform(X)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_[0])

    # 3. Model Serialization
    dump(reg, "model.strata")
    var restored = load[LinearRegression[]]("model.strata")
    var preds = restored.predict(X)
    print("Restored Model Predictions:", preds[0], preds[1], preds[2], preds[3])
```

Run with linker flags:
```bash
pixi run mojo -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack lapack_example.mojo
```
Or via your Pixi task:
```bash
pixi run start
```

---

## 4. Key Concepts

- **`Matrix[dtype]`**: Contiguous 2D dense matrix format with native SIMD layouts.
- **`fit(X, y)` and `predict(X)`**: Standard estimator lifecycle methods across all models.
- **`ColumnTransformer` & `Pipeline`**: Composable preprocessor chains with compile-time type safety.
- **`dump` / `load`**: Zero-copy binary serialization format.

For multi-stage preprocessing pipelines, see [Composing End-to-End ML Pipelines](end_to_end_pipeline.md).
