# Contributors & Development Guide

## Getting Started

Strata is built entirely in [Mojo](https://docs.modular.com/mojo/) and uses [Pixi](https://pixi.sh/) for reproducible environment and dependency management.

### 1. Environment Setup

#### Linux / macOS
Make sure you have [Pixi installed](https://pixi.sh/#installation), then clone and initialize the environment:

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata

# Install dependencies (Mojo, Python, NumPy, SciPy)
pixi install
```

#### Windows (via WSL 2)
Mojo runs natively on Linux (and macOS), so Windows contributors should use **WSL 2** (Windows Subsystem for Linux):

1. **Install WSL 2** (if not already installed):
   Open PowerShell as Administrator and run:
   ```powershell
   wsl --install
   ```
   Restart your PC if prompted, then launch the **Ubuntu** terminal from your Start menu.

2. **Install base packages in Ubuntu**:
   ```bash
   sudo apt update && sudo apt install -y curl git build-essential
   ```

3. **Install Pixi inside WSL**:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   source ~/.bashrc
   ```

4. **Clone and install (inside the Linux filesystem)**:
   > **Note**: Always clone the repository into your WSL Linux home directory (`~/Code/` or `/home/<user>/`), **not** on the Windows mount (`/mnt/c/...`), for optimal disk I/O and compiler performance.

   ```bash
   mkdir -p ~/Code && cd ~/Code
   git clone https://github.com/ethqnol/Strata.git
   cd Strata
   pixi install
   ```

5. **VS Code / Cursor Integration**:
   - Install the **WSL** extension in VS Code / Cursor.
   - Run `code .` from the `Strata` directory inside your WSL terminal.
   - Install the official **Mojo** extension inside the remote WSL session.

---

### 2. Code Formatting

We use Mojo's built-in official formatter (`mojo format`). Since it is bundled with the compiler in our Pixi environment, no extra installations are required.

```bash
# Auto-format all source and test files
pixi run format

# Check formatting without modifying files
pixi run format-check
```

**Editor Configuration (VS Code / Cursor):**
To format automatically on save, add this to your `settings.json`:
```json
"[mojo]": {
    "editor.defaultFormatter": "modular.mojo",
    "editor.formatOnSave": true
}
```

---

### 3. Running Tests

We keep our test suites modular. You can run all tests or target specific subsystems:

```bash
# Run the full test suite (16 test suites)
pixi run test-all

# Run specific test modules
pixi run test-matrix          # Dense GEMM, dot_vec, transpose, cast, eye, axes
pixi run test-linalg          # LAPACK SVD, Eigh, QR, Cholesky, Least-Squares, Solve, Inv, Norm, Matrix + - *
pixi run test-sparse          # CSR/CSC, SpMV, SpVM, SpMM, SpGEMM, SDDMM, check_sparse
pixi run test-math            # Numerically stable sigmoid, softmax, log_sum_exp, PRNG
pixi run test-dataset         # Dataset containers, splitting invariants, edge cases
pixi run test-pipelines       # Composable N-step transformer, regressor, classifier pipelines
pixi run test-view            # Strided MatrixView, 2D zero-copy slicing, bounds checks
pixi run test-preprocessing   # StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, Binarizer
pixi run test-split           # Train/test split, shuffling reproducibility
pixi run test-metrics         # Regression & classification metrics, averaging strategies
pixi run test-interop         # NumPy / SciPy conversion roundtrips
pixi run test-large           # Large matrix benchmarks & stress tests
pixi run test-core            # Error types, validation routines, base traits
pixi run test-linear-regression   # OLS solvers, dtype flexibility, degenerate designs
pixi run test-ridge               # L2 closed-form solvers, regularization paths
pixi run test-logistic-regression # Binary & multinomial softmax classification
```

---

### 4. Precompiling the Package

To ensure all generic structs, traits, and module interfaces typecheck and compile cleanly:

```bash
pixi run build
```

---

## Codebase Architecture

Here is how the project is organized:

```
Strata/
├── strata/
│   ├── base/             # Core traits (Estimator, Transformer, Regressor, Classifier, Clusterer, Pipelines)
│   ├── core/             # Matrix, MatrixView, CSRMatrix, CSCMatrix, linalg, sparse_ops, dataset, interop
│   ├── exceptions/       # Domain errors (DimensionMismatchError, NotFittedError, InvalidParameterError, etc.)
│   ├── utils/            # Validation helpers (check_X_y, check_sparse), math (softmax), random (PRNG, shuffle)
│   ├── preprocessing/    # Data transformers (StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, Binarizer)
│   ├── model_selection/  # Data splitting (train_test_split)
│   ├── metrics/          # Evaluation metrics (MSE, R2, Accuracy, F1)
│   ├── linear_model/     # Linear regression, Ridge, Logistic regression
│   ├── cluster/          # KMeans, KModes, DBSCAN
│   ├── tree/             # Decision trees
│   ├── ensemble/         # Random forests, gradient boosting
│   ├── decomposition/    # PCA, TruncatedSVD
│   └── neighbors/        # KNN, KD-Tree
└── tests/                # Unit and integration test suites
```

---

## Core Design Principles & Mojo Idioms

When contributing code to Strata, please keep these conventions in mind:

### 1. Ownership & Memory Efficiency
- **Avoid unnecessary copies**: For internal matrix operations and read-only slicing, use `MatrixView` rather than allocating new `Matrix` buffers.
- **Explicit transfers**: Use `^` (move operator) when transferring ownership of large arrays or structs into estimators or return values.
- **Explicit copying**: When an explicit clone is needed, call `.copy()`.

### 2. Multi-Precision & 3-Tiered Type Policy

Strata follows a clear, 3-tiered type policy across the stack:

| Layer | Type Policy | Examples & Constraints |
| :--- | :--- | :--- |
| **Low-Level LinAlg** | Generic numeric types (including `int32`, `int64`, `float32`, `float64`, `bfloat16`, `float16`) via compile-time parameters for general-purpose utility. Operations are strictly homogeneous ($A \times B \rightarrow C$). | `gemm`, `dense_dot_vec`, `transpose`, `spmv`, `spmm`, `sddmm`, `spgemm` |
| **Advanced LinAlg** | Constrained to floating-point types (`constrained[dtype.is_floating_point(), "Floating-point type required"]()`). | `solve`, `inv`, `svd`, `qr`, `norm`, `cholesky` |
| **ML Estimators** | Automatically cast incoming integer arrays to `float32` or `float64` immediately upon entry via `X.cast[compute_dtype]()`. | `LinearRegression`, `Ridge`, `LogisticRegression`, `PCA`, `KMeans` |

- **Strictly Homogeneous Execution**: Low-level kernels never mix precision inside inner loops (e.g. `f32 × f32 → f32`, `f64 × f64 → f64`, `i32 × i32 → i32`).
- **Promote Immediately Once**: Estimators accept flexible inputs at `fit` time and promote to `compute_dtype` upfront once, ensuring all downstream iterative optimizations run on pure floating-point tensors at hardware peak.

### 3. Trait Contracts & Composable Pipelines
To ensure compile-time trait enforcement, type safety, and effortless `model.predict(X)` call syntax matching scikit-learn:

- **Streamlined Trait Design (The 2-Method Pattern)**:
  Estimators conform to `Movable` and the appropriate base trait (`Regressor`, `Classifier`, `Transformer`, or `Clusterer`). Estimators only need to implement the core array methods (`fit(X, y)` and `predict(X)` / `transform(X)` / `predict_proba(X)`). `Dataset` support is handled automatically via generic functional helpers or optional convenience methods:
  ```mojo
  struct LinearRegression[
      compute_dtype: DType = DType.float64,
  ](Regressor, Movable):
      ...
  ```

- **Transformers & Fitted DType Consistency**:
  Transformers (e.g. `StandardScaler`) implement `fit`, `transform`, and `fit_transform`. They accept arbitrary input precisions at `fit` time, enforce fitted `fit_dtype` consistency at `transform` time, and compute in `compute_dtype` (default `DType.float64`) for maximum numerical stability:
  ```mojo
  struct StandardScaler[
      compute_dtype: DType = DType.float64,
  ](Transformer, Movable, Copyable):
      ...
  ```

- **Clusterers & Discrete Cluster Assignments**:
  Clusterers (e.g. `KMeans`, `KModes`) implement generic `fit[dtype]`, `predict[dtype]`, and `fit_predict[dtype]` methods returning `List[Int]` cluster indices.

- **Clean, Composable N-Step Pipelines**:
  `PipelineTransformer[T1, T2]` chains transformers together, and `PipelineRegressor` / `PipelineClassifier` bind transformers to models with zero call-site type parameters:
  ```mojo
  var scaler = StandardScaler()
  var pca = PCA(n_components=5)
  var prep = PipelineTransformer(scaler^, pca^)
  var model = LinearRegression()
  var pipe = PipelineRegressor(prep^, model^)

  pipe.fit(X_train, y_train)
  var preds = pipe.predict(X_test)  # Fully inferred from X_test!
  ```

### 4. FFI for Heavy LAPACK Factorizations
When implementing algorithms requiring complex matrix factorizations (such as full `PCA`, `TruncatedSVD`, or exact `LinearRegression` via QR/SVD in `strata/decomposition/` and `strata/linear_model/`):
- **Use `sys.ffi.DLHandle`**: Bind dynamically to system OpenBLAS or LAPACK routines (e.g. `dgesdd`, `dgelss`, `dgeqrf`, `dorgqr`, `sgesdd`, `sgelss`) for robust $O(N^3)$ factorizations.
- **Direct Pointer Passing**: Pass the internal `self.data.unsafe_ptr()` from the Mojo `Matrix` directly across the C ABI boundary without allocating intermediate copies.
- **Wrap Outputs into Mojo Matrix**: Wrap the resulting singular values, eigenvectors, or solution vectors directly back into owned `Matrix[compute_dtype]` structs.

### 5. Validation & Clear Error Messages
- Use the shared validation functions in `strata.utils.validation`:
  - `check_array[dtype](X)` — checks for non-empty 2D matrices.
  - `check_X_y(X, y)` — verifies consistent sample counts between features and targets.
  - `check_sparse[dtype](rows, cols, data, indices, indptr, is_csr)` — validates sparse matrix structure and index invariants.
  - `check_is_fitted("EstimatorName", self.is_fitted)` — ensures models are trained before calling `predict` or `transform`.
- Raise domain-specific errors from `strata.exceptions.errors` (`DimensionMismatchError`, `InvalidParameterError`, `NotFittedError`).

---

## How to Add a New Estimator (Step-by-Step)

If you're implementing an estimator (e.g. from ROADMAP.md):

### Step 1: Define the Struct & Constructor
Implement the estimator conforming to `Movable` and the appropriate base trait (`Regressor`, `Classifier`, `Transformer`, or `Clusterer`), storing internal model parameters in `compute_dtype`:

```mojo
from ..base.estimator import Regressor
from ..core.matrix import Matrix
from ..utils.validation import check_is_fitted, check_X_y
from ..exceptions.errors import NotFittedError

struct MyRegressor[
    compute_dtype: DType = DType.float64,
](Regressor, Movable):
    var is_fitted: Bool
    var coef_: List[Scalar[Self.compute_dtype]]      # Accumulates in compute precision!
    var intercept_: Scalar[Self.compute_dtype]

    def __init__(out self):
        self.is_fitted = False
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0
```

### Step 2: Implement `fit` and `predict` (The 2-Method Pattern)
Implement `fit` and `predict` to satisfy trait conformance with zero call-site type parameters:

```mojo
    # 1. Fit accepting arbitrary input feature and target precisions
    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        check_X_y(X, y)
        # Compute coefficients using Self.compute_dtype arithmetic
        self.is_fitted = True

    # 2. Predict returning predictions matching input feature precision
    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        check_is_fitted("MyRegressor", self.is_fitted)
        var preds = List[Scalar[feat_dtype]](capacity=X.rows)
        # Compute predictions
        return preds^
```

### Step 3: Export in Subpackage `__init__.mojo`
Export your struct in its folder's `__init__.mojo` (e.g., `strata/linear_model/__init__.mojo`) and add it to `strata/__init__.mojo`.

### Step 4: Write Unit Tests & Validate Parity
Create `tests/test_<feature>.mojo`. Where applicable, compare outputs against `scikit-learn` references to ensure numerical correctness.

---

## Pull Request Checklist

Before submitting a PR:

1. [ ] **Format code**: Run `pixi run format` to ensure code conforms to Mojo styling.
2. [ ] **Tests pass**: Run `pixi run test-all` and ensure all tests pass without errors.
3. [ ] **Package builds**: Run `pixi run build` to confirm package precompilation succeeds.
4. [ ] **Docstrings**: Ensure new public structs and methods have clear docstrings.
5. [ ] **Update Roadmap**: If completing an item from [ROADMAP.md](./ROADMAP.md), check off the task!
