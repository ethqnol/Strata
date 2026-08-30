# Contributing to Strata

This guide covers environment setup, test workflows, and coding conventions for contributing to Strata.

---

## Environment Setup

Strata requires [Pixi](https://pixi.sh/) for environment management.

### Linux / macOS

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

### Windows (WSL 2)

Mojo runs on Linux and macOS. Windows users should use WSL 2 with Ubuntu:

1. Open PowerShell and install WSL:
   ```powershell
   wsl --install
   ```
2. In Ubuntu, install base dependencies and Pixi:
   ```bash
   sudo apt update && sudo apt install -y curl git build-essential
   curl -fsSL https://pixi.sh/install.sh | bash
   source ~/.bashrc
   ```
3. Clone into your Linux home directory (avoid `/mnt/c/` for better disk performance):
   ```bash
   git clone https://github.com/ethqnol/Strata.git
   cd Strata
   pixi install
   ```

---

## Development Workflow

### Formatting

Strata uses `mojo format`. Format before committing:

```bash
# Auto-format all source and test files
pixi run format
```

For VS Code or Cursor, enable format-on-save in `.vscode/settings.json`:
```json
{
  "[mojo]": {
    "editor.defaultFormatter": "modular.mojo",
    "editor.formatOnSave": true
  }
}
```

### Running Tests

We organize tests into modular suites under `tests/`:

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
pixi run test-metrics         # Regression, classification, ranking & clustering metrics, averaging strategies
pixi run test-interop         # NumPy / SciPy conversion roundtrips
pixi run test-large           # Large matrix benchmarks & stress tests
pixi run test-core            # Error types, validation routines, base traits
pixi run test-linear-regression   # OLS solvers, dtype flexibility, degenerate designs
pixi run test-ridge               # L2 closed-form solvers, regularization paths
pixi run test-coordinate-descent  # Lasso, ElasticNet, sparsity, coordinate descent
pixi run test-logistic-regression # Binary & multinomial softmax classification
pixi run test-sgd                 # Stochastic Gradient Descent regressor & classifier
```

### Typechecking & Building

Compile all package modules to verify trait implementations and type safety:

```bash
pixi run build
```

---

## Codebase Layout

```
Strata/
├── strata/
│   ├── base/             # Core traits (Estimator, Transformer, Regressor, Classifier, Clusterer, Pipelines)
│   ├── core/             # Matrix, MatrixView, CSRMatrix, CSCMatrix, linalg, sparse_ops, dataset, interop
│   ├── exceptions/       # Domain errors (DimensionMismatchError, NotFittedError, InvalidParameterError, etc.)
│   ├── utils/            # Validation helpers (check_X_y, check_sparse), math (softmax), random (PRNG, shuffle)
│   ├── preprocessing/    # Data transformers (StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, Binarizer)
│   ├── model_selection/  # Data splitting (train_test_split)
│   ├── metrics/          # Evaluation metrics (MSE, R2, Accuracy, F1, LogLoss, ROC AUC, Silhouette)
│   ├── linear_model/     # Linear regression, Ridge, Logistic regression
│   ├── cluster/          # KMeans, KModes, DBSCAN
│   ├── tree/             # Decision trees
│   ├── ensemble/         # Random forests, gradient boosting
│   ├── decomposition/    # PCA, TruncatedSVD
│   └── neighbors/        # KNN, KD-Tree
└── tests/                # Unit and integration test suites
```

---

## Coding Conventions

### 1. Memory and Ownership
- Use `MatrixView` for non-owning, strided reads or slices to avoid intermediate matrix allocations.
- Use `^` (move operator) when transferring ownership of large arrays or fitted state into struct fields.
- Call `.copy()` explicitly when a duplicate buffer is required.

### 2. DType and Precision Policy
- **Low-level LinAlg**: Kernels (`gemm`, `spmm`, `dense_dot_vec`) are parameterized by `dtype` and operate homogeneously (`f64 x f64 -> f64`, `f32 x f32 -> f32`).
- **Estimators**: Estimators default to `compute_dtype = DType.float64`. Incoming matrices with other dtypes are converted upfront via `X.cast[Self.compute_dtype]()` at `fit` or `transform` time.

### 3. Trait Implementation
Estimators implement the relevant base trait from `strata.base`:
- `Transformer`: `fit(X)`, `transform(X) -> Matrix`, `fit_transform(X) -> Matrix`, and optional `inverse_transform(X) -> Matrix`.
- `Regressor`: `fit(X, y)`, `predict(X) -> List[Scalar]`.
- `Classifier`: `fit(X, y)`, `predict(X) -> List[Scalar]`, `predict_proba(X) -> Matrix`.
- `Clusterer`: `fit(X)`, `predict(X) -> List[Int]`, `fit_predict(X) -> List[Int]`.

### 4. Validation and Errors
- Validate input matrices at the entry point of `fit` and `transform` using `check_array` or `check_sparse`.
- Check fitted state in `transform` and `predict` using `check_is_fitted("EstimatorName", self.is_fitted)`.
- Raise specific error types from `strata.exceptions` (`DimensionMismatchError`, `NotFittedError`, `InvalidParameterError`).

---

## Pre-PR Submission Checklist

Before submitting a Pull Request, run through this step-by-step checklist to ensure CI passes cleanly on the first try.

> [!TIP]
> You can run the entire automated verification and documentation pipeline with a single command:
> ```bash
> pixi run pre-pr
> ```

### 1. Code Formatting
- [ ] Run `pixi run format-check` to verify code formatting.
- [ ] If there are formatting diffs, run `pixi run format` to auto-format all `strata/` and `tests/` files.

### 2. Precompile & Docstring Check
- [ ] Run `pixi run build` (`mojo precompile strata -o strata.mojoc`).
- [ ] Verify there are **0 compiler warnings** regarding docstrings.
  - Compile-time struct parameters (`[compute_dtype: DType]`) belong under `Parameters:`.
  - Runtime constructor parameters (`n_clusters`, `fit_intercept`) belong under `Args:`.

### 3. Run Unit Test Suites
- [ ] Run `pixi run test-runner` (or `pixi run test`) to execute unit test suites.
- [ ] Verify **100% pass rate** across all test files with zero failures.

### 4. Regenerate API Documentation & Site Bundle
- [ ] Run `pixi run generate-docs` (or `python scripts/generate_docs.py`).
- [ ] Ensure reference pages, search index (`docs/search_index.json`), and site bundle (`docs/data.js`) are updated.

### 5. Git Branch & Pull Request
- [ ] Ensure work is on a dedicated feature branch (`git checkout -b feat/your-feature-name`).
- [ ] Commit changes with concise, descriptive commit messages.
- [ ] Push branch to remote: `git push -u origin <your-branch-name>`.
- [ ] Open Pull Request against `main`.
