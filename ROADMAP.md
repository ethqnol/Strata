# Strata Roadmap

Strata is a native machine learning library for Mojo designed around familiar scikit-learn APIs, explicit memory control, and hardware-accelerated linear algebra kernels.

---

## Current Status

- **Data Structures & Memory**:
  - `Matrix[dtype]` (Dense 2D row-major matrix with SIMD alignment and strided views)
  - `MatrixView[dtype, origin]` (Non-owning strided 2D view with compile-time origin tracking)
  - `CSRMatrix[dtype]` & `CSCMatrix[dtype]` (Compressed sparse formats with $O(\text{nnz})$ conversions)
  - `Dataset[feat_dtype, target_dtype]` (Container pairing feature matrices with targets and feature names)
- **Linear Algebra & Kernels**:
  - Homogeneous SIMD execution across `Float64`, `Float32`, `BFloat16`, and `Float16`
  - Dense: `gemm`, `dense_dot_vec`
  - Sparse: `spmv`, `spvm`, `spmm`, Gustavson `spgemm`, and `sddmm`
  - LAPACK FFI bindings for exact SVD (`dgesdd`), Eigh (`dsyevd`), QR (`dgeqrf`/`dorgqr`), Cholesky (`dpotrf`), Least-Squares (`dgelss`), LU Solve (`dgesv`), and Inversion (`dgetrf`/`dgetri`)
- **Pipelines & Composition**:
  - Unified traits: `Transformer`, `Regressor`, `Classifier`, `Clusterer`
  - Composable pipelines: `PipelineTransformer`, `PipelineRegressor`, `PipelineClassifier`
- **Validation & Math**:
  - Numerically stable `softmax`, `log_sum_exp`, `sigmoid`
  - 64-bit SplitMix64 `PRNG` with unbiased Lemire rejection sampling, Fisher-Yates `permutation` & `shuffle`
  - Domain exceptions: `DimensionMismatchError`, `NotFittedError`, `InvalidParameterError`, `DataConversionError`
- **Tree-Based & Ensemble Models**:
  - `DecisionTreeClassifier` (Gini impurity, Shannon entropy, log-loss, discrete probability matrices)
  - `DecisionTreeRegressor` (Squared error MSE, Friedman MSE, sample median MAE)
  - `RandomForestClassifier` (Bootstrap aggregation, soft-voting probability averaging, batched OOB accuracy)
  - `RandomForestRegressor` (Bootstrap aggregation, mean prediction averaging, batched OOB R² score)
  - Contiguous flat-array `Tree` / `Node` buffers with $O(1)$ streaming histogram split calculations
- **Evaluation Metrics**:
  - Regression: `mean_squared_error` / `root_mean_squared_error`, `mean_absolute_error`, `r2_score`
  - Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score` (binary, micro, macro, weighted), `confusion_matrix`, `log_loss`
  - Ranking & clustering: rank-based `roc_auc_score` with tie correction, `silhouette_score`
- **Test Coverage**:
  - 22 modular test suites (880+ passing tests)

---

## Feature Roadmap

### Core Architecture & Composition
- [x] Dense Matrix (`Matrix[DType]`) with SIMD alignment and contiguous memory
- [x] Sparse Matrices (`CSRMatrix[DType]`, `CSCMatrix[DType]`) and Sparse Ops
- [x] Memory-efficient `Dataset[feat_dtype, target_dtype]` container
- [x] Sequential `Pipeline` chaining (`PipelineTransformer`, `PipelineRegressor`, `PipelineClassifier`)
- [ ] `ColumnTransformer` (sub-matrix slicing for heterogeneous features)
- [ ] Zero-copy buffer views and Python/NumPy array interop
- [ ] Multi-threaded matrix multiplication and batch ops via `parallel_for`

### Preprocessing & Encoders
- [x] `StandardScaler` (SIMD-vectorized mean/variance standardization)
- [x] `MinMaxScaler` & `RobustScaler`
- [x] `Normalizer` (L1, L2, Max row-wise normalization)
- [x] `OneHotEncoder` (Dense category expansion with `drop` and `handle_unknown` policies)
- [x] `OrdinalEncoder` & `LabelEncoder`
- [x] `SimpleImputer` (mean, median, most_frequent, constant)
- [x] `Binarizer` (threshold-based indicator mapping)
- [x] `PolynomialFeatures` (interaction and degree expansion)

### Linear Models
- [x] `LinearRegression` (Ordinary Least Squares via QR, SVD, Cholesky, Solve)
- [x] `Ridge` (Analytical $L_2$-regularization closed-form solver)
- [ ] `Lasso` (Coordinate Descent solver)
- [ ] `ElasticNet` ($L_1 + L_2$ Coordinate Descent solver)
- [x] `LogisticRegression` (Binary and multinomial softmax classification)
- [ ] `SGDClassifier` & `SGDRegressor` (Streaming and online mini-batch optimization)

### Tree-Based & Ensemble Models
- [x] `DecisionTreeClassifier` (Gini impurity / Entropy / Log-Loss)
- [x] `DecisionTreeRegressor` (MSE / Friedman MSE / MAE split criteria)
- [x] `RandomForestClassifier` (Bagging with soft voting and batched OOB score)
- [x] `RandomForestRegressor` (Bagging with mean averaging and batched OOB score)
- [ ] `HistGradientBoostingClassifier` (UInt8 binning + parallel histogram building)
- [ ] `HistGradientBoostingRegressor`
- [ ] `VotingClassifier` & `StackingClassifier`

### Cluster Analysis & Dimensionality Reduction
- [x] `PCA` (Exact SVD, mean centering, whitening, and inverse reconstruction)
- [x] `TruncatedSVD` (Linear projection for dense and sparse `CSRMatrix` via SpMM)
- [x] `KMeans` (k-means++ initialization, SIMD Lloyd's algorithm)
- [x] `MiniBatchKMeans` (Streaming cluster updates)
- [ ] `KModes` (Categorical frequency-based clustering)
- [ ] `DBSCAN` (Density-based spatial clustering)

### Nearest Neighbors
- [ ] SIMD Distance Kernels (Euclidean, Manhattan, Cosine, Minkowski)
- [ ] `KNeighborsClassifier` (Brute-force & Index-backed)
- [ ] `KNeighborsRegressor`
- [ ] `KDTree` / `BallTree` Spatial Indexing

### Model Selection & Validation
- [x] `train_test_split` (with deterministic RNG seed)
- [x] `KFold` & `StratifiedKFold` cross-validation splitters
- [x] `cross_val_score` (Single-metric cross-validated scoring)
- [x] `TimeSeriesSplit` (Rolling / expanding temporal cross-validation with gap embargo)
- [x] `ShuffleSplit` (Monte Carlo random train/test splits)
- [x] `StratifiedShuffleSplit` (Class-balanced Monte Carlo splits)
- [x] `GridSearchRegressor` & `GridSearchClassifier` (Exhaustive hyperparameter grid search)
- [x] `RandomizedSearchRegressor` & `RandomizedSearchClassifier` (Sampled hyperparameter search)
- [x] `cross_val_predict` (Out-of-fold prediction matrices)
- [x] `cross_validate` (Multi-metric evaluation)

### Evaluation Metrics
- [x] `accuracy_score`
- [x] `precision_score`, `recall_score`, `f1_score`
- [x] `confusion_matrix`
- [x] `roc_auc_score` & `log_loss`
- [x] `mean_squared_error` ($MSE$ & $RMSE$)
- [x] `mean_absolute_error` ($MAE$)
- [x] `r2_score` ($R^2$)
- [x] `silhouette_score`
