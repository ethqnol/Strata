# Strata vs. Scikit-Learn High-Performance Benchmark Suite

**Date**: 2026-09-01 14:02:46 UTC
**Workload Scale**: `QUICK`
**Methodology**: Zero-overhead in-memory allocation, dedicated warmup iterations, nanosecond-precision monotonic timing (`perf_counter_ns`), median execution times across repetitions, strict single-thread environment control.

---

## Executive Summary

- **Total Benchmarks Evaluated**: 53
- **Overall Geometric Mean Speedup**: **1.79x**
- **Strata Faster (≥ 1.05x)**: **42** (79.2%)
- **Equivalent / Parity (0.95x - 1.05x)**: **2**
- **Scikit-Learn Faster (< 0.95x)**: **9**

## 1. Model Training & Fitting Performance (`fit`)

> **Category Summary**: Geometric Mean Speedup = **1.91x** (20/24 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LinearRegression` | `fit` | 2,000 x 10 | 428.0 µs | 1.04 ms | **2.44x** | 4,672,619 samples/s | r2_score: 1.0000 (Exact) |
| `Ridge` | `fit` | 2,000 x 10 | 474.4 µs | 808.6 µs | **1.70x** | 4,215,478 samples/s | r2_score: 1.0000 (Exact) |
| `Lasso` | `fit` | 2,000 x 10 | 379.0 µs | 655.7 µs | **1.73x** | 5,276,641 samples/s | r2_score: 1.0000 (Exact) |
| `ElasticNet` | `fit` | 2,000 x 10 | 399.7 µs | 983.7 µs | **2.46x** | 5,003,596 samples/s | r2_score: 1.0000 (Exact) |
| `LogisticRegression` | `fit` | 2,000 x 10 | 9.16 ms | 5.59 ms | 0.61x | 218,342 samples/s | accuracy: 1.0000 (Exact) |
| `SGDRegressor` | `fit` | 2,000 x 10 | 1.81 ms | 2.37 ms | **1.31x** | 1,106,442 samples/s | r2_score: 1.0000 (Exact) |
| `SGDClassifier` | `fit` | 2,000 x 10 | 946.7 µs | 2.25 ms | **2.38x** | 2,112,668 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeClassifier` | `fit` | 2,000 x 10 | 2.11 ms | 4.75 ms | **2.25x** | 946,393 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeRegressor` | `fit` | 2,000 x 10 | 14.60 ms | 20.53 ms | **1.41x** | 136,965 samples/s | r2_score: 0.9839 vs 0.9890 (Δ=5.0e-03) |
| `RandomForestClassifier` | `fit` | 2,000 x 10 | 23.28 ms | 35.02 ms | **1.50x** | 85,927 samples/s | accuracy: 1.0000 (Exact) |
| `RandomForestRegressor` | `fit` | 2,000 x 10 | 113.26 ms | 230.05 ms | **2.03x** | 17,658 samples/s | r2_score: 0.9687 vs 0.9853 (Δ=1.7e-02) |
| `HistGradientBoostingClassifier` | `fit` | 2,000 x 10 | 77.46 ms | 27.66 ms | 0.36x | 25,820 samples/s | accuracy: 1.0000 (Exact) |
| `HistGradientBoostingRegressor` | `fit` | 2,000 x 10 | 106.22 ms | 63.64 ms | 0.60x | 18,829 samples/s | r2_score: 0.9462 vs 0.9709 (Δ=2.5e-02) |
| `KMeans` | `fit` | 2,000 x 10 | 778.8 µs | 2.49 ms | **3.19x** | 2,567,966 samples/s | inertia: 6579.5800 vs 6562.3084 (Δ=1.7e+01) |
| `MiniBatchKMeans` | `fit` | 2,000 x 10 | 3.76 ms | 6.52 ms | **1.74x** | 532,605 samples/s | inertia: 6584.3252 vs 6569.3755 (Δ=1.5e+01) |
| `PCA` | `fit` | 2,000 x 10 | 561.7 µs | 545.5 µs | 0.97x (parity) | 3,560,743 samples/s | explained_variance_ratio_0: 0.9804 vs 0.9773 (Δ=3.1e-03) |
| `TruncatedSVD_CSR` | `fit` | 2,000 x 10 | 1.31 ms | 3.15 ms | **2.41x** | 1,526,846 samples/s | Exact Match |
| `NearestNeighbors` | `fit` | 2,000 x 10 | 120.9 µs | 516.7 µs | **4.27x** | 16,541,024 samples/s | Exact Match |
| `KNeighborsClassifier` | `fit` | 2,000 x 10 | 176.3 µs | 633.2 µs | **3.59x** | 11,345,587 samples/s | accuracy: 1.0000 (Exact) |
| `KNeighborsRegressor` | `fit` | 2,000 x 10 | 162.1 µs | 328.8 µs | **2.03x** | 12,337,454 samples/s | r2_score: 0.9194 vs 0.9118 (Δ=7.6e-03) |
| `StandardScaler` | `fit` | 5,000 x 10 | 254.6 µs | 1.16 ms | **4.56x** | 19,637,376 samples/s | Exact Match |
| `MinMaxScaler` | `fit` | 5,000 x 10 | 151.7 µs | 796.7 µs | **5.25x** | 32,964,461 samples/s | Exact Match |
| `RobustScaler` | `fit` | 5,000 x 10 | 4.00 ms | 4.38 ms | **1.09x** | 1,248,877 samples/s | Exact Match |
| `PolynomialFeatures_d2` | `fit` | 5,000 x 10 | 82.4 µs | 602.4 µs | **7.31x** | 60,709,451 samples/s | Exact Match |

## 2. Model Inference & Transformation Throughput (`predict`, `transform`, `query`)

> **Category Summary**: Geometric Mean Speedup = **2.15x** (21/24 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LinearRegression` | `predict` | 2,000 x 10 | 52.9 µs | 146.6 µs | **2.77x** | 37,836,866 samples/s | r2_score: 1.0000 (Exact) |
| `Ridge` | `predict` | 2,000 x 10 | 50.5 µs | 131.6 µs | **2.61x** | 39,608,666 samples/s | r2_score: 1.0000 (Exact) |
| `Lasso` | `predict` | 2,000 x 10 | 39.0 µs | 137.2 µs | **3.52x** | 51,297,835 samples/s | r2_score: 1.0000 (Exact) |
| `ElasticNet` | `predict` | 2,000 x 10 | 44.2 µs | 185.6 µs | **4.20x** | 45,235,564 samples/s | r2_score: 1.0000 (Exact) |
| `LogisticRegression` | `predict` | 2,000 x 10 | 90.3 µs | 524.8 µs | **5.81x** | 22,143,858 samples/s | accuracy: 1.0000 (Exact) |
| `SGDRegressor` | `predict` | 2,000 x 10 | 35.2 µs | 225.3 µs | **6.40x** | 56,816,568 samples/s | r2_score: 1.0000 (Exact) |
| `SGDClassifier` | `predict` | 2,000 x 10 | 100.5 µs | 226.8 µs | **2.26x** | 19,897,924 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeClassifier` | `predict` | 2,000 x 10 | 53.9 µs | 296.6 µs | **5.50x** | 37,097,836 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeRegressor` | `predict` | 2,000 x 10 | 219.9 µs | 1.01 ms | **4.60x** | 9,096,015 samples/s | r2_score: 0.9839 vs 0.9890 (Δ=5.0e-03) |
| `RandomForestClassifier` | `predict` | 2,000 x 10 | 2.03 ms | 1.74 ms | 0.86x | 986,470 samples/s | accuracy: 1.0000 (Exact) |
| `RandomForestRegressor` | `predict` | 2,000 x 10 | 6.75 ms | 8.24 ms | **1.22x** | 296,321 samples/s | r2_score: 0.9687 vs 0.9853 (Δ=1.7e-02) |
| `HistGradientBoostingClassifier` | `predict` | 2,000 x 10 | 2.64 ms | 3.10 ms | **1.18x** | 758,034 samples/s | accuracy: 1.0000 (Exact) |
| `HistGradientBoostingRegressor` | `predict` | 2,000 x 10 | 2.81 ms | 5.53 ms | **1.96x** | 710,571 samples/s | r2_score: 0.9462 vs 0.9709 (Δ=2.5e-02) |
| `KMeans` | `predict` | 2,000 x 10 | 140.3 µs | 268.6 µs | **1.91x** | 14,252,374 samples/s | inertia: 6579.5800 vs 6562.3084 (Δ=1.7e+01) |
| `MiniBatchKMeans` | `predict` | 2,000 x 10 | 135.4 µs | 279.8 µs | **2.07x** | 14,769,576 samples/s | inertia: 6584.3252 vs 6569.3755 (Δ=1.5e+01) |
| `PCA` | `transform` | 2,000 x 10 | 219.0 µs | 166.5 µs | 0.76x | 9,130,356 samples/s | explained_variance_ratio_0: 0.9804 vs 0.9773 (Δ=3.1e-03) |
| `TruncatedSVD_CSR` | `transform` | 2,000 x 10 | 336.8 µs | 274.4 µs | 0.81x | 5,939,124 samples/s | Exact Match |
| `NearestNeighbors` | `kneighbors` | 500 x 10 | 4.21 ms | 5.58 ms | **1.33x** | 118,814 samples/s | Exact Match |
| `KNeighborsClassifier` | `predict` | 500 x 10 | 4.34 ms | 6.75 ms | **1.55x** | 115,113 samples/s | accuracy: 1.0000 (Exact) |
| `KNeighborsRegressor` | `predict` | 500 x 10 | 4.31 ms | 5.95 ms | **1.38x** | 115,939 samples/s | r2_score: 0.9194 vs 0.9118 (Δ=7.6e-03) |
| `StandardScaler` | `transform` | 5,000 x 10 | 125.0 µs | 330.5 µs | **2.64x** | 40,010,083 samples/s | Exact Match |
| `MinMaxScaler` | `transform` | 5,000 x 10 | 122.8 µs | 292.5 µs | **2.38x** | 40,718,270 samples/s | Exact Match |
| `RobustScaler` | `transform` | 5,000 x 10 | 129.5 µs | 506.2 µs | **3.91x** | 38,604,971 samples/s | Exact Match |
| `PolynomialFeatures_d2` | `transform` | 5,000 x 10 | 2.26 ms | 2.37 ms | **1.05x** | 2,214,936 samples/s | Exact Match |

## 3. Low-Level Linear Algebra & Sparse Kernels (`gemm`, `spmv`, `spmm`)

> **Category Summary**: Geometric Mean Speedup = **0.53x** (1/5 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Dense_GEMM` | `matmul` | 256 x 256 | 1.37 ms | 1.41 ms | 1.03x (parity) | 187,083 samples/s | Exact Match |
| `Dense_Dot_Vec` | `matvec` | 10,000 x 50 | 81.9 µs | 113.1 µs | **1.38x** | 122,088,197 samples/s | Exact Match |
| `Sparse_SpMV` | `matvec` | 10,000 x 10000 | 1.08 ms | 273.1 µs | 0.25x | 9,266,999 samples/s | Exact Match |
| `Sparse_SpMM` | `matmul` | 2,000 x 64 | 4.08 ms | 1.13 ms | 0.28x | 489,933 samples/s | Exact Match |
| `Sparse_SpGEMM` | `matmul` | 2,000 x 2000 | 3.89 ms | 1.65 ms | 0.42x | 514,775 samples/s | Exact Match |
