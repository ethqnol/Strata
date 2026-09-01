# Strata vs. Scikit-Learn High-Performance Benchmark Suite

**Date**: 2026-09-01 12:58:41 UTC  
**Workload Scale**: `QUICK`  
**Methodology**: Zero-overhead in-memory allocation, dedicated warmup iterations, nanosecond-precision monotonic timing (`perf_counter_ns`), median execution times across repetitions, strict single-thread environment control.

---

## Executive Summary

- **Total Benchmarks Evaluated**: 53
- **Overall Geometric Mean Speedup**: **1.27x**
- **Strata Faster (≥ 1.05x)**: **30** (56.6%)
- **Equivalent / Parity (0.95x - 1.05x)**: **2**
- **Scikit-Learn Faster (< 0.95x)**: **21**

## 1. Model Training & Fitting Performance (`fit`)

> **Category Summary**: Geometric Mean Speedup = **2.08x** (17/24 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LinearRegression` | `fit` | 2,000 x 10 | 475.3 µs | 1.10 ms | **2.31x** | 4,207,718 samples/s | r2_score: 1.0000 (Exact) |
| `Ridge` | `fit` | 2,000 x 10 | 451.9 µs | 843.2 µs | **1.87x** | 4,426,218 samples/s | r2_score: 1.0000 (Exact) |
| `Lasso` | `fit` | 2,000 x 10 | 373.2 µs | 623.7 µs | **1.67x** | 5,359,746 samples/s | r2_score: 1.0000 (Exact) |
| `ElasticNet` | `fit` | 2,000 x 10 | 378.7 µs | 667.9 µs | **1.76x** | 5,281,895 samples/s | r2_score: 1.0000 (Exact) |
| `LogisticRegression` | `fit` | 2,000 x 10 | 54.26 ms | 3.30 ms | 0.06x | 36,856 samples/s | accuracy: 1.0000 (Exact) |
| `SGDRegressor` | `fit` | 2,000 x 10 | 1.83 ms | 1.91 ms | 1.04x (parity) | 1,090,496 samples/s | r2_score: 1.0000 (Exact) |
| `SGDClassifier` | `fit` | 2,000 x 10 | 947.0 µs | 2.17 ms | **2.29x** | 2,111,927 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeClassifier` | `fit` | 2,000 x 10 | 2.06 ms | 21.15 ms | **10.27x** | 970,751 samples/s | accuracy: 1.0000 vs 0.9065 (Δ=9.4e-02) |
| `DecisionTreeRegressor` | `fit` | 2,000 x 10 | 14.40 ms | 17.35 ms | **1.21x** | 138,901 samples/s | r2_score: 0.9839 vs 0.9890 (Δ=5.0e-03) |
| `RandomForestClassifier` | `fit` | 2,000 x 10 | 17.47 ms | 101.62 ms | **5.82x** | 114,511 samples/s | accuracy: 1.0000 vs 0.9150 (Δ=8.5e-02) |
| `RandomForestRegressor` | `fit` | 2,000 x 10 | 107.92 ms | 222.34 ms | **2.06x** | 18,532 samples/s | r2_score: 0.9687 vs 0.9853 (Δ=1.7e-02) |
| `HistGradientBoostingClassifier` | `fit` | 2,000 x 10 | 82.69 ms | 40.84 ms | 0.49x | 24,187 samples/s | accuracy: 1.0000 vs 0.9150 (Δ=8.5e-02) |
| `HistGradientBoostingRegressor` | `fit` | 2,000 x 10 | 101.00 ms | 63.31 ms | 0.63x | 19,803 samples/s | r2_score: 0.9462 vs 0.9709 (Δ=2.5e-02) |
| `KMeans` | `fit` | 2,000 x 10 | 2.44 ms | 2.64 ms | **1.08x** | 818,756 samples/s | inertia: 6579.5800 vs 6562.3084 (Δ=1.7e+01) |
| `MiniBatchKMeans` | `fit` | 2,000 x 10 | 12.66 ms | 6.06 ms | 0.48x | 157,954 samples/s | inertia: 6584.3252 vs 6569.3755 (Δ=1.5e+01) |
| `PCA` | `fit` | 2,000 x 10 | 559.6 µs | 546.6 µs | 0.98x (parity) | 3,573,662 samples/s | explained_variance_ratio_0: 0.9804 vs 0.9773 (Δ=3.1e-03) |
| `TruncatedSVD_CSR` | `fit` | 2,000 x 10 | 1.10 ms | 3.13 ms | **2.84x** | 1,814,623 samples/s | Exact Match |
| `NearestNeighbors` | `fit` | 2,000 x 10 | 110.3 µs | 2.35 ms | **21.31x** | 18,139,602 samples/s | Exact Match |
| `KNeighborsClassifier` | `fit` | 2,000 x 10 | 155.8 µs | 2.63 ms | **16.86x** | 12,834,005 samples/s | accuracy: 1.0000 (Exact) |
| `KNeighborsRegressor` | `fit` | 2,000 x 10 | 159.9 µs | 2.54 ms | **15.87x** | 12,507,896 samples/s | r2_score: 0.9194 vs 0.9118 (Δ=7.6e-03) |
| `StandardScaler` | `fit` | 5,000 x 10 | 263.7 µs | 1.19 ms | **4.52x** | 18,964,069 samples/s | Exact Match |
| `MinMaxScaler` | `fit` | 5,000 x 10 | 152.9 µs | 1.02 ms | **6.68x** | 32,705,283 samples/s | Exact Match |
| `RobustScaler` | `fit` | 5,000 x 10 | 4.51 ms | 3.05 ms | 0.68x | 1,109,423 samples/s | Exact Match |
| `PolynomialFeatures_d2` | `fit` | 5,000 x 10 | 84.4 µs | 269.9 µs | **3.20x** | 59,232,934 samples/s | Exact Match |

## 2. Model Inference & Transformation Throughput (`predict`, `transform`, `query`)

> **Category Summary**: Geometric Mean Speedup = **1.00x** (13/24 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LinearRegression` | `predict` | 2,000 x 10 | 46.2 µs | 146.8 µs | **3.17x** | 43,251,192 samples/s | r2_score: 1.0000 (Exact) |
| `Ridge` | `predict` | 2,000 x 10 | 45.5 µs | 133.4 µs | **2.93x** | 43,927,081 samples/s | r2_score: 1.0000 (Exact) |
| `Lasso` | `predict` | 2,000 x 10 | 45.6 µs | 139.4 µs | **3.06x** | 43,854,360 samples/s | r2_score: 1.0000 (Exact) |
| `ElasticNet` | `predict` | 2,000 x 10 | 45.5 µs | 135.7 µs | **2.98x** | 43,951,214 samples/s | r2_score: 1.0000 (Exact) |
| `LogisticRegression` | `predict` | 2,000 x 10 | 204.9 µs | 234.7 µs | **1.15x** | 9,760,716 samples/s | accuracy: 1.0000 (Exact) |
| `SGDRegressor` | `predict` | 2,000 x 10 | 46.2 µs | 138.1 µs | **2.99x** | 43,326,618 samples/s | r2_score: 1.0000 (Exact) |
| `SGDClassifier` | `predict` | 2,000 x 10 | 111.1 µs | 221.6 µs | **2.00x** | 18,002,205 samples/s | accuracy: 1.0000 (Exact) |
| `DecisionTreeClassifier` | `predict` | 2,000 x 10 | 53.6 µs | 425.4 µs | **7.93x** | 37,302,994 samples/s | accuracy: 1.0000 vs 0.9065 (Δ=9.4e-02) |
| `DecisionTreeRegressor` | `predict` | 2,000 x 10 | 218.3 µs | 534.8 µs | **2.45x** | 9,160,970 samples/s | r2_score: 0.9839 vs 0.9890 (Δ=5.0e-03) |
| `RandomForestClassifier` | `predict` | 2,000 x 10 | 1.37 ms | 2.95 ms | **2.15x** | 1,460,401 samples/s | accuracy: 1.0000 vs 0.9150 (Δ=8.5e-02) |
| `RandomForestRegressor` | `predict` | 2,000 x 10 | 6.16 ms | 7.95 ms | **1.29x** | 324,593 samples/s | r2_score: 0.9687 vs 0.9853 (Δ=1.7e-02) |
| `HistGradientBoostingClassifier` | `predict` | 2,000 x 10 | 2.62 ms | 3.76 ms | **1.44x** | 764,197 samples/s | accuracy: 1.0000 vs 0.9150 (Δ=8.5e-02) |
| `HistGradientBoostingRegressor` | `predict` | 2,000 x 10 | 2.70 ms | 5.55 ms | **2.06x** | 740,629 samples/s | r2_score: 0.9462 vs 0.9709 (Δ=2.5e-02) |
| `KMeans` | `predict` | 2,000 x 10 | 568.0 µs | 323.0 µs | 0.57x | 3,520,876 samples/s | inertia: 6579.5800 vs 6562.3084 (Δ=1.7e+01) |
| `MiniBatchKMeans` | `predict` | 2,000 x 10 | 569.2 µs | 251.8 µs | 0.44x | 3,513,592 samples/s | inertia: 6584.3252 vs 6569.3755 (Δ=1.5e+01) |
| `PCA` | `transform` | 2,000 x 10 | 312.8 µs | 177.7 µs | 0.57x | 6,393,678 samples/s | explained_variance_ratio_0: 0.9804 vs 0.9773 (Δ=3.1e-03) |
| `TruncatedSVD_CSR` | `transform` | 2,000 x 10 | 330.1 µs | 266.1 µs | 0.81x | 6,058,183 samples/s | Exact Match |
| `NearestNeighbors` | `kneighbors` | 500 x 10 | 113.46 ms | 6.29 ms | 0.06x | 4,407 samples/s | Exact Match |
| `KNeighborsClassifier` | `predict` | 500 x 10 | 117.28 ms | 11.65 ms | 0.10x | 4,263 samples/s | accuracy: 1.0000 (Exact) |
| `KNeighborsRegressor` | `predict` | 500 x 10 | 120.51 ms | 18.62 ms | 0.15x | 4,149 samples/s | r2_score: 0.9194 vs 0.9118 (Δ=7.6e-03) |
| `StandardScaler` | `transform` | 5,000 x 10 | 433.7 µs | 330.8 µs | 0.76x | 11,528,733 samples/s | Exact Match |
| `MinMaxScaler` | `transform` | 5,000 x 10 | 793.4 µs | 337.8 µs | 0.43x | 6,301,745 samples/s | Exact Match |
| `RobustScaler` | `transform` | 5,000 x 10 | 427.8 µs | 319.2 µs | 0.75x | 11,686,912 samples/s | Exact Match |
| `PolynomialFeatures_d2` | `transform` | 5,000 x 10 | 3.99 ms | 1.65 ms | 0.41x | 1,251,829 samples/s | Exact Match |

## 3. Low-Level Linear Algebra & Sparse Kernels (`gemm`, `spmv`, `spmm`)

> **Category Summary**: Geometric Mean Speedup = **0.39x** (0/5 benchmarks faster in Strata)

| Estimator / Component | Phase | Workload (N x D) | Strata Median | Scikit-Learn Median | Speedup | Throughput (Strata) | Parity / Quality Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Dense_GEMM` | `matmul` | 256 x 256 | 4.54 ms | 1.40 ms | 0.31x | 56,412 samples/s | Exact Match |
| `Dense_Dot_Vec` | `matvec` | 10,000 x 50 | 153.4 µs | 108.3 µs | 0.71x | 65,205,413 samples/s | Exact Match |
| `Sparse_SpMV` | `matvec` | 10,000 x 10000 | 955.4 µs | 263.2 µs | 0.28x | 10,466,344 samples/s | Exact Match |
| `Sparse_SpMM` | `matmul` | 2,000 x 64 | 3.37 ms | 1.07 ms | 0.32x | 594,242 samples/s | Exact Match |
| `Sparse_SpGEMM` | `matmul` | 2,000 x 2000 | 3.53 ms | 1.67 ms | 0.47x | 567,104 samples/s | Exact Match |

