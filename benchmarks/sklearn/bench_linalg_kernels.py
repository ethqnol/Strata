import sys
import json
import time
import math
import numpy as np
import scipy.sparse as sp


def compute_stats(benchmark, phase, samples, features, times_ns, metric_name="none", metric_val=0.0):
    n = len(times_ns)
    ms_list = [t / 1_000_000.0 for t in times_ns]
    mean_ms = sum(ms_list) / n
    min_ms = min(ms_list)
    max_ms = max(ms_list)
    sorted_ms = sorted(ms_list)
    if n % 2 == 1:
        median_ms = sorted_ms[n // 2]
    else:
        median_ms = (sorted_ms[n // 2 - 1] + sorted_ms[n // 2]) / 2.0

    sum_sq_diff = sum((x - mean_ms) ** 2 for x in ms_list)
    std_ms = math.sqrt(sum_sq_diff / n)
    throughput = (samples / median_ms) * 1000.0 if median_ms > 0 else 0.0

    return {
        "benchmark": benchmark,
        "phase": phase,
        "samples": samples,
        "features": features,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "throughput_samples_per_sec": throughput,
        "metric_name": metric_name,
        "metric_val": metric_val,
        "iterations": n,
    }


def make_synthetic_sparse(rows, cols, nnz_per_row=10, seed=42):
    rng = np.random.RandomState(seed)
    data = []
    indices = []
    indptr = [0]
    for _ in range(rows):
        for _ in range(nnz_per_row):
            c = rng.randint(0, cols)
            val = (rng.randint(1, 1000)) / 100.0
            data.append(val)
            indices.append(c)
        indptr.append(len(data))
    return sp.csr_matrix((data, indices, indptr), shape=(rows, cols), dtype=np.float64)


def run_gemm(size, warmups, iters):
    A = np.full((size, size), 1.5, dtype=np.float64)
    B = np.full((size, size), 2.5, dtype=np.float64)

    for _ in range(warmups):
        _ = A.dot(B)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = A.dot(B)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    print(json.dumps(compute_stats("Dense_GEMM", "matmul", size, size, times)))


def run_dense_dot_vec(rows, cols, warmups, iters):
    A = np.full((rows, cols), 1.25, dtype=np.float64)
    x = np.full(cols, 0.75, dtype=np.float64)

    for _ in range(warmups):
        _ = A.dot(x)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = A.dot(x)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    print(json.dumps(compute_stats("Dense_Dot_Vec", "matvec", rows, cols, times)))


def run_spmv(dim, nnz_per_row, warmups, iters):
    A = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=42)
    x = np.ones(dim, dtype=np.float64)

    for _ in range(warmups):
        _ = A.dot(x)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = A.dot(x)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    print(json.dumps(compute_stats("Sparse_SpMV", "matvec", dim, dim, times)))


def run_spmm(dim, k_cols, nnz_per_row, warmups, iters):
    A = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=42)
    B = np.ones((dim, k_cols), dtype=np.float64)

    for _ in range(warmups):
        _ = A.dot(B)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = A.dot(B)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    print(json.dumps(compute_stats("Sparse_SpMM", "matmul", dim, k_cols, times)))


def run_spgemm(dim, nnz_per_row, warmups, iters):
    A = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=42)
    B = make_synthetic_sparse(dim, dim, nnz_per_row=nnz_per_row, seed=43)

    for _ in range(warmups):
        _ = A.dot(B)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _ = A.dot(B)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    print(json.dumps(compute_stats("Sparse_SpGEMM", "matmul", dim, dim, times)))


def main():
    gemm_size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    cols = 50
    sparse_dim = 10000
    warmups = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    run_gemm(gemm_size, warmups, iters)
    run_dense_dot_vec(rows, cols, warmups, iters)
    run_spmv(sparse_dim, 20, warmups, iters)
    run_spmm(min(sparse_dim, 2000), 64, 20, warmups, iters)
    run_spgemm(min(sparse_dim, 2000), 10, warmups, iters)


if __name__ == "__main__":
    main()
