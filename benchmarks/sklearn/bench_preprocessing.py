import sys
import json
import time
import math
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)


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


def make_synthetic_blobs(n_samples, n_features, n_clusters=5, seed=42):
    rng = np.random.RandomState(seed)
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    for k in range(n_clusters):
        centers[k] = (k * 5.0) + (rng.randint(-100, 100, size=n_features)) / 50.0

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples):
        k = i % n_clusters
        noise = (rng.randint(-1000, 1000, size=n_features)) / 1000.0
        X[i] = centers[k] + noise

    return np.ascontiguousarray(X, dtype=np.float64)


def run_standard_scaler(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        scaler = StandardScaler()
        scaler.fit(X)
        _ = scaler.transform(X)

    fit_times, trans_times = [], []

    for _ in range(iters):
        scaler = StandardScaler()
        t0 = time.perf_counter_ns()
        scaler.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = scaler.transform(X)
        t3 = time.perf_counter_ns()
        trans_times.append(t3 - t2)

    print(json.dumps(compute_stats("StandardScaler", "fit", samples, features, fit_times)))
    print(json.dumps(compute_stats("StandardScaler", "transform", samples, features, trans_times)))


def run_minmax_scaler(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        scaler = MinMaxScaler()
        scaler.fit(X)
        _ = scaler.transform(X)

    fit_times, trans_times = [], []

    for _ in range(iters):
        scaler = MinMaxScaler()
        t0 = time.perf_counter_ns()
        scaler.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = scaler.transform(X)
        t3 = time.perf_counter_ns()
        trans_times.append(t3 - t2)

    print(json.dumps(compute_stats("MinMaxScaler", "fit", samples, features, fit_times)))
    print(json.dumps(compute_stats("MinMaxScaler", "transform", samples, features, trans_times)))


def run_robust_scaler(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)

    for _ in range(warmups):
        scaler = RobustScaler()
        scaler.fit(X)
        _ = scaler.transform(X)

    fit_times, trans_times = [], []

    for _ in range(iters):
        scaler = RobustScaler()
        t0 = time.perf_counter_ns()
        scaler.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = scaler.transform(X)
        t3 = time.perf_counter_ns()
        trans_times.append(t3 - t2)

    print(json.dumps(compute_stats("RobustScaler", "fit", samples, features, fit_times)))
    print(json.dumps(compute_stats("RobustScaler", "transform", samples, features, trans_times)))


def run_polynomial_features(samples, features, warmups, iters):
    poly_features = min(features, 10)
    X = make_synthetic_blobs(samples, poly_features, n_clusters=3, seed=42)

    for _ in range(warmups):
        poly = PolynomialFeatures(degree=2, include_bias=True)
        poly.fit(X)
        _ = poly.transform(X)

    fit_times, trans_times = [], []

    for _ in range(iters):
        poly = PolynomialFeatures(degree=2, include_bias=True)
        t0 = time.perf_counter_ns()
        poly.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = poly.transform(X)
        t3 = time.perf_counter_ns()
        trans_times.append(t3 - t2)

    print(json.dumps(compute_stats("PolynomialFeatures_d2", "fit", samples, poly_features, fit_times)))
    print(json.dumps(compute_stats("PolynomialFeatures_d2", "transform", samples, poly_features, trans_times)))


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    warmups = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    run_standard_scaler(samples, features, warmups, iters)
    run_minmax_scaler(samples, features, warmups, iters)
    run_robust_scaler(samples, features, warmups, iters)
    run_polynomial_features(samples, features, warmups, iters)


if __name__ == "__main__":
    main()
