import sys
import json
import time
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score


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


def make_synthetic_classification(n_samples, n_features, n_classes=2, seed=42):
    rng = np.random.RandomState(seed)
    centroids = np.zeros((n_classes, n_features), dtype=np.float64)
    for c in range(n_classes):
        centroids[c] = (c * 3.0) + (rng.randint(-100, 100, size=n_features)) / 100.0

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        c = i % n_classes
        y[i] = c
        offset = (rng.randint(-1000, 1000, size=n_features)) / 1000.0
        X[i] = centroids[c] + offset

    return np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.int32)


def make_synthetic_regression(n_samples, n_features, seed=42):
    rng = np.random.RandomState(seed)
    weights = (rng.randint(-1000, 1000, size=n_features)) / 100.0
    X = (rng.randint(-1000, 1000, size=(n_samples, n_features))) / 100.0
    dot_val = 2.5 + X.dot(weights)
    noise = (rng.randint(-100, 100, size=n_samples)) / 500.0
    y = dot_val + noise
    return np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)


def run_nearest_neighbors(samples, features, warmups, iters):
    X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)
    query_samples = min(samples, 500)
    X_query = X[:query_samples]

    for _ in range(warmups):
        nn = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=1)
        nn.fit(X)
        _ = nn.kneighbors(X_query)

    fit_times, query_times = [], []

    for _ in range(iters):
        nn = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=1)
        t0 = time.perf_counter_ns()
        nn.fit(X)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        _ = nn.kneighbors(X_query)
        t3 = time.perf_counter_ns()
        query_times.append(t3 - t2)

    print(json.dumps(compute_stats("NearestNeighbors", "fit", samples, features, fit_times)))
    print(json.dumps(compute_stats("NearestNeighbors", "kneighbors", query_samples, features, query_times)))


def run_kneighbors_classifier(samples, features, warmups, iters):
    X, y = make_synthetic_classification(samples, features, n_classes=2, seed=42)
    query_samples = min(samples, 500)
    X_query = X[:query_samples]
    y_query = y[:query_samples]

    for _ in range(warmups):
        clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=1)
        clf.fit(X, y)
        _ = clf.predict(X_query)

    fit_times, pred_times = [], []
    last_acc = 0.0

    for _ in range(iters):
        clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=1)
        t0 = time.perf_counter_ns()
        clf.fit(X, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = clf.predict(X_query)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_acc = float(accuracy_score(y_query, preds))

    print(json.dumps(compute_stats("KNeighborsClassifier", "fit", samples, features, fit_times, "accuracy", last_acc)))
    print(json.dumps(compute_stats("KNeighborsClassifier", "predict", query_samples, features, pred_times, "accuracy", last_acc)))


def run_kneighbors_regressor(samples, features, warmups, iters):
    X, y = make_synthetic_regression(samples, features, seed=42)
    query_samples = min(samples, 500)
    X_query = X[:query_samples]
    y_query = y[:query_samples]

    for _ in range(warmups):
        reg = KNeighborsRegressor(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=1)
        reg.fit(X, y)
        _ = reg.predict(X_query)

    fit_times, pred_times = [], []
    last_r2 = 0.0

    for _ in range(iters):
        reg = KNeighborsRegressor(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=1)
        t0 = time.perf_counter_ns()
        reg.fit(X, y)
        t1 = time.perf_counter_ns()
        fit_times.append(t1 - t0)

        t2 = time.perf_counter_ns()
        preds = reg.predict(X_query)
        t3 = time.perf_counter_ns()
        pred_times.append(t3 - t2)

        last_r2 = float(r2_score(y_query, preds))

    print(json.dumps(compute_stats("KNeighborsRegressor", "fit", samples, features, fit_times, "r2_score", last_r2)))
    print(json.dumps(compute_stats("KNeighborsRegressor", "predict", query_samples, features, pred_times, "r2_score", last_r2)))


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    warmups = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    run_nearest_neighbors(samples, features, warmups, iters)
    run_kneighbors_classifier(samples, features, warmups, iters)
    run_kneighbors_regressor(samples, features, warmups, iters)


if __name__ == "__main__":
    main()
