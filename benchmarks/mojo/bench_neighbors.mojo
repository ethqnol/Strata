from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    NearestNeighbors,
    KNeighborsClassifier,
    KNeighborsRegressor,
    accuracy_score,
    r2_score,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_regression,
    make_synthetic_classification,
    make_synthetic_blobs,
)


def run_nearest_neighbors(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var X = make_synthetic_blobs(samples, features, n_clusters=5, seed=42)
    var query_samples = min(samples, 500)
    var X_query = X.slice_rows(0, query_samples).to_matrix()

    for _ in range(warmups):
        var nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn.fit(X)
        _ = nn.kneighbors(X_query)

    var fit_timer = BenchTimer()
    var query_timer = BenchTimer()

    for _ in range(iters):
        var nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        var t0 = perf_counter_ns()
        nn.fit(X)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        _ = nn.kneighbors(X_query)
        var t3 = perf_counter_ns()
        query_timer.add(t3 - t2)

    var fit_res = fit_timer.compute_stats(
        "NearestNeighbors",
        "fit",
        samples,
        features,
    )
    var query_res = query_timer.compute_stats(
        "NearestNeighbors",
        "kneighbors",
        query_samples,
        features,
    )
    print(fit_res.to_json())
    print(query_res.to_json())


def run_kneighbors_classifier(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()
    var query_samples = min(samples, 500)
    var X_query = X.slice_rows(0, query_samples).to_matrix()
    var y_query = List[Scalar[DType.int32]](capacity=query_samples)
    for i in range(query_samples):
        y_query.append(y[i])

    for _ in range(warmups):
        var clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        clf.fit(X, y)
        _ = clf.predict(X_query)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X_query)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y_query, preds)

    var fit_res = fit_timer.compute_stats(
        "KNeighborsClassifier",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "KNeighborsClassifier",
        "predict",
        query_samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_kneighbors_regressor(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()
    var query_samples = min(samples, 500)
    var X_query = X.slice_rows(0, query_samples).to_matrix()
    var y_query = List[Scalar[DType.float64]](capacity=query_samples)
    for i in range(query_samples):
        y_query.append(y[i])

    for _ in range(warmups):
        var reg = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
        reg.fit(X, y)
        _ = reg.predict(X_query)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X_query)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y_query, preds)

    var fit_res = fit_timer.compute_stats(
        "KNeighborsRegressor",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "KNeighborsRegressor",
        "predict",
        query_samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def main() raises:
    var args = argv()
    var samples = 5000
    var features = 15
    var warmups = 1
    var iters = 3

    if len(args) > 1:
        samples = Int(args[1])
    if len(args) > 2:
        features = Int(args[2])
    if len(args) > 3:
        warmups = Int(args[3])
    if len(args) > 4:
        iters = Int(args[4])

    run_nearest_neighbors(samples, features, warmups, iters)
    run_kneighbors_classifier(samples, features, warmups, iters)
    run_kneighbors_regressor(samples, features, warmups, iters)
