from std.time import perf_counter_ns
from std.sys import argv
from strata import (
    Matrix,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    accuracy_score,
    r2_score,
)
from benchmarks.mojo.bench_utils import (
    BenchTimer,
    make_synthetic_regression,
    make_synthetic_classification,
)


def run_decision_tree_classifier(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var clf = DecisionTreeClassifier(max_depth=10, random_state=42)
        clf.fit(X, y)
        _ = clf.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = DecisionTreeClassifier(max_depth=10, random_state=42)
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "DecisionTreeClassifier",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "DecisionTreeClassifier",
        "predict",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_decision_tree_regressor(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = DecisionTreeRegressor(max_depth=10, random_state=42)
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = DecisionTreeRegressor(max_depth=10, random_state=42)
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "DecisionTreeRegressor",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "DecisionTreeRegressor",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_random_forest_classifier(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var clf = RandomForestClassifier(
            n_estimators=20, max_depth=10, random_state=42
        )
        clf.fit(X, y)
        _ = clf.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = RandomForestClassifier(
            n_estimators=20, max_depth=10, random_state=42
        )
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "RandomForestClassifier",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "RandomForestClassifier",
        "predict",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_random_forest_regressor(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = RandomForestRegressor(
            n_estimators=20, max_depth=10, random_state=42
        )
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = RandomForestRegressor(
            n_estimators=20, max_depth=10, random_state=42
        )
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "RandomForestRegressor",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "RandomForestRegressor",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_hist_gradient_boosting_classifier(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_classification(
        samples, features, n_classes=2, seed=42
    )
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var clf = HistGradientBoostingClassifier(
            max_iter=30, max_depth=6, random_state=42, early_stopping=False
        )
        clf.fit(X, y)
        _ = clf.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_acc: Float64 = 0.0

    for _ in range(iters):
        var clf = HistGradientBoostingClassifier(
            max_iter=30, max_depth=6, random_state=42, early_stopping=False
        )
        var t0 = perf_counter_ns()
        clf.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = clf.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_acc = accuracy_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "HistGradientBoostingClassifier",
        "fit",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    var pred_res = pred_timer.compute_stats(
        "HistGradientBoostingClassifier",
        "predict",
        samples,
        features,
        "accuracy",
        last_acc,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def run_hist_gradient_boosting_regressor(
    samples: Int, features: Int, warmups: Int, iters: Int
) raises:
    var data = make_synthetic_regression(samples, features, seed=42)
    var X = data[0].copy()
    var y = data[1].copy()

    for _ in range(warmups):
        var reg = HistGradientBoostingRegressor(
            max_iter=30, max_depth=6, random_state=42, early_stopping=False
        )
        reg.fit(X, y)
        _ = reg.predict(X)

    var fit_timer = BenchTimer()
    var pred_timer = BenchTimer()
    var last_r2: Float64 = 0.0

    for _ in range(iters):
        var reg = HistGradientBoostingRegressor(
            max_iter=30, max_depth=6, random_state=42, early_stopping=False
        )
        var t0 = perf_counter_ns()
        reg.fit(X, y)
        var t1 = perf_counter_ns()
        fit_timer.add(t1 - t0)

        var t2 = perf_counter_ns()
        var preds = reg.predict(X)
        var t3 = perf_counter_ns()
        pred_timer.add(t3 - t2)

        last_r2 = r2_score(y, preds)

    var fit_res = fit_timer.compute_stats(
        "HistGradientBoostingRegressor",
        "fit",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    var pred_res = pred_timer.compute_stats(
        "HistGradientBoostingRegressor",
        "predict",
        samples,
        features,
        "r2_score",
        last_r2,
    )
    print(fit_res.to_json())
    print(pred_res.to_json())


def main() raises:
    var args = argv()
    var samples = 10000
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

    run_decision_tree_classifier(samples, features, warmups, iters)
    run_decision_tree_regressor(samples, features, warmups, iters)
    run_random_forest_classifier(samples, features, warmups, iters)
    run_random_forest_regressor(samples, features, warmups, iters)
    run_hist_gradient_boosting_classifier(samples, features, warmups, iters)
    run_hist_gradient_boosting_regressor(samples, features, warmups, iters)
