from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    StandardScaler,
    PipelineRegressor,
    PipelineClassifier,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)
from strata.ensemble._binning import (
    BinnedMatrix,
    _find_bin_idx,
    _compute_bin_thresholds,
    _map_to_bins,
)
from strata.ensemble._gb_loss import (
    LeastSquaresLoss,
    BinaryCrossEntropyLoss,
    MulticlassCrossEntropyLoss,
)
from strata.ensemble._hist_tree import (
    FeatureHistogram,
    HistNode,
    HistTree,
    HistSplit,
    _build_node_histograms,
    _find_best_split_in_histograms,
)
from strata.ensemble.hist_gradient_boosting_regressor import (
    HistGradientBoostingRegressor,
)
from strata.ensemble.hist_gradient_boosting_classifier import (
    HistGradientBoostingClassifier,
)


def test_find_bin_idx_boundaries() raises:
    var thresholds: List[Float64] = [2.0, 5.0, 8.0]
    # Bins:
    # bin 0: val <= 2.0
    # bin 1: 2.0 < val <= 5.0
    # bin 2: 5.0 < val <= 8.0
    # bin 3: val > 8.0

    assert_equal(_find_bin_idx(1.0, thresholds), 0)
    assert_equal(_find_bin_idx(2.0, thresholds), 0)
    assert_equal(_find_bin_idx(2.1, thresholds), 1)
    assert_equal(_find_bin_idx(5.0, thresholds), 1)
    assert_equal(_find_bin_idx(5.1, thresholds), 2)
    assert_equal(_find_bin_idx(8.0, thresholds), 2)
    assert_equal(_find_bin_idx(8.1, thresholds), 3)
    assert_equal(_find_bin_idx(100.0, thresholds), 3)

    # Empty thresholds (constant feature)
    var empty_thresh = List[Float64]()
    assert_equal(_find_bin_idx(42.0, empty_thresh), 0)


def test_binned_matrix_basic() raises:
    var data: List[UInt8] = [0, 1, 2, 3, 0, 1]
    var th1: List[Float64] = [1.5, 3.5]
    var th2: List[Float64] = [10.0]
    var thresholds = List[List[Float64]]()
    thresholds.append(th1^)
    thresholds.append(th2^)
    var n_bins: List[Int] = [3, 2]

    var bm = BinnedMatrix(3, 2, data^, thresholds^, n_bins^)
    assert_equal(bm.rows, 3)
    assert_equal(bm.cols, 2)
    assert_equal(bm.get(0, 0), 0)
    assert_equal(bm.get(0, 1), 1)
    assert_equal(bm.get(1, 0), 2)
    assert_equal(bm.get(1, 1), 3)
    assert_equal(bm.get(2, 0), 0)
    assert_equal(bm.get(2, 1), 1)

    bm.set(0, 0, 7)
    assert_equal(bm.get(0, 0), 7)

    var r0 = bm.row(0)
    assert_equal(len(r0), 2)
    assert_equal(r0[0], 7)
    assert_equal(r0[1], 1)

    var c1 = bm.col(1)
    assert_equal(len(c1), 3)
    assert_equal(c1[0], 1)
    assert_equal(c1[1], 3)
    assert_equal(c1[2], 1)

    var copy_bm = BinnedMatrix(copy=bm)
    assert_equal(copy_bm.rows, 3)
    assert_equal(copy_bm.cols, 2)
    assert_equal(copy_bm.get(0, 0), 7)


def test_compute_bin_thresholds_exact_distinct() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 10.0
    X[1, 0] = 2.0
    X[1, 1] = 20.0
    X[2, 0] = 3.0
    X[2, 1] = 30.0
    X[3, 0] = 4.0
    X[3, 1] = 40.0

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=10)
    assert_equal(len(thresholds), 2)

    # Column 0 thresholds: (1+2)/2=1.5, (2+3)/2=2.5, (3+4)/2=3.5
    assert_equal(len(thresholds[0]), 3)
    assert_almost_equal(thresholds[0][0], 1.5)
    assert_almost_equal(thresholds[0][1], 2.5)
    assert_almost_equal(thresholds[0][2], 3.5)

    # Column 1 thresholds: 15.0, 25.0, 35.0
    assert_equal(len(thresholds[1]), 3)
    assert_almost_equal(thresholds[1][0], 15.0)
    assert_almost_equal(thresholds[1][1], 25.0)
    assert_almost_equal(thresholds[1][2], 35.0)


def test_compute_bin_thresholds_subsampled_quantiles() raises:
    var N = 500
    var X = Matrix[DType.float64](N, 1, 0)
    for i in range(N):
        X[i, 0] = Float64(i)

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=16)
    assert_equal(len(thresholds), 1)
    assert_true(len(thresholds[0]) <= 15)
    assert_true(len(thresholds[0]) > 0)

    # Monotonicity check
    for k in range(1, len(thresholds[0])):
        assert_true(thresholds[0][k] > thresholds[0][k - 1])


def test_compute_bin_thresholds_constant_features() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 7.0
    X[1, 0] = 2.0
    X[1, 1] = 7.0
    X[2, 0] = 3.0
    X[2, 1] = 7.0
    X[3, 0] = 4.0
    X[3, 1] = 7.0
    X[4, 0] = 5.0
    X[4, 1] = 7.0

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=256)
    assert_equal(len(thresholds[0]), 4)
    assert_equal(len(thresholds[1]), 0)  # Constant column produces 0 thresholds

    var binned = _map_to_bins[DType.float64](X, thresholds)
    assert_equal(binned.n_bins_per_feature[1], 1)
    for i in range(5):
        assert_equal(binned.get(i, 1), 0)


def test_map_to_bins_matrix_shape() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 10.0
    X[1, 0] = 2.0
    X[1, 1] = 20.0
    X[2, 0] = 3.0
    X[2, 1] = 30.0
    X[3, 0] = 4.0
    X[3, 1] = 40.0

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=256)
    var binned = _map_to_bins[DType.float64](X, thresholds)

    assert_equal(binned.rows, 4)
    assert_equal(binned.cols, 2)
    assert_equal(binned.get(0, 0), 0)
    assert_equal(binned.get(1, 0), 1)
    assert_equal(binned.get(2, 0), 2)
    assert_equal(binned.get(3, 0), 3)


def test_binning_invalid_parameters() raises:
    var X = Matrix[DType.float64](3, 2, 1.0)
    with assert_raises():
        _ = _compute_bin_thresholds[DType.float64](X, max_bins=1)

    with assert_raises():
        _ = _compute_bin_thresholds[DType.float64](X, max_bins=300)

    var bad_thresh = List[List[Float64]]()
    with assert_raises():
        _ = _map_to_bins[DType.float64](X, bad_thresh)


def test_binning_float32() raises:
    var X = Matrix[DType.float32](3, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0

    var thresholds = _compute_bin_thresholds[DType.float32](X, max_bins=256)
    var binned = _map_to_bins[DType.float32](X, thresholds)

    assert_equal(binned.rows, 3)
    assert_equal(binned.cols, 1)
    assert_equal(binned.get(0, 0), 0)
    assert_equal(binned.get(1, 0), 1)
    assert_equal(binned.get(2, 0), 2)


def test_binning_edge_cases() raises:
    # 1. Extreme float values
    var X_ext = Matrix[DType.float64](3, 1, 0)
    X_ext[0, 0] = -1e15
    X_ext[1, 0] = 0.0
    X_ext[2, 0] = 1e15

    var th_ext = _compute_bin_thresholds[DType.float64](X_ext, max_bins=256)
    assert_equal(len(th_ext[0]), 2)
    var b_ext = _map_to_bins[DType.float64](X_ext, th_ext)
    assert_equal(b_ext.get(0, 0), 0)
    assert_equal(b_ext.get(1, 0), 1)
    assert_equal(b_ext.get(2, 0), 2)

    # 2. Raw buffer pointer access
    var ptr = b_ext.unsafe_ptr()
    assert_equal(ptr.unsafe_load(), 0)
    assert_equal(ptr.unsafe_offset(1).unsafe_load(), 1)
    assert_equal(ptr.unsafe_offset(2).unsafe_load(), 2)

    # 3. Empty 0x0 matrix
    var X_empty = Matrix[DType.float64](0, 0, 0)
    var th_empty = _compute_bin_thresholds[DType.float64](X_empty)
    assert_equal(len(th_empty), 0)
    var b_empty = _map_to_bins[DType.float64](X_empty, th_empty)
    assert_equal(b_empty.rows, 0)
    assert_equal(b_empty.cols, 0)


def test_least_squares_loss() raises:
    var loss_fn = LeastSquaresLoss()

    # 1. Scalar loss: 0.5 * (y_hat - y)^2
    assert_almost_equal(loss_fn.loss(3.0, 5.0), 2.0)
    assert_almost_equal(loss_fn.loss(5.0, 5.0), 0.0)

    # 2. Gradient: y_hat - y, Hessian: 1.0
    assert_almost_equal(loss_fn.gradient(3.0, 5.0), 2.0)
    assert_almost_equal(loss_fn.gradient(5.0, 3.0), -2.0)
    assert_equal(loss_fn.hessian(3.0, 5.0), 1.0)

    # 3. Initial baseline raw prediction = mean(y)
    var y_vals: List[Float64] = [2.0, 4.0, 6.0, 8.0]
    assert_almost_equal(loss_fn.init_raw_prediction(y_vals), 5.0)

    # 4. Batched gradients and hessians
    var raw_preds: List[Float64] = [2.5, 4.5, 5.5, 7.5]
    var grads = List[Float64]()
    var hess = List[Float64]()
    loss_fn.update_gradients_and_hessians(y_vals, raw_preds, grads, hess)
    assert_equal(len(grads), 4)
    assert_equal(len(hess), 4)
    assert_almost_equal(grads[0], 0.5)
    assert_almost_equal(grads[1], 0.5)
    assert_almost_equal(grads[2], -0.5)
    assert_almost_equal(grads[3], -0.5)
    assert_equal(hess[0], 1.0)


def test_binary_cross_entropy_loss() raises:
    var loss_fn = BinaryCrossEntropyLoss()

    # 1. Predict proba: sigmoid(0.0) = 0.5, sigmoid(large) ~ 1.0
    assert_almost_equal(loss_fn.predict_proba(0.0), 0.5)
    assert_true(loss_fn.predict_proba(10.0) > 0.99)
    assert_true(loss_fn.predict_proba(-10.0) < 0.01)

    # 2. Loss at margin 0: -ln(0.5) = ln(2) ~ 0.693147
    assert_almost_equal(loss_fn.loss(1.0, 0.0), 0.69314718056, rtol=1e-4)
    assert_almost_equal(loss_fn.loss(0.0, 0.0), 0.69314718056, rtol=1e-4)

    # 3. Gradient: p - y. At margin 0, p = 0.5
    assert_almost_equal(loss_fn.gradient(1.0, 0.0), -0.5)
    assert_almost_equal(loss_fn.gradient(0.0, 0.0), 0.5)

    # 4. Hessian: p*(1 - p) = 0.25 at margin 0
    assert_almost_equal(loss_fn.hessian(1.0, 0.0), 0.25)

    # 5. Baseline raw prediction: log-odds
    var y_balanced: List[Float64] = [0.0, 1.0]
    assert_almost_equal(loss_fn.init_raw_prediction(y_balanced), 0.0, atol=1e-4)


def test_multiclass_cross_entropy_loss() raises:
    var loss_fn = MulticlassCrossEntropyLoss(3)

    # 1. Softmax probability distribution sums to 1.0
    var raw_preds: List[Float64] = [1.0, 2.0, 3.0]
    var probs = loss_fn.predict_proba(raw_preds)
    assert_equal(len(probs), 3)
    var sum_p = probs[0] + probs[1] + probs[2]
    assert_almost_equal(sum_p, 1.0)
    assert_true(probs[2] > probs[1])
    assert_true(probs[1] > probs[0])

    # 2. Negative log likelihood loss
    var loss_val = loss_fn.loss(2, raw_preds)
    assert_true(loss_val > 0.0)

    # 3. Initial raw predictions (priors) for 3 classes
    var y_labels: List[Int] = [0, 1, 1, 2, 2, 2]
    var init_p = loss_fn.init_raw_predictions(y_labels)
    assert_equal(len(init_p), 3)
    assert_true(init_p[2] > init_p[1])
    assert_true(init_p[1] > init_p[0])

    # 4. Class-specific gradient and hessian update
    var raw_all: List[Float64] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    var grads = List[Float64]()
    var hess = List[Float64]()
    loss_fn.update_gradients_and_hessians_for_class(
        y_labels, raw_all, 2, grads, hess
    )
    assert_equal(len(grads), 6)
    assert_equal(len(hess), 6)
    # At uniform logits 0, p_2 = 1/3 ~ 0.3333
    # For sample 0 (y=0 != 2): g = 1/3 - 0 = 0.3333
    assert_almost_equal(grads[3], -2.0 / 3.0, rtol=1e-3)


def test_gb_loss_edge_cases() raises:
    # 1. Extreme margins (+/- 500)
    var bce = BinaryCrossEntropyLoss()
    assert_true(bce.loss(1.0, 500.0) >= 0.0)
    assert_true(bce.loss(0.0, -500.0) >= 0.0)
    assert_almost_equal(bce.hessian(1.0, 500.0), 1e-16)
    assert_almost_equal(bce.hessian(1.0, -500.0), 1e-16)

    # 2. Pure binary targets (all 0 or all 1)
    var y_all_0: List[Float64] = [0.0, 0.0, 0.0]
    var y_all_1: List[Float64] = [1.0, 1.0, 1.0]
    var init_0 = bce.init_raw_prediction(y_all_0)
    var init_1 = bce.init_raw_prediction(y_all_1)
    assert_true(init_0 < -10.0)
    assert_true(init_1 > 10.0)

    # 3. Empty input list
    var y_empty = List[Float64]()
    var ls = LeastSquaresLoss()
    assert_equal(ls.init_raw_prediction(y_empty), 0.0)
    assert_equal(bce.init_raw_prediction(y_empty), 0.0)


def test_feature_histogram_subtraction() raises:
    var parent = FeatureHistogram(4)
    parent.grad_sum[0] = 5.0
    parent.grad_sum[1] = 10.0
    parent.grad_sum[2] = 15.0
    parent.grad_sum[3] = 20.0
    parent.hess_sum[0] = 1.0
    parent.hess_sum[1] = 2.0
    parent.hess_sum[2] = 3.0
    parent.hess_sum[3] = 4.0
    parent.count[0] = 10
    parent.count[1] = 20
    parent.count[2] = 30
    parent.count[3] = 40

    var child = FeatureHistogram(4)
    child.grad_sum[0] = 2.0
    child.grad_sum[1] = 4.0
    child.grad_sum[2] = 6.0
    child.grad_sum[3] = 8.0
    child.hess_sum[0] = 0.5
    child.hess_sum[1] = 1.0
    child.hess_sum[2] = 1.5
    child.hess_sum[3] = 2.0
    child.count[0] = 4
    child.count[1] = 8
    child.count[2] = 12
    child.count[3] = 16

    var sibling = parent.subtract(child)
    assert_almost_equal(sibling.grad_sum[0], 3.0)
    assert_almost_equal(sibling.grad_sum[1], 6.0)
    assert_almost_equal(sibling.grad_sum[2], 9.0)
    assert_almost_equal(sibling.grad_sum[3], 12.0)
    assert_almost_equal(sibling.hess_sum[0], 0.5)
    assert_almost_equal(sibling.hess_sum[3], 2.0)
    assert_equal(sibling.count[0], 6)
    assert_equal(sibling.count[3], 24)

    # Identical parent-child subtraction yields exact 0
    var exact_zero_sibling = child.subtract(child)
    assert_equal(exact_zero_sibling.hess_sum[0], 0.0)
    assert_equal(exact_zero_sibling.count[0], 0)


def test_hist_tree_single_split() raises:
    # 6 samples along feature 0: 0, 1, 2 (left) and 10, 11, 12 (right)
    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 10.0
    X[4, 0] = 11.0
    X[5, 0] = 12.0

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=256)
    var binned = _map_to_bins[DType.float64](X, thresholds)

    # Gradients: negative for left, positive for right
    var grads: List[Float64] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    var tree = HistTree(
        max_depth=2,
        min_samples_leaf=2,
        l2_regularization=1.0,
        shrinkage=1.0,
    )
    tree.build(binned, grads, hess)

    assert_true(len(tree.nodes) >= 3)
    assert_equal(tree.nodes[0].is_leaf, False)
    assert_equal(tree.nodes[0].feature_idx, 0)

    var preds = tree.predict(X)
    assert_equal(len(preds), 6)
    # Left leaf value: -(-3.0 / (3 + 1)) = +0.75
    assert_almost_equal(preds[0], 0.75)
    assert_almost_equal(preds[1], 0.75)
    assert_almost_equal(preds[2], 0.75)
    # Right leaf value: -(3.0 / (3 + 1)) = -0.75
    assert_almost_equal(preds[3], -0.75)
    assert_almost_equal(preds[4], -0.75)
    assert_almost_equal(preds[5], -0.75)


def test_hist_tree_regularization() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=256)
    var binned = _map_to_bins[DType.float64](X, thresholds)

    var grads: List[Float64] = [-2.0, -2.0, 2.0, 2.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0]

    # Tree with lambda = 0
    var tree_no_reg = HistTree(
        max_depth=1, min_samples_leaf=1, l2_regularization=0.0, shrinkage=1.0
    )
    tree_no_reg.build(binned, grads, hess)
    var preds_no_reg = tree_no_reg.predict(X)

    # Tree with lambda = 10
    var tree_reg = HistTree(
        max_depth=1, min_samples_leaf=1, l2_regularization=10.0, shrinkage=1.0
    )
    tree_reg.build(binned, grads, hess)
    var preds_reg = tree_reg.predict(X)

    # Regularization strictly shrinks magnitude of leaf updates
    assert_true(preds_no_reg[0] > preds_reg[0])
    assert_true(preds_no_reg[3] < preds_reg[3])


def test_hist_tree_binned_vs_continuous_parity() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.5
    X[1, 0] = 2.0
    X[1, 1] = 1.5
    X[2, 0] = 3.0
    X[2, 1] = 2.5
    X[3, 0] = 10.0
    X[3, 1] = 0.5
    X[4, 0] = 11.0
    X[4, 1] = 1.5
    X[5, 0] = 12.0
    X[5, 1] = 2.5

    var thresholds = _compute_bin_thresholds[DType.float64](X, max_bins=256)
    var binned = _map_to_bins[DType.float64](X, thresholds)

    var grads: List[Float64] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    var tree = HistTree(max_depth=3, min_samples_leaf=1, l2_regularization=1.0)
    tree.build(binned, grads, hess)

    for i in range(6):
        var cont_pred = tree.predict_row(X, i)
        var bin_pred = tree.predict_binned(binned, i)
        assert_almost_equal(cont_pred, bin_pred)


def test_hist_gradient_boosting_regressor_1d() raises:
    # 8 samples along y = 2.0 * x + 1.0
    var X = Matrix[DType.float64](8, 1, 0)
    var y: List[Scalar[DType.float64]] = [
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        15.0,
    ]
    for i in range(8):
        X[i, 0] = Float64(i)

    var reg = HistGradientBoostingRegressor(
        learning_rate=0.2,
        max_iter=50,
        min_samples_leaf=1,
        early_stopping=False,
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 1)

    var preds = reg.predict(X)
    assert_equal(len(preds), 8)
    var r2 = reg.score(X, y)
    assert_true(r2 > 0.90)


def test_hist_gradient_boosting_regressor_multivariate() raises:
    # y = 3*x0 - 2*x1 + 5
    var N = 12
    var X = Matrix[DType.float64](N, 2, 0)
    var y = List[Scalar[DType.float64]](capacity=N)
    for i in range(N):
        var x0 = Float64(i % 4)
        var x1 = Float64(i // 4)
        X[i, 0] = x0
        X[i, 1] = x1
        y.append(3.0 * x0 - 2.0 * x1 + 5.0)

    var reg = HistGradientBoostingRegressor(
        learning_rate=0.2,
        max_iter=60,
        min_samples_leaf=1,
        early_stopping=False,
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)
    var r2 = reg.score(X, y)
    assert_true(r2 > 0.90)


def test_hist_gradient_boosting_regressor_early_stopping() raises:
    var N = 40
    var X = Matrix[DType.float64](N, 1, 0)
    var y = List[Scalar[DType.float64]](capacity=N)
    for i in range(N):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 2))

    var reg = HistGradientBoostingRegressor(
        learning_rate=0.2,
        max_iter=100,
        min_samples_leaf=1,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)
    assert_true(reg.n_iter_ <= 100)


def test_hist_gradient_boosting_regressor_pipeline() raises:
    var scaler = StandardScaler()
    var reg = HistGradientBoostingRegressor(
        learning_rate=0.2, max_iter=30, min_samples_leaf=1, early_stopping=False
    )
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](6, 1, 0)
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    for i in range(6):
        X[i, 0] = Float64(i + 1)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 6)


def test_hist_gradient_boosting_regressor_float32() raises:
    var X = Matrix[DType.float32](6, 1, 0)
    var y: List[Scalar[DType.float32]] = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    for i in range(6):
        X[i, 0] = Float32(i)

    var reg = HistGradientBoostingRegressor[DType.float32](
        learning_rate=0.2, max_iter=30, min_samples_leaf=1, early_stopping=False
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)
    var preds = reg.predict(X)
    assert_equal(len(preds), 6)


def test_hist_gradient_boosting_classifier_binary() raises:
    # 6 points: 3 for class 0, 3 for class 1
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -3.0
    X[0, 1] = -3.0
    X[1, 0] = -2.0
    X[1, 1] = -4.0
    X[2, 0] = -4.0
    X[2, 1] = -2.0
    X[3, 0] = 3.0
    X[3, 1] = 3.0
    X[4, 0] = 2.0
    X[4, 1] = 4.0
    X[5, 0] = 4.0
    X[5, 1] = 2.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    var clf = HistGradientBoostingClassifier(
        learning_rate=0.3,
        max_iter=30,
        min_samples_leaf=1,
        early_stopping=False,
    )
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(len(clf.classes_), 2)

    var preds = clf.predict(X)
    for i in range(3):
        assert_equal(preds[i], 0)
    for i in range(3, 6):
        assert_equal(preds[i], 1)

    var acc = clf.score(X, y)
    assert_almost_equal(acc, 1.0)


def test_hist_gradient_boosting_classifier_predict_proba() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -5.0
    X[1, 0] = -4.0
    X[2, 0] = 4.0
    X[3, 0] = 5.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf = HistGradientBoostingClassifier(
        learning_rate=0.3, max_iter=30, min_samples_leaf=1, early_stopping=False
    )
    clf.fit(X, y)

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)

    # Probabilities sum to 1
    for i in range(4):
        assert_almost_equal(Float64(probs[i, 0] + probs[i, 1]), 1.0)

    assert_true(probs[0, 0] > 0.8)
    assert_true(probs[3, 1] > 0.8)


def test_hist_gradient_boosting_classifier_multiclass_3class() raises:
    # 6 samples: 2 per class
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -10.0
    X[0, 1] = -10.0
    X[1, 0] = -9.0
    X[1, 1] = -9.0
    X[2, 0] = 0.0
    X[2, 1] = 0.0
    X[3, 0] = 0.5
    X[3, 1] = -0.5
    X[4, 0] = 10.0
    X[4, 1] = 10.0
    X[5, 0] = 9.0
    X[5, 1] = 9.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    var clf = HistGradientBoostingClassifier(
        learning_rate=0.3,
        max_iter=40,
        min_samples_leaf=1,
        early_stopping=False,
    )
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 3)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)
    assert_equal(preds[4], 2)
    assert_equal(preds[5], 2)

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 3)
    for i in range(6):
        var sum_p = Float64(probs[i, 0] + probs[i, 1] + probs[i, 2])
        assert_almost_equal(sum_p, 1.0)


def test_hist_gradient_boosting_classifier_multiclass_4class() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    X[0, 0] = -10.0
    X[0, 1] = -10.0
    X[1, 0] = -9.0
    X[1, 1] = -9.0
    X[2, 0] = -10.0
    X[2, 1] = 10.0
    X[3, 0] = -9.0
    X[3, 1] = 9.0
    X[4, 0] = 10.0
    X[4, 1] = -10.0
    X[5, 0] = 9.0
    X[5, 1] = -9.0
    X[6, 0] = 10.0
    X[6, 1] = 10.0
    X[7, 0] = 9.0
    X[7, 1] = 9.0

    var y: List[Scalar[DType.float64]] = [
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        3.0,
    ]

    var clf = HistGradientBoostingClassifier(
        learning_rate=0.3,
        max_iter=40,
        min_samples_leaf=1,
        early_stopping=False,
    )
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 4)
    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[4], 2)
    assert_equal(preds[6], 3)


def test_hist_gradient_boosting_classifier_pipeline() raises:
    var scaler = StandardScaler()
    var clf = HistGradientBoostingClassifier(
        learning_rate=0.3, max_iter=20, min_samples_leaf=1, early_stopping=False
    )
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)


def test_hist_gradient_boosting_invalid_params() raises:
    with assert_raises():
        _ = HistGradientBoostingRegressor(loss="unknown_loss")

    with assert_raises():
        _ = HistGradientBoostingRegressor(learning_rate=-0.1)

    with assert_raises():
        _ = HistGradientBoostingRegressor(max_iter=0)

    with assert_raises():
        _ = HistGradientBoostingRegressor(max_leaf_nodes=1)

    with assert_raises():
        _ = HistGradientBoostingRegressor(max_depth=0)

    with assert_raises():
        _ = HistGradientBoostingRegressor(max_bins=1)

    with assert_raises():
        _ = HistGradientBoostingRegressor(max_bins=300)

    with assert_raises():
        _ = HistGradientBoostingClassifier(loss="invalid_loss")

    with assert_raises():
        _ = HistGradientBoostingClassifier(validation_fraction=1.5)


def test_hist_gradient_boosting_not_fitted_and_mismatch() raises:
    var reg = HistGradientBoostingRegressor()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = reg.predict(X)

    var y: List[Scalar[DType.float64]] = [1.0, 2.0]
    reg.fit(X, y)

    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = reg.predict(X_wrong)

    var clf = HistGradientBoostingClassifier()
    with assert_raises():
        _ = clf.predict(X)

    with assert_raises():
        _ = clf.predict_proba(X)


def test_hist_gradient_boosting_copy() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf1 = HistGradientBoostingClassifier(
        learning_rate=0.2, max_iter=10, min_samples_leaf=1, early_stopping=False
    )
    clf1.fit(X, y)

    var clf2 = HistGradientBoostingClassifier(copy=clf1)
    assert_true(clf2.is_fitted)
    assert_equal(len(clf2.classes_), len(clf1.classes_))
    assert_equal(len(clf2.trees_), len(clf1.trees_))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
