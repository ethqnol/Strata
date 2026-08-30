from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    Dataset,
    StandardScaler,
    PipelineRegressor,
    PipelineClassifier,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)
from strata.base import fit, predict
from strata.linear_model.sgd_regressor import SGDRegressor
from strata.linear_model.sgd_classifier import SGDClassifier


def test_sgd_regressor_squared_error_1d() raises:
    # y = 2 * x + 1
    var X = Matrix[DType.float64](5, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    X[4, 0] = 4.0

    var y: List[Scalar[DType.float64]] = [1.0, 3.0, 5.0, 7.0, 9.0]

    var reg = SGDRegressor(
        loss="squared_error",
        penalty="none",
        learning_rate="constant",
        eta0=0.05,
        max_iter=500,
    )
    reg.fit(X, y)

    assert_true(reg.is_fitted)
    assert_equal(reg.n_features_in_, 1)
    assert_almost_equal(reg.coef_[0], 2.0, rtol=0.1)
    assert_almost_equal(reg.intercept_, 1.0, rtol=0.1)

    var preds = reg.predict(X)
    assert_equal(len(preds), 5)
    assert_almost_equal(preds[2], 5.0, rtol=0.1)


def test_sgd_regressor_multivariate() raises:
    # y = 1.5*x0 - 2.0*x1 + 3.0
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 2.0
    X[2, 1] = 1.0
    X[3, 0] = 1.0
    X[3, 1] = 2.0
    X[4, 0] = 3.0
    X[4, 1] = 1.0
    X[5, 0] = 2.0
    X[5, 1] = 3.0

    var y = List[Scalar[DType.float64]](capacity=6)
    for i in range(6):
        y.append(1.5 * Float64(X[i, 0]) - 2.0 * Float64(X[i, 1]) + 3.0)

    var reg = SGDRegressor(
        loss="squared_error",
        penalty="none",
        learning_rate="constant",
        eta0=0.05,
        max_iter=800,
    )
    reg.fit(X, y)

    assert_almost_equal(reg.coef_[0], 1.5, rtol=0.15)
    assert_almost_equal(reg.coef_[1], -2.0, rtol=0.15)
    assert_almost_equal(reg.intercept_, 3.0, rtol=0.15)


def test_sgd_regressor_huber_loss() raises:
    var X = Matrix[DType.float64](5, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    X[4, 0] = 4.0

    var y: List[Scalar[DType.float64]] = [1.0, 3.0, 20.0, 7.0, 9.0]

    var reg = SGDRegressor(
        loss="huber",
        epsilon=1.0,
        learning_rate="constant",
        eta0=0.05,
        max_iter=600,
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)


def test_sgd_regressor_epsilon_insensitive() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var reg = SGDRegressor(
        loss="epsilon_insensitive",
        epsilon=0.1,
        learning_rate="constant",
        eta0=0.05,
        max_iter=400,
    )
    reg.fit(X, y)
    assert_true(reg.is_fitted)


def test_sgd_regressor_penalties() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 2.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [2.0, 5.0, 8.0, 11.0]

    var reg_l2 = SGDRegressor(penalty="l2", alpha=0.01, max_iter=200)
    reg_l2.fit(X, y)
    assert_true(reg_l2.is_fitted)

    var reg_l1 = SGDRegressor(penalty="l1", alpha=0.01, max_iter=200)
    reg_l1.fit(X, y)
    assert_true(reg_l1.is_fitted)

    var reg_en = SGDRegressor(
        penalty="elasticnet", alpha=0.01, l1_ratio=0.5, max_iter=200
    )
    reg_en.fit(X, y)
    assert_true(reg_en.is_fitted)

    var reg_none = SGDRegressor(penalty="none", max_iter=200)
    reg_none.fit(X, y)
    assert_true(reg_none.is_fitted)


def test_sgd_regressor_learning_rate_schedules() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var reg_const = SGDRegressor(learning_rate="constant", eta0=0.01)
    reg_const.fit(X, y)
    assert_true(reg_const.is_fitted)

    var reg_inv = SGDRegressor(learning_rate="invscaling", eta0=0.01)
    reg_inv.fit(X, y)
    assert_true(reg_inv.is_fitted)

    var reg_opt = SGDRegressor(learning_rate="optimal", alpha=0.01, eta0=0.01)
    reg_opt.fit(X, y)
    assert_true(reg_opt.is_fitted)


def test_sgd_regressor_no_intercept() raises:
    # y = 3.0 * x
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [3.0, 6.0, 9.0]

    var reg = SGDRegressor(
        fit_intercept=False,
        learning_rate="constant",
        eta0=0.05,
        max_iter=500,
    )
    reg.fit(X, y)
    assert_equal(reg.intercept_, 0.0)
    assert_almost_equal(reg.coef_[0], 3.0, rtol=0.1)


def test_sgd_regressor_float32() raises:
    var X = Matrix[DType.float32](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float32]] = [2.0, 4.0, 6.0]

    var reg = SGDRegressor[DType.float32](learning_rate="constant", eta0=0.05)
    reg.fit(X, y)
    assert_true(reg.is_fitted)


def test_sgd_regressor_dataset_container() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    var ds = Dataset[DType.float64, DType.float64](X^, y^)
    var reg = SGDRegressor()
    fit(reg, ds)
    assert_true(reg.is_fitted)


def test_sgd_regressor_pipeline() raises:
    var scaler = StandardScaler()
    var sgd = SGDRegressor(learning_rate="constant", eta0=0.05)
    var pipe = PipelineRegressor(scaler^, sgd^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)


def test_sgd_regressor_invalid_params() raises:
    with assert_raises():
        _ = SGDRegressor(loss="invalid_loss")

    with assert_raises():
        _ = SGDRegressor(penalty="invalid_penalty")

    with assert_raises():
        _ = SGDRegressor(alpha=-1.0)

    with assert_raises():
        _ = SGDRegressor(l1_ratio=-0.1)

    with assert_raises():
        _ = SGDRegressor(l1_ratio=1.1)

    with assert_raises():
        _ = SGDRegressor(max_iter=0)

    with assert_raises():
        _ = SGDRegressor(tol=-0.1)

    with assert_raises():
        _ = SGDRegressor(eta0=0.0)


def test_sgd_regressor_not_fitted_and_mismatch() raises:
    var reg = SGDRegressor()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = reg.predict(X)

    var y: List[Scalar[DType.float64]] = [1.0, 2.0]
    reg.fit(X, y)

    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = reg.predict(X_wrong)


def test_sgd_classifier_binary_hinge() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -3.0
    X[1, 1] = -1.0
    X[2, 0] = -1.0
    X[2, 1] = -3.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0
    X[4, 0] = 3.0
    X[4, 1] = 1.0
    X[5, 0] = 1.0
    X[5, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    var clf = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        max_iter=300,
    )
    clf.fit(X, y)

    assert_true(clf.is_fitted)
    assert_equal(len(clf.classes_), 2)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 0)
    assert_equal(preds[3], 1)
    assert_equal(preds[4], 1)
    assert_equal(preds[5], 1)


def test_sgd_classifier_log_loss_predict_proba() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -5.0
    X[1, 0] = -3.0
    X[2, 0] = 3.0
    X[3, 0] = 5.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.001,
        learning_rate="constant",
        eta0=0.1,
        max_iter=300,
    )
    clf.fit(X, y)

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)

    assert_true(probs[0, 0] > 0.8)
    assert_true(probs[1, 0] > 0.8)
    assert_true(probs[2, 1] > 0.8)
    assert_true(probs[3, 1] > 0.8)


def test_sgd_classifier_modified_huber_and_squared_hinge() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -3.0
    X[1, 0] = -2.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf_hub = SGDClassifier(loss="modified_huber", eta0=0.1, max_iter=300)
    clf_hub.fit(X, y)
    assert_true(clf_hub.is_fitted)

    var clf_sqh = SGDClassifier(loss="squared_hinge", eta0=0.1, max_iter=300)
    clf_sqh.fit(X, y)
    assert_true(clf_sqh.is_fitted)


def test_sgd_classifier_decision_function() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -4.0
    X[1, 0] = -2.0
    X[2, 0] = 2.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf = SGDClassifier(loss="hinge", eta0=0.1, max_iter=300)
    clf.fit(X, y)

    var scores = clf.decision_function(X)
    assert_equal(scores.rows, 4)
    assert_equal(scores.cols, 1)

    assert_true(scores[0, 0] < 0.0)
    assert_true(scores[3, 0] > 0.0)


def test_sgd_classifier_multiclass_ovr_3class() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -5.0
    X[0, 1] = -5.0
    X[1, 0] = -4.0
    X[1, 1] = -4.0
    X[2, 0] = 0.0
    X[2, 1] = 0.0
    X[3, 0] = 0.5
    X[3, 1] = -0.5
    X[4, 0] = 5.0
    X[4, 1] = 5.0
    X[5, 0] = 4.0
    X[5, 1] = 4.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    var clf = SGDClassifier(
        loss="modified_huber",
        learning_rate="constant",
        eta0=0.05,
        max_iter=500,
    )
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 3)
    assert_equal(clf.coef_.rows, 3)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[4], 2)
    assert_equal(preds[5], 2)

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 3)


def test_sgd_classifier_multiclass_ovr_4class() raises:
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

    var clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.05,
        max_iter=500,
    )
    clf.fit(X, y)

    assert_equal(len(clf.classes_), 4)
    assert_equal(clf.coef_.rows, 4)

    var preds = clf.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[4], 2)
    assert_equal(preds[6], 3)


def test_sgd_classifier_penalties() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -2.0
    X[1, 0] = -1.0
    X[1, 1] = -1.0
    X[2, 0] = 1.0
    X[2, 1] = 1.0
    X[3, 0] = 2.0
    X[3, 1] = 2.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf_l1 = SGDClassifier(penalty="l1", alpha=0.01)
    clf_l1.fit(X, y)
    assert_true(clf_l1.is_fitted)

    var clf_en = SGDClassifier(penalty="elasticnet", alpha=0.01, l1_ratio=0.5)
    clf_en.fit(X, y)
    assert_true(clf_en.is_fitted)

    var clf_none = SGDClassifier(penalty="none")
    clf_none.fit(X, y)
    assert_true(clf_none.is_fitted)


def test_sgd_classifier_pipeline() raises:
    var scaler = StandardScaler()
    var clf = SGDClassifier(eta0=0.1, max_iter=300)
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


def test_sgd_classifier_float32() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.float32]] = [0.0, 0.0, 1.0, 1.0]

    var clf = SGDClassifier[DType.float32](eta0=0.1, max_iter=300)
    clf.fit(X, y)
    assert_true(clf.is_fitted)


def test_sgd_classifier_copy() raises:
    var clf1 = SGDClassifier(loss="log_loss")
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = -1.0
    X[1, 0] = 1.0
    var y: List[Scalar[DType.float64]] = [0.0, 1.0]
    clf1.fit(X, y)

    var clf2 = SGDClassifier(copy=clf1)
    assert_true(clf2.is_fitted)
    assert_equal(clf2.loss, "log_loss")
    assert_equal(len(clf2.classes_), 2)


def test_sgd_classifier_invalid_params() raises:
    with assert_raises():
        _ = SGDClassifier(loss="unknown_loss")

    with assert_raises():
        _ = SGDClassifier(penalty="unknown_penalty")

    with assert_raises():
        _ = SGDClassifier(alpha=-0.1)

    with assert_raises():
        _ = SGDClassifier(l1_ratio=-0.1)

    with assert_raises():
        _ = SGDClassifier(l1_ratio=1.1)

    with assert_raises():
        _ = SGDClassifier(max_iter=0)

    with assert_raises():
        _ = SGDClassifier(tol=-1.0)


def test_sgd_classifier_not_fitted_and_mismatch() raises:
    var clf = SGDClassifier()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = clf.predict(X)

    with assert_raises():
        _ = clf.predict_proba(X)

    var y: List[Scalar[DType.float64]] = [0.0, 1.0]
    clf.fit(X, y)

    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = clf.predict(X_wrong)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
