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
    MinMaxScaler,
    PolynomialFeatures,
    PipelineRegressor,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)
from strata.base import fit, predict
from strata.linear_model._coordinate_descent import (
    _soft_threshold,
    _coordinate_descent_elastic_net,
)
from strata.linear_model.lasso import Lasso
from strata.linear_model.elastic_net import ElasticNet


def test_soft_threshold() raises:
    assert_almost_equal(_soft_threshold[DType.float64](3.0, 1.0), 2.0)
    assert_almost_equal(_soft_threshold[DType.float64](-3.0, 1.0), -2.0)
    assert_equal(_soft_threshold[DType.float64](1.0, 1.0), 0.0)
    assert_equal(_soft_threshold[DType.float64](-1.0, 1.0), 0.0)
    assert_equal(_soft_threshold[DType.float64](0.5, 1.0), 0.0)
    assert_equal(_soft_threshold[DType.float64](-0.5, 1.0), 0.0)
    assert_equal(_soft_threshold[DType.float64](0.0, 1.0), 0.0)


def test_lasso_1d_fit_intercept() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0

    # y = 2.0 * x + 1.0
    var y: List[Scalar[DType.float64]] = [1.0, 3.0, 5.0, 7.0]

    var model = Lasso(alpha=1e-5, fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 1.0, rtol=1e-3)
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-3)
    assert_true(model.is_fitted)
    assert_equal(model.n_features_in_, 1)

    var preds = model.predict(X)
    assert_equal(len(preds), 4)
    assert_almost_equal(preds[0], 1.0, rtol=1e-3)
    assert_almost_equal(preds[3], 7.0, rtol=1e-3)


def test_lasso_multivariate_dense() raises:
    # y = 2.0*x0 - 1.5*x1 + 3.0*x2 + 5.0
    var X = Matrix[DType.float64](6, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[0, 2] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[1, 2] = 0.0
    X[2, 0] = 0.0
    X[2, 1] = 3.0
    X[2, 2] = 1.0
    X[3, 0] = 1.0
    X[3, 1] = 1.0
    X[3, 2] = 1.0
    X[4, 0] = 2.0
    X[4, 1] = 2.0
    X[4, 2] = 2.0
    X[5, 0] = 3.0
    X[5, 1] = 0.0
    X[5, 2] = 1.0

    var y = List[Scalar[DType.float64]](capacity=6)
    for i in range(6):
        var val = (
            2.0 * Float64(X[i, 0])
            - 1.5 * Float64(X[i, 1])
            + 3.0 * Float64(X[i, 2])
            + 5.0
        )
        y.append(val)

    var model = Lasso(alpha=1e-5, fit_intercept=True, max_iter=2000)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 5.0, rtol=1e-2)
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-2)
    assert_almost_equal(model.coef_[1], -1.5, rtol=1e-2)
    assert_almost_equal(model.coef_[2], 3.0, rtol=1e-2)


def test_lasso_sparsity_feature_selection() raises:
    # Target depends only on feature 0: y = 5.0 * x0
    var X = Matrix[DType.float64](6, 4, 0)
    X[0, 0] = -3.0
    X[0, 1] = 0.1
    X[0, 2] = -0.2
    X[0, 3] = 0.05
    X[1, 0] = -1.0
    X[1, 1] = -0.1
    X[1, 2] = 0.3
    X[1, 3] = -0.05
    X[2, 0] = 0.0
    X[2, 1] = 0.2
    X[2, 2] = -0.1
    X[2, 3] = 0.1
    X[3, 0] = 1.0
    X[3, 1] = -0.2
    X[3, 2] = 0.1
    X[3, 3] = -0.1
    X[4, 0] = 2.0
    X[4, 1] = 0.1
    X[4, 2] = -0.3
    X[4, 3] = 0.02
    X[5, 0] = 3.0
    X[5, 1] = -0.1
    X[5, 2] = 0.2
    X[5, 3] = -0.02

    var y: List[Scalar[DType.float64]] = [
        -15.0,
        -5.0,
        0.0,
        5.0,
        10.0,
        15.0,
    ]

    var model = Lasso(alpha=0.5, fit_intercept=False)
    model.fit(X, y)

    assert_true(model.coef_[0] > 4.0)
    assert_equal(model.coef_[1], 0.0)
    assert_equal(model.coef_[2], 0.0)
    assert_equal(model.coef_[3], 0.0)


def test_lasso_monotonic_shrinkage() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.5
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 1.5
    X[3, 0] = 4.0
    X[3, 1] = 2.0
    X[4, 0] = 5.0
    X[4, 1] = 2.5

    var y: List[Scalar[DType.float64]] = [3.0, 6.0, 9.0, 12.0, 15.0]

    var m1 = Lasso(alpha=0.01, fit_intercept=False)
    m1.fit(X, y)

    var m2 = Lasso(alpha=0.5, fit_intercept=False)
    m2.fit(X, y)

    var m3 = Lasso(alpha=2.0, fit_intercept=False)
    m3.fit(X, y)

    var norm1 = m1.coef_[0] + m1.coef_[1]
    var norm2 = m2.coef_[0] + m2.coef_[1]
    var norm3 = m3.coef_[0] + m3.coef_[1]

    assert_true(norm1 >= norm2)
    assert_true(norm2 >= norm3)


def test_lasso_extreme_alpha_zeros_all_weights() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [4.0, 6.0, 8.0, 10.0]

    var model = Lasso(alpha=100.0, fit_intercept=True)
    model.fit(X, y)

    assert_equal(model.coef_[0], 0.0)
    assert_equal(model.coef_[1], 0.0)
    assert_almost_equal(model.intercept_, 7.0, rtol=1e-4)


def test_lasso_zero_variance_constant_features() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 0.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0
    X[3, 0] = 4.0
    X[3, 1] = 0.0

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var model = Lasso(alpha=0.01, fit_intercept=False)
    model.fit(X, y)

    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-2)
    assert_equal(model.coef_[1], 0.0)


def test_lasso_no_intercept() raises:
    # y = 3.0*x1 + 4.0*x2
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0

    var y: List[Scalar[DType.float64]] = [11.0, 10.0, 9.0]

    var model = Lasso(alpha=1e-5, fit_intercept=False)
    model.fit(X, y)

    assert_equal(model.intercept_, 0.0)
    assert_almost_equal(model.coef_[0], 3.0, rtol=1e-3)
    assert_almost_equal(model.coef_[1], 4.0, rtol=1e-3)


def test_lasso_positive_constraint() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [-4.0, 1.0, -6.0, -1.0]

    var model = Lasso(alpha=0.01, fit_intercept=False, positive=True)
    model.fit(X, y)

    for j in range(len(model.coef_)):
        assert_true(model.coef_[j] >= 0.0)


def test_lasso_underdetermined_wide_matrix() raises:
    var N = 3
    var D = 5
    var X = Matrix[DType.float64](N, D, 0)
    for i in range(N):
        for j in range(D):
            X[i, j] = Float64((i + 1) * (j + 1))
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    var model = Lasso(alpha=0.1, fit_intercept=False)
    model.fit(X, y)
    assert_equal(len(model.coef_), D)
    var preds = model.predict(X)
    assert_equal(len(preds), N)


def test_lasso_float32() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    var y: List[Scalar[DType.float32]] = [1.0, 3.0, 5.0, 7.0]

    var model = Lasso[DType.float32](alpha=1e-4, fit_intercept=True)
    model.fit(X, y)
    assert_almost_equal(Float64(model.intercept_), 1.0, rtol=1e-2)
    assert_almost_equal(Float64(model.coef_[0]), 2.0, rtol=1e-2)


def test_lasso_dataset_container() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    var dataset = Dataset[DType.float64, DType.float64](X^, y^)
    var model = Lasso(alpha=1e-5, fit_intercept=False)
    fit(model, dataset)

    assert_true(model.is_fitted)
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-3)


def test_lasso_pipeline() raises:
    var scaler = StandardScaler()
    var lasso = Lasso(alpha=0.01)
    var pipe = PipelineRegressor(scaler^, lasso^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)
    assert_almost_equal(preds[0], 2.0, rtol=0.05)
    assert_almost_equal(preds[3], 8.0, rtol=0.05)


def test_lasso_copy_and_independence() raises:
    var m1 = Lasso(alpha=0.2, fit_intercept=True)
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]
    m1.fit(X, y)

    var m2 = Lasso(copy=m1)
    assert_equal(m2.is_fitted, True)
    assert_almost_equal(m2.alpha, 0.2)
    assert_almost_equal(m2.coef_[0], m1.coef_[0])
    assert_almost_equal(m2.intercept_, m1.intercept_)


def test_lasso_invalid_parameters() raises:
    with assert_raises():
        _ = Lasso(alpha=-1.0)

    with assert_raises():
        _ = Lasso(max_iter=0)

    with assert_raises():
        _ = Lasso(tol=-0.1)


def test_lasso_not_fitted_and_mismatch() raises:
    var model = Lasso()
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = model.predict(X)

    var y: List[Scalar[DType.float64]] = [1.0, 2.0]
    model.fit(X, y)

    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = model.predict(X_wrong)


def test_elastic_net_1d_fit_intercept() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0

    var y: List[Scalar[DType.float64]] = [1.0, 3.0, 5.0, 7.0]

    var model = ElasticNet(alpha=1e-5, l1_ratio=0.5, fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 1.0, rtol=1e-3)
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-3)
    assert_true(model.is_fitted)


def test_elastic_net_multivariate() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0
    X[3, 0] = 0.0
    X[3, 1] = 3.0
    X[4, 0] = 2.0
    X[4, 1] = 2.0

    # y = 3.0*x0 + 4.0*x1 + 1.0
    var y: List[Scalar[DType.float64]] = [12.0, 11.0, 10.0, 13.0, 15.0]

    var model = ElasticNet(alpha=1e-4, l1_ratio=0.7, fit_intercept=True)
    model.fit(X, y)

    assert_almost_equal(model.intercept_, 1.0, rtol=1e-2)
    assert_almost_equal(model.coef_[0], 3.0, rtol=1e-2)
    assert_almost_equal(model.coef_[1], 4.0, rtol=1e-2)


def test_elastic_net_mixing_extremes() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0

    var y: List[Scalar[DType.float64]] = [1.0, 3.0, 5.0, 7.0]

    var en_lasso = ElasticNet(alpha=1e-5, l1_ratio=1.0, fit_intercept=True)
    en_lasso.fit(X, y)
    assert_almost_equal(en_lasso.intercept_, 1.0, rtol=1e-3)
    assert_almost_equal(en_lasso.coef_[0], 2.0, rtol=1e-3)

    var en_ridge = ElasticNet(alpha=1e-5, l1_ratio=0.0, fit_intercept=True)
    en_ridge.fit(X, y)
    assert_almost_equal(en_ridge.intercept_, 1.0, rtol=1e-3)
    assert_almost_equal(en_ridge.coef_[0], 2.0, rtol=1e-3)

    var en_half = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True)
    en_half.fit(X, y)
    assert_true(en_half.coef_[0] > 0.0)
    assert_true(en_half.coef_[0] < 2.0)


def test_elastic_net_correlated_features_grouping() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.01
    X[1, 0] = 2.0
    X[1, 1] = 1.99
    X[2, 0] = 3.0
    X[2, 1] = 3.02
    X[3, 0] = 4.0
    X[3, 1] = 3.98
    X[4, 0] = 5.0
    X[4, 1] = 5.01

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0]

    var model = ElasticNet(alpha=0.1, l1_ratio=0.2, fit_intercept=False)
    model.fit(X, y)

    assert_true(model.coef_[0] > 0.0)
    assert_true(model.coef_[1] > 0.0)


def test_elastic_net_positive_constraint() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [-5.0, 0.0, -8.0, -2.0]

    var model = ElasticNet(
        alpha=0.01, l1_ratio=0.5, fit_intercept=False, positive=True
    )
    model.fit(X, y)

    for j in range(len(model.coef_)):
        assert_true(model.coef_[j] >= 0.0)


def test_elastic_net_extreme_alpha() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0

    var y: List[Scalar[DType.float64]] = [4.0, 6.0, 8.0, 10.0]

    var model = ElasticNet(alpha=100.0, l1_ratio=0.5, fit_intercept=True)
    model.fit(X, y)

    assert_equal(model.coef_[0], 0.0)
    assert_equal(model.coef_[1], 0.0)
    assert_almost_equal(model.intercept_, 7.0, rtol=1e-4)


def test_elastic_net_zero_variance_features() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 0.0
    X[2, 0] = 3.0
    X[2, 1] = 0.0
    X[3, 0] = 4.0
    X[3, 1] = 0.0

    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var model = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=False)
    model.fit(X, y)

    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-2)
    assert_equal(model.coef_[1], 0.0)


def test_elastic_net_underdetermined_wide_matrix() raises:
    var N = 3
    var D = 5
    var X = Matrix[DType.float64](N, D, 0)
    for i in range(N):
        for j in range(D):
            X[i, j] = Float64((i + 1) * (j + 1))
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    var model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False)
    model.fit(X, y)
    assert_equal(len(model.coef_), D)
    var preds = model.predict(X)
    assert_equal(len(preds), N)


def test_elastic_net_float32() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0
    var y: List[Scalar[DType.float32]] = [1.0, 3.0, 5.0, 7.0]

    var model = ElasticNet[DType.float32](
        alpha=1e-4, l1_ratio=0.5, fit_intercept=True
    )
    model.fit(X, y)
    assert_almost_equal(Float64(model.intercept_), 1.0, rtol=1e-2)
    assert_almost_equal(Float64(model.coef_[0]), 2.0, rtol=1e-2)


def test_elastic_net_dataset_container() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    var dataset = Dataset[DType.float64, DType.float64](X^, y^)
    var model = ElasticNet(alpha=1e-5, l1_ratio=0.5, fit_intercept=False)
    fit(model, dataset)

    assert_true(model.is_fitted)
    assert_almost_equal(model.coef_[0], 2.0, rtol=1e-3)


def test_elastic_net_invalid_parameters() raises:
    with assert_raises():
        _ = ElasticNet(alpha=-0.5)

    with assert_raises():
        _ = ElasticNet(l1_ratio=-0.1)

    with assert_raises():
        _ = ElasticNet(l1_ratio=1.1)

    with assert_raises():
        _ = ElasticNet(max_iter=0)

    with assert_raises():
        _ = ElasticNet(tol=-1e-4)


def test_elastic_net_copy() raises:
    var m1 = ElasticNet(alpha=0.2, l1_ratio=0.8)
    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]
    m1.fit(X, y)

    var m2 = ElasticNet(copy=m1)
    assert_equal(m2.is_fitted, True)
    assert_almost_equal(m2.alpha, 0.2)
    assert_almost_equal(m2.l1_ratio, 0.8)
    assert_almost_equal(m2.coef_[0], m1.coef_[0])
    assert_almost_equal(m2.intercept_, m1.intercept_)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
