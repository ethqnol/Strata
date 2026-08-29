from std.math import sqrt, pow, nan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    euclidean_distance,
    sqeuclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
    cosine_distance,
    pairwise_distances,
)


def test_euclidean_and_sqeuclidean_distance() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0

    var d_sq = sqeuclidean_distance(X, 0, X, 1)
    var d = euclidean_distance(X, 0, X, 1)

    assert_almost_equal(d_sq, 25.0, atol=1e-12)
    assert_almost_equal(d, 5.0, atol=1e-12)


def test_manhattan_distance() raises:
    var X = Matrix[DType.float64](1, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[0, 2] = 3.0

    var Y = Matrix[DType.float64](1, 3, 0)
    Y[0, 0] = 4.0
    Y[0, 1] = 0.0
    Y[0, 2] = -1.0

    # |1 - 4| + |2 - 0| + |3 - (-1)| = 3 + 2 + 4 = 9.0
    var d = manhattan_distance(X, 0, Y, 0)
    assert_almost_equal(d, 9.0, atol=1e-12)


def test_chebyshev_distance() raises:
    var X = Matrix[DType.float64](1, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[0, 2] = 3.0

    var Y = Matrix[DType.float64](1, 3, 0)
    Y[0, 0] = 4.0
    Y[0, 1] = 0.0
    Y[0, 2] = -1.0

    # max(|1 - 4|, |2 - 0|, |3 - (-1)|) = max(3, 2, 4) = 4.0
    var d = chebyshev_distance(X, 0, Y, 0)
    assert_almost_equal(d, 4.0, atol=1e-12)


def test_minkowski_distance() raises:
    var X = Matrix[DType.float64](1, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[0, 2] = 3.0

    var Y = Matrix[DType.float64](1, 3, 0)
    Y[0, 0] = 4.0
    Y[0, 1] = 0.0
    Y[0, 2] = -1.0

    # p=1.0 -> 9.0
    var d1 = minkowski_distance(X, 0, Y, 0, p=1.0)
    assert_almost_equal(d1, 9.0, atol=1e-12)

    # p=2.0 -> sqrt(3^2 + 2^2 + 4^2) = sqrt(29)
    var d2 = minkowski_distance(X, 0, Y, 0, p=2.0)
    assert_almost_equal(d2, sqrt(Float64(29.0)), atol=1e-12)

    # p=3.0 -> (27 + 8 + 64)^(1/3) = 99^(1/3)
    var d3 = minkowski_distance(X, 0, Y, 0, p=3.0)
    assert_almost_equal(d3, pow(Float64(99.0), 1.0 / 3.0), atol=1e-12)


def test_cosine_distance() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    # Orthogonal: [1, 0] and [0, 1] -> cos = 0 -> dist = 1.0
    X[0, 0] = 1.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0

    # Parallel: [1, 2] and [2, 4] -> cos = 1 -> dist = 0.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 4.0

    var d_ortho = cosine_distance(X, 0, X, 1)
    var d_paral = cosine_distance(X, 2, X, 3)

    assert_almost_equal(d_ortho, 1.0, atol=1e-12)
    assert_almost_equal(d_paral, 0.0, atol=1e-12)

    # Anti-parallel: [1, 0] and [-1, 0] -> cos = -1 -> dist = 2.0
    var Y = Matrix[DType.float64](1, 2, 0)
    Y[0, 0] = -1.0
    Y[0, 1] = 0.0
    var d_anti = cosine_distance(X, 0, Y, 0)
    assert_almost_equal(d_anti, 2.0, atol=1e-12)

    # Zero vector vs non-zero vector -> dist = 1.0 (similarity = 0.0)
    var Z = Matrix[DType.float64](2, 2, 0)
    Z[0, 0] = 0.0
    Z[0, 1] = 0.0
    Z[1, 0] = 5.0
    Z[1, 1] = 5.0
    var d_zero_nonzero = cosine_distance(Z, 0, Z, 1)
    assert_almost_equal(d_zero_nonzero, 1.0, atol=1e-12)

    # Zero vector vs zero vector -> dist = 0.0 (identical)
    var d_zero_zero = cosine_distance(Z, 0, Z, 0)
    assert_almost_equal(d_zero_zero, 0.0, atol=1e-12)


def test_pairwise_distances_two_matrices() raises:
    var X = Matrix[DType.float64](2, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 1.0
    X[1, 1] = 1.0

    var Y = Matrix[DType.float64](3, 2, 0)
    Y[0, 0] = 0.0
    Y[0, 1] = 0.0
    Y[1, 0] = 1.0
    Y[1, 1] = 0.0
    Y[2, 0] = 0.0
    Y[2, 1] = 1.0

    var D_euc = pairwise_distances(X, Y, metric="euclidean")
    assert_equal(D_euc.rows, 2)
    assert_equal(D_euc.cols, 3)

    # X[0] vs Y:
    assert_almost_equal(D_euc[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(D_euc[0, 1], 1.0, atol=1e-12)
    assert_almost_equal(D_euc[0, 2], 1.0, atol=1e-12)

    # X[1] vs Y:
    assert_almost_equal(D_euc[1, 0], sqrt(Float64(2.0)), atol=1e-12)
    assert_almost_equal(D_euc[1, 1], 1.0, atol=1e-12)
    assert_almost_equal(D_euc[1, 2], 1.0, atol=1e-12)

    # Manhattan
    var D_man = pairwise_distances(X, Y, metric="manhattan")
    assert_almost_equal(D_man[1, 0], 2.0, atol=1e-12)


def test_pairwise_distances_self_symmetric() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 4.0
    X[1, 1] = 6.0
    X[2, 0] = 7.0
    X[2, 1] = 10.0

    var D = pairwise_distances(X, metric="euclidean")
    assert_equal(D.rows, 3)
    assert_equal(D.cols, 3)

    for i in range(3):
        assert_almost_equal(D[i, i], 0.0, atol=1e-12)
        for j in range(3):
            assert_almost_equal(D[i, j], D[j, i], atol=1e-12)


def test_distance_float32() raises:
    var X = Matrix[DType.float32](2, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 3.0
    X[1, 1] = 4.0

    var d = euclidean_distance(X, 0, X, 1)
    assert_almost_equal(d, 5.0, atol=1e-6)

    var D = pairwise_distances(X, metric="euclidean")
    assert_almost_equal(D[0, 1], 5.0, atol=1e-6)
    assert_almost_equal(D[1, 0], 5.0, atol=1e-6)


def test_distance_invalid_parameters_and_errors() raises:
    var X = Matrix[DType.float64](2, 2, 1.0)
    var Y = Matrix[DType.float64](2, 3, 1.0)

    # Column dimension mismatch
    with assert_raises():
        _ = pairwise_distances(X, Y)

    # Invalid metric name
    with assert_raises():
        _ = pairwise_distances(X, metric="unsupported_metric")

    # Minkowski p < 1.0
    with assert_raises():
        _ = pairwise_distances(X, metric="minkowski", p=0.5)

    # NaN in input
    var nan_val = nan[DType.float64]()
    var X_nan = Matrix[DType.float64](2, 2, nan_val)
    with assert_raises():
        _ = pairwise_distances(X_nan)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
