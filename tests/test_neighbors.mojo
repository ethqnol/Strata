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
    Dataset,
    euclidean_distance,
    sqeuclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
    cosine_distance,
    pairwise_distances,
    NearestNeighbors,
    KNeighborsClassifier,
    KNeighborsRegressor,
    KDTree,
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


def test_nearest_neighbors_basic_kneighbors() raises:
    # 1D points on a line
    var X_train = Matrix[DType.float64](6, 1, 0)
    X_train[0, 0] = 0.0
    X_train[1, 0] = 1.0
    X_train[2, 0] = 2.0
    X_train[3, 0] = 3.0
    X_train[4, 0] = 10.0
    X_train[5, 0] = 12.0

    var nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X_train)

    assert_true(nn.is_fitted)
    assert_equal(nn.n_samples_fit_, 6)
    assert_equal(nn.n_features_in_, 1)

    var X_query = Matrix[DType.float64](1, 1, 2.2)
    var res = nn.kneighbors(X_query)
    var dists = res[0].copy()
    var idxs = res[1].copy()

    assert_equal(dists.rows, 1)
    assert_equal(dists.cols, 3)
    assert_equal(idxs.rows, 1)
    assert_equal(idxs.cols, 3)

    # Closest: index 2 (val 2.0, dist 0.2), index 3 (val 3.0, dist 0.8), index 1 (val 1.0, dist 1.2)
    assert_equal(Int(idxs[0, 0]), 2)
    assert_equal(Int(idxs[0, 1]), 3)
    assert_equal(Int(idxs[0, 2]), 1)

    assert_almost_equal(dists[0, 0], 0.2, atol=1e-12)
    assert_almost_equal(dists[0, 1], 0.8, atol=1e-12)
    assert_almost_equal(dists[0, 2], 1.2, atol=1e-12)


def test_nearest_neighbors_self_query() raises:
    var X_train = Matrix[DType.float64](4, 2, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 0.0
    X_train[1, 1] = 1.0
    X_train[2, 0] = 1.0
    X_train[2, 1] = 0.0
    X_train[3, 0] = 1.0
    X_train[3, 1] = 1.0

    var nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_train)

    var res = nn.kneighbors()
    var dists = res[0].copy()
    var idxs = res[1].copy()

    assert_equal(dists.rows, 4)
    assert_equal(dists.cols, 2)

    # 1st neighbor of every sample is itself with distance 0.0
    for i in range(4):
        assert_equal(Int(idxs[i, 0]), i)
        assert_almost_equal(dists[i, 0], 0.0, atol=1e-12)
        assert_almost_equal(dists[i, 1], 1.0, atol=1e-12)


def test_nearest_neighbors_radius_query() raises:
    var X_train = Matrix[DType.float64](5, 1, 0)
    X_train[0, 0] = 0.0
    X_train[1, 0] = 1.0
    X_train[2, 0] = 2.0
    X_train[3, 0] = 3.0
    X_train[4, 0] = 10.0

    var nn = NearestNeighbors(radius=1.5)
    nn.fit(X_train)

    var X_query = Matrix[DType.float64](1, 1, 2.0)
    var res = nn.radius_neighbors(X_query)
    var dists_list = res[0].copy()
    var idxs_list = res[1].copy()

    # Query 2.0 with radius 1.5 matches 2.0 (dist 0.0), 1.0 (dist 1.0), 3.0 (dist 1.0)
    assert_equal(len(idxs_list[0]), 3)
    assert_equal(idxs_list[0][0], 2)
    assert_almost_equal(dists_list[0][0], 0.0, atol=1e-12)


def test_nearest_neighbors_manhattan_and_chebyshev() raises:
    var X_train = Matrix[DType.float64](3, 2, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 2.0
    X_train[1, 1] = 2.0
    X_train[2, 0] = 5.0
    X_train[2, 1] = 0.0

    var query = Matrix[DType.float64](1, 2, 0)
    query[0, 0] = 1.0
    query[0, 1] = 1.0

    var nn_man = NearestNeighbors(n_neighbors=2, metric="manhattan")
    nn_man.fit(X_train)
    var res_man = nn_man.kneighbors(query)
    # dist to (0,0): |1-0|+|1-0| = 2.0 (index 0)
    # dist to (2,2): |1-2|+|1-2| = 2.0 (index 1)
    assert_almost_equal(res_man[0][0, 0], 2.0, atol=1e-12)
    assert_almost_equal(res_man[0][0, 1], 2.0, atol=1e-12)

    var nn_cheb = NearestNeighbors(n_neighbors=1, metric="chebyshev")
    nn_cheb.fit(X_train)
    var res_cheb = nn_cheb.kneighbors(query)
    # max(|1-0|, |1-0|) = 1.0
    assert_almost_equal(res_cheb[0][0, 0], 1.0, atol=1e-12)
    assert_equal(Int(res_cheb[1][0, 0]), 0)


def test_nearest_neighbors_float32() raises:
    var X_train = Matrix[DType.float32](3, 2, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 3.0
    X_train[1, 1] = 4.0
    X_train[2, 0] = 6.0
    X_train[2, 1] = 8.0

    var nn = NearestNeighbors[DType.float32](n_neighbors=2)
    nn.fit(X_train)

    var q = Matrix[DType.float32](1, 2, 0)
    var res = nn.kneighbors(q)
    assert_almost_equal(res[0][0, 0], 0.0, atol=1e-6)
    assert_almost_equal(res[0][0, 1], 5.0, atol=1e-6)
    assert_equal(Int(res[1][0, 0]), 0)
    assert_equal(Int(res[1][0, 1]), 1)


def test_nearest_neighbors_copy_semantics() raises:
    var X_train = Matrix[DType.float64](3, 2, 1.0)
    var nn1 = NearestNeighbors(n_neighbors=2)
    nn1.fit(X_train)

    var nn2 = NearestNeighbors(copy=nn1)
    assert_true(nn2.is_fitted)
    assert_equal(nn2.n_samples_fit_, 3)
    assert_equal(nn2.n_features_in_, 2)


def test_nearest_neighbors_invalid_parameters_and_errors() raises:
    # Invalid constructor parameters
    with assert_raises():
        _ = NearestNeighbors(n_neighbors=0)
    with assert_raises():
        _ = NearestNeighbors(radius=-1.0)
    with assert_raises():
        _ = NearestNeighbors(algorithm="unsupported")
    with assert_raises():
        _ = NearestNeighbors(metric="unsupported")

    var nn = NearestNeighbors(n_neighbors=2)
    var X = Matrix[DType.float64](2, 2, 1.0)

    # Unfitted query
    with assert_raises():
        _ = nn.kneighbors(X)
    with assert_raises():
        _ = nn.radius_neighbors(X)

    # n_neighbors > n_samples_fit_
    var X_small = Matrix[DType.float64](1, 2, 1.0)
    with assert_raises():
        nn.fit(X_small)

    nn.fit(X)

    # Dimension mismatch
    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = nn.kneighbors(X_wrong)

    # DataConversionError on dtype mismatch
    var X_f32 = Matrix[DType.float32](2, 2, 1.0)
    with assert_raises():
        _ = nn.kneighbors(X_f32)

    # Query with invalid k
    with assert_raises():
        _ = nn.kneighbors(X, n_neighbors=5)
    with assert_raises():
        _ = nn.kneighbors(X, n_neighbors=0)

    # Invalid radius in radius_neighbors
    with assert_raises():
        _ = nn.radius_neighbors(X, radius=-5.0)
    with assert_raises():
        _ = nn.radius_neighbors(X, radius=0.0)


def test_nearest_neighbors_float32_self_query() raises:
    var X_train = Matrix[DType.float32](3, 2, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 1.0
    X_train[1, 1] = 1.0
    X_train[2, 0] = 2.0
    X_train[2, 1] = 2.0

    var nn = NearestNeighbors[DType.float32](n_neighbors=2)
    nn.fit(X_train)

    var res = nn.kneighbors[DType.float32]()
    assert_equal(res[0].rows, 3)
    assert_equal(res[0].cols, 2)
    assert_equal(Int(res[1][0, 0]), 0)
    assert_equal(Int(res[1][1, 0]), 1)
    assert_equal(Int(res[1][2, 0]), 2)


def test_kneighbors_classifier_binary_uniform() raises:
    var X_train = Matrix[DType.float64](6, 2, 0)
    # Class 0: near (0, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 0.5
    X_train[1, 1] = 0.5
    X_train[2, 0] = 1.0
    X_train[2, 1] = 0.0

    # Class 1: near (5, 5)
    X_train[3, 0] = 5.0
    X_train[3, 1] = 5.0
    X_train[4, 0] = 5.5
    X_train[4, 1] = 5.0
    X_train[5, 0] = 4.5
    X_train[5, 1] = 5.5

    var y_train = List[Float64]()
    y_train.append(0.0)
    y_train.append(0.0)
    y_train.append(0.0)
    y_train.append(1.0)
    y_train.append(1.0)
    y_train.append(1.0)

    var clf = KNeighborsClassifier(n_neighbors=3, weights="uniform")
    clf.fit[DType.float64, DType.float64](X_train, y_train)

    assert_true(clf.is_fitted)
    assert_equal(clf.n_classes_, 2)
    assert_equal(clf.classes_[0], 0)
    assert_equal(clf.classes_[1], 1)

    var X_query = Matrix[DType.float64](2, 2, 0)
    X_query[0, 0] = 0.2
    X_query[0, 1] = 0.1
    X_query[1, 0] = 5.1
    X_query[1, 1] = 4.9

    var preds = clf.predict(X_query)
    assert_equal(len(preds), 2)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 1)

    var proba = clf.predict_proba(X_query)
    assert_equal(proba.rows, 2)
    assert_equal(proba.cols, 2)
    assert_almost_equal(proba[0, 0], 1.0, atol=1e-12)
    assert_almost_equal(proba[0, 1], 0.0, atol=1e-12)
    assert_almost_equal(proba[1, 0], 0.0, atol=1e-12)
    assert_almost_equal(proba[1, 1], 1.0, atol=1e-12)


def test_kneighbors_classifier_multiclass_distance_weighted() raises:
    var X_train = Matrix[DType.float64](4, 1, 0)
    X_train[0, 0] = 0.0   # Class 0
    X_train[1, 0] = 1.0   # Class 0
    X_train[2, 0] = 2.0   # Class 1
    X_train[3, 0] = 10.0  # Class 2

    var y_train = List[Float64]()
    y_train.append(0.0)
    y_train.append(0.0)
    y_train.append(1.0)
    y_train.append(2.0)

    var clf = KNeighborsClassifier(n_neighbors=2, weights="distance")
    clf.fit[DType.float64, DType.float64](X_train, y_train)

    assert_equal(clf.n_classes_, 3)

    # Query 1.9 is distance 0.1 to sample 2 (Class 1) and distance 0.9 to sample 1 (Class 0)
    var q = Matrix[DType.float64](1, 1, 1.9)
    var preds = clf.predict(q)
    assert_equal(preds[0], 1)

    var proba = clf.predict_proba(q)
    assert_true(proba[0, 1] > proba[0, 0])
    assert_almost_equal(proba[0, 2], 0.0, atol=1e-12)


def test_kneighbors_classifier_exact_match_zero_distance() raises:
    var X_train = Matrix[DType.float64](3, 2, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 10.0
    X_train[1, 1] = 10.0
    X_train[2, 0] = 20.0
    X_train[2, 1] = 20.0

    var y_train = List[Float64]()
    y_train.append(0.0)
    y_train.append(1.0)
    y_train.append(2.0)

    var clf = KNeighborsClassifier(n_neighbors=2, weights="distance")
    clf.fit[DType.float64, DType.float64](X_train, y_train)

    # Query exactly matches sample 1: (10, 10)
    var q = Matrix[DType.float64](1, 2, 0)
    q[0, 0] = 10.0
    q[0, 1] = 10.0

    var proba = clf.predict_proba(q)
    assert_almost_equal(proba[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(proba[0, 1], 1.0, atol=1e-12)
    assert_almost_equal(proba[0, 2], 0.0, atol=1e-12)

    var preds = clf.predict(q)
    assert_equal(preds[0], 1)


def test_kneighbors_classifier_dataset_overload() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 0.0
    X[1, 0] = 0.0
    X[1, 1] = 1.0
    X[2, 0] = 10.0
    X[2, 1] = 10.0
    X[3, 0] = 10.0
    X[3, 1] = 11.0

    var y = List[Float64]()
    y.append(0.0)
    y.append(0.0)
    y.append(1.0)
    y.append(1.0)

    var ds = Dataset[DType.float64, DType.float64](X^, y^)
    var clf = KNeighborsClassifier(n_neighbors=2)
    clf.fit(ds)

    var preds = clf.predict(ds)
    assert_equal(preds[0], 0)
    assert_equal(preds[1], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[3], 1)


def test_kneighbors_classifier_float32() raises:
    var X = Matrix[DType.float32](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 0.5
    X[2, 0] = 5.0
    X[3, 0] = 5.5

    var y = List[Scalar[DType.float32]]()
    y.append(0.0)
    y.append(0.0)
    y.append(1.0)
    y.append(1.0)

    var clf = KNeighborsClassifier[DType.float32](n_neighbors=2)
    clf.fit[DType.float32, DType.float32](X, y)

    var q = Matrix[DType.float32](1, 1, 0.1)
    var preds = clf.predict(q)
    assert_equal(preds[0], 0)


def test_kneighbors_classifier_copy_semantics() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    var y = List[Float64]()
    y.append(0.0)
    y.append(1.0)
    y.append(0.0)

    var clf1 = KNeighborsClassifier(n_neighbors=2)
    clf1.fit[DType.float64, DType.float64](X, y)

    var clf2 = KNeighborsClassifier(copy=clf1)
    assert_true(clf2.is_fitted)
    assert_equal(clf2.n_classes_, 2)
    assert_equal(clf2.n_samples_fit_, 3)


def test_kneighbors_classifier_invalid_parameters_and_errors() raises:
    with assert_raises():
        _ = KNeighborsClassifier(n_neighbors=0)
    with assert_raises():
        _ = KNeighborsClassifier(weights="invalid_weights")

    var clf = KNeighborsClassifier(n_neighbors=2)
    var X = Matrix[DType.float64](3, 2, 1.0)
    var y = List[Float64]()
    y.append(0.0)
    y.append(1.0)

    # Inconsistent length (3 rows vs 2 targets)
    with assert_raises():
        clf.fit[DType.float64, DType.float64](X, y)

    y.append(0.0)
    # Unfitted predict
    with assert_raises():
        _ = clf.predict(X)

    clf.fit[DType.float64, DType.float64](X, y)

    # Dimension mismatch
    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = clf.predict(X_wrong)

    # Dtype mismatch
    var X_f32 = Matrix[DType.float32](2, 2, 1.0)
    with assert_raises():
        _ = clf.predict(X_f32)


def test_kneighbors_regressor_uniform() raises:
    var X_train = Matrix[DType.float64](4, 1, 0)
    X_train[0, 0] = 0.0
    X_train[1, 0] = 1.0
    X_train[2, 0] = 2.0
    X_train[3, 0] = 3.0

    var y_train = List[Float64]()
    y_train.append(0.0)
    y_train.append(2.0)
    y_train.append(4.0)
    y_train.append(6.0)

    var reg = KNeighborsRegressor(n_neighbors=2, weights="uniform")
    reg.fit[DType.float64, DType.float64](X_train, y_train)

    assert_true(reg.is_fitted)
    assert_equal(reg.n_samples_fit_, 4)
    assert_equal(reg.n_features_in_, 1)

    # Query 1.5: 2 nearest neighbors are (1.0, 2.0) and (2.0, 4.0) -> mean is 3.0
    var q = Matrix[DType.float64](1, 1, 1.5)
    var preds = reg.predict(q)
    assert_equal(len(preds), 1)
    assert_almost_equal(preds[0], 3.0, atol=1e-12)


def test_kneighbors_regressor_distance_weighted() raises:
    var X_train = Matrix[DType.float64](3, 1, 0)
    X_train[0, 0] = 0.0
    X_train[1, 0] = 1.0
    X_train[2, 0] = 2.0

    var y_train = List[Float64]()
    y_train.append(0.0)
    y_train.append(10.0)
    y_train.append(100.0)

    var reg = KNeighborsRegressor(n_neighbors=2, weights="distance")
    reg.fit[DType.float64, DType.float64](X_train, y_train)

    # Query 0.9: dist to sample 1 is 0.1 (w=10), dist to sample 0 is 0.9 (w=1.111111)
    # y = (10*10 + 1.111111*0) / (10 + 1.111111) = 100 / 11.111111 = 9.0
    var q = Matrix[DType.float64](1, 1, 0.9)
    var preds = reg.predict(q)
    assert_almost_equal(preds[0], 9.0, atol=1e-12)


def test_kneighbors_regressor_exact_match_zero_distance() raises:
    var X_train = Matrix[DType.float64](3, 2, 0)
    X_train[0, 0] = 0.0
    X_train[0, 1] = 0.0
    X_train[1, 0] = 10.0
    X_train[1, 1] = 10.0
    X_train[2, 0] = 20.0
    X_train[2, 1] = 20.0

    var y_train = List[Float64]()
    y_train.append(5.0)
    y_train.append(50.0)
    y_train.append(500.0)

    var reg = KNeighborsRegressor(n_neighbors=2, weights="distance")
    reg.fit[DType.float64, DType.float64](X_train, y_train)

    # Query exactly matches sample 1: (10, 10)
    var q = Matrix[DType.float64](1, 2, 0)
    q[0, 0] = 10.0
    q[0, 1] = 10.0

    var preds = reg.predict(q)
    assert_almost_equal(preds[0], 50.0, atol=1e-12)


def test_kneighbors_regressor_dataset_overload() raises:
    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0
    X[3, 0] = 3.0

    var y = List[Float64]()
    y.append(10.0)
    y.append(20.0)
    y.append(30.0)
    y.append(40.0)

    var ds = Dataset[DType.float64, DType.float64](X^, y^)
    var reg = KNeighborsRegressor(n_neighbors=2)
    reg.fit(ds)

    var preds = reg.predict(ds)
    assert_equal(len(preds), 4)
    assert_almost_equal(preds[0], 15.0, atol=1e-12)  # (10 + 20)/2


def test_kneighbors_regressor_float32() raises:
    var X = Matrix[DType.float32](3, 1, 0)
    X[0, 0] = 0.0
    X[1, 0] = 1.0
    X[2, 0] = 2.0

    var y = List[Scalar[DType.float32]]()
    y.append(0.0)
    y.append(10.0)
    y.append(20.0)

    var reg = KNeighborsRegressor[DType.float32](n_neighbors=2)
    reg.fit[DType.float32, DType.float32](X, y)

    var q = Matrix[DType.float32](1, 1, 1.5)
    var preds = reg.predict(q)
    assert_almost_equal(preds[0], 15.0, atol=1e-6)


def test_kneighbors_regressor_copy_semantics() raises:
    var X = Matrix[DType.float64](3, 1, 0)
    var y = List[Float64]()
    y.append(1.0)
    y.append(2.0)
    y.append(3.0)

    var reg1 = KNeighborsRegressor(n_neighbors=2)
    reg1.fit[DType.float64, DType.float64](X, y)

    var reg2 = KNeighborsRegressor(copy=reg1)
    assert_true(reg2.is_fitted)
    assert_equal(reg2.n_samples_fit_, 3)


def test_kneighbors_regressor_invalid_parameters_and_errors() raises:
    with assert_raises():
        _ = KNeighborsRegressor(n_neighbors=0)
    with assert_raises():
        _ = KNeighborsRegressor(weights="unsupported")
    with assert_raises():
        _ = KNeighborsRegressor(metric="minkowski", p=0.5)

    var reg = KNeighborsRegressor(n_neighbors=2)
    var X = Matrix[DType.float64](3, 2, 1.0)
    var y = List[Float64]()
    y.append(1.0)
    y.append(2.0)

    # Inconsistent length
    with assert_raises():
        reg.fit[DType.float64, DType.float64](X, y)

    y.append(3.0)
    # Unfitted predict
    with assert_raises():
        _ = reg.predict(X)

    reg.fit[DType.float64, DType.float64](X, y)

    # Dimension mismatch
    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = reg.predict(X_wrong)

    # Dtype mismatch
    var X_f32 = Matrix[DType.float32](2, 2, 1.0)
    with assert_raises():
        _ = reg.predict(X_f32)


def test_kd_tree_query_basic() raises:
    var data = Matrix[DType.float64](6, 2, 0)
    data[0, 0] = 0.0
    data[0, 1] = 0.0
    data[1, 0] = 1.0
    data[1, 1] = 1.0
    data[2, 0] = 2.0
    data[2, 1] = 2.0
    data[3, 0] = 10.0
    data[3, 1] = 10.0
    data[4, 0] = 11.0
    data[4, 1] = 11.0
    data[5, 0] = 12.0
    data[5, 1] = 12.0

    var tree = KDTree(data, metric="euclidean")
    assert_equal(tree.n_samples_, 6)
    assert_equal(tree.n_features_, 2)

    var query = Matrix[DType.float64](1, 2, 0)
    query[0, 0] = 1.8
    query[0, 1] = 2.1

    var res = tree.query(query, k=2)
    var dists = res[0].copy()
    var idxs = res[1].copy()

    assert_equal(dists.rows, 1)
    assert_equal(dists.cols, 2)
    assert_equal(idxs.rows, 1)
    assert_equal(idxs.cols, 2)

    # Closest should be (2, 2) at index 2, then (1, 1) at index 1
    assert_equal(Int(idxs[0, 0]), 2)
    assert_equal(Int(idxs[0, 1]), 1)


def test_kd_tree_query_radius() raises:
    var data = Matrix[DType.float64](5, 2, 0)
    data[0, 0] = 0.0
    data[0, 1] = 0.0
    data[1, 0] = 1.0
    data[1, 1] = 1.0
    data[2, 0] = 2.0
    data[2, 1] = 2.0
    data[3, 0] = 10.0
    data[3, 1] = 10.0
    data[4, 0] = 20.0
    data[4, 1] = 20.0

    var tree = KDTree(data, metric="euclidean")

    var query = Matrix[DType.float64](1, 2, 0)
    query[0, 0] = 2.0
    query[0, 1] = 2.0

    var res = tree.query_radius(query, r=2.0)
    var dists_list = res[0].copy()
    var idxs_list = res[1].copy()

    # Matches (2,2) at index 2 (dist 0.0) and (1,1) at index 1 (dist sqrt(2))
    assert_equal(len(idxs_list[0]), 2)
    assert_equal(idxs_list[0][0], 2)
    assert_equal(idxs_list[0][1], 1)
    assert_almost_equal(dists_list[0][0], 0.0, atol=1e-12)
    assert_almost_equal(dists_list[0][1], sqrt(Float64(2.0)), atol=1e-12)


def test_kd_tree_manhattan_and_chebyshev() raises:
    var data = Matrix[DType.float64](4, 2, 0)
    data[0, 0] = 0.0
    data[0, 1] = 0.0
    data[1, 0] = 2.0
    data[1, 1] = 2.0
    data[2, 0] = 5.0
    data[2, 1] = 5.0
    data[3, 0] = 10.0
    data[3, 1] = 10.0

    var tree_man = KDTree(data, metric="manhattan")
    var tree_cheb = KDTree(data, metric="chebyshev")

    var query = Matrix[DType.float64](1, 2, 0)
    query[0, 0] = 1.0
    query[0, 1] = 1.0

    var res_man = tree_man.query(query, k=2)
    # dist to (0,0) is |1-0|+|1-0|=2, dist to (2,2) is |1-2|+|1-2|=2
    assert_almost_equal(res_man[0][0, 0], 2.0, atol=1e-12)
    assert_almost_equal(res_man[0][0, 1], 2.0, atol=1e-12)

    var res_cheb = tree_cheb.query(query, k=1)
    # max(|1-0|, |1-0|) = 1.0
    assert_almost_equal(res_cheb[0][0, 0], 1.0, atol=1e-12)


def test_kd_tree_float32() raises:
    var data = Matrix[DType.float32](3, 2, 0)
    data[0, 0] = 0.0
    data[0, 1] = 0.0
    data[1, 0] = 3.0
    data[1, 1] = 4.0
    data[2, 0] = 6.0
    data[2, 1] = 8.0

    var tree = KDTree[DType.float32](data, metric="euclidean")
    var q = Matrix[DType.float32](1, 2, 0)
    var res = tree.query(q, k=2)

    assert_almost_equal(res[0][0, 0], 0.0, atol=1e-6)
    assert_almost_equal(res[0][0, 1], 5.0, atol=1e-6)
    assert_equal(Int(res[1][0, 0]), 0)
    assert_equal(Int(res[1][0, 1]), 1)


def test_kd_tree_copy_semantics() raises:
    var data = Matrix[DType.float64](3, 2, 1.0)
    var tree1 = KDTree(data)
    var tree2 = KDTree(copy=tree1)

    assert_equal(tree2.n_samples_, 3)
    assert_equal(tree2.n_features_, 2)


def test_kd_tree_invalid_parameters_and_errors() raises:
    var data = Matrix[DType.float64](3, 2, 1.0)

    with assert_raises():
        _ = KDTree(data, metric="unsupported")

    var tree = KDTree(data)

    # Invalid k
    with assert_raises():
        _ = tree.query(data, k=0)
    with assert_raises():
        _ = tree.query(data, k=10)

    # Invalid radius
    with assert_raises():
        _ = tree.query_radius(data, r=0.0)
    with assert_raises():
        _ = tree.query_radius(data, r=-1.0)

    # Dimension mismatch
    var q_wrong = Matrix[DType.float64](1, 3, 1.0)
    with assert_raises():
        _ = tree.query(q_wrong)
    with assert_raises():
        _ = tree.query_radius(q_wrong, r=1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
