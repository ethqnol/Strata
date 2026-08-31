from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from strata import Matrix, DimensionMismatchError, InvalidParameterError
from strata.ensemble._binning import (
    BinnedMatrix,
    _find_bin_idx,
    _compute_bin_thresholds,
    _map_to_bins,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
