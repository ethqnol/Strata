from std.math import isnan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)
from strata import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    silhouette_score,
)
from strata.core.matrix import Matrix


def test_mse_sklearn_reference() raises:
    var doc_t: List[Scalar[DType.float64]] = [3.0, -0.5, 2.0, 7.0]
    var doc_p: List[Scalar[DType.float64]] = [2.5, 0.0, 2.0, 8.0]
    assert_almost_equal(mean_squared_error(doc_t, doc_p), 0.375)

    var mixed_t: List[Scalar[DType.float64]] = [-5.0, -1.0, 0.0, 4.5]
    var mixed_p: List[Scalar[DType.float64]] = [0.0, -3.0, 2.0, 4.5]
    assert_almost_equal(mean_squared_error(mixed_t, mixed_p), 8.25)

    var mean_t: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var mean_p: List[Scalar[DType.float64]] = [2.0, 2.0, 2.0]
    assert_almost_equal(mean_squared_error(mean_t, mean_p), 0.6666666666666666)


def test_mse_perfect_prediction_is_zero() raises:
    var exact: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    assert_equal(mean_squared_error(exact, exact), 0.0)

    var const_t: List[Scalar[DType.float64]] = [5.0, 5.0, 5.0]
    var const_p: List[Scalar[DType.float64]] = [5.0, 5.0, 5.0]
    assert_equal(mean_squared_error(const_t, const_p), 0.0)


def test_mse_single_sample_and_symmetry() raises:
    var one_t: List[Scalar[DType.float64]] = [4.0]
    var one_p: List[Scalar[DType.float64]] = [1.0]
    assert_almost_equal(mean_squared_error(one_t, one_p), 9.0)

    var a: List[Scalar[DType.float64]] = [3.0, -0.5, 2.0, 7.0]
    var b: List[Scalar[DType.float64]] = [2.5, 0.0, 2.0, 8.0]
    assert_equal(mean_squared_error(a, b), mean_squared_error(b, a))


def test_mse_large_magnitudes() raises:
    var big_t: List[Scalar[DType.float64]] = [1e6, 2e6, 3e6]
    var big_p: List[Scalar[DType.float64]] = [1.1e6, 1.9e6, 3.2e6]
    assert_almost_equal(mean_squared_error(big_t, big_p), 20000000000.0)


def test_mse_dtype_variants() raises:
    var int_t: List[Scalar[DType.int32]] = [1, 2, 3, 4]
    var f32_p: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0, 5.0]
    assert_almost_equal(mean_squared_error(int_t, f32_p), 0.25)

    var f32_t: List[Scalar[DType.float32]] = [3.0, -0.5, 2.0, 7.0]
    var f32_pp: List[Scalar[DType.float32]] = [2.5, 0.0, 2.0, 8.0]
    assert_almost_equal(mean_squared_error(f32_t, f32_pp), 0.375)

    var bf_t: List[Scalar[DType.bfloat16]] = [1.0, 2.0, 3.0]
    var bf_p: List[Scalar[DType.bfloat16]] = [1.0, 2.0, 4.0]
    assert_almost_equal(mean_squared_error(bf_t, bf_p), 0.3333333333333333)

    var i64_t: List[Scalar[DType.int64]] = [10, 20]
    var i64_p: List[Scalar[DType.int64]] = [12, 18]
    assert_almost_equal(mean_squared_error(i64_t, i64_p), 4.0)


def test_mse_validation() raises:
    var y_true: List[Scalar[DType.float64]] = [1.0, 2.0]
    var short: List[Scalar[DType.float64]] = [1.0]
    with assert_raises():
        _ = mean_squared_error(y_true, short)

    var empty = List[Scalar[DType.float64]]()
    with assert_raises():
        _ = mean_squared_error(empty, empty)


def test_rmse_sklearn_reference() raises:
    var doc_t: List[Scalar[DType.float64]] = [3.0, -0.5, 2.0, 7.0]
    var doc_p: List[Scalar[DType.float64]] = [2.5, 0.0, 2.0, 8.0]
    assert_almost_equal(
        root_mean_squared_error(doc_t, doc_p), 0.6123724356957945
    )

    var mixed_t: List[Scalar[DType.float64]] = [-5.0, -1.0, 0.0, 4.5]
    var mixed_p: List[Scalar[DType.float64]] = [0.0, -3.0, 2.0, 4.5]
    assert_almost_equal(
        root_mean_squared_error(mixed_t, mixed_p), 2.8722813232690143
    )


def test_rmse_is_sqrt_of_mse() raises:
    var y_true: List[Scalar[DType.float64]] = [1.0, 4.0, -2.0, 7.5]
    var y_pred: List[Scalar[DType.float64]] = [1.5, 3.0, 0.0, 9.5]
    var rmse = root_mean_squared_error(y_true, y_pred)
    assert_almost_equal(rmse * rmse, mean_squared_error(y_true, y_pred))


def test_rmse_perfect_and_constant_offset() raises:
    var exact: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    assert_equal(root_mean_squared_error(exact, exact), 0.0)

    # A uniform offset of 3.0 gives an RMSE of exactly 3.0
    var a: List[Scalar[DType.float64]] = [0.0, 0.0]
    var b: List[Scalar[DType.float64]] = [3.0, 3.0]
    assert_almost_equal(root_mean_squared_error(a, b), 3.0)

    var one_t: List[Scalar[DType.float64]] = [4.0]
    var one_p: List[Scalar[DType.float64]] = [1.0]
    assert_almost_equal(root_mean_squared_error(one_t, one_p), 3.0)


def test_rmse_large_magnitudes() raises:
    var big_t: List[Scalar[DType.float64]] = [1e6, 2e6, 3e6]
    var big_p: List[Scalar[DType.float64]] = [1.1e6, 1.9e6, 3.2e6]
    assert_almost_equal(
        root_mean_squared_error(big_t, big_p), 141421.35623730952
    )


def test_rmse_dtype_variants_and_validation() raises:
    var int_t: List[Scalar[DType.int32]] = [1, 2, 3, 4]
    var f32_p: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0, 5.0]
    assert_almost_equal(root_mean_squared_error(int_t, f32_p), 0.5)

    var short: List[Scalar[DType.float32]] = [1.0]
    with assert_raises():
        _ = root_mean_squared_error(int_t, short)

    var empty = List[Scalar[DType.float64]]()
    with assert_raises():
        _ = root_mean_squared_error(empty, empty)


def test_mae_sklearn_reference() raises:
    var doc_t: List[Scalar[DType.float64]] = [3.0, -0.5, 2.0, 7.0]
    var doc_p: List[Scalar[DType.float64]] = [2.5, 0.0, 2.0, 8.0]
    assert_almost_equal(mean_absolute_error(doc_t, doc_p), 0.5)

    var mixed_t: List[Scalar[DType.float64]] = [-5.0, -1.0, 0.0, 4.5]
    var mixed_p: List[Scalar[DType.float64]] = [0.0, -3.0, 2.0, 4.5]
    assert_almost_equal(mean_absolute_error(mixed_t, mixed_p), 2.25)

    var mean_t: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var mean_p: List[Scalar[DType.float64]] = [2.0, 2.0, 2.0]
    assert_almost_equal(mean_absolute_error(mean_t, mean_p), 0.6666666666666666)


def test_mae_is_symmetric_and_nonnegative() raises:
    var a: List[Scalar[DType.float64]] = [-5.0, -1.0, 0.0, 4.5]
    var b: List[Scalar[DType.float64]] = [0.0, -3.0, 2.0, 4.5]
    assert_equal(mean_absolute_error(a, b), mean_absolute_error(b, a))

    # Errors are absolute, so opposite-signed residuals must not cancel
    var t: List[Scalar[DType.float64]] = [0.0, 0.0]
    var p: List[Scalar[DType.float64]] = [-4.0, 4.0]
    assert_almost_equal(mean_absolute_error(t, p), 4.0)


def test_mae_perfect_and_single_sample() raises:
    var exact: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    assert_equal(mean_absolute_error(exact, exact), 0.0)

    var one_t: List[Scalar[DType.float64]] = [4.0]
    var one_p: List[Scalar[DType.float64]] = [1.0]
    assert_almost_equal(mean_absolute_error(one_t, one_p), 3.0)


def test_mae_never_exceeds_rmse() raises:
    var y_true: List[Scalar[DType.float64]] = [3.0, -0.5, 2.0, 7.0]
    var y_pred: List[Scalar[DType.float64]] = [2.5, 0.0, 2.0, 8.0]
    var mae = mean_absolute_error(y_true, y_pred)
    var rmse = root_mean_squared_error(y_true, y_pred)
    assert_equal(mae <= rmse, True)

    var big_t: List[Scalar[DType.float64]] = [1e6, 2e6, 3e6]
    var big_p: List[Scalar[DType.float64]] = [1.1e6, 1.9e6, 3.2e6]
    assert_almost_equal(mean_absolute_error(big_t, big_p), 133333.33333333334)


def test_mae_dtype_variants_and_validation() raises:
    var int_t: List[Scalar[DType.int32]] = [1, 2, 3, 4]
    var f32_p: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0, 5.0]
    assert_almost_equal(mean_absolute_error(int_t, f32_p), 0.25)

    var i64_t: List[Scalar[DType.int64]] = [10, 20]
    var i64_p: List[Scalar[DType.int64]] = [12, 18]
    assert_almost_equal(mean_absolute_error(i64_t, i64_p), 2.0)

    var short: List[Scalar[DType.float32]] = [1.0]
    with assert_raises():
        _ = mean_absolute_error(int_t, short)

    var empty = List[Scalar[DType.float64]]()
    with assert_raises():
        _ = mean_absolute_error(empty, empty)


def test_r2_sklearn_reference() raises:
    var doc_t: List[Scalar[DType.float64]] = [3.0, -0.5, 2.0, 7.0]
    var doc_p: List[Scalar[DType.float64]] = [2.5, 0.0, 2.0, 8.0]
    assert_almost_equal(r2_score(doc_t, doc_p), 0.9486081370449679)

    var mixed_t: List[Scalar[DType.float64]] = [-5.0, -1.0, 0.0, 4.5]
    var mixed_p: List[Scalar[DType.float64]] = [0.0, -3.0, 2.0, 4.5]
    assert_almost_equal(r2_score(mixed_t, mixed_p), 0.2777017783857729)

    var big_t: List[Scalar[DType.float64]] = [1e6, 2e6, 3e6]
    var big_p: List[Scalar[DType.float64]] = [1.1e6, 1.9e6, 3.2e6]
    assert_almost_equal(r2_score(big_t, big_p), 0.97)


def test_r2_perfect_is_one_and_mean_predictor_is_zero() raises:
    var exact: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    assert_almost_equal(r2_score(exact, exact), 1.0)

    var vals: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var mean_pred: List[Scalar[DType.float64]] = [2.0, 2.0, 2.0]
    assert_almost_equal(r2_score(vals, mean_pred), 0.0)


def test_r2_can_be_negative() raises:
    # Inverted predictions score far worse than predicting the mean
    var y_true: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var y_pred: List[Scalar[DType.float64]] = [3.0, 2.0, 1.0]
    assert_almost_equal(r2_score(y_true, y_pred), -3.0)

    var tiny_t: List[Scalar[DType.float64]] = [1e-8, 2e-8]
    var tiny_p: List[Scalar[DType.float64]] = [2e-8, 4e-8]
    assert_almost_equal(r2_score(tiny_t, tiny_p), -9.000000000000002)


def test_r2_constant_target() raises:
    var const_t: List[Scalar[DType.float64]] = [5.0, 5.0, 5.0]

    # Zero residual on a zero-variance target scores 1.0
    var exact_p: List[Scalar[DType.float64]] = [5.0, 5.0, 5.0]
    assert_equal(r2_score(const_t, exact_p), 1.0)

    # Any residual on a zero-variance target scores 0.0
    var off_p: List[Scalar[DType.float64]] = [5.0, 6.0, 5.0]
    assert_equal(r2_score(const_t, off_p), 0.0)


def test_r2_dtype_variants_and_validation() raises:
    var int_t: List[Scalar[DType.int32]] = [1, 2, 3, 4]
    var f32_p: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0, 5.0]
    assert_almost_equal(r2_score(int_t, f32_p), 0.8)

    # Fewer than two samples leaves the variance undefined
    var one_t: List[Scalar[DType.float64]] = [1.0]
    var one_p: List[Scalar[DType.float64]] = [1.0]
    with assert_raises():
        _ = r2_score(one_t, one_p)

    var short: List[Scalar[DType.float32]] = [1.0]
    with assert_raises():
        _ = r2_score(int_t, short)

    var empty = List[Scalar[DType.float64]]()
    with assert_raises():
        _ = r2_score(empty, empty)


def test_accuracy_sklearn_reference() raises:
    var bin_t: List[Scalar[DType.int32]] = [0, 1, 1, 0, 1, 1]
    var bin_p: List[Scalar[DType.int32]] = [0, 1, 0, 1, 1, 1]
    assert_almost_equal(accuracy_score(bin_t, bin_p), 0.6666666666666666)

    var multi_t: List[Scalar[DType.int32]] = [0, 1, 2, 0, 1, 2]
    var multi_p: List[Scalar[DType.int32]] = [0, 2, 1, 0, 0, 1]
    assert_almost_equal(accuracy_score(multi_t, multi_p), 0.3333333333333333)

    var four_t: List[Scalar[DType.int32]] = [0, 1, 2, 3, 0, 1]
    var four_p: List[Scalar[DType.int32]] = [0, 2, 2, 3, 1, 1]
    assert_almost_equal(accuracy_score(four_t, four_p), 0.6666666666666666)


def test_accuracy_bounds() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 1]
    assert_equal(accuracy_score(y_true, y_true), 1.0)

    var all_wrong_t: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    var all_wrong_p: List[Scalar[DType.int32]] = [1, 0, 1, 0]
    assert_equal(accuracy_score(all_wrong_t, all_wrong_p), 0.0)


def test_accuracy_normalize_false_returns_count() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1, 0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 1, 0, 1, 1, 1]
    assert_equal(accuracy_score(y_true, y_pred, normalize=False), 4.0)

    var all_wrong_t: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    var all_wrong_p: List[Scalar[DType.int32]] = [1, 0, 1, 0]
    assert_equal(accuracy_score(all_wrong_t, all_wrong_p, normalize=False), 0.0)


def test_accuracy_label_encodings() raises:
    # Labels need not be contiguous or start at zero
    var sparse_t: List[Scalar[DType.int32]] = [10, 3, 10, 3]
    var sparse_p: List[Scalar[DType.int32]] = [3, 3, 10, 10]
    assert_equal(accuracy_score(sparse_t, sparse_p), 0.5)

    var neg_t: List[Scalar[DType.int32]] = [-1, 1, -1, 1]
    var neg_p: List[Scalar[DType.int32]] = [1, 1, -1, -1]
    assert_equal(accuracy_score(neg_t, neg_p), 0.5)

    var single_t: List[Scalar[DType.int32]] = [1, 1, 1]
    assert_equal(accuracy_score(single_t, single_t), 1.0)


def test_accuracy_dtype_variants_and_validation() raises:
    var u8_t: List[Scalar[DType.uint8]] = [0, 1, 1, 0]
    var u8_p: List[Scalar[DType.uint8]] = [0, 1, 0, 0]
    assert_equal(accuracy_score(u8_t, u8_p), 0.75)

    var f64_t: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 0.0]
    var i64_p: List[Scalar[DType.int64]] = [0, 1, 0, 0]
    assert_equal(accuracy_score(f64_t, i64_p), 0.75)

    var short: List[Scalar[DType.uint8]] = [0]
    with assert_raises():
        _ = accuracy_score(u8_t, short)

    var empty = List[Scalar[DType.int32]]()
    with assert_raises():
        _ = accuracy_score(empty, empty)


def test_confusion_matrix_sklearn_reference() raises:
    var y_true: List[Scalar[DType.int32]] = [2, 0, 2, 2, 0, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 2, 2, 0, 2]
    var cm = confusion_matrix(y_true, y_pred)

    assert_equal(cm.rows, 3)
    assert_equal(cm.cols, 3)
    var expected: List[Int64] = [2, 0, 0, 0, 0, 1, 1, 0, 2]
    for i in range(3):
        for j in range(3):
            assert_equal(cm[i, j], expected[i * 3 + j])


def test_confusion_matrix_binary() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1, 0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 1, 0, 1, 1, 1]
    var cm = confusion_matrix(y_true, y_pred)

    assert_equal(cm.rows, 2)
    assert_equal(cm.cols, 2)
    assert_equal(cm[0, 0], 1)
    assert_equal(cm[0, 1], 1)
    assert_equal(cm[1, 0], 1)
    assert_equal(cm[1, 1], 3)

    var perfect_t: List[Scalar[DType.int32]] = [0, 1, 2, 1]
    var perfect = confusion_matrix(perfect_t, perfect_t)
    assert_equal(perfect[0, 0], 1)
    assert_equal(perfect[1, 1], 2)
    assert_equal(perfect[2, 2], 1)
    assert_equal(perfect[0, 1], 0)


def test_confusion_matrix_label_ordering() raises:
    # Rows and columns follow the sorted union of the labels: [3, 10]
    var sparse_t: List[Scalar[DType.int32]] = [10, 3, 10, 3]
    var sparse_p: List[Scalar[DType.int32]] = [3, 3, 10, 10]
    var sparse_cm = confusion_matrix(sparse_t, sparse_p)
    assert_equal(sparse_cm.rows, 2)
    assert_equal(sparse_cm[0, 0], 1)
    assert_equal(sparse_cm[0, 1], 1)
    assert_equal(sparse_cm[1, 0], 1)
    assert_equal(sparse_cm[1, 1], 1)

    # Negative labels sort ahead of positive ones: [-1, 1]
    var neg_t: List[Scalar[DType.int32]] = [-1, 1, -1, 1]
    var neg_p: List[Scalar[DType.int32]] = [1, 1, -1, -1]
    var neg_cm = confusion_matrix(neg_t, neg_p)
    assert_equal(neg_cm[0, 0], 1)
    assert_equal(neg_cm[0, 1], 1)

    var single_t: List[Scalar[DType.int32]] = [1, 1, 1]
    var single_cm = confusion_matrix(single_t, single_t)
    assert_equal(single_cm.rows, 1)
    assert_equal(single_cm[0, 0], 3)


def test_confusion_matrix_invariants() raises:
    # A label predicted but never observed yields an all-zero row
    var y_true: List[Scalar[DType.int32]] = [0, 0, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 1]
    var cm = confusion_matrix(y_true, y_pred)

    assert_equal(cm.rows, 3)
    assert_equal(cm[0, 0], 1)
    assert_equal(cm[0, 2], 1)
    assert_equal(cm[1, 1], 1)
    assert_equal(cm[2, 0], 0)
    assert_equal(cm[2, 1], 0)
    assert_equal(cm[2, 2], 0)

    # Every sample lands in exactly one cell, and the trace is the hit count
    var total: Int64 = 0
    var trace: Int64 = 0
    for i in range(cm.rows):
        for j in range(cm.cols):
            total += cm[i, j]
        trace += cm[i, i]
    assert_equal(total, 3)
    assert_equal(
        Float64(trace), accuracy_score(y_true, y_pred, normalize=False)
    )


def test_confusion_matrix_dtype_variants_and_validation() raises:
    var f64_t: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 0.0]
    var i64_p: List[Scalar[DType.int64]] = [0, 1, 0, 0]
    var cm = confusion_matrix(f64_t, i64_p)
    assert_equal(cm.rows, 2)
    assert_equal(cm[0, 0], 2)
    assert_equal(cm[1, 0], 1)
    assert_equal(cm[1, 1], 1)

    var short: List[Scalar[DType.int64]] = [0]
    with assert_raises():
        _ = confusion_matrix(f64_t, short)

    var empty = List[Scalar[DType.int32]]()
    with assert_raises():
        _ = confusion_matrix(empty, empty)


def test_precision_binary_sklearn_reference() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1, 0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 1, 0, 1, 1, 1]
    assert_almost_equal(precision_score(y_true, y_pred), 0.75)
    assert_almost_equal(precision_score(y_true, y_pred, pos_label=0.0), 0.5)

    var imb_t: List[Scalar[DType.int32]] = [0, 0, 0, 0, 1, 1]
    var imb_p: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 0]
    assert_almost_equal(precision_score(imb_t, imb_p), 0.5)
    assert_almost_equal(precision_score(imb_t, imb_p, pos_label=0.0), 0.75)


def test_precision_multiclass_averages() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 0, 1, 2]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 1, 0, 0, 1]
    assert_almost_equal(
        precision_score(y_true, y_pred, average="micro"), 0.3333333333333333
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="macro"), 0.2222222222222222
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="weighted"),
        0.2222222222222222,
    )

    var four_t: List[Scalar[DType.int32]] = [0, 1, 2, 3, 0, 1]
    var four_p: List[Scalar[DType.int32]] = [0, 2, 2, 3, 1, 1]
    assert_almost_equal(precision_score(four_t, four_p, average="macro"), 0.75)
    assert_almost_equal(
        precision_score(four_t, four_p, average="weighted"), 0.75
    )


def test_precision_zero_division_returns_zero() raises:
    # Nothing is predicted positive, so the denominator is zero
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 0]
    assert_equal(precision_score(y_true, y_pred), 0.0)
    assert_almost_equal(
        precision_score(y_true, y_pred, average="macro"), 0.16666666666666666
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="weighted"),
        0.1111111111111111,
    )

    var all_wrong_t: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    var all_wrong_p: List[Scalar[DType.int32]] = [1, 0, 1, 0]
    assert_equal(precision_score(all_wrong_t, all_wrong_p), 0.0)
    assert_equal(
        precision_score(all_wrong_t, all_wrong_p, average="macro"), 0.0
    )


def test_precision_micro_equals_accuracy() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 3, 0, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 2, 3, 1, 1]
    assert_almost_equal(
        precision_score(y_true, y_pred, average="micro"),
        accuracy_score(y_true, y_pred),
    )

    var perfect: List[Scalar[DType.int32]] = [0, 1, 2, 1]
    assert_equal(precision_score(perfect, perfect, average="macro"), 1.0)
    assert_equal(precision_score(perfect, perfect, average="micro"), 1.0)


def test_precision_validation() raises:
    var multi_t: List[Scalar[DType.int32]] = [0, 1, 2]
    var multi_p: List[Scalar[DType.int32]] = [0, 1, 2]

    # 'binary' averaging is undefined for more than two labels
    with assert_raises():
        _ = precision_score(multi_t, multi_p)

    with assert_raises():
        _ = precision_score(multi_t, multi_p, average="samples")

    # pos_label must actually occur in the data
    var sparse_t: List[Scalar[DType.int32]] = [10, 3, 10, 3]
    var sparse_p: List[Scalar[DType.int32]] = [3, 3, 10, 10]
    with assert_raises():
        _ = precision_score(sparse_t, sparse_p)
    assert_equal(precision_score(sparse_t, sparse_p, pos_label=10.0), 0.5)

    var short: List[Scalar[DType.int32]] = [0]
    with assert_raises():
        _ = precision_score(multi_t, short)


def test_recall_binary_sklearn_reference() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1, 0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 1, 0, 1, 1, 1]
    assert_almost_equal(recall_score(y_true, y_pred), 0.75)
    assert_almost_equal(recall_score(y_true, y_pred, pos_label=0.0), 0.5)

    var imb_t: List[Scalar[DType.int32]] = [0, 0, 0, 0, 1, 1]
    var imb_p: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 0]
    assert_almost_equal(recall_score(imb_t, imb_p), 0.5)
    assert_almost_equal(recall_score(imb_t, imb_p, pos_label=0.0), 0.75)


def test_recall_multiclass_averages() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 0, 1, 2]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 1, 0, 0, 1]
    assert_almost_equal(
        recall_score(y_true, y_pred, average="micro"), 0.3333333333333333
    )
    assert_almost_equal(
        recall_score(y_true, y_pred, average="macro"), 0.3333333333333333
    )
    assert_almost_equal(
        recall_score(y_true, y_pred, average="weighted"), 0.3333333333333333
    )

    var cm_t: List[Scalar[DType.int32]] = [2, 0, 2, 2, 0, 1]
    var cm_p: List[Scalar[DType.int32]] = [0, 0, 2, 2, 0, 2]
    assert_almost_equal(
        recall_score(cm_t, cm_p, average="macro"), 0.5555555555555555
    )
    assert_almost_equal(
        recall_score(cm_t, cm_p, average="weighted"), 0.6666666666666666
    )


def test_recall_ignores_unpredicted_labels_in_denominator() raises:
    # Recall for a label only ever predicted (never observed) is zero
    var y_true: List[Scalar[DType.int32]] = [0, 0, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 1]
    assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.5)
    assert_almost_equal(
        recall_score(y_true, y_pred, average="weighted"), 0.6666666666666666
    )


def test_recall_zero_division_and_perfect() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 0]
    assert_equal(recall_score(y_true, y_pred), 0.0)
    assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.5)

    var all_wrong_t: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    var all_wrong_p: List[Scalar[DType.int32]] = [1, 0, 1, 0]
    assert_equal(recall_score(all_wrong_t, all_wrong_p), 0.0)

    var perfect: List[Scalar[DType.int32]] = [0, 1, 2, 1]
    assert_equal(recall_score(perfect, perfect, average="macro"), 1.0)


def test_recall_weighted_equals_accuracy() raises:
    # For single-label targets, weighted recall collapses to accuracy
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 3, 0, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 2, 3, 1, 1]
    assert_almost_equal(
        recall_score(y_true, y_pred, average="weighted"),
        accuracy_score(y_true, y_pred),
    )

    var short: List[Scalar[DType.int32]] = [0]
    with assert_raises():
        _ = recall_score(y_true, short)


def test_f1_binary_sklearn_reference() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1, 0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 1, 0, 1, 1, 1]
    assert_almost_equal(f1_score(y_true, y_pred), 0.75)
    assert_almost_equal(f1_score(y_true, y_pred, pos_label=0.0), 0.5)

    var nopos_t: List[Scalar[DType.int32]] = [0, 1, 1]
    var nopos_p: List[Scalar[DType.int32]] = [0, 0, 0]
    assert_almost_equal(f1_score(nopos_t, nopos_p, pos_label=0.0), 0.5)


def test_f1_multiclass_averages() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 0, 1, 2]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 1, 0, 0, 1]
    assert_almost_equal(
        f1_score(y_true, y_pred, average="micro"), 0.3333333333333333
    )
    assert_almost_equal(
        f1_score(y_true, y_pred, average="macro"), 0.26666666666666666
    )
    assert_almost_equal(
        f1_score(y_true, y_pred, average="weighted"), 0.26666666666666666
    )

    var cm_t: List[Scalar[DType.int32]] = [2, 0, 2, 2, 0, 1]
    var cm_p: List[Scalar[DType.int32]] = [0, 0, 2, 2, 0, 2]
    assert_almost_equal(
        f1_score(cm_t, cm_p, average="macro"), 0.48888888888888893
    )
    assert_almost_equal(f1_score(cm_t, cm_p, average="weighted"), 0.6)

    var four_t: List[Scalar[DType.int32]] = [0, 1, 2, 3, 0, 1]
    var four_p: List[Scalar[DType.int32]] = [0, 2, 2, 3, 1, 1]
    assert_almost_equal(
        f1_score(four_t, four_p, average="macro"), 0.7083333333333333
    )
    assert_almost_equal(
        f1_score(four_t, four_p, average="weighted"), 0.6666666666666666
    )


def test_f1_is_harmonic_mean_of_precision_and_recall() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 0, 0, 0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 0]
    var p = precision_score(y_true, y_pred)
    var r = recall_score(y_true, y_pred)
    assert_almost_equal(f1_score(y_true, y_pred), 2.0 * p * r / (p + r))

    var cm_t: List[Scalar[DType.int32]] = [2, 0, 2, 2, 0, 1]
    var cm_p: List[Scalar[DType.int32]] = [0, 0, 2, 2, 0, 2]
    var pm = precision_score(cm_t, cm_p, average="micro")
    var rm = recall_score(cm_t, cm_p, average="micro")
    assert_almost_equal(
        f1_score(cm_t, cm_p, average="micro"), 2.0 * pm * rm / (pm + rm)
    )


def test_f1_bounds_and_zero_division() raises:
    var perfect: List[Scalar[DType.int32]] = [0, 1, 2, 1]
    assert_equal(f1_score(perfect, perfect, average="macro"), 1.0)
    assert_equal(f1_score(perfect, perfect, average="micro"), 1.0)

    var all_wrong_t: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    var all_wrong_p: List[Scalar[DType.int32]] = [1, 0, 1, 0]
    assert_equal(f1_score(all_wrong_t, all_wrong_p), 0.0)
    assert_equal(f1_score(all_wrong_t, all_wrong_p, average="macro"), 0.0)
    assert_equal(f1_score(all_wrong_t, all_wrong_p, average="weighted"), 0.0)

    var nopos_t: List[Scalar[DType.int32]] = [0, 1, 1]
    var nopos_p: List[Scalar[DType.int32]] = [0, 0, 0]
    assert_equal(f1_score(nopos_t, nopos_p), 0.0)
    assert_equal(f1_score(nopos_t, nopos_p, average="macro"), 0.25)


def test_f1_single_class_and_label_encodings() raises:
    var single_t: List[Scalar[DType.int32]] = [1, 1, 1]
    assert_equal(f1_score(single_t, single_t), 1.0)
    assert_equal(f1_score(single_t, single_t, average="macro"), 1.0)

    var neg_t: List[Scalar[DType.int32]] = [-1, 1, -1, 1]
    var neg_p: List[Scalar[DType.int32]] = [1, 1, -1, -1]
    assert_equal(f1_score(neg_t, neg_p), 0.5)
    assert_equal(f1_score(neg_t, neg_p, pos_label=-1.0), 0.5)

    var sparse_t: List[Scalar[DType.int32]] = [10, 3, 10, 3]
    var sparse_p: List[Scalar[DType.int32]] = [3, 3, 10, 10]
    assert_equal(f1_score(sparse_t, sparse_p, pos_label=3.0), 0.5)
    assert_equal(f1_score(sparse_t, sparse_p, average="macro"), 0.5)


def test_nan_labels_are_rejected() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var y_true: List[Scalar[DType.float64]] = [0.0, 1.0, nan, 1.0]
    var y_pred: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 1.0]

    with assert_raises():
        _ = accuracy_score(y_true, y_pred)
    with assert_raises():
        _ = confusion_matrix(y_true, y_pred)
    with assert_raises():
        _ = f1_score(y_true, y_pred, average="macro")
    with assert_raises():
        _ = precision_score(y_pred, y_true, average="macro")
    with assert_raises():
        _ = recall_score(y_true, y_pred, average="micro")


def test_binary_single_class_absent_pos_label() raises:
    # Only one label is present, so pos_label=1 is simply never matched:
    # the score is ill-defined and falls back to zero_division
    var y: List[Scalar[DType.int32]] = [0, 0, 0]
    assert_equal(precision_score(y, y), 0.0)
    assert_equal(recall_score(y, y), 0.0)
    assert_equal(f1_score(y, y), 0.0)

    assert_equal(precision_score(y, y, zero_division=1.0), 1.0)
    assert_equal(recall_score(y, y, zero_division=1.0), 1.0)
    assert_equal(f1_score(y, y, zero_division=1.0), 1.0)

    var nan = Float64(0.0) / Float64(0.0)
    assert_equal(isnan(precision_score(y, y, zero_division=nan)), True)
    assert_equal(isnan(f1_score(y, y, zero_division=nan)), True)

    # The label that is present still scores normally
    assert_equal(precision_score(y, y, pos_label=0.0), 1.0)
    assert_equal(recall_score(y, y, pos_label=0.0), 1.0)


def test_binary_absent_pos_label_raises_when_two_labels() raises:
    # With two labels present, an absent pos_label is a caller error
    var y_true: List[Scalar[DType.int32]] = [10, 3, 10, 3]
    var y_pred: List[Scalar[DType.int32]] = [3, 3, 10, 10]
    with assert_raises():
        _ = precision_score(y_true, y_pred)
    with assert_raises():
        _ = recall_score(y_true, y_pred)
    with assert_raises():
        _ = f1_score(y_true, y_pred)

    assert_equal(precision_score(y_true, y_pred, pos_label=3.0), 0.5)


def test_zero_division_binary() raises:
    # Nothing is predicted positive: precision is ill-defined, recall is not
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 0]

    assert_equal(precision_score(y_true, y_pred), 0.0)
    assert_equal(precision_score(y_true, y_pred, zero_division=1.0), 1.0)

    # Recall has a non-zero denominator, so zero_division does not apply
    assert_equal(recall_score(y_true, y_pred), 0.0)
    assert_equal(recall_score(y_true, y_pred, zero_division=1.0), 0.0)
    assert_equal(f1_score(y_true, y_pred, zero_division=1.0), 0.0)

    var nan = Float64(0.0) / Float64(0.0)
    assert_equal(
        isnan(precision_score(y_true, y_pred, zero_division=nan)), True
    )
    assert_equal(recall_score(y_true, y_pred, zero_division=nan), 0.0)


def test_zero_division_macro_and_weighted() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 0]

    assert_almost_equal(
        precision_score(y_true, y_pred, average="macro"), 0.16666666666666666
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="macro", zero_division=1.0),
        0.6666666666666666,
    )

    assert_almost_equal(
        precision_score(y_true, y_pred, average="weighted"),
        0.1111111111111111,
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="weighted", zero_division=1.0),
        0.7777777777777778,
    )

    # Well-defined labels are unaffected by the fallback
    assert_almost_equal(
        recall_score(y_true, y_pred, average="macro", zero_division=1.0), 0.5
    )
    assert_almost_equal(
        f1_score(y_true, y_pred, average="macro", zero_division=1.0), 0.25
    )


def test_zero_division_nan_drops_ill_defined_labels() raises:
    # A NaN fallback removes the ill-defined label from the average entirely
    var nan = Float64(0.0) / Float64(0.0)
    var y_true: List[Scalar[DType.int32]] = [0, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [0, 0, 0]

    assert_almost_equal(
        precision_score(y_true, y_pred, average="macro", zero_division=nan),
        0.3333333333333333,
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="weighted", zero_division=nan),
        0.3333333333333333,
    )

    # Micro pools the counts, so its denominator is still non-zero here
    assert_almost_equal(
        precision_score(y_true, y_pred, average="micro", zero_division=nan),
        0.3333333333333333,
    )


def test_many_classes_label_lookup() raises:
    var y_true: List[Scalar[DType.int32]] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 5, 7]
    var y_pred: List[Scalar[DType.int32]] = [0, 2, 2, 3, 4, 6, 6, 7, 1, 3, 5, 0]

    assert_almost_equal(accuracy_score(y_true, y_pred), 0.6666666666666666)
    assert_almost_equal(
        precision_score(y_true, y_pred, average="macro"), 0.6875
    )
    assert_almost_equal(recall_score(y_true, y_pred, average="macro"), 0.6875)
    assert_almost_equal(
        f1_score(y_true, y_pred, average="macro"), 0.6458333333333333
    )
    assert_almost_equal(
        precision_score(y_true, y_pred, average="weighted"), 0.75
    )
    assert_almost_equal(
        f1_score(y_true, y_pred, average="weighted"), 0.6666666666666666
    )

    var cm = confusion_matrix(y_true, y_pred)
    assert_equal(cm.rows, 8)
    var expected: List[Int64] = [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    for i in range(8):
        for j in range(8):
            assert_equal(cm[i, j], expected[i * 8 + j])


def test_labels_sorted_regardless_of_input_order() raises:
    # Labels first appear as 7, 3, 5, 1 but must be indexed as [1, 3, 5, 7]
    var y_true: List[Scalar[DType.int32]] = [7, 7, 3, 3, 5, 5, 1, 1]
    var y_pred: List[Scalar[DType.int32]] = [3, 7, 3, 5, 5, 1, 1, 7]

    var cm = confusion_matrix(y_true, y_pred)
    assert_equal(cm.rows, 4)
    var expected: List[Int64] = [
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
    ]
    for i in range(4):
        for j in range(4):
            assert_equal(cm[i, j], expected[i * 4 + j])

    assert_almost_equal(f1_score(y_true, y_pred, average="macro"), 0.5)


def test_infinite_values_are_rejected() raises:
    var inf = Float64(1.0) / Float64(0.0)

    var cls_t: List[Scalar[DType.float64]] = [0.0, 1.0, inf]
    var cls_p: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0]
    with assert_raises():
        _ = accuracy_score(cls_t, cls_p)
    with assert_raises():
        _ = confusion_matrix(cls_t, cls_p)
    with assert_raises():
        _ = f1_score(cls_t, cls_p, average="macro")

    var reg_t: List[Scalar[DType.float64]] = [1.0, 2.0, inf]
    var reg_p: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    with assert_raises():
        _ = mean_squared_error(reg_t, reg_p)
    with assert_raises():
        _ = r2_score(reg_t, reg_p)


def test_regression_rejects_nan() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var y_true: List[Scalar[DType.float64]] = [1.0, 2.0, nan]
    var y_pred: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]

    with assert_raises():
        _ = mean_squared_error(y_true, y_pred)
    with assert_raises():
        _ = root_mean_squared_error(y_true, y_pred)
    with assert_raises():
        _ = mean_absolute_error(y_true, y_pred)
    with assert_raises():
        _ = r2_score(y_true, y_pred)
    with assert_raises():
        _ = mean_absolute_error(y_pred, y_true)


def _binary_proba() raises -> Matrix[DType.float64]:
    var vals: List[Scalar[DType.float64]] = [
        0.9,
        0.1,
        0.8,
        0.2,
        0.3,
        0.7,
        0.01,
        0.99,
    ]
    return Matrix[DType.float64](4, 2, vals^)


def test_log_loss_binary_reference() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    assert_almost_equal(log_loss(y, _binary_proba()), 0.1738073366910675)


def test_log_loss_single_column_matches_two_column() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var one: List[Scalar[DType.float64]] = [0.1, 0.2, 0.7, 0.99]
    var P1 = Matrix[DType.float64](4, 1, one^)
    assert_almost_equal(log_loss(y, P1), log_loss(y, _binary_proba()))


def test_log_loss_normalize_false_returns_the_sum() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var P = _binary_proba()
    assert_almost_equal(log_loss(y, P, normalize=False), 0.69522934676427)
    assert_almost_equal(log_loss(y, P, normalize=False), log_loss(y, P) * 4.0)


def test_log_loss_multiclass_reference() raises:
    var y: List[Scalar[DType.float64]] = [2.0, 5.0, 5.0, 9.0]
    var vals: List[Scalar[DType.float64]] = [
        0.7,
        0.2,
        0.1,
        0.1,
        0.6,
        0.3,
        0.2,
        0.2,
        0.6,
        0.05,
        0.15,
        0.8,
    ]
    var P = Matrix[DType.float64](4, 3, vals^)
    assert_almost_equal(log_loss(y, P), 0.6750205078632583)


def test_log_loss_columns_follow_sorted_labels() raises:
    var y: List[Scalar[DType.float64]] = [10.0, 20.0]
    var vals: List[Scalar[DType.float64]] = [0.9, 0.1, 0.9, 0.1]
    var P = Matrix[DType.float64](2, 2, vals^)
    assert_almost_equal(log_loss(y, P), 1.203972804325936)


def test_log_loss_uniform_probabilities_is_log_two() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 1.0]
    var P = Matrix[DType.float64](2, 2, 0.5)
    assert_almost_equal(log_loss(y, P), 0.6931471805599453)


def test_log_loss_clips_certain_predictions() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 1.0]
    var best: List[Scalar[DType.float64]] = [1.0, 0.0, 0.0, 1.0]
    var worst: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 0.0]
    assert_almost_equal(log_loss(y, Matrix[DType.float64](2, 2, best^)), 0.0)
    assert_almost_equal(
        log_loss(y, Matrix[DType.float64](2, 2, worst^)), 36.04365338911715
    )


def test_log_loss_renormalizes_rows() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 1.0]
    var vals: List[Scalar[DType.float64]] = [0.45, 0.45, 0.2, 0.6]
    var P = Matrix[DType.float64](2, 2, vals^)
    assert_almost_equal(log_loss(y, P), 0.4904146265058632)


def test_log_loss_dtype_variants() raises:
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
    var vals: List[Scalar[DType.float32]] = [
        0.9,
        0.1,
        0.8,
        0.2,
        0.3,
        0.7,
        0.01,
        0.99,
    ]
    var P = Matrix[DType.float32](4, 2, vals^)
    assert_almost_equal(log_loss(y, P), 0.1738073366910675, atol=1e-6)


def test_log_loss_invalid_inputs() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 1.0]

    with assert_raises():
        _ = log_loss(y, Matrix[DType.float64](3, 2, 0.5))

    with assert_raises():
        _ = log_loss(y, Matrix[DType.float64](2, 3, 0.3))

    var single: List[Scalar[DType.float64]] = [1.0, 1.0]
    with assert_raises():
        _ = log_loss(single, Matrix[DType.float64](2, 2, 0.5))

    var empty = List[Scalar[DType.float64]]()
    with assert_raises():
        _ = log_loss(empty, Matrix[DType.float64](0, 2, 0.5))

    var nan = Float64(0.0) / Float64(0.0)
    with assert_raises():
        _ = log_loss(y, Matrix[DType.float64](2, 2, nan))


def test_roc_auc_reference_value() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var s: List[Scalar[DType.float64]] = [0.1, 0.4, 0.35, 0.8]
    assert_almost_equal(roc_auc_score(y, s), 0.75)


def test_roc_auc_perfect_and_inverted_rankings() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var good: List[Scalar[DType.float64]] = [0.1, 0.2, 0.8, 0.9]
    var bad: List[Scalar[DType.float64]] = [0.9, 0.8, 0.2, 0.1]
    assert_equal(roc_auc_score(y, good), 1.0)
    assert_equal(roc_auc_score(y, bad), 0.0)


def test_roc_auc_ties_count_as_half() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var tied: List[Scalar[DType.float64]] = [0.5, 0.5, 0.5, 0.5]
    assert_equal(roc_auc_score(y, tied), 0.5)

    var y7: List[Scalar[DType.float64]] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    var s7: List[Scalar[DType.float64]] = [0.3, 0.3, 0.6, 0.6, 0.9, 0.1, 0.6]
    assert_almost_equal(roc_auc_score(y7, s7), 0.7916666666666666)


def test_roc_auc_depends_only_on_ranking() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var s: List[Scalar[DType.float64]] = [0.1, 0.4, 0.35, 0.8]
    var rescaled: List[Scalar[DType.float64]] = [15.0, 45.0, 40.0, 85.0]
    assert_equal(roc_auc_score(y, s), roc_auc_score(y, rescaled))

    var y3: List[Scalar[DType.float64]] = [1.0, 0.0, 1.0]
    var raw: List[Scalar[DType.float64]] = [-3.0, 5.0, 12.0]
    assert_almost_equal(roc_auc_score(y3, raw), 0.5)


def test_roc_auc_pos_label_selects_positive_class() raises:
    var y: List[Scalar[DType.float64]] = [1.0, 1.0, 2.0, 2.0, 2.0]
    var s: List[Scalar[DType.float64]] = [0.2, 0.7, 0.4, 0.6, 0.9]
    var pos_two = roc_auc_score(y, s, pos_label=2.0)
    assert_almost_equal(pos_two, 0.6666666666666666)
    assert_almost_equal(roc_auc_score(y, s, pos_label=1.0), 1.0 - pos_two)


def _pairwise_auc(y: List[Float64], s: List[Float64]) -> Float64:
    var wins: Float64 = 0.0
    var pairs: Float64 = 0.0
    for i in range(len(y)):
        if y[i] != 1.0:
            continue
        for j in range(len(y)):
            if y[j] != 0.0:
                continue
            pairs += 1.0
            if s[i] > s[j]:
                wins += 1.0
            elif s[i] == s[j]:
                wins += 0.5
    return wins / pairs


def test_roc_auc_matches_pairwise_definition() raises:
    var y: List[Scalar[DType.float64]] = [
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    var s: List[Scalar[DType.float64]] = [
        0.9,
        0.9,
        0.2,
        0.6,
        0.6,
        0.1,
        0.4,
        0.4,
    ]
    assert_almost_equal(roc_auc_score(y, s), _pairwise_auc(y, s))


def test_roc_auc_dtype_variants() raises:
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
    var s: List[Scalar[DType.float32]] = [0.1, 0.4, 0.35, 0.8]
    assert_almost_equal(roc_auc_score(y, s), 0.75)


def test_roc_auc_invalid_inputs() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var s: List[Scalar[DType.float64]] = [0.1, 0.4, 0.35, 0.8]

    with assert_raises():
        _ = roc_auc_score(y, s, pos_label=7.0)

    var three: List[Scalar[DType.float64]] = [0.0, 1.0, 2.0, 1.0]
    with assert_raises():
        _ = roc_auc_score(three, s)

    var one: List[Scalar[DType.float64]] = [1.0, 1.0, 1.0, 1.0]
    with assert_raises():
        _ = roc_auc_score(one, s)

    var short: List[Scalar[DType.float64]] = [0.1, 0.4]
    with assert_raises():
        _ = roc_auc_score(y, short)

    var inf = Float64(1.0) / Float64(0.0)
    var bad: List[Scalar[DType.float64]] = [0.1, 0.4, 0.35, inf]
    with assert_raises():
        _ = roc_auc_score(y, bad)


def _two_blobs() raises -> Matrix[DType.float64]:
    var vals: List[Scalar[DType.float64]] = [
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.5,
        10.0,
        10.0,
        10.5,
        10.0,
        10.0,
        10.5,
    ]
    return Matrix[DType.float64](6, 2, vals^)


def test_silhouette_well_separated_clusters() raises:
    var labels: List[Int] = [0, 0, 0, 1, 1, 1]
    assert_almost_equal(
        silhouette_score(_two_blobs(), labels), 0.9597751259823862
    )


def test_silhouette_penalises_mixed_clusters() raises:
    var X = _two_blobs()
    var mixed: List[Int] = [0, 1, 0, 1, 0, 1]
    var good: List[Int] = [0, 0, 0, 1, 1, 1]
    assert_almost_equal(silhouette_score(X, mixed), -0.05331968611082519)
    assert_true(silhouette_score(X, mixed) < silhouette_score(X, good))


def test_silhouette_label_values_do_not_matter() raises:
    var X = _two_blobs()
    var plain: List[Int] = [0, 0, 0, 1, 1, 1]
    var odd: List[Int] = [7, 7, 7, -1, -1, -1]
    assert_equal(silhouette_score(X, plain), silhouette_score(X, odd))


def test_silhouette_three_clusters_and_singletons() raises:
    var vals: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 8.0, 9.0, 20.0]
    var X = Matrix[DType.float64](6, 1, vals^)

    var three: List[Int] = [0, 0, 0, 1, 1, 2]
    assert_almost_equal(silhouette_score(X, three), 0.6773171273171273)

    var singleton: List[Int] = [0, 0, 0, 0, 0, 1]
    assert_almost_equal(silhouette_score(X, singleton), 0.5755181458548332)


def test_silhouette_duplicate_points_across_clusters() raises:
    var vals: List[Scalar[DType.float64]] = [0.0, 0.0, 0.0, 1.0]
    var X = Matrix[DType.float64](4, 1, vals^)
    var labels: List[Int] = [0, 0, 1, 1]

    var score = silhouette_score(X, labels)
    assert_true(not isnan(score))
    assert_almost_equal(score, 0.25)


def test_silhouette_all_identical_points_score_zero() raises:
    var X = Matrix[DType.float64](4, 1, 0.0)
    var labels: List[Int] = [0, 0, 1, 1]

    var score = silhouette_score(X, labels)
    assert_true(not isnan(score))
    assert_equal(score, 0.0)


def test_silhouette_dtype_variants() raises:
    var vals: List[Scalar[DType.float32]] = [
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.5,
        10.0,
        10.0,
        10.5,
        10.0,
        10.0,
        10.5,
    ]
    var X = Matrix[DType.float32](6, 2, vals^)
    var labels: List[Int] = [0, 0, 0, 1, 1, 1]
    assert_almost_equal(
        silhouette_score(X, labels), 0.9597751259823862, atol=1e-6
    )


def test_silhouette_invalid_inputs() raises:
    var X = _two_blobs()

    var one_cluster: List[Int] = [0, 0, 0, 0, 0, 0]
    with assert_raises():
        _ = silhouette_score(X, one_cluster)

    var all_singletons: List[Int] = [0, 1, 2, 3, 4, 5]
    with assert_raises():
        _ = silhouette_score(X, all_singletons)

    var too_few: List[Int] = [0, 0, 1]
    with assert_raises():
        _ = silhouette_score(X, too_few)

    # 2 samples with 2 clusters (k=2 > n-1=1, invalid)
    var X2 = Matrix[DType.float64](2, 2, 1.0)
    var labels2: List[Int] = [0, 1]
    with assert_raises():
        _ = silhouette_score(X2, labels2)

    # NaN in feature matrix
    var nan = Float64(0.0) / Float64(0.0)
    var X_nan = Matrix[DType.float64](6, 2, nan)
    var labels_valid: List[Int] = [0, 0, 0, 1, 1, 1]
    with assert_raises():
        _ = silhouette_score(X_nan, labels_valid)


def test_silhouette_minimal_dataset_three_samples() raises:
    # Minimal valid configuration: n=3, k=2 (one cluster of size 2, one singleton)
    var vals: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 0.0, 10.0, 10.0]
    var X = Matrix[DType.float64](3, 2, vals^)
    var labels: List[Int] = [0, 0, 1]
    var score = silhouette_score(X, labels)
    assert_true(not isnan(score))
    assert_true(score > 0.0)


def test_silhouette_negative_labels_match_positive() raises:
    var X = _two_blobs()
    var pos_labels: List[Int] = [0, 0, 0, 1, 1, 1]
    var neg_labels: List[Int] = [-5, -5, -5, -2, -2, -2]
    assert_equal(
        silhouette_score(X, pos_labels), silhouette_score(X, neg_labels)
    )


def test_roc_auc_negative_labels() raises:
    var y: List[Scalar[DType.float64]] = [-1.0, -1.0, 1.0, 1.0]
    var s: List[Scalar[DType.float64]] = [-10.0, -5.0, 2.0, 8.0]
    assert_almost_equal(roc_auc_score(y, s, pos_label=1.0), 1.0)
    assert_almost_equal(roc_auc_score(y, s, pos_label=-1.0), 0.0)


def test_roc_auc_all_tied_imbalanced() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 1.0, 1.0, 1.0]
    var s: List[Scalar[DType.float64]] = [0.5, 0.5, 0.5, 0.5]
    assert_almost_equal(roc_auc_score(y, s), 0.5)


def test_roc_auc_nan_and_empty_rejected() raises:
    var empty = List[Scalar[DType.float64]]()
    with assert_raises():
        _ = roc_auc_score(empty, empty)

    var nan = Float64(0.0) / Float64(0.0)
    var y_nan: List[Scalar[DType.float64]] = [0.0, 1.0, nan, 1.0]
    var s_valid: List[Scalar[DType.float64]] = [0.1, 0.4, 0.5, 0.8]
    with assert_raises():
        _ = roc_auc_score(y_nan, s_valid)

    var y_valid: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]
    var s_nan: List[Scalar[DType.float64]] = [0.1, nan, 0.5, 0.8]
    with assert_raises():
        _ = roc_auc_score(y_valid, s_nan)


def test_log_loss_arbitrary_binary_labels() raises:
    var y: List[Scalar[DType.float64]] = [-1.0, 1.0, 1.0, -1.0]
    var one: List[Scalar[DType.float64]] = [0.1, 0.9, 0.8, 0.2]
    var P1 = Matrix[DType.float64](4, 1, one^)
    assert_true(log_loss(y, P1) < 0.25)


def test_log_loss_column_mismatch_with_three_classes() raises:
    var y: List[Scalar[DType.float64]] = [0.0, 1.0, 2.0]
    var P_single = Matrix[DType.float64](3, 1, 0.5)
    with assert_raises():
        _ = log_loss(y, P_single)

    var P_four = Matrix[DType.float64](3, 4, 0.25)
    with assert_raises():
        _ = log_loss(y, P_four)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
