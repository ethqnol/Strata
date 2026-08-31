from std.testing import (
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
    TestSuite,
)
from strata import (
    Matrix,
    Dataset,
    NotFittedError,
    Regressor,
    check_is_fitted,
    check_array,
    check_X_y,
    sigmoid,
    softmax,
)


def test_domain_errors() raises:
    var err = NotFittedError.error("Estimator", "call fit first")
    assert_true(String(err).byte_length() > 0)

    with assert_raises():
        check_is_fitted("TestModel", False)


def test_validation() raises:
    var empty_m = Matrix[DType.float64](0, 0, 0)
    with assert_raises():
        check_array(empty_m, allow_empty=False)

    var valid_m = Matrix[DType.float64](3, 2, 1.0)
    var wrong_y: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        check_X_y(valid_m, wrong_y)

    var valid_y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    check_X_y(valid_m, valid_y)


def test_math_utils() raises:
    var s0 = sigmoid[DType.float64](0.0)
    assert_equal(s0, 0.5)
    var s_neg = sigmoid[DType.float64](-1000.0)
    assert_equal(s_neg, 0.0)
    var s_pos = sigmoid[DType.float64](1000.0)
    assert_equal(s_pos, 1.0)

    # Numerically stable softmax on extreme logits
    var logits_extreme: List[Scalar[DType.float64]] = [1000.0, 1000.0]
    var probs_extreme = softmax[DType.float64](logits_extreme)
    assert_equal(probs_extreme[0], 0.5)
    assert_equal(probs_extreme[1], 0.5)

    # Numerically stable log_sum_exp
    from strata import log_sum_exp, PRNG
    from std.math import log

    var lse = log_sum_exp[DType.float64](logits_extreme)
    # LSE([1000, 1000]) = 1000 + ln(2)
    assert_almost_equal(lse, 1000.0 + log(2.0))

    # Test PRNG
    var rng = PRNG(123)
    var r_val = rng.next_int(10)
    assert_true(r_val >= 0 and r_val < 10)

    # Verify negative seeds do not alias positive seeds
    var rng_pos = PRNG(5)
    var rng_neg = PRNG(-5)
    assert_true(rng_pos.next_u64() != rng_neg.next_u64())


@fieldwise_init
struct MockRegressor(Copyable, Movable, Regressor):
    var slope: Float64

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        var res = List[Scalar[feat_dtype]](capacity=X.rows)
        for r in range(X.rows):
            var val = Float64(X[r, 0]) * self.slope
            res.append(Scalar[feat_dtype](val))
        return res^


def test_pipeline_and_traits() raises:
    from strata import (
        StandardScaler,
        PipelineRegressor,
        PipelineTransformer,
        Dataset,
    )
    from strata.base.estimator import fit as fit_ds, predict as predict_ds

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0

    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]

    # Standard Float64 usage with 2-method MockRegressor
    var scaler = StandardScaler()
    var reg = MockRegressor(2.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var ds = Dataset(X^, y^)
    # Fits using generic Dataset free function or matrix methods
    fit_ds(pipe, ds)
    var preds = pipe.predict(ds.records)
    assert_equal(len(preds), 4)

    # Test N-step composable PipelineTransformer
    var scaler1 = StandardScaler(with_mean=True, with_std=False)
    var scaler2 = StandardScaler(with_mean=False, with_std=True)
    var chained_prep = PipelineTransformer((scaler1^, scaler2^))
    var reg2 = MockRegressor(3.0)
    var deep_pipe = PipelineRegressor(chained_prep^, reg2^)

    var X_deep = Matrix[DType.float64](3, 1, 10.0)
    var y_deep: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    deep_pipe.fit(X_deep, y_deep)
    var deep_preds = deep_pipe.predict(X_deep)
    assert_equal(len(deep_preds), 3)


from strata import (
    DimensionMismatchError,
    ConvergenceError,
    InvalidParameterError,
    DataConversionError,
    Classifier,
    Transformer,
)
from strata.utils.validation import check_consistent_length, check_sparse


def test_dimension_mismatch_error_formatting() raises:
    var err1 = DimensionMismatchError.error("10", "5")
    assert_true(String(err1).byte_length() > 0)

    var err2 = DimensionMismatchError.error("10", "5", context="test_fn")
    assert_true("test_fn" in String(err2))

    var struct_err = DimensionMismatchError("10", "5", "mismatch")
    assert_true(String(struct_err).byte_length() > 0)


def test_convergence_error_formatting() raises:
    var err = ConvergenceError.error("Optimizer", 100, 0.5)
    assert_true("Optimizer" in String(err))
    assert_true("100" in String(err))

    var struct_err = ConvergenceError("Optimizer", 100, 0.5)
    assert_true(String(struct_err).byte_length() > 0)


def test_invalid_parameter_error_formatting() raises:
    var err = InvalidParameterError.error("learning_rate", "must be > 0")
    assert_true("learning_rate" in String(err))

    var struct_err = InvalidParameterError("learning_rate", "must be > 0")
    assert_true(String(struct_err).byte_length() > 0)


def test_data_conversion_error_formatting() raises:
    var err = DataConversionError.error("Cannot cast string to float")
    assert_true("DataConversionError" in String(err))

    var struct_err = DataConversionError("Cannot cast string to float")
    assert_true(String(struct_err).byte_length() > 0)


def test_not_fitted_error_custom_message() raises:
    var default_err = NotFittedError.error("Ridge")
    assert_true("Ridge" in String(default_err))

    var custom_err = NotFittedError.error("Ridge", "fit was skipped")
    assert_true("fit was skipped" in String(custom_err))


def test_check_is_fitted_passes_when_true() raises:
    check_is_fitted("FittedModel", True)


def test_check_array_empty_allowed() raises:
    var empty_m = Matrix[DType.float64](0, 0, 0)
    check_array(empty_m, allow_empty=True)


def test_check_array_nan_check() raises:
    var nan_val = Float64(0.0) / Float64(0.0)
    var m = Matrix[DType.float64](2, 2, 1.0)
    m[0, 1] = nan_val

    with assert_raises():
        check_array(m, force_all_finite=True)


def test_check_array_inf_check() raises:
    var inf_val = Float64(1.0) / Float64(0.0)
    var m = Matrix[DType.float64](2, 2, 1.0)
    m[1, 0] = inf_val

    with assert_raises():
        check_array(m, force_all_finite=True)


def test_check_consistent_length_validation() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    var y_good: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    check_consistent_length(X, y_good)

    var y_bad: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        check_consistent_length(X, y_bad)


def test_check_sparse_validation() raises:
    var data: List[Scalar[DType.float64]] = [1.0, 2.0]
    var indices: List[Int] = [0, 1]
    var indptr: List[Int] = [0, 1, 2]

    # Valid 2x2 CSR
    check_sparse(2, 2, data, indices, indptr, is_csr=True)

    # Invalid indptr length
    var bad_indptr: List[Int] = [0, 1]
    with assert_raises():
        check_sparse(2, 2, data, indices, bad_indptr, is_csr=True)


def test_check_sparse_out_of_bounds_indices() raises:
    var data: List[Scalar[DType.float64]] = [1.0]
    var indices: List[Int] = [5]  # cols=2, so 5 is invalid
    var indptr: List[Int] = [0, 1, 1]

    with assert_raises():
        check_sparse(2, 2, data, indices, indptr, is_csr=True)


@fieldwise_init
struct CustomClassifier(Classifier, Copyable, Movable):
    var decision_threshold: Float64

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        var res = List[Int](capacity=X.rows)
        for r in range(X.rows):
            res.append(1 if Float64(X[r, 0]) >= self.decision_threshold else 0)
        return res^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        var res = Matrix[feat_dtype](X.rows, 2, 0)
        for r in range(X.rows):
            var p1 = 0.9 if Float64(X[r, 0]) >= self.decision_threshold else 0.1
            res[r, 0] = Scalar[feat_dtype](1.0 - p1)
            res[r, 1] = Scalar[feat_dtype](p1)
        return res^


def test_custom_classifier_trait_implementation() raises:
    var clf = CustomClassifier(0.0)
    var X = Matrix[DType.float64](2, 1, 0)
    X[0, 0] = 5.0
    X[1, 0] = -5.0
    var y: List[Scalar[DType.int32]] = [1, 0]

    clf.fit(X, y)
    var preds = clf.predict(X)
    assert_equal(preds[0], 1)
    assert_equal(preds[1], 0)

    var probs = clf.predict_proba(X)
    assert_equal(probs.rows, 2)
    assert_equal(probs.cols, 2)


@fieldwise_init
struct CustomTransformer(Copyable, Movable, Transformer):
    var scale_factor: Float64

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        pass

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                res[r, c] = Scalar[in_dtype](
                    Float64(X[r, c]) * self.scale_factor
                )
        return res^

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit(X)
        return self.transform(X)


def test_custom_transformer_trait_implementation() raises:
    var trans = CustomTransformer(3.0)
    var X = Matrix[DType.float64](2, 2, 2.0)
    trans.fit(X)
    var X_scaled = trans.transform(X)
    assert_equal(X_scaled[0, 0], 6.0)
    assert_equal(X_scaled[1, 1], 6.0)


def test_check_array_dim_mismatch() raises:
    var X = Matrix[DType.float64](3, 3, 1.0)
    check_array(X)
    assert_equal(X.rows, 3)


def test_check_array_single_element() raises:
    var X = Matrix[DType.float64](1, 1, 10.0)
    check_array(X)
    assert_equal(X.rows, 1)


def test_check_X_y_integer_target_type() raises:
    var X = Matrix[DType.float64](4, 2, 1.0)
    var y: List[Scalar[DType.int32]] = [0, 1, 0, 1]
    check_X_y(X, y)


def test_check_X_y_float32_feature_type() raises:
    var X = Matrix[DType.float32](3, 2, 1.5)
    var y: List[Scalar[DType.float32]] = [1.0, 2.0, 3.0]
    check_X_y(X, y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
