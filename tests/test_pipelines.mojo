from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
)
from strata import (
    Matrix,
    Dataset,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PipelineTransformer,
    PipelineRegressor,
    PipelineClassifier,
    Regressor,
    Classifier,
)
from strata.base.estimator import (
    fit as fit_ds,
    predict as predict_ds,
    predict_proba as predict_proba_ds,
    transform as transform_ds,
    fit_transform as fit_transform_ds,
)


@fieldwise_init
struct MockLinearRegressor(Movable, Regressor):
    var weight: Float64
    var bias: Float64

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        var res = List[Scalar[feat_dtype]](capacity=X.rows)
        for r in range(X.rows):
            var val = Float64(X[r, 0]) * self.weight + self.bias
            res.append(Scalar[feat_dtype](val))
        return res^


@fieldwise_init
struct MockLogisticClassifier(Classifier, Movable):
    var threshold: Float64

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        pass

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        var res = List[Int](capacity=X.rows)
        for r in range(X.rows):
            if Float64(X[r, 0]) >= self.threshold:
                res.append(1)
            else:
                res.append(0)
        return res^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        var res = Matrix[feat_dtype](X.rows, 2, 0)
        for r in range(X.rows):
            var prob1: Float64 = (
                0.8 if Float64(X[r, 0]) >= self.threshold else 0.2
            )
            var prob0: Float64 = 1.0 - prob1
            res[r, 0] = Scalar[feat_dtype](prob0)
            res[r, 1] = Scalar[feat_dtype](prob1)
        return res^


def test_pipeline_transformer_chaining() raises:
    # Scale with mean then scale with std in 2 chained steps
    var s1 = StandardScaler(with_mean=True, with_std=False)
    var s2 = StandardScaler(with_mean=False, with_std=True)
    var prep = PipelineTransformer((s1^, s2^))

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 10.0
    X[1, 0] = 20.0
    X[2, 0] = 30.0
    X[3, 0] = 40.0

    var X_trans = prep.fit_transform(X)
    assert_equal(X_trans.rows, 4)
    assert_equal(X_trans.cols, 1)


def test_pipeline_regressor_end_to_end() raises:
    var scaler = StandardScaler()
    var reg = MockLinearRegressor(2.0, 1.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 2.0
    X[1, 0] = 4.0
    X[2, 0] = 6.0
    var y: List[Scalar[DType.float64]] = [5.0, 9.0, 13.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 3)


def test_pipeline_classifier_and_probabilities() raises:
    var scaler = StandardScaler(with_mean=False, with_std=False)
    var clf = MockLogisticClassifier(0.0)
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float32](2, 1, 0)
    X[0, 0] = 5.0
    X[1, 0] = -5.0
    var y: List[Scalar[DType.int32]] = [1, 0]

    pipe.fit(X, y)

    # 1. Discrete Int predictions
    var preds = pipe.predict(X)
    assert_equal(len(preds), 2)
    assert_equal(preds[0], 1)
    assert_equal(preds[1], 0)

    # 2. Probability matrix matching Float32 precision
    var probs = pipe.predict_proba(X)
    assert_equal(probs.rows, 2)
    assert_equal(probs.cols, 2)
    assert_equal(probs[0, 1], 0.8)
    assert_equal(probs[1, 1], 0.2)


def test_generic_dataset_helpers() raises:
    var scaler = StandardScaler()
    var reg = MockLinearRegressor(1.0, 0.0)
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 10.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var ds = Dataset(X^, y^)

    # Test top-level generic functional helpers
    fit_ds(pipe, ds)
    var preds = predict_ds(pipe, ds)
    assert_equal(len(preds), 3)


from strata import (
    Matrix,
    Dataset,
    StandardScaler,
    LinearRegression,
    Ridge,
    LogisticRegression,
    PipelineTransformer,
    PipelineRegressor,
    PipelineClassifier,
    Regressor,
    Classifier,
    NotFittedError,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_pipeline_regressor_with_real_linear_regression() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [3.0, 5.0, 7.0, 9.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)
    assert_almost_equal(preds[0], 3.0, rtol=1e-3)
    assert_almost_equal(preds[3], 9.0, rtol=1e-3)


def test_pipeline_regressor_with_real_ridge() raises:
    var scaler = StandardScaler()
    var ridge = Ridge(alpha=1.0)
    var pipe = PipelineRegressor(scaler^, ridge^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)


def test_pipeline_classifier_with_real_logistic_regression() raises:
    var scaler = StandardScaler()
    var clf = LogisticRegression(max_iter=150)
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = -30.0
    X[1, 0] = -20.0
    X[2, 0] = -10.0
    X[3, 0] = 10.0
    X[4, 0] = 20.0
    X[5, 0] = 30.0
    var y: List[Scalar[DType.int32]] = [0, 0, 0, 1, 1, 1]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[5], 1)

    var probs = pipe.predict_proba(X)
    assert_equal(probs.rows, 6)
    assert_equal(probs.cols, 2)
    assert_true(probs[0, 0] > 0.5)
    assert_true(probs[5, 1] > 0.5)


def test_pipeline_classifier_multiclass_real() raises:
    var scaler = StandardScaler()
    var clf = LogisticRegression(max_iter=200)
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -5.0
    X[0, 1] = -5.0
    X[1, 0] = -4.0
    X[1, 1] = -4.0
    X[2, 0] = 0.0
    X[2, 1] = 5.0
    X[3, 0] = 1.0
    X[3, 1] = 4.0
    X[4, 0] = 5.0
    X[4, 1] = -5.0
    X[5, 0] = 4.0
    X[5, 1] = -4.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1, 2, 2]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(preds[0], 0)
    assert_equal(preds[2], 1)
    assert_equal(preds[4], 2)


def test_pipeline_regressor_copy_constructor() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe1 = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]
    pipe1.fit(X, y)

    var pipe2 = pipe1.copy()
    var p1 = pipe1.predict(X)
    var p2 = pipe2.predict(X)
    for i in range(3):
        assert_almost_equal(p1[i], p2[i], rtol=1e-4)


def test_pipeline_classifier_copy_constructor() raises:
    var scaler = StandardScaler()
    var clf = LogisticRegression(max_iter=100)
    var pipe1 = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
    pipe1.fit(X, y)

    var pipe2 = pipe1.copy()
    var p1 = pipe1.predict(X)
    var p2 = pipe2.predict(X)
    assert_equal(p1[0], p2[0])
    assert_equal(p1[3], p2[3])


def test_pipeline_transformer_copy_constructor() raises:
    var s1 = StandardScaler(with_mean=True, with_std=False)
    var s2 = StandardScaler(with_mean=False, with_std=True)
    var t1 = PipelineTransformer((s1^, s2^))

    var X = Matrix[DType.float64](3, 1, 10.0)
    _ = t1.fit_transform(X)

    var t2 = t1.copy()
    var X_trans = t2.transform(X)
    assert_equal(X_trans.rows, 3)


def test_pipeline_float32_casting() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X_f32 = Matrix[DType.float32](3, 1, 0)
    X_f32[0, 0] = 1.0
    X_f32[1, 0] = 2.0
    X_f32[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    pipe.fit(X_f32, y)
    var preds = pipe.predict(X_f32)
    assert_equal(len(preds), 3)


def test_pipeline_bfloat16_casting() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X_bf = Matrix[DType.bfloat16](3, 1, 0)
    X_bf[0, 0] = 1.0
    X_bf[1, 0] = 2.0
    X_bf[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    pipe.fit(X_bf, y)
    var preds = pipe.predict(X_bf)
    assert_equal(len(preds), 3)


def test_pipeline_regressor_unfitted_predict_error() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64].ones(3, 2)
    with assert_raises():
        _ = pipe.predict(X)


def test_pipeline_classifier_unfitted_predict_proba_error() raises:
    var scaler = StandardScaler()
    var clf = LogisticRegression()
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float64].ones(3, 2)
    with assert_raises():
        _ = pipe.predict_proba(X)


def test_pipeline_feature_dimension_mismatch_error() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X_train = Matrix[DType.float64].ones(4, 2)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0, 4.0]
    pipe.fit(X_train, y)

    var X_bad = Matrix[DType.float64].ones(3, 4)
    with assert_raises():
        _ = pipe.predict(X_bad)


def test_pipeline_regressor_dataset_fit_predict() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]
    var ds = Dataset(X^, y^)

    fit_ds(pipe, ds)
    var preds = predict_ds(pipe, ds)
    assert_equal(len(preds), 3)


def test_pipeline_classifier_dataset_fit_predict_proba() raises:
    var scaler = StandardScaler()
    var clf = LogisticRegression(max_iter=100)
    var pipe = PipelineClassifier(scaler^, clf^)

    var X = Matrix[DType.float64](4, 1, 0)
    X[0, 0] = -2.0
    X[1, 0] = -1.0
    X[2, 0] = 1.0
    X[3, 0] = 2.0
    var y: List[Scalar[DType.int32]] = [0, 0, 1, 1]
    var ds = Dataset[DType.float64, DType.int32](X^, y^)

    fit_ds(pipe, ds)
    var probs = predict_proba_ds(pipe, ds)
    assert_equal(probs.rows, 4)
    assert_equal(probs.cols, 2)


def test_pipeline_transformer_dataset_transform() raises:
    var s1 = StandardScaler()
    var s2 = StandardScaler(with_mean=False, with_std=False)
    var pipe = PipelineTransformer((s1^, s2^))

    var X = Matrix[DType.float64](3, 2, 5.0)
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var ds = Dataset(X^, y^)

    var ds_trans = fit_transform_ds(pipe, ds)
    assert_equal(ds_trans.n_samples(), 3)
    assert_equal(ds_trans.n_features(), 2)


def test_pipeline_high_dimensional_features() raises:
    var scaler = StandardScaler()
    var reg = Ridge(alpha=0.1)
    var pipe = PipelineRegressor(scaler^, reg^)

    var N = 20
    var D = 6
    var X = Matrix[DType.float64](N, D, 1.0)
    var y = List[Scalar[DType.float64]](capacity=N)
    for i in range(N):
        X[i, 0] = Float64(i)
        y.append(Float64(i * 2))

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), N)


def test_pipeline_identity_scaler() raises:
    var noop_scaler = StandardScaler(with_mean=False, with_std=False)
    var reg = LinearRegression()
    var pipe = PipelineRegressor(noop_scaler^, reg^)

    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_almost_equal(preds[0], 2.0, rtol=1e-3)
    assert_almost_equal(preds[2], 6.0, rtol=1e-3)


def test_pipeline_repeated_fit_reset() raises:
    var scaler = StandardScaler()
    var reg = LinearRegression()
    var pipe = PipelineRegressor(scaler^, reg^)

    var X1 = Matrix[DType.float64](3, 1, 0)
    X1[0, 0] = 1.0
    X1[1, 0] = 2.0
    X1[2, 0] = 3.0
    var y1: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0]
    pipe.fit(X1, y1)

    var X2 = Matrix[DType.float64](3, 1, 0)
    X2[0, 0] = 1.0
    X2[1, 0] = 2.0
    X2[2, 0] = 3.0
    var y2: List[Scalar[DType.float64]] = [10.0, 20.0, 30.0]
    pipe.fit(X2, y2)

    var preds = pipe.predict(X2)
    assert_almost_equal(preds[0], 10.0, rtol=1e-3)


def test_pipeline_transformer_three_steps() raises:
    # Test variadic chaining of 3 transformers in series
    var pipe_trans = PipelineTransformer(
        (
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
        )
    )
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 10.0
    X[0, 1] = 100.0
    X[1, 0] = 20.0
    X[1, 1] = 200.0
    X[2, 0] = 30.0
    X[2, 1] = 300.0
    X[3, 0] = 40.0
    X[3, 1] = 400.0

    var Xt = pipe_trans.fit_transform(X)
    assert_equal(Xt.rows, 4)
    assert_equal(Xt.cols, 2)
    assert_true(pipe_trans.is_fitted)


def test_pipeline_transformer_variadic_regressor() raises:
    var pipe_trans = PipelineTransformer(
        (
            StandardScaler(),
            MinMaxScaler(),
        )
    )
    var reg = MockLinearRegressor(1.0, 0.0)
    var pipe = PipelineRegressor(pipe_trans^, reg^)

    var X = Matrix[DType.float64](3, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    var y: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
