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
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LinearRegression,
    LogisticRegression,
    PipelineRegressor,
    PipelineClassifier,
    ColumnTransformer,
    DimensionMismatchError,
    InvalidParameterError,
    NotFittedError,
)


def test_column_transformer_two_scalers() raises:
    # 4 samples, 3 features
    # col 0: [10, 20, 30, 40] -> StandardScaler
    # col 1: [100, 200, 300, 400] -> MinMaxScaler
    # col 2: [5, 5, 5, 5] -> dropped
    var X = Matrix[DType.float64](4, 3, 0)
    X[0, 0] = 10.0
    X[0, 1] = 100.0
    X[0, 2] = 5.0
    X[1, 0] = 20.0
    X[1, 1] = 200.0
    X[1, 2] = 5.0
    X[2, 0] = 30.0
    X[2, 1] = 300.0
    X[2, 2] = 5.0
    X[3, 0] = 40.0
    X[3, 1] = 400.0
    X[3, 2] = 5.0

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c1: List[Int] = [1]
    specs.append(c0^)
    specs.append(c1^)

    var ct = ColumnTransformer(
        (StandardScaler(), MinMaxScaler()), specs, remainder="drop"
    )
    var Xt = ct.fit_transform(X)

    assert_true(ct.is_fitted)
    assert_equal(ct.n_features_in_, 3)
    assert_equal(Xt.rows, 4)
    assert_equal(Xt.cols, 2)

    # Col 1 (MinMaxScaler on [100, 400]) maps to [0, 1]
    assert_almost_equal(Xt[0, 1], 0.0)
    assert_almost_equal(Xt[3, 1], 1.0)


def test_column_transformer_with_passthrough_remainder() raises:
    var X = Matrix[DType.float64](3, 3, 0)
    X[0, 0] = 10.0
    X[0, 1] = 1.0
    X[0, 2] = 2.0
    X[1, 0] = 20.0
    X[1, 1] = 3.0
    X[1, 2] = 4.0
    X[2, 0] = 30.0
    X[2, 1] = 5.0
    X[2, 2] = 6.0

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    specs.append(c0^)

    var ct = ColumnTransformer(
        (StandardScaler(),), specs, remainder="passthrough"
    )
    var Xt = ct.fit_transform(X)

    assert_equal(Xt.rows, 3)
    assert_equal(Xt.cols, 3)
    # Remainder columns [1, 2] passed through exactly
    assert_equal(Xt[0, 1], 1.0)
    assert_equal(Xt[0, 2], 2.0)
    assert_equal(Xt[2, 1], 5.0)
    assert_equal(Xt[2, 2], 6.0)


def test_column_transformer_with_one_hot_encoder() raises:
    # 4 samples, 2 numerical features + 1 categorical feature (0, 1, 2)
    var X = Matrix[DType.float64](4, 3, 0)
    X[0, 0] = 1.0
    X[0, 1] = 10.0
    X[0, 2] = 0.0
    X[1, 0] = 2.0
    X[1, 1] = 20.0
    X[1, 2] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 30.0
    X[2, 2] = 2.0
    X[3, 0] = 4.0
    X[3, 1] = 40.0
    X[3, 2] = 0.0

    var specs = List[List[Int]]()
    var c01: List[Int] = [0, 1]
    var c2: List[Int] = [2]
    specs.append(c01^)
    specs.append(c2^)

    var ct = ColumnTransformer(
        (StandardScaler(), OneHotEncoder()), specs, remainder="drop"
    )
    var Xt = ct.fit_transform(X)

    assert_equal(Xt.rows, 4)
    # 2 scaled cols + 3 one-hot category cols = 5 total cols
    assert_equal(Xt.cols, 5)


def test_column_transformer_four_transformers() raises:
    # Testing a 4-way heterogeneous transformer pipeline
    var X = Matrix[DType.float64](4, 4, 0)
    for r in range(4):
        X[r, 0] = Float64(r * 10)
        X[r, 1] = Float64(r * 100)
        X[r, 2] = Float64(r * 1000)
        X[r, 3] = Float64(r % 2)

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c1: List[Int] = [1]
    var c2: List[Int] = [2]
    var c3: List[Int] = [3]
    specs.append(c0^)
    specs.append(c1^)
    specs.append(c2^)
    specs.append(c3^)

    var ct = ColumnTransformer(
        (StandardScaler(), MinMaxScaler(), RobustScaler(), OneHotEncoder()),
        specs,
        remainder="drop",
    )
    var Xt = ct.fit_transform(X)

    assert_equal(Xt.rows, 4)
    # 1 + 1 + 1 + 2 (one-hot binary) = 5 cols
    assert_equal(Xt.cols, 5)


def test_column_transformer_in_pipeline_regressor() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 100.0
    X[1, 0] = 2.0
    X[1, 1] = 200.0
    X[2, 0] = 3.0
    X[2, 1] = 300.0
    X[3, 0] = 4.0
    X[3, 1] = 400.0

    var y: List[Scalar[DType.float64]] = [5.0, 10.0, 15.0, 20.0]

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c1: List[Int] = [1]
    specs.append(c0^)
    specs.append(c1^)

    var ct = ColumnTransformer(
        (StandardScaler(), MinMaxScaler()), specs, remainder="drop"
    )
    var reg = LinearRegression()
    var pipe = PipelineRegressor(ct^, reg^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)


def test_column_transformer_in_pipeline_classifier() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = 10.0
    X[1, 0] = -1.0
    X[1, 1] = 20.0
    X[2, 0] = 1.0
    X[2, 1] = 30.0
    X[3, 0] = 2.0
    X[3, 1] = 40.0

    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c1: List[Int] = [1]
    specs.append(c0^)
    specs.append(c1^)

    var ct = ColumnTransformer(
        (StandardScaler(), StandardScaler()), specs, remainder="drop"
    )
    var clf = LogisticRegression()
    var pipe = PipelineClassifier(ct^, clf^)

    pipe.fit(X, y)
    var preds = pipe.predict(X)
    assert_equal(len(preds), 4)


def test_column_transformer_invalid_params_and_mismatch() raises:
    var specs = List[List[Int]]()
    with assert_raises():
        _ = ColumnTransformer(
            (StandardScaler(),),
            specs, # Length mismatch: 0 specs for 1 transformer
            remainder="drop",
        )

    var c0 = List[Int]()
    specs.append(c0^)
    with assert_raises():
        _ = ColumnTransformer(
            (StandardScaler(),),
            specs,
            remainder="invalid_remainder",
        )

    var ct = ColumnTransformer(
        (StandardScaler(),), specs, remainder="drop"
    )
    var X = Matrix[DType.float64](2, 2, 1.0)
    with assert_raises():
        _ = ct.transform(X)

    ct.fit(X)
    var X_wrong = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = ct.transform(X_wrong)


def test_column_transformer_copy() raises:
    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c1: List[Int] = [1]
    specs.append(c0^)
    specs.append(c1^)

    var ct1 = ColumnTransformer(
        (StandardScaler(), MinMaxScaler()), specs, remainder="drop"
    )
    var X = Matrix[DType.float64](3, 2, 1.0)
    ct1.fit(X)

    var ct2 = ColumnTransformer(copy=ct1)
    assert_true(ct2.is_fitted)
    assert_equal(ct2.n_features_in_, 2)


def test_column_transformer_float32() raises:
    var X = Matrix[DType.float32](4, 2, 0)
    X[0, 0] = 10.0
    X[0, 1] = 100.0
    X[1, 0] = 20.0
    X[1, 1] = 200.0
    X[2, 0] = 30.0
    X[2, 1] = 300.0
    X[3, 0] = 40.0
    X[3, 1] = 400.0

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c1: List[Int] = [1]
    specs.append(c0^)
    specs.append(c1^)

    var ct = ColumnTransformer(
        (StandardScaler(), MinMaxScaler()), specs, remainder="drop"
    )
    var Xt = ct.fit_transform(X)
    assert_equal(Xt.rows, 4)
    assert_equal(Xt.cols, 2)


def test_column_transformer_overlapping_columns() raises:
    # Multiple transformers selecting the same column (e.g. col 0)
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 10.0
    X[1, 0] = 2.0
    X[1, 1] = 20.0
    X[2, 0] = 3.0
    X[2, 1] = 30.0

    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    var c0_again: List[Int] = [0]
    specs.append(c0^)
    specs.append(c0_again^)

    var ct = ColumnTransformer(
        (StandardScaler(), MinMaxScaler()), specs, remainder="passthrough"
    )
    var Xt = ct.fit_transform(X)

    # 1 (scaled col 0) + 1 (minmax col 0) + 1 (remainder col 1) = 3 cols
    assert_equal(Xt.rows, 3)
    assert_equal(Xt.cols, 3)


def test_column_transformer_single_sample() raises:
    var X = Matrix[DType.float64](1, 2, 5.0)
    var specs = List[List[Int]]()
    var c0: List[Int] = [0]
    specs.append(c0^)

    var ct = ColumnTransformer(
        (StandardScaler(),), specs, remainder="passthrough"
    )
    var Xt = ct.fit_transform(X)
    assert_equal(Xt.rows, 1)
    assert_equal(Xt.cols, 2)


def test_column_transformer_out_of_bounds_columns() raises:
    var X = Matrix[DType.float64](3, 2, 1.0)
    var specs = List[List[Int]]()
    var bad_cols: List[Int] = [0, 5]
    specs.append(bad_cols^)

    var ct = ColumnTransformer((StandardScaler(),), specs)
    with assert_raises():
        _ = ct.fit(X)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
