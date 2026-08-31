from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_almost_equal,
    assert_raises,
)
from strata import (
    Matrix,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    BufferWriter,
    BufferReader,
    dump,
    load,
    dumps,
    loads,
    DataConversionError,
    InvalidParameterError,
)


def test_buffer_writer_and_reader_primitives() raises:
    var writer = BufferWriter()
    writer.write_byte(42)
    writer.write_bool(True)
    writer.write_bool(False)
    writer.write_int(123456789)
    writer.write_int(-987654321)
    writer.write_float64(3.141592653589793)
    writer.write_float32(2.71828)
    writer.write_string("hello strata serialization")
    writer.write_int_list([1, 2, 3, 4, 5])
    writer.write_float64_list([10.5, 20.25, -30.75])

    var m = Matrix[DType.float64](2, 2, 0)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[1, 0] = 3.0
    m[1, 1] = 4.0
    writer.write_matrix[DType.float64](m)

    var bytes = writer.get_bytes()
    var reader = BufferReader(bytes)

    assert_equal(Int(reader.read_byte()), 42)
    assert_true(reader.read_bool())
    assert_false(reader.read_bool())
    assert_equal(reader.read_int(), 123456789)
    assert_equal(reader.read_int(), -987654321)
    assert_almost_equal(reader.read_float64(), 3.141592653589793)
    assert_almost_equal(Float64(reader.read_float32()), 2.71828, atol=1e-4)
    assert_equal(reader.read_string(), "hello strata serialization")

    var ilist = reader.read_int_list()
    assert_equal(len(ilist), 5)
    assert_equal(ilist[0], 1)
    assert_equal(ilist[4], 5)

    var flist = reader.read_float64_list()
    assert_equal(len(flist), 3)
    assert_almost_equal(flist[0], 10.5)
    assert_almost_equal(flist[2], -30.75)

    var m_rec = reader.read_matrix[DType.float64]()
    assert_equal(m_rec.rows, 2)
    assert_equal(m_rec.cols, 2)
    assert_almost_equal(m_rec[0, 0], 1.0)
    assert_almost_equal(m_rec[1, 1], 4.0)


def test_linear_regression_persistence() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0
    var y: List[Scalar[DType.float64]] = [5.0, 5.0, 11.0, 11.0]

    var reg = LinearRegression(solver="cholesky")
    reg.fit(X, y)
    var preds_orig = reg.predict(X)

    var path = "/tmp/test_linear_regression.strata"
    dump(reg, path)

    var loaded = load[LinearRegression[DType.float64]](path)
    assert_true(loaded.is_fitted)
    assert_equal(len(loaded.coef_), len(reg.coef_))
    assert_almost_equal(loaded.intercept_, reg.intercept_)

    var preds_loaded = loaded.predict(X)
    for i in range(4):
        assert_almost_equal(preds_loaded[i], preds_orig[i])


def test_ridge_persistence() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 3.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 5.0
    var y: List[Scalar[DType.float64]] = [3.0, 5.0, 7.0, 9.0]

    var model = Ridge(alpha=1.0)
    model.fit(X, y)
    var preds_orig = model.predict(X)

    var bytes = dumps(model)
    var loaded = loads[Ridge[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_almost_equal(loaded.alpha, 1.0)
    var preds_loaded = loaded.predict(X)
    for i in range(4):
        assert_almost_equal(preds_loaded[i], preds_orig[i])


def test_lasso_persistence() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 0.5
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 1.5
    X[3, 0] = 4.0
    X[3, 1] = 2.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var model = Lasso(alpha=0.1)
    model.fit(X, y)
    var preds_orig = model.predict(X)

    var path = "/tmp/test_lasso.strata"
    dump(model, path)
    var loaded = load[Lasso[DType.float64]](path)

    assert_true(loaded.is_fitted)
    var preds_loaded = loaded.predict(X)
    for i in range(4):
        assert_almost_equal(preds_loaded[i], preds_orig[i])


def test_elastic_net_persistence() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 2.0
    X[1, 1] = 2.0
    X[2, 0] = 3.0
    X[2, 1] = 3.0
    X[3, 0] = 4.0
    X[3, 1] = 4.0
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0]

    var model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)
    var preds_orig = model.predict(X)

    var path = "/tmp/test_elastic_net.strata"
    dump(model, path)
    var loaded = load[ElasticNet[DType.float64]](path)

    assert_true(loaded.is_fitted)
    assert_almost_equal(loaded.l1_ratio, 0.5)
    var preds_loaded = loaded.predict(X)
    for i in range(4):
        assert_almost_equal(preds_loaded[i], preds_orig[i])


def test_logistic_regression_persistence() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -1.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 1.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0]

    var clf = LogisticRegression(C=1.0)
    clf.fit(X, y)
    var preds_orig = clf.predict(X)
    var proba_orig = clf.predict_proba(X)

    var path = "/tmp/test_logreg.strata"
    dump(clf, path)
    var loaded = load[LogisticRegression[DType.float64]](path)

    assert_true(loaded.is_fitted)
    assert_equal(len(loaded.classes_), 2)

    var preds_loaded = loaded.predict(X)
    var proba_loaded = loaded.predict_proba(X)

    for i in range(4):
        assert_equal(preds_loaded[i], preds_orig[i])
        assert_almost_equal(proba_loaded[i, 0], proba_orig[i, 0])
        assert_almost_equal(proba_loaded[i, 1], proba_orig[i, 1])


def test_standard_scaler_persistence() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 10.0
    X[0, 1] = 100.0
    X[1, 0] = 20.0
    X[1, 1] = 200.0
    X[2, 0] = 30.0
    X[2, 1] = 300.0

    var scaler = StandardScaler()
    scaler.fit(X)
    var Xt_orig = scaler.transform(X)

    var path = "/tmp/test_scaler.strata"
    dump(scaler, path)
    var loaded = load[StandardScaler[DType.float64]](path)

    assert_true(loaded.is_fitted)
    var Xt_loaded = loaded.transform(X)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(Xt_loaded[r, c], Xt_orig[r, c])


def test_minmax_scaler_persistence() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 10.0
    X[1, 0] = 5.0
    X[1, 1] = 20.0
    X[2, 0] = 10.0
    X[2, 1] = 30.0

    var scaler = MinMaxScaler(feature_range_min=0.0, feature_range_max=1.0)
    scaler.fit(X)
    var Xt_orig = scaler.transform(X)

    var bytes = dumps(scaler)
    var loaded = loads[MinMaxScaler[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    var Xt_loaded = loaded.transform(X)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(Xt_loaded[r, c], Xt_orig[r, c])


def test_robust_scaler_persistence() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    for i in range(5):
        X[i, 0] = Float64(i * 10)
        X[i, 1] = Float64(i * 100)

    var scaler = RobustScaler()
    scaler.fit(X)
    var Xt_orig = scaler.transform(X)

    var path = "/tmp/test_robust_scaler.strata"
    dump(scaler, path)
    var loaded = load[RobustScaler[DType.float64]](path)

    assert_true(loaded.is_fitted)
    var Xt_loaded = loaded.transform(X)
    for r in range(5):
        for c in range(2):
            assert_almost_equal(Xt_loaded[r, c], Xt_orig[r, c])


def test_decision_tree_regressor_persistence() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 2.0
    X[3, 0] = 4.0
    X[3, 1] = 2.0
    X[4, 0] = 5.0
    X[4, 1] = 3.0
    X[5, 0] = 6.0
    X[5, 1] = 3.0
    var y: List[Scalar[DType.float64]] = [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]

    var reg = DecisionTreeRegressor(max_depth=3)
    reg.fit(X, y)
    var preds_orig = reg.predict(X)

    var path = "/tmp/test_dt_reg.strata"
    dump(reg, path)
    var loaded = load[DecisionTreeRegressor[DType.float64]](path)

    assert_true(loaded.is_fitted)
    assert_equal(loaded.n_features_in_, 2)
    var preds_loaded = loaded.predict(X)
    for i in range(6):
        assert_almost_equal(preds_loaded[i], preds_orig[i])


def test_decision_tree_classifier_persistence() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 6.0
    X[3, 1] = 5.0
    X[4, 0] = 10.0
    X[4, 1] = 10.0
    X[5, 0] = 11.0
    X[5, 1] = 10.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    var clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    var preds_orig = clf.predict(X)

    var bytes = dumps(clf)
    var loaded = loads[DecisionTreeClassifier[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_equal(loaded.n_classes_, 3)
    var preds_loaded = loaded.predict(X)
    for i in range(6):
        assert_equal(preds_loaded[i], preds_orig[i])


def test_random_forest_regressor_persistence() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    for i in range(6):
        X[i, 0] = Float64(i + 1)
        X[i, 1] = Float64((i + 1) * 2)
    var y: List[Scalar[DType.float64]] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

    var rf = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=42)
    rf.fit(X, y)
    var preds_orig = rf.predict(X)

    var path = "/tmp/test_rf_reg.strata"
    dump(rf, path)
    var loaded = load[RandomForestRegressor[DType.float64]](path)

    assert_true(loaded.is_fitted)
    assert_equal(len(loaded.estimators_), 5)
    var preds_loaded = loaded.predict(X)
    for i in range(6):
        assert_almost_equal(preds_loaded[i], preds_orig[i])


def test_random_forest_classifier_persistence() raises:
    var X = Matrix[DType.float64](6, 2, 0)
    X[0, 0] = -2.0
    X[0, 1] = -1.0
    X[1, 0] = -1.0
    X[1, 1] = -2.0
    X[2, 0] = 1.0
    X[2, 1] = 2.0
    X[3, 0] = 2.0
    X[3, 1] = 1.0
    X[4, 0] = 5.0
    X[4, 1] = 5.0
    X[5, 0] = 6.0
    X[5, 1] = 6.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    var rf = RandomForestClassifier(
        n_estimators=5, max_depth=2, random_state=42
    )
    rf.fit(X, y)
    var preds_orig = rf.predict(X)

    var path = "/tmp/test_rf_clf.strata"
    dump(rf, path)
    var loaded = load[RandomForestClassifier[DType.float64]](path)

    assert_true(loaded.is_fitted)
    assert_equal(len(loaded.estimators_), 5)
    var preds_loaded = loaded.predict(X)
    for i in range(6):
        assert_equal(preds_loaded[i], preds_orig[i])


def test_unfitted_model_persistence() raises:
    var lr = LinearRegression(fit_intercept=False, solver="cholesky")
    var bytes = dumps(lr)
    var loaded = loads[LinearRegression[DType.float64]](bytes)
    assert_false(loaded.is_fitted)
    assert_false(loaded.fit_intercept)
    assert_equal(loaded.solver, "cholesky")

    var scaler = StandardScaler(with_mean=False, with_std=True)
    var sc_bytes = dumps(scaler)
    var sc_loaded = loads[StandardScaler[DType.float64]](sc_bytes)
    assert_false(sc_loaded.is_fitted)
    assert_false(sc_loaded.with_mean)
    assert_true(sc_loaded.with_std)


def test_multiclass_logistic_regression_3class_persistence() raises:
    var X = Matrix[DType.float64](6, 3, 0)
    X[0, 0] = -5.0
    X[0, 1] = -5.0
    X[0, 2] = -5.0
    X[1, 0] = -4.0
    X[1, 1] = -4.0
    X[1, 2] = -4.0
    X[2, 0] = 0.0
    X[2, 1] = 0.0
    X[2, 2] = 0.0
    X[3, 0] = 1.0
    X[3, 1] = 0.5
    X[3, 2] = 0.5
    X[4, 0] = 5.0
    X[4, 1] = 5.0
    X[4, 2] = 5.0
    X[5, 0] = 6.0
    X[5, 1] = 6.0
    X[5, 2] = 6.0
    var y: List[Scalar[DType.float64]] = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    var clf = LogisticRegression(C=1.0, max_iter=200)
    clf.fit(X, y)
    var preds_orig = clf.predict(X)
    var proba_orig = clf.predict_proba(X)

    var bytes = dumps(clf)
    var loaded = loads[LogisticRegression[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_equal(len(loaded.classes_), 3)
    assert_equal(loaded.coef_.rows, 3)
    assert_equal(loaded.coef_.cols, 3)

    var preds_loaded = loaded.predict(X)
    var proba_loaded = loaded.predict_proba(X)
    for i in range(6):
        assert_equal(preds_loaded[i], preds_orig[i])
        for c in range(3):
            assert_almost_equal(proba_loaded[i, c], proba_orig[i, c], atol=1e-5)


def test_float32_linear_models_persistence() raises:
    var X = Matrix[DType.float32](3, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 3.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    var y: List[Scalar[DType.float32]] = [5.0, 7.0, 9.0]

    var reg = LinearRegression[DType.float32]()
    reg.fit(X, y)
    var preds_orig = reg.predict(X)

    var path = "/tmp/test_lr_f32.strata"
    dump(reg, path)
    var loaded = load[LinearRegression[DType.float32]](path)
    assert_true(loaded.is_fitted)

    var preds_loaded = loaded.predict(X)
    for i in range(3):
        assert_almost_equal(preds_loaded[i], preds_orig[i], atol=1e-4)


def test_minmax_scaler_custom_range_and_clipping() raises:
    var X = Matrix[DType.float64](3, 2, 0)
    X[0, 0] = 0.0
    X[0, 1] = 10.0
    X[1, 0] = 5.0
    X[1, 1] = 20.0
    X[2, 0] = 10.0
    X[2, 1] = 30.0

    var scaler = MinMaxScaler(
        feature_range_min=-5.0, feature_range_max=15.0, clip=True
    )
    scaler.fit(X)
    var orig_tf = scaler.transform(X)

    var bytes = dumps(scaler)
    var loaded = loads[MinMaxScaler[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_true(loaded.clip)
    assert_almost_equal(loaded.feature_range_min, -5.0)
    assert_almost_equal(loaded.feature_range_max, 15.0)

    var loaded_tf = loaded.transform(X)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(loaded_tf[r, c], orig_tf[r, c])

    var X_out_of_bounds = Matrix[DType.float64](1, 2, 0)
    X_out_of_bounds[0, 0] = 100.0
    X_out_of_bounds[0, 1] = -100.0
    var clipped_out = loaded.transform(X_out_of_bounds)
    assert_almost_equal(clipped_out[0, 0], 15.0)
    assert_almost_equal(clipped_out[0, 1], -5.0)


def test_robust_scaler_custom_quantiles() raises:
    var X = Matrix[DType.float64](5, 2, 0)
    for r in range(5):
        X[r, 0] = Float64(r * 10)
        X[r, 1] = Float64(r * 2)

    var scaler = RobustScaler(
        with_centering=False,
        with_scaling=True,
        quantile_min=10.0,
        quantile_max=90.0,
    )
    scaler.fit(X)
    var orig_tf = scaler.transform(X)

    var bytes = dumps(scaler)
    var loaded = loads[RobustScaler[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_false(loaded.with_centering)
    assert_true(loaded.with_scaling)
    assert_almost_equal(loaded.quantile_min, 10.0)
    assert_almost_equal(loaded.quantile_max, 90.0)

    var loaded_tf = loaded.transform(X)
    for r in range(5):
        for c in range(2):
            assert_almost_equal(loaded_tf[r, c], orig_tf[r, c])


def test_decision_tree_classifier_entropy_and_random_state() raises:
    var X = Matrix[DType.float64](8, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 1.0
    X[1, 0] = 1.5
    X[1, 1] = 1.2
    X[2, 0] = 5.0
    X[2, 1] = 5.0
    X[3, 0] = 5.5
    X[3, 1] = 5.2
    X[4, 0] = 9.0
    X[4, 1] = 9.0
    X[5, 0] = 9.5
    X[5, 1] = 9.2
    X[6, 0] = 1.2
    X[6, 1] = 1.1
    X[7, 0] = 5.2
    X[7, 1] = 5.1
    var y: List[Scalar[DType.float64]] = [
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        0.0,
        1.0,
    ]

    var dt = DecisionTreeClassifier(
        criterion="entropy", splitter="best", max_depth=3, random_state=999
    )
    dt.fit(X, y)
    var orig_preds = dt.predict(X)
    var orig_proba = dt.predict_proba(X)

    var bytes = dumps(dt)
    var loaded = loads[DecisionTreeClassifier[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_equal(loaded.criterion, "entropy")
    assert_equal(loaded.random_state, 999)

    var loaded_preds = loaded.predict(X)
    var loaded_proba = loaded.predict_proba(X)
    for i in range(8):
        assert_equal(loaded_preds[i], orig_preds[i])
        for c in range(3):
            assert_almost_equal(loaded_proba[i, c], orig_proba[i, c])


def test_decision_tree_regressor_friedman_mse() raises:
    var X = Matrix[DType.float64](6, 1, 0)
    X[0, 0] = 1.0
    X[1, 0] = 2.0
    X[2, 0] = 3.0
    X[3, 0] = 4.0
    X[4, 0] = 5.0
    X[5, 0] = 6.0
    var y: List[Scalar[DType.float64]] = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

    var reg = DecisionTreeRegressor(
        criterion="friedman_mse", max_depth=2, min_samples_leaf=2
    )
    reg.fit(X, y)
    var orig_preds = reg.predict(X)

    var bytes = dumps(reg)
    var loaded = loads[DecisionTreeRegressor[DType.float64]](bytes)

    assert_true(loaded.is_fitted)
    assert_equal(loaded.criterion, "friedman_mse")
    assert_equal(loaded.min_samples_leaf, 2)

    var loaded_preds = loaded.predict(X)
    for i in range(6):
        assert_almost_equal(loaded_preds[i], orig_preds[i])


def test_repeated_cycles_and_chain_persistence() raises:
    var X = Matrix[DType.float64](4, 2, 0)
    X[0, 0] = 1.0
    X[0, 1] = 2.0
    X[1, 0] = 2.0
    X[1, 1] = 1.0
    X[2, 0] = 3.0
    X[2, 1] = 4.0
    X[3, 0] = 4.0
    X[3, 1] = 3.0
    var y: List[Scalar[DType.float64]] = [3.0, 3.0, 7.0, 7.0]

    # Cycle 1: fit & dump
    var m1 = LinearRegression()
    m1.fit(X, y)
    var path1 = "/tmp/cycle_m1.strata"
    dump(m1, path1)

    # Cycle 2: load & verify
    var m2 = load[LinearRegression[DType.float64]](path1)
    var preds2 = m2.predict(X)

    # Cycle 3: dump m2 & load as m3
    var path2 = "/tmp/cycle_m2.strata"
    dump(m2, path2)
    var m3 = load[LinearRegression[DType.float64]](path2)
    var preds3 = m3.predict(X)

    for i in range(4):
        assert_almost_equal(preds2[i], preds3[i])


def test_empty_lists_and_zero_dimension_persistence() raises:
    var writer = BufferWriter()
    writer.write_int_list(List[Int]())
    writer.write_float64_list(List[Float64]())
    var empty_mat = Matrix[DType.float64](0, 0, 0)
    writer.write_matrix[DType.float64](empty_mat)

    var bytes = writer.get_bytes()
    var reader = BufferReader(bytes)

    var ilist = reader.read_int_list()
    assert_equal(len(ilist), 0)

    var flist = reader.read_float64_list()
    assert_equal(len(flist), 0)

    var mat = reader.read_matrix[DType.float64]()
    assert_equal(mat.rows, 0)
    assert_equal(mat.cols, 0)


def test_error_handling_and_corrupt_files() raises:
    # 1. Loading corrupt magic
    var writer = BufferWriter()
    writer.write_string("CORRUPT_MAGIC")
    writer.write_int(1)
    writer.write_string("LinearRegression")
    var corrupt_bytes = writer.get_bytes()

    with assert_raises():
        _ = loads[LinearRegression[DType.float64]](corrupt_bytes)

    # 2. Corrupt / unsupported version tag (version 99)
    var v99_writer = BufferWriter()
    v99_writer.write_string("STRATA")
    v99_writer.write_int(99)
    v99_writer.write_string("LinearRegression")
    var v99_bytes = v99_writer.get_bytes()

    with assert_raises():
        _ = loads[LinearRegression[DType.float64]](v99_bytes)

    # 3. Type mismatch (loading Ridge bytes into LinearRegression)
    var ridge = Ridge(alpha=1.0)
    var ridge_bytes = dumps(ridge)
    with assert_raises():
        _ = loads[LinearRegression[DType.float64]](ridge_bytes)

    # 4. Truncated buffer EOF
    var truncated = List[UInt8](capacity=4)
    truncated.append(83)
    truncated.append(84)
    with assert_raises():
        _ = loads[LinearRegression[DType.float64]](truncated)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
