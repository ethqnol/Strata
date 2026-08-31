# Saving and Loading Models with Binary Serialization (`strata.io`)

Strata provides a fast, zero-copy, endian-safe binary serialization engine for saving fitted estimators, transformers, decision trees, and random forests to disk or in-memory byte buffers.

---

## 1. Quick Start: Saving and Loading an Estimator

To persist any fitted model to disk, call `dump(model, path)`. To restore it, call `load[ModelType](path)`:

```mojo
from strata import Matrix, LinearRegression, dump, load

def main() raises:
    # 1. Prepare training data
    var X_train = Matrix[DType.float64](4, 2, 0)
    X_train[0, 0] = 1.0; X_train[0, 1] = 2.0
    X_train[1, 0] = 2.0; X_train[1, 1] = 1.0
    X_train[2, 0] = 3.0; X_train[2, 1] = 4.0
    X_train[3, 0] = 4.0; X_train[3, 1] = 3.0
    var y_train: List[Scalar[DType.float64]] = [5.0, 5.0, 11.0, 11.0]

    # 2. Fit the model
    var reg = LinearRegression(solver="cholesky")
    reg.fit(X_train, y_train)

    # 3. Save fitted model to a binary file
    dump(reg, "linear_regression.strata")

    # 4. Load the fitted model back into memory
    var loaded_reg = load[LinearRegression]("linear_regression.strata")

    # 5. Execute inference immediately with exact mathematical parity
    var preds = loaded_reg.predict(X_train)
    print("Inference successful. Fitted intercept:", loaded_reg.intercept_)
```

---

## 2. In-Memory Byte Serialization (`dumps` & `loads`)

When transmitting models over network sockets, storing checkpoints in databases, or caching pipelines in memory, use `dumps` and `loads`:

```mojo
from strata import dumps, loads, Ridge, Matrix

def main() raises:
    var ridge = Ridge(alpha=0.5)
    ridge.fit(X, y)

    # Serialize model to a List[UInt8] byte buffer
    var payload: List[UInt8] = dumps(ridge)
    print("Serialized payload size:", len(payload), "bytes")

    # Reconstruct model from raw bytes
    var loaded_ridge = loads[Ridge](payload)
    var preds = loaded_ridge.predict(X)
```

---

## 3. Saving & Loading Preprocessors and Scalers

All preprocessing scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) conform to `Serializable` and retain all learned statistical parameters:

```mojo
from strata import StandardScaler, MinMaxScaler, RobustScaler, dump, load

def main() raises:
    # Fit scaler on training features
    var scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)

    # Save to disk
    dump(scaler, "scaler.strata")

    # Load in production inference service
    var deployed_scaler = load[StandardScaler]("scaler.strata")
    var X_scaled = deployed_scaler.transform(X_test)
```

---

## 4. Saving & Loading Trees and Ensembles

Tree models (`DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`) recursively serialize the full binary split routing tree, feature indices, continuous thresholds, leaf impurity values, class histograms, and probability distributions:

```mojo
from strata import RandomForestClassifier, dump, load

def main() raises:
    var rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Persist the 100-tree forest to disk
    dump(rf, "random_forest_100.strata")

    # Load the forest back
    var loaded_rf = load[RandomForestClassifier]("random_forest_100.strata")

    # Validate that OOB score and feature importances are preserved
    print("Loaded trees:", len(loaded_rf.estimators_))
    print("OOB Score:", loaded_rf.get_oob_score())
```

---

## 5. Type Precision and Generic Parameters

Strata estimators support both `Float64` and `Float32` precision. When loading a model with a non-default precision type, pass the concrete type argument to `load` / `loads`:

```mojo
from strata import LinearRegression, load

def main() raises:
    # For models trained with Float32:
    var model_f32 = load[LinearRegression[DType.float32]]("model_f32.strata")

    # For default Float64 models:
    var model_f64 = load[LinearRegression]("model_f64.strata")
```

---

## 6. Safety, Header Verification, & Error Handling

Every serialized payload includes a magic header (`STRATA\x01`) and an explicit model type identifier. If you attempt to load a corrupt file or load bytes with the wrong struct type, Strata raises a `DataConversionError`:

```mojo
from strata import dumps, loads, LinearRegression, Ridge

def main() raises:
    var ridge = Ridge(alpha=1.0)
    var ridge_bytes = dumps(ridge)

    try:
        # Raises DataConversionError: Model type mismatch (expected 'LinearRegression', got 'Ridge')
        var bad = loads[LinearRegression](ridge_bytes)
    except e:
        print("Safely caught type mismatch:", e)
```

---

## 7. Implementing `Serializable` for Custom Estimators

To make custom estimators or transformers persistable with `dump` and `load`, implement the `Serializable` trait:

```mojo
from strata.io import BufferWriter, BufferReader, Serializable, write_header, check_header

struct CustomEstimator(Copyable, Movable, Serializable):
    var is_fitted: Bool
    var weight: Float64
    var bias: Float64

    def __init__(out self, weight: Float64 = 1.0, bias: Float64 = 0.0):
        self.is_fitted = False
        self.weight = weight
        self.bias = bias

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.weight = copy.weight
        self.bias = copy.bias

    def serialize(self, mut writer: BufferWriter):
        write_header(writer, "CustomEstimator")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.weight)
        writer.write_float64(self.bias)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        check_header(reader, "CustomEstimator")
        var is_fitted = reader.read_bool()
        var weight = reader.read_float64()
        var bias = reader.read_float64()

        var model = CustomEstimator(weight=weight, bias=bias)
        model.is_fitted = is_fitted
        return model^
```
