from ..core.matrix import Matrix
from ..base.estimator import Regressor, Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.random import PRNG
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)
from ..tree.regressor import DecisionTreeRegressor
from ..tree.classifier import DecisionTreeClassifier
from ._forest import (
    generate_bootstrap_indices,
    get_oob_mask,
    compute_tree_feature_importances,
    normalize_feature_importances,
)


def _find_class_index(classes: List[Int], target: Int) -> Int:
    """Finds index of target in sorted classes list using binary search."""
    var low = 0
    var high = len(classes) - 1
    while low <= high:
        var mid = low + (high - low) // 2
        if classes[mid] == target:
            return mid
        elif classes[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


struct RandomForestRegressor[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor, Serializable):
    """Random Forest Regressor ensemble estimator.

    An ensemble of decision trees trained via bootstrap aggregation (bagging).
    Predictions are computed as the arithmetic mean of individual tree predictions.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        n_estimators: Number of trees in the forest. Default 100.
        criterion: Impurity split criterion ('squared_error', 'friedman_mse', 'absolute_error'). Default 'squared_error'.
        max_depth: Maximum tree depth. -1 means unlimited. Default -1.
        min_samples_split: Minimum samples required to split an internal node. Default 2.
        min_samples_leaf: Minimum samples required to be a leaf node. Default 1.
        min_impurity_decrease: Split threshold if impurity decrease >= this value. Default 0.0.

        max_features: Number of features to consider per split ('all', 'sqrt', 'log2'). Default 'sqrt'.
        max_features_count: Exact number of features per split. Default -1 (disabled).
        max_features_ratio: Proportion of features per split. Default 0.0 (disabled).
        bootstrap: Whether to use bootstrap sampling. Default True.
        max_samples_ratio: Proportion of samples drawn per tree when bootstrap=True. Default 1.0.
        max_samples_count: Exact number of samples drawn per tree. Default -1 (disabled).
        oob_score: Whether to compute out-of-bag $R^2$ score after fitting. Default False.
        random_state: PRNG seed for deterministic tree builds. Default 42.

    Attributes:
        n_features_in_: Number of features seen during fit.
        feature_importances_: Normalized impurity feature importance vector.
        oob_score_: Out-of-bag $R^2$ score (available when oob_score=True).
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.ensemble import RandomForestRegressor
        from strata.core import Matrix

        var rf = RandomForestRegressor[DType.float64](n_estimators=50, max_depth=6)
        rf.fit(X_train, y_train)
        var preds = rf.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var n_estimators: Int
    var criterion: String
    var max_depth: Int
    var min_samples_split: Int
    var min_samples_leaf: Int
    var min_impurity_decrease: Float64
    var max_features: String
    var max_features_count: Int
    var max_features_ratio: Float64
    var bootstrap: Bool
    var max_samples_ratio: Float64
    var max_samples_count: Int
    var oob_score: Bool
    var random_state: Int

    # Fitted attributes
    var n_features_in_: Int
    var estimators_: List[DecisionTreeRegressor[Self.compute_dtype]]
    var feature_importances_: List[Float64]
    var oob_score_: Float64
    var rng: PRNG

    def __init__(
        out self,
        n_estimators: Int = 100,
        criterion: String = "squared_error",
        max_depth: Int = -1,
        min_samples_split: Int = 2,
        min_samples_leaf: Int = 1,
        min_impurity_decrease: Float64 = 0.0,
        max_features: String = "sqrt",
        max_features_count: Int = -1,
        max_features_ratio: Float64 = 0.0,
        bootstrap: Bool = True,
        max_samples_ratio: Float64 = 1.0,
        max_samples_count: Int = -1,
        oob_score: Bool = False,
        random_state: Int = 42,
    ) raises:
        check_floating_dtype[Self.compute_dtype, "RandomForestRegressor"]()

        if n_estimators < 1:
            raise InvalidParameterError.error(
                "n_estimators",
                "n_estimators must be >= 1, got " + String(n_estimators),
            )

        if (
            criterion != "squared_error"
            and criterion != "friedman_mse"
            and criterion != "absolute_error"
        ):
            raise InvalidParameterError.error(
                "criterion",
                "RandomForestRegressor requires criterion in ['squared_error',"
                " 'friedman_mse', 'absolute_error'], got '"
                + criterion
                + "'",
            )

        if max_depth < 1 and max_depth != -1:
            raise InvalidParameterError.error(
                "max_depth",
                "max_depth must be >= 1 or -1 (unlimited), got "
                + String(max_depth),
            )

        if min_samples_split < 2:
            raise InvalidParameterError.error(
                "min_samples_split",
                "min_samples_split must be >= 2, got "
                + String(min_samples_split),
            )

        if min_samples_leaf < 1:
            raise InvalidParameterError.error(
                "min_samples_leaf",
                "min_samples_leaf must be >= 1, got "
                + String(min_samples_leaf),
            )

        if min_impurity_decrease < 0.0:
            raise InvalidParameterError.error(
                "min_impurity_decrease",
                "min_impurity_decrease must be >= 0.0, got "
                + String(min_impurity_decrease),
            )

        if (
            max_features_count == -1
            and max_features_ratio == 0.0
            and max_features != "all"
            and max_features != "sqrt"
            and max_features != "log2"
        ):
            raise InvalidParameterError.error(
                "max_features",
                "max_features must be 'all', 'sqrt', or 'log2', got '"
                + max_features
                + "'",
            )

        if max_features_count != -1 and max_features_count < 1:
            raise InvalidParameterError.error(
                "max_features_count",
                "max_features_count must be >= 1, got "
                + String(max_features_count),
            )

        if max_features_ratio != 0.0 and (
            max_features_ratio <= 0.0 or max_features_ratio > 1.0
        ):
            raise InvalidParameterError.error(
                "max_features_ratio",
                "max_features_ratio must be in (0.0, 1.0], got "
                + String(max_features_ratio),
            )

        if max_samples_ratio <= 0.0 or max_samples_ratio > 1.0:
            raise InvalidParameterError.error(
                "max_samples_ratio",
                "max_samples_ratio must be in (0.0, 1.0], got "
                + String(max_samples_ratio),
            )

        if max_samples_count != -1 and max_samples_count < 1:
            raise InvalidParameterError.error(
                "max_samples_count",
                "max_samples_count must be >= 1, got "
                + String(max_samples_count),
            )

        if oob_score and not bootstrap:
            raise InvalidParameterError.error(
                "oob_score",
                "oob_score=True requires bootstrap=True",
            )

        self.is_fitted = False
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_features_count = max_features_count
        self.max_features_ratio = max_features_ratio
        self.bootstrap = bootstrap
        self.max_samples_ratio = max_samples_ratio
        self.max_samples_count = max_samples_count
        self.oob_score = oob_score
        self.random_state = random_state

        self.n_features_in_ = 0
        self.estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]]()
        self.feature_importances_ = List[Float64]()
        self.oob_score_ = 0.0
        self.rng = PRNG(random_state if random_state >= 0 else 42)

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_estimators = copy.n_estimators
        self.criterion = copy.criterion
        self.max_depth = copy.max_depth
        self.min_samples_split = copy.min_samples_split
        self.min_samples_leaf = copy.min_samples_leaf
        self.min_impurity_decrease = copy.min_impurity_decrease
        self.max_features = copy.max_features
        self.max_features_count = copy.max_features_count
        self.max_features_ratio = copy.max_features_ratio
        self.bootstrap = copy.bootstrap
        self.max_samples_ratio = copy.max_samples_ratio
        self.max_samples_count = copy.max_samples_count
        self.oob_score = copy.oob_score
        self.random_state = copy.random_state
        self.n_features_in_ = copy.n_features_in_
        self.estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]]()
        for i in range(len(copy.estimators_)):
            self.estimators_.append(
                DecisionTreeRegressor[Self.compute_dtype](
                    copy=copy.estimators_[i]
                )
            )
        self.feature_importances_ = copy.feature_importances_.copy()
        self.oob_score_ = copy.oob_score_
        self.rng = PRNG(copy.random_state if copy.random_state >= 0 else 42)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]],) raises:
        """Fits the random forest on (X, y).

        Each tree is trained on a bootstrap sample (or the full dataset when
        bootstrap=False). After fitting, `feature_importances_` holds averaged
        MDI importances and, when oob_score=True, `oob_score_` holds the OOB R² score.
        """
        check_X_y(X, y)
        self.n_features_in_ = X.cols

        var n_samples = X.rows
        self.estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]](
            capacity=self.n_estimators
        )
        self.rng = PRNG(self.random_state if self.random_state >= 0 else 42)

        # OOB accumulation buffers: sum of predictions and count per sample.
        var oob_preds = List[Float64](capacity=n_samples)
        var oob_counts = List[Int](capacity=n_samples)
        for _ in range(n_samples):
            oob_preds.append(0.0)
            oob_counts.append(0)

        # Accumulate raw (un-normalized) MDI importances across all trees.
        var raw_importances = List[Float64](capacity=self.n_features_in_)
        for _ in range(self.n_features_in_):
            raw_importances.append(0.0)

        for _ in range(self.n_estimators):
            # Each tree gets its own independent PRNG streams seeded from the parent.
            # Splitting into two separate streams prevents tree-seed and
            # bootstrap-index generation from interleaving on the same state.
            var tree_seed = Int(self.rng.next_u64() % UInt64(2147483647))
            var boot_rng = PRNG(Int(self.rng.next_u64() % UInt64(2147483647)))

            var tree = DecisionTreeRegressor[Self.compute_dtype](
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_features_count=self.max_features_count,
                max_features_ratio=self.max_features_ratio,
                random_state=tree_seed,
            )

            if self.bootstrap:
                var sample_indices = generate_bootstrap_indices(
                    n_samples,
                    self.max_samples_count,
                    self.max_samples_ratio,
                    boot_rng,
                )

                # Build bootstrap X and y views (index-filtered copies).
                var k = len(sample_indices)
                var X_boot = Matrix[feat_dtype](k, X.cols)
                var y_boot = List[Scalar[target_dtype]](capacity=k)
                for i in range(k):
                    var src = sample_indices[i]
                    for j in range(X.cols):
                        X_boot[i, j] = X[src, j]
                    y_boot.append(y[src])

                tree.fit[feat_dtype, target_dtype](X_boot, y_boot)

                if self.oob_score:
                    var oob_mask = get_oob_mask(n_samples, sample_indices)
                    # Collect all OOB row indices to perform a single batched inference pass per tree.
                    var oob_row_indices = List[Int]()
                    for i in range(n_samples):
                        if oob_mask[i]:
                            oob_row_indices.append(i)

                    var n_oob = len(oob_row_indices)
                    if n_oob > 0:
                        var X_oob = Matrix[feat_dtype](n_oob, X.cols)
                        for i in range(n_oob):
                            var src = oob_row_indices[i]
                            for j in range(X.cols):
                                X_oob[i, j] = X[src, j]

                        var oob_preds_tree = tree.predict[feat_dtype](X_oob)
                        for i in range(n_oob):
                            var orig = oob_row_indices[i]
                            oob_preds[orig] += oob_preds_tree[i].cast[
                                DType.float64
                            ]()
                            oob_counts[orig] += 1
            else:
                # No bootstrapping: train on full dataset.
                tree.fit[feat_dtype, target_dtype](X, y)

            # Accumulate MDI importances from this tree.
            var tree_imp = compute_tree_feature_importances(
                tree.tree_, self.n_features_in_, n_samples
            )
            for j in range(self.n_features_in_):
                raw_importances[j] += tree_imp[j]

            self.estimators_.append(tree^)

        # L1-normalize directly — dividing by n_estimators before normalization
        # is mathematically redundant since the sum cancels in the ratio.
        self.feature_importances_ = normalize_feature_importances(
            raw_importances, self.n_features_in_
        )

        # Compute OOB R² if requested.
        if self.oob_score and self.bootstrap:
            var ss_res: Float64 = 0.0
            var ss_tot: Float64 = 0.0
            var y_mean: Float64 = 0.0
            var oob_n: Int = 0

            for i in range(n_samples):
                if oob_counts[i] > 0:
                    y_mean += y[i].cast[DType.float64]()
                    oob_n += 1
            if oob_n > 0:
                y_mean /= Float64(oob_n)

            for i in range(n_samples):
                if oob_counts[i] > 0:
                    var pred = oob_preds[i] / Float64(oob_counts[i])
                    var diff = y[i].cast[DType.float64]() - pred
                    var mean_diff = y[i].cast[DType.float64]() - y_mean
                    ss_res += diff * diff
                    ss_tot += mean_diff * mean_diff

            self.oob_score_ = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        self.is_fitted = True

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predicts regression targets as the arithmetic mean across all tree predictions.
        """
        check_is_fitted("RandomForestRegressor", self.is_fitted)
        check_array(X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                String(self.n_features_in_) + " features",
                String(X.cols) + " features",
                "predict",
            )

        var n_samples = X.rows
        var sums = List[Float64](capacity=n_samples)
        for _ in range(n_samples):
            sums.append(0.0)

        for t in range(len(self.estimators_)):
            var preds = self.estimators_[t].predict[feat_dtype](X)
            for i in range(n_samples):
                sums[i] += preds[i].cast[DType.float64]()

        var n_trees = Float64(len(self.estimators_))
        var out = List[Scalar[feat_dtype]](capacity=n_samples)
        for i in range(n_samples):
            out.append((sums[i] / n_trees).cast[feat_dtype]())
        return out^

    def get_n_estimators(self) -> Int:
        """Returns the number of fitted trees."""
        return len(self.estimators_)

    def get_feature_importances(self) raises -> List[Float64]:
        """Returns normalized MDI feature importances (sums to 1.0)."""
        check_is_fitted("RandomForestRegressor", self.is_fitted)
        return self.feature_importances_.copy()

    def get_oob_score(self) raises -> Float64:
        """Returns out-of-bag R² score. Requires oob_score=True and bootstrap=True.
        """
        check_is_fitted("RandomForestRegressor", self.is_fitted)
        if not self.oob_score:
            raise InvalidParameterError.error(
                "oob_score",
                "oob_score was not enabled at construction time",
            )
        return self.oob_score_

    def serialize(self, mut writer: BufferWriter):
        """Serializes RandomForestRegressor parameters and fitted state into BufferWriter.
        """
        write_header(writer, "RandomForestRegressor")
        writer.write_bool(self.is_fitted)
        writer.write_int(self.n_estimators)
        writer.write_string(self.criterion)
        writer.write_int(self.max_depth)
        writer.write_int(self.min_samples_split)
        writer.write_int(self.min_samples_leaf)
        writer.write_float64(self.min_impurity_decrease)
        writer.write_string(self.max_features)
        writer.write_int(self.max_features_count)
        writer.write_float64(self.max_features_ratio)
        writer.write_bool(self.bootstrap)
        writer.write_float64(self.max_samples_ratio)
        writer.write_int(self.max_samples_count)
        writer.write_bool(self.oob_score)
        writer.write_int(self.random_state)
        writer.write_int(self.n_features_in_)
        writer.write_float64_list(self.feature_importances_)
        writer.write_float64(self.oob_score_)

        writer.write_int(len(self.estimators_))
        for i in range(len(self.estimators_)):
            self.estimators_[i].serialize(writer)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes RandomForestRegressor from BufferReader."""
        check_header(reader, "RandomForestRegressor")
        var is_fitted = reader.read_bool()
        var n_estimators = reader.read_int()
        var criterion = reader.read_string()
        var max_depth = reader.read_int()
        var min_samples_split = reader.read_int()
        var min_samples_leaf = reader.read_int()
        var min_impurity_decrease = reader.read_float64()
        var max_features = reader.read_string()
        var max_features_count = reader.read_int()
        var max_features_ratio = reader.read_float64()
        var bootstrap = reader.read_bool()
        var max_samples_ratio = reader.read_float64()
        var max_samples_count = reader.read_int()
        var oob_score = reader.read_bool()
        var random_state = reader.read_int()
        var n_features_in_ = reader.read_int()
        var feature_importances_ = reader.read_float64_list()
        var oob_score_ = reader.read_float64()

        var n_trees = reader.read_int()
        var estimators_ = List[DecisionTreeRegressor[Self.compute_dtype]](
            capacity=n_trees
        )
        for _ in range(n_trees):
            estimators_.append(
                DecisionTreeRegressor[Self.compute_dtype].deserialize(reader)
            )

        var model = Self(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_features_count=max_features_count,
            max_features_ratio=max_features_ratio,
            bootstrap=bootstrap,
            max_samples_ratio=max_samples_ratio,
            max_samples_count=max_samples_count,
            oob_score=oob_score,
            random_state=random_state,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.feature_importances_ = feature_importances_^
        model.oob_score_ = oob_score_
        model.estimators_ = estimators_^
        return model^


struct RandomForestClassifier[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable, Serializable):
    """Random Forest Classifier ensemble estimator.

    An ensemble of decision trees trained via bootstrap aggregation (bagging).
    Predictions are computed via soft voting (averaging predicted class probabilities
    across all trees and selecting the argmax class).

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        n_estimators: Number of trees in the forest. Default 100.
        criterion: Impurity split criterion ('gini', 'entropy', 'log_loss'). Default 'gini'.
        max_depth: Maximum tree depth. -1 means unlimited. Default -1.
        min_samples_split: Minimum samples required to split an internal node. Default 2.
        min_samples_leaf: Minimum samples required to be a leaf node. Default 1.
        min_impurity_decrease: Split threshold if impurity decrease >= this value. Default 0.0.

        max_features: Number of features to consider per split ('all', 'sqrt', 'log2'). Default 'sqrt'.
        max_features_count: Exact number of features per split. Default -1 (disabled).
        max_features_ratio: Proportion of features per split. Default 0.0 (disabled).
        bootstrap: Whether to use bootstrap sampling. Default True.
        max_samples_ratio: Proportion of samples drawn per tree when bootstrap=True. Default 1.0.
        max_samples_count: Exact number of samples drawn per tree. Default -1 (disabled).
        oob_score: Whether to compute out-of-bag accuracy score after fitting. Default False.
        random_state: PRNG seed for deterministic tree builds. Default 42.

    Attributes:
        classes_: Sorted list of unique class labels seen during fit.
        n_features_in_: Number of features seen during fit.
        feature_importances_: Normalized impurity feature importance vector.
        oob_score_: Out-of-bag accuracy score (available when oob_score=True).
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.ensemble import RandomForestClassifier
        from strata.core import Matrix

        var rf = RandomForestClassifier[DType.float64](n_estimators=50, max_depth=6)
        rf.fit(X_train, y_train)
        var preds = rf.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var n_estimators: Int
    var criterion: String
    var max_depth: Int
    var min_samples_split: Int
    var min_samples_leaf: Int
    var min_impurity_decrease: Float64
    var max_features: String
    var max_features_count: Int
    var max_features_ratio: Float64
    var bootstrap: Bool
    var max_samples_ratio: Float64
    var max_samples_count: Int
    var oob_score: Bool
    var random_state: Int

    # Fitted attributes
    var n_features_in_: Int
    var n_classes_: Int
    var classes_: List[Int]
    var estimators_: List[DecisionTreeClassifier[Self.compute_dtype]]
    var feature_importances_: List[Float64]
    var oob_score_: Float64
    var rng: PRNG

    def __init__(
        out self,
        n_estimators: Int = 100,
        criterion: String = "gini",
        max_depth: Int = -1,
        min_samples_split: Int = 2,
        min_samples_leaf: Int = 1,
        min_impurity_decrease: Float64 = 0.0,
        max_features: String = "sqrt",
        max_features_count: Int = -1,
        max_features_ratio: Float64 = 0.0,
        bootstrap: Bool = True,
        max_samples_ratio: Float64 = 1.0,
        max_samples_count: Int = -1,
        oob_score: Bool = False,
        random_state: Int = 42,
    ) raises:
        check_floating_dtype[Self.compute_dtype, "RandomForestClassifier"]()

        if n_estimators < 1:
            raise InvalidParameterError.error(
                "n_estimators",
                "n_estimators must be >= 1, got " + String(n_estimators),
            )

        if (
            criterion != "gini"
            and criterion != "entropy"
            and criterion != "log_loss"
        ):
            raise InvalidParameterError.error(
                "criterion",
                "RandomForestClassifier requires criterion in ['gini',"
                " 'entropy', 'log_loss'], got '"
                + criterion
                + "'",
            )

        if max_depth < 1 and max_depth != -1:
            raise InvalidParameterError.error(
                "max_depth",
                "max_depth must be >= 1 or -1 (unlimited), got "
                + String(max_depth),
            )

        if min_samples_split < 2:
            raise InvalidParameterError.error(
                "min_samples_split",
                "min_samples_split must be >= 2, got "
                + String(min_samples_split),
            )

        if min_samples_leaf < 1:
            raise InvalidParameterError.error(
                "min_samples_leaf",
                "min_samples_leaf must be >= 1, got "
                + String(min_samples_leaf),
            )

        if min_impurity_decrease < 0.0:
            raise InvalidParameterError.error(
                "min_impurity_decrease",
                "min_impurity_decrease must be >= 0.0, got "
                + String(min_impurity_decrease),
            )

        if (
            max_features_count == -1
            and max_features_ratio == 0.0
            and max_features != "all"
            and max_features != "sqrt"
            and max_features != "log2"
        ):
            raise InvalidParameterError.error(
                "max_features",
                "max_features must be 'all', 'sqrt', or 'log2', got '"
                + max_features
                + "'",
            )

        if max_features_count != -1 and max_features_count < 1:
            raise InvalidParameterError.error(
                "max_features_count",
                "max_features_count must be >= 1, got "
                + String(max_features_count),
            )

        if max_features_ratio != 0.0 and (
            max_features_ratio <= 0.0 or max_features_ratio > 1.0
        ):
            raise InvalidParameterError.error(
                "max_features_ratio",
                "max_features_ratio must be in (0.0, 1.0], got "
                + String(max_features_ratio),
            )

        if max_samples_ratio <= 0.0 or max_samples_ratio > 1.0:
            raise InvalidParameterError.error(
                "max_samples_ratio",
                "max_samples_ratio must be in (0.0, 1.0], got "
                + String(max_samples_ratio),
            )

        if max_samples_count != -1 and max_samples_count < 1:
            raise InvalidParameterError.error(
                "max_samples_count",
                "max_samples_count must be >= 1, got "
                + String(max_samples_count),
            )

        if oob_score and not bootstrap:
            raise InvalidParameterError.error(
                "oob_score",
                "oob_score=True requires bootstrap=True",
            )

        self.is_fitted = False
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_features_count = max_features_count
        self.max_features_ratio = max_features_ratio
        self.bootstrap = bootstrap
        self.max_samples_ratio = max_samples_ratio
        self.max_samples_count = max_samples_count
        self.oob_score = oob_score
        self.random_state = random_state

        self.n_features_in_ = 0
        self.n_classes_ = 0
        self.classes_ = List[Int]()
        self.estimators_ = List[DecisionTreeClassifier[Self.compute_dtype]]()
        self.feature_importances_ = List[Float64]()
        self.oob_score_ = 0.0
        self.rng = PRNG(random_state if random_state >= 0 else 42)

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.n_estimators = copy.n_estimators
        self.criterion = copy.criterion
        self.max_depth = copy.max_depth
        self.min_samples_split = copy.min_samples_split
        self.min_samples_leaf = copy.min_samples_leaf
        self.min_impurity_decrease = copy.min_impurity_decrease
        self.max_features = copy.max_features
        self.max_features_count = copy.max_features_count
        self.max_features_ratio = copy.max_features_ratio
        self.bootstrap = copy.bootstrap
        self.max_samples_ratio = copy.max_samples_ratio
        self.max_samples_count = copy.max_samples_count
        self.oob_score = copy.oob_score
        self.random_state = copy.random_state
        self.n_features_in_ = copy.n_features_in_
        self.n_classes_ = copy.n_classes_
        self.classes_ = copy.classes_.copy()
        self.estimators_ = List[DecisionTreeClassifier[Self.compute_dtype]]()
        for i in range(len(copy.estimators_)):
            self.estimators_.append(
                DecisionTreeClassifier[Self.compute_dtype](
                    copy=copy.estimators_[i]
                )
            )
        self.feature_importances_ = copy.feature_importances_.copy()
        self.oob_score_ = copy.oob_score_
        self.rng = PRNG(copy.random_state if copy.random_state >= 0 else 42)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]],) raises:
        """Fits the random forest classifier on (X, y).

        Each tree is trained on a bootstrap sample (or the full dataset when
        bootstrap=False). After fitting, `feature_importances_` holds averaged
        MDI importances and, when oob_score=True, `oob_score_` holds the OOB accuracy.
        """
        check_X_y(X, y)
        self.n_features_in_ = X.cols

        var n_samples = len(y)
        if n_samples == 0:
            raise InvalidParameterError.error("y", "y cannot be empty")

        # Extract sorted unique class labels
        var sorted_y = List[Int](capacity=n_samples)
        for i in range(n_samples):
            sorted_y.append(Int(y[i]))
        sort(sorted_y)

        var unique_labels = List[Int]()
        unique_labels.append(sorted_y[0])
        for i in range(1, n_samples):
            if sorted_y[i] != sorted_y[i - 1]:
                unique_labels.append(sorted_y[i])

        self.classes_ = unique_labels.copy()
        self.n_classes_ = len(unique_labels)

        self.estimators_ = List[DecisionTreeClassifier[Self.compute_dtype]](
            capacity=self.n_estimators
        )
        self.rng = PRNG(self.random_state if self.random_state >= 0 else 42)

        # OOB accumulation buffers: class probability sums and counts per sample
        var oob_probs = Matrix[DType.float64](n_samples, self.n_classes_, 0.0)
        var oob_counts = List[Int](capacity=n_samples)
        for _ in range(n_samples):
            oob_counts.append(0)

        # Accumulate raw (un-normalized) MDI importances across all trees
        var raw_importances = List[Float64](capacity=self.n_features_in_)
        for _ in range(self.n_features_in_):
            raw_importances.append(0.0)

        for _ in range(self.n_estimators):
            # Derive independent PRNG streams for tree seed and bootstrap generator
            var tree_seed = Int(self.rng.next_u64() % UInt64(2147483647))
            var boot_rng = PRNG(Int(self.rng.next_u64() % UInt64(2147483647)))

            var tree = DecisionTreeClassifier[Self.compute_dtype](
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_features_count=self.max_features_count,
                max_features_ratio=self.max_features_ratio,
                random_state=tree_seed,
            )

            if self.bootstrap:
                var sample_indices = generate_bootstrap_indices(
                    n_samples,
                    self.max_samples_count,
                    self.max_samples_ratio,
                    boot_rng,
                )

                # Build bootstrap X and y views (index-filtered copies)
                var k = len(sample_indices)
                var X_boot = Matrix[feat_dtype](k, X.cols)
                var y_boot = List[Scalar[target_dtype]](capacity=k)
                for i in range(k):
                    var src = sample_indices[i]
                    for j in range(X.cols):
                        X_boot[i, j] = X[src, j]
                    y_boot.append(y[src])

                tree.fit[feat_dtype, target_dtype](X_boot, y_boot)

                if self.oob_score:
                    var oob_mask = get_oob_mask(n_samples, sample_indices)
                    var oob_row_indices = List[Int]()
                    for i in range(n_samples):
                        if oob_mask[i]:
                            oob_row_indices.append(i)

                    var n_oob = len(oob_row_indices)
                    if n_oob > 0:
                        var X_oob = Matrix[feat_dtype](n_oob, X.cols)
                        for i in range(n_oob):
                            var src = oob_row_indices[i]
                            for j in range(X.cols):
                                X_oob[i, j] = X[src, j]

                        var tree_oob_probs = tree.predict_proba[feat_dtype](
                            X_oob
                        )

                        # Map tree class indices to global forest class indices
                        var class_map = List[Int](capacity=tree.n_classes_)
                        for c in range(tree.n_classes_):
                            class_map.append(
                                _find_class_index(
                                    self.classes_, tree.classes_[c]
                                )
                            )

                        for r in range(n_oob):
                            var orig = oob_row_indices[r]
                            for c in range(tree.n_classes_):
                                var forest_c = class_map[c]
                                if forest_c >= 0:
                                    oob_probs[orig, forest_c] += Float64(
                                        tree_oob_probs[r, c]
                                    )
                            oob_counts[orig] += 1
            else:
                # No bootstrapping: train on full dataset
                tree.fit[feat_dtype, target_dtype](X, y)

            # Accumulate MDI importances from this tree
            var tree_imp = compute_tree_feature_importances(
                tree.tree_, self.n_features_in_, n_samples
            )
            for j in range(self.n_features_in_):
                raw_importances[j] += tree_imp[j]

            self.estimators_.append(tree^)

        # L1-normalize MDI feature importances
        self.feature_importances_ = normalize_feature_importances(
            raw_importances, self.n_features_in_
        )

        # Compute OOB accuracy if requested
        if self.oob_score and self.bootstrap:
            var correct: Int = 0
            var oob_total: Int = 0

            for i in range(n_samples):
                if oob_counts[i] > 0:
                    oob_total += 1
                    var max_c = 0
                    var max_p = oob_probs[i, 0]
                    for c in range(1, self.n_classes_):
                        if oob_probs[i, c] > max_p:
                            max_p = oob_probs[i, c]
                            max_c = c
                    var pred_label = self.classes_[max_c]
                    if pred_label == Int(y[i]):
                        correct += 1

            self.oob_score_ = (
                Float64(correct) / Float64(oob_total) if oob_total > 0 else 0.0
            )

        self.is_fitted = True

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Generates class probability estimates for input matrix X by averaging tree probabilities.
        """
        check_is_fitted("RandomForestClassifier", self.is_fitted)
        check_array(X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                String(self.n_features_in_) + " features",
                String(X.cols) + " features",
                "predict_proba",
            )

        var n_rows = X.rows
        var prob_sums = Matrix[DType.float64](n_rows, self.n_classes_, 0.0)

        for t in range(len(self.estimators_)):
            var tree_probs = self.estimators_[t].predict_proba[feat_dtype](X)
            var class_map = List[Int](capacity=self.estimators_[t].n_classes_)
            for c in range(self.estimators_[t].n_classes_):
                class_map.append(
                    _find_class_index(
                        self.classes_, self.estimators_[t].classes_[c]
                    )
                )

            for r in range(n_rows):
                for c in range(self.estimators_[t].n_classes_):
                    var forest_c = class_map[c]
                    if forest_c >= 0:
                        prob_sums[r, forest_c] += Float64(tree_probs[r, c])

        var n_trees = Float64(len(self.estimators_))
        var out = Matrix[feat_dtype](n_rows, self.n_classes_)
        for r in range(n_rows):
            for c in range(self.n_classes_):
                out[r, c] = Scalar[feat_dtype](prob_sums[r, c] / n_trees)

        return out^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Generates discrete class predictions via soft-voting argmax over predicted class probabilities.
        """
        check_is_fitted("RandomForestClassifier", self.is_fitted)
        var proba = self.predict_proba[feat_dtype](X)

        var n_rows = X.rows
        var preds = List[Int](capacity=n_rows)
        for r in range(n_rows):
            var max_c = 0
            var max_p = proba[r, 0]
            for c in range(1, self.n_classes_):
                if proba[r, c] > max_p:
                    max_p = proba[r, c]
                    max_c = c
            preds.append(self.classes_[max_c])

        return preds^

    def get_n_estimators(self) -> Int:
        """Returns the number of fitted trees."""
        return len(self.estimators_)

    def get_feature_importances(self) raises -> List[Float64]:
        """Returns normalized MDI feature importances (sums to 1.0)."""
        check_is_fitted("RandomForestClassifier", self.is_fitted)
        return self.feature_importances_.copy()

    def get_oob_score(self) raises -> Float64:
        """Returns out-of-bag accuracy score. Requires oob_score=True and bootstrap=True.
        """
        check_is_fitted("RandomForestClassifier", self.is_fitted)
        if not self.oob_score:
            raise InvalidParameterError.error(
                "oob_score",
                "oob_score was not enabled at construction time",
            )
        return self.oob_score_

    def get_classes(self) raises -> List[Int]:
        """Returns the sorted list of known class labels."""
        check_is_fitted("RandomForestClassifier", self.is_fitted)
        return self.classes_.copy()

    def serialize(self, mut writer: BufferWriter):
        """Serializes RandomForestClassifier parameters and fitted state into BufferWriter.
        """
        write_header(writer, "RandomForestClassifier")
        writer.write_bool(self.is_fitted)
        writer.write_int(self.n_estimators)
        writer.write_string(self.criterion)
        writer.write_int(self.max_depth)
        writer.write_int(self.min_samples_split)
        writer.write_int(self.min_samples_leaf)
        writer.write_float64(self.min_impurity_decrease)
        writer.write_string(self.max_features)
        writer.write_int(self.max_features_count)
        writer.write_float64(self.max_features_ratio)
        writer.write_bool(self.bootstrap)
        writer.write_float64(self.max_samples_ratio)
        writer.write_int(self.max_samples_count)
        writer.write_bool(self.oob_score)
        writer.write_int(self.random_state)
        writer.write_int(self.n_features_in_)
        writer.write_int(self.n_classes_)
        writer.write_int_list(self.classes_)
        writer.write_float64_list(self.feature_importances_)
        writer.write_float64(self.oob_score_)

        writer.write_int(len(self.estimators_))
        for i in range(len(self.estimators_)):
            self.estimators_[i].serialize(writer)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes RandomForestClassifier from BufferReader."""
        check_header(reader, "RandomForestClassifier")
        var is_fitted = reader.read_bool()
        var n_estimators = reader.read_int()
        var criterion = reader.read_string()
        var max_depth = reader.read_int()
        var min_samples_split = reader.read_int()
        var min_samples_leaf = reader.read_int()
        var min_impurity_decrease = reader.read_float64()
        var max_features = reader.read_string()
        var max_features_count = reader.read_int()
        var max_features_ratio = reader.read_float64()
        var bootstrap = reader.read_bool()
        var max_samples_ratio = reader.read_float64()
        var max_samples_count = reader.read_int()
        var oob_score = reader.read_bool()
        var random_state = reader.read_int()
        var n_features_in_ = reader.read_int()
        var n_classes_ = reader.read_int()
        var classes_ = reader.read_int_list()
        var feature_importances_ = reader.read_float64_list()
        var oob_score_ = reader.read_float64()

        var n_trees = reader.read_int()
        var estimators_ = List[DecisionTreeClassifier[Self.compute_dtype]](
            capacity=n_trees
        )
        for _ in range(n_trees):
            estimators_.append(
                DecisionTreeClassifier[Self.compute_dtype].deserialize(reader)
            )

        var model = Self(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_features_count=max_features_count,
            max_features_ratio=max_features_ratio,
            bootstrap=bootstrap,
            max_samples_ratio=max_samples_ratio,
            max_samples_count=max_samples_count,
            oob_score=oob_score,
            random_state=random_state,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.n_classes_ = n_classes_
        model.classes_ = classes_^
        model.feature_importances_ = feature_importances_^
        model.oob_score_ = oob_score_
        model.estimators_ = estimators_^
        return model^
