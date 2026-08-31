from ..core.matrix import Matrix
from ..base.estimator import Regressor
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
from .tree import Node, Tree
from .criterion import (
    squared_error_impurity,
    absolute_error_impurity,
)
from .splitter import find_best_split_regression


struct DecisionTreeRegressor[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor, Serializable):
    """Decision Tree Regressor for non-parametric continuous target regression.

    Builds a regression tree by minimizing sample variance (mean squared error)
    or mean absolute deviation across recursive binary splits:

    $$
    H_{\\text{MSE}}(Q) = \\frac{1}{|Q|} \\sum_{i \\in Q} (y_i - \\bar{y}_Q)^2
    $$

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        criterion: The function to measure split quality ('squared_error', 'friedman_mse', 'absolute_error'). Default 'squared_error'.
        splitter: Strategy used to choose the split at each node ('best', 'random'). Default 'best'.
        max_depth: Maximum tree depth. -1 indicates unlimited depth. Default -1.
        min_samples_split: Minimum samples required to split an internal node. Default 2.
        min_samples_leaf: Minimum samples required to be at a leaf node. Default 1.
        min_impurity_decrease: Split threshold if impurity decrease >= this value. Default 0.0.

        max_features: Number of features to consider when looking for best split ('all', 'sqrt', 'log2', 'custom'). Default 'all'.
        max_features_count: Explicit number of features to evaluate when max_features='custom'. Default -1.
        max_features_ratio: Proportion of features to evaluate when max_features='custom'. Default 0.0.
        random_state: PRNG seed for deterministic feature and split selection. Default 42.

    Attributes:
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.tree import DecisionTreeRegressor
        from strata.core import Matrix

        var reg = DecisionTreeRegressor[DType.float64](max_depth=4)
        reg.fit(X_train, y_train)
        var preds = reg.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var criterion: String
    var splitter: String
    var max_depth: Int
    var min_samples_split: Int
    var min_samples_leaf: Int
    var min_impurity_decrease: Float64
    var max_features: String
    var max_features_count: Int
    var max_features_ratio: Float64
    var random_state: Int
    var n_features_in_: Int
    var tree_: Tree
    var rng: PRNG

    def __init__(
        out self,
        criterion: String = "squared_error",
        splitter: String = "best",
        max_depth: Int = -1,
        min_samples_split: Int = 2,
        min_samples_leaf: Int = 1,
        min_impurity_decrease: Float64 = 0.0,
        max_features: String = "all",
        max_features_count: Int = -1,
        max_features_ratio: Float64 = 0.0,
        random_state: Int = 42,
    ) raises:
        check_floating_dtype[Self.compute_dtype, "DecisionTreeRegressor"]()

        if (
            criterion != "squared_error"
            and criterion != "friedman_mse"
            and criterion != "absolute_error"
        ):
            raise InvalidParameterError.error(
                "criterion",
                "DecisionTreeRegressor requires criterion in ['squared_error',"
                " 'friedman_mse', 'absolute_error'], got '"
                + criterion
                + "'",
            )

        if splitter != "best" and splitter != "random":
            raise InvalidParameterError.error(
                "splitter",
                "DecisionTreeRegressor requires splitter in ['best',"
                " 'random'], got '"
                + splitter
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

        self.is_fitted = False
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_features_count = max_features_count
        self.max_features_ratio = max_features_ratio
        self.random_state = random_state
        self.n_features_in_ = 0
        self.tree_ = Tree()
        self.rng = PRNG(random_state if random_state >= 0 else 42)

    def __init__(out self, *, copy: Self):
        self.is_fitted = copy.is_fitted
        self.criterion = copy.criterion
        self.splitter = copy.splitter
        self.max_depth = copy.max_depth
        self.min_samples_split = copy.min_samples_split
        self.min_samples_leaf = copy.min_samples_leaf
        self.min_impurity_decrease = copy.min_impurity_decrease
        self.max_features = copy.max_features
        self.max_features_count = copy.max_features_count
        self.max_features_ratio = copy.max_features_ratio
        self.random_state = copy.random_state
        self.n_features_in_ = copy.n_features_in_
        self.tree_ = copy.tree_.copy()
        self.rng = PRNG(copy.random_state if copy.random_state >= 0 else 42)

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]],) raises:
        """Fits the decision tree regressor on (X, y)."""
        check_X_y(X, y)
        self.n_features_in_ = X.cols
        self.tree_ = Tree()
        self.rng = PRNG(self.random_state if self.random_state >= 0 else 42)

        var n_samples = X.rows
        var initial_indices = List[Int](capacity=n_samples)
        for i in range(n_samples):
            initial_indices.append(i)

        _ = self._build_tree(X, y, initial_indices, 0)
        self.is_fitted = True

    def _build_tree[
        feat_dtype: DType, target_dtype: DType
    ](
        mut self,
        X: Matrix[feat_dtype],
        y: List[Scalar[target_dtype]],
        node_indices: List[Int],
        depth: Int,
    ) -> Int:
        var n_samples = len(node_indices)
        if n_samples == 0:
            return -1

        var node_val: Float64
        if self.criterion == "absolute_error":
            var vals = List[Float64](capacity=n_samples)
            for i in range(n_samples):
                vals.append(y[node_indices[i]].cast[DType.float64]())
            sort(vals)
            node_val = vals[n_samples // 2]
        else:
            var sum_y: Float64 = 0.0
            for i in range(n_samples):
                sum_y += y[node_indices[i]].cast[DType.float64]()
            node_val = sum_y / Float64(n_samples)

        var impurity: Float64
        if self.criterion == "absolute_error":
            impurity = absolute_error_impurity(y, node_indices)
        else:
            impurity = squared_error_impurity(y, node_indices)

        var stop_due_to_depth = self.max_depth != -1 and depth >= self.max_depth
        var stop_due_to_samples = n_samples < self.min_samples_split
        var stop_due_to_impurity = impurity <= 1e-12

        if stop_due_to_depth or stop_due_to_samples or stop_due_to_impurity:
            var leaf_node = Node(node_val, impurity, n_samples)
            return self.tree_.add_node(leaf_node^)

        var split = find_best_split_regression(
            X,
            y,
            node_indices,
            self.criterion,
            self.splitter,
            self.max_features,
            self.max_features_count,
            self.max_features_ratio,
            self.min_samples_split,
            self.min_samples_leaf,
            self.min_impurity_decrease,
            self.rng,
        )

        if (
            not split.found
            or len(split.left_indices) == 0
            or len(split.right_indices) == 0
        ):
            var leaf_node = Node(node_val, impurity, n_samples)
            return self.tree_.add_node(leaf_node^)

        var split_node = Node(
            split.feature_idx,
            split.threshold,
            -1,
            -1,
            impurity,
            n_samples,
            value_reg=node_val,
        )
        var node_idx = self.tree_.add_node(split_node^)

        var left_child = self._build_tree(X, y, split.left_indices, depth + 1)
        var right_child = self._build_tree(X, y, split.right_indices, depth + 1)

        self.tree_.nodes[node_idx].left_child = left_child
        self.tree_.nodes[node_idx].right_child = right_child
        return node_idx

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Generates regression predictions for input matrix X."""
        check_is_fitted("DecisionTreeRegressor", self.is_fitted)
        check_array(X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                String(self.n_features_in_) + " features",
                String(X.cols) + " features",
                "predict",
            )

        return self.tree_.predict_regression[feat_dtype, feat_dtype](X)

    def get_depth(self) raises -> Int:
        """Returns the maximum depth of the fitted tree."""
        check_is_fitted("DecisionTreeRegressor", self.is_fitted)
        return self.tree_.max_depth()

    def get_n_leaves(self) raises -> Int:
        """Returns the total number of leaf nodes in the fitted tree."""
        check_is_fitted("DecisionTreeRegressor", self.is_fitted)
        var count = 0
        for i in range(len(self.tree_.nodes)):
            if self.tree_.nodes[i].is_leaf:
                count += 1
        return count

    def serialize(self, mut writer: BufferWriter):
        """Serializes DecisionTreeRegressor parameters and fitted state into BufferWriter.
        """
        write_header(writer, "DecisionTreeRegressor")
        writer.write_bool(self.is_fitted)
        writer.write_string(self.criterion)
        writer.write_string(self.splitter)
        writer.write_int(self.max_depth)
        writer.write_int(self.min_samples_split)
        writer.write_int(self.min_samples_leaf)
        writer.write_float64(self.min_impurity_decrease)
        writer.write_string(self.max_features)
        writer.write_int(self.max_features_count)
        writer.write_float64(self.max_features_ratio)
        writer.write_int(self.random_state)
        writer.write_int(self.n_features_in_)
        self.tree_.serialize(writer)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes DecisionTreeRegressor from BufferReader."""
        check_header(reader, "DecisionTreeRegressor")
        var is_fitted = reader.read_bool()
        var criterion = reader.read_string()
        var splitter = reader.read_string()
        var max_depth = reader.read_int()
        var min_samples_split = reader.read_int()
        var min_samples_leaf = reader.read_int()
        var min_impurity_decrease = reader.read_float64()
        var max_features = reader.read_string()
        var max_features_count = reader.read_int()
        var max_features_ratio = reader.read_float64()
        var random_state = reader.read_int()
        var n_features_in_ = reader.read_int()
        var tree_ = Tree.deserialize(reader)

        var model = Self(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_features_count=max_features_count,
            max_features_ratio=max_features_ratio,
            random_state=random_state,
        )
        model.is_fitted = is_fitted
        model.n_features_in_ = n_features_in_
        model.tree_ = tree_^
        return model^
