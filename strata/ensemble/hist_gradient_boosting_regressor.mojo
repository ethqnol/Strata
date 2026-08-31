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
from ._binning import BinnedMatrix, _compute_bin_thresholds, _map_to_bins
from ._gb_loss import LeastSquaresLoss
from ._hist_tree import HistTree


struct HistGradientBoostingRegressor[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor):
    """Histogram-based Gradient Boosting Regressor.

    Builds an ensemble of regression trees iteratively fit to negative gradients
    of the squared error loss function using discrete UInt8 binning and
    histogram subtraction.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Attributes:
        loss: Loss function to optimize. Currently 'squared_error'. Default 'squared_error'.
        learning_rate: Shrinkage multiplier for tree leaf outputs $\\eta \\in (0, 1]$. Default 0.1.
        max_iter: Maximum number of boosting iterations (number of trees). Default 100.
        max_leaf_nodes: Maximum number of leaves per tree. Default 31.
        max_depth: Maximum tree depth. Default 6.
        min_samples_leaf: Minimum samples required per leaf node. Default 20.
        l2_regularization: $L_2$ leaf weight regularization parameter $\\lambda \\ge 0$. Default 0.0.
        max_bins: Maximum discrete histogram bins ($2 \\le \\text{max\\_bins} \\le 256$). Default 256.
        min_gain_to_split: Minimum gain required to split an internal node. Default 0.0.
        early_stopping: Whether to use validation-based early stopping. Default True.
        validation_fraction: Proportion of training data set aside for early stopping. Default 0.1.
        n_iter_no_change: Maximum consecutive iterations with non-improving validation loss. Default 10.
        tol: Minimum relative loss improvement threshold. Default 1e-7.
        random_state: Seed for PRNG shuffling. Default 42.

    Fitted Attributes:
        init_raw_prediction_: Baseline raw prediction offset (sample mean $\\bar{y}$).
        bin_thresholds_: Discovered continuous partition boundaries per feature.
        trees_: List of sequential fitted HistTree models.
        n_iter_: Actual number of boosting iterations completed.
        is_fitted: Boolean flag indicating if estimator has been fitted.
        n_features_in_: Number of feature columns seen during fit.
    """

    var loss: String
    var learning_rate: Float64
    var max_iter: Int
    var max_leaf_nodes: Int
    var max_depth: Int
    var min_samples_leaf: Int
    var l2_regularization: Float64
    var max_bins: Int
    var min_gain_to_split: Float64
    var early_stopping: Bool
    var validation_fraction: Float64
    var n_iter_no_change: Int
    var tol: Float64
    var random_state: Int

    var init_raw_prediction_: Float64
    var bin_thresholds_: List[List[Float64]]
    var trees_: List[HistTree]
    var n_iter_: Int
    var is_fitted: Bool
    var n_features_in_: Int

    def __init__(
        out self,
        loss: String = "squared_error",
        learning_rate: Float64 = 0.1,
        max_iter: Int = 100,
        max_leaf_nodes: Int = 31,
        max_depth: Int = 6,
        min_samples_leaf: Int = 20,
        l2_regularization: Float64 = 0.0,
        max_bins: Int = 256,
        min_gain_to_split: Float64 = 0.0,
        early_stopping: Bool = True,
        validation_fraction: Float64 = 0.1,
        n_iter_no_change: Int = 10,
        tol: Float64 = 1e-7,
        random_state: Int = 42,
    ) raises:
        """Initializes a HistGradientBoostingRegressor with validated hyperparameters."""
        if loss != "squared_error":
            raise InvalidParameterError.error(
                "loss",
                "Supported loss is 'squared_error', got '" + loss + "'",
            )
        if learning_rate <= 0.0:
            raise InvalidParameterError.error(
                "learning_rate",
                "learning_rate must be > 0, got " + String(learning_rate),
            )
        if max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter", "max_iter must be > 0, got " + String(max_iter)
            )
        if max_leaf_nodes < 2:
            raise InvalidParameterError.error(
                "max_leaf_nodes",
                "max_leaf_nodes must be >= 2, got " + String(max_leaf_nodes),
            )
        if max_depth < 1:
            raise InvalidParameterError.error(
                "max_depth", "max_depth must be >= 1, got " + String(max_depth)
            )
        if min_samples_leaf < 1:
            raise InvalidParameterError.error(
                "min_samples_leaf",
                "min_samples_leaf must be >= 1, got "
                + String(min_samples_leaf),
            )
        if l2_regularization < 0.0:
            raise InvalidParameterError.error(
                "l2_regularization",
                "l2_regularization must be >= 0, got "
                + String(l2_regularization),
            )
        if max_bins < 2 or max_bins > 256:
            raise InvalidParameterError.error(
                "max_bins",
                "max_bins must be between 2 and 256, got " + String(max_bins),
            )
        if (
            early_stopping
            and (validation_fraction <= 0.0 or validation_fraction >= 1.0)
        ):
            raise InvalidParameterError.error(
                "validation_fraction",
                "validation_fraction must be in (0, 1), got "
                + String(validation_fraction),
            )
        if n_iter_no_change < 1:
            raise InvalidParameterError.error(
                "n_iter_no_change",
                "n_iter_no_change must be >= 1, got "
                + String(n_iter_no_change),
            )
        if tol < 0.0:
            raise InvalidParameterError.error(
                "tol", "tol must be >= 0, got " + String(tol)
            )

        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.min_gain_to_split = min_gain_to_split
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

        self.init_raw_prediction_ = 0.0
        self.bin_thresholds_ = List[List[Float64]]()
        self.trees_ = List[HistTree]()
        self.n_iter_ = 0
        self.is_fitted = False
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Deep copies an existing HistGradientBoostingRegressor."""
        self.loss = copy.loss
        self.learning_rate = copy.learning_rate
        self.max_iter = copy.max_iter
        self.max_leaf_nodes = copy.max_leaf_nodes
        self.max_depth = copy.max_depth
        self.min_samples_leaf = copy.min_samples_leaf
        self.l2_regularization = copy.l2_regularization
        self.max_bins = copy.max_bins
        self.min_gain_to_split = copy.min_gain_to_split
        self.early_stopping = copy.early_stopping
        self.validation_fraction = copy.validation_fraction
        self.n_iter_no_change = copy.n_iter_no_change
        self.tol = copy.tol
        self.random_state = copy.random_state

        self.init_raw_prediction_ = copy.init_raw_prediction_
        self.bin_thresholds_ = List[List[Float64]](
            capacity=len(copy.bin_thresholds_)
        )
        for j in range(len(copy.bin_thresholds_)):
            self.bin_thresholds_.append(copy.bin_thresholds_[j].copy())

        self.trees_ = List[HistTree](capacity=len(copy.trees_))
        for t in range(len(copy.trees_)):
            self.trees_.append(copy.trees_[t].copy())

        self.n_iter_ = copy.n_iter_
        self.is_fitted = copy.is_fitted
        self.n_features_in_ = copy.n_features_in_

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](
        mut self,
        X: Matrix[feat_dtype],
        y: List[Scalar[target_dtype]],
    ) raises:
        """Fits the ensemble of histogram gradient boosted trees on training data $(X, y)$.

        Args:
            X: Training feature matrix of shape $(N, D)$.
            y: Continuous target labels of length $N$.
        """
        check_X_y[feat_dtype, target_dtype](X, y)
        var N = X.rows
        var D = X.cols
        self.n_features_in_ = D
        self.trees_.clear()

        # Split into training and validation subsets if early_stopping enabled and N >= 20
        var use_early_stopping = self.early_stopping and N >= 20
        var n_val = Int(
            Float64(N) * self.validation_fraction
        ) if use_early_stopping else 0
        if n_val < 2:
            use_early_stopping = False
            n_val = 0

        var n_train = N - n_val
        var rng = PRNG(self.random_state)
        var indices = List[Int](capacity=N)
        for i in range(N):
            indices.append(i)

        if use_early_stopping:
            for i in range(N - 1, 0, -1):
                var j = rng.next_int(i + 1)
                var tmp = indices[i]
                indices[i] = indices[j]
                indices[j] = tmp

        var X_train = Matrix[Self.compute_dtype](n_train, D, 0)
        var y_train = List[Float64](capacity=n_train)
        for i in range(n_train):
            var orig_idx = indices[i]
            y_train.append(Float64(y[orig_idx]))
            for j in range(D):
                X_train[i, j] = Scalar[Self.compute_dtype](X[orig_idx, j])

        var X_val = Matrix[Self.compute_dtype](n_val, D, 0)
        var y_val = List[Float64](capacity=n_val)
        if use_early_stopping:
            for i in range(n_val):
                var orig_idx = indices[n_train + i]
                y_val.append(Float64(y[orig_idx]))
                for j in range(D):
                    X_val[i, j] = Scalar[Self.compute_dtype](X[orig_idx, j])

        # Compute bin thresholds on training data
        self.bin_thresholds_ = _compute_bin_thresholds[Self.compute_dtype](
            X_train, self.max_bins
        )
        var binned_train = _map_to_bins[Self.compute_dtype](
            X_train, self.bin_thresholds_
        )
        var binned_val = _map_to_bins[Self.compute_dtype](
            X_val, self.bin_thresholds_
        ) if use_early_stopping else BinnedMatrix(
            0, D, List[UInt8](), List[List[Float64]](), List[Int]()
        )

        var loss_fn = LeastSquaresLoss()
        self.init_raw_prediction_ = loss_fn.init_raw_prediction(y_train)

        var raw_train = List[Float64](capacity=n_train)
        for _ in range(n_train):
            raw_train.append(self.init_raw_prediction_)

        var raw_val = List[Float64](capacity=n_val)
        if use_early_stopping:
            for _ in range(n_val):
                raw_val.append(self.init_raw_prediction_)

        var gradients = List[Float64]()
        var hessians = List[Float64]()

        var best_val_loss: Float64 = 1e30
        var no_improvement_count = 0
        self.n_iter_ = 0

        for iter_idx in range(self.max_iter):
            self.n_iter_ = iter_idx + 1

            # 1. Update gradients and hessians
            loss_fn.update_gradients_and_hessians(
                y_train, raw_train, gradients, hessians
            )

            # 2. Build histogram tree
            var tree = HistTree(
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                min_gain_to_split=self.min_gain_to_split,
                shrinkage=self.learning_rate,
            )
            tree.build(binned_train, gradients, hessians)

            # 3. Update raw training predictions
            for i in range(n_train):
                raw_train[i] += tree.predict_binned(binned_train, i)

            # 4. Check validation loss if early stopping active
            if use_early_stopping:
                var val_loss_total: Float64 = 0.0
                for i in range(n_val):
                    raw_val[i] += tree.predict_binned(binned_val, i)
                    val_loss_total += loss_fn.loss(y_val[i], raw_val[i])

                var val_loss = val_loss_total / Float64(n_val)

                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.n_iter_no_change:
                        self.trees_.append(tree^)
                        break

            self.trees_.append(tree^)

        self.is_fitted = True

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predicts continuous target values for feature matrix $X$.

        Args:
            X: Feature matrix of shape $(N, D)$ to predict on.

        Returns:
            List[Scalar[feat_dtype]]: Predicted target vector of length $N$.
        """
        check_is_fitted("HistGradientBoostingRegressor", self.is_fitted)
        check_array[feat_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "HistGradientBoostingRegressor.predict",
            )

        var N = X.rows
        var raw_preds = List[Float64](capacity=N)
        for _ in range(N):
            raw_preds.append(self.init_raw_prediction_)

        for t in range(len(self.trees_)):
            var tree_preds = self.trees_[t].predict[feat_dtype](X)
            for i in range(N):
                raw_preds[i] += tree_preds[i]

        var res = List[Scalar[feat_dtype]](capacity=N)
        for i in range(N):
            res.append(Scalar[feat_dtype](raw_preds[i]))

        return res^

    def score[
        feat_dtype: DType, target_dtype: DType
    ](
        self,
        X: Matrix[feat_dtype],
        y: List[Scalar[target_dtype]],
    ) raises -> Float64:
        """Returns the coefficient of determination $R^2$ on test data $(X, y)$."""
        var preds = self.predict(X)
        var N = len(y)
        if N == 0:
            return 0.0

        var sum_y: Float64 = 0.0
        for i in range(N):
            sum_y += Float64(y[i])
        var mean_y = sum_y / Float64(N)

        var ss_tot: Float64 = 0.0
        var ss_res: Float64 = 0.0
        for i in range(N):
            var diff_tot = Float64(y[i]) - mean_y
            var diff_res = Float64(y[i]) - Float64(preds[i])
            ss_tot += diff_tot * diff_tot
            ss_res += diff_res * diff_res

        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0

        return 1.0 - (ss_res / ss_tot)
