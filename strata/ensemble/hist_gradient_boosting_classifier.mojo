from ..core.matrix import Matrix
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.random import PRNG
from ..utils.math import sigmoid, softmax
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ._binning import BinnedMatrix, _compute_bin_thresholds, _map_to_bins
from ._gb_loss import BinaryCrossEntropyLoss, MulticlassCrossEntropyLoss
from ._hist_tree import HistTree


def _find_class_idx(classes: List[Int], target: Int) -> Int:
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


struct HistGradientBoostingClassifier[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable):
    """Histogram-based Gradient Boosting Classifier.

    Builds an ensemble of classification trees using discrete UInt8 binning,
    histogram subtraction, and exact 1st/2nd order gradients for binary and
    multiclass cross-entropy loss.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Attributes:
        loss: Loss function to optimize ('log_loss' or 'auto'). Default 'log_loss'.
        learning_rate: Shrinkage multiplier for tree leaf outputs $\\eta \\in (0, 1]$. Default 0.1.
        max_iter: Maximum number of boosting iterations. Default 100.
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
        classes_: Discovered distinct class labels in sorted order.
        init_raw_predictions_: Baseline raw margin offsets per class.
        bin_thresholds_: Discovered continuous partition boundaries per feature.
        trees_: Nested array of fitted trees ($T \\times 1$ for binary, $T \\times K$ for multiclass).
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

    var classes_: List[Int]
    var init_raw_predictions_: List[Float64]
    var bin_thresholds_: List[List[Float64]]
    var trees_: List[List[HistTree]]
    var n_iter_: Int
    var is_fitted: Bool
    var n_features_in_: Int

    def __init__(
        out self,
        loss: String = "log_loss",
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
        """Initializes a HistGradientBoostingClassifier with validated hyperparameters."""
        if loss != "log_loss" and loss != "auto":
            raise InvalidParameterError.error(
                "loss",
                "Supported loss is 'log_loss' or 'auto', got '" + loss + "'",
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

        self.classes_ = List[Int]()
        self.init_raw_predictions_ = List[Float64]()
        self.bin_thresholds_ = List[List[Float64]]()
        self.trees_ = List[List[HistTree]]()
        self.n_iter_ = 0
        self.is_fitted = False
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Deep copies an existing HistGradientBoostingClassifier."""
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

        self.classes_ = copy.classes_.copy()
        self.init_raw_predictions_ = copy.init_raw_predictions_.copy()

        self.bin_thresholds_ = List[List[Float64]](
            capacity=len(copy.bin_thresholds_)
        )
        for j in range(len(copy.bin_thresholds_)):
            self.bin_thresholds_.append(copy.bin_thresholds_[j].copy())

        self.trees_ = List[List[HistTree]](capacity=len(copy.trees_))
        for t in range(len(copy.trees_)):
            var round_trees = List[HistTree](capacity=len(copy.trees_[t]))
            for k in range(len(copy.trees_[t])):
                round_trees.append(copy.trees_[t][k].copy())
            self.trees_.append(round_trees^)

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
        """Fits the ensemble of classification trees on training data $(X, y)$.

        Args:
            X: Training feature matrix of shape $(N, D)$.
            y: Discrete target classification labels of length $N$.
        """
        check_X_y[feat_dtype, target_dtype](X, y)
        var N = X.rows
        var D = X.cols
        self.n_features_in_ = D
        self.trees_.clear()

        # Extract sorted distinct class labels
        var sorted_y = List[Int](capacity=N)
        for i in range(N):
            sorted_y.append(Int(y[i]))
        sort(sorted_y)

        self.classes_.clear()
        self.classes_.append(sorted_y[0])
        for i in range(1, N):
            if sorted_y[i] != sorted_y[i - 1]:
                self.classes_.append(sorted_y[i])

        var K = len(self.classes_)

        # Map y to 0-indexed class targets
        var y_mapped = List[Int](capacity=N)
        for i in range(N):
            var c_idx = _find_class_idx(self.classes_, Int(y[i]))
            y_mapped.append(c_idx)

        # Early stopping partition setup
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
        var y_train = List[Int](capacity=n_train)
        for i in range(n_train):
            var orig_idx = indices[i]
            y_train.append(y_mapped[orig_idx])
            for j in range(D):
                X_train[i, j] = Scalar[Self.compute_dtype](X[orig_idx, j])

        var X_val = Matrix[Self.compute_dtype](n_val, D, 0)
        var y_val = List[Int](capacity=n_val)
        if use_early_stopping:
            for i in range(n_val):
                var orig_idx = indices[n_train + i]
                y_val.append(y_mapped[orig_idx])
                for j in range(D):
                    X_val[i, j] = Scalar[Self.compute_dtype](X[orig_idx, j])

        # Compute bin thresholds
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

        self.init_raw_predictions_.clear()

        # Binary vs Multiclass execution paths
        if K <= 2:
            var loss_bce = BinaryCrossEntropyLoss()
            var y_train_f64 = List[Float64](capacity=n_train)
            for i in range(n_train):
                y_train_f64.append(Float64(y_train[i]))

            var init_pred = loss_bce.init_raw_prediction(y_train_f64)
            self.init_raw_predictions_.append(init_pred)

            var raw_train = List[Float64](capacity=n_train)
            for _ in range(n_train):
                raw_train.append(init_pred)

            var raw_val = List[Float64](capacity=n_val)
            if use_early_stopping:
                for _ in range(n_val):
                    raw_val.append(init_pred)

            var grads = List[Float64]()
            var hess = List[Float64]()
            var best_val_loss: Float64 = 1e30
            var no_improvement_count = 0
            self.n_iter_ = 0

            for iter_idx in range(self.max_iter):
                self.n_iter_ = iter_idx + 1

                loss_bce.update_gradients_and_hessians(
                    y_train_f64, raw_train, grads, hess
                )

                var tree = HistTree(
                    max_depth=self.max_depth,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_samples_leaf=self.min_samples_leaf,
                    l2_regularization=self.l2_regularization,
                    min_gain_to_split=self.min_gain_to_split,
                    shrinkage=self.learning_rate,
                )
                tree.build(binned_train, grads, hess)

                for i in range(n_train):
                    raw_train[i] += tree.predict_binned(binned_train, i)

                if use_early_stopping:
                    var val_loss_total: Float64 = 0.0
                    for i in range(n_val):
                        raw_val[i] += tree.predict_binned(binned_val, i)
                        val_loss_total += loss_bce.loss(
                            Float64(y_val[i]), raw_val[i]
                        )

                    var val_loss = val_loss_total / Float64(n_val)
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= self.n_iter_no_change:
                            var round_trees = List[HistTree]()
                            round_trees.append(tree^)
                            self.trees_.append(round_trees^)
                            break

                var round_trees = List[HistTree]()
                round_trees.append(tree^)
                self.trees_.append(round_trees^)

        else:
            # Multiclass K > 2
            var loss_multi = MulticlassCrossEntropyLoss(K)
            self.init_raw_predictions_ = loss_multi.init_raw_predictions(
                y_train
            )

            var raw_train_all = List[Float64](capacity=n_train * K)
            for _ in range(n_train):
                for k in range(K):
                    raw_train_all.append(self.init_raw_predictions_[k])

            var raw_val_all = List[Float64](capacity=n_val * K)
            if use_early_stopping:
                for _ in range(n_val):
                    for k in range(K):
                        raw_val_all.append(self.init_raw_predictions_[k])

            var grads = List[Float64]()
            var hess = List[Float64]()
            var best_val_loss: Float64 = 1e30
            var no_improvement_count = 0
            self.n_iter_ = 0

            for iter_idx in range(self.max_iter):
                self.n_iter_ = iter_idx + 1
                var round_trees = List[HistTree](capacity=K)

                for k in range(K):
                    loss_multi.update_gradients_and_hessians_for_class(
                        y_train, raw_train_all, k, grads, hess
                    )

                    var tree = HistTree(
                        max_depth=self.max_depth,
                        max_leaf_nodes=self.max_leaf_nodes,
                        min_samples_leaf=self.min_samples_leaf,
                        l2_regularization=self.l2_regularization,
                        min_gain_to_split=self.min_gain_to_split,
                        shrinkage=self.learning_rate,
                    )
                    tree.build(binned_train, grads, hess)

                    for i in range(n_train):
                        raw_train_all[i * K + k] += tree.predict_binned(
                            binned_train, i
                        )

                    round_trees.append(tree^)

                if use_early_stopping:
                    var val_loss_total: Float64 = 0.0
                    var sample_val_logits = List[Float64](capacity=K)
                    for _ in range(K):
                        sample_val_logits.append(0.0)

                    for i in range(n_val):
                        for k in range(K):
                            raw_val_all[
                                i * K + k
                            ] += round_trees[k].predict_binned(binned_val, i)
                            sample_val_logits[k] = raw_val_all[i * K + k]
                        val_loss_total += loss_multi.loss(
                            y_val[i], sample_val_logits
                        )

                    var val_loss = val_loss_total / Float64(n_val)
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= self.n_iter_no_change:
                            self.trees_.append(round_trees^)
                            break

                self.trees_.append(round_trees^)

        self.is_fitted = True

    def decision_function[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Calculates raw margin predictions for samples in $X$."""
        check_is_fitted("HistGradientBoostingClassifier", self.is_fitted)
        check_array[feat_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "HistGradientBoostingClassifier.decision_function",
            )

        var N = X.rows
        var K = len(self.classes_)

        if K <= 2:
            var scores = Matrix[feat_dtype](N, 1, 0)
            var init_val = Scalar[feat_dtype](self.init_raw_predictions_[0])
            for i in range(N):
                scores[i, 0] = init_val

            for t in range(len(self.trees_)):
                var tree_preds = self.trees_[t][0].predict[feat_dtype](X)
                for i in range(N):
                    scores[i, 0] += Scalar[feat_dtype](tree_preds[i])

            return scores^
        else:
            var scores = Matrix[feat_dtype](N, K, 0)
            for i in range(N):
                for k in range(K):
                    scores[i, k] = Scalar[feat_dtype](
                        self.init_raw_predictions_[k]
                    )

            for t in range(len(self.trees_)):
                for k in range(K):
                    var tree_preds = self.trees_[t][k].predict[feat_dtype](X)
                    for i in range(N):
                        scores[i, k] += Scalar[feat_dtype](tree_preds[i])

            return scores^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Calculates calibrated class probability distributions for $X$."""
        var scores = self.decision_function(X)
        var N = scores.rows
        var K = len(self.classes_)
        var probs = Matrix[feat_dtype](N, K, 0)

        if K <= 1:
            for i in range(N):
                probs[i, 0] = 1.0
        elif K == 2:
            for i in range(N):
                var s = scores[i, 0]
                var p1 = sigmoid[feat_dtype](s)
                probs[i, 0] = Scalar[feat_dtype](1.0) - p1
                probs[i, 1] = p1
        else:
            for i in range(N):
                var row_scores = scores.row(i)
                var row_probs = softmax[feat_dtype](row_scores)
                for k in range(K):
                    probs[i, k] = row_probs[k]

        return probs^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predicts class labels for samples in feature matrix $X$."""
        var scores = self.decision_function(X)
        var N = scores.rows
        var preds = List[Int](capacity=N)

        if len(self.classes_) <= 1:
            for _ in range(N):
                preds.append(self.classes_[0])
        elif len(self.classes_) == 2:
            for i in range(N):
                if scores[i, 0] > 0:
                    preds.append(self.classes_[1])
                else:
                    preds.append(self.classes_[0])
        else:
            for i in range(N):
                var best_k = 0
                var best_score = scores[i, 0]
                for k in range(1, scores.cols):
                    if scores[i, k] > best_score:
                        best_score = scores[i, k]
                        best_k = k
                preds.append(self.classes_[best_k])

        return preds^

    def score[
        feat_dtype: DType, target_dtype: DType
    ](
        self,
        X: Matrix[feat_dtype],
        y: List[Scalar[target_dtype]],
    ) raises -> Float64:
        """Returns the mean accuracy on the given test data and labels."""
        var preds = self.predict(X)
        var N = len(y)
        if N == 0:
            return 0.0
        var correct = 0
        for i in range(N):
            if preds[i] == Int(y[i]):
                correct += 1
        return Float64(correct) / Float64(N)
