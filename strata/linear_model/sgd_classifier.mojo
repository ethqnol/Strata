from std.math import abs, exp, max, min, sqrt
from ..core.matrix import Matrix
from ..core.linalg import dense_dot_vec
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.random import PRNG
from ..utils.math import softmax, sigmoid
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ._sgd_fast import _compute_eta, _dloss_classification, _apply_penalty_step


struct SGDClassifier[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable):
    """Linear classifier with SGD training.

    Supports linear SVM (`loss='hinge'`), Logistic Regression (`loss='log_loss'`),
    Modified Huber (`loss='modified_huber'`), and Squared Hinge (`loss='squared_hinge'`).

    Minimizes the regularized loss:

    $$
    \\min_{W, b} \\frac{1}{N} \\sum_{i=1}^N \\mathcal{L}(W x_i + b, y_i) + \\alpha \\mathcal{R}(W)
    $$

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        loss: Loss function to use ('hinge', 'log_loss', 'modified_huber', 'squared_hinge'). Default 'hinge'.
        penalty: Regularization penalty ('l2', 'l1', 'elasticnet', 'none'). Default 'l2'.
        alpha: Regularization constant multiplier ($\\alpha \\ge 0$). Default 1e-4.
        l1_ratio: ElasticNet mixing parameter in [0, 1]. Default 0.15.
        fit_intercept: Whether to estimate an independent intercept term. Default True.
        max_iter: Maximum number of passes over training data (epochs). Default 1000.
        tol: Stopping criterion threshold for loss changes. Default 1e-3.
        shuffle_data: Whether to shuffle data per epoch. Default True.
        epsilon: Epsilon parameter for huber loss. Default 0.1.
        random_state: Seed for random shuffling. Default 42.
        learning_rate: Learning rate schedule ('optimal', 'constant', 'invscaling', 'adaptive'). Default 'optimal'.
        eta0: Initial learning rate. Default 0.0.
        power_t: Exponent for inverse scaling schedule. Default 0.5.

    Attributes:
        classes_: Unique sorted class labels observed in training data.
        coef_: Learned weights matrix of shape $(K, D)$ (or $(1, D)$ for binary).
        intercept_: Learned bias vector of length $K$ (or 1 for binary).
        n_iter_: Actual number of epochs executed before convergence.
        is_fitted: Boolean flag indicating if estimator has been fitted.
        n_features_in_: Number of features seen during fitting.

    Examples:
        ```mojo
        from strata.linear_model import SGDClassifier
        from strata.core import Matrix

        var clf = SGDClassifier[DType.float64](loss="log_loss", penalty="l2")
        clf.fit(X_train, y_train)
        var preds = clf.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var loss: String
    var penalty: String
    var alpha: Scalar[Self.compute_dtype]
    var l1_ratio: Scalar[Self.compute_dtype]
    var fit_intercept: Bool
    var max_iter: Int
    var tol: Scalar[Self.compute_dtype]
    var shuffle_data: Bool
    var epsilon: Scalar[Self.compute_dtype]
    var random_state: Int
    var learning_rate: String
    var eta0: Scalar[Self.compute_dtype]
    var power_t: Scalar[Self.compute_dtype]
    var classes_: List[Int]
    var coef_: Matrix[Self.compute_dtype]
    var intercept_: List[Scalar[Self.compute_dtype]]
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self,
        loss: String = "hinge",
        penalty: String = "l2",
        alpha: Scalar[Self.compute_dtype] = 1e-4,
        l1_ratio: Scalar[Self.compute_dtype] = 0.15,
        fit_intercept: Bool = True,
        max_iter: Int = 1000,
        tol: Scalar[Self.compute_dtype] = 1e-3,
        shuffle_data: Bool = True,
        epsilon: Scalar[Self.compute_dtype] = 0.1,
        random_state: Int = 42,
        learning_rate: String = "optimal",
        eta0: Scalar[Self.compute_dtype] = 0.01,
        power_t: Scalar[Self.compute_dtype] = 0.5,
    ) raises:
        """Initialize the SGDClassifier estimator."""
        check_floating_dtype[Self.compute_dtype, "SGDClassifier"]()

        if (
            loss != "hinge"
            and loss != "log_loss"
            and loss != "log"
            and loss != "modified_huber"
            and loss != "squared_hinge"
        ):
            raise InvalidParameterError.error(
                "loss",
                "Unsupported loss '"
                + loss
                + "'. Expected 'hinge', 'log_loss', 'modified_huber', or"
                " 'squared_hinge'.",
            )

        if (
            penalty != "l2"
            and penalty != "l1"
            and penalty != "elasticnet"
            and penalty != "none"
        ):
            raise InvalidParameterError.error(
                "penalty",
                "Unsupported penalty '"
                + penalty
                + "'. Expected 'l2', 'l1', 'elasticnet', or 'none'.",
            )

        if alpha < 0:
            raise InvalidParameterError.error(
                "alpha", "alpha must be non-negative, got " + String(alpha)
            )
        if l1_ratio < 0 or l1_ratio > 1:
            raise InvalidParameterError.error(
                "l1_ratio",
                "l1_ratio must be between 0 and 1, got " + String(l1_ratio),
            )
        if max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter",
                "max_iter must be strictly positive, got " + String(max_iter),
            )
        if tol < 0:
            raise InvalidParameterError.error(
                "tol", "tol must be non-negative, got " + String(tol)
            )

        self.is_fitted = False
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle_data = shuffle_data
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.classes_ = List[Int]()
        self.coef_ = Matrix[Self.compute_dtype].zeros(0, 0)
        self.intercept_ = List[Scalar[Self.compute_dtype]]()
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing SGDClassifier instance."""
        self.is_fitted = copy.is_fitted
        self.loss = copy.loss
        self.penalty = copy.penalty
        self.alpha = copy.alpha
        self.l1_ratio = copy.l1_ratio
        self.fit_intercept = copy.fit_intercept
        self.max_iter = copy.max_iter
        self.tol = copy.tol
        self.shuffle_data = copy.shuffle_data
        self.epsilon = copy.epsilon
        self.random_state = copy.random_state
        self.learning_rate = copy.learning_rate
        self.eta0 = copy.eta0
        self.power_t = copy.power_t
        self.classes_ = copy.classes_.copy()
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_.copy()
        self.n_iter_ = copy.n_iter_
        self.n_features_in_ = copy.n_features_in_

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fit linear model classifier with Stochastic Gradient Descent."""
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols
        self.n_features_in_ = D

        # Discover unique classes
        var classes = List[Int]()
        for i in range(N):
            var c = Int(y[i])
            var found = False
            for j in range(len(classes)):
                if classes[j] == c:
                    found = True
                    break
            if not found:
                classes.append(c)

        # Sort classes ascending
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                if classes[j] < classes[i]:
                    var tmp = classes[i]
                    classes[i] = classes[j]
                    classes[j] = tmp

        var K = len(classes)
        self.classes_ = classes^

        var X_comp = X.cast[Self.compute_dtype]()
        comptime simd_width = 4 if Self.compute_dtype == DType.float64 else 8

        # One-vs-Rest or Binary classification
        var n_classifiers = 1 if K <= 2 else K
        var coef_mat = Matrix[Self.compute_dtype].zeros(n_classifiers, D)
        var intercept_list = List[Scalar[Self.compute_dtype]](
            capacity=n_classifiers
        )
        for _ in range(n_classifiers):
            intercept_list.append(0)

        var rng = PRNG(self.random_state)
        var indices = List[Int](capacity=N)
        for i in range(N):
            indices.append(i)

        var max_n_iter = 0

        for clf_idx in range(n_classifiers):
            var target_class = (
                self.classes_[1] if K == 2 else self.classes_[clf_idx]
            )

            # Construct binary +1/-1 target vector
            var y_bin = List[Scalar[Self.compute_dtype]](capacity=N)
            for i in range(N):
                if Int(y[i]) == target_class:
                    y_bin.append(1.0)
                else:
                    y_bin.append(-1.0)

            var w = List[Scalar[Self.compute_dtype]](capacity=D)
            for _ in range(D):
                w.append(0)
            var b: Scalar[Self.compute_dtype] = 0

            var global_step = 0
            var t0 = Scalar[Self.compute_dtype](1.0)
            if self.learning_rate == "optimal":
                if self.alpha > 0 and self.eta0 > 0:
                    t0 = Scalar[Self.compute_dtype](1.0) / (
                        self.alpha * self.eta0
                    )

            var x_ptr = X_comp.data.unsafe_ptr()
            var w_ptr = w.unsafe_ptr()
            var prev_loss: Scalar[Self.compute_dtype] = 1e30

            for epoch in range(self.max_iter):
                if epoch + 1 > max_n_iter:
                    max_n_iter = epoch + 1

                if self.shuffle_data:
                    for i in range(N - 1, 0, -1):
                        var j = rng.next_int(i + 1)
                        var tmp = indices[i]
                        indices[i] = indices[j]
                        indices[j] = tmp

                var epoch_loss: Scalar[Self.compute_dtype] = 0

                for s_idx in range(N):
                    var i = indices[s_idx]
                    var i_offset = i * D

                    var dot_simd = SIMD[Self.compute_dtype, simd_width](0)
                    var k = 0
                    while k + simd_width <= D:
                        var x_chunk = x_ptr.unsafe_offset(
                            i_offset + k
                        ).unsafe_load[width=simd_width]()
                        var w_chunk = w_ptr.unsafe_offset(k).unsafe_load[
                            width=simd_width
                        ]()
                        dot_simd = x_chunk.fma(w_chunk, dot_simd)
                        k += simd_width

                    var score: Scalar[
                        Self.compute_dtype
                    ] = dot_simd.reduce_add()
                    while k < D:
                        score += X_comp.data[i_offset + k] * w[k]
                        k += 1

                    score += b
                    var d_loss = _dloss_classification[Self.compute_dtype](
                        self.loss, y_bin[i], score, self.epsilon
                    )

                    var eta = _compute_eta[Self.compute_dtype](
                        self.learning_rate,
                        self.eta0,
                        self.power_t,
                        self.alpha,
                        global_step,
                        t0,
                    )
                    global_step += 1

                    var grad_step = -eta * d_loss
                    if grad_step != 0:
                        var grad_simd = SIMD[Self.compute_dtype, simd_width](
                            grad_step
                        )
                        k = 0
                        while k + simd_width <= D:
                            var x_chunk = x_ptr.unsafe_offset(
                                i_offset + k
                            ).unsafe_load[width=simd_width]()
                            var w_chunk = w_ptr.unsafe_offset(k).unsafe_load[
                                width=simd_width
                            ]()
                            var w_updated = grad_simd.fma(x_chunk, w_chunk)
                            w_ptr.unsafe_offset(k).unsafe_store[
                                width=simd_width
                            ](w_updated)
                            k += simd_width

                        while k < D:
                            w[k] += grad_step * X_comp.data[i_offset + k]
                            k += 1

                    _apply_penalty_step[Self.compute_dtype](
                        w, self.penalty, self.alpha, self.l1_ratio, eta
                    )

                    if self.fit_intercept:
                        b -= eta * d_loss

                    epoch_loss += abs(d_loss)

                epoch_loss /= Scalar[Self.compute_dtype](N)
                if abs(prev_loss - epoch_loss) < self.tol:
                    break
                prev_loss = epoch_loss

            # Save learned weights
            for j in range(D):
                coef_mat[clf_idx, j] = w[j]
            intercept_list[clf_idx] = b

        self.coef_ = coef_mat^
        self.intercept_ = intercept_list^
        self.n_iter_ = max_n_iter
        self.is_fitted = True

    def decision_function[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict linear margin decision function."""
        check_is_fitted("SGDClassifier", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "SGDClassifier.decision_function",
            )

        var N = X.rows
        var K = self.coef_.rows
        var res = Matrix[feat_dtype](N, K, 0)

        for c in range(K):
            var w = self.coef_.row(c)
            var b = self.intercept_[c]
            var preds = dense_dot_vec[Self.compute_dtype](
                X.cast[Self.compute_dtype](), w, bias=b
            )
            for i in range(N):
                res[i, c] = Scalar[feat_dtype](preds[i])

        return res^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Probability estimates for each class."""
        var scores = self.decision_function(X)
        var N = scores.rows
        var K = len(self.classes_)
        var probs = Matrix[feat_dtype](N, K, 0)

        if K == 2:
            for i in range(N):
                var s = Float64(scores[i, 0])
                var p1 = 1.0 / (1.0 + exp(-s))
                probs[i, 0] = Scalar[feat_dtype](1.0 - p1)
                probs[i, 1] = Scalar[feat_dtype](p1)
        else:
            for i in range(N):
                var row_scores = scores.row(i)
                var row_probs = softmax[feat_dtype](row_scores)
                for c in range(K):
                    probs[i, c] = row_probs[c]

        return probs^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in X."""
        var scores = self.decision_function(X)
        var N = scores.rows
        var preds = List[Int](capacity=N)

        if len(self.classes_) == 2:
            for i in range(N):
                if scores[i, 0] > 0:
                    preds.append(self.classes_[1])
                else:
                    preds.append(self.classes_[0])
        else:
            for i in range(N):
                var best_c = 0
                var best_score = scores[i, 0]
                for c in range(1, scores.cols):
                    if scores[i, c] > best_score:
                        best_score = scores[i, c]
                        best_c = c
                preds.append(self.classes_[best_c])

        return preds^
