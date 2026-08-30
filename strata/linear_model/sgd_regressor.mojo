from std.math import abs, sqrt
from ..core.matrix import Matrix
from ..core.linalg import dense_dot_vec
from ..base.estimator import Regressor
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.random import PRNG
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ._sgd_fast import _compute_eta, _dloss_regression, _apply_penalty_step


struct SGDRegressor[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor):
    """Linear model fitted by minimizing a regularized empirical loss with SGD.

    Minimizes the objective function using stochastic gradient descent:

    $$
    \\min_{w, b} \\frac{1}{N} \\sum_{i=1}^N \\mathcal{L}(w^T x_i + b, y_i) + \\alpha \\mathcal{R}(w)
    $$

    where $\\mathcal{L}$ is the regression loss function and $\\mathcal{R}$ is the penalty norm ($L_2, L_1, \\text{ElasticNet}$).

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        loss: Loss function to be used ('squared_error', 'huber', 'epsilon_insensitive'). Default 'squared_error'.
        penalty: Regularization penalty ('l2', 'l1', 'elasticnet', 'none'). Default 'l2'.
        alpha: Regularization multiplier ($\\alpha \\ge 0$). Default 1e-4.
        l1_ratio: ElasticNet mixing parameter in [0, 1]. Default 0.15.
        fit_intercept: Whether the intercept should be estimated. Default True.
        max_iter: Maximum number of passes over the training data (epochs). Default 1000.
        tol: Stopping criterion threshold for epoch loss improvement. Default 1e-3.
        shuffle_data: Whether to shuffle data after each epoch. Default True.
        epsilon: Epsilon parameter in the epsilon-insensitive or huber loss. Default 0.1.
        random_state: Seed for random data shuffling. Default 42.
        learning_rate: Learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive'). Default 'invscaling'.
        eta0: Initial learning rate. Default 0.01.
        power_t: Exponent for inverse scaling learning rate. Default 0.25.

    Attributes:
        coef_: Weight vector coefficients of length $D$.
        intercept_: Independent bias intercept term.
        n_iter_: Actual number of epochs executed before convergence.
        is_fitted: Boolean flag indicating if estimator has been fitted.
        n_features_in_: Number of features seen during fitting.

    Examples:
        ```mojo
        from strata.linear_model import SGDRegressor
        from strata.core import Matrix

        var reg = SGDRegressor[DType.float64](loss="squared_error", penalty="l2", alpha=1e-4)
        reg.fit(X_train, y_train)
        var preds = reg.predict(X_test)
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
    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self,
        loss: String = "squared_error",
        penalty: String = "l2",
        alpha: Scalar[Self.compute_dtype] = 1e-4,
        l1_ratio: Scalar[Self.compute_dtype] = 0.15,
        fit_intercept: Bool = True,
        max_iter: Int = 1000,
        tol: Scalar[Self.compute_dtype] = 1e-3,
        shuffle_data: Bool = True,
        epsilon: Scalar[Self.compute_dtype] = 0.1,
        random_state: Int = 42,
        learning_rate: String = "invscaling",
        eta0: Scalar[Self.compute_dtype] = 0.01,
        power_t: Scalar[Self.compute_dtype] = 0.25,
    ) raises:
        """Initialize the SGDRegressor estimator."""
        check_floating_dtype[Self.compute_dtype, "SGDRegressor"]()

        if (
            loss != "squared_error"
            and loss != "huber"
            and loss != "epsilon_insensitive"
        ):
            raise InvalidParameterError.error(
                "loss",
                "Unsupported loss '"
                + loss
                + "'. Expected 'squared_error', 'huber', or"
                " 'epsilon_insensitive'.",
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
        if eta0 <= 0:
            raise InvalidParameterError.error(
                "eta0", "eta0 must be strictly positive, got " + String(eta0)
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
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing SGDRegressor instance."""
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
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_
        self.n_iter_ = copy.n_iter_
        self.n_features_in_ = copy.n_features_in_

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fit linear model with Stochastic Gradient Descent.

        Args:
            X: Training feature matrix of shape $(N, D)$.
            y: Target values vector of length $N$.

        Raises:
            DimensionMismatchError: If row count of $X$ does not match length of $y$.
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols
        self.n_features_in_ = D

        var X_comp = X.cast[Self.compute_dtype]()
        var y_comp = List[Scalar[Self.compute_dtype]](capacity=N)
        for i in range(N):
            y_comp.append(Scalar[Self.compute_dtype](y[i]))

        var w = List[Scalar[Self.compute_dtype]](capacity=D)
        for _ in range(D):
            w.append(0)
        var b: Scalar[Self.compute_dtype] = 0

        var rng = PRNG(self.random_state)
        var indices = List[Int](capacity=N)
        for i in range(N):
            indices.append(i)

        var global_step = 0
        var t0 = Scalar[Self.compute_dtype](1.0)
        if self.learning_rate == "optimal":
            t0 = Scalar[Self.compute_dtype](1.0) / (self.alpha * self.eta0)

        comptime simd_width = 4 if Self.compute_dtype == DType.float64 else 8
        var x_ptr = X_comp.data.unsafe_ptr()
        var w_ptr = w.unsafe_ptr()

        var prev_epoch_loss: Scalar[Self.compute_dtype] = 1e30

        for epoch in range(self.max_iter):
            self.n_iter_ = epoch + 1

            if self.shuffle_data:
                for i in range(N - 1, 0, -1):
                    var j = rng.next_int(i + 1)
                    var tmp = indices[i]
                    indices[i] = indices[j]
                    indices[j] = tmp

            var epoch_loss: Scalar[Self.compute_dtype] = 0

            for sample_idx in range(N):
                var i = indices[sample_idx]
                var i_offset = i * D

                var dot_simd = SIMD[Self.compute_dtype, simd_width](0)
                var k = 0
                while k + simd_width <= D:
                    var x_chunk = x_ptr.unsafe_offset(i_offset + k).unsafe_load[
                        width=simd_width
                    ]()
                    var w_chunk = w_ptr.unsafe_offset(k).unsafe_load[
                        width=simd_width
                    ]()
                    dot_simd = x_chunk.fma(w_chunk, dot_simd)
                    k += simd_width

                var dot_val: Scalar[Self.compute_dtype] = dot_simd.reduce_add()
                while k < D:
                    dot_val += X_comp.data[i_offset + k] * w[k]
                    k += 1

                var y_pred = dot_val + b
                var d_loss = _dloss_regression[Self.compute_dtype](
                    self.loss, y_comp[i], y_pred, self.epsilon
                )

                epoch_loss += abs(y_pred - y_comp[i])

                var eta = _compute_eta[Self.compute_dtype](
                    self.learning_rate,
                    self.eta0,
                    self.power_t,
                    self.alpha,
                    global_step,
                    t0,
                )
                global_step += 1

                # Gradient step
                var grad_step = -eta * d_loss
                var grad_simd = SIMD[Self.compute_dtype, simd_width](grad_step)

                k = 0
                while k + simd_width <= D:
                    var x_chunk = x_ptr.unsafe_offset(i_offset + k).unsafe_load[
                        width=simd_width
                    ]()
                    var w_chunk = w_ptr.unsafe_offset(k).unsafe_load[
                        width=simd_width
                    ]()
                    var w_updated = grad_simd.fma(x_chunk, w_chunk)
                    w_ptr.unsafe_offset(k).unsafe_store[width=simd_width](
                        w_updated
                    )
                    k += simd_width

                while k < D:
                    w[k] += grad_step * X_comp.data[i_offset + k]
                    k += 1

                _apply_penalty_step[Self.compute_dtype](
                    w, self.penalty, self.alpha, self.l1_ratio, eta
                )

                if self.fit_intercept:
                    b -= eta * d_loss

            epoch_loss /= Scalar[Self.compute_dtype](N)
            if abs(prev_epoch_loss - epoch_loss) < self.tol:
                break
            prev_epoch_loss = epoch_loss

        self.coef_ = w^
        self.intercept_ = b
        self.is_fitted = True

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predict continuous values using the linear SGD model."""
        check_is_fitted("SGDRegressor", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != len(self.coef_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.coef_)),
                "X.cols == " + String(X.cols),
                "SGDRegressor.predict",
            )

        comptime if feat_dtype == Self.compute_dtype:
            var coef_copy = List[Scalar[feat_dtype]](capacity=len(self.coef_))
            for i in range(len(self.coef_)):
                coef_copy.append(Scalar[feat_dtype](self.coef_[i]))
            return dense_dot_vec[feat_dtype](
                X, coef_copy, bias=Scalar[feat_dtype](self.intercept_)
            )
        else:
            var X_comp = X.cast[Self.compute_dtype]()
            var preds_comp = dense_dot_vec[Self.compute_dtype](
                X_comp, self.coef_, bias=self.intercept_
            )
            var preds = List[Scalar[feat_dtype]](capacity=len(preds_comp))
            for i in range(len(preds_comp)):
                comptime if feat_dtype.is_integral():
                    preds.append(Scalar[feat_dtype](round(preds_comp[i])))
                else:
                    preds.append(Scalar[feat_dtype](preds_comp[i]))
            return preds^
