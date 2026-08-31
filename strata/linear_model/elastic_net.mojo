from ..core.matrix import Matrix
from ..core.linalg import dense_dot_vec
from ..base.estimator import Regressor
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)
from ._coordinate_descent import _coordinate_descent_elastic_net


struct ElasticNet[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor, Serializable):
    """Linear regression with combined L1 and L2 regularization (ElasticNet).

    Minimizes the penalized least-squares objective function using coordinate descent:

    $$
    \\min_{w, b} \\frac{1}{2N} \\|y - (Xw + b)\\|_2^2 + \\alpha \\cdot \\rho \\|w\\|_1 + \\frac{\\alpha (1 - \\rho)}{2} \\|w\\|_2^2
    $$

    where $\\rho \\in [0, 1]$ corresponds to `l1_ratio`.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        alpha: Regularization constant multiplier ($\\alpha \\ge 0$). Default 1.0.
        l1_ratio: ElasticNet mixing parameter $\\rho \\in [0, 1]$. For `l1_ratio = 1.0` the penalty is L1 (Lasso); for `l1_ratio = 0.0` the penalty is L2 (Ridge). Default 0.5.
        fit_intercept: Whether to calculate the independent intercept bias term. Default True.
        max_iter: Maximum number of coordinate descent iterations. Default 1000.
        tol: Convergence tolerance threshold for maximum coefficient update. Default 1e-4.
        positive: When set to True, forces coefficients to be non-negative. Default False.

    Attributes:
        coef_: Weight vector coefficients of length $D$.
        intercept_: Independent bias intercept term.
        n_iter_: Actual number of coordinate descent iterations run.
        is_fitted: Boolean flag indicating if the model has been fitted.
        n_features_in_: Number of input features observed during fitting.

    Examples:
        ```mojo
        from strata.linear_model import ElasticNet
        from strata.core import Matrix

        var reg = ElasticNet[DType.float64](alpha=0.1, l1_ratio=0.7)
        reg.fit(X_train, y_train)
        var preds = reg.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var alpha: Scalar[Self.compute_dtype]
    var l1_ratio: Scalar[Self.compute_dtype]
    var fit_intercept: Bool
    var max_iter: Int
    var tol: Scalar[Self.compute_dtype]
    var positive: Bool
    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]
    var n_iter_: Int
    var n_features_in_: Int

    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        l1_ratio: Scalar[Self.compute_dtype] = 0.5,
        fit_intercept: Bool = True,
        max_iter: Int = 1000,
        tol: Scalar[Self.compute_dtype] = 1e-4,
        positive: Bool = False,
    ) raises:
        """Initialize the ElasticNet regression estimator.

        Args:
            alpha: Regularization strength (must be non-negative). Default 1.0.
            l1_ratio: Mixing parameter in [0, 1]. Default 0.5.
            fit_intercept: Whether to fit an intercept term. Default True.
            max_iter: Maximum iterations for coordinate descent. Default 1000.
            tol: Convergence tolerance threshold. Default 1e-4.
            positive: Force non-negative coefficients. Default False.

        Raises:
            InvalidParameterError: If alpha < 0, l1_ratio not in [0, 1], max_iter <= 0, or tol < 0.
        """
        check_floating_dtype[Self.compute_dtype, "ElasticNet"]()
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
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0
        self.n_iter_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing ElasticNet instance."""
        self.is_fitted = copy.is_fitted
        self.alpha = copy.alpha
        self.l1_ratio = copy.l1_ratio
        self.fit_intercept = copy.fit_intercept
        self.max_iter = copy.max_iter
        self.tol = copy.tol
        self.positive = copy.positive
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_
        self.n_iter_ = copy.n_iter_
        self.n_features_in_ = copy.n_features_in_

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fit the ElasticNet linear model via coordinate descent.

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

        if self.fit_intercept:
            var X_means = X_comp.mean_along_axis_0()
            var y_sum: Scalar[Self.compute_dtype] = 0
            for i in range(N):
                y_sum += y_comp[i]
            var y_mean = y_sum / Scalar[Self.compute_dtype](N)

            var X_centered = Matrix[Self.compute_dtype](N, D, 0)
            for r in range(N):
                for c in range(D):
                    X_centered[r, c] = X_comp[r, c] - X_means[c]

            var y_centered = List[Scalar[Self.compute_dtype]](capacity=N)
            for i in range(N):
                y_centered.append(y_comp[i] - y_mean)

            var cd_res = _coordinate_descent_elastic_net[Self.compute_dtype](
                X_centered,
                y_centered,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
                positive=self.positive,
            )

            var w = cd_res[0].copy()
            self.n_iter_ = cd_res[1]

            var w_dot_mean: Scalar[Self.compute_dtype] = 0
            for j in range(D):
                w_dot_mean += w[j] * X_means[j]

            self.intercept_ = y_mean - w_dot_mean
            self.coef_ = w^
        else:
            var cd_res = _coordinate_descent_elastic_net[Self.compute_dtype](
                X_comp,
                y_comp,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
                positive=self.positive,
            )
            self.coef_ = cd_res[0].copy()
            self.n_iter_ = cd_res[1]
            self.intercept_ = 0

        self.is_fitted = True

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predict continuous target values using the fitted ElasticNet model.

        Args:
            X: Feature matrix of shape $(N, D)$ to predict on.

        Returns:
            List[Scalar[feat_dtype]]: Predicted target vector of length $N$.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            DimensionMismatchError: If the column count of $X$ does not match `n_features_in_`.
        """
        check_is_fitted("ElasticNet", self.is_fitted)
        check_array[feat_dtype](X)

        if X.cols != len(self.coef_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.coef_)),
                "X.cols == " + String(X.cols),
                "ElasticNet.predict",
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

    def serialize(self, mut writer: BufferWriter):
        """Serializes ElasticNet parameters and fitted state into BufferWriter.
        """
        write_header(writer, "ElasticNet")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.alpha.cast[DType.float64]())
        writer.write_float64(self.l1_ratio.cast[DType.float64]())
        writer.write_bool(self.fit_intercept)
        writer.write_int(self.max_iter)
        writer.write_float64(self.tol.cast[DType.float64]())
        writer.write_bool(self.positive)
        writer.write_float_list[Self.compute_dtype](self.coef_)
        writer.write_float64(self.intercept_.cast[DType.float64]())
        writer.write_int(self.n_iter_)
        writer.write_int(self.n_features_in_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes ElasticNet from BufferReader."""
        check_header(reader, "ElasticNet")
        var is_fitted = reader.read_bool()
        var alpha = Scalar[Self.compute_dtype](reader.read_float64())
        var l1_ratio = Scalar[Self.compute_dtype](reader.read_float64())
        var fit_intercept = reader.read_bool()
        var max_iter = reader.read_int()
        var tol = Scalar[Self.compute_dtype](reader.read_float64())
        var positive = reader.read_bool()
        var coef_ = reader.read_float_list[Self.compute_dtype]()
        var intercept_ = Scalar[Self.compute_dtype](reader.read_float64())
        var n_iter_ = reader.read_int()
        var n_features_in_ = reader.read_int()

        var model = Self(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            positive=positive,
        )
        model.is_fitted = is_fitted
        model.coef_ = coef_^
        model.intercept_ = intercept_
        model.n_iter_ = n_iter_
        model.n_features_in_ = n_features_in_
        return model^
