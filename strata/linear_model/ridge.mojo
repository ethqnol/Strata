from ..core.matrix import Matrix
from ..core.linalg import (
    gemm,
    dense_dot_vec,
    svd,
    solve,
    solve_cholesky,
)
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


struct Ridge[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor, Serializable):
    """Ridge regression with L2 regularization.

    Minimizes the penalized objective function:

    $$
    \\min_{w} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_2^2
    $$

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        alpha: Regularization strength ($\\alpha \\ge 0$). Larger values enforce stronger shrinkage. Default 1.0.
        fit_intercept: Whether to calculate the intercept bias term. Default True.
        solver: Solver algorithm to use ('auto', 'cholesky', 'svd', 'solve'). Default 'auto'.


    Attributes:
        coef_: Weight vector coefficients of length $D$.
        intercept_: Independent bias intercept term.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.linear_model import Ridge
        from strata.core import Matrix

        var model = Ridge[DType.float64](alpha=0.5)
        model.fit(X_train, y_train)
        var preds = model.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var alpha: Scalar[Self.compute_dtype]
    var fit_intercept: Bool
    var solver: String
    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]

    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        fit_intercept: Bool = True,
        solver: String = "auto",
    ) raises:
        """Initialize the Ridge regression estimator.

        Args:
            alpha: Regularization strength (must be non-negative). Default 1.0.
            fit_intercept: Whether to calculate the intercept bias term. Default True.
            solver: Solver algorithm ('auto', 'cholesky', 'svd', 'solve'). Default 'auto'.

        Raises:
            InvalidParameterError: If alpha is negative.
        """

        check_floating_dtype[Self.compute_dtype, "Ridge"]()
        if alpha < 0:
            raise InvalidParameterError.error(
                "alpha",
                "alpha must be non-negative, got " + String(alpha),
            )
        self.is_fitted = False
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing Ridge instance."""
        self.is_fitted = copy.is_fitted
        self.alpha = copy.alpha
        self.fit_intercept = copy.fit_intercept
        self.solver = copy.solver
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fit the Ridge regression model from training data.

        Args:
            X: Training feature matrix of shape $(N, D)$.
            y: Target values vector of length $N$.

        Raises:
            DimensionMismatchError: If $X$ rows do not match length of $y$.
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols

        # Promote upfront to compute_dtype once
        var X_comp = X.cast[Self.compute_dtype]()

        var y_comp = List[Scalar[Self.compute_dtype]](capacity=N)
        for i in range(N):
            y_comp.append(Scalar[Self.compute_dtype](y[i]))

        if self.fit_intercept:
            # compute means along axis 0
            var X_means = X_comp.mean_along_axis_0()
            var y_sum: Scalar[Self.compute_dtype] = 0
            for i in range(N):
                y_sum += y_comp[i]
            var y_mean = y_sum / Scalar[Self.compute_dtype](N)

            # center X and y
            var X_centered = Matrix[Self.compute_dtype](N, D, 0)
            for r in range(N):
                for c in range(D):
                    X_centered[r, c] = X_comp[r, c] - X_means[c]

            var y_centered = List[Scalar[Self.compute_dtype]](capacity=N)
            for i in range(N):
                y_centered.append(y_comp[i] - y_mean)

            # solve for beta on centered data
            var beta = self._solve(X_centered, y_centered)

            # calculate intercept: y_mean - sum(beta_j * X_means_j)
            var beta_dot_mean: Scalar[Self.compute_dtype] = 0
            for j in range(D):
                beta_dot_mean += beta[j] * X_means[j]

            self.intercept_ = y_mean - beta_dot_mean
            self.coef_ = beta^
        else:
            var beta = self._solve(X_comp, y_comp)
            self.intercept_ = 0
            self.coef_ = beta^

        self.is_fitted = True

    def _solve(
        self,
        X: Matrix[Self.compute_dtype],
        y: List[Scalar[Self.compute_dtype]],
    ) raises -> List[Scalar[Self.compute_dtype]]:
        """Internal solver dispatch for regularized least-squares optimization.
        """
        var D = X.cols
        var N = X.rows

        if self.solver == "auto" or self.solver == "cholesky":
            var Xt = X.transpose()
            var XtX = gemm(Xt, X)
            for i in range(D):
                XtX[i, i] += self.alpha
            var Xty = dense_dot_vec(Xt, y)
            return solve_cholesky(XtX, Xty)
        elif self.solver == "svd" or self.solver == "lsqr":
            var svd_res = svd(X)
            var K = len(svd_res.S)
            var z = List[Scalar[Self.compute_dtype]](capacity=K)
            for i in range(K):
                var s = svd_res.S[i]
                var ut_y: Scalar[Self.compute_dtype] = 0
                for r in range(N):
                    ut_y += svd_res.U[r, i] * y[r]
                var denom = s * s + self.alpha
                if denom > 0:
                    z.append(ut_y * (s / denom))
                else:
                    z.append(0)

            var beta = List[Scalar[Self.compute_dtype]](capacity=D)
            for j in range(D):
                var val: Scalar[Self.compute_dtype] = 0
                for i in range(K):
                    val += svd_res.Vt[i, j] * z[i]
                beta.append(val)
            return beta^
        elif self.solver == "solve" or self.solver == "lu":
            var Xt = X.transpose()
            var XtX = gemm(Xt, X)
            for i in range(D):
                XtX[i, i] += self.alpha
            var Xty = dense_dot_vec(Xt, y)
            return solve(XtX, Xty)
        else:
            raise InvalidParameterError.error(
                "solver",
                "Unsupported solver '"
                + self.solver
                + "'. Expected 'auto', 'cholesky', 'svd', or 'solve'.",
            )

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predict continuous target values using the fitted linear model.

        Args:
            X: Feature matrix of shape $(N, D)$ to predict on.

        Returns:
            List[Scalar[feat_dtype]]: Predicted target vector of length $N$.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            DimensionMismatchError: If the number of columns in $X$ does not match `n_features_in_`.
        """
        check_is_fitted("Ridge", self.is_fitted)

        check_array[feat_dtype](X)
        if X.cols != len(self.coef_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.coef_)),
                "X.cols == " + String(X.cols),
                "Ridge.predict",
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
                preds.append(Scalar[feat_dtype](preds_comp[i]))
            return preds^

    def serialize(self, mut writer: BufferWriter):
        """Serializes Ridge parameters and fitted state into BufferWriter."""
        write_header(writer, "Ridge")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.alpha.cast[DType.float64]())
        writer.write_bool(self.fit_intercept)
        writer.write_string(self.solver)
        writer.write_float_list[Self.compute_dtype](self.coef_)
        writer.write_float64(self.intercept_.cast[DType.float64]())

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes Ridge from BufferReader."""
        check_header(reader, "Ridge")
        var is_fitted = reader.read_bool()
        var alpha = Scalar[Self.compute_dtype](reader.read_float64())
        var fit_intercept = reader.read_bool()
        var solver = reader.read_string()
        var coef_ = reader.read_float_list[Self.compute_dtype]()
        var intercept_ = Scalar[Self.compute_dtype](reader.read_float64())

        var model = Self(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
        )
        model.is_fitted = is_fitted
        model.coef_ = coef_^
        model.intercept_ = intercept_
        return model^
