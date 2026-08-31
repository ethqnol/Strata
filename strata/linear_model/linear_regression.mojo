from ..core.matrix import Matrix
from ..core.linalg import (
    gemm,
    dense_dot_vec,
    qr,
    lstsq,
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


struct LinearRegression[
    compute_dtype: DType = DType.float64,
](Copyable, Movable, Regressor, Serializable):
    """Ordinary Least Squares Linear Regression.

    Fits a linear model with coefficients $w = (w_1, \\dots, w_D)$ and intercept $b$
    to minimize the residual sum of squares between observed targets and predictions:

    $$
    \\min_{w, b} \\frac{1}{2N} \\|y - (Xw + b)\\|_2^2
    $$


    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        fit_intercept: Whether to calculate the intercept bias term. Default True.
        solver: Solver algorithm to use ('lstsq', 'qr', 'cholesky', 'solve'). Default 'lstsq'.

    Attributes:
        coef_: Weight vector coefficients of length $D$.
        intercept_: Independent bias intercept term.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.linear_model import LinearRegression
        from strata.core import Matrix

        var reg = LinearRegression[DType.float64](solver="cholesky")
        reg.fit(X_train, y_train)
        var preds = reg.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var fit_intercept: Bool
    var solver: String
    var coef_: List[Scalar[Self.compute_dtype]]
    var intercept_: Scalar[Self.compute_dtype]

    def __init__(
        out self,
        fit_intercept: Bool = True,
        solver: String = "lstsq",
    ):
        """Initialize the linear regression estimator.

        Args:
            fit_intercept: Whether to calculate the intercept bias term. Default True.
            solver: Solver algorithm ('lstsq', 'qr', 'cholesky', 'solve'). Default 'lstsq'.
        """
        check_floating_dtype[Self.compute_dtype, "LinearRegression"]()
        self.is_fitted = False
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.coef_ = List[Scalar[Self.compute_dtype]]()
        self.intercept_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing LinearRegression instance."""
        self.is_fitted = copy.is_fitted
        self.fit_intercept = copy.fit_intercept
        self.solver = copy.solver
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fit the linear model from training data.

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
        """Internal solver dispatch for least-squares optimization."""
        if self.solver == "lstsq" or self.solver == "svd":
            return lstsq(X, y)
        elif self.solver == "qr":
            var qr_res = qr(X)
            var Qt = qr_res.Q.transpose()
            var Qt_y = dense_dot_vec(Qt, y)
            return solve(qr_res.R, Qt_y)
        elif self.solver == "cholesky":
            var Xt = X.transpose()
            var XtX = gemm(Xt, X)
            var Xty = dense_dot_vec(Xt, y)
            return solve_cholesky(XtX, Xty)
        elif self.solver == "solve" or self.solver == "lu":
            var Xt = X.transpose()
            var XtX = gemm(Xt, X)
            var Xty = dense_dot_vec(Xt, y)
            return solve(XtX, Xty)
        else:
            raise InvalidParameterError.error(
                "solver",
                "Unsupported solver '"
                + self.solver
                + "'. Expected 'lstsq', 'qr', 'cholesky', or 'solve'.",
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
        check_is_fitted("LinearRegression", self.is_fitted)

        check_array[feat_dtype](X)
        if X.cols != len(self.coef_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.coef_)),
                "X.cols == " + String(X.cols),
                "LinearRegression.predict",
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
        """Serializes LinearRegression parameters and fitted state into BufferWriter.
        """
        write_header(writer, "LinearRegression")
        writer.write_bool(self.is_fitted)
        writer.write_bool(self.fit_intercept)
        writer.write_string(self.solver)
        writer.write_float_list[Self.compute_dtype](self.coef_)
        writer.write_float64(self.intercept_.cast[DType.float64]())

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes LinearRegression from BufferReader."""
        check_header(reader, "LinearRegression")
        var is_fitted = reader.read_bool()
        var fit_intercept = reader.read_bool()
        var solver = reader.read_string()
        var coef_ = reader.read_float_list[Self.compute_dtype]()
        var intercept_ = Scalar[Self.compute_dtype](reader.read_float64())

        var model = Self(fit_intercept=fit_intercept, solver=solver)
        model.is_fitted = is_fitted
        model.coef_ = coef_^
        model.intercept_ = intercept_
        return model^
