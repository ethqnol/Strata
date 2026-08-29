from std.math import nan, isnan
from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..utils.validation import (
    check_is_fitted,
    check_floating_dtype,
)
from ..exceptions.errors import (
    NotFittedError,
    DataConversionError,
    DimensionMismatchError,
    InvalidParameterError,
)


def _is_missing[
    dtype: DType
](val: Scalar[dtype], missing_values: Float64) -> Bool:
    comptime if dtype.is_floating_point():
        if isnan(missing_values):
            return isnan(val)
    return Float64(val) == missing_values


struct SimpleImputer[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Univariate imputer for completing missing values with simple statistics.

    Replaces missing values (`NaN` or a specified sentinel value) using a
    chosen statistical strategy along each column.

    Strategies:
    - `"mean"`: Replace missing values using the mean along each column.
    - `"median"`: Replace missing values using the median along each column.
    - `"most_frequent"`: Replace missing values using the most frequent value (mode) along each column.
    - `"constant"`: Replace missing values with `fill_value`.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        missing_values: The placeholder for missing values. Can be `NaN` or a scalar value. Default `NaN`.
        strategy: The imputation strategy ('mean', 'median', 'most_frequent', 'constant'). Default 'mean'.
        fill_value: Value used when strategy='constant'. Default 0.0.

    Attributes:
        statistics_: The imputation fill value for each feature column calculated during fit.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import SimpleImputer
        from strata.core import Matrix

        var imputer = SimpleImputer[DType.float64](strategy="mean")
        imputer.fit(X_train)
        var X_imputed = imputer.transform(X_train)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var missing_values: Float64
    var strategy: String
    var fill_value: Scalar[Self.compute_dtype]
    var statistics_: List[Scalar[Self.compute_dtype]]
    var n_features_in_: Int

    def __init__(
        out self,
        missing_values: Float64 = nan[DType.float64](),
        strategy: String = "mean",
        fill_value: Scalar[Self.compute_dtype] = 0.0,
    ) raises:
        """Initialize the SimpleImputer.

        Args:
            missing_values: Value representing missing entries (default NaN).
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant'). Default 'mean'.
            fill_value: Value to substitute when strategy='constant'. Default 0.0.

        Raises:
            InvalidParameterError: If strategy is unrecognized.
        """
        check_floating_dtype[Self.compute_dtype, "SimpleImputer"]()
        if (
            strategy != "mean"
            and strategy != "median"
            and strategy != "most_frequent"
            and strategy != "constant"
        ):
            raise InvalidParameterError.error(
                "strategy",
                "expected 'mean', 'median', 'most_frequent' or 'constant', got"
                " '"
                + strategy
                + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = List[Scalar[Self.compute_dtype]]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing SimpleImputer instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.missing_values = copy.missing_values
        self.strategy = copy.strategy
        self.fill_value = copy.fill_value
        self.statistics_ = copy.statistics_.copy()
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fit the imputer on feature matrix X.

        Args:
            X: Input data matrix with shape (n_samples, n_features).

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If a column contains only missing values when strategy is not 'constant'.
        """
        if X.rows == 0 or X.cols == 0:
            raise DimensionMismatchError.error(
                "non-empty 2D array",
                "array shape (" + String(X.rows) + ", " + String(X.cols) + ")",
                "SimpleImputer.fit",
            )

        var n_rows = X.rows
        var n_cols = X.cols

        self.statistics_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)

        for c in range(n_cols):
            if self.strategy == "constant":
                self.statistics_.append(self.fill_value)
                continue

            var valid = List[Scalar[Self.compute_dtype]](capacity=n_rows)
            for r in range(n_rows):
                if not _is_missing[in_dtype](X[r, c], self.missing_values):
                    valid.append(Scalar[Self.compute_dtype](X[r, c]))

            if len(valid) == 0:
                raise InvalidParameterError.error(
                    "X",
                    "Cannot compute "
                    + self.strategy
                    + " on column "
                    + String(c)
                    + " because all values are missing",
                )

            if self.strategy == "mean":
                var sum_val: Scalar[Self.compute_dtype] = 0.0
                for i in range(len(valid)):
                    sum_val += valid[i]
                var mean_val = sum_val / Scalar[Self.compute_dtype](len(valid))
                self.statistics_.append(mean_val)

            elif self.strategy == "median":
                sort(valid)
                var n_valid = len(valid)
                var median_val: Scalar[Self.compute_dtype]
                if n_valid % 2 == 1:
                    median_val = valid[n_valid // 2]
                else:
                    median_val = (
                        valid[n_valid // 2 - 1] + valid[n_valid // 2]
                    ) / 2.0
                self.statistics_.append(median_val)

            elif self.strategy == "most_frequent":
                sort(valid)
                var best_val = valid[0]
                var best_count = 1
                var curr_val = valid[0]
                var curr_count = 1

                for i in range(1, len(valid)):
                    if valid[i] == curr_val:
                        curr_count += 1
                    else:
                        if curr_count > best_count:
                            best_count = curr_count
                            best_val = curr_val
                        curr_val = valid[i]
                        curr_count = 1

                if curr_count > best_count:
                    best_val = curr_val

                self.statistics_.append(best_val)

        self.n_features_in_ = n_cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Impute all missing values in X.

        Args:
            X: Input data matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Matrix with missing values imputed.

        Raises:
            NotFittedError: If the imputer is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X is empty or X.cols != n_features_in_.
        """
        check_is_fitted("SimpleImputer", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "SimpleImputer.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.rows == 0 or X.cols == 0:
            raise DimensionMismatchError.error(
                "non-empty 2D array",
                "array shape (" + String(X.rows) + ", " + String(X.cols) + ")",
                "SimpleImputer.transform",
            )
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "SimpleImputer.transform",
            )

        var n_rows = X.rows
        var n_cols = X.cols
        var res = Matrix[in_dtype](n_rows, n_cols, 0)

        for r in range(n_rows):
            for c in range(n_cols):
                if _is_missing[in_dtype](X[r, c], self.missing_values):
                    res[r, c] = Scalar[in_dtype](self.statistics_[c])
                else:
                    res[r, c] = X[r, c]

        return res^

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fit to data, then transform it.

        Args:
            X: Input matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Matrix with missing values imputed.

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If a column contains only missing values under a non-constant strategy.
        """
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)
