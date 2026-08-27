from std.math import sqrt, floor
from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import (
    check_is_fitted,
    check_array,
    check_floating_dtype,
)
from ..exceptions.errors import (
    NotFittedError,
    DataConversionError,
    DimensionMismatchError,
    InvalidParameterError,
)


struct StandardScaler[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Standardizes features by removing the mean and scaling to unit variance.
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var with_mean: Bool
    var with_std: Bool
    var mean_: List[Scalar[Self.compute_dtype]]
    var scale_: List[Scalar[Self.compute_dtype]]

    def __init__(out self, with_mean: Bool = True, with_std: Bool = True):
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = List[Scalar[Self.compute_dtype]]()
        self.scale_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        """Copies an existing StandardScaler instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.with_mean = copy.with_mean
        self.with_std = copy.with_std
        self.mean_ = copy.mean_.copy()
        self.scale_ = copy.scale_.copy()

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.mean_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.scale_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)

        for c in range(n_cols):
            var col_sum: Scalar[Self.compute_dtype] = 0
            for r in range(n_rows):
                col_sum += Scalar[Self.compute_dtype](X[r, c])
            var mean_val = col_sum / Scalar[Self.compute_dtype](n_rows)
            self.mean_.append(mean_val)

            var var_sum: Scalar[Self.compute_dtype] = 0
            for r in range(n_rows):
                var diff = Scalar[Self.compute_dtype](X[r, c]) - mean_val
                var_sum += diff * diff
            var std_val = sqrt(var_sum / Scalar[Self.compute_dtype](n_rows))
            if std_val == 0:
                std_val = 1
            self.scale_.append(std_val)

        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        check_is_fitted("StandardScaler", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "StandardScaler.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.cols != len(self.mean_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.mean_)),
                "X.cols == " + String(X.cols),
                "StandardScaler.transform",
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = Scalar[Self.compute_dtype](X[r, c])
                if self.with_mean:
                    val -= self.mean_[c]
                if self.with_std:
                    val /= self.scale_[c]
                res[r, c] = Scalar[in_dtype](val)
        return res^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var scaled_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            scaled_records^,
            dataset.targets.copy(),
            dataset.feature_names.copy(),
            dataset.target_names.copy(),
        )

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)


struct MinMaxScaler[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Transforms features by scaling each one into a given range.

    Each feature is scaled and translated individually so that it spans
    [feature_range_min, feature_range_max] on the training set.
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var feature_range_min: Scalar[Self.compute_dtype]
    var feature_range_max: Scalar[Self.compute_dtype]
    var clip: Bool
    var data_min_: List[Scalar[Self.compute_dtype]]
    var data_max_: List[Scalar[Self.compute_dtype]]
    var data_range_: List[Scalar[Self.compute_dtype]]
    var scale_: List[Scalar[Self.compute_dtype]]
    var min_: List[Scalar[Self.compute_dtype]]

    def __init__(
        out self,
        feature_range_min: Scalar[Self.compute_dtype] = 0.0,
        feature_range_max: Scalar[Self.compute_dtype] = 1.0,
        clip: Bool = False,
    ) raises:
        """Initializes the MinMaxScaler.

        Args:
            feature_range_min: Lower bound of the transformed range.
            feature_range_max: Upper bound of the transformed range.
            clip: Whether to clip transformed values to the feature range.
        """
        check_floating_dtype[Self.compute_dtype, "MinMaxScaler"]()
        if not feature_range_min < feature_range_max:
            raise InvalidParameterError.error(
                "feature_range",
                "feature_range_min must be strictly less than"
                " feature_range_max, got ("
                + String(feature_range_min)
                + ", "
                + String(feature_range_max)
                + ")",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.feature_range_min = feature_range_min
        self.feature_range_max = feature_range_max
        self.clip = clip
        self.data_min_ = List[Scalar[Self.compute_dtype]]()
        self.data_max_ = List[Scalar[Self.compute_dtype]]()
        self.data_range_ = List[Scalar[Self.compute_dtype]]()
        self.scale_ = List[Scalar[Self.compute_dtype]]()
        self.min_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        """Copies an existing MinMaxScaler instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.feature_range_min = copy.feature_range_min
        self.feature_range_max = copy.feature_range_max
        self.clip = copy.clip
        self.data_min_ = copy.data_min_.copy()
        self.data_max_ = copy.data_max_.copy()
        self.data_range_ = copy.data_range_.copy()
        self.scale_ = copy.scale_.copy()
        self.min_ = copy.min_.copy()

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.data_min_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.data_max_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.data_range_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.scale_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.min_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)

        var span = self.feature_range_max - self.feature_range_min

        for c in range(n_cols):
            var col_min = Scalar[Self.compute_dtype](X[0, c])
            var col_max = col_min
            for r in range(1, n_rows):
                var val = Scalar[Self.compute_dtype](X[r, c])
                if val < col_min:
                    col_min = val
                if val > col_max:
                    col_max = val

            var col_range = col_max - col_min
            var denom = col_range
            if denom == 0:
                denom = 1

            self.data_min_.append(col_min)
            self.data_max_.append(col_max)
            self.data_range_.append(col_range)
            self.scale_.append(span / denom)
            self.min_.append(self.feature_range_min - col_min * (span / denom))

        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        check_is_fitted("MinMaxScaler", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "MinMaxScaler.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.cols != len(self.scale_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.scale_)),
                "X.cols == " + String(X.cols),
                "MinMaxScaler.transform",
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = (
                    Scalar[Self.compute_dtype](X[r, c]) * self.scale_[c]
                    + self.min_[c]
                )
                if self.clip:
                    if val < self.feature_range_min:
                        val = self.feature_range_min
                    elif val > self.feature_range_max:
                        val = self.feature_range_max
                res[r, c] = Scalar[in_dtype](val)
        return res^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var scaled_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            scaled_records^,
            dataset.targets.copy(),
            dataset.feature_names.copy(),
            dataset.target_names.copy(),
        )

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)

    def inverse_transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Undoes the scaling of X according to the fitted feature range."""
        check_is_fitted("MinMaxScaler", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "MinMaxScaler.inverse_transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.cols != len(self.scale_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.scale_)),
                "X.cols == " + String(X.cols),
                "MinMaxScaler.inverse_transform",
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = (
                    Scalar[Self.compute_dtype](X[r, c]) - self.min_[c]
                ) / self.scale_[c]
                res[r, c] = Scalar[in_dtype](val)
        return res^


def _quantile[
    dtype: DType
](sorted_vals: List[Scalar[dtype]], q: Float64) -> Scalar[dtype]:
    var n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    var pos = (q / 100.0) * Float64(n - 1)
    var lo = Int(floor(pos))
    if lo >= n - 1:
        return sorted_vals[n - 1]

    var frac = Scalar[dtype](pos - Float64(lo))
    return sorted_vals[lo] + (sorted_vals[lo + 1] - sorted_vals[lo]) * frac


struct RobustScaler[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Scales features using statistics that are robust to outliers.

    Centers on the median and scales by the configured quantile range,
    which defaults to the interquartile range (25th to 75th percentile).
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var with_centering: Bool
    var with_scaling: Bool
    var quantile_min: Float64
    var quantile_max: Float64
    var center_: List[Scalar[Self.compute_dtype]]
    var scale_: List[Scalar[Self.compute_dtype]]

    def __init__(
        out self,
        with_centering: Bool = True,
        with_scaling: Bool = True,
        quantile_min: Float64 = 25.0,
        quantile_max: Float64 = 75.0,
    ) raises:
        """Initializes the RobustScaler.

        Args:
            with_centering: Whether to center the data on the median.
            with_scaling: Whether to scale the data to the quantile range.
            quantile_min: Lower quantile percentage of the scaling range.
            quantile_max: Upper quantile percentage of the scaling range.
        """
        check_floating_dtype[Self.compute_dtype, "RobustScaler"]()
        if (
            quantile_min < 0.0
            or quantile_max > 100.0
            or not quantile_min < quantile_max
        ):
            raise InvalidParameterError.error(
                "quantile_range",
                "expected 0 <= quantile_min < quantile_max <= 100, got ("
                + String(quantile_min)
                + ", "
                + String(quantile_max)
                + ")",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_min = quantile_min
        self.quantile_max = quantile_max
        self.center_ = List[Scalar[Self.compute_dtype]]()
        self.scale_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        """Copies an existing RobustScaler instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.with_centering = copy.with_centering
        self.with_scaling = copy.with_scaling
        self.quantile_min = copy.quantile_min
        self.quantile_max = copy.quantile_max
        self.center_ = copy.center_.copy()
        self.scale_ = copy.scale_.copy()

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.center_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)
        self.scale_ = List[Scalar[Self.compute_dtype]](capacity=n_cols)

        var col = List[Scalar[Self.compute_dtype]](capacity=n_rows)

        for c in range(n_cols):
            col.clear()
            for r in range(n_rows):
                col.append(Scalar[Self.compute_dtype](X[r, c]))
            sort(col)

            self.center_.append(_quantile[Self.compute_dtype](col, 50.0))

            var q_lo = _quantile[Self.compute_dtype](col, self.quantile_min)
            var q_hi = _quantile[Self.compute_dtype](col, self.quantile_max)
            var iqr = q_hi - q_lo
            if iqr == 0:
                iqr = 1
            self.scale_.append(iqr)

        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        check_is_fitted("RobustScaler", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "RobustScaler.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.cols != len(self.center_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.center_)),
                "X.cols == " + String(X.cols),
                "RobustScaler.transform",
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = Scalar[Self.compute_dtype](X[r, c])
                if self.with_centering:
                    val -= self.center_[c]
                if self.with_scaling:
                    val /= self.scale_[c]
                res[r, c] = Scalar[in_dtype](val)
        return res^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var scaled_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            scaled_records^,
            dataset.targets.copy(),
            dataset.feature_names.copy(),
            dataset.target_names.copy(),
        )

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)

    def inverse_transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Undoes the centering and scaling of X."""
        check_is_fitted("RobustScaler", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "RobustScaler.inverse_transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.cols != len(self.center_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.center_)),
                "X.cols == " + String(X.cols),
                "RobustScaler.inverse_transform",
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = Scalar[Self.compute_dtype](X[r, c])
                if self.with_scaling:
                    val *= self.scale_[c]
                if self.with_centering:
                    val += self.center_[c]
                res[r, c] = Scalar[in_dtype](val)
        return res^
