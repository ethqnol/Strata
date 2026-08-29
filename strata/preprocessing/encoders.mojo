from ..base.estimator import Transformer
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import (
    check_is_fitted,
    check_array,
    check_floating_dtype,
    check_finite,
)
from ..exceptions.errors import (
    NotFittedError,
    DataConversionError,
    DimensionMismatchError,
    InvalidParameterError,
)


def _index_of[
    dtype: DType
](categories: List[Scalar[dtype]], value: Scalar[dtype]) -> Int:
    var lo = 0
    var hi = len(categories)
    while lo < hi:
        var mid = (lo + hi) // 2
        if categories[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(categories) and categories[lo] == value:
        return lo
    return -1


struct OneHotEncoder[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Encode categorical features as a one-hot numeric array.

    The input to this transformer should be a 2D matrix of integer or float
    categorical features. The features are encoded using a one-hot (also known as
    'one-of-K' or 'dummy') encoding scheme.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        drop: Category dropping strategy ('none', 'first', 'if_binary'). Default 'none'.
        handle_unknown: Behavior for unseen categories during transform ('error', 'ignore'). Default 'error'.

    Attributes:
        categories_: Categories of each feature determined during fitting.
        drop_idx_: Indices of dropped categories for each feature.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import OneHotEncoder
        from strata.core import Matrix

        var encoder = OneHotEncoder[DType.float64](drop="if_binary")
        encoder.fit(X_cat)
        var X_encoded = encoder.transform(X_cat)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var handle_unknown: String
    var drop: String
    var categories_: List[List[Scalar[Self.compute_dtype]]]
    var drop_idx_: List[Int]
    var n_features_in_: Int

    def __init__(
        out self, drop: String = "none", handle_unknown: String = "error"
    ) raises:
        """Initialize the OneHotEncoder.

        Args:
            drop: Category dropping strategy ('none', 'first', 'if_binary'). Default 'none'.
            handle_unknown: Behavior for unseen categories ('error', 'ignore'). Default 'error'.

        Raises:
            InvalidParameterError: If drop or handle_unknown strategies are unrecognized.
        """

        check_floating_dtype[Self.compute_dtype, "OneHotEncoder"]()
        if drop != "none" and drop != "first" and drop != "if_binary":
            raise InvalidParameterError.error(
                "drop",
                "expected 'none', 'first' or 'if_binary', got '" + drop + "'",
            )
        if handle_unknown != "error" and handle_unknown != "ignore":
            raise InvalidParameterError.error(
                "handle_unknown",
                "expected 'error' or 'ignore', got '" + handle_unknown + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.categories_ = List[List[Scalar[Self.compute_dtype]]]()
        self.drop_idx_ = List[Int]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing OneHotEncoder instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.handle_unknown = copy.handle_unknown
        self.drop = copy.drop
        self.categories_ = copy.categories_.copy()
        self.drop_idx_ = copy.drop_idx_.copy()
        self.n_features_in_ = copy.n_features_in_

    def n_features_out(self) -> Int:
        """Number of indicator columns produced by transform."""
        var total = 0
        for f in range(len(self.categories_)):
            total += len(self.categories_[f])
            if self.drop_idx_[f] >= 0:
                total -= 1
        return total

    def _column_offsets(self) -> List[Int]:
        var offsets = List[Int](capacity=len(self.categories_) + 1)
        var running = 0
        for f in range(len(self.categories_)):
            offsets.append(running)
            running += len(self.categories_[f])
            if self.drop_idx_[f] >= 0:
                running -= 1
        offsets.append(running)
        return offsets^

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.categories_ = List[List[Scalar[Self.compute_dtype]]](
            capacity=n_cols
        )
        self.drop_idx_ = List[Int](capacity=n_cols)

        for c in range(n_cols):
            var col = List[Scalar[Self.compute_dtype]](capacity=n_rows)
            for r in range(n_rows):
                col.append(Scalar[Self.compute_dtype](X[r, c]))
            sort(col)

            var uniques = List[Scalar[Self.compute_dtype]]()
            for i in range(len(col)):
                if i == 0 or col[i] != col[i - 1]:
                    uniques.append(col[i])

            if self.drop == "first" and len(uniques) > 0:
                self.drop_idx_.append(0)
            elif self.drop == "if_binary" and len(uniques) == 2:
                self.drop_idx_.append(0)
            else:
                self.drop_idx_.append(-1)

            self.categories_.append(uniques^)

        self.n_features_in_ = n_cols
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
        check_is_fitted("OneHotEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OneHotEncoder.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != len(self.categories_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.categories_)),
                "X.cols == " + String(X.cols),
                "OneHotEncoder.transform",
            )

        var offsets = self._column_offsets()
        var res = Matrix[in_dtype](X.rows, offsets[X.cols], 0)

        for r in range(X.rows):
            for c in range(X.cols):
                var value = Scalar[Self.compute_dtype](X[r, c])
                var idx = _index_of[Self.compute_dtype](
                    self.categories_[c], value
                )
                if idx < 0:
                    if self.handle_unknown == "error":
                        raise InvalidParameterError.error(
                            "X",
                            "OneHotEncoder.transform found unknown category "
                            + String(value)
                            + " in column "
                            + String(c),
                        )
                    continue

                var dropped = self.drop_idx_[c]
                if idx == dropped:
                    continue
                if dropped >= 0 and idx > dropped:
                    idx -= 1
                res[r, offsets[c] + idx] = 1

        return res^

    def get_feature_names_out(
        self, input_features: List[String] = List[String]()
    ) raises -> List[String]:
        """Output column names as '<feature>_<category>' pairs."""
        check_is_fitted("OneHotEncoder", self.is_fitted)
        if len(input_features) != 0 and len(input_features) != len(
            self.categories_
        ):
            raise DimensionMismatchError.error(
                "len(input_features) == " + String(len(self.categories_)),
                "len(input_features) == " + String(len(input_features)),
                "OneHotEncoder.get_feature_names_out",
            )

        var names = List[String](capacity=self.n_features_out())
        for f in range(len(self.categories_)):
            var base: String
            if len(input_features) != 0:
                base = input_features[f]
            else:
                base = "x" + String(f)
            for i in range(len(self.categories_[f])):
                if i == self.drop_idx_[f]:
                    continue
                names.append(base + "_" + String(self.categories_[f][i]))
        return names^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var encoded_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            encoded_records^,
            dataset.targets.copy(),
            self.get_feature_names_out(dataset.feature_names),
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
        """Recovers the original categorical values from a one-hot matrix."""
        check_is_fitted("OneHotEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OneHotEncoder.inverse_transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X, allow_empty=X.cols == 0)
        var n_features = len(self.categories_)
        var offsets = self._column_offsets()
        if X.cols != offsets[n_features]:
            raise DimensionMismatchError.error(
                "X.cols == " + String(offsets[n_features]),
                "X.cols == " + String(X.cols),
                "OneHotEncoder.inverse_transform",
            )

        var res = Matrix[in_dtype](X.rows, n_features, 0)
        for r in range(X.rows):
            for f in range(n_features):
                var block = offsets[f + 1] - offsets[f]
                var active = -1
                for j in range(block):
                    if X[r, offsets[f] + j] != 0:
                        active = j
                        break

                var dropped = self.drop_idx_[f]
                if active < 0:
                    if dropped < 0:
                        raise InvalidParameterError.error(
                            "X",
                            "OneHotEncoder.inverse_transform found an"
                            " all-zero block for column "
                            + String(f),
                        )
                    res[r, f] = Scalar[in_dtype](self.categories_[f][dropped])
                    continue

                if dropped >= 0 and active >= dropped:
                    active += 1
                res[r, f] = Scalar[in_dtype](self.categories_[f][active])

        return res^


struct OrdinalEncoder[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Encode categorical features as an integer array.

    The input to this transformer should be a 2D matrix of integer or float
    categorical features. The features are converted to ordinal integers
    ($0, 1, \\dots, K_c - 1$) corresponding to the sorted order of unique
    categories per feature.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        handle_unknown: When set to 'error' an error is raised on unknown categories.
            When set to 'use_encoded_value', unknown categories are set to unknown_value. Default 'error'.
        unknown_value: Value used when handle_unknown='use_encoded_value'. Default -1.0.

    Attributes:
        categories_: Categories of each feature determined during fitting.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import OrdinalEncoder
        from strata.core import Matrix

        var encoder = OrdinalEncoder[DType.float64]()
        encoder.fit(X_cat)
        var X_ord = encoder.transform(X_cat)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var handle_unknown: String
    var unknown_value: Scalar[Self.compute_dtype]
    var categories_: List[List[Scalar[Self.compute_dtype]]]
    var n_features_in_: Int

    def __init__(
        out self,
        handle_unknown: String = "error",
        unknown_value: Scalar[Self.compute_dtype] = -1.0,
    ) raises:
        """Initialize the OrdinalEncoder.

        Args:
            handle_unknown: Behavior for unseen categories ('error', 'use_encoded_value'). Default 'error'.
            unknown_value: Numerical value assigned to unseen categories when handle_unknown='use_encoded_value'. Default -1.0.

        Raises:
            InvalidParameterError: If handle_unknown is not 'error' or 'use_encoded_value'.
        """
        check_floating_dtype[Self.compute_dtype, "OrdinalEncoder"]()
        if handle_unknown != "error" and handle_unknown != "use_encoded_value":
            raise InvalidParameterError.error(
                "handle_unknown",
                "expected 'error' or 'use_encoded_value', got '"
                + handle_unknown
                + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_ = List[List[Scalar[Self.compute_dtype]]]()
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing OrdinalEncoder instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.handle_unknown = copy.handle_unknown
        self.unknown_value = copy.unknown_value
        self.categories_ = copy.categories_.copy()
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fit the OrdinalEncoder on feature matrix X.

        Args:
            X: Categorical data matrix with shape (n_samples, n_features).

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity.
        """
        check_array[in_dtype](X)
        var n_rows = X.rows
        var n_cols = X.cols

        self.categories_ = List[List[Scalar[Self.compute_dtype]]](
            capacity=n_cols
        )

        for c in range(n_cols):
            var col = List[Scalar[Self.compute_dtype]](capacity=n_rows)
            for r in range(n_rows):
                col.append(Scalar[Self.compute_dtype](X[r, c]))
            sort(col)

            var uniques = List[Scalar[Self.compute_dtype]]()
            for i in range(len(col)):
                if i == 0 or col[i] != col[i - 1]:
                    uniques.append(col[i])

            self.categories_.append(uniques^)

        self.n_features_in_ = n_cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Transform X to ordinal integer codes.

        Args:
            X: Data matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Encoded ordinal matrix.

        Raises:
            NotFittedError: If the encoder is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
            InvalidParameterError: If unseen category is encountered when handle_unknown='error', or if inputs contain NaN/Inf.
        """
        check_is_fitted("OrdinalEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OrdinalEncoder.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != len(self.categories_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.categories_)),
                "X.cols == " + String(X.cols),
                "OrdinalEncoder.transform",
            )

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var val = Scalar[Self.compute_dtype](X[r, c])
                var idx = _index_of(self.categories_[c], val)
                if idx < 0:
                    if self.handle_unknown == "error":
                        raise InvalidParameterError.error(
                            "X",
                            "Found unknown category "
                            + String(val)
                            + " in column "
                            + String(c)
                            + " during transform",
                        )
                    res[r, c] = Scalar[in_dtype](self.unknown_value)
                else:
                    res[r, c] = Scalar[in_dtype](idx)
        return res^

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fit to data, then transform it.

        Args:
            X: Input matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Encoded ordinal matrix.

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity.
        """
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def inverse_transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Convert the ordinal codes back to original category values.

        Args:
            X: Encoded data matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Decoded matrix containing reconstructed categories.

        Raises:
            NotFittedError: If the encoder is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
            InvalidParameterError: If a category index is not an integer or is out of bounds for the respective column.
        """
        check_is_fitted("OrdinalEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "OrdinalEncoder.inverse_transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != len(self.categories_):
            raise DimensionMismatchError.error(
                "X.cols == " + String(len(self.categories_)),
                "X.cols == " + String(X.cols),
                "OrdinalEncoder.inverse_transform",
            )

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                var raw_idx = X[r, c]
                var idx = Int(raw_idx)
                if (
                    raw_idx != Scalar[in_dtype](idx)
                    or idx < 0
                    or idx >= len(self.categories_[c])
                ):
                    raise InvalidParameterError.error(
                        "X",
                        "Cannot invert non-integer or out-of-bounds category"
                        " index "
                        + String(raw_idx)
                        + " in column "
                        + String(c),
                    )
                res[r, c] = Scalar[in_dtype](self.categories_[c][idx])
        return res^


struct LabelEncoder[compute_dtype: DType = DType.float64](Copyable, Movable):
    """Encode target labels with value between 0 and n_classes-1.

    Used to transform non-numerical or non-consecutive 1D target labels
    into continuous integer labels for classification tasks.

    Parameters:
        compute_dtype: Precision used for internal class representation. Default DType.float64.

    Attributes:
        classes_: Distinct classes seen during fit, in sorted order.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import LabelEncoder

        var encoder = LabelEncoder[DType.float64]()
        encoder.fit(y_train)
        var y_encoded = encoder.transform(y_train)
        var y_original = encoder.inverse_transform(y_encoded)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var classes_: List[Scalar[Self.compute_dtype]]

    def __init__(out self) raises:
        """Initialize the LabelEncoder."""
        check_floating_dtype[Self.compute_dtype, "LabelEncoder"]()
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.classes_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        """Copies an existing LabelEncoder instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.classes_ = copy.classes_.copy()

    def fit[in_dtype: DType](mut self, y: List[Scalar[in_dtype]]) raises:
        """Fit label encoder on target vector y.

        Args:
            y: Target label vector.

        Raises:
            InvalidParameterError: If y is empty or contains NaN / Inf.
        """
        if len(y) == 0:
            raise InvalidParameterError.error(
                "y", "LabelEncoder.fit requires at least one sample"
            )
        check_finite[in_dtype](y, "y", "LabelEncoder.fit")

        var col = List[Scalar[Self.compute_dtype]](capacity=len(y))
        for i in range(len(y)):
            col.append(Scalar[Self.compute_dtype](y[i]))
        sort(col)

        var uniques = List[Scalar[Self.compute_dtype]]()
        for i in range(len(col)):
            if i == 0 or col[i] != col[i - 1]:
                uniques.append(col[i])

        self.classes_ = uniques^
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def transform[
        in_dtype: DType
    ](self, y: List[Scalar[in_dtype]]) raises -> List[Int]:
        """Transform target labels to normalized encoding indices.

        Args:
            y: Target label vector to transform.

        Returns:
            List[Int]: Encoded integer labels in range [0, n_classes - 1].

        Raises:
            NotFittedError: If the encoder is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            InvalidParameterError: If y contains unseen labels or NaN / Inf.
        """
        check_is_fitted("LabelEncoder", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "LabelEncoder.transform received List["
                + String(in_dtype)
                + "] but was fitted on List["
                + String(self.fit_dtype)
                + "]"
            )
        check_finite[in_dtype](y, "y", "LabelEncoder.transform")

        var res = List[Int](capacity=len(y))
        for i in range(len(y)):
            var val = Scalar[Self.compute_dtype](y[i])
            var idx = _index_of(self.classes_, val)
            if idx < 0:
                raise InvalidParameterError.error(
                    "y",
                    "y contains previously unseen labels: " + String(val),
                )
            res.append(idx)
        return res^

    def fit_transform[
        in_dtype: DType
    ](mut self, y: List[Scalar[in_dtype]]) raises -> List[Int]:
        """Fit label encoder and return encoded integer labels.

        Args:
            y: Target label vector.

        Returns:
            List[Int]: Encoded integer labels.

        Raises:
            InvalidParameterError: If y is empty or contains NaN / Inf.
        """
        self.fit[in_dtype](y)
        return self.transform[in_dtype](y)

    def inverse_transform[
        out_dtype: DType = Self.compute_dtype
    ](self, y: List[Int]) raises -> List[Scalar[out_dtype]]:
        """Transform integer labels back to original encoding.

        Parameters:
            out_dtype: Output precision data type. Default compute_dtype.

        Args:
            y: Encoded integer labels.

        Returns:
            List[Scalar[out_dtype]]: Reconstructed original labels.

        Raises:
            NotFittedError: If the encoder is not fitted.
            InvalidParameterError: If an integer code is out of bounds.
        """
        check_is_fitted("LabelEncoder", self.is_fitted)

        var res = List[Scalar[out_dtype]](capacity=len(y))
        for i in range(len(y)):
            var idx = y[i]
            if idx < 0 or idx >= len(self.classes_):
                raise InvalidParameterError.error(
                    "y",
                    "Cannot invert unseen label index: " + String(idx),
                )
            res.append(Scalar[out_dtype](self.classes_[idx]))
        return res^
