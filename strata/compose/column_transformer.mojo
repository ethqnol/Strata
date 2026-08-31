from ..core.matrix import Matrix, hstack
from ..base.estimator import Transformer
from ..utils.validation import check_array, check_is_fitted
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct ColumnTransformer[*Transformers: Transformer](
    Copyable, Movable, Transformer
):
    """Applies a heterogeneous variadic sequence of transformers to column subsets of a feature matrix.

    Transforms designated column subsets independently using compile-time unrolled variadic
    tuples and horizontally stacks the output representations with configurable remainder handling.

    Parameters:
        Transformers: Variadic pack of Transformer types.

    Attributes:
        transformers: Variadic tuple of transformer instances.
        column_specs: List of column index lists corresponding to each transformer step.
        remainder: Strategy for unselected columns ('drop' or 'passthrough'). Default 'drop'.

    Fitted Attributes:
        remainder_cols_: Discovered unselected column indices when remainder='passthrough'.
        n_features_in_: Total number of feature columns seen during fit.
        is_fitted: Boolean flag indicating if transformer has been fitted.

    Examples:
        ```mojo
        from strata import Matrix, StandardScaler, OneHotEncoder, ColumnTransformer

        var ct = ColumnTransformer(
            (StandardScaler(), OneHotEncoder()),
            List[List[Int]]([0, 1], [2]),
            remainder="passthrough",
        )
        ```
    """

    var transformers: Tuple[*Self.Transformers]
    var column_specs: List[List[Int]]
    var remainder: String

    var remainder_cols_: List[Int]
    var n_features_in_: Int
    var is_fitted: Bool

    def __init__(
        out self,
        transformers: Tuple[*Self.Transformers],
        column_specs: List[List[Int]],
        remainder: String = "drop",
    ) raises:
        """Initializes a ColumnTransformer with variadic transformers and column specs.

        Args:
            transformers: Tuple of transformer instances satisfying the Transformer trait.
            column_specs: List of column index lists matching the count of transformers.
            remainder: Strategy for unselected columns ('drop' or 'passthrough'). Default 'drop'.

        Raises:
            InvalidParameterError: If remainder is invalid or column_specs count != transformers count.
        """
        if remainder != "drop" and remainder != "passthrough":
            raise InvalidParameterError.error(
                "remainder",
                "Supported remainder strategies are 'drop' or 'passthrough',"
                " got '"
                + remainder
                + "'",
            )
        if len(column_specs) != len(Self.Transformers):
            raise InvalidParameterError.error(
                "column_specs",
                "Number of column specifications ("
                + String(len(column_specs))
                + ") must match number of transformers ("
                + String(len(Self.Transformers))
                + ")",
            )

        self.transformers = transformers.copy()
        self.column_specs = column_specs.copy()
        self.remainder = remainder

        self.remainder_cols_ = List[Int]()
        self.n_features_in_ = 0
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        """Creates a deep copy of an existing ColumnTransformer instance."""
        self.transformers = copy.transformers.copy()
        self.column_specs = List[List[Int]](capacity=len(copy.column_specs))
        for i in range(len(copy.column_specs)):
            self.column_specs.append(copy.column_specs[i].copy())
        self.remainder = copy.remainder

        self.remainder_cols_ = copy.remainder_cols_.copy()
        self.n_features_in_ = copy.n_features_in_
        self.is_fitted = copy.is_fitted

    def _compute_remainder_cols(self, n_cols: Int) -> List[Int]:
        """Calculates column indices not assigned to any transformer step."""
        var rem = List[Int]()
        for c in range(n_cols):
            var found = False
            for step_idx in range(len(self.column_specs)):
                for i in range(len(self.column_specs[step_idx])):
                    if self.column_specs[step_idx][i] == c:
                        found = True
                        break
                if found:
                    break
            if not found:
                rem.append(c)
        return rem^

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fits all transformers on their respective designated column subsets.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Raises:
            InvalidParameterError: If any column index is out of bounds $[0, D-1]$.
        """
        check_array[in_dtype](X)
        self.n_features_in_ = X.cols

        for step_idx in range(len(self.column_specs)):
            for i in range(len(self.column_specs[step_idx])):
                var c = self.column_specs[step_idx][i]
                if c < 0 or c >= X.cols:
                    raise InvalidParameterError.error(
                        "column_specs["
                        + String(step_idx)
                        + "]["
                        + String(i)
                        + "]",
                        "Column index "
                        + String(c)
                        + " is out of bounds for feature matrix with "
                        + String(X.cols)
                        + " columns",
                    )

        self.remainder_cols_ = self._compute_remainder_cols(X.cols)

        comptime for i in range(len(Self.Transformers)):
            if len(self.column_specs[i]) > 0:
                var Xi = X.select_columns(self.column_specs[i])
                var trans = self.transformers[i].copy()
                trans.fit[in_dtype](Xi)
                self.transformers[i] = trans^

        self.is_fitted = True

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Transforms column subsets and horizontally stacks output representations.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Matrix[in_dtype]: Concatenated transformed feature matrix.

        Raises:
            NotFittedError: If transformer has not been fitted yet.
            DimensionMismatchError: If column count of X != n_features_in_.
        """
        check_is_fitted("ColumnTransformer", self.is_fitted)
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "ColumnTransformer.transform",
            )

        var parts = List[Matrix[in_dtype]]()

        comptime for i in range(len(Self.Transformers)):
            if len(self.column_specs[i]) > 0:
                var Xi = X.select_columns(self.column_specs[i])
                var Ti_res = self.transformers[i].transform[in_dtype](Xi)
                parts.append(Ti_res^)

        if self.remainder == "passthrough" and len(self.remainder_cols_) > 0:
            var X_rem = X.select_columns(self.remainder_cols_)
            parts.append(X_rem^)

        if len(parts) == 0:
            return Matrix[in_dtype](X.rows, 0)

        return hstack[in_dtype](parts)

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fits all transformers and returns concatenated transformed output.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Matrix[in_dtype]: Concatenated transformed feature matrix.
        """
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)
