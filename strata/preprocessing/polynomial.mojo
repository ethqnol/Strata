from ..base.estimator import Transformer
from ..core.matrix import Matrix
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


def _combinations_with_replacement_helper(
    n_features: Int,
    k: Int,
    start_idx: Int,
    mut current: List[Int],
    mut result: List[List[Int]],
):
    if len(current) == k:
        result.append(current.copy())
        return

    for i in range(start_idx, n_features):
        current.append(i)
        _combinations_with_replacement_helper(n_features, k, i, current, result)
        _ = current.pop()


def _combinations_helper(
    n_features: Int,
    k: Int,
    start_idx: Int,
    mut current: List[Int],
    mut result: List[List[Int]],
):
    if len(current) == k:
        result.append(current.copy())
        return

    for i in range(start_idx, n_features):
        current.append(i)
        _combinations_helper(n_features, k, i + 1, current, result)
        _ = current.pop()


def _generate_combinations(
    n_features: Int,
    min_degree: Int,
    max_degree: Int,
    interaction_only: Bool,
) -> List[List[Int]]:
    var result = List[List[Int]]()

    for d in range(min_degree, max_degree + 1):
        if d == 0:
            result.append(List[Int]())
        elif interaction_only:
            var current = List[Int]()
            _combinations_helper(n_features, d, 0, current, result)
        else:
            var current = List[Int]()
            _combinations_with_replacement_helper(
                n_features, d, 0, current, result
            )

    return result^


struct PolynomialFeatures[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Generate polynomial and interaction features.

    Generates a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is 2D and of the form $[a, b]$, the
    degree-2 polynomial features with bias are $[1, a, b, a^2, ab, b^2]$.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        degree: The maximal degree of polynomial features. Default 2.
        interaction_only: If True, only interaction features are produced:
            products of at most `degree` distinct input features. Default False.
        include_bias: If True, includes a bias column (all 1s) acting as an
            intercept term. Default True.

    Attributes:
        powers_: Exponent matrix with shape (n_output_features_, n_features_in_).
        n_features_in_: Number of features seen during fit.
        n_output_features_: Total number of polynomial output features.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import PolynomialFeatures
        from strata.core import Matrix

        var poly = PolynomialFeatures[DType.float64](degree=2)
        poly.fit(X)
        var X_poly = poly.transform(X)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var degree: Int
    var interaction_only: Bool
    var include_bias: Bool
    var combinations_: List[List[Int]]
    var powers_: Matrix[DType.int32]
    var n_features_in_: Int
    var n_output_features_: Int

    def __init__(
        out self,
        degree: Int = 2,
        interaction_only: Bool = False,
        include_bias: Bool = True,
    ) raises:
        """Initialize the PolynomialFeatures transformer.

        Args:
            degree: Maximum polynomial degree (>= 0). Default 2.
            interaction_only: Whether to produce only interaction terms. Default False.
            include_bias: Whether to include a bias column (degree 0). Default True.

        Raises:
            InvalidParameterError: If degree < 0.
        """
        check_floating_dtype[Self.compute_dtype, "PolynomialFeatures"]()
        if degree < 0:
            raise InvalidParameterError.error(
                "degree",
                "expected non-negative integer, got " + String(degree),
            )
        if degree == 0 and not include_bias:
            raise InvalidParameterError.error(
                "include_bias",
                "degree=0 and include_bias=False would yield 0 output features",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.combinations_ = List[List[Int]]()
        self.powers_ = Matrix[DType.int32](0, 0, 0)
        self.n_features_in_ = 0
        self.n_output_features_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing PolynomialFeatures instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.degree = copy.degree
        self.interaction_only = copy.interaction_only
        self.include_bias = copy.include_bias
        self.combinations_ = copy.combinations_.copy()
        self.powers_ = copy.powers_.copy()
        self.n_features_in_ = copy.n_features_in_
        self.n_output_features_ = copy.n_output_features_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Compute the number of output features and combination powers.

        Args:
            X: Input feature matrix with shape (n_samples, n_features).

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity.
        """
        check_array[in_dtype](X)
        var n_features = X.cols
        var min_degree = 0 if self.include_bias else 1

        self.combinations_ = _generate_combinations(
            n_features, min_degree, self.degree, self.interaction_only
        )
        self.n_features_in_ = n_features
        self.n_output_features_ = len(self.combinations_)

        # Build powers_ matrix of shape (n_output_features_, n_features_in_)
        self.powers_ = Matrix[DType.int32](
            self.n_output_features_, self.n_features_in_, 0
        )
        for i in range(self.n_output_features_):
            for j in range(len(self.combinations_[i])):
                var feat_idx = self.combinations_[i][j]
                self.powers_[i, feat_idx] += 1

        self.fit_dtype = in_dtype
        self.is_fitted = True

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Transform data matrix X to polynomial feature combinations.

        Args:
            X: Data matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Expanded polynomial matrix with shape (n_samples, n_output_features_).

        Raises:
            NotFittedError: If the transformer is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
            InvalidParameterError: If X contains NaN or infinity.
        """
        check_is_fitted("PolynomialFeatures", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "PolynomialFeatures.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[in_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "PolynomialFeatures.transform",
            )

        var n_rows = X.rows
        var res = Matrix[in_dtype](n_rows, self.n_output_features_, 0)

        for r in range(n_rows):
            for k in range(self.n_output_features_):
                var term_prod: Scalar[Self.compute_dtype] = 1.0
                for idx in range(len(self.combinations_[k])):
                    var f = self.combinations_[k][idx]
                    term_prod *= Scalar[Self.compute_dtype](X[r, f])
                res[r, k] = Scalar[in_dtype](term_prod)

        return res^

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fit to data, then transform it.

        Args:
            X: Input matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Expanded polynomial matrix.

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity.
        """
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)
