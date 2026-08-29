from std.math import sqrt, abs
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


struct Normalizer[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one non-zero
    component is rescaled independently of other samples so that its norm
    ($L_1$, $L_2$ or $\\text{max}$) equals one.

    $$
    x_{\\text{norm}} = \\frac{x}{\\|x\\|_p}
    $$

    where $\\|x\\|_p$ is the chosen vector norm:
    - $L_1$: $\\|x\\|_1 = \\sum_j |x_j|$
    - $L_2$: $\\|x\\|_2 = \\sqrt{\\sum_j x_j^2}$
    - $\\text{max}$: $\\|x\\|_\\infty = \\max_j |x_j|$

    Rows of all zeros remain all zeros.

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        norm: The norm to use to normalize each non-zero sample ('l1', 'l2', or 'max'). Default 'l2'.

    Attributes:
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.preprocessing import Normalizer
        from strata.core import Matrix

        var normalizer = Normalizer[DType.float64](norm="l2")
        normalizer.fit(X_train)
        var X_norm = normalizer.transform(X_train)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var norm: String
    var n_features_in_: Int

    def __init__(out self, norm: String = "l2") raises:
        """Initialize the Normalizer.

        Args:
            norm: The norm to normalize each sample with ('l1', 'l2', or 'max'). Default 'l2'.

        Raises:
            InvalidParameterError: If norm is not 'l1', 'l2', or 'max'.
        """
        check_floating_dtype[Self.compute_dtype, "Normalizer"]()
        if norm != "l1" and norm != "l2" and norm != "max":
            raise InvalidParameterError.error(
                "norm",
                "expected 'l1', 'l2' or 'max', got '" + norm + "'",
            )
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.norm = norm
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing Normalizer instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.norm = copy.norm
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fit the transformer on feature matrix X.

        Args:
            X: Training data matrix with shape (n_samples, n_features).

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity.
        """
        check_array[in_dtype](X)
        self.n_features_in_ = X.cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fit the transformer on Dataset records.

        Args:
            dataset: Training Dataset containing records matrix.

        Raises:
            DimensionMismatchError: If records matrix is empty.
            InvalidParameterError: If records contain NaN or infinity.
        """
        self.fit[feat_dtype](dataset.records)

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Scale each non-zero sample in X to unit norm.

        Args:
            X: Data matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Normalized feature matrix.

        Raises:
            NotFittedError: If the transformer is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
            InvalidParameterError: If X contains NaN or infinity.
        """
        check_is_fitted("Normalizer", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "Normalizer.transform received Matrix["
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
                "Normalizer.transform",
            )

        var n_rows = X.rows
        var n_cols = X.cols
        var res = Matrix[in_dtype](n_rows, n_cols, 0)

        for r in range(n_rows):
            var norm_val: Scalar[Self.compute_dtype] = 0.0

            if self.norm == "l1":
                for c in range(n_cols):
                    norm_val += abs(Scalar[Self.compute_dtype](X[r, c]))
            elif self.norm == "l2":
                var sum_sq: Scalar[Self.compute_dtype] = 0.0
                for c in range(n_cols):
                    var v = Scalar[Self.compute_dtype](X[r, c])
                    sum_sq += v * v
                norm_val = sqrt(sum_sq)
            else:  # "max"
                for c in range(n_cols):
                    var v_abs = abs(Scalar[Self.compute_dtype](X[r, c]))
                    if v_abs > norm_val:
                        norm_val = v_abs

            if norm_val > 0.0:
                for c in range(n_cols):
                    var scaled = Scalar[Self.compute_dtype](X[r, c]) / norm_val
                    res[r, c] = Scalar[in_dtype](scaled)

        return res^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        """Scale records of a Dataset to unit norm.

        Args:
            dataset: Dataset container to transform.

        Returns:
            Dataset[feat_dtype, target_dtype]: Transformed Dataset with normalized records.

        Raises:
            NotFittedError: If the transformer is not fitted.
            DataConversionError: If feat_dtype does not match fit_dtype.
            DimensionMismatchError: If records cols != n_features_in_.
        """
        var normalized_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            normalized_records^,
            dataset.targets.copy(),
            dataset.feature_names.copy(),
            dataset.target_names.copy(),
        )

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fit to data, then transform it.

        Args:
            X: Input matrix with shape (n_samples, n_features).

        Returns:
            Matrix[in_dtype]: Normalized feature matrix.

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity.
        """
        self.fit[in_dtype](X)
        return self.transform[in_dtype](X)

    def fit_transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        """Fit to dataset, then transform it.

        Args:
            dataset: Input Dataset.

        Returns:
            Dataset[feat_dtype, target_dtype]: Transformed Dataset with normalized records.

        Raises:
            DimensionMismatchError: If records matrix is empty.
            InvalidParameterError: If records contain NaN or infinity.
        """
        self.fit[feat_dtype, target_dtype](dataset)
        return self.transform[feat_dtype, target_dtype](dataset)
