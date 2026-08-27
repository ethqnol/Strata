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
)


struct Binarizer[compute_dtype: DType = DType.float64](
    Copyable, Movable, Transformer
):
    """Binarizes features according to a threshold.

    Values strictly greater than the threshold map to 1, all others map to 0.
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var threshold: Scalar[Self.compute_dtype]
    var n_features_in_: Int

    def __init__(out self, threshold: Scalar[Self.compute_dtype] = 0.0) raises:
        """Initializes the Binarizer.

        Args:
            threshold: Feature values above this are mapped to 1, others to 0.
        """
        check_floating_dtype[Self.compute_dtype, "Binarizer"]()
        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.threshold = threshold
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing Binarizer instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.threshold = copy.threshold
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        check_array[in_dtype](X)
        self.n_features_in_ = X.cols
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
        check_is_fitted("Binarizer", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "Binarizer.transform received Matrix["
                + String(in_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "Binarizer.transform",
            )
        check_array[in_dtype](X)

        var res = Matrix[in_dtype](X.rows, X.cols, 0)
        for r in range(X.rows):
            for c in range(X.cols):
                if Scalar[Self.compute_dtype](X[r, c]) > self.threshold:
                    res[r, c] = 1
        return res^

    def transform[
        feat_dtype: DType,
        target_dtype: DType,
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> Dataset[
        feat_dtype, target_dtype
    ]:
        var binarized_records = self.transform[feat_dtype](dataset.records)
        return Dataset[feat_dtype, target_dtype](
            binarized_records^,
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
