from ..core.matrix import Matrix
from ..core.dataset import Dataset
from .estimator import Estimator, Transformer, Regressor, Classifier
from ..utils.validation import check_array, check_is_fitted
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct PipelineTransformer[*Transformers: Transformer](
    Copyable, Movable, Transformer
):
    """Chains an arbitrary variadic sequence of data transformers into a single composite transformer.

    Transforms data sequentially across steps:

    $$
    X \\to T_1(X) \\to T_2(T_1(X)) \\to \\dots \\to T_N(\\dots)
    $$

    Parameters:
        Transformers: Variadic pack of Transformer types.

    Attributes:
        transformers: Variadic tuple of transformer instances.
        is_fitted: Boolean flag indicating if transformer steps are fitted.

    Examples:
        ```mojo
        from strata import StandardScaler, RobustScaler, PCA, PipelineTransformer

        var pipe = PipelineTransformer(
            (StandardScaler(), RobustScaler(), PCA(n_components=2))
        )
        ```
    """

    var transformers: Tuple[*Self.Transformers]
    var is_fitted: Bool

    def __init__(out self, transformers: Tuple[*Self.Transformers]):
        """Initializes a PipelineTransformer with a variadic tuple of transformer steps.

        Args:
            transformers: Tuple of transformer instances satisfying the Transformer trait.
        """
        self.transformers = transformers.copy()
        self.is_fitted = False

    def __init__(out self, *, copy: Self):
        """Creates a deep copy of an existing PipelineTransformer."""
        self.transformers = copy.transformers.copy()
        self.is_fitted = copy.is_fitted

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fits all transformers sequentially.

        Args:
            X: Input feature matrix of shape $(N, D)$.
        """
        check_array[in_dtype](X)
        var cur = X.copy()
        comptime for i in range(len(Self.Transformers)):
            var trans = self.transformers[i].copy()
            cur = trans.fit_transform[in_dtype](cur)
            self.transformers[i] = trans^
        self.is_fitted = True

    def transform[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Sequentially applies transformations across all pipeline steps.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Matrix[in_dtype]: Transformed output matrix.

        Raises:
            NotFittedError: If the pipeline has not been fitted yet.
        """
        check_is_fitted("PipelineTransformer", self.is_fitted)
        check_array[in_dtype](X)
        var cur = X.copy()
        comptime for i in range(len(Self.Transformers)):
            cur = self.transformers[i].transform[in_dtype](cur)
        return cur^

    def fit_transform[
        in_dtype: DType
    ](mut self, X: Matrix[in_dtype]) raises -> Matrix[in_dtype]:
        """Fits all transformers and returns the final transformed output representation.

        Args:
            X: Input feature matrix of shape $(N, D)$.

        Returns:
            Matrix[in_dtype]: Final transformed output matrix.
        """
        check_array[in_dtype](X)
        var cur = X.copy()
        comptime for i in range(len(Self.Transformers)):
            var trans = self.transformers[i].copy()
            cur = trans.fit_transform[in_dtype](cur)
            self.transformers[i] = trans^
        self.is_fitted = True
        return cur^


@fieldwise_init
struct PipelineRegressor[
    T: Transformer,
    R: Regressor,
    target_dtype: DType = DType.float64,
](Copyable, Movable, Regressor):
    """Sequentially applies a transformer pipeline before fitting a regressor.

    Parameters:
        T: Type of the feature transformer step.
        R: Type of the regressor step.
        target_dtype: Data type of target values. Default DType.float64.

    Attributes:
        transformer: Transformer step instance.
        regressor: Regressor step instance.
    """

    var transformer: Self.T
    var regressor: Self.R

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fits the transformer on X, then fits the regressor on transformed features.

        Args:
            X: Feature matrix of shape $(N, D)$.
            y: Target values of length $N$.
        """
        var X_trans = self.transformer.fit_transform[feat_dtype](X)
        self.regressor.fit[feat_dtype, in_target_dtype](X_trans, y)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Transforms input features and computes regression predictions.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            List[Scalar[feat_dtype]]: Predicted target values.
        """
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.regressor.predict[feat_dtype](X_trans)


@fieldwise_init
struct PipelineClassifier[
    T: Transformer,
    C: Classifier,
    target_dtype: DType = DType.int32,
](Classifier, Copyable, Movable):
    """Sequentially applies a transformer pipeline before fitting a classifier.

    Parameters:
        T: Type of the feature transformer step.
        C: Type of the classifier step.
        target_dtype: Data type of target labels. Default DType.int32.

    Attributes:
        transformer: Transformer step instance.
        classifier: Classifier step instance.
    """

    var transformer: Self.T
    var classifier: Self.C

    def fit[
        feat_dtype: DType, in_target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]]) raises:
        """Fits the transformer on X, then fits the classifier on transformed features.

        Args:
            X: Feature matrix of shape $(N, D)$.
            y: Target class labels of length $N$.
        """
        var X_trans = self.transformer.fit_transform[feat_dtype](X)
        self.classifier.fit[feat_dtype, in_target_dtype](X_trans, y)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Transforms input features and predicts discrete class labels.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            List[Int]: Predicted discrete class labels.
        """
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.classifier.predict[feat_dtype](X_trans)

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Transforms input features and predicts class probability distributions.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Probability distribution matrix of shape $(N, K)$.
        """
        var X_trans = self.transformer.transform[feat_dtype](X)
        return self.classifier.predict_proba[feat_dtype](X_trans)
