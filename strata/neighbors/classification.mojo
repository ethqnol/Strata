from ..base.estimator import Classifier
from ..core.matrix import Matrix
from ..core.dataset import Dataset
from ..utils.validation import (
    check_array,
    check_floating_dtype,
    check_is_fitted,
    check_consistent_length,
)
from ..exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
    NotFittedError,
    DataConversionError,
)
from .distance import _validate_metric_and_p
from .base import NearestNeighbors


struct KNeighborsClassifier[compute_dtype: DType = DType.float64](
    Classifier, Copyable, Movable
):
    """Classifier implementing the k-nearest neighbors vote.

    Supports uniform voting and inverse-distance weighted voting:

    - **Uniform weights**:
      $$
      P(y = c \\mid x) = \\frac{1}{K} \\sum_{i \\in N_K(x)} \\mathbb{I}(y_i = c)
      $$
    - **Distance weights**:
      $$
      w_i = \\frac{1}{d(x, x_i)}, \\quad P(y = c \\mid x) = \\frac{\\sum_{i \\in N_K(x)} w_i \\mathbb{I}(y_i = c)}{\\sum_{i \\in N_K(x)} w_i}
      $$

    Parameters:
        compute_dtype: Precision used for internal distance calculations. Default DType.float64.

    Args:
        n_neighbors: Number of neighbors to use for queries. Default 5.
        weights: Weight function used in prediction ('uniform', 'distance'). Default 'uniform'.
        algorithm: Algorithm used to compute nearest neighbors ('auto', 'brute'). Default 'auto'.
        metric: Distance metric to use. Default 'euclidean'.
        p: Power parameter for the Minkowski metric. Default 2.0.

    Attributes:
        classes_: Distinct class labels matrix of shape (n_classes,).
        n_classes_: Number of distinct classes seen during fit.
        n_samples_fit_: Number of samples in the fitted data.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.neighbors import KNeighborsClassifier
        from strata.core import Matrix

        var clf = KNeighborsClassifier[DType.float64](n_neighbors=3, weights="distance")
        clf.fit(X_train, y_train)
        var y_pred = clf.predict(X_test)
        var proba = clf.predict_proba(X_test)
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var n_neighbors: Int
    var weights: String
    var algorithm: String
    var metric: String
    var p: Float64

    var _nn: NearestNeighbors[Self.compute_dtype]
    var _fit_y: List[Int]
    var classes_: List[Int]
    var n_classes_: Int
    var n_samples_fit_: Int
    var n_features_in_: Int

    def __init__(
        out self,
        n_neighbors: Int = 5,
        weights: String = "uniform",
        algorithm: String = "auto",
        metric: String = "euclidean",
        p: Float64 = 2.0,
    ) raises:
        """Initialize KNeighborsClassifier.

        Args:
            n_neighbors: Number of nearest neighbors (>= 1). Default 5.
            weights: Voting weight policy ('uniform', 'distance'). Default 'uniform'.
            algorithm: Neighbor search algorithm ('auto', 'brute'). Default 'auto'.
            metric: Distance metric identifier. Default 'euclidean'.
            p: Minkowski metric exponent (>= 1.0). Default 2.0.

        Raises:
            InvalidParameterError: If any parameter is out of valid bounds.
        """
        check_floating_dtype[Self.compute_dtype, "KNeighborsClassifier"]()
        if n_neighbors < 1:
            raise InvalidParameterError.error(
                "n_neighbors",
                "expected n_neighbors >= 1, got " + String(n_neighbors),
            )
        if weights != "uniform" and weights != "distance":
            raise InvalidParameterError.error(
                "weights",
                "expected weights 'uniform' or 'distance', got '"
                + weights
                + "'",
            )
        _validate_metric_and_p(metric, p)

        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p

        self._nn = NearestNeighbors[Self.compute_dtype](
            n_neighbors=n_neighbors,
            radius=1.0,
            algorithm=algorithm,
            metric=metric,
            p=p,
        )
        self._fit_y = List[Int]()
        self.classes_ = List[Int]()
        self.n_classes_ = 0
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing KNeighborsClassifier instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.n_neighbors = copy.n_neighbors
        self.weights = copy.weights
        self.algorithm = copy.algorithm
        self.metric = copy.metric
        self.p = copy.p

        self._nn = NearestNeighbors[Self.compute_dtype](copy=copy._nn)
        self._fit_y = copy._fit_y.copy()
        self.classes_ = copy.classes_.copy()
        self.n_classes_ = copy.n_classes_
        self.n_samples_fit_ = copy.n_samples_fit_
        self.n_features_in_ = copy.n_features_in_

    def _class_index(self, label: Int) -> Int:
        for idx in range(self.n_classes_):
            if self.classes_[idx] == label:
                return idx
        return 0

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](
        mut self,
        X: Matrix[feat_dtype],
        y: List[Scalar[target_dtype]],
    ) raises:
        """Fit the k-nearest neighbors classifier from the training dataset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target class labels vector of length n_samples.

        Raises:
            DimensionMismatchError: If len(y) != X.rows or X is empty.
            InvalidParameterError: If X contains NaN/Inf or n_samples < n_neighbors.
        """
        check_array[feat_dtype](X)
        check_consistent_length(X, y)

        if X.rows < self.n_neighbors:
            raise InvalidParameterError.error(
                "n_neighbors",
                "n_neighbors="
                + String(self.n_neighbors)
                + " cannot be greater than number of samples="
                + String(X.rows),
            )

        var n_rows = X.rows
        var n_cols = X.cols

        # Discover unique classes
        var distinct = List[Int]()
        var fit_y = List[Int](capacity=n_rows)

        for i in range(n_rows):
            var val_int = Int(Float64(y[i]))
            fit_y.append(val_int)
            var found = False
            for c in range(len(distinct)):
                if distinct[c] == val_int:
                    found = True
                    break
            if not found:
                distinct.append(val_int)

        sort(distinct)

        self._fit_y = fit_y^
        self.classes_ = distinct^
        self.n_classes_ = len(self.classes_)

        # Fit internal NearestNeighbors model
        self._nn.fit[feat_dtype](X)

        self.n_samples_fit_ = n_rows
        self.n_features_in_ = n_cols
        self.fit_dtype = feat_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fit the k-nearest neighbors classifier using a Dataset container."""
        self.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Return probability estimates for the test data X.

        Args:
            X: Query feature matrix of shape (n_queries, n_features).

        Returns:
            Matrix[feat_dtype]: Probabilities of shape (n_queries, n_classes).

        Raises:
            NotFittedError: If estimator is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
        """
        check_is_fitted("KNeighborsClassifier", self.is_fitted)
        if feat_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "KNeighborsClassifier.predict_proba received Matrix["
                + String(feat_dtype)
                + "] but was fitted on Matrix["
                + String(self.fit_dtype)
                + "]"
            )
        check_array[feat_dtype](X)
        if X.cols != self.n_features_in_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_in_),
                "X.cols == " + String(X.cols),
                "KNeighborsClassifier.predict_proba",
            )

        var res = self._nn.kneighbors[feat_dtype](X, self.n_neighbors)
        var dists = res[0].copy()
        var idxs = res[1].copy()

        var n_queries = X.rows
        var proba = Matrix[feat_dtype](n_queries, self.n_classes_, 0)

        if self.weights == "uniform":
            var inv_k = Scalar[feat_dtype](1.0 / Float64(self.n_neighbors))
            for q in range(n_queries):
                for j in range(self.n_neighbors):
                    var neighbor_idx = Int(idxs[q, j])
                    var target_class = self._fit_y[neighbor_idx]
                    var c_idx = self._class_index(target_class)
                    proba[q, c_idx] += inv_k
        else:  # "distance"
            for q in range(n_queries):
                var has_zero_dist = False
                var zero_dist_class = 0

                for j in range(self.n_neighbors):
                    if Float64(dists[q, j]) == 0.0:
                        has_zero_dist = True
                        var neighbor_idx = Int(idxs[q, j])
                        zero_dist_class = self._fit_y[neighbor_idx]
                        break

                if has_zero_dist:
                    var c_idx = self._class_index(zero_dist_class)
                    proba[q, c_idx] = Scalar[feat_dtype](1.0)
                else:
                    var total_weight: Float64 = 0.0
                    for j in range(self.n_neighbors):
                        var d_val = Float64(dists[q, j])
                        var weight = 1.0 / d_val
                        var neighbor_idx = Int(idxs[q, j])
                        var target_class = self._fit_y[neighbor_idx]
                        var c_idx = self._class_index(target_class)
                        proba[q, c_idx] += Scalar[feat_dtype](weight)
                        total_weight += weight

                    if total_weight > 0.0:
                        for c in range(self.n_classes_):
                            proba[q, c] /= Scalar[feat_dtype](total_weight)

        return proba^

    def predict_proba[
        feat_dtype: DType, target_dtype: DType
    ](
        self, dataset: Dataset[feat_dtype, target_dtype]
    ) raises -> Matrix[feat_dtype]:
        """Return probability estimates for a Dataset."""
        return self.predict_proba[feat_dtype](dataset.records)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict the class labels for the provided data.

        Args:
            X: Query feature matrix of shape (n_queries, n_features).

        Returns:
            List[Int]: Predicted class label for each sample.

        Raises:
            NotFittedError: If estimator is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
        """
        var proba = self.predict_proba[feat_dtype](X)
        var n_queries = X.rows
        var preds = List[Int](capacity=n_queries)

        for q in range(n_queries):
            var best_class_idx = 0
            var best_prob = proba[q, 0]

            for c in range(1, self.n_classes_):
                if proba[q, c] > best_prob:
                    best_prob = proba[q, c]
                    best_class_idx = c

            preds.append(self.classes_[best_class_idx])

        return preds^

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[Int]:
        """Predict class labels for a Dataset."""
        return self.predict[feat_dtype](dataset.records)
