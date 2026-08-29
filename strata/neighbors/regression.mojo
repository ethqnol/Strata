from ..base.estimator import Regressor
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


struct KNeighborsRegressor[compute_dtype: DType = DType.float64](
    Copyable, Movable, Regressor
):
    """Regression based on k-nearest neighbors.

    Predicts the target value for query points by local interpolation:

    - **Uniform weights**:
      $$
      \\hat{y}(x) = \\frac{1}{K} \\sum_{i \\in N_K(x)} y_i
      $$
    - **Distance weights**:
      $$
      w_i = \\frac{1}{d(x, x_i)}, \\quad \\hat{y}(x) = \\frac{\\sum_{i \\in N_K(x)} w_i y_i}{\\sum_{i \\in N_K(x)} w_i}
      $$

    Parameters:
        compute_dtype: Computational precision for distance arithmetic. Default DType.float64.

    Args:
        n_neighbors: Number of neighbors to use for prediction. Default 5.
        weights: Weight function used in prediction ('uniform', 'distance'). Default 'uniform'.
        algorithm: Neighbor search algorithm ('auto', 'brute'). Default 'auto'.
        metric: Distance metric to use ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'.
        p: Power parameter for the Minkowski metric. Default 2.0.

    Attributes:
        n_samples_fit_: Number of samples in the fitted data.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.neighbors import KNeighborsRegressor
        from strata.core import Matrix

        var reg = KNeighborsRegressor[DType.float64](n_neighbors=3, weights="distance")
        reg.fit(X_train, y_train)
        var y_pred = reg.predict(X_test)
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
    var _fit_y: List[Scalar[Self.compute_dtype]]
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
        """Initialize KNeighborsRegressor.

        Args:
            n_neighbors: Number of nearest neighbors (>= 1). Default 5.
            weights: Voting weight policy ('uniform', 'distance'). Default 'uniform'.
            algorithm: Neighbor search algorithm ('auto', 'brute'). Default 'auto'.
            metric: Distance metric identifier. Default 'euclidean'.
            p: Minkowski metric exponent (>= 1.0). Default 2.0.

        Raises:
            InvalidParameterError: If any parameter is out of valid bounds.
        """
        check_floating_dtype[Self.compute_dtype, "KNeighborsRegressor"]()
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
        self._fit_y = List[Scalar[Self.compute_dtype]]()
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing KNeighborsRegressor instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.n_neighbors = copy.n_neighbors
        self.weights = copy.weights
        self.algorithm = copy.algorithm
        self.metric = copy.metric
        self.p = copy.p

        self._nn = NearestNeighbors[Self.compute_dtype](copy=copy._nn)
        self._fit_y = copy._fit_y.copy()
        self.n_samples_fit_ = copy.n_samples_fit_
        self.n_features_in_ = copy.n_features_in_

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]],) raises:
        """Fit the k-nearest neighbors regressor from the training dataset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target continuous values vector of length n_samples.

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

        var fit_y = List[Scalar[Self.compute_dtype]](capacity=n_rows)
        for i in range(n_rows):
            fit_y.append(Scalar[Self.compute_dtype](y[i]))

        self._fit_y = fit_y^

        # Fit internal NearestNeighbors model
        self._nn.fit[feat_dtype](X)

        self.n_samples_fit_ = n_rows
        self.n_features_in_ = n_cols
        self.fit_dtype = feat_dtype
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, dataset: Dataset[feat_dtype, target_dtype]) raises:
        """Fit the k-nearest neighbors regressor using a Dataset container."""
        self.fit[feat_dtype, target_dtype](dataset.records, dataset.targets)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Scalar[feat_dtype]]:
        """Predict the continuous target values for the provided data.

        Args:
            X: Query feature matrix of shape (n_queries, n_features).

        Returns:
            List[Scalar[feat_dtype]]: Predicted regression values.

        Raises:
            NotFittedError: If estimator is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
        """
        check_is_fitted("KNeighborsRegressor", self.is_fitted)
        if feat_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "KNeighborsRegressor.predict received Matrix["
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
                "KNeighborsRegressor.predict",
            )

        var res = self._nn.kneighbors[feat_dtype](X, self.n_neighbors)
        var dists = res[0].copy()
        var idxs = res[1].copy()

        var n_queries = X.rows
        var preds = List[Scalar[feat_dtype]](capacity=n_queries)

        if self.weights == "uniform":
            var inv_k = 1.0 / Float64(self.n_neighbors)
            for q in range(n_queries):
                var sum_y: Float64 = 0.0
                for j in range(self.n_neighbors):
                    var neighbor_idx = Int(idxs[q, j])
                    sum_y += Float64(self._fit_y[neighbor_idx])
                preds.append(Scalar[feat_dtype](sum_y * inv_k))
        else:  # "distance"
            for q in range(n_queries):
                var zero_dist_count = 0
                var zero_dist_sum_y: Float64 = 0.0
                var total_weight: Float64 = 0.0
                var weighted_sum_y: Float64 = 0.0

                for j in range(self.n_neighbors):
                    var d_val = Float64(dists[q, j])
                    var neighbor_idx = Int(idxs[q, j])
                    var y_val = Float64(self._fit_y[neighbor_idx])

                    if d_val == 0.0:
                        zero_dist_count += 1
                        zero_dist_sum_y += y_val
                    elif zero_dist_count == 0:
                        var weight = 1.0 / d_val
                        weighted_sum_y += weight * y_val
                        total_weight += weight

                if zero_dist_count > 0:
                    preds.append(
                        Scalar[feat_dtype](
                            zero_dist_sum_y / Float64(zero_dist_count)
                        )
                    )
                elif total_weight > 0.0:
                    preds.append(
                        Scalar[feat_dtype](weighted_sum_y / total_weight)
                    )
                else:
                    preds.append(Scalar[feat_dtype](0.0))

        return preds^

    def predict[
        feat_dtype: DType, target_dtype: DType
    ](self, dataset: Dataset[feat_dtype, target_dtype]) raises -> List[
        Scalar[feat_dtype]
    ]:
        """Predict continuous targets for a Dataset."""
        return self.predict[feat_dtype](dataset.records)
