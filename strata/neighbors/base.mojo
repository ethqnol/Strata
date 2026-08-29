from ..core.matrix import Matrix
from ..utils.validation import (
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..exceptions.errors import (
    InvalidParameterError,
    DimensionMismatchError,
    NotFittedError,
    DataConversionError,
)
from .distance import (
    euclidean_distance,
    sqeuclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
    cosine_distance,
    _validate_metric_and_p,
)


struct NeighborDistIdx(Comparable, Copyable, Movable):
    """Container holding a sample distance and its training dataset row index.
    """

    var dist: Float64
    var idx: Int

    def __init__(out self, dist: Float64, idx: Int):
        self.dist = dist
        self.idx = idx

    def __init__(out self, *, copy: Self):
        self.dist = copy.dist
        self.idx = copy.idx

    def __lt__(self, other: Self) -> Bool:
        if self.dist == other.dist:
            return self.idx < other.idx
        return self.dist < other.dist

    def __le__(self, other: Self) -> Bool:
        if self.dist == other.dist:
            return self.idx <= other.idx
        return self.dist <= other.dist

    def __gt__(self, other: Self) -> Bool:
        if self.dist == other.dist:
            return self.idx > other.idx
        return self.dist > other.dist

    def __ge__(self, other: Self) -> Bool:
        if self.dist == other.dist:
            return self.idx >= other.idx
        return self.dist >= other.dist

    def __eq__(self, other: Self) -> Bool:
        return self.dist == other.dist and self.idx == other.idx

    def __ne__(self, other: Self) -> Bool:
        return self.dist != other.dist or self.idx != other.idx


struct NearestNeighbors[compute_dtype: DType = DType.float64](
    Copyable, Movable
):
    """Unsupervised learner for implementing neighbor searches.

    Finds the $k$-nearest neighbors or all neighbors within a given radius
    using brute-force or index-backed spatial distance metrics.

    Parameters:
        compute_dtype: Precision used for distance computations. Default DType.float64.

    Args:
        n_neighbors: Number of neighbors to use by default for `kneighbors` queries. Default 5.
        radius: Range of parameter space to use by default for `radius_neighbors` queries. Default 1.0.
        algorithm: Algorithm used to compute nearest neighbors ('auto', 'brute'). Default 'auto'.
        metric: Distance metric to use ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'.
        p: Parameter for the Minkowski metric. Default 2.0.

    Attributes:
        n_samples_fit_: Number of samples in the fitted data.
        n_features_in_: Number of features seen during fit.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.neighbors import NearestNeighbors
        from strata.core import Matrix

        var nn = NearestNeighbors[DType.float64](n_neighbors=2)
        nn.fit(X_train)
        var res = nn.kneighbors(X_test)
        var distances = res[0]
        var indices = res[1]
        ```
    """

    var is_fitted: Bool
    var fit_dtype: DType
    var n_neighbors: Int
    var radius: Float64
    var algorithm: String
    var metric: String
    var p: Float64
    var _fit_X: Matrix[Self.compute_dtype]
    var n_samples_fit_: Int
    var n_features_in_: Int

    def __init__(
        out self,
        n_neighbors: Int = 5,
        radius: Float64 = 1.0,
        algorithm: String = "auto",
        metric: String = "euclidean",
        p: Float64 = 2.0,
    ) raises:
        """Initialize NearestNeighbors estimator.

        Args:
            n_neighbors: Number of nearest neighbors to query (>= 1). Default 5.
            radius: Spatial neighborhood radius (> 0.0). Default 1.0.
            algorithm: Neighbor search algorithm ('auto', 'brute'). Default 'auto'.
            metric: Distance metric name. Default 'euclidean'.
            p: Minkowski metric exponent (>= 1.0). Default 2.0.

        Raises:
            InvalidParameterError: If any parameter is out of valid bounds.
        """
        check_floating_dtype[Self.compute_dtype, "NearestNeighbors"]()
        if n_neighbors < 1:
            raise InvalidParameterError.error(
                "n_neighbors",
                "expected n_neighbors >= 1, got " + String(n_neighbors),
            )
        if radius <= 0.0:
            raise InvalidParameterError.error(
                "radius", "expected radius > 0.0, got " + String(radius)
            )
        if algorithm != "auto" and algorithm != "brute":
            raise InvalidParameterError.error(
                "algorithm",
                "expected 'auto' or 'brute', got '" + algorithm + "'",
            )
        _validate_metric_and_p(metric, p)

        self.is_fitted = False
        self.fit_dtype = DType.float64
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self._fit_X = Matrix[Self.compute_dtype](0, 0, 0)
        self.n_samples_fit_ = 0
        self.n_features_in_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing NearestNeighbors instance."""
        self.is_fitted = copy.is_fitted
        self.fit_dtype = copy.fit_dtype
        self.n_neighbors = copy.n_neighbors
        self.radius = copy.radius
        self.algorithm = copy.algorithm
        self.metric = copy.metric
        self.p = copy.p
        self._fit_X = copy._fit_X.copy()
        self.n_samples_fit_ = copy.n_samples_fit_
        self.n_features_in_ = copy.n_features_in_

    def fit[in_dtype: DType](mut self, X: Matrix[in_dtype]) raises:
        """Fit the nearest neighbors estimator from the training dataset.

        Args:
            X: Training data matrix with shape (n_samples, n_features).

        Raises:
            DimensionMismatchError: If X is empty.
            InvalidParameterError: If X contains NaN or infinity, or n_samples < n_neighbors.
        """
        check_array[in_dtype](X)
        if X.rows < self.n_neighbors:
            raise InvalidParameterError.error(
                "n_neighbors",
                "n_neighbors="
                + String(self.n_neighbors)
                + " cannot be greater than the number of samples in X="
                + String(X.rows),
            )

        var n_rows = X.rows
        var n_cols = X.cols
        self._fit_X = Matrix[Self.compute_dtype](n_rows, n_cols, 0)
        for r in range(n_rows):
            for c in range(n_cols):
                self._fit_X[r, c] = Scalar[Self.compute_dtype](X[r, c])

        self.n_samples_fit_ = n_rows
        self.n_features_in_ = n_cols
        self.fit_dtype = in_dtype
        self.is_fitted = True

    def _query_point[
        in_dtype: DType
    ](
        self,
        X_query: Matrix[in_dtype],
        query_row: Int,
    ) raises -> List[
        NeighborDistIdx
    ]:
        var pairs = List[NeighborDistIdx](capacity=self.n_samples_fit_)

        if self.metric == "euclidean" or self.metric == "l2":
            for j in range(self.n_samples_fit_):
                var d = euclidean_distance(X_query, query_row, self._fit_X, j)
                pairs.append(NeighborDistIdx(d, j))
        elif self.metric == "sqeuclidean":
            for j in range(self.n_samples_fit_):
                var d = sqeuclidean_distance(X_query, query_row, self._fit_X, j)
                pairs.append(NeighborDistIdx(d, j))
        elif (
            self.metric == "manhattan"
            or self.metric == "cityblock"
            or self.metric == "l1"
        ):
            for j in range(self.n_samples_fit_):
                var d = manhattan_distance(X_query, query_row, self._fit_X, j)
                pairs.append(NeighborDistIdx(d, j))
        elif (
            self.metric == "chebyshev"
            or self.metric == "infinity"
            or self.metric == "max"
        ):
            for j in range(self.n_samples_fit_):
                var d = chebyshev_distance(X_query, query_row, self._fit_X, j)
                pairs.append(NeighborDistIdx(d, j))
        elif self.metric == "minkowski":
            for j in range(self.n_samples_fit_):
                var d = minkowski_distance(
                    X_query, query_row, self._fit_X, j, self.p
                )
                pairs.append(NeighborDistIdx(d, j))
        elif self.metric == "cosine":
            for j in range(self.n_samples_fit_):
                var d = cosine_distance(X_query, query_row, self._fit_X, j)
                pairs.append(NeighborDistIdx(d, j))

        sort(pairs)
        return pairs^

    def kneighbors[
        in_dtype: DType
    ](
        self,
        X: Matrix[in_dtype],
        n_neighbors: Int = -1,
    ) raises -> Tuple[
        Matrix[in_dtype], Matrix[DType.int32]
    ]:
        """Find the K-neighbors of points in X.

        Args:
            X: Query points matrix with shape (n_queries, n_features).
            n_neighbors: Number of neighbors to return per query. If -1, uses estimator default.

        Returns:
            Tuple of:
            - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, n_neighbors).
            - Matrix[DType.int32]: Indices of neighbors in the training dataset with shape (n_queries, n_neighbors).

        Raises:
            NotFittedError: If the estimator is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_ or X is empty.
            InvalidParameterError: If n_neighbors > n_samples_fit_ or n_neighbors < 1.
        """
        check_is_fitted("NearestNeighbors", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "NearestNeighbors.kneighbors received Matrix["
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
                "NearestNeighbors.kneighbors",
            )

        var k = self.n_neighbors if n_neighbors == -1 else n_neighbors
        if k < 1 or k > self.n_samples_fit_:
            raise InvalidParameterError.error(
                "n_neighbors",
                "expected 1 <= n_neighbors <= "
                + String(self.n_samples_fit_)
                + ", got "
                + String(k),
            )

        var n_queries = X.rows
        var dist_mat = Matrix[in_dtype](n_queries, k, 0)
        var idx_mat = Matrix[DType.int32](n_queries, k, 0)

        for q in range(n_queries):
            var sorted_pairs = self._query_point[in_dtype](X, q)
            for j in range(k):
                dist_mat[q, j] = Scalar[in_dtype](sorted_pairs[j].dist)
                idx_mat[q, j] = Scalar[DType.int32](sorted_pairs[j].idx)

        return dist_mat^, idx_mat^

    def kneighbors[
        in_dtype: DType = DType.float64
    ](
        self,
        n_neighbors: Int = -1,
    ) raises -> Tuple[
        Matrix[in_dtype], Matrix[DType.int32]
    ]:
        """Find the K-neighbors of the training data itself.

        Args:
            n_neighbors: Number of neighbors to return. Default is estimator n_neighbors.

        Returns:
            Tuple of distance matrix and index matrix.
        """
        check_is_fitted("NearestNeighbors", self.is_fitted)
        var X_mat = Matrix[in_dtype](
            self.n_samples_fit_, self.n_features_in_, 0
        )
        for r in range(self.n_samples_fit_):
            for c in range(self.n_features_in_):
                X_mat[r, c] = Scalar[in_dtype](self._fit_X[r, c])
        return self.kneighbors[in_dtype](X_mat, n_neighbors)

    def radius_neighbors[
        in_dtype: DType
    ](
        self,
        X: Matrix[in_dtype],
        radius: Float64 = -1.0,
    ) raises -> Tuple[
        List[List[Scalar[in_dtype]]], List[List[Int]]
    ]:
        """Find the neighbors within a given radius of points in X.

        Args:
            X: Query points matrix with shape (n_queries, n_features).
            radius: Neighborhood radius. If negative, uses estimator radius.

        Returns:
            Tuple of:
            - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius.
            - List[List[Int]]: Indices of neighbors in the training dataset.

        Raises:
            NotFittedError: If the estimator is not fitted.
            DataConversionError: If in_dtype does not match fit_dtype.
            DimensionMismatchError: If X.cols != n_features_in_.
            InvalidParameterError: If radius <= 0.0.
        """
        check_is_fitted("NearestNeighbors", self.is_fitted)
        if in_dtype != self.fit_dtype:
            raise DataConversionError.error(
                "NearestNeighbors.radius_neighbors received Matrix["
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
                "NearestNeighbors.radius_neighbors",
            )

        if radius <= 0.0 and radius != -1.0:
            raise InvalidParameterError.error(
                "radius", "expected radius > 0.0, got " + String(radius)
            )

        var r_val = self.radius if radius == -1.0 else radius
        var n_queries = X.rows

        var all_dists = List[List[Scalar[in_dtype]]](capacity=n_queries)
        var all_indices = List[List[Int]](capacity=n_queries)

        for q in range(n_queries):
            var sorted_pairs = self._query_point[in_dtype](X, q)
            var q_dists = List[Scalar[in_dtype]]()
            var q_idxs = List[Int]()

            for j in range(len(sorted_pairs)):
                if sorted_pairs[j].dist <= r_val:
                    q_dists.append(Scalar[in_dtype](sorted_pairs[j].dist))
                    q_idxs.append(sorted_pairs[j].idx)

            all_dists.append(q_dists^)
            all_indices.append(q_idxs^)

        return all_dists^, all_indices^
