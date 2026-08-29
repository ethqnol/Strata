from std.math import sqrt, abs
from ..core.matrix import Matrix
from ..utils.validation import check_array, check_floating_dtype
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from .distance import (
    euclidean_distance,
    sqeuclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    _validate_metric_and_p,
)


struct KDNode(Copyable, Movable):
    """Contiguous node in a flat KD-Tree buffer."""

    var point_idx: Int
    var split_dim: Int
    var split_val: Float64
    var left_child: Int
    var right_child: Int

    def __init__(
        out self,
        point_idx: Int,
        split_dim: Int,
        split_val: Float64,
        left_child: Int = -1,
        right_child: Int = -1,
    ):
        self.point_idx = point_idx
        self.split_dim = split_dim
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child

    def __init__(out self, *, copy: Self):
        self.point_idx = copy.point_idx
        self.split_dim = copy.split_dim
        self.split_val = copy.split_val
        self.left_child = copy.left_child
        self.right_child = copy.right_child


struct _AxisIndexPair(Comparable, Copyable, Movable):
    var val: Float64
    var idx: Int

    def __init__(out self, val: Float64, idx: Int):
        self.val = val
        self.idx = idx

    def __init__(out self, *, copy: Self):
        self.val = copy.val
        self.idx = copy.idx

    def __lt__(self, other: Self) -> Bool:
        if self.val == other.val:
            return self.idx < other.idx
        return self.val < other.val

    def __le__(self, other: Self) -> Bool:
        if self.val == other.val:
            return self.idx <= other.idx
        return self.val <= other.val

    def __gt__(self, other: Self) -> Bool:
        if self.val == other.val:
            return self.idx > other.idx
        return self.val > other.val

    def __ge__(self, other: Self) -> Bool:
        if self.val == other.val:
            return self.idx >= other.idx
        return self.val >= other.val

    def __eq__(self, other: Self) -> Bool:
        return self.val == other.val and self.idx == other.idx

    def __ne__(self, other: Self) -> Bool:
        return self.val != other.val or self.idx != other.idx


struct KDTree[compute_dtype: DType = DType.float64](Copyable, Movable):
    """Fast spatial index for nearest neighbor and radius queries in low dimensions.

    Organizes $N$ points in $D$-dimensional space into a binary space-partitioning
    tree for $O(K \\log N)$ neighbor lookups.

    Parameters:
        compute_dtype: Precision for spatial coordinate representation. Default DType.float64.

    Args:
        data: Matrix of training points with shape (n_samples, n_features).
        metric: Distance metric to use ('euclidean', 'manhattan', 'chebyshev'). Default 'euclidean'.

    Attributes:
        n_samples_: Number of samples indexed in the tree.
        n_features_: Dimensionality of the indexed space.
        root_idx_: Index of the root node in the internal buffer.

    Examples:
        ```mojo
        from strata.neighbors import KDTree
        from strata.core import Matrix

        var tree = KDTree[DType.float64](X_train, metric="euclidean")
        var res = tree.query(X_query, k=3)
        var dists = res[0]
        var idxs = res[1]
        ```
    """

    var data_: Matrix[Self.compute_dtype]
    var nodes: List[KDNode]
    var root_idx_: Int
    var n_samples_: Int
    var n_features_: Int
    var metric: String

    def __init__(
        out self,
        data: Matrix[Self.compute_dtype],
        metric: String = "euclidean",
    ) raises:
        """Construct a KDTree from a matrix of spatial coordinates.

        Args:
            data: Input point matrix of shape (n_samples, n_features).
            metric: Distance metric ('euclidean', 'manhattan', 'chebyshev'). Default 'euclidean'.

        Raises:
            InvalidParameterError: If data has NaNs or unsupported metric.
            DimensionMismatchError: If data is empty.
        """
        check_floating_dtype[Self.compute_dtype, "KDTree"]()
        check_array[Self.compute_dtype](data)
        if (
            metric != "euclidean"
            and metric != "l2"
            and metric != "manhattan"
            and metric != "cityblock"
            and metric != "l1"
            and metric != "chebyshev"
        ):
            raise InvalidParameterError.error(
                "metric",
                "KDTree supports 'euclidean', 'manhattan', or 'chebyshev',"
                " got '"
                + metric
                + "'",
            )

        self.data_ = data.copy()
        self.nodes = List[KDNode]()
        self.root_idx_ = -1
        self.n_samples_ = data.rows
        self.n_features_ = data.cols
        self.metric = metric

        var sample_indices = List[Int](capacity=self.n_samples_)
        for i in range(self.n_samples_):
            sample_indices.append(i)

        self.root_idx_ = self._build(sample_indices, 0, self.n_samples_, 0)

    def __init__(out self, *, copy: Self):
        """Copies an existing KDTree."""
        self.data_ = copy.data_.copy()
        self.nodes = copy.nodes.copy()
        self.root_idx_ = copy.root_idx_
        self.n_samples_ = copy.n_samples_
        self.n_features_ = copy.n_features_
        self.metric = copy.metric

    def _build(
        mut self,
        mut indices: List[Int],
        start: Int,
        end: Int,
        depth: Int,
    ) -> Int:
        var count = end - start
        if count <= 0:
            return -1

        var axis = depth % self.n_features_

        # Collect and sort subset range [start, end)
        var pairs = List[_AxisIndexPair](capacity=count)
        for i in range(start, end):
            var s_idx = indices[i]
            var coord = Float64(self.data_[s_idx, axis])
            pairs.append(_AxisIndexPair(coord, s_idx))

        sort(pairs)

        for i in range(count):
            indices[start + i] = pairs[i].idx

        var mid = start + count // 2
        var mid_idx = indices[mid]
        var split_val = Float64(self.data_[mid_idx, axis])

        var left_child = self._build(indices, start, mid, depth + 1)
        var right_child = self._build(indices, mid + 1, end, depth + 1)

        var node_idx = len(self.nodes)
        self.nodes.append(
            KDNode(mid_idx, axis, split_val, left_child, right_child)
        )
        return node_idx

    def _point_dist[
        in_dtype: DType
    ](self, X_query: Matrix[in_dtype], q_row: Int, sample_idx: Int,) -> Float64:
        if (
            self.metric == "manhattan"
            or self.metric == "cityblock"
            or self.metric == "l1"
        ):
            return manhattan_distance(X_query, q_row, self.data_, sample_idx)
        elif self.metric == "chebyshev":
            return chebyshev_distance(X_query, q_row, self.data_, sample_idx)
        else:  # euclidean
            return euclidean_distance(X_query, q_row, self.data_, sample_idx)

    def _search_knn[
        in_dtype: DType
    ](
        self,
        node_idx: Int,
        X_query: Matrix[in_dtype],
        q_row: Int,
        k: Int,
        mut best_dists: List[Float64],
        mut best_indices: List[Int],
    ):
        if node_idx < 0:
            return

        var node = self.nodes[node_idx].copy()
        var d = self._point_dist[in_dtype](X_query, q_row, node.point_idx)

        # Insert into sorted bounded top-k buffer
        if len(best_dists) < k or d < best_dists[len(best_dists) - 1]:
            var ins_pos = 0
            while ins_pos < len(best_dists) and best_dists[ins_pos] <= d:
                ins_pos += 1

            best_dists.insert(ins_pos, d)
            best_indices.insert(ins_pos, node.point_idx)

            if len(best_dists) > k:
                _ = best_dists.pop()
                _ = best_indices.pop()

        var q_coord = Float64(X_query[q_row, node.split_dim])
        var plane_dist = abs(q_coord - node.split_val)

        var near_child = (
            node.left_child if q_coord < node.split_val else node.right_child
        )
        var far_child = (
            node.right_child if q_coord < node.split_val else node.left_child
        )

        self._search_knn[in_dtype](
            near_child, X_query, q_row, k, best_dists, best_indices
        )

        # Prune search: only explore opposite side if plane distance is within current top-k boundary
        var current_max_dist = (
            best_dists[len(best_dists) - 1] if len(best_dists) == k else 1e308
        )
        if plane_dist < current_max_dist:
            self._search_knn[in_dtype](
                far_child, X_query, q_row, k, best_dists, best_indices
            )

    def _search_radius[
        in_dtype: DType
    ](
        self,
        node_idx: Int,
        X_query: Matrix[in_dtype],
        q_row: Int,
        r: Float64,
        mut res_dists: List[Float64],
        mut res_indices: List[Int],
    ):
        if node_idx < 0:
            return

        var node = self.nodes[node_idx].copy()
        var d = self._point_dist[in_dtype](X_query, q_row, node.point_idx)

        if d <= r:
            res_dists.append(d)
            res_indices.append(node.point_idx)

        var q_coord = Float64(X_query[q_row, node.split_dim])
        var plane_dist = abs(q_coord - node.split_val)

        var near_child = (
            node.left_child if q_coord < node.split_val else node.right_child
        )
        var far_child = (
            node.right_child if q_coord < node.split_val else node.left_child
        )

        self._search_radius[in_dtype](
            near_child, X_query, q_row, r, res_dists, res_indices
        )

        if plane_dist <= r:
            self._search_radius[in_dtype](
                far_child, X_query, q_row, r, res_dists, res_indices
            )

    def query[
        in_dtype: DType
    ](
        self,
        X: Matrix[in_dtype],
        k: Int = 1,
    ) raises -> Tuple[
        Matrix[in_dtype], Matrix[DType.int32]
    ]:
        """Query the KDTree for the k-nearest neighbors of points in X.

        Args:
            X: Query point matrix with shape (n_queries, n_features).
            k: Number of nearest neighbors to return per query. Default 1.

        Returns:
            Tuple of:
            - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, k).
            - Matrix[DType.int32]: Indices of neighbors in original data with shape (n_queries, k).

        Raises:
            DimensionMismatchError: If X.cols != n_features_ or X is empty.
            InvalidParameterError: If k < 1 or k > n_samples_.
        """
        check_array[in_dtype](X)
        if X.cols != self.n_features_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_),
                "X.cols == " + String(X.cols),
                "KDTree.query",
            )
        if k < 1 or k > self.n_samples_:
            raise InvalidParameterError.error(
                "k",
                "expected 1 <= k <= "
                + String(self.n_samples_)
                + ", got "
                + String(k),
            )

        var n_queries = X.rows
        var dist_mat = Matrix[in_dtype](n_queries, k, 0)
        var idx_mat = Matrix[DType.int32](n_queries, k, 0)

        for q in range(n_queries):
            var best_dists = List[Float64](capacity=k)
            var best_indices = List[Int](capacity=k)

            self._search_knn[in_dtype](
                self.root_idx_, X, q, k, best_dists, best_indices
            )

            var n_found = len(best_dists)
            for j in range(n_found):
                dist_mat[q, j] = Scalar[in_dtype](best_dists[j])
                idx_mat[q, j] = Scalar[DType.int32](best_indices[j])

        return dist_mat^, idx_mat^

    def query_radius[
        in_dtype: DType
    ](
        self,
        X: Matrix[in_dtype],
        r: Float64,
    ) raises -> Tuple[
        List[List[Scalar[in_dtype]]], List[List[Int]]
    ]:
        """Find all points within distance r of points in X.

        Args:
            X: Query point matrix with shape (n_queries, n_features).
            r: Distance radius (must be > 0.0).

        Returns:
            Tuple of:
            - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius.
            - List[List[Int]]: Dataset indices of neighbors within radius.

        Raises:
            DimensionMismatchError: If X.cols != n_features_.
            InvalidParameterError: If r <= 0.0.
        """
        check_array[in_dtype](X)
        if X.cols != self.n_features_:
            raise DimensionMismatchError.error(
                "X.cols == " + String(self.n_features_),
                "X.cols == " + String(X.cols),
                "KDTree.query_radius",
            )
        if r <= 0.0:
            raise InvalidParameterError.error(
                "r", "expected radius r > 0.0, got " + String(r)
            )

        var n_queries = X.rows
        var all_dists = List[List[Scalar[in_dtype]]](capacity=n_queries)
        var all_indices = List[List[Int]](capacity=n_queries)

        for q in range(n_queries):
            var res_dists = List[Float64]()
            var res_indices = List[Int]()

            self._search_radius[in_dtype](
                self.root_idx_, X, q, r, res_dists, res_indices
            )

            # Sort by distance
            var pairs = List[_AxisIndexPair](capacity=len(res_dists))
            for j in range(len(res_dists)):
                pairs.append(_AxisIndexPair(res_dists[j], res_indices[j]))

            sort(pairs)

            var q_dists = List[Scalar[in_dtype]](capacity=len(pairs))
            var q_idxs = List[Int](capacity=len(pairs))

            for j in range(len(pairs)):
                q_dists.append(Scalar[in_dtype](pairs[j].val))
                q_idxs.append(pairs[j].idx)

            all_dists.append(q_dists^)
            all_indices.append(q_idxs^)

        return all_dists^, all_indices^
