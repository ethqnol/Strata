from std.math import sqrt
from ..core.matrix import Matrix
from ..utils.random import PRNG, permutation
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


def _squared_euclidean_distance[
    dtype: DType
](X: Matrix[dtype], row_x: Int, C: Matrix[dtype], row_c: Int,) -> Scalar[dtype]:
    var d = X.cols
    comptime simd_w = 4 if dtype == DType.float64 else 8
    var x_ptr = X.data.unsafe_ptr()
    var c_ptr = C.data.unsafe_ptr()
    var x_offset = row_x * d
    var c_offset = row_c * d

    var sum_simd = SIMD[dtype, simd_w](0)
    var j = 0
    while j + simd_w <= d:
        var x_simd = x_ptr.unsafe_offset(x_offset + j).unsafe_load[
            width=simd_w
        ]()
        var c_simd = c_ptr.unsafe_offset(c_offset + j).unsafe_load[
            width=simd_w
        ]()
        var diff = x_simd - c_simd
        sum_simd += diff * diff
        j += simd_w

    var sum_sq: Scalar[dtype] = sum_simd.reduce_add()
    while j < d:
        var diff = (
            x_ptr.unsafe_offset(x_offset + j).unsafe_load()
            - c_ptr.unsafe_offset(c_offset + j).unsafe_load()
        )
        sum_sq += diff * diff
        j += 1

    return sum_sq


def _euclidean_distance[
    dtype: DType
](X: Matrix[dtype], row_x: Int, C: Matrix[dtype], row_c: Int,) -> Scalar[dtype]:
    return sqrt(_squared_euclidean_distance(X, row_x, C, row_c))


def _compute_distances[
    dtype: DType
](X: Matrix[dtype], centers: Matrix[dtype]) -> Matrix[dtype]:
    var n = X.rows
    var k = centers.rows
    var dists = Matrix[dtype](n, k, 0)
    for i in range(n):
        for c in range(k):
            dists[i, c] = _euclidean_distance(X, i, centers, c)
    return dists^


def _assign_nearest_and_inertia[
    dtype: DType
](
    X: Matrix[dtype],
    centers: Matrix[dtype],
    mut labels: List[Int],
    mut min_sq_dists: List[Scalar[dtype]],
) -> Scalar[dtype]:
    var n = X.rows
    var k = centers.rows
    var total_inertia: Scalar[dtype] = 0

    if len(labels) != n:
        labels = List[Int](capacity=n)
        for _ in range(n):
            labels.append(0)

    if len(min_sq_dists) != n:
        min_sq_dists = List[Scalar[dtype]](capacity=n)
        for _ in range(n):
            min_sq_dists.append(0)

    for i in range(n):
        var best_cluster = 0
        var best_dist_sq = _squared_euclidean_distance(X, i, centers, 0)

        for c in range(1, k):
            var dist_sq = _squared_euclidean_distance(X, i, centers, c)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_cluster = c

        labels[i] = best_cluster
        min_sq_dists[i] = best_dist_sq
        total_inertia += best_dist_sq

    return total_inertia


def _init_centroids_random[
    dtype: DType
](X: Matrix[dtype], n_clusters: Int, seed: Int) raises -> Matrix[dtype]:
    var n = X.rows
    var d = X.cols

    if n_clusters > n:
        raise InvalidParameterError.error(
            "n_clusters",
            "n_clusters ("
            + String(n_clusters)
            + ") cannot exceed n_samples ("
            + String(n)
            + ")",
        )

    var perm = permutation(n, seed)
    var centers = Matrix[dtype](n_clusters, d, 0)

    for c in range(n_clusters):
        var sample_idx = perm[c]
        for j in range(d):
            centers[c, j] = X[sample_idx, j]

    return centers^


def _init_centroids_kmeans_plus_plus[
    dtype: DType
](X: Matrix[dtype], n_clusters: Int, seed: Int) raises -> Matrix[dtype]:
    var n = X.rows
    var d = X.cols

    if n_clusters > n:
        raise InvalidParameterError.error(
            "n_clusters",
            "n_clusters ("
            + String(n_clusters)
            + ") cannot exceed n_samples ("
            + String(n)
            + ")",
        )

    var rng = PRNG(seed)
    var centers = Matrix[dtype](n_clusters, d, 0)

    var first_idx = rng.next_int(n)
    for j in range(d):
        centers[0, j] = X[first_idx, j]

    if n_clusters == 1:
        return centers^

    var closest_dist_sq = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        closest_dist_sq.append(_squared_euclidean_distance(X, i, centers, 0))

    for c in range(1, n_clusters):
        var sum_sq: Float64 = 0.0
        for i in range(n):
            sum_sq += Float64(closest_dist_sq[i])

        var chosen_idx = n - 1
        if sum_sq <= 1e-12:
            chosen_idx = rng.next_int(n)
        else:
            var r = (Float64(rng.next_u64() >> 11) / Float64(1 << 53)) * sum_sq
            var cumsum: Float64 = 0.0
            for i in range(n):
                cumsum += Float64(closest_dist_sq[i])
                if cumsum >= r:
                    chosen_idx = i
                    break

        for j in range(d):
            centers[c, j] = X[chosen_idx, j]

        if c + 1 < n_clusters:
            for i in range(n):
                var dist_sq = _squared_euclidean_distance(X, i, centers, c)
                if dist_sq < closest_dist_sq[i]:
                    closest_dist_sq[i] = dist_sq

    return centers^
