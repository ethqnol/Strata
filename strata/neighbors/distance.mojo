from std.math import sqrt, abs, pow
from ..core.matrix import Matrix
from ..utils.validation import check_array, check_floating_dtype
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


def sqeuclidean_distance[
    dtype_x: DType,
    dtype_y: DType = dtype_x,
](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64:
    """Compute the squared Euclidean distance between row X[row_x] and row Y[row_y].

    $$
    d^2(u, v) = \\sum_{j=1}^D (u_j - v_j)^2
    $$

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.

    Returns:
        Float64: Squared Euclidean distance.
    """
    var d = X.cols
    var sum_sq: Float64 = 0.0
    for j in range(d):
        var diff = Float64(X[row_x, j]) - Float64(Y[row_y, j])
        sum_sq += diff * diff
    return sum_sq


def euclidean_distance[
    dtype_x: DType,
    dtype_y: DType = dtype_x,
](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64:
    """Compute the Euclidean ($L_2$) distance between row X[row_x] and row Y[row_y].

    $$
    d(u, v) = \\sqrt{\\sum_{j=1}^D (u_j - v_j)^2}
    $$

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.

    Returns:
        Float64: Euclidean distance.
    """
    return sqrt(sqeuclidean_distance(X, row_x, Y, row_y))


def manhattan_distance[
    dtype_x: DType,
    dtype_y: DType = dtype_x,
](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64:
    """Compute the Manhattan ($L_1$ / taxicab / cityblock) distance between row X[row_x] and row Y[row_y].

    $$
    d(u, v) = \\sum_{j=1}^D |u_j - v_j|
    $$

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.

    Returns:
        Float64: Manhattan distance.
    """
    var d = X.cols
    var sum_abs: Float64 = 0.0
    for j in range(d):
        sum_abs += abs(Float64(X[row_x, j]) - Float64(Y[row_y, j]))
    return sum_abs


def chebyshev_distance[
    dtype_x: DType,
    dtype_y: DType = dtype_x,
](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64:
    """Compute the Chebyshev ($L_\\infty$ / max) distance between row X[row_x] and row Y[row_y].

    $$
    d(u, v) = \\max_{1 \\le j \\le D} |u_j - v_j|
    $$

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.

    Returns:
        Float64: Chebyshev distance.
    """
    var d = X.cols
    var max_val: Float64 = 0.0
    for j in range(d):
        var diff = abs(Float64(X[row_x, j]) - Float64(Y[row_y, j]))
        if diff > max_val:
            max_val = diff
    return max_val


def minkowski_distance[
    dtype_x: DType,
    dtype_y: DType = dtype_x,
](
    X: Matrix[dtype_x],
    row_x: Int,
    Y: Matrix[dtype_y],
    row_y: Int,
    p: Float64 = 2.0,
) raises -> Float64:
    """Compute the Minkowski ($L_p$) distance between row X[row_x] and row Y[row_y].

    $$
    d(u, v) = \\left( \\sum_{j=1}^D |u_j - v_j|^p \\right)^{1/p}
    $$

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.
        p: Minkowski norm order ($p \\ge 1.0$). Default 2.0.

    Returns:
        Float64: Minkowski distance.

    Raises:
        InvalidParameterError: If p < 1.0.
    """
    if p < 1.0:
        raise InvalidParameterError.error(
            "p", "minkowski distance requires p >= 1.0, got " + String(p)
        )
    if p == 1.0:
        return manhattan_distance(X, row_x, Y, row_y)
    if p == 2.0:
        return euclidean_distance(X, row_x, Y, row_y)

    var d = X.cols
    var sum_pow: Float64 = 0.0
    for j in range(d):
        var diff = abs(Float64(X[row_x, j]) - Float64(Y[row_y, j]))
        sum_pow += pow(diff, p)
    return pow(sum_pow, 1.0 / p)


def cosine_distance[
    dtype_x: DType,
    dtype_y: DType = dtype_x,
](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64:
    """Compute the Cosine distance between row X[row_x] and row Y[row_y].

    $$
    d(u, v) = 1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}
    $$

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.

    Returns:
        Float64: Cosine distance in range $[0, 2]$.
    """
    var d = X.cols
    var dot_val: Float64 = 0.0
    var norm_x_sq: Float64 = 0.0
    var norm_y_sq: Float64 = 0.0

    for j in range(d):
        var vx = Float64(X[row_x, j])
        var vy = Float64(Y[row_y, j])
        dot_val += vx * vy
        norm_x_sq += vx * vx
        norm_y_sq += vy * vy

    if norm_x_sq <= 0.0 and norm_y_sq <= 0.0:
        return 0.0
    if norm_x_sq <= 0.0 or norm_y_sq <= 0.0:
        return 1.0

    var sim = dot_val / (sqrt(norm_x_sq) * sqrt(norm_y_sq))
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return 1.0 - sim



def _validate_metric_and_p(metric: String, p: Float64) raises:
    if (
        metric != "euclidean"
        and metric != "l2"
        and metric != "sqeuclidean"
        and metric != "manhattan"
        and metric != "cityblock"
        and metric != "l1"
        and metric != "chebyshev"
        and metric != "infinity"
        and metric != "max"
        and metric != "minkowski"
        and metric != "cosine"
    ):
        raise InvalidParameterError.error(
            "metric",
            "unrecognized distance metric '"
            + metric
            + "'. Supported: 'euclidean', 'sqeuclidean', 'manhattan',"
            " 'chebyshev', 'minkowski', 'cosine'",
        )
    if metric == "minkowski" and p < 1.0:
        raise InvalidParameterError.error(
            "p", "minkowski distance requires p >= 1.0, got " + String(p)
        )


def row_distance[
    dtype: DType
](
    X: Matrix[dtype],
    row_x: Int,
    Y: Matrix[dtype],
    row_y: Int,
    metric: String = "euclidean",
    p: Float64 = 2.0,
) raises -> Scalar[dtype]:
    """Compute pairwise distance between sample row X[row_x] and sample row Y[row_y] under metric.

    Supported metrics:
    - `"euclidean"` or `"l2"`
    - `"sqeuclidean"`
    - `"manhattan"`, `"cityblock"`, or `"l1"`
    - `"chebyshev"`, `"infinity"`, or `"max"`
    - `"minkowski"` (with order parameter `p >= 1.0`)
    - `"cosine"`

    Args:
        X: First matrix container.
        row_x: Row index in matrix X.
        Y: Second matrix container.
        row_y: Row index in matrix Y.
        metric: Distance metric identifier. Default 'euclidean'.
        p: Minkowski order parameter used when metric='minkowski'. Default 2.0.

    Returns:
        Scalar[dtype]: Calculated distance scalar.

    Raises:
        InvalidParameterError: If metric is unknown or p < 1.0 for minkowski.
    """
    _validate_metric_and_p(metric, p)

    if metric == "euclidean" or metric == "l2":
        return Scalar[dtype](euclidean_distance(X, row_x, Y, row_y))
    elif metric == "sqeuclidean":
        return Scalar[dtype](sqeuclidean_distance(X, row_x, Y, row_y))
    elif (
        metric == "manhattan"
        or metric == "cityblock"
        or metric == "l1"
    ):
        return Scalar[dtype](manhattan_distance(X, row_x, Y, row_y))
    elif (
        metric == "chebyshev"
        or metric == "infinity"
        or metric == "max"
    ):
        return Scalar[dtype](chebyshev_distance(X, row_x, Y, row_y))
    elif metric == "minkowski":
        return Scalar[dtype](minkowski_distance(X, row_x, Y, row_y, p))
    else:  # "cosine"
        return Scalar[dtype](cosine_distance(X, row_x, Y, row_y))


def pairwise_distances[
    dtype: DType
](
    X: Matrix[dtype],
    Y: Matrix[dtype],
    metric: String = "euclidean",
    p: Float64 = 2.0,
) raises -> Matrix[dtype]:
    """Compute the full pairwise distance matrix between rows of X and rows of Y.

    Args:
        X: First feature matrix of shape $(N_X, D)$.
        Y: Second feature matrix of shape $(N_Y, D)$.
        metric: Distance metric to compute. Default 'euclidean'.
        p: Minkowski order when metric='minkowski'. Default 2.0.

    Returns:
        Matrix[dtype]: Pairwise distance matrix of shape $(N_X, N_Y)$.

    Raises:
        DimensionMismatchError: If X and Y have different number of columns or are empty.
        InvalidParameterError: If inputs contain NaN/Inf or metric/p are invalid.
    """
    check_floating_dtype[dtype, "pairwise_distances"]()
    _validate_metric_and_p(metric, p)
    check_array[dtype](X)
    check_array[dtype](Y)

    if X.cols != Y.cols:
        raise DimensionMismatchError.error(
            "X.cols == " + String(X.cols),
            "Y.cols == " + String(Y.cols),
            "pairwise_distances",
        )

    var nx = X.rows
    var ny = Y.rows
    var D = Matrix[dtype](nx, ny, 0)

    if metric == "euclidean" or metric == "l2":
        for i in range(nx):
            for j in range(ny):
                D[i, j] = Scalar[dtype](euclidean_distance(X, i, Y, j))
    elif metric == "sqeuclidean":
        for i in range(nx):
            for j in range(ny):
                D[i, j] = Scalar[dtype](sqeuclidean_distance(X, i, Y, j))
    elif (
        metric == "manhattan"
        or metric == "cityblock"
        or metric == "l1"
    ):
        for i in range(nx):
            for j in range(ny):
                D[i, j] = Scalar[dtype](manhattan_distance(X, i, Y, j))
    elif (
        metric == "chebyshev"
        or metric == "infinity"
        or metric == "max"
    ):
        for i in range(nx):
            for j in range(ny):
                D[i, j] = Scalar[dtype](chebyshev_distance(X, i, Y, j))
    elif metric == "minkowski":
        for i in range(nx):
            for j in range(ny):
                D[i, j] = Scalar[dtype](
                    minkowski_distance(X, i, Y, j, p)
                )
    elif metric == "cosine":
        for i in range(nx):
            for j in range(ny):
                D[i, j] = Scalar[dtype](cosine_distance(X, i, Y, j))

    return D^


def pairwise_distances[
    dtype: DType
](
    X: Matrix[dtype],
    metric: String = "euclidean",
    p: Float64 = 2.0,
) raises -> Matrix[dtype]:
    """Compute the self-pairwise distance matrix between all pairs of rows in X.

    Exploits symmetry $D_{i, j} = D_{j, i}$ and $D_{i, i} = 0$ for symmetric distance metrics.

    Args:
        X: Feature matrix of shape $(N, D)$.
        metric: Distance metric to compute. Default 'euclidean'.
        p: Minkowski order when metric='minkowski'. Default 2.0.

    Returns:
        Matrix[dtype]: Symmetric distance matrix of shape $(N, N)$.

    Raises:
        DimensionMismatchError: If X is empty.
        InvalidParameterError: If X contains NaN/Inf or metric/p are invalid.
    """
    check_floating_dtype[dtype, "pairwise_distances"]()
    _validate_metric_and_p(metric, p)
    check_array[dtype](X)

    var n = X.rows
    var D = Matrix[dtype](n, n, 0)

    if metric == "euclidean" or metric == "l2":
        for i in range(n):
            D[i, i] = Scalar[dtype](0)
            for j in range(i + 1, n):
                var dist = Scalar[dtype](euclidean_distance(X, i, X, j))
                D[i, j] = dist
                D[j, i] = dist
    elif metric == "sqeuclidean":
        for i in range(n):
            D[i, i] = Scalar[dtype](0)
            for j in range(i + 1, n):
                var dist = Scalar[dtype](sqeuclidean_distance(X, i, X, j))
                D[i, j] = dist
                D[j, i] = dist
    elif (
        metric == "manhattan"
        or metric == "cityblock"
        or metric == "l1"
    ):
        for i in range(n):
            D[i, i] = Scalar[dtype](0)
            for j in range(i + 1, n):
                var dist = Scalar[dtype](manhattan_distance(X, i, X, j))
                D[i, j] = dist
                D[j, i] = dist
    elif (
        metric == "chebyshev"
        or metric == "infinity"
        or metric == "max"
    ):
        for i in range(n):
            D[i, i] = Scalar[dtype](0)
            for j in range(i + 1, n):
                var dist = Scalar[dtype](chebyshev_distance(X, i, X, j))
                D[i, j] = dist
                D[j, i] = dist
    elif metric == "minkowski":
        for i in range(n):
            D[i, i] = Scalar[dtype](0)
            for j in range(i + 1, n):
                var dist = Scalar[dtype](
                    minkowski_distance(X, i, X, j, p)
                )
                D[i, j] = dist
                D[j, i] = dist
    elif metric == "cosine":
        for i in range(n):
            D[i, i] = Scalar[dtype](0)
            for j in range(i + 1, n):
                var dist = Scalar[dtype](cosine_distance(X, i, X, j))
                D[i, j] = dist
                D[j, i] = dist

    return D^
