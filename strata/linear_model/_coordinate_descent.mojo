from std.math import abs, max
from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


def _soft_threshold[
    dtype: DType = DType.float64
](z: Scalar[dtype], gamma: Scalar[dtype]) -> Scalar[dtype]:
    """Applies the soft-thresholding operator: sign(z) * max(0, |z| - gamma).

    Parameters:
        dtype: Numerical data type. Default DType.float64.

    Args:
        z: Input scalar value.
        gamma: Threshold penalty parameter (gamma >= 0).

    Returns:
        Scalar[dtype]: Soft-thresholded output.
    """
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0


def _coordinate_descent_elastic_net[
    dtype: DType = DType.float64
](
    X: Matrix[dtype],
    y: List[Scalar[dtype]],
    alpha: Scalar[dtype],
    l1_ratio: Scalar[dtype],
    max_iter: Int,
    tol: Scalar[dtype],
    positive: Bool = False,
) raises -> Tuple[List[Scalar[dtype]], Int, Scalar[dtype]]:
    """Optimizes the ElasticNet / Lasso objective via cyclic coordinate descent.

    Minimizes the penalized objective:

    $$
    \\min_{w} \\frac{1}{2N} \\|y - Xw\\|_2^2 + \\alpha \\cdot \\text{l1\\_ratio} \\|w\\|_1 + \\frac{\\alpha (1 - \\text{l1\\_ratio})}{2} \\|w\\|_2^2
    $$

    Parameters:
        dtype: Computational data type. Default DType.float64.

    Args:
        X: Centered or raw feature matrix of shape $(N, D)$.
        y: Centered or raw target vector of length $N$.
        alpha: Overall regularization multiplier (alpha >= 0).
        l1_ratio: ElasticNet mixing parameter in [0, 1]. (1.0 for Lasso, 0.0 for Ridge).
        max_iter: Maximum number of coordinate descent iterations over all features.
        tol: Convergence tolerance on max absolute coefficient update.
        positive: When set to True, forces coefficients to be non-negative.

    Returns:
        Tuple[List[Scalar[dtype]], Int, Scalar[dtype]]:
            - Fitted coefficient vector w of length D.
            - Total number of iterations executed.
            - Final maximum coordinate change at termination.
    """
    var N = X.rows
    var D = X.cols

    if N == 0 or D == 0:
        return (List[Scalar[dtype]](), 0, Scalar[dtype](0))

    if len(y) != N:
        raise DimensionMismatchError.error(
            "len(y) == " + String(N),
            "len(y) == " + String(len(y)),
            "_coordinate_descent_elastic_net",
        )

    var l1_penalty = alpha * l1_ratio
    var l2_penalty = alpha * (1.0 - l1_ratio)
    var inv_N = Scalar[dtype](1.0) / Scalar[dtype](N)

    # Transpose X to D x N for column-contiguous memory access
    var Xt = X.transpose()
    var xt_ptr = Xt.data.unsafe_ptr()

    # Precalculate column squared L2 norms normalized by N
    var c_norms = List[Scalar[dtype]](capacity=D)
    comptime simd_width = 4 if dtype == DType.float64 else 8

    for j in range(D):
        var j_offset = j * N
        var sum_sq_simd = SIMD[dtype, simd_width](0)
        var i = 0
        while i + simd_width <= N:
            var val_simd = xt_ptr.unsafe_offset(j_offset + i).unsafe_load[
                width=simd_width
            ]()
            sum_sq_simd = val_simd.fma(val_simd, sum_sq_simd)
            i += simd_width

        var sum_sq: Scalar[dtype] = sum_sq_simd.reduce_add()
        while i < N:
            var v = Xt.data[j_offset + i]
            sum_sq += v * v
            i += 1

        c_norms.append(sum_sq * inv_N)

    var w = List[Scalar[dtype]](capacity=D)
    for _ in range(D):
        w.append(0)

    var r = y.copy()
    var r_ptr = r.unsafe_ptr()

    var n_iter = 0
    var max_d_w: Scalar[dtype] = 0

    for it in range(max_iter):
        n_iter = it + 1
        max_d_w = 0

        for j in range(D):
            var c_j = c_norms[j]
            if c_j == 0:
                continue

            var j_offset = j * N
            var dot_simd = SIMD[dtype, simd_width](0)
            var i = 0
            while i + simd_width <= N:
                var x_simd = xt_ptr.unsafe_offset(j_offset + i).unsafe_load[
                    width=simd_width
                ]()
                var r_simd = r_ptr.unsafe_offset(i).unsafe_load[
                    width=simd_width
                ]()
                dot_simd = x_simd.fma(r_simd, dot_simd)
                i += simd_width

            var dot_val: Scalar[dtype] = dot_simd.reduce_add()
            while i < N:
                dot_val += Xt.data[j_offset + i] * r[i]
                i += 1

            var rho_j = dot_val * inv_N + c_j * w[j]

            var w_j_new: Scalar[dtype]
            if positive:
                if l1_penalty == 0:
                    w_j_new = max(Scalar[dtype](0), rho_j) / (c_j + l2_penalty)
                else:
                    w_j_new = max(Scalar[dtype](0), rho_j - l1_penalty) / (
                        c_j + l2_penalty
                    )
            else:
                if l1_penalty == 0:
                    w_j_new = rho_j / (c_j + l2_penalty)
                else:
                    w_j_new = _soft_threshold[dtype](rho_j, l1_penalty) / (
                        c_j + l2_penalty
                    )

            var d_w = w_j_new - w[j]
            if d_w != 0:
                # Update residual: r = r - d_w * X_j
                var neg_d_w_simd = SIMD[dtype, simd_width](-d_w)
                var k = 0
                while k + simd_width <= N:
                    var x_chunk = xt_ptr.unsafe_offset(
                        j_offset + k
                    ).unsafe_load[width=simd_width]()
                    var r_chunk = r_ptr.unsafe_offset(k).unsafe_load[
                        width=simd_width
                    ]()
                    var r_updated = neg_d_w_simd.fma(x_chunk, r_chunk)
                    r_ptr.unsafe_offset(k).unsafe_store[width=simd_width](
                        r_updated
                    )
                    k += simd_width

                while k < N:
                    r[k] -= d_w * Xt.data[j_offset + k]
                    k += 1

                w[j] = w_j_new
                var abs_d_w = abs(d_w)
                if abs_d_w > max_d_w:
                    max_d_w = abs_d_w

        if max_d_w < tol:
            break

    return (w^, n_iter, max_d_w)
