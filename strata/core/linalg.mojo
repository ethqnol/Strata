from std.ffi import external_call, c_int, c_char, c_double, c_float
from std.math import sqrt
from .matrix import Matrix
from ..exceptions.errors import DimensionMismatchError, InvalidParameterError


struct SVDResult[dtype: DType = DType.float64](Copyable, Movable):
    """Result of Singular Value Decomposition ($A = U \\Sigma V^T$).

    Attributes:
        U: Left singular vectors matrix of shape $(M, K)$.
        S: Singular values vector of length $K$ in descending order.
        Vt: Right singular vectors transposed matrix of shape $(K, N)$.
    """

    var U: Matrix[Self.dtype]
    var S: List[Scalar[Self.dtype]]
    var Vt: Matrix[Self.dtype]

    def __init__(
        out self,
        var U: Matrix[Self.dtype],
        var S: List[Scalar[Self.dtype]],
        var Vt: Matrix[Self.dtype],
    ):
        """Initialize an SVDResult container."""
        self.U = U^
        self.S = S^
        self.Vt = Vt^


struct QRResult[dtype: DType = DType.float64](Copyable, Movable):
    """Result of QR Decomposition ($A = Q R$).

    Attributes:
        Q: Orthogonal matrix of shape $(M, K)$.
        R: Upper triangular matrix of shape $(K, N)$.
    """

    var Q: Matrix[Self.dtype]
    var R: Matrix[Self.dtype]

    def __init__(
        out self, var Q: Matrix[Self.dtype], var R: Matrix[Self.dtype]
    ):
        """Initialize a QRResult container."""
        self.Q = Q^
        self.R = R^


struct EigResult[dtype: DType = DType.float64](Copyable, Movable):
    """Result of Symmetric Eigenvalue Decomposition ($A V = V \\Lambda$).

    Attributes:
        eigenvalues: Real eigenvalues vector of length $N$ in ascending order.
        eigenvectors: Eigenvector matrix of shape $(N, N)$ with columns representing eigenvectors.
    """

    var eigenvalues: List[Scalar[Self.dtype]]
    var eigenvectors: Matrix[Self.dtype]

    def __init__(
        out self,
        var eigenvalues: List[Scalar[Self.dtype]],
        var eigenvectors: Matrix[Self.dtype],
    ):
        """Initialize an EigResult container."""
        self.eigenvalues = eigenvalues^
        self.eigenvectors = eigenvectors^


def gemm[
    dtype: DType = DType.float64
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[dtype]:
    """Compute dense matrix product $C = A B$.

    Hardware-vectorized with SIMD registers and scalar tail handling, supporting
    arbitrary matrix dimensions and numeric types with zero external dependencies.

    Args:
        A: Left matrix of shape $(M, K)$.
        B: Right matrix of shape $(K, N)$.

    Returns:
        Matrix[dtype]: Output matrix product $C$ of shape $(M, N)$.

    Raises:
        DimensionMismatchError: If `A.cols != B.rows`.
    """

    if A.cols != B.rows:
        raise DimensionMismatchError.error(
            "A.cols == B.rows",
            "A("
            + String(A.rows)
            + "x"
            + String(A.cols)
            + ") @ B("
            + String(B.rows)
            + "x"
            + String(B.cols)
            + ")",
            "gemm",
        )

    var M = A.rows
    var K = A.cols
    var N = B.cols

    var C = Matrix[dtype].zeros(M, N)

    comptime if dtype == DType.float64:
        var layout: c_int = 101  # CblasRowMajor
        var transa: c_int = 111  # CblasNoTrans
        var transb: c_int = 111  # CblasNoTrans
        var a_ptr = Pointer[Scalar[dtype], origin_of(A.data)](
            A.data.unsafe_ptr()
        )
        var b_ptr = Pointer[Scalar[dtype], origin_of(B.data)](
            B.data.unsafe_ptr()
        )
        var c_ptr = Pointer[Scalar[dtype], origin_of(C.data)](
            C.data.unsafe_ptr()
        )
        external_call[
            "cblas_dgemm",
            NoneType,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_double,
            Pointer[Scalar[dtype], origin_of(A.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(B.data)],
            c_int,
            c_double,
            Pointer[Scalar[dtype], origin_of(C.data)],
            c_int,
        ](
            layout,
            transa,
            transb,
            c_int(M),
            c_int(N),
            c_int(K),
            c_double(1.0),
            a_ptr,
            c_int(K),
            b_ptr,
            c_int(N),
            c_double(0.0),
            c_ptr,
            c_int(N),
        )
    elif dtype == DType.float32:
        var layout: c_int = 101  # CblasRowMajor
        var transa: c_int = 111  # CblasNoTrans
        var transb: c_int = 111  # CblasNoTrans
        var a_ptr = Pointer[Scalar[dtype], origin_of(A.data)](
            A.data.unsafe_ptr()
        )
        var b_ptr = Pointer[Scalar[dtype], origin_of(B.data)](
            B.data.unsafe_ptr()
        )
        var c_ptr = Pointer[Scalar[dtype], origin_of(C.data)](
            C.data.unsafe_ptr()
        )
        external_call[
            "cblas_sgemm",
            NoneType,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_float,
            Pointer[Scalar[dtype], origin_of(A.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(B.data)],
            c_int,
            c_float,
            Pointer[Scalar[dtype], origin_of(C.data)],
            c_int,
        ](
            layout,
            transa,
            transb,
            c_int(M),
            c_int(N),
            c_int(K),
            c_float(1.0),
            a_ptr,
            c_int(K),
            b_ptr,
            c_int(N),
            c_float(0.0),
            c_ptr,
            c_int(N),
        )
    else:
        comptime simd_width = 4 if dtype == DType.float64 else 8

        var b_ptr = B.data.unsafe_ptr()
        var c_ptr = C.data.unsafe_ptr()

        var M_B = 64
        var K_B = 64
        var N_B = 64

        for i0 in range(0, M, M_B):
            var i_max = min(i0 + M_B, M)
            for k0 in range(0, K, K_B):
                var k_max = min(k0 + K_B, K)
                for j0 in range(0, N, N_B):
                    var j_max = min(j0 + N_B, N)

                    for i in range(i0, i_max):
                        var c_offset = i * N
                        for k in range(k0, k_max):
                            var a_ik = A[i, k]
                            if a_ik == 0:
                                continue
                            var b_offset = k * N

                            var j = j0
                            while j + simd_width <= j_max:
                                var b_simd = b_ptr.unsafe_offset(
                                    b_offset + j
                                ).unsafe_load[width=simd_width]()
                                var c_simd = c_ptr.unsafe_offset(
                                    c_offset + j
                                ).unsafe_load[width=simd_width]()
                                var res_simd = c_simd + a_ik * b_simd
                                c_ptr.unsafe_offset(c_offset + j).unsafe_store(
                                    res_simd
                                )
                                j += simd_width

                            while j < j_max:
                                var cur = c_ptr.unsafe_offset(
                                    c_offset + j
                                ).unsafe_load()
                                c_ptr.unsafe_offset(c_offset + j).unsafe_store(
                                    cur
                                    + a_ik
                                    * b_ptr.unsafe_offset(
                                        b_offset + j
                                    ).unsafe_load()
                                )
                                j += 1

    return C^


def dense_dot_vec[
    dtype: DType = DType.float64
](
    A: Matrix[dtype],
    x: List[Scalar[dtype]],
    bias: Scalar[dtype] = 0,
) raises -> List[Scalar[dtype]]:
    """Dense matrix-vector product: y = A @ x + bias.

    Hardware-vectorized with SIMD registers, dual-row register tiling, and horizontal reduction.
    """
    if A.cols != len(x):
        raise DimensionMismatchError.error(
            "len(x) == " + String(A.cols),
            "len(x) == " + String(len(x)),
            "dense_dot_vec",
        )

    var res = List[Scalar[dtype]](capacity=A.rows)
    comptime simd_width = 4 if dtype == DType.float64 else 8

    var a_ptr = A.data.unsafe_ptr()
    var x_ptr = x.unsafe_ptr()
    var n_cols = A.cols
    var n_rows = A.rows

    for r in range(0, n_rows - 1, 2):
        var r0_offset = r * n_cols
        var r1_offset = (r + 1) * n_cols
        var sum0 = SIMD[dtype, simd_width](0)
        var sum1 = SIMD[dtype, simd_width](0)
        var c = 0
        while c + simd_width <= n_cols:
            var x_simd = x_ptr.unsafe_offset(c).unsafe_load[width=simd_width]()
            var a0_simd = a_ptr.unsafe_offset(r0_offset + c).unsafe_load[
                width=simd_width
            ]()
            var a1_simd = a_ptr.unsafe_offset(r1_offset + c).unsafe_load[
                width=simd_width
            ]()
            sum0 += a0_simd * x_simd
            sum1 += a1_simd * x_simd
            c += simd_width

        var row_sum0: Scalar[dtype] = sum0.reduce_add()
        var row_sum1: Scalar[dtype] = sum1.reduce_add()
        while c < n_cols:
            var x_val = x_ptr.unsafe_offset(c).unsafe_load()
            row_sum0 += a_ptr.unsafe_offset(r0_offset + c).unsafe_load() * x_val
            row_sum1 += a_ptr.unsafe_offset(r1_offset + c).unsafe_load() * x_val
            c += 1

        res.append(row_sum0 + bias)
        res.append(row_sum1 + bias)

    if n_rows % 2 == 1:
        var r = n_rows - 1
        var row_offset = r * n_cols
        var sum_simd = SIMD[dtype, simd_width](0)
        var c = 0
        while c + simd_width <= n_cols:
            var a_simd = a_ptr.unsafe_offset(row_offset + c).unsafe_load[
                width=simd_width
            ]()
            var x_simd = x_ptr.unsafe_offset(c).unsafe_load[width=simd_width]()
            sum_simd += a_simd * x_simd
            c += simd_width

        var row_sum: Scalar[dtype] = sum_simd.reduce_add()
        while c < n_cols:
            row_sum += (
                a_ptr.unsafe_offset(row_offset + c).unsafe_load()
                * x_ptr.unsafe_offset(c).unsafe_load()
            )
            c += 1

        res.append(row_sum + bias)

    return res^


def svd[
    dtype: DType = DType.float64
](A: Matrix[dtype], full_matrices: Bool = False) raises -> SVDResult[dtype]:
    """Computes the Singular Value Decomposition (SVD) of a matrix A = U * S * Vt.

    Uses LAPACK's divide-and-conquer algorithm (dgesdd/sgesdd).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for SVD decomposition"

    var M = A.rows
    var N = A.cols
    var K = min(M, N)
    var u_cols = M if full_matrices else K
    var vt_rows = N if full_matrices else K
    var jobz = c_char(ord("A")) if full_matrices else c_char(ord("S"))

    var A_copy = A.copy()
    var S = List[Scalar[dtype]](capacity=K)
    for _ in range(K):
        S.append(0)

    var U = Matrix[dtype].zeros(M, u_cols)
    var Vt = Matrix[dtype].zeros(vt_rows, N)

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dgesdd",
            c_int,
            c_int,
            c_char,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(S)],
            Pointer[Scalar[dtype], origin_of(U.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(Vt.data)],
            c_int,
        ](
            c_int(101),
            jobz,
            c_int(M),
            c_int(N),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            S.unsafe_ptr(),
            U.data.unsafe_ptr(),
            c_int(u_cols),
            Vt.data.unsafe_ptr(),
            c_int(N),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sgesdd",
            c_int,
            c_int,
            c_char,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(S)],
            Pointer[Scalar[dtype], origin_of(U.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(Vt.data)],
            c_int,
        ](
            c_int(101),
            jobz,
            c_int(M),
            c_int(N),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            S.unsafe_ptr(),
            U.data.unsafe_ptr(),
            c_int(u_cols),
            Vt.data.unsafe_ptr(),
            c_int(N),
        )

    if info != 0:
        raise InvalidParameterError.error(
            "LAPACK SVD did not converge (info=" + String(info) + ")",
            "svd",
        )

    return SVDResult[dtype](U^, S^, Vt^)


def qr[
    dtype: DType = DType.float64
](A: Matrix[dtype]) raises -> QRResult[dtype]:
    """Computes the QR Decomposition of matrix A = Q * R using LAPACK Householder reflectors (dgeqrf/dorgqr).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for QR decomposition"

    var M = A.rows
    var N = A.cols
    var K = min(M, N)

    var A_qr = A.copy()
    var tau = List[Scalar[dtype]](capacity=K)
    for _ in range(K):
        tau.append(0)

    var info_geqrf: c_int = 0

    comptime if dtype == DType.float64:
        info_geqrf = external_call[
            "LAPACKE_dgeqrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_qr.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            A_qr.data.unsafe_ptr(),
            c_int(N),
            tau.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info_geqrf = external_call[
            "LAPACKE_sgeqrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_qr.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            A_qr.data.unsafe_ptr(),
            c_int(N),
            tau.unsafe_ptr(),
        )

    if info_geqrf != 0:
        raise InvalidParameterError.error(
            "LAPACK QR factorization failed (info=" + String(info_geqrf) + ")",
            "qr",
        )

    # Extract upper triangular R (K x N)
    var R = Matrix[dtype].zeros(K, N)
    for r in range(K):
        for c in range(r, N):
            R[r, c] = A_qr[r, c]

    # Generate orthogonal matrix Q (M x K)
    var Q = Matrix[dtype].zeros(M, K)
    for r in range(M):
        for c in range(min(r + 1, K)):
            Q[r, c] = A_qr[r, c]

    var info_orgqr: c_int = 0

    comptime if dtype == DType.float64:
        info_orgqr = external_call[
            "LAPACKE_dorgqr",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(Q.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(K),
            c_int(K),
            Q.data.unsafe_ptr(),
            c_int(K),
            tau.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info_orgqr = external_call[
            "LAPACKE_sorgqr",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(Q.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(tau)],
        ](
            c_int(101),
            c_int(M),
            c_int(K),
            c_int(K),
            Q.data.unsafe_ptr(),
            c_int(K),
            tau.unsafe_ptr(),
        )

    if info_orgqr != 0:
        raise InvalidParameterError.error(
            "LAPACK Q matrix reconstruction failed (info="
            + String(info_orgqr)
            + ")",
            "qr",
        )

    return QRResult[dtype](Q^, R^)


def cholesky[
    dtype: DType = DType.float64
](A: Matrix[dtype], lower: Bool = True) raises -> Matrix[dtype]:
    """Computes the Cholesky factorization of a symmetric positive-definite matrix A = L * L^T.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for Cholesky factorization"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix (A.rows == A.cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "cholesky",
        )

    var N = A.rows
    var L = A.copy()
    var uplo_char = c_char(ord("L")) if lower else c_char(ord("U"))

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dpotrf",
            c_int,
            c_int,
            c_char,
            c_int,
            Pointer[Scalar[dtype], origin_of(L.data)],
            c_int,
        ](
            c_int(101),
            uplo_char,
            c_int(N),
            L.data.unsafe_ptr(),
            c_int(N),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_spotrf",
            c_int,
            c_int,
            c_char,
            c_int,
            Pointer[Scalar[dtype], origin_of(L.data)],
            c_int,
        ](
            c_int(101),
            uplo_char,
            c_int(N),
            L.data.unsafe_ptr(),
            c_int(N),
        )

    if info > 0:
        raise InvalidParameterError.error(
            "Matrix A is not positive-definite (leading minor "
            + String(info)
            + " is not positive)",
            "cholesky",
        )
    elif info < 0:
        raise InvalidParameterError.error(
            "Illegal argument in LAPACK cholesky (info=" + String(info) + ")",
            "cholesky",
        )

    # Zero out non-triangular elements
    if lower:
        for r in range(N):
            for c in range(r + 1, N):
                L[r, c] = 0
    else:
        for r in range(N):
            for c in range(r):
                L[r, c] = 0

    return L^


def lstsq[
    dtype: DType = DType.float64
](
    A: Matrix[dtype],
    b: List[Scalar[dtype]],
    rcond: Float64 = -1.0,
) raises -> List[Scalar[dtype]]:
    """Solve linear least-squares problem $\\min_x \\|A x - b\\|_2$ using SVD.

    Uses LAPACK `dgelss`/`sgelss` to compute the minimum-norm least-squares solution.


    Args:
        A: Coefficient matrix of shape $(M, N)$.
        b: Right-hand side vector of length $M$.
        rcond: Cutoff for small singular values. Default -1.0 (machine precision).

    Returns:
        List[Scalar[dtype]]: Least-squares solution vector $x$ of length $N$.

    Raises:
        DimensionMismatchError: If `A.rows != len(b)`.
    """

    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for least squares"

    if A.rows != len(b):
        raise DimensionMismatchError.error(
            "A.rows == len(b)",
            "A.rows=" + String(A.rows) + ", len(b)=" + String(len(b)),
            "lstsq",
        )

    var M = A.rows
    var N = A.cols
    var K = min(M, N)
    var max_mn = max(M, N)

    var A_copy = A.copy()
    # b_buf must have size max(M, N) allocated and zeroed out for underdetermined systems
    var b_buf = List[Scalar[dtype]](capacity=max_mn)
    for i in range(M):
        b_buf.append(b[i])
    for _ in range(M, max_mn):
        b_buf.append(0)

    var s = List[Scalar[dtype]](capacity=K)
    for _ in range(K):
        s.append(0)

    var rank_buf = List[c_int](capacity=1)
    rank_buf.append(0)

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dgelss",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(b_buf)],
            c_int,
            Pointer[Scalar[dtype], origin_of(s)],
            c_double,
            Pointer[c_int, origin_of(rank_buf)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            b_buf.unsafe_ptr(),
            c_int(1),  # ldb = 1 for 1 column RHS in row-major layout
            s.unsafe_ptr(),
            rcond,
            rank_buf.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sgelss",
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(b_buf)],
            c_int,
            Pointer[Scalar[dtype], origin_of(s)],
            c_float,
            Pointer[c_int, origin_of(rank_buf)],
        ](
            c_int(101),
            c_int(M),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            b_buf.unsafe_ptr(),
            c_int(1),
            s.unsafe_ptr(),
            Float32(rcond),
            rank_buf.unsafe_ptr(),
        )

    if info != 0:
        raise InvalidParameterError.error(
            "LAPACK least squares solver failed (info=" + String(info) + ")",
            "lstsq",
        )

    var x = List[Scalar[dtype]](capacity=N)
    for i in range(N):
        x.append(b_buf[i])

    return x^


def solve[
    dtype: DType = DType.float64
](A: Matrix[dtype], b: List[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """Solves a square linear system A * x = b using LU decomposition (dgesv/sgesv).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for linear solve"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix A (rows == cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "solve",
        )
    if A.rows != len(b):
        raise DimensionMismatchError.error(
            "A.rows == len(b)",
            "A.rows=" + String(A.rows) + ", len(b)=" + String(len(b)),
            "solve",
        )

    var N = A.rows
    var A_copy = A.copy()
    var b_copy = b.copy()
    var ipiv = List[c_int](capacity=N)
    for _ in range(N):
        ipiv.append(0)

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dgesv",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
            Pointer[Scalar[dtype], origin_of(b_copy)],
            c_int,
        ](
            c_int(101),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
            b_copy.unsafe_ptr(),
            c_int(1),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sgesv",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
            Pointer[Scalar[dtype], origin_of(b_copy)],
            c_int,
        ](
            c_int(101),
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
            b_copy.unsafe_ptr(),
            c_int(1),
        )

    if info > 0:
        raise InvalidParameterError.error(
            "Matrix A is singular and cannot be inverted (minor "
            + String(info)
            + " is exactly zero)",
            "solve",
        )
    elif info < 0:
        raise InvalidParameterError.error(
            "Illegal argument in LAPACK solve (info=" + String(info) + ")",
            "solve",
        )

    return b_copy^


def solve_cholesky[
    dtype: DType = DType.float64
](
    A: Matrix[dtype],
    b: List[Scalar[dtype]],
    lower: Bool = True,
) raises -> List[
    Scalar[dtype]
]:
    """Solves a symmetric positive definite linear system A * x = b using Cholesky (dposv/sposv).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for Cholesky solve"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix A (rows == cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "solve_cholesky",
        )
    if A.rows != len(b):
        raise DimensionMismatchError.error(
            "A.rows == len(b)",
            "A.rows=" + String(A.rows) + ", len(b)=" + String(len(b)),
            "solve_cholesky",
        )

    var N = A.rows
    var A_copy = A.copy()
    var b_copy = b.copy()
    var uplo_char = c_char(ord("L")) if lower else c_char(ord("U"))
    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dposv",
            c_int,
            c_int,
            c_char,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(b_copy)],
            c_int,
        ](
            c_int(101),
            uplo_char,
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            b_copy.unsafe_ptr(),
            c_int(1),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_sposv",
            c_int,
            c_int,
            c_char,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(b_copy)],
            c_int,
        ](
            c_int(101),
            uplo_char,
            c_int(N),
            c_int(1),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            b_copy.unsafe_ptr(),
            c_int(1),
        )

    if info > 0:
        raise InvalidParameterError.error(
            "Matrix A is not positive definite (leading minor "
            + String(info)
            + " is not positive)",
            "solve_cholesky",
        )
    elif info < 0:
        raise InvalidParameterError.error(
            "Illegal argument in LAPACK Cholesky solve (info="
            + String(info)
            + ")",
            "solve_cholesky",
        )

    return b_copy^


def inv[dtype: DType = DType.float64](A: Matrix[dtype]) raises -> Matrix[dtype]:
    """Computes the multiplicative inverse of a square matrix A using LU decomposition (dgetrf/dgetri).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for matrix inversion"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix A (rows == cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "inv",
        )

    var N = A.rows
    var A_inv = A.copy()
    var ipiv = List[c_int](capacity=N)
    for _ in range(N):
        ipiv.append(0)

    var info_getrf: c_int = 0
    var info_getri: c_int = 0

    comptime if dtype == DType.float64:
        info_getrf = external_call[
            "LAPACKE_dgetrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_inv.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
        ](
            c_int(101),
            c_int(N),
            c_int(N),
            A_inv.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
        )
        if info_getrf == 0:
            info_getri = external_call[
                "LAPACKE_dgetri",
                c_int,
                c_int,
                c_int,
                Pointer[Scalar[dtype], origin_of(A_inv.data)],
                c_int,
                Pointer[c_int, origin_of(ipiv)],
            ](
                c_int(101),
                c_int(N),
                A_inv.data.unsafe_ptr(),
                c_int(N),
                ipiv.unsafe_ptr(),
            )
    elif dtype == DType.float32:
        info_getrf = external_call[
            "LAPACKE_sgetrf",
            c_int,
            c_int,
            c_int,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_inv.data)],
            c_int,
            Pointer[c_int, origin_of(ipiv)],
        ](
            c_int(101),
            c_int(N),
            c_int(N),
            A_inv.data.unsafe_ptr(),
            c_int(N),
            ipiv.unsafe_ptr(),
        )
        if info_getrf == 0:
            info_getri = external_call[
                "LAPACKE_sgetri",
                c_int,
                c_int,
                c_int,
                Pointer[Scalar[dtype], origin_of(A_inv.data)],
                c_int,
                Pointer[c_int, origin_of(ipiv)],
            ](
                c_int(101),
                c_int(N),
                A_inv.data.unsafe_ptr(),
                c_int(N),
                ipiv.unsafe_ptr(),
            )

    if info_getrf > 0 or info_getri > 0:
        raise InvalidParameterError.error(
            "Matrix A is singular and cannot be inverted",
            "inv",
        )

    return A_inv^


def norm[
    dtype: DType = DType.float64
](A: Matrix[dtype], ord: String = "fro") raises -> Float64:
    """Computes the Frobenius matrix norm: ||A||_F = sqrt(sum(A_ij^2))."""
    if ord != "fro":
        raise InvalidParameterError.error("ord", "Only 'fro' is supported")

    var total: Float64 = 0.0
    var count = A.rows * A.cols
    for i in range(count):
        var v = Float64(A.data[i])
        total += v * v

    return sqrt(total)


def eigh[
    dtype: DType = DType.float64
](A: Matrix[dtype], UPLO: String = "L") raises -> EigResult[dtype]:
    """Computes the eigenvalues and eigenvectors of a real symmetric matrix.

    Uses LAPACK's divide-and-conquer algorithm (dsyevd / ssyevd).
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Floating-point type required for eigenvalue decomposition"

    if A.rows != A.cols:
        raise DimensionMismatchError.error(
            "Square matrix (A.rows == A.cols)",
            "A(" + String(A.rows) + "x" + String(A.cols) + ")",
            "eigh",
        )

    var N = A.rows
    var A_copy = A.copy()
    var eigenvalues = List[Scalar[dtype]](capacity=N)
    for _ in range(N):
        eigenvalues.append(0)

    var jobz = c_char(ord("V"))
    var uplo_char = c_char(ord("U")) if UPLO == "U" else c_char(ord("L"))

    var info: c_int = 0

    comptime if dtype == DType.float64:
        info = external_call[
            "LAPACKE_dsyevd",
            c_int,
            c_int,
            c_char,
            c_char,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(eigenvalues)],
        ](
            c_int(101),
            jobz,
            uplo_char,
            c_int(N),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            eigenvalues.unsafe_ptr(),
        )
    elif dtype == DType.float32:
        info = external_call[
            "LAPACKE_ssyevd",
            c_int,
            c_int,
            c_char,
            c_char,
            c_int,
            Pointer[Scalar[dtype], origin_of(A_copy.data)],
            c_int,
            Pointer[Scalar[dtype], origin_of(eigenvalues)],
        ](
            c_int(101),
            jobz,
            uplo_char,
            c_int(N),
            A_copy.data.unsafe_ptr(),
            c_int(N),
            eigenvalues.unsafe_ptr(),
        )

    if info != 0:
        raise InvalidParameterError.error(
            "LAPACK symmetric eigenvalue solver did not converge (info="
            + String(info)
            + ")",
            "eigh",
        )

    return EigResult[dtype](eigenvalues^, A_copy^)
