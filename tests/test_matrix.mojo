from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from strata import (
    Matrix,
    CSRMatrix,
    CSCMatrix,
    gemm,
    spmv,
    spmm,
)


def test_dense_matrix_gemm() raises:
    var A = Matrix[DType.float32](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0

    var B = Matrix[DType.float32](2, 2, 0)
    B[0, 0] = 5.0
    B[0, 1] = 6.0
    B[1, 0] = 7.0
    B[1, 1] = 8.0

    var C = gemm(A, B)
    assert_equal(C[0, 0], 19.0)
    assert_equal(C[0, 1], 22.0)
    assert_equal(C[1, 0], 43.0)
    assert_equal(C[1, 1], 50.0)


def test_sparse_csr_spmv_spmm() raises:
    var dense = Matrix[DType.float64](3, 3, 0)
    dense[0, 0] = 10.0
    dense[1, 2] = 20.0
    dense[2, 1] = 30.0

    var csr = CSRMatrix[DType.float64].from_dense(dense)
    assert_equal(csr.nnz(), 3)

    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var y = spmv[DType.float64](csr, x)
    assert_equal(y[0], 10.0)
    assert_equal(y[1], 60.0)
    assert_equal(y[2], 60.0)

    var B = Matrix[DType.float64].ones(3, 2)
    var C = spmm[DType.float64](csr, B)
    assert_equal(C[0, 0], 10.0)
    assert_equal(C[1, 0], 20.0)
    assert_equal(C[2, 0], 30.0)


def test_dense_dot_vec_mixed_precision() raises:
    from strata import dense_dot_vec

    # Int32 input matrix
    var A = Matrix[DType.int32](2, 2, 0)
    A[0, 0] = 2
    A[0, 1] = 3
    A[1, 0] = 4
    A[1, 1] = 5

    # Float64 weights & bias
    var weights: List[Scalar[DType.float64]] = [0.5, 2.0]
    var bias: Scalar[DType.float64] = 1.0

    # Mixed precision matrix-vector product: Int32 @ Float64 -> Float64
    var y = dense_dot_vec(A, weights, bias)
    assert_equal(y[0], 8.0)  # 2*0.5 + 3*2.0 + 1.0 = 1 + 6 + 1 = 8.0
    assert_equal(y[1], 13.0)  # 4*0.5 + 5*2.0 + 1.0 = 2 + 10 + 1 = 13.0


def test_gemm_mixed_precision() raises:
    # Int32 matrix @ Float64 matrix -> Float64 matrix
    var A = Matrix[DType.int32](2, 2, 0)
    A[0, 0] = 1
    A[0, 1] = 2
    A[1, 0] = 3
    A[1, 1] = 4

    var B = Matrix[DType.float64](2, 2, 0.0)
    B[0, 0] = 0.5
    B[0, 1] = 1.5
    B[1, 0] = 2.0
    B[1, 1] = 3.0

    var C = gemm[DType.int32, DType.float64, DType.float64](A, B)
    assert_equal(C[0, 0], 4.5)  # 1*0.5 + 2*2.0 = 0.5 + 4 = 4.5
    assert_equal(C[0, 1], 7.5)  # 1*1.5 + 2*3.0 = 1.5 + 6 = 7.5
    assert_equal(C[1, 0], 9.5)  # 3*0.5 + 4*2.0 = 1.5 + 8 = 9.5
    assert_equal(C[1, 1], 16.5)  # 3*1.5 + 4*3.0 = 4.5 + 12 = 16.5


def test_sparse_mixed_precision() raises:
    # Int32 sparse matrix (e.g. word counts)
    var dense = Matrix[DType.int32](2, 2, 0)
    dense[0, 0] = 3
    dense[1, 1] = 4

    var csr = CSRMatrix[DType.int32].from_dense(dense)

    # Float64 weights
    var weights: List[Scalar[DType.float64]] = [0.5, 2.5]
    var bias: Scalar[DType.float64] = 1.0

    # SpMV: Int32 CSR @ Float64 weights + Float64 bias
    var y = spmv(csr, weights, bias)
    assert_equal(y[0], 2.5)  # 3*0.5 + 1.0 = 2.5
    assert_equal(y[1], 11.0)  # 4*2.5 + 1.0 = 11.0

    # SpMM: Int32 CSR @ Float64 Dense Matrix -> Float64 Dense Matrix
    var B = Matrix[DType.float64].ones(2, 2)
    var C = spmm[DType.int32, DType.float64, DType.float64](csr, B)
    assert_equal(C[0, 0], 3.0)
    assert_equal(C[0, 1], 3.0)
    assert_equal(C[1, 0], 4.0)
    assert_equal(C[1, 1], 4.0)


def test_sparse_validation() raises:
    from std.testing import assert_raises
    from strata import CSCMatrix

    # Invalid indptr length for 2x2 matrix (needs len=3, given len=2)
    var data: List[Scalar[DType.float64]] = [1.0, 2.0]
    var indices: List[Int] = [0, 1]
    var bad_indptr: List[Int] = [0, 2]

    with assert_raises():
        _ = CSCMatrix(2, 2, data.copy(), indices.copy(), bad_indptr.copy())

    with assert_raises():
        _ = CSRMatrix(2, 2, data.copy(), indices.copy(), bad_indptr.copy())


def test_csc_matrix_ops_and_conversion() raises:
    from strata import CSCMatrix

    var dense = Matrix[DType.float64](3, 3, 0)
    dense[0, 0] = 10.0
    dense[1, 2] = 20.0
    dense[2, 1] = 30.0

    var csc = CSCMatrix[DType.float64].from_dense(dense)
    assert_equal(csc.nnz(), 3)
    assert_equal(csc.shape()[0], 3)
    assert_equal(csc.shape()[1], 3)

    # Conversion roundtrip CSC -> CSR -> CSC
    var csr = csc.to_csr()
    assert_equal(csr.nnz(), 3)
    var csc_roundtrip = csr.to_csc()
    assert_equal(csc_roundtrip.nnz(), 3)

    # Direct CSC matrix-vector dot product
    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var y = csc.dot_vec(x)
    assert_equal(y[0], 10.0)
    assert_equal(y[1], 60.0)
    assert_equal(y[2], 60.0)

    # CSC dense matrix multiplication
    var B = Matrix[DType.float64].ones(3, 2)
    var C = csc.dot_dense(B)
    assert_equal(C[0, 0], 10.0)
    assert_equal(C[1, 0], 20.0)
    assert_equal(C[2, 0], 30.0)


def test_csc_dot_vec_wide_accumulation() raises:
    var n = 1000
    var data = List[Scalar[DType.float32]](capacity=n)
    var indices = List[Int](capacity=n)
    var indptr = List[Int](capacity=n + 1)
    indptr.append(0)
    for i in range(n):
        data.append(0.1)
        indices.append(0)
        indptr.append(i + 1)

    var wide = CSCMatrix[DType.float32](1, n, data^, indices^, indptr^)
    var ones = List[Scalar[DType.float32]](capacity=n)
    for _ in range(n):
        ones.append(1.0)

    var y = wide.dot_vec(ones)
    var err = Float64(y[0]) - 100.0
    if err < 0.0:
        err = -err
    assert_true(err < 1e-8)


def test_sparse_allows_empty_dimensions() raises:
    var empty_csr = CSRMatrix[DType.float64].empty(0, 4)
    assert_equal(empty_csr.nnz(), 0)
    assert_equal(empty_csr.shape()[0], 0)
    assert_equal(empty_csr.shape()[1], 4)

    var empty_csc = CSCMatrix[DType.float64].from_dense(
        Matrix[DType.float64](0, 3, 0)
    )
    assert_equal(empty_csc.nnz(), 0)


def test_sparse_rejects_negative_dimensions() raises:
    var data = List[Scalar[DType.float64]]()
    var indices = List[Int]()
    var indptr: List[Int] = [0]
    with assert_raises():
        _ = CSRMatrix[DType.float64](-1, 2, data^, indices^, indptr^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
