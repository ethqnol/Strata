from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_raises,
    assert_almost_equal,
from strata import (
    Matrix,
    gemm,
    dense_dot_vec,
    hstack,
    vstack,
    DimensionMismatchError,
    InvalidParameterError,
)


def test_matrix_construction_and_access() raises:
    # Zeros, ones, custom fill
    var z = Matrix[DType.float64].zeros(3, 4)
    assert_equal(z.rows, 3)
    assert_equal(z.cols, 4)
    assert_equal(z.num_elements(), 12)
    assert_equal(z.shape()[0], 3)
    assert_equal(z.shape()[1], 4)
    for r in range(3):
        for c in range(4):
            assert_equal(z[r, c], 0.0)

    var o = Matrix[DType.float32].ones(2, 5)
    assert_equal(o.rows, 2)
    assert_equal(o.cols, 5)
    for r in range(2):
        for c in range(5):
            assert_equal(o[r, c], 1.0)

    var filled = Matrix[DType.int32](3, 3, -7)
    for r in range(3):
        for c in range(3):
            assert_equal(filled[r, c], -7)

    # Set item
    filled[1, 2] = 42
    assert_equal(filled[1, 2], 42)


def test_matrix_eye_properties() raises:
    for n in range(1, 6):
        var Eye = Matrix[DType.float64].eye(n)
        assert_equal(Eye.rows, n)
        assert_equal(Eye.cols, n)
        for r in range(n):
            for c in range(n):
                var expected: Float64 = 1.0 if r == c else 0.0
                assert_equal(Eye[r, c], expected)


def test_matrix_row_and_col_extraction() raises:
    var m = Matrix[DType.float64](3, 3, 0)
    var counter: Float64 = 1.0
    for r in range(3):
        for c in range(3):
            m[r, c] = counter
            counter += 1.0

    var r1 = m.row(1)
    assert_equal(len(r1), 3)
    assert_equal(r1[0], 4.0)
    assert_equal(r1[1], 5.0)
    assert_equal(r1[2], 6.0)

    var c2 = m.col(2)
    assert_equal(len(c2), 3)
    assert_equal(c2[0], 3.0)
    assert_equal(c2[1], 6.0)
    assert_equal(c2[2], 9.0)


def test_matrix_transpose() raises:
    # Non-square matrix transpose (2x4 -> 4x2)
    var A = Matrix[DType.float64](2, 4, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[0, 3] = 4.0
    A[1, 0] = 5.0
    A[1, 1] = 6.0
    A[1, 2] = 7.0
    A[1, 3] = 8.0

    var At = A.transpose()
    assert_equal(At.rows, 4)
    assert_equal(At.cols, 2)
    assert_equal(At[0, 0], 1.0)
    assert_equal(At[0, 1], 5.0)
    assert_equal(At[3, 0], 4.0)
    assert_equal(At[3, 1], 8.0)

    # Double transpose identity: (A^T)^T == A
    var Att = At.transpose()
    assert_equal(Att.rows, 2)
    assert_equal(Att.cols, 4)
    for r in range(2):
        for c in range(4):
            assert_equal(Att[r, c], A[r, c])


def test_matrix_copy_independence() raises:
    var A = Matrix[DType.float64](2, 2, 10.0)
    var B = A.copy()
    B[0, 0] = 999.0

    assert_equal(A[0, 0], 10.0)  # Original remains unmutated
    assert_equal(B[0, 0], 999.0)


def test_matrix_cast_promotions() raises:
    var m_int = Matrix[DType.int32](2, 2, 0)
    m_int[0, 0] = 10
    m_int[0, 1] = -20
    m_int[1, 0] = 30
    m_int[1, 1] = 40

    # Int32 -> Float64
    var m_f64 = m_int.cast[DType.float64]()
    assert_equal(m_f64.rows, 2)
    assert_equal(m_f64.cols, 2)
    assert_equal(m_f64[0, 0], 10.0)
    assert_equal(m_f64[0, 1], -20.0)

    # Float64 -> Float32
    var m_f32 = m_f64.cast[DType.float32]()
    assert_equal(m_f32[1, 0], 30.0)
    assert_equal(m_f32[1, 1], 40.0)

    # Float32 -> Int64
    var m_i64 = m_f32.cast[DType.int64]()
    assert_equal(m_i64[0, 1], -20)


def test_matrix_mean_and_std_axis_0() raises:
    var m = Matrix[DType.float64](4, 3, 0)
    # Col 0: [10, 20, 30, 40] -> Mean = 25, Std = sqrt(125)
    m[0, 0] = 10.0
    m[1, 0] = 20.0
    m[2, 0] = 30.0
    m[3, 0] = 40.0

    # Col 1: Constant [5, 5, 5, 5] -> Mean = 5, Std fallback = 1.0
    m[0, 1] = 5.0
    m[1, 1] = 5.0
    m[2, 1] = 5.0
    m[3, 1] = 5.0

    # Col 2: [0, 0, 0, 0] -> Mean = 0, Std fallback = 1.0
    m[0, 2] = 0.0
    m[1, 2] = 0.0
    m[2, 2] = 0.0
    m[3, 2] = 0.0

    var means = m.mean_along_axis_0()
    assert_equal(len(means), 3)
    assert_equal(means[0], 25.0)
    assert_equal(means[1], 5.0)
    assert_equal(means[2], 0.0)

    var stds = m.std_along_axis_0(means)
    assert_equal(len(stds), 3)
    assert_equal(stds[1], 1.0)  # Constant column fallback
    assert_equal(stds[2], 1.0)

    # Zero-row matrix guard
    var empty_m = Matrix[DType.float64](0, 3, 0)
    var empty_means = empty_m.mean_along_axis_0()
    assert_equal(len(empty_means), 3)
    var empty_stds = empty_m.std_along_axis_0(empty_means)
    assert_equal(len(empty_stds), 3)


def test_gemm_properties() raises:
    # 1. Standard non-square GEMM: (2x3) @ (3x2) -> (2x2)
    var A = Matrix[DType.float64](2, 3, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0

    var B = Matrix[DType.float64](3, 2, 0)
    B[0, 0] = 7.0
    B[0, 1] = 8.0
    B[1, 0] = 9.0
    B[1, 1] = 1.0
    B[2, 0] = 2.0
    B[2, 1] = 3.0

    var C = gemm(A, B)
    assert_equal(C.rows, 2)
    assert_equal(C.cols, 2)
    # C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    assert_equal(C[0, 0], 31.0)
    # C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    assert_equal(C[0, 1], 19.0)
    # C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    assert_equal(C[1, 0], 85.0)
    # C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert_equal(C[1, 1], 55.0)

    # 2. Identity matrix multiplication: A @ I == A and I @ A == A
    var I3 = Matrix[DType.float64].eye(3)
    var AI = gemm(A, I3)
    for r in range(2):
        for c in range(3):
            assert_equal(AI[r, c], A[r, c])

    var I2 = Matrix[DType.float64].eye(2)
    var IA = gemm(I2, A)
    for r in range(2):
        for c in range(3):
            assert_equal(IA[r, c], A[r, c])

    # 3. Dimension mismatch error check
    var BadB = Matrix[DType.float64].ones(4, 2)
    with assert_raises():
        _ = gemm(A, BadB)


def test_integer_gemm_and_outer_products() raises:
    # 1. Integer GEMM (Int32) fallback loop verification
    var A_i32 = Matrix[DType.int32](2, 2, 0)
    A_i32[0, 0] = 1
    A_i32[0, 1] = 2
    A_i32[1, 0] = 3
    A_i32[1, 1] = 4

    var B_i32 = Matrix[DType.int32](2, 2, 0)
    B_i32[0, 0] = 5
    B_i32[0, 1] = 6
    B_i32[1, 0] = 7
    B_i32[1, 1] = 8

    var C_i32 = gemm(A_i32, B_i32)
    assert_equal(C_i32[0, 0], 19)
    assert_equal(C_i32[0, 1], 22)
    assert_equal(C_i32[1, 0], 43)
    assert_equal(C_i32[1, 1], 50)

    # 2. Outer product: (3x1) @ (1x3) -> (3x3)
    var Col = Matrix[DType.float64](3, 1, 0)
    Col[0, 0] = 1.0
    Col[1, 0] = 2.0
    Col[2, 0] = 3.0

    var Row = Matrix[DType.float64](1, 3, 0)
    Row[0, 0] = 4.0
    Row[0, 1] = 5.0
    Row[0, 2] = 6.0

    var Outer = gemm(Col, Row)
    assert_equal(Outer.rows, 3)
    assert_equal(Outer.cols, 3)
    assert_equal(Outer[0, 0], 4.0)
    assert_equal(Outer[1, 1], 10.0)
    assert_equal(Outer[2, 2], 18.0)

    # 3. Inner product: (1x3) @ (3x1) -> (1x1)
    var Inner = gemm(Row, Col)
    assert_equal(Inner.rows, 1)
    assert_equal(Inner.cols, 1)
    assert_equal(Inner[0, 0], 32.0)  # 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32


def test_dense_dot_vec_properties() raises:
    var A = Matrix[DType.float64](2, 3, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0

    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var bias: Scalar[DType.float64] = 10.0

    # y = A @ x + bias: [1*1 + 2*2 + 3*3 + 10 = 24, 4*1 + 5*2 + 6*3 + 10 = 42]
    var y = dense_dot_vec(A, x, bias)
    assert_equal(len(y), 2)
    assert_equal(y[0], 24.0)
    assert_equal(y[1], 42.0)

    # Zero bias test
    var y_nobias = dense_dot_vec(A, x, 0.0)
    assert_equal(y_nobias[0], 14.0)
    assert_equal(y_nobias[1], 32.0)

    # Integer dot vec
    var A_i32 = Matrix[DType.int32](2, 2, 3)
    var x_i32: List[Scalar[DType.int32]] = [2, 4]
    var y_i32 = dense_dot_vec(A_i32, x_i32, 5)
    assert_equal(y_i32[0], 23)
    assert_equal(y_i32[1], 23)

    # Dimension mismatch check
    var bad_x: List[Scalar[DType.float64]] = [1.0, 2.0]
    with assert_raises():
        _ = dense_dot_vec(A, bad_x)


def test_matrix_string_representation() raises:
    var m = Matrix[DType.float64](2, 2, 3.14)
    var s = String(m)
    assert_true(s.byte_length() > 0)


def test_matrix_mean_along_axis_0_computation() raises:
    var A = Matrix[DType.float64](3, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 10.0
    A[1, 0] = 2.0
    A[1, 1] = 20.0
    A[2, 0] = 3.0
    A[2, 1] = 30.0

    var means = A.mean_along_axis_0()
    assert_equal(len(means), 2)
    assert_almost_equal(means[0], 2.0)
    assert_almost_equal(means[1], 20.0)


def test_matrix_scalar_multiplication_and_division() raises:
    var A = Matrix[DType.float64](2, 2, 6.0)
    var B_mul = A * 2.0
    assert_equal(B_mul[0, 0], 12.0)
    assert_equal(B_mul[1, 1], 12.0)

    var B_div = A / 3.0
    assert_equal(B_div[0, 0], 2.0)
    assert_equal(B_div[1, 1], 2.0)


def test_matrix_gemm_associativity() raises:
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0

    var B = Matrix[DType.float64](2, 2, 0)
    B[0, 0] = 2.0
    B[0, 1] = 0.0
    B[1, 0] = 1.0
    B[1, 1] = 2.0

    var C = Matrix[DType.float64](2, 2, 0)
    C[0, 0] = 0.0
    C[0, 1] = 1.0
    C[1, 0] = 1.0
    C[1, 1] = 0.0

    # (A @ B) @ C == A @ (B @ C)
    var AB_C = gemm(gemm(A, B), C)
    var A_BC = gemm(A, gemm(B, C))

    for r in range(2):
        for c in range(2):
            assert_almost_equal(AB_C[r, c], A_BC[r, c], rtol=1e-5)


def test_matrix_gemm_distributivity() raises:
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0

    var B = Matrix[DType.float64](2, 2, 1.0)
    var C = Matrix[DType.float64](2, 2, 2.0)

    # A @ (B + C) == A @ B + A @ C
    var lhs = gemm(A, B + C)
    var rhs = gemm(A, B) + gemm(A, C)

    for r in range(2):
        for c in range(2):
            assert_almost_equal(lhs[r, c], rhs[r, c], rtol=1e-5)


def test_matrix_transpose_product_identity() raises:
    var A = Matrix[DType.float64](2, 3, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0

    var B = Matrix[DType.float64](3, 2, 0)
    B[0, 0] = 7.0
    B[0, 1] = 8.0
    B[1, 0] = 9.0
    B[1, 1] = 1.0
    B[2, 0] = 2.0
    B[2, 1] = 3.0

    # (A @ B)^T == B^T @ A^T
    var lhs = gemm(A, B).transpose()
    var rhs = gemm(B.transpose(), A.transpose())

    for r in range(2):
        for c in range(2):
            assert_almost_equal(lhs[r, c], rhs[r, c], rtol=1e-5)


def test_matrix_cast_precision_conversions() raises:
    var A_f64 = Matrix[DType.float64](2, 2, 3.14159265)
    var A_f32 = A_f64.cast[DType.float32]()
    assert_equal(A_f32.rows, 2)
    assert_equal(A_f32.cols, 2)
    assert_almost_equal(Float64(A_f32[0, 0]), 3.14159265, rtol=1e-4)

    var A_i32 = A_f64.cast[DType.int32]()
    assert_equal(A_i32[0, 0], 3)
    assert_equal(A_i32[1, 1], 3)


def test_matrix_1x1_scalar_multiplication() raises:
    var A = Matrix[DType.float64](1, 1, 5.0)
    var B = Matrix[DType.float64](1, 1, 4.0)
    var C = gemm(A, B)
    assert_equal(C.rows, 1)
    assert_equal(C.cols, 1)
    assert_equal(C[0, 0], 20.0)


def test_matrix_tall_and_wide_products() raises:
    var Col = Matrix[DType.float64].ones(10, 1)
    var Row = Matrix[DType.float64].ones(1, 10)
    var Outer = gemm(Col, Row)
    assert_equal(Outer.rows, 10)
    assert_equal(Outer.cols, 10)
    for r in range(10):
        for c in range(10):
            assert_equal(Outer[r, c], 1.0)


def test_matrix_bfloat16_gemm() raises:
    var A = Matrix[DType.bfloat16](2, 2, 2.0)
    var B = Matrix[DType.bfloat16](2, 2, 3.0)
    var C = gemm(A, B)
    assert_equal(C.rows, 2)
    assert_equal(C.cols, 2)
    assert_almost_equal(Float64(C[0, 0]), 12.0, rtol=1e-2)


def test_matrix_dimension_mismatch_errors() raises:
    var A = Matrix[DType.float64](2, 3, 1.0)
    var B = Matrix[DType.float64](2, 3, 1.0)

    with assert_raises():
        _ = gemm(A, B)

    var C = Matrix[DType.float64](3, 2, 1.0)
    with assert_raises():
        _ = A + C


def test_matrix_fill_zeros_ones_factories() raises:
    var z = Matrix[DType.float64].zeros(4, 4)
    for r in range(4):
        for c in range(4):
            assert_equal(z[r, c], 0.0)

    var o = Matrix[DType.float64].ones(4, 4)
    for r in range(4):
        for c in range(4):
            assert_equal(o[r, c], 1.0)


def test_matrix_subtraction_self_is_zeros() raises:
    var A = Matrix[DType.float64](3, 3, 42.0)
    var Diff = A - A
    for r in range(3):
        for c in range(3):
            assert_equal(Diff[r, c], 0.0)


def test_matrix_hadamard_with_identity() raises:
    var A = Matrix[DType.float64](3, 3, 5.0)
    var Eye = Matrix[DType.float64].eye(3)
    var H = A * Eye
    for r in range(3):
        for c in range(3):
            assert_equal(H[r, c], 5.0 if r == c else 0.0)


def test_matrix_transpose_symmetry_for_identity() raises:
    var Eye = Matrix[DType.float64].eye(4)
    var Eye_t = Eye.transpose()
    for r in range(4):
        for c in range(4):
            assert_equal(Eye[r, c], Eye_t[r, c])


def test_matrix_select_columns_and_rows() raises:
    # 3x4 Matrix
    # [ 1,  2,  3,  4 ]
    # [ 5,  6,  7,  8 ]
    # [ 9, 10, 11, 12 ]
    var A = Matrix[DType.float64](3, 4, 0)
    for r in range(3):
        for c in range(4):
            A[r, c] = Float64(r * 4 + c + 1)

    # 1. Select columns [0, 2] -> 3x2 submatrix
    var cols_idx: List[Int] = [0, 2]
    var sub_cols = A.select_columns(cols_idx)
    assert_equal(sub_cols.rows, 3)
    assert_equal(sub_cols.cols, 2)
    assert_equal(sub_cols[0, 0], 1.0)
    assert_equal(sub_cols[0, 1], 3.0)
    assert_equal(sub_cols[1, 0], 5.0)
    assert_equal(sub_cols[1, 1], 7.0)
    assert_equal(sub_cols[2, 0], 9.0)
    assert_equal(sub_cols[2, 1], 11.0)

    # 2. Select rows [2, 0] (permutation) -> 2x4 submatrix
    var rows_idx: List[Int] = [2, 0]
    var sub_rows = A.select_rows(rows_idx)
    assert_equal(sub_rows.rows, 2)
    assert_equal(sub_rows.cols, 4)
    assert_equal(sub_rows[0, 0], 9.0)
    assert_equal(sub_rows[0, 3], 12.0)
    assert_equal(sub_rows[1, 0], 1.0)
    assert_equal(sub_rows[1, 3], 4.0)


def test_matrix_hstack_and_vstack() raises:
    var A = Matrix[DType.float64](2, 2, 0)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0

    var B = Matrix[DType.float64](2, 3, 0)
    B[0, 0] = 10.0
    B[0, 1] = 20.0
    B[0, 2] = 30.0
    B[1, 0] = 40.0
    B[1, 1] = 50.0
    B[1, 2] = 60.0

    # 1. Horizontal stacking (2x2) hstack (2x3) -> (2x5)
    var H = hstack(A, B)
    assert_equal(H.rows, 2)
    assert_equal(H.cols, 5)
    assert_equal(H[0, 0], 1.0)
    assert_equal(H[0, 1], 2.0)
    assert_equal(H[0, 2], 10.0)
    assert_equal(H[0, 4], 30.0)
    assert_equal(H[1, 0], 3.0)
    assert_equal(H[1, 4], 60.0)

    # 2. Vertical stacking (2x2) vstack (2x2) -> (4x2)
    var V = vstack(A, A)
    assert_equal(V.rows, 4)
    assert_equal(V.cols, 2)
    assert_equal(V[0, 0], 1.0)
    assert_equal(V[1, 1], 4.0)
    assert_equal(V[2, 0], 1.0)
    assert_equal(V[3, 1], 4.0)


def test_matrix_ops_bounds_and_mismatch() raises:
    var A = Matrix[DType.float64](2, 2, 1.0)

    # Invalid column index
    var bad_cols: List[Int] = [0, 5]
    with assert_raises():
        _ = A.select_columns(bad_cols)

    # Invalid row index
    var bad_rows: List[Int] = [-1, 0]
    with assert_raises():
        _ = A.select_rows(bad_rows)

    # hstack row count mismatch
    var C_3rows = Matrix[DType.float64](3, 2, 1.0)
    with assert_raises():
        _ = hstack(A, C_3rows)

    # vstack col count mismatch
    var D_3cols = Matrix[DType.float64](2, 3, 1.0)
    with assert_raises():
        _ = vstack(A, D_3cols)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
