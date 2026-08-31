from std.math import sqrt
from std.python import PythonObject
from .types import ArrayLike
from .view import MatrixView
from ..exceptions.errors import DimensionMismatchError, InvalidParameterError


struct Matrix[dtype: DType = DType.float64](
    ArrayLike, Copyable, Movable, Writable
):
    """Dense 2D row-major matrix container with striding and view support.

    Provides contiguous buffer allocation, SIMD-compatible row-major layout,
    slicing, element-wise arithmetic, and BLAS/LAPACK interop.

    Parameters:
        dtype: Numerical data type of matrix elements. Default DType.float64.

    Attributes:
        rows: Number of matrix rows ($N$).
        cols: Number of matrix columns ($D$).
        data: Flat 1D buffer of matrix elements in row-major order.

    Examples:
        ```mojo
        from strata.core import Matrix

        var A = Matrix[DType.float64](2, 3, fill=1.0)
        var B = Matrix[DType.float64].eye(3)
        var C = A.dot(B)
        ```
    """

    var rows: Int
    var cols: Int
    var data: List[Scalar[Self.dtype]]

    def __init__(out self, rows: Int, cols: Int, fill: Scalar[Self.dtype] = 0):
        """Initialize a Matrix of shape (rows, cols) filled with a constant value.

        Args:
            rows: Number of rows ($N >= 0$).
            cols: Number of columns ($D >= 0$).
            fill: Constant scalar value to fill the buffer. Default 0.
        """
        self.rows = rows
        self.cols = cols
        var total = rows * cols
        var d = List[Scalar[Self.dtype]](capacity=total)
        for _ in range(total):
            d.append(fill)
        self.data = d^

    def __init__(
        out self, rows: Int, cols: Int, var data: List[Scalar[Self.dtype]]
    ):
        """Initialize a Matrix from an existing 1D buffer.

        Args:
            rows: Number of rows ($N >= 0$).
            cols: Number of columns ($D >= 0$).
            data: Flat row-major element buffer of length `rows * cols`.
        """

        self.rows = rows
        self.cols = cols
        self.data = data^

    @staticmethod
    def zeros(rows: Int, cols: Int) -> Self:
        """Create a zero-filled Matrix of shape (rows, cols)."""
        return Self(rows, cols, 0)

    @staticmethod
    def ones(rows: Int, cols: Int) -> Self:
        """Create a one-filled Matrix of shape (rows, cols)."""
        return Self(rows, cols, 1)

    @staticmethod
    def eye(n: Int) -> Self:
        """Create an identity Matrix of shape (n, n)."""
        var res = Self.zeros(n, n)
        for i in range(n):
            res[i, i] = 1
        return res^

    @staticmethod
    def from_numpy(np_arr: PythonObject) raises -> Self:
        from .interop import matrix_from_numpy

        return matrix_from_numpy[Self.dtype](np_arr)

    def to_numpy(self) raises -> PythonObject:
        from .interop import matrix_to_numpy

        return matrix_to_numpy[Self.dtype](self)

    def num_rows(self) -> Int:
        return self.rows

    def num_cols(self) -> Int:
        return self.cols

    def num_elements(self) -> Int:
        return self.rows * self.cols

    def shape(self) -> Tuple[Int, Int]:
        return (self.rows, self.cols)

    def view(ref self) -> MatrixView[Self.dtype, origin_of(self.data)]:
        return MatrixView[Self.dtype, origin_of(self.data)](
            self.data.unsafe_ptr(), self.rows, self.cols, self.cols, 1
        )

    def slice_rows(
        ref self, start_row: Int, end_row: Int
    ) raises -> MatrixView[Self.dtype, origin_of(self.data)]:
        return self.view().slice_rows(start_row, end_row)

    def slice_cols(
        ref self, start_col: Int, end_col: Int
    ) raises -> MatrixView[Self.dtype, origin_of(self.data)]:
        return self.view().slice_cols(start_col, end_col)

    def slice_2d(
        ref self,
        start_row: Int,
        end_row: Int,
        start_col: Int,
        end_col: Int,
    ) raises -> MatrixView[Self.dtype, origin_of(self.data)]:
        return self.view().slice_2d(start_row, end_row, start_col, end_col)

    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.dtype]:
        debug_assert(r >= 0 and r < self.rows, "row index out of bounds")
        debug_assert(c >= 0 and c < self.cols, "column index out of bounds")
        return self.data[r * self.cols + c]

    def __setitem__(mut self, r: Int, c: Int, val: Scalar[Self.dtype]):
        debug_assert(r >= 0 and r < self.rows, "row index out of bounds")
        debug_assert(c >= 0 and c < self.cols, "column index out of bounds")
        self.data[r * self.cols + c] = val

    def row(self, r: Int) -> List[Scalar[Self.dtype]]:
        var res = List[Scalar[Self.dtype]](capacity=self.cols)
        var start_idx = r * self.cols
        for c in range(self.cols):
            res.append(self.data[start_idx + c])
        return res^

    def col(self, c: Int) -> List[Scalar[Self.dtype]]:
        var res = List[Scalar[Self.dtype]](capacity=self.rows)
        for r in range(self.rows):
            res.append(self.data[r * self.cols + c])
        return res^

    def cast[target_dtype: DType](self) -> Matrix[target_dtype]:
        """Promotes or converts the Matrix elements to target_dtype."""
        var total = self.rows * self.cols
        var new_data = List[Scalar[target_dtype]](capacity=total)
        for i in range(total):
            new_data.append(Scalar[target_dtype](self.data[i]))
        return Matrix[target_dtype](self.rows, self.cols, new_data^)

    def transpose(self) -> Self:
        var res = Self(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                res[c, r] = self[r, c]
        return res^

    def select_columns(self, indices: List[Int]) raises -> Self:
        """Extracts an $N \\times K$ submatrix containing only the specified column indices in order.

        Args:
            indices: List of column indices to extract.

        Returns:
            Matrix[dtype]: Submatrix of shape (self.rows, len(indices)).

        Raises:
            InvalidParameterError: If any index is negative or >= self.cols.
        """
        var n_sub_cols = len(indices)
        for i in range(n_sub_cols):
            var c = indices[i]
            if c < 0 or c >= self.cols:
                raise InvalidParameterError.error(
                    "indices[" + String(i) + "]",
                    "Column index "
                    + String(c)
                    + " is out of bounds for matrix with "
                    + String(self.cols)
                    + " columns",
                )

        var res = Self(self.rows, n_sub_cols)
        for r in range(self.rows):
            var row_offset = r * self.cols
            var out_offset = r * n_sub_cols
            for c_idx in range(n_sub_cols):
                var src_col = indices[c_idx]
                res.data[out_offset + c_idx] = self.data[row_offset + src_col]
        return res^

    def select_rows(self, indices: List[Int]) raises -> Self:
        """Extracts an $M \\times D$ submatrix containing only the specified row indices in order.

        Args:
            indices: List of row indices to extract.

        Returns:
            Matrix[dtype]: Submatrix of shape (len(indices), self.cols).

        Raises:
            InvalidParameterError: If any index is negative or >= self.rows.
        """
        var n_sub_rows = len(indices)
        for i in range(n_sub_rows):
            var r = indices[i]
            if r < 0 or r >= self.rows:
                raise InvalidParameterError.error(
                    "indices[" + String(i) + "]",
                    "Row index "
                    + String(r)
                    + " is out of bounds for matrix with "
                    + String(self.rows)
                    + " rows",
                )

        var res = Self(n_sub_rows, self.cols)
        for r_idx in range(n_sub_rows):
            var src_row = indices[r_idx]
            var src_offset = src_row * self.cols
            var out_offset = r_idx * self.cols
            for c in range(self.cols):
                res.data[out_offset + c] = self.data[src_offset + c]
        return res^

    def __add__(self, other: Self) raises -> Self:
        """Element-wise matrix addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionMismatchError.error(
                "Matching shapes for addition ("
                + String(self.rows)
                + "x"
                + String(self.cols)
                + ")",
                "(" + String(other.rows) + "x" + String(other.cols) + ")",
                "Matrix.__add__",
            )
        var total = self.rows * self.cols
        var res = List[Scalar[Self.dtype]](capacity=total)
        for i in range(total):
            res.append(self.data[i] + other.data[i])
        return Matrix[Self.dtype](self.rows, self.cols, res^)

    def __sub__(self, other: Self) raises -> Self:
        """Element-wise matrix subtraction."""
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionMismatchError.error(
                "Matching shapes for subtraction ("
                + String(self.rows)
                + "x"
                + String(self.cols)
                + ")",
                "(" + String(other.rows) + "x" + String(other.cols) + ")",
                "Matrix.__sub__",
            )
        var total = self.rows * self.cols
        var res = List[Scalar[Self.dtype]](capacity=total)
        for i in range(total):
            res.append(self.data[i] - other.data[i])
        return Matrix[Self.dtype](self.rows, self.cols, res^)

    def __mul__(self, other: Self) raises -> Self:
        """Element-wise Hadamard matrix product."""
        if self.rows != other.rows or self.cols != other.cols:
            raise DimensionMismatchError.error(
                "Matching shapes for element-wise multiplication ("
                + String(self.rows)
                + "x"
                + String(self.cols)
                + ")",
                "(" + String(other.rows) + "x" + String(other.cols) + ")",
                "Matrix.__mul__",
            )
        var total = self.rows * self.cols
        var res = List[Scalar[Self.dtype]](capacity=total)
        for i in range(total):
            res.append(self.data[i] * other.data[i])
        return Matrix[Self.dtype](self.rows, self.cols, res^)

    def __mul__(self, scalar: Scalar[Self.dtype]) -> Self:
        """Scalar multiplication."""
        var total = self.rows * self.cols
        var res = List[Scalar[Self.dtype]](capacity=total)
        for i in range(total):
            res.append(self.data[i] * scalar)
        return Matrix[Self.dtype](self.rows, self.cols, res^)

    def __rmul__(self, scalar: Scalar[Self.dtype]) -> Self:
        """Right scalar multiplication."""
        return self * scalar

    def __truediv__(self, scalar: Scalar[Self.dtype]) -> Self:
        """Scalar division."""
        var total = self.rows * self.cols
        var res = List[Scalar[Self.dtype]](capacity=total)
        for i in range(total):
            res.append(self.data[i] / scalar)
        return Matrix[Self.dtype](self.rows, self.cols, res^)

    def __neg__(self) -> Self:
        """Unary negation."""
        var total = self.rows * self.cols
        var res = List[Scalar[Self.dtype]](capacity=total)
        for i in range(total):
            res.append(-self.data[i])
        return Matrix[Self.dtype](self.rows, self.cols, res^)

    def dot(self, other: Self) raises -> Self:
        from .linalg import gemm

        return gemm[Self.dtype](self, other)

    def dot_vec(
        self, vec: List[Scalar[Self.dtype]]
    ) raises -> List[Scalar[Self.dtype]]:
        from .linalg import dense_dot_vec

        return dense_dot_vec[Self.dtype](self, vec)

    def mean_along_axis_0(self) -> List[Scalar[Self.dtype]]:
        var means = List[Scalar[Self.dtype]](capacity=self.cols)
        if self.rows == 0:
            for _ in range(self.cols):
                means.append(0)
            return means^
        var n_rows = Float64(self.rows)
        for c in range(self.cols):
            var total: Float64 = 0.0
            for r in range(self.rows):
                total += Float64(self[r, c])
            means.append(Scalar[Self.dtype](total / n_rows))
        return means^

    def std_along_axis_0(
        self, means: List[Scalar[Self.dtype]]
    ) -> List[Scalar[Self.dtype]]:
        var stds = List[Scalar[Self.dtype]](capacity=self.cols)
        if self.rows == 0:
            for _ in range(self.cols):
                stds.append(1)
            return stds^
        var n_rows = Float64(self.rows)
        for c in range(self.cols):
            var var_sum: Float64 = 0.0
            var col_mean = Float64(means[c])
            for r in range(self.rows):
                var diff = Float64(self[r, c]) - col_mean
                var_sum += diff * diff
            var s = sqrt(var_sum / n_rows)
            if s == 0.0:
                s = 1.0
            stds.append(Scalar[Self.dtype](s))
        return stds^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Matrix[",
            String(Self.dtype),
            "](",
            self.rows,
            "x",
            self.cols,
            ")\n[",
        )
        for r in range(self.rows):
            if r > 0:
                writer.write(" ")
            writer.write("[")
            for c in range(self.cols):
                writer.write(self[r, c])
                if c + 1 < self.cols:
                    writer.write(", ")
            writer.write("]")
            if r + 1 < self.rows:
                writer.write("\n")
        writer.write("]")


def hstack[
    dtype: DType = DType.float64
](matrices: List[Matrix[dtype]]) raises -> Matrix[dtype]:
    """Horizontally stacks a list of matrices with matching row counts.

    Args:
        matrices: List of Matrix instances to concatenate along axis 1.

    Returns:
        Matrix[dtype]: Concatenated matrix of shape (rows, sum(cols)).

    Raises:
        InvalidParameterError: If matrices list is empty.
        DimensionMismatchError: If any matrix has a different row count.
    """
    var n_mats = len(matrices)
    if n_mats == 0:
        raise InvalidParameterError.error(
            "matrices", "Cannot hstack empty list of matrices"
        )
    if n_mats == 1:
        return matrices[0].copy()

    var rows = matrices[0].rows
    var total_cols = 0
    for i in range(n_mats):
        if matrices[i].rows != rows:
            raise DimensionMismatchError.error(
                "Matrix 0 has " + String(rows) + " rows",
                "Matrix "
                + String(i)
                + " has "
                + String(matrices[i].rows)
                + " rows",
                "hstack",
            )
        total_cols += matrices[i].cols

    var res = Matrix[dtype](rows, total_cols)
    var col_offset = 0
    for m_idx in range(n_mats):
        var m_cols = matrices[m_idx].cols
        for r in range(rows):
            var src_offset = r * m_cols
            var out_offset = r * total_cols + col_offset
            for c in range(m_cols):
                res.data[out_offset + c] = matrices[m_idx].data[src_offset + c]
        col_offset += m_cols

    return res^


def hstack[
    dtype: DType = DType.float64
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[dtype]:
    """Horizontally stacks two matrices with matching row counts."""
    var mats = List[Matrix[dtype]](capacity=2)
    mats.append(A.copy())
    mats.append(B.copy())
    return hstack[dtype](mats)


def vstack[
    dtype: DType = DType.float64
](matrices: List[Matrix[dtype]]) raises -> Matrix[dtype]:
    """Vertically stacks a list of matrices with matching column counts.

    Args:
        matrices: List of Matrix instances to concatenate along axis 0.

    Returns:
        Matrix[dtype]: Concatenated matrix of shape (sum(rows), cols).

    Raises:
        InvalidParameterError: If matrices list is empty.
        DimensionMismatchError: If any matrix has a different column count.
    """
    var n_mats = len(matrices)
    if n_mats == 0:
        raise InvalidParameterError.error(
            "matrices", "Cannot vstack empty list of matrices"
        )
    if n_mats == 1:
        return matrices[0].copy()

    var cols = matrices[0].cols
    var total_rows = 0
    for i in range(n_mats):
        if matrices[i].cols != cols:
            raise DimensionMismatchError.error(
                "Matrix 0 has " + String(cols) + " cols",
                "Matrix "
                + String(i)
                + " has "
                + String(matrices[i].cols)
                + " cols",
                "vstack",
            )
        total_rows += matrices[i].rows

    var res = Matrix[dtype](total_rows, cols)
    var row_offset = 0
    for m_idx in range(n_mats):
        var m_rows = matrices[m_idx].rows
        var m_size = m_rows * cols
        var out_offset = row_offset * cols
        for i in range(m_size):
            res.data[out_offset + i] = matrices[m_idx].data[i]
        row_offset += m_rows

    return res^


def vstack[
    dtype: DType = DType.float64
](A: Matrix[dtype], B: Matrix[dtype]) raises -> Matrix[dtype]:
    """Vertically stacks two matrices with matching column counts."""
    var mats = List[Matrix[dtype]](capacity=2)
    mats.append(A.copy())
    mats.append(B.copy())
    return vstack[dtype](mats)
