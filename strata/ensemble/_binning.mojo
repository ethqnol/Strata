from std.math import round
from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct BinnedMatrix(Copyable, Movable):
    """Compact uint8 row-major matrix representation for binned feature data.

    Each feature value $x_{i, j}$ is mapped to a discrete integer bin in $[0, K_j - 1]$,
    where $K_j \\le 256$ is the number of distinct bins for feature $j$.

    Attributes:
        rows: Number of samples $N$.
        cols: Number of features $D$.
        data: Flattened $N \\times D$ array of bin indices stored as unsigned 8-bit integers.
        bin_thresholds: List of monotonic threshold vectors for each feature dimension.
        n_bins_per_feature: Number of active discrete bins per feature dimension ($K_j$).
    """

    var rows: Int
    var cols: Int
    var data: List[UInt8]
    var bin_thresholds: List[List[Float64]]
    var n_bins_per_feature: List[Int]

    def __init__(
        out self,
        rows: Int,
        cols: Int,
        var data: List[UInt8],
        var bin_thresholds: List[List[Float64]],
        var n_bins_per_feature: List[Int],
    ):
        """Initializes a BinnedMatrix with owned buffer and threshold metadata.

        Args:
            rows: Number of sample rows $N$.
            cols: Number of feature columns $D$.
            data: Raw $N \\times D$ array of UInt8 bin identifiers.
            bin_thresholds: Discovered partition thresholds per feature.
            n_bins_per_feature: Total bin counts per feature.
        """
        self.rows = rows
        self.cols = cols
        self.data = data^
        self.bin_thresholds = bin_thresholds^
        self.n_bins_per_feature = n_bins_per_feature^

    def __init__(out self, *, copy: Self):
        """Deep copies an existing BinnedMatrix."""
        self.rows = copy.rows
        self.cols = copy.cols
        self.data = copy.data.copy()
        self.n_bins_per_feature = copy.n_bins_per_feature.copy()
        self.bin_thresholds = List[List[Float64]](
            capacity=len(copy.bin_thresholds)
        )
        for j in range(len(copy.bin_thresholds)):
            self.bin_thresholds.append(copy.bin_thresholds[j].copy())

    def get(self, r: Int, c: Int) -> UInt8:
        """Retrieves the bin index for sample $r$ and feature $c$.

        Args:
            r: Row index ($0 \\le r < N$).
            c: Column index ($0 \\le c < D$).

        Returns:
            UInt8: Bin integer in $[0, K_c - 1]$.
        """
        return self.data[r * self.cols + c]

    def set(mut self, r: Int, c: Int, val: UInt8):
        """Sets the bin index for sample $r$ and feature $c$.

        Args:
            r: Row index.
            c: Column index.
            val: UInt8 bin value.
        """
        self.data[r * self.cols + c] = val

    def row(self, r: Int) -> List[UInt8]:
        """Extracts a copy of the $D$-dimensional binned row vector.

        Args:
            r: Row index ($0 \\le r < N$).

        Returns:
            List[UInt8]: Vector of length $D$.
        """
        var res = List[UInt8](capacity=self.cols)
        var offset = r * self.cols
        for c in range(self.cols):
            res.append(self.data[offset + c])
        return res^

    def col(self, c: Int) -> List[UInt8]:
        """Extracts a copy of the $N$-dimensional binned column vector.

        Args:
            c: Column index ($0 \\le c < D$).

        Returns:
            List[UInt8]: Vector of length $N$.
        """
        var res = List[UInt8](capacity=self.rows)
        for r in range(self.rows):
            res.append(self.data[r * self.cols + c])
        return res^


    def unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns the raw unsafe pointer to the underlying contiguous UInt8 buffer."""
        return self.data.unsafe_ptr()


def _find_bin_idx(val: Float64, thresholds: List[Float64]) -> UInt8:
    """Finds the bin index for a scalar value using binary search over thresholds.

    Given monotonic thresholds $t_0 < t_1 < \\dots < t_{K-2}$:
    - Returns $0$ if $\\text{val} \\le t_0$.
    - Returns $k$ if $t_{k-1} < \\text{val} \\le t_k$.
    - Returns $K - 1$ if $\\text{val} > t_{K-2}$.

    Args:
        val: Continuous feature value.
        thresholds: Sorted array of threshold split points.

    Returns:
        UInt8: Bin index in $[0, \\text{len}(thresholds)]$.
    """
    var low = 0
    var high = len(thresholds)
    while low < high:
        var mid = (low + high) // 2
        if val <= thresholds[mid]:
            high = mid
        else:
            low = mid + 1
    return UInt8(low)


def _compute_bin_thresholds[
    dtype: DType = DType.float64
](X: Matrix[dtype], max_bins: Int = 256) raises -> List[List[Float64]]:
    """Calculates quantile bin boundary thresholds for each column in $X$.

    Parameters:
        dtype: Data type of the feature matrix.

    Args:
        X: Feature matrix of shape $(N, D)$.
        max_bins: Maximum number of discrete bins ($2 \\le \\text{max\\_bins} \\le 256$).

    Returns:
        List[List[Float64]]: List of length $D$ containing sorted threshold vectors.

    Raises:
        InvalidParameterError: If max_bins < 2 or max_bins > 256.
    """
    if max_bins < 2 or max_bins > 256:
        raise InvalidParameterError.error(
            "max_bins",
            "max_bins must be between 2 and 256, got " + String(max_bins),
        )

    var N = X.rows
    var D = X.cols
    var all_thresholds = List[List[Float64]](capacity=D)

    if N == 0 or D == 0:
        for _ in range(D):
            all_thresholds.append(List[Float64]())
        return all_thresholds^

    for j in range(D):
        var col_vals = List[Float64](capacity=N)
        for i in range(N):
            var v = Float64(X[i, j])
            # Filter NaN if any
            if v == v:
                col_vals.append(v)

        if len(col_vals) == 0:
            all_thresholds.append(List[Float64]())
            continue

        sort(col_vals)

        # Extract unique sorted distinct values
        var distinct = List[Float64](capacity=len(col_vals))
        distinct.append(col_vals[0])
        for i in range(1, len(col_vals)):
            if col_vals[i] != col_vals[i - 1]:
                distinct.append(col_vals[i])

        var M = len(distinct)
        var thresholds = List[Float64]()

        if M <= 1:
            # Constant feature: all samples map to bin 0
            all_thresholds.append(thresholds^)
            continue

        if M <= max_bins:
            # All distinct midpoints become thresholds
            thresholds.reserve(M - 1)
            for k in range(M - 1):
                thresholds.append((distinct[k] + distinct[k + 1]) * 0.5)
        else:
            # Subsample quantiles
            var n_thresholds = max_bins - 1
            thresholds.reserve(n_thresholds)
            for k in range(1, max_bins):
                var idx = Int(
                    round(
                        Float64(k) * Float64(M - 1) / Float64(max_bins)
                    )
                )
                if idx >= M - 1:
                    idx = M - 2
                var t = (distinct[idx] + distinct[idx + 1]) * 0.5
                if len(thresholds) == 0 or t > thresholds[len(thresholds) - 1]:
                    thresholds.append(t)

        all_thresholds.append(thresholds^)

    return all_thresholds^


def _map_to_bins[
    dtype: DType = DType.float64
](
    X: Matrix[dtype],
    bin_thresholds: List[List[Float64]],
) raises -> BinnedMatrix:
    """Discretizes a continuous feature matrix into a compact BinnedMatrix.

    Optimized with column-outer iteration to keep feature threshold arrays
    in high-speed L1 cache throughout column scans.

    Parameters:
        dtype: Data type of the input matrix.

    Args:
        X: Feature matrix of shape $(N, D)$.
        bin_thresholds: Precomputed threshold list of length $D$.

    Returns:
        BinnedMatrix: Flattened UInt8 matrix of bin indices.

    Raises:
        DimensionMismatchError: If len(bin_thresholds) != X.cols.
    """
    var N = X.rows
    var D = X.cols

    if len(bin_thresholds) != D:
        raise DimensionMismatchError.error(
            "len(bin_thresholds) == " + String(D),
            "len(bin_thresholds) == " + String(len(bin_thresholds)),
            "_map_to_bins",
        )

    var n_bins = List[Int](capacity=D)
    for j in range(D):
        n_bins.append(len(bin_thresholds[j]) + 1)

    var binned_data = List[UInt8](capacity=N * D)
    for _ in range(N * D):
        binned_data.append(0)

    var b_ptr = binned_data.unsafe_ptr()

    # Column-outer iteration: bin_thresholds[j] cached across all N samples
    for j in range(D):
        var th = bin_thresholds[j].copy()
        for i in range(N):
            var val = Float64(X[i, j])
            var b_idx = _find_bin_idx(val, th)
            b_ptr.unsafe_offset(i * D + j).unsafe_store(b_idx)

    # Copy thresholds to store on BinnedMatrix
    var thresholds_copy = List[List[Float64]](capacity=D)
    for j in range(D):
        thresholds_copy.append(bin_thresholds[j].copy())

    return BinnedMatrix(
        N, D, binned_data^, thresholds_copy^, n_bins^
    )
