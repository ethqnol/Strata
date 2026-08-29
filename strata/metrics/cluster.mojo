from ..core.matrix import Matrix
from ..cluster._common import _euclidean_distance
from ..utils.validation import check_array, check_consistent_length
from ..exceptions.errors import InvalidParameterError
from .classification import _insert_label, _search_sorted


def silhouette_score[
    dtype: DType = DType.float64
](X: Matrix[dtype], labels: List[Int]) raises -> Float64:
    """Compute the mean Silhouette Coefficient of all samples.

    $$
    s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}
    $$

    where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance.
    Samples alone in their cluster score 0.0.

    Args:
        X: Feature matrix with one row per sample.
        labels: Predicted cluster labels for each sample.

    Returns:
        Float64: Mean Silhouette Coefficient between -1.0 and 1.0.

    Raises:
        DimensionMismatchError: If sample count of X does not match length of labels.
        InvalidParameterError: If number of distinct labels is less than 2 or greater than n_samples - 1, or if inputs contain NaN/Inf.
    """
    check_array(X)
    check_consistent_length(X, labels)

    var n = X.rows
    var unique = List[Float64]()
    for i in range(n):
        _insert_label(unique, Float64(labels[i]))

    var k = len(unique)
    if k < 2 or k > n - 1:
        raise InvalidParameterError.error(
            "labels",
            "silhouette_score requires 2 to n_samples - 1 distinct labels, but "
            + String(k)
            + " were found for "
            + String(n)
            + " samples",
        )

    var encoded = List[Int](capacity=n)
    var sizes = List[Float64](length=k, fill=0.0)
    for i in range(n):
        var c = _search_sorted(unique, Float64(labels[i]))
        encoded.append(c)
        sizes[c] += 1.0

    var X_comp = X.cast[DType.float64]()
    var sums = Matrix[DType.float64](n, k, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            var d = Float64(_euclidean_distance(X_comp, i, X_comp, j))
            sums[i, encoded[j]] += d
            sums[j, encoded[i]] += d

    var total: Float64 = 0.0
    for i in range(n):
        var own = encoded[i]
        if sizes[own] == 1.0:
            continue

        var a = sums[i, own] / (sizes[own] - 1.0)
        var b = -1.0
        for c in range(k):
            if c == own:
                continue
            var mean_c = sums[i, c] / sizes[c]
            if b < 0.0 or mean_c < b:
                b = mean_c

        var denom = max(a, b)
        if denom == 0.0:
            continue
        total += (b - a) / denom

    return total / Float64(n)
