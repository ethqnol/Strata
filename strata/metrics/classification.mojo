from std.math import isnan, log
from ..core.matrix import Matrix
from ..utils.validation import (
    check_array,
    check_consistent_length,
    check_finite,
)
from ..exceptions.errors import DimensionMismatchError, InvalidParameterError


def _insertion_point(labels: List[Float64], value: Float64) -> Int:
    var lo = 0
    var hi = len(labels)
    while lo < hi:
        var mid = (lo + hi) // 2
        if labels[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _search_sorted(labels: List[Float64], value: Float64) -> Int:
    var i = _insertion_point(labels, value)
    if i < len(labels) and labels[i] == value:
        return i
    return -1


def _insert_label(mut labels: List[Float64], value: Float64):
    var i = _insertion_point(labels, value)
    if i < len(labels) and labels[i] == value:
        return
    labels.insert(i, value)


def unique_labels[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> List[Float64]:
    """Sorted list of the distinct labels appearing in y_true or y_pred."""
    check_finite(y_true, "y_true", "unique_labels")
    check_finite(y_pred, "y_pred", "unique_labels")

    var labels = List[Float64]()
    for i in range(len(y_true)):
        _insert_label(labels, Float64(y_true[i]))
    for i in range(len(y_pred)):
        _insert_label(labels, Float64(y_pred[i]))
    return labels^


def _check_classification_targets[
    true_dtype: DType, pred_dtype: DType
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    caller: String,
) raises:
    check_consistent_length(y_true, y_pred)
    if len(y_true) == 0:
        raise InvalidParameterError.error(
            "y_true", caller + " requires at least one sample"
        )
    check_finite(y_true, "y_true", caller)
    check_finite(y_pred, "y_pred", caller)


def accuracy_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    normalize: Bool = True,
) raises -> Float64:
    """Fraction (or count, if normalize is False) of correctly classified samples.
    """
    _check_classification_targets(y_true, y_pred, "accuracy_score")

    var correct: Int = 0
    for i in range(len(y_true)):
        if Float64(y_true[i]) == Float64(y_pred[i]):
            correct += 1

    if not normalize:
        return Float64(correct)
    return Float64(correct) / Float64(len(y_true))


def confusion_matrix[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]
) raises -> Matrix[DType.int64]:
    """Confusion matrix C where C[i, j] counts samples of label i predicted as label j.

    Rows and columns are indexed by the sorted distinct labels.
    """
    _check_classification_targets(y_true, y_pred, "confusion_matrix")

    var labels = unique_labels(y_true, y_pred)
    var cm = Matrix[DType.int64](len(labels), len(labels), 0)
    for i in range(len(y_true)):
        var r = _search_sorted(labels, Float64(y_true[i]))
        var c = _search_sorted(labels, Float64(y_pred[i]))
        cm[r, c] = cm[r, c] + 1
    return cm^


def _numerator(tp: Float64, which: Int) -> Float64:
    if which == 2:
        return 2.0 * tp
    return tp


def _denominator(
    tp: Float64, fp: Float64, false_neg: Float64, which: Int
) -> Float64:
    if which == 0:
        return tp + fp
    if which == 1:
        return tp + false_neg
    return 2.0 * tp + fp + false_neg


def _averaged_score[
    true_dtype: DType, pred_dtype: DType
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String,
    pos_label: Float64,
    zero_division: Float64,
    which: Int,
    caller: String,
) raises -> Float64:
    _check_classification_targets(y_true, y_pred, caller)

    var labels = unique_labels(y_true, y_pred)
    var k = len(labels)
    var n = len(y_true)

    var tp = List[Float64](length=k, fill=0.0)
    var fp = List[Float64](length=k, fill=0.0)
    var false_neg = List[Float64](length=k, fill=0.0)
    var support = List[Float64](length=k, fill=0.0)

    for i in range(n):
        var t = _search_sorted(labels, Float64(y_true[i]))
        var p = _search_sorted(labels, Float64(y_pred[i]))
        support[t] += 1.0
        if t == p:
            tp[t] += 1.0
        else:
            fp[p] += 1.0
            false_neg[t] += 1.0

    if average == "binary":
        if k > 2:
            raise InvalidParameterError.error(
                "average",
                "'binary' is only supported for binary targets, but "
                + String(k)
                + " labels were found. Use 'micro', 'macro' or 'weighted'.",
            )
        var idx = _search_sorted(labels, pos_label)
        if idx < 0:
            if k >= 2:
                raise InvalidParameterError.error(
                    "pos_label",
                    String(pos_label) + " is not present in y_true or y_pred",
                )
            return zero_division
        var den = _denominator(tp[idx], fp[idx], false_neg[idx], which)
        if den == 0.0:
            return zero_division
        return _numerator(tp[idx], which) / den

    if average == "micro":
        var tp_sum: Float64 = 0.0
        var fp_sum: Float64 = 0.0
        var fn_sum: Float64 = 0.0
        for i in range(k):
            tp_sum += tp[i]
            fp_sum += fp[i]
            fn_sum += false_neg[i]
        var micro_den = _denominator(tp_sum, fp_sum, fn_sum, which)
        if micro_den == 0.0:
            return zero_division
        return _numerator(tp_sum, which) / micro_den

    if average != "macro" and average != "weighted":
        raise InvalidParameterError.error(
            "average",
            "'"
            + average
            + "' is not supported. Use 'binary', 'micro', 'macro' or"
            " 'weighted'.",
        )

    var weighted = average == "weighted"
    var total: Float64 = 0.0
    var weight_total: Float64 = 0.0
    for i in range(k):
        var weight = support[i] if weighted else 1.0
        var den = _denominator(tp[i], fp[i], false_neg[i], which)
        if den == 0.0:
            # A NaN fallback drops ill-defined labels from the average
            if isnan(zero_division):
                continue
            total += zero_division * weight
        else:
            total += (_numerator(tp[i], which) / den) * weight
        weight_total += weight

    if weight_total == 0.0:
        return zero_division
    return total / weight_total


def precision_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
    zero_division: Float64 = 0.0,
) raises -> Float64:
    """Compute classification precision score.

    $$
    \\text{Precision} = \\frac{TP}{TP + FP}
    $$

    Args:
        y_true: Ground truth target labels.
        y_pred: Estimated target labels.
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted'). Default 'binary'.
        pos_label: Label of the positive class when average='binary'. Default 1.0.
        zero_division: Value returned when there is zero division (e.g. 0.0). Default 0.0.

    Returns:
        Float64: Precision score ratio.

    Raises:
        InvalidParameterError: If inputs are empty or average strategy is unrecognized.
    """
    return _averaged_score(
        y_true, y_pred, average, pos_label, zero_division, 0, "precision_score"
    )


def recall_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
    zero_division: Float64 = 0.0,
) raises -> Float64:
    """Compute classification recall (sensitivity) score.

    $$
    \\text{Recall} = \\frac{TP}{TP + FN}
    $$

    Args:
        y_true: Ground truth target labels.
        y_pred: Estimated target labels.
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted'). Default 'binary'.
        pos_label: Label of the positive class when average='binary'. Default 1.0.
        zero_division: Value returned when there is zero division. Default 0.0.

    Returns:
        Float64: Recall score ratio.

    Raises:
        InvalidParameterError: If inputs are empty or average strategy is unrecognized.
    """
    return _averaged_score(
        y_true, y_pred, average, pos_label, zero_division, 1, "recall_score"
    )


def f1_score[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: List[Scalar[pred_dtype]],
    average: String = "binary",
    pos_label: Float64 = 1.0,
    zero_division: Float64 = 0.0,
) raises -> Float64:
    """Compute classification F1 score (harmonic mean of precision and recall).

    $$
    F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2 TP}{2 TP + FP + FN}
    $$


    Args:
        y_true: Ground truth target labels.
        y_pred: Estimated target labels.
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted'). Default 'binary'.
        pos_label: Label of the positive class when average='binary'. Default 1.0.
        zero_division: Value returned when there is zero division. Default 0.0.

    Returns:
        Float64: F1 score between 0.0 and 1.0.

    Raises:
        InvalidParameterError: If inputs are empty or average strategy is unrecognized.
    """
    return _averaged_score(
        y_true, y_pred, average, pos_label, zero_division, 2, "f1_score"
    )


def log_loss[
    true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_pred: Matrix[pred_dtype],
    normalize: Bool = True,
) raises -> Float64:
    """Compute log loss (cross-entropy loss), the negative log-likelihood of true labels.

    $$
    \\text{Log Loss} = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{k=1}^K y_{i, k} \\log(p_{i, k})
    $$

    Args:
        y_true: Ground truth labels, one per sample.
        y_pred: Predicted probabilities, one row per sample and one column per
            class ordered by the sorted distinct labels of y_true. A single
            column is read as the probability of the larger of two labels.
        normalize: Return the mean loss per sample, otherwise the total. Default True.

    Returns:
        Float64: The mean (or total) cross-entropy between y_true and y_pred.

    Raises:
        DimensionMismatchError: If sample count of y_pred does not match y_true or column count does not match class count.
        InvalidParameterError: If inputs are empty, contain fewer than 2 classes, or contain NaN/Inf.
    """
    check_consistent_length(y_pred, y_true)
    if len(y_true) == 0:
        raise InvalidParameterError.error(
            "y_true", "log_loss requires at least one sample"
        )
    check_finite(y_true, "y_true", "log_loss")
    check_array(y_pred)

    var labels = unique_labels(y_true, y_true)
    var k = len(labels)
    if k == 1:
        raise InvalidParameterError.error(
            "y_true",
            "log_loss requires at least 2 distinct labels in y_true, but only "
            + String(labels[0])
            + " was found",
        )

    var binarized = y_pred.cols == 1
    var n_cols = 2 if binarized else y_pred.cols
    if n_cols != k:
        raise DimensionMismatchError.error(
            "y_pred with " + String(k) + " columns",
            "y_pred with " + String(y_pred.cols) + " columns",
            "log_loss",
        )

    var eps = 2.220446049250313e-16
    var probs = List[Float64](length=n_cols, fill=0.0)
    var total: Float64 = 0.0

    for i in range(len(y_true)):
        var row_sum: Float64 = 0.0
        for j in range(n_cols):
            var p = Float64(y_pred[i, 0]) if binarized else Float64(
                y_pred[i, j]
            )
            if binarized and j == 0:
                p = 1.0 - p
            if p < eps:
                p = eps
            elif p > 1.0 - eps:
                p = 1.0 - eps
            probs[j] = p
            row_sum += p
        var col = _search_sorted(labels, Float64(y_true[i]))
        total -= log(probs[col] / row_sum)

    if not normalize:
        return total
    return total / Float64(len(y_true))


def roc_auc_score[
    true_dtype: DType = DType.float64, score_dtype: DType = DType.float64
](
    y_true: List[Scalar[true_dtype]],
    y_score: List[Scalar[score_dtype]],
    pos_label: Float64 = 1.0,
) raises -> Float64:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    $$
    \\text{ROC AUC} = \\frac{R_1 - \\frac{n_1(n_1 + 1)}{2}}{n_1 n_0}
    $$

    Args:
        y_true: Ground truth labels, one per sample, holding exactly 2 labels.
        y_score: Target scores or probabilities for the positive class.
        pos_label: The label of the positive class. Default 1.0.

    Returns:
        Float64: The area under the ROC curve, between 0.0 and 1.0.

    Raises:
        DimensionMismatchError: If sample count of y_true does not match y_score.
        InvalidParameterError: If y_true does not have exactly 2 distinct classes, pos_label is missing, or inputs contain NaN/Inf.
    """
    _check_classification_targets(y_true, y_score, "roc_auc_score")

    var labels = unique_labels(y_true, y_true)
    var k = len(labels)
    if k != 2:
        raise InvalidParameterError.error(
            "y_true",
            "roc_auc_score requires exactly 2 distinct labels in y_true, but "
            + String(k)
            + " were found",
        )
    if _search_sorted(labels, pos_label) < 0:
        raise InvalidParameterError.error(
            "pos_label", String(pos_label) + " is not present in y_true"
        )

    var n = len(y_true)
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)

    @parameter
    def _lower_score(a: Int, b: Int) -> Bool:
        return Float64(y_score[a]) < Float64(y_score[b])

    sort[_lower_score](order)

    var rank_sum: Float64 = 0.0
    var n_pos: Float64 = 0.0
    var start = 0
    while start < n:
        var stop = start
        while stop + 1 < n and Float64(y_score[order[stop + 1]]) == Float64(
            y_score[order[start]]
        ):
            stop += 1
        var mid_rank = Float64(start + stop + 2) / 2.0
        for t in range(start, stop + 1):
            if Float64(y_true[order[t]]) == pos_label:
                rank_sum += mid_rank
                n_pos += 1.0
        start = stop + 1

    var n_neg = Float64(n) - n_pos
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
