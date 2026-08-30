from std.math import abs, exp, max, min, pow, sqrt
from ..utils.math import sigmoid
from ..utils.random import PRNG
from ..core.matrix import Matrix
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


def _soft_threshold_scalar[
    dtype: DType = DType.float64
](z: Scalar[dtype], gamma: Scalar[dtype]) -> Scalar[dtype]:
    """Applies soft-thresholding to a scalar."""
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0


def _compute_eta[
    dtype: DType = DType.float64
](
    learning_rate: String,
    eta0: Scalar[dtype],
    power_t: Scalar[dtype],
    alpha: Scalar[dtype],
    t: Int,
    t0: Scalar[dtype] = 1.0,
) -> Scalar[dtype]:
    """Computes the step size at global step t according to the chosen schedule.
    """
    if learning_rate == "constant":
        return eta0
    elif learning_rate == "invscaling":
        var denom = pow(Scalar[dtype](t + 1), power_t)
        return eta0 / denom
    elif learning_rate == "optimal":
        var denom = alpha * (Scalar[dtype](t) + t0)
        if denom <= 0:
            return eta0
        return Scalar[dtype](1.0) / denom
    elif learning_rate == "adaptive":
        return eta0
    else:
        return eta0


def _dloss_regression[
    dtype: DType = DType.float64
](
    loss: String,
    y_true: Scalar[dtype],
    y_pred: Scalar[dtype],
    epsilon: Scalar[dtype],
) -> Scalar[dtype]:
    """Derivative of regression loss with respect to y_pred."""
    var diff = y_pred - y_true
    if loss == "squared_error":
        return diff
    elif loss == "huber":
        var abs_diff = abs(diff)
        if abs_diff <= epsilon:
            return diff
        elif diff > 0:
            return epsilon
        else:
            return -epsilon
    elif loss == "epsilon_insensitive":
        var abs_diff = abs(diff)
        if abs_diff <= epsilon:
            return 0
        elif diff > 0:
            return 1
        else:
            return -1
    else:
        return diff


def _dloss_classification[
    dtype: DType = DType.float64
](
    loss: String,
    y_true: Scalar[dtype],
    raw_score: Scalar[dtype],
    epsilon: Scalar[dtype] = 0.1,
) -> Scalar[dtype]:
    """Derivative of classification loss with respect to raw linear score.

    y_true is assumed to be +1 or -1.
    """
    var margin = y_true * raw_score
    if loss == "hinge":
        if margin < 1.0:
            return -y_true
        else:
            return 0
    elif loss == "log_loss" or loss == "log":
        # d/d(raw_score) ln(1 + e^(-y*raw_score)) = -y * sigma(-y * raw_score)
        var sig = sigmoid[dtype](-margin)
        return -y_true * sig
    elif loss == "modified_huber":
        if margin >= 1.0:
            return 0
        elif margin >= -1.0:
            return -Scalar[dtype](2.0) * y_true * (Scalar[dtype](1.0) - margin)
        else:
            return -Scalar[dtype](4.0) * y_true
    elif loss == "squared_hinge":
        if margin < 1.0:
            return -Scalar[dtype](2.0) * y_true * (Scalar[dtype](1.0) - margin)
        else:
            return 0
    else:
        if margin < 1.0:
            return -y_true
        else:
            return 0


def _apply_penalty_step[
    dtype: DType = DType.float64
](
    mut w: List[Scalar[dtype]],
    penalty: String,
    alpha: Scalar[dtype],
    l1_ratio: Scalar[dtype],
    eta: Scalar[dtype],
):
    """Applies weight decay and proximal shrinkage to the weight vector w."""
    var D = len(w)
    if penalty == "none" or alpha == 0:
        return
    elif penalty == "l2":
        var decay = Scalar[dtype](1.0) - eta * alpha
        if decay < 0:
            decay = 0
        for j in range(D):
            w[j] *= decay
    elif penalty == "l1":
        var threshold = eta * alpha
        for j in range(D):
            w[j] = _soft_threshold_scalar[dtype](w[j], threshold)
    elif penalty == "elasticnet":
        var l1_strength = alpha * l1_ratio
        var l2_strength = alpha * (Scalar[dtype](1.0) - l1_ratio)
        var decay = Scalar[dtype](1.0) - eta * l2_strength
        if decay < 0:
            decay = 0
        var threshold = eta * l1_strength
        for j in range(D):
            var shrunk = w[j] * decay
            w[j] = _soft_threshold_scalar[dtype](shrunk, threshold)
