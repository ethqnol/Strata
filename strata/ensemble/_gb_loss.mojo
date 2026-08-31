from std.math import exp, log, log1p, max, min
from ..utils.math import sigmoid, softmax
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError


struct LeastSquaresLoss(Copyable, Movable):
    """Least-squares loss for regression gradient boosting: $L(y, \\hat{y}) = \\frac{1}{2}(y - \\hat{y})^2$.
    """

    def __init__(out self):
        """Initializes the LeastSquaresLoss instance."""
        pass

    def __init__(out self, *, copy: Self):
        """Copies a LeastSquaresLoss instance."""
        pass

    def loss(self, y_true: Float64, raw_pred: Float64) -> Float64:
        """Evaluates single-sample scalar squared error loss."""
        var diff = raw_pred - y_true
        return 0.5 * diff * diff

    def gradient(self, y_true: Float64, raw_pred: Float64) -> Float64:
        """Computes 1st order derivative $\\frac{\\partial L}{\\partial \\hat{y}} = \\hat{y} - y$.
        """
        return raw_pred - y_true

    def hessian(self, y_true: Float64, raw_pred: Float64) -> Float64:
        """Computes 2nd order derivative $\\frac{\\partial^2 L}{\\partial \\hat{y}^2} = 1.0$.
        """
        return 1.0

    def init_raw_prediction(self, y: List[Float64]) -> Float64:
        """Computes baseline initial prediction as target sample mean $\\bar{y}$.
        """
        var n = len(y)
        if n == 0:
            return 0.0
        var total: Float64 = 0.0
        for i in range(n):
            total += y[i]
        return total / Float64(n)

    def update_gradients_and_hessians(
        self,
        y: List[Float64],
        raw_preds: List[Float64],
        mut gradients: List[Float64],
        mut hessians: List[Float64],
    ):
        """Batched calculation of 1st and 2nd order derivatives."""
        var n = len(y)
        gradients.clear()
        hessians.clear()
        gradients.reserve(n)
        hessians.reserve(n)
        for i in range(n):
            gradients.append(raw_preds[i] - y[i])
            hessians.append(1.0)


struct BinaryCrossEntropyLoss(Copyable, Movable):
    """Binary logistic cross-entropy loss: $L(y, \\hat{y}) = \\ln(1 + e^{\\hat{y}}) - y\\hat{y}$.
    """

    def __init__(out self):
        """Initializes the BinaryCrossEntropyLoss instance."""
        pass

    def __init__(out self, *, copy: Self):
        """Copies a BinaryCrossEntropyLoss instance."""
        pass

    def loss(self, y_true: Float64, raw_pred: Float64) -> Float64:
        """Evaluates single-sample scalar logistic cross-entropy loss."""
        if raw_pred >= 0.0:
            return log1p(exp(-raw_pred)) + (1.0 - y_true) * raw_pred
        else:
            return log1p(exp(raw_pred)) - y_true * raw_pred

    def predict_proba(self, raw_pred: Float64) -> Float64:
        """Maps raw score margin to probability $p = \\sigma(\\hat{y})$."""
        return Float64(sigmoid[DType.float64](raw_pred))

    def gradient(self, y_true: Float64, raw_pred: Float64) -> Float64:
        """Computes 1st order derivative $g = \\sigma(\\hat{y}) - y$."""
        var p = Float64(sigmoid[DType.float64](raw_pred))
        return p - y_true

    def hessian(self, y_true: Float64, raw_pred: Float64) -> Float64:
        """Computes 2nd order derivative $h = p(1 - p)$, bounded below by $10^{-16}$.
        """
        var p = Float64(sigmoid[DType.float64](raw_pred))
        var h = p * (1.0 - p)
        return max(h, 1e-16)

    def init_raw_prediction(self, y: List[Float64]) -> Float64:
        """Computes baseline initial prediction as empirical log-odds $\\ln \\frac{\\bar{y}}{1 - \\bar{y}}$.
        """
        var n = len(y)
        if n == 0:
            return 0.0
        var pos_count: Float64 = 0.0
        for i in range(n):
            if y[i] > 0.5:
                pos_count += 1.0

        var mean_y = pos_count / Float64(n)
        # Clamp to avoid log(0)
        mean_y = min(max(mean_y, 1e-7), 1.0 - 1e-7)
        return log(mean_y / (1.0 - mean_y))

    def update_gradients_and_hessians(
        self,
        y: List[Float64],
        raw_preds: List[Float64],
        mut gradients: List[Float64],
        mut hessians: List[Float64],
    ):
        """Batched calculation of logistic gradients and hessians."""
        var n = len(y)
        gradients.clear()
        hessians.clear()
        gradients.reserve(n)
        hessians.reserve(n)
        for i in range(n):
            var p = Float64(sigmoid[DType.float64](raw_preds[i]))
            gradients.append(p - y[i])
            var h = p * (1.0 - p)
            hessians.append(max(h, 1e-16))


struct MulticlassCrossEntropyLoss(Copyable, Movable):
    """Categorical cross-entropy loss with Softmax probabilities for $K$-class classification.
    """

    var n_classes: Int

    def __init__(out self, n_classes: Int):
        """Initializes the MulticlassCrossEntropyLoss instance with $K$ classes.
        """
        self.n_classes = n_classes

    def __init__(out self, *, copy: Self):
        """Copies a MulticlassCrossEntropyLoss instance."""
        self.n_classes = copy.n_classes

    def predict_proba(self, raw_preds: List[Float64]) -> List[Float64]:
        """Calculates normalized Softmax probability distribution over all $K$ classes.
        """
        var raw_scalars = List[Scalar[DType.float64]](capacity=len(raw_preds))
        for k in range(len(raw_preds)):
            raw_scalars.append(Scalar[DType.float64](raw_preds[k]))
        var p_scalars = softmax[DType.float64](raw_scalars)
        var probs = List[Float64](capacity=len(p_scalars))
        for k in range(len(p_scalars)):
            probs.append(Float64(p_scalars[k]))
        return probs^

    def loss(self, y_true: Int, raw_preds: List[Float64]) -> Float64:
        """Evaluates negative log-likelihood $-\\ln p_{y_{\\text{true}}}$ for a single sample.
        """
        var probs = self.predict_proba(raw_preds)
        if y_true < 0 or y_true >= len(probs):
            return 0.0
        var p_y = max(probs[y_true], 1e-16)
        return -log(p_y)

    def init_raw_predictions(self, y: List[Int]) -> List[Float64]:
        """Computes initial baseline log-prior margins $\\ln \\frac{\\text{count}(k)}{N}$ per class.
        """
        var counts = List[Float64](capacity=self.n_classes)
        for _ in range(self.n_classes):
            counts.append(0.0)

        var n = len(y)
        for i in range(n):
            var c = y[i]
            if c >= 0 and c < self.n_classes:
                counts[c] += 1.0

        var init_preds = List[Float64](capacity=self.n_classes)
        var n_f = Float64(max(n, 1))
        for k in range(self.n_classes):
            var prior = max(counts[k] / n_f, 1e-7)
            init_preds.append(log(prior))

        return init_preds^

    def update_gradients_and_hessians_for_class(
        self,
        y: List[Int],
        raw_preds_all: List[Float64],
        class_k: Int,
        mut gradients: List[Float64],
        mut hessians: List[Float64],
    ):
        """Computes gradients and hessians for target class tree $k$ across all $N$ samples.

        `raw_preds_all` contains the flattened $N \\times K$ margin values.
        """
        var n = len(y)
        var K = self.n_classes
        gradients.clear()
        hessians.clear()
        gradients.reserve(n)
        hessians.reserve(n)

        var raw_ptr = raw_preds_all.unsafe_ptr()

        for i in range(n):
            var offset = i * K
            var max_logit = raw_ptr.unsafe_offset(offset).unsafe_load()
            for k in range(1, K):
                var val = raw_ptr.unsafe_offset(offset + k).unsafe_load()
                if val > max_logit:
                    max_logit = val

            var sum_exp: Float64 = 0.0
            for k in range(K):
                sum_exp += exp(
                    raw_ptr.unsafe_offset(offset + k).unsafe_load() - max_logit
                )

            var target_exp = exp(
                raw_ptr.unsafe_offset(offset + class_k).unsafe_load()
                - max_logit
            )
            var p_k = target_exp / sum_exp
            var indicator: Float64 = 1.0 if y[i] == class_k else 0.0

            gradients.append(p_k - indicator)
            var h = p_k * (1.0 - p_k)
            hessians.append(max(h, 1e-16))
