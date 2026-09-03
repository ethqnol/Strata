from std.math import sqrt, log, exp
from ..core.matrix import Matrix
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.math import log_sum_exp
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)

comptime LN_2PI: Float64 = 1.8378770664093454835606594728112


struct GaussianNB[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable, Serializable):
    """Gaussian Naive Bayes classifier for continuous feature modeling.

    Assumes that the likelihood of continuous features within each class is
    normally distributed:

    $$
    P(x_j \\mid y = c) = \\frac{1}{\\sqrt{2\\pi \\sigma_{c, j}^2}} \\exp\\left(-\\frac{(x_j - \\mu_{c, j})^2}{2\\sigma_{c, j}^2}\\right)
    $$

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        var_smoothing: Portion of the largest variance of all features that is
            added to variances for calculation stability. Default 1e-9.
        priors: Prior probabilities of the classes. If specified, the priors are
            not adjusted according to the data. Default empty (empirical priors).

    Attributes:
        classes_: Unique class labels observed during fit.
        class_count_: Number of training samples observed in each class.
        class_prior_: Probability of each class.
        theta_: Mean of each feature per class of shape $(K, D)$.
        var_: Variance of each feature per class with variance smoothing added,
            of shape $(K, D)$.
        epsilon_: Absolute additive variance smoothing factor.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.naive_bayes import GaussianNB
        from strata.core import Matrix

        var gnb = GaussianNB[DType.float64](var_smoothing=1e-9)
        gnb.fit(X_train, y_train)
        var preds = gnb.predict(X_test)
        var probs = gnb.predict_proba(X_test)
        ```
    """

    var is_fitted: Bool
    var var_smoothing: Scalar[Self.compute_dtype]
    var priors: List[Scalar[Self.compute_dtype]]
    var classes_: List[Int]
    var class_count_: List[Int]
    var class_prior_: List[Scalar[Self.compute_dtype]]
    var theta_: Matrix[Self.compute_dtype]
    var var_: Matrix[Self.compute_dtype]
    var epsilon_: Scalar[Self.compute_dtype]

    def __init__(
        out self,
        var_smoothing: Scalar[Self.compute_dtype] = 1e-9,
        priors: List[Scalar[Self.compute_dtype]] = List[
            Scalar[Self.compute_dtype]
        ](),
    ) raises:
        """Initialize the GaussianNB classifier.

        Args:
            var_smoothing: Portion of maximum variance added for numerical stability.
                Must be non-negative. Default 1e-9.
            priors: Prior probabilities of the classes. Default empty.

        Raises:
            InvalidParameterError: If var_smoothing is negative.
        """
        check_floating_dtype[Self.compute_dtype, "GaussianNB"]()
        if var_smoothing < 0:
            raise InvalidParameterError.error(
                "var_smoothing",
                "var_smoothing must be non-negative, got "
                + String(var_smoothing),
            )
        self.is_fitted = False
        self.var_smoothing = var_smoothing
        self.priors = priors.copy()
        self.classes_ = List[Int]()
        self.class_count_ = List[Int]()
        self.class_prior_ = List[Scalar[Self.compute_dtype]]()
        self.theta_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.var_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.epsilon_ = 0

    def __init__(out self, *, copy: Self):
        """Copies an existing GaussianNB instance."""
        self.is_fitted = copy.is_fitted
        self.var_smoothing = copy.var_smoothing
        self.priors = copy.priors.copy()
        self.classes_ = copy.classes_.copy()
        self.class_count_ = copy.class_count_.copy()
        self.class_prior_ = copy.class_prior_.copy()
        self.theta_ = copy.theta_.copy()
        self.var_ = copy.var_.copy()
        self.epsilon_ = copy.epsilon_

    def __init__(out self, *, deinit move: Self):
        """Moves an existing GaussianNB instance."""
        self.is_fitted = move.is_fitted
        self.var_smoothing = move.var_smoothing
        self.priors = move.priors^
        self.classes_ = move.classes_^
        self.class_count_ = move.class_count_^
        self.class_prior_ = move.class_prior_^
        self.theta_ = move.theta_^
        self.var_ = move.var_^
        self.epsilon_ = move.epsilon_

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit Gaussian Naive Bayes according to X, y.

        Args:
            X: Training feature matrix of shape $(N, D)$.
            y: Target class labels of length $N$.

        Raises:
            InvalidParameterError: If classes < 2 or prior configuration is invalid.
            DimensionMismatchError: If sample count of X does not match length of y.
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols

        # Extract unique classes and sort ascending
        var raw_classes = List[Int]()
        for i in range(len(y)):
            var label = Int(y[i])
            var found = False
            for c_idx in range(len(raw_classes)):
                if raw_classes[c_idx] == label:
                    found = True
                    break
            if not found:
                raw_classes.append(label)

        for i in range(len(raw_classes)):
            for j in range(i + 1, len(raw_classes)):
                if raw_classes[j] < raw_classes[i]:
                    var temp = raw_classes[i]
                    raw_classes[i] = raw_classes[j]
                    raw_classes[j] = temp

        var K = len(raw_classes)
        if K < 2:
            raise InvalidParameterError.error(
                "y",
                "GaussianNB requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

        # Map each sample target to class index [0, K)
        var y_idx = List[Int](capacity=N)
        var class_counts = List[Int](capacity=K)
        for _ in range(K):
            class_counts.append(0)

        for i in range(N):
            var target_val = Int(y[i])
            var matched_k = 0
            for k in range(K):
                if self.classes_[k] == target_val:
                    matched_k = k
                    break
            y_idx.append(matched_k)
            class_counts[matched_k] += 1

        self.class_count_ = class_counts^

        # Verify all classes have samples
        for k in range(K):
            if self.class_count_[k] == 0:
                raise InvalidParameterError.error(
                    "y",
                    "Class "
                    + String(self.classes_[k])
                    + " has zero training samples.",
                )

        var X_comp = X.cast[Self.compute_dtype]()
        var n_samples_scalar = Scalar[Self.compute_dtype](N)

        # Calculate global variance across all samples for adaptive variance smoothing
        var col_sums = List[Scalar[Self.compute_dtype]](capacity=D)
        for _ in range(D):
            col_sums.append(0.0)

        for i in range(N):
            for j in range(D):
                col_sums[j] += X_comp[i, j]

        var col_means = List[Scalar[Self.compute_dtype]](capacity=D)
        for j in range(D):
            col_means.append(col_sums[j] / n_samples_scalar)

        var sq_diff_sums = List[Scalar[Self.compute_dtype]](capacity=D)
        for _ in range(D):
            sq_diff_sums.append(0.0)

        for i in range(N):
            for j in range(D):
                var d = X_comp[i, j] - col_means[j]
                sq_diff_sums[j] += d * d

        var max_global_var: Scalar[Self.compute_dtype] = 0.0
        for j in range(D):
            var feat_var = sq_diff_sums[j] / n_samples_scalar
            if feat_var > max_global_var:
                max_global_var = feat_var

        self.epsilon_ = self.var_smoothing * max_global_var
        if self.epsilon_ <= 0:
            self.epsilon_ = Scalar[Self.compute_dtype](1e-9)

        # Calculate per-class mean in a single cache-friendly pass over N
        var class_sums = Matrix[Self.compute_dtype](K, D, 0)
        for i in range(N):
            var k = y_idx[i]
            for j in range(D):
                class_sums[k, j] += X_comp[i, j]

        self.theta_ = Matrix[Self.compute_dtype](K, D, 0)
        for k in range(K):
            var n_k = Scalar[Self.compute_dtype](self.class_count_[k])
            for j in range(D):
                self.theta_[k, j] = class_sums[k, j] / n_k

        # Calculate per-class variance in a second cache-friendly pass over N
        var class_sq_diffs = Matrix[Self.compute_dtype](K, D, 0)
        for i in range(N):
            var k = y_idx[i]
            for j in range(D):
                var diff = X_comp[i, j] - self.theta_[k, j]
                class_sq_diffs[k, j] += diff * diff

        self.var_ = Matrix[Self.compute_dtype](K, D, 0)
        for k in range(K):
            var n_k = Scalar[Self.compute_dtype](self.class_count_[k])
            for j in range(D):
                self.var_[k, j] = (class_sq_diffs[k, j] / n_k) + self.epsilon_

        # Configure class priors
        self.class_prior_ = List[Scalar[Self.compute_dtype]](capacity=K)
        if len(self.priors) > 0:
            if len(self.priors) != K:
                raise InvalidParameterError.error(
                    "priors",
                    "Number of priors ("
                    + String(len(self.priors))
                    + ") must match number of classes ("
                    + String(K)
                    + ")",
                )
            var p_sum: Scalar[Self.compute_dtype] = 0.0
            for k in range(K):
                if self.priors[k] < 0:
                    raise InvalidParameterError.error(
                        "priors",
                        "Class prior at index "
                        + String(k)
                        + " must be non-negative.",
                    )
                p_sum += self.priors[k]
            var diff_from_one = Float64(p_sum) - 1.0 if Float64(
                p_sum
            ) >= 1.0 else 1.0 - Float64(p_sum)
            if diff_from_one > 1e-3:
                raise InvalidParameterError.error(
                    "priors",
                    "Sum of class priors must equal 1.0, got " + String(p_sum),
                )
            self.class_prior_ = self.priors.copy()
        else:
            for k in range(K):
                self.class_prior_.append(
                    Scalar[Self.compute_dtype](self.class_count_[k])
                    / n_samples_scalar
                )

        self.is_fitted = True

    def _joint_log_likelihood[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Calculates unnormalized joint log likelihood log P(y=c) + log P(x | y=c).
        """
        check_is_fitted("GaussianNB", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.theta_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "GaussianNB._joint_log_likelihood",
            )

        var M = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()
        var joint_ll = Matrix[Self.compute_dtype](M, K, 0)

        # Precompute per-class constants and inverse 2*variance to replace division with multiplication
        var base_log = List[Float64](capacity=K)
        var inv_two_var = Matrix[DType.float64](K, D, 0)
        for k in range(K):
            var prior_val = Float64(self.class_prior_[k])
            var log_prior = log(prior_val) if prior_val > 1e-15 else -34.538776
            var c_k: Float64 = 0.0
            for j in range(D):
                var v = Float64(self.var_[k, j])
                c_k += LN_2PI + log(v)
                inv_two_var[k, j] = 1.0 / (2.0 * v)
            base_log.append(log_prior - 0.5 * c_k)

        # Fast evaluation using precomputed reciprocal variance factors
        for i in range(M):
            for k in range(K):
                var quad: Float64 = 0.0
                for j in range(D):
                    var diff = Float64(X_comp[i, j]) - Float64(
                        self.theta_[k, j]
                    )
                    quad += (diff * diff) * inv_two_var[k, j]
                joint_ll[i, k] = Scalar[Self.compute_dtype](base_log[k] - quad)

        return joint_ll^

    def predict_log_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class log probabilities for samples in X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Log-probability matrix of shape $(N, K)$.
        """
        var joint_ll = self._joint_log_likelihood(X)
        var M = joint_ll.rows
        var K = joint_ll.cols
        var log_probs = Matrix[feat_dtype](M, K, 0)

        for i in range(M):
            var row_vals = List[Scalar[Self.compute_dtype]](capacity=K)
            for k in range(K):
                row_vals.append(joint_ll[i, k])
            var lse = log_sum_exp[Self.compute_dtype](row_vals)
            for k in range(K):
                log_probs[i, k] = Scalar[feat_dtype](joint_ll[i, k] - lse)

        return log_probs^

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probability distributions for samples in X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Normalized probability matrix of shape $(N, K)$,
                where each row sums to 1.0.
        """
        var log_probs = self.predict_log_proba(X)
        var M = log_probs.rows
        var K = log_probs.cols
        var probs = Matrix[feat_dtype](M, K, 0)

        for i in range(M):
            var row_sum: Float64 = 0.0
            for k in range(K):
                var p = exp(Float64(log_probs[i, k]))
                row_sum += p
                probs[i, k] = Scalar[feat_dtype](p)
            if row_sum > 0:
                var inv_sum = 1.0 / row_sum
                for k in range(K):
                    probs[i, k] = Scalar[feat_dtype](
                        Float64(probs[i, k]) * inv_sum
                    )

        return probs^

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for samples in X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            List[Int]: Predicted class labels of length $N$.
        """
        var joint_ll = self._joint_log_likelihood(X)
        var M = joint_ll.rows
        var K = joint_ll.cols
        var preds = List[Int](capacity=M)

        for i in range(M):
            var best_k = 0
            var max_val = joint_ll[i, 0]
            for k in range(1, K):
                if joint_ll[i, k] > max_val:
                    max_val = joint_ll[i, k]
                    best_k = k
            preds.append(self.classes_[best_k])

        return preds^

    def serialize(self, mut writer: BufferWriter):
        """Serializes GaussianNB state into BufferWriter."""
        write_header(writer, "GaussianNB")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.var_smoothing.cast[DType.float64]())
        writer.write_float64(self.epsilon_.cast[DType.float64]())
        writer.write_float_list[Self.compute_dtype](self.priors)
        writer.write_int_list(self.classes_)
        writer.write_int_list(self.class_count_)
        writer.write_float_list[Self.compute_dtype](self.class_prior_)
        writer.write_matrix[Self.compute_dtype](self.theta_)
        writer.write_matrix[Self.compute_dtype](self.var_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes GaussianNB from BufferReader."""
        check_header(reader, "GaussianNB")
        var is_fitted = reader.read_bool()
        var var_smoothing = Scalar[Self.compute_dtype](reader.read_float64())
        var epsilon_ = Scalar[Self.compute_dtype](reader.read_float64())
        var priors = reader.read_float_list[Self.compute_dtype]()
        var classes_ = reader.read_int_list()
        var class_count_ = reader.read_int_list()
        var class_prior_ = reader.read_float_list[Self.compute_dtype]()
        var theta_ = reader.read_matrix[Self.compute_dtype]()
        var var_ = reader.read_matrix[Self.compute_dtype]()

        var model = Self(
            var_smoothing=var_smoothing,
            priors=priors,
        )
        model.is_fitted = is_fitted
        model.epsilon_ = epsilon_
        model.classes_ = classes_^
        model.class_count_ = class_count_^
        model.class_prior_ = class_prior_^
        model.theta_ = theta_^
        model.var_ = var_^
        return model^
