from std.math import log, exp
from ..core.matrix import Matrix
from ..core.csr_matrix import CSRMatrix
from ..core.linalg import gemm
from ..core.sparse_ops import spmm
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


struct MultinomialNB[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable, Serializable):
    """Multinomial Naive Bayes classifier for multinomially distributed count data.

    Suitable for discrete features (e.g. word counts for text classification):

    $$
    P(x \\mid y = c) \\propto \\prod_{j=1}^{D} \\theta_{c, j}^{x_j}
    $$

    where the smoothed feature probabilities are:

    $$
    \\theta_{c, j} = \\frac{N_{c, j} + \\alpha}{N_c + \\alpha \\cdot D}
    $$

    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        alpha: Additive (Laplace/Lidstone) smoothing parameter. Must be non-negative.
            Default 1.0.
        fit_prior: Whether to learn class prior probabilities or assume a uniform
            prior. Default True.
        class_prior: Prior probabilities of the classes. If specified, the priors
            are not adjusted according to the data. Default empty.

    Attributes:
        classes_: Unique class labels observed during fit.
        class_count_: Number of training samples observed in each class.
        class_log_prior_: Smoothed empirical log-probability of each class.
        feature_count_: Number of samples encountered for each (class, feature)
            pair of shape $(K, D)$.
        feature_log_prob_: Empirical log probability of features given a class,
            $\\ln P(x_j \\mid y=c)$, of shape $(K, D)$.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.naive_bayes import MultinomialNB
        from strata.core import Matrix

        var mnb = MultinomialNB[DType.float64](alpha=1.0)
        mnb.fit(X_train, y_train)
        var preds = mnb.predict(X_test)
        var probs = mnb.predict_proba(X_test)
        ```
    """

    var is_fitted: Bool
    var alpha: Scalar[Self.compute_dtype]
    var fit_prior: Bool
    var class_prior: List[Scalar[Self.compute_dtype]]
    var classes_: List[Int]
    var class_count_: List[Int]
    var class_log_prior_: List[Scalar[Self.compute_dtype]]
    var feature_count_: Matrix[Self.compute_dtype]
    var feature_log_prob_: Matrix[Self.compute_dtype]

    def __init__(
        out self,
        alpha: Scalar[Self.compute_dtype] = 1.0,
        fit_prior: Bool = True,
        class_prior: List[Scalar[Self.compute_dtype]] = List[
            Scalar[Self.compute_dtype]
        ](),
    ) raises:
        """Initialize the MultinomialNB classifier.

        Args:
            alpha: Additive smoothing parameter (>= 0). Default 1.0.
            fit_prior: Whether to learn class prior probabilities. Default True.
            class_prior: Prior probabilities of the classes. Default empty.

        Raises:
            InvalidParameterError: If alpha is negative.
        """
        check_floating_dtype[Self.compute_dtype, "MultinomialNB"]()
        if alpha < 0:
            raise InvalidParameterError.error(
                "alpha", "alpha must be non-negative, got " + String(alpha)
            )
        self.is_fitted = False
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior.copy()
        self.classes_ = List[Int]()
        self.class_count_ = List[Int]()
        self.class_log_prior_ = List[Scalar[Self.compute_dtype]]()
        self.feature_count_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.feature_log_prob_ = Matrix[Self.compute_dtype](0, 0, 0)

    def __init__(out self, *, copy: Self):
        """Copies an existing MultinomialNB instance."""
        self.is_fitted = copy.is_fitted
        self.alpha = copy.alpha
        self.fit_prior = copy.fit_prior
        self.class_prior = copy.class_prior.copy()
        self.classes_ = copy.classes_.copy()
        self.class_count_ = copy.class_count_.copy()
        self.class_log_prior_ = copy.class_log_prior_.copy()
        self.feature_count_ = copy.feature_count_.copy()
        self.feature_log_prob_ = copy.feature_log_prob_.copy()

    def __init__(out self, *, deinit move: Self):
        """Moves an existing MultinomialNB instance."""
        self.is_fitted = move.is_fitted
        self.alpha = move.alpha
        self.fit_prior = move.fit_prior
        self.class_prior = move.class_prior^
        self.classes_ = move.classes_^
        self.class_count_ = move.class_count_^
        self.class_log_prior_ = move.class_log_prior_^
        self.feature_count_ = move.feature_count_^
        self.feature_log_prob_ = move.feature_log_prob_^

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit Multinomial Naive Bayes using dense feature matrix X and target labels y.

        Args:
            X: Non-negative training feature matrix of shape $(N, D)$.
            y: Target class labels of length $N$.

        Raises:
            InvalidParameterError: If X contains negative values or classes < 2.
            DimensionMismatchError: If sample count of X does not match length of y.
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols

        # Verify non-negativity of features
        for i in range(N):
            for j in range(D):
                if X[i, j] < 0:
                    raise InvalidParameterError.error(
                        "X",
                        "Negative values in data passed to MultinomialNB at ("
                        + String(i)
                        + ", "
                        + String(j)
                        + ")",
                    )

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
                "MultinomialNB requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

        # Map targets to index
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

        var X_comp = X.cast[Self.compute_dtype]()

        # Accumulate feature counts per class
        self.feature_count_ = Matrix[Self.compute_dtype](K, D, 0)
        for i in range(N):
            var k = y_idx[i]
            for j in range(D):
                self.feature_count_[k, j] += X_comp[i, j]

        self._compute_log_probabilities(N, K, D)
        self.is_fitted = True

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: CSRMatrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fit Multinomial Naive Bayes using sparse CSRMatrix and target labels y.

        Args:
            X: Non-negative sparse CSRMatrix of shape $(N, D)$.
            y: Target class labels of length $N$.

        Raises:
            InvalidParameterError: If X contains negative values or classes < 2.
            DimensionMismatchError: If sample count of X does not match length of y.
        """
        var N = X.rows
        var D = X.cols
        if N != len(y):
            raise DimensionMismatchError.error(
                "X.rows == len(y)",
                "X.rows=" + String(N) + ", len(y)=" + String(len(y)),
                "MultinomialNB.fit",
            )
        if N == 0 or D == 0:
            raise InvalidParameterError.error(
                "X", "Input matrix cannot be empty."
            )

        # Extract unique classes
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
                "MultinomialNB requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

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

        # Accumulate sparse feature counts
        self.feature_count_ = Matrix[Self.compute_dtype](K, D, 0)
        for i in range(N):
            var k = y_idx[i]
            var start_idx = X.indptr[i]
            var end_idx = X.indptr[i + 1]
            for idx in range(start_idx, end_idx):
                var col = X.indices[idx]
                var val = Scalar[Self.compute_dtype](X.data[idx])
                if val < 0:
                    raise InvalidParameterError.error(
                        "X",
                        (
                            "Negative values in sparse data passed to"
                            " MultinomialNB"
                        ),
                    )
                self.feature_count_[k, col] += val

        self._compute_log_probabilities(N, K, D)
        self.is_fitted = True

    def _compute_log_probabilities(mut self, N: Int, K: Int, D: Int) raises:
        """Internal helper to compute smoothed feature log probs and class log priors.
        """
        self.feature_log_prob_ = Matrix[Self.compute_dtype](K, D, 0)

        # Smoothed feature probabilities
        for k in range(K):
            var total_count: Scalar[Self.compute_dtype] = 0.0
            for j in range(D):
                total_count += self.feature_count_[k, j]

            var smoothed_denom = total_count + self.alpha * Scalar[
                Self.compute_dtype
            ](D)
            var log_denom = (
                log(Float64(smoothed_denom)) if Float64(smoothed_denom)
                > 1e-15 else -34.538776
            )

            for j in range(D):
                var smoothed_num = self.feature_count_[k, j] + self.alpha
                if Float64(smoothed_num) > 1e-15:
                    self.feature_log_prob_[k, j] = Scalar[Self.compute_dtype](
                        log(Float64(smoothed_num)) - log_denom
                    )
                else:
                    self.feature_log_prob_[k, j] = Scalar[Self.compute_dtype](
                        -1e9
                    )

        # Class log priors
        self.class_log_prior_ = List[Scalar[Self.compute_dtype]](capacity=K)
        if len(self.class_prior) > 0:
            if len(self.class_prior) != K:
                raise InvalidParameterError.error(
                    "class_prior",
                    "Number of priors ("
                    + String(len(self.class_prior))
                    + ") must match number of classes ("
                    + String(K)
                    + ")",
                )
            var p_sum: Float64 = 0.0
            for k in range(K):
                var p = Float64(self.class_prior[k])
                if p <= 0:
                    raise InvalidParameterError.error(
                        "class_prior", "Class prior must be strictly positive."
                    )
                p_sum += p
            var diff_from_one = p_sum - 1.0 if p_sum >= 1.0 else 1.0 - p_sum
            if diff_from_one > 1e-3:
                raise InvalidParameterError.error(
                    "class_prior",
                    "Sum of class priors must equal 1.0, got " + String(p_sum),
                )
            for k in range(K):
                self.class_log_prior_.append(
                    Scalar[Self.compute_dtype](
                        log(Float64(self.class_prior[k]))
                    )
                )
        elif self.fit_prior:
            var n_scalar = Float64(N)
            for k in range(K):
                var p = Float64(self.class_count_[k]) / n_scalar
                var lp = log(p) if p > 1e-15 else -34.538776
                self.class_log_prior_.append(Scalar[Self.compute_dtype](lp))
        else:
            # Uniform prior: log(1/K)
            var unif_lp = -log(Float64(K))
            for _ in range(K):
                self.class_log_prior_.append(
                    Scalar[Self.compute_dtype](unif_lp)
                )

    def _joint_log_likelihood[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Calculates joint log likelihood: X @ log_prob.T + class_log_prior."""
        check_is_fitted("MultinomialNB", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.feature_log_prob_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "MultinomialNB._joint_log_likelihood",
            )

        var M = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()

        # GEMM: (M x D) @ (D x K) -> (M x K)
        var log_prob_T = self.feature_log_prob_.transpose()
        var joint_ll = gemm(X_comp, log_prob_T)

        for i in range(M):
            for k in range(K):
                joint_ll[i, k] += self.class_log_prior_[k]

        return joint_ll^

    def _joint_log_likelihood[
        feat_dtype: DType
    ](self, X: CSRMatrix[feat_dtype]) raises -> Matrix[Self.compute_dtype]:
        """Calculates joint log likelihood for sparse CSRMatrix: X @ log_prob.T + class_log_prior.
        """
        check_is_fitted("MultinomialNB", self.is_fitted)
        var D = self.feature_log_prob_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "MultinomialNB._joint_log_likelihood",
            )

        var M = X.rows
        var K = len(self.classes_)

        # Fast direct sparse accumulation reading non-zero elements once per row
        var joint_ll = Matrix[Self.compute_dtype](M, K, 0)
        for i in range(M):
            for k in range(K):
                joint_ll[i, k] = self.class_log_prior_[k]

            var start_idx = X.indptr[i]
            var end_idx = X.indptr[i + 1]
            for idx in range(start_idx, end_idx):
                var col = X.indices[idx]
                var val = Scalar[Self.compute_dtype](X.data[idx])
                for k in range(K):
                    joint_ll[i, k] += val * self.feature_log_prob_[k, col]

        return joint_ll^

    def predict_log_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class log probabilities for dense matrix X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Log probability matrix of shape $(N, K)$.
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
        """Predict class probability distributions for dense matrix X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Normalized probability matrix of shape $(N, K)$.
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
        """Predict class labels for dense feature matrix X.

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

    def predict_log_proba[
        feat_dtype: DType
    ](self, X: CSRMatrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class log probabilities for sparse CSRMatrix X."""
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
    ](self, X: CSRMatrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probability distributions for sparse CSRMatrix X."""
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
    ](self, X: CSRMatrix[feat_dtype]) raises -> List[Int]:
        """Predict class labels for sparse CSRMatrix X."""
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
        """Serializes MultinomialNB state into BufferWriter."""
        write_header(writer, "MultinomialNB")
        writer.write_bool(self.is_fitted)
        writer.write_float64(self.alpha.cast[DType.float64]())
        writer.write_bool(self.fit_prior)
        writer.write_float_list[Self.compute_dtype](self.class_prior)
        writer.write_int_list(self.classes_)
        writer.write_int_list(self.class_count_)
        writer.write_float_list[Self.compute_dtype](self.class_log_prior_)
        writer.write_matrix[Self.compute_dtype](self.feature_count_)
        writer.write_matrix[Self.compute_dtype](self.feature_log_prob_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes MultinomialNB from BufferReader."""
        check_header(reader, "MultinomialNB")
        var is_fitted = reader.read_bool()
        var alpha = Scalar[Self.compute_dtype](reader.read_float64())
        var fit_prior = reader.read_bool()
        var class_prior = reader.read_float_list[Self.compute_dtype]()
        var classes_ = reader.read_int_list()
        var class_count_ = reader.read_int_list()
        var class_log_prior_ = reader.read_float_list[Self.compute_dtype]()
        var feature_count_ = reader.read_matrix[Self.compute_dtype]()
        var feature_log_prob_ = reader.read_matrix[Self.compute_dtype]()

        var model = Self(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )
        model.is_fitted = is_fitted
        model.classes_ = classes_^
        model.class_count_ = class_count_^
        model.class_log_prior_ = class_log_prior_^
        model.feature_count_ = feature_count_^
        model.feature_log_prob_ = feature_log_prob_^
        return model^
