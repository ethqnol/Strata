from ..core.matrix import Matrix
from ..core.linalg import gemm
from ..base.estimator import Classifier
from ..utils.validation import (
    check_X_y,
    check_array,
    check_floating_dtype,
    check_is_fitted,
)
from ..utils.math import softmax
from std.math import exp
from ..exceptions.errors import InvalidParameterError, DimensionMismatchError
from ..io.serializer import (
    BufferWriter,
    BufferReader,
    Serializable,
    write_header,
    check_header,
)


struct LogisticRegression[
    compute_dtype: DType = DType.float64,
](Classifier, Copyable, Movable, Serializable):
    """Logistic Regression classifier with L2 regularization.

    Supports binary and multiclass (multinomial) classification by minimizing
    the regularized cross-entropy loss with gradient optimization:

    $$
    \\min_{W, b} -\\frac{1}{N} \\sum_{i=1}^{N} \\ln P(y_i \\mid x_i; W, b) + \\frac{1}{2C} \\|W\\|_F^2
    $$


    Parameters:
        compute_dtype: Computational precision data type. Default DType.float64.

    Args:
        penalty: Regularization norm ('l2' or 'none'). Default 'l2'.
        C: Inverse regularization strength ($C > 0$). Smaller values specify stronger regularization. Default 1.0.
        fit_intercept: Whether to calculate the intercept bias vector. Default True.
        max_iter: Maximum number of gradient optimization iterations. Default 100.
        tol: Tolerance threshold for stopping criterion based on gradient norm. Default 1e-4.
        learning_rate: Step size for gradient descent optimization updates. Default 0.1.

    Attributes:
        classes_: Sorted list of unique class labels seen during fit.
        coef_: Learned weight coefficient matrix of shape $(K, D)$.
        intercept_: Learned bias intercept vector of length $K$.
        is_fitted: Boolean flag indicating if estimator has been fitted.

    Examples:
        ```mojo
        from strata.linear_model import LogisticRegression
        from strata.core import Matrix

        var clf = LogisticRegression[DType.float64](C=1.0, max_iter=200)
        clf.fit(X_train, y_train)
        var probs = clf.predict_proba(X_test)
        var preds = clf.predict(X_test)
        ```
    """

    var is_fitted: Bool
    var penalty: String
    var C: Scalar[Self.compute_dtype]
    var fit_intercept: Bool
    var max_iter: Int
    var tol: Scalar[Self.compute_dtype]
    var learning_rate: Scalar[Self.compute_dtype]
    var classes_: List[Int]
    var coef_: Matrix[Self.compute_dtype]
    var intercept_: List[Scalar[Self.compute_dtype]]

    def __init__(
        out self,
        penalty: String = "l2",
        C: Scalar[Self.compute_dtype] = 1.0,
        fit_intercept: Bool = True,
        max_iter: Int = 100,
        tol: Scalar[Self.compute_dtype] = 1e-4,
        learning_rate: Scalar[Self.compute_dtype] = 0.1,
    ) raises:
        """Initialize the LogisticRegression estimator.

        Args:
            penalty: Regularization norm ('l2' or 'none'). Default 'l2'.
            C: Inverse regularization strength (must be strictly positive). Default 1.0.
            fit_intercept: Whether to calculate the intercept bias term. Default True.
            max_iter: Maximum number of optimization iterations. Default 100.
            tol: Tolerance for stopping criterion. Default 1e-4.
            learning_rate: Initial step size for gradient updates. Default 0.1.

        Raises:
            InvalidParameterError: If penalty is unsupported, C <= 0, max_iter <= 0, or tol < 0.
        """

        check_floating_dtype[Self.compute_dtype, "LogisticRegression"]()
        if penalty != "l2" and penalty != "none" and penalty != "None":
            raise InvalidParameterError.error(
                "penalty",
                "Unsupported penalty '"
                + penalty
                + "'. Expected 'l2' or 'none'.",
            )
        if C <= 0:
            raise InvalidParameterError.error(
                "C",
                "C must be strictly positive, got " + String(C),
            )
        if max_iter <= 0:
            raise InvalidParameterError.error(
                "max_iter",
                "max_iter must be strictly positive, got " + String(max_iter),
            )
        if tol < 0:
            raise InvalidParameterError.error(
                "tol",
                "tol must be non-negative, got " + String(tol),
            )
        self.is_fitted = False
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.classes_ = List[Int]()
        self.coef_ = Matrix[Self.compute_dtype](0, 0, 0)
        self.intercept_ = List[Scalar[Self.compute_dtype]]()

    def __init__(out self, *, copy: Self):
        """Copies an existing LogisticRegression instance."""
        self.is_fitted = copy.is_fitted
        self.penalty = copy.penalty
        self.C = copy.C
        self.fit_intercept = copy.fit_intercept
        self.max_iter = copy.max_iter
        self.tol = copy.tol
        self.learning_rate = copy.learning_rate
        self.classes_ = copy.classes_.copy()
        self.coef_ = copy.coef_.copy()
        self.intercept_ = copy.intercept_.copy()

    def fit[
        feat_dtype: DType, target_dtype: DType
    ](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) raises:
        """Fits the logistic regression model on training data (X, y).

        Args:
            X: Training feature matrix (N x D).
            y: Discrete target class labels (length N).
        """
        check_X_y(X, y)

        var N = X.rows
        var D = X.cols

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
                "LogisticRegression requires at least 2 distinct classes, got "
                + String(K),
            )
        self.classes_ = raw_classes^

        var y_idx = List[Int](capacity=N)
        for i in range(N):
            var target_val = Int(y[i])
            for k in range(K):
                if self.classes_[k] == target_val:
                    y_idx.append(k)
                    break

        var X_comp = X.cast[Self.compute_dtype]()

        var W = Matrix[Self.compute_dtype](K, D, 0)
        var b = List[Scalar[Self.compute_dtype]](capacity=K)
        for _ in range(K):
            b.append(0)

        var use_l2 = self.penalty == "l2"
        var n_samples = Scalar[Self.compute_dtype](N)
        var lambda_reg: Scalar[Self.compute_dtype] = (
            1.0 / (self.C * n_samples) if use_l2 else 0.0
        )

        var E = Matrix[Self.compute_dtype](N, K, 0)
        var grad_b = List[Scalar[Self.compute_dtype]](capacity=K)
        for _ in range(K):
            grad_b.append(0)

        for _ in range(self.max_iter):
            var W_T = W.transpose()
            var Z = gemm(X_comp, W_T)

            var z_ptr = Z.data.unsafe_ptr()
            var e_ptr = E.data.unsafe_ptr()

            for k in range(K):
                grad_b[k] = 0

            for i in range(N):
                var z_offset = i * K
                var max_logit: Float64 = Float64(
                    z_ptr.unsafe_offset(z_offset).unsafe_load()
                ) + (Float64(b[0]) if self.fit_intercept else 0.0)
                for k in range(1, K):
                    var lk: Float64 = Float64(
                        z_ptr.unsafe_offset(z_offset + k).unsafe_load()
                    ) + (Float64(b[k]) if self.fit_intercept else 0.0)
                    if lk > max_logit:
                        max_logit = lk

                var sum_exp: Float64 = 0.0
                for k in range(K):
                    var lk: Float64 = Float64(
                        z_ptr.unsafe_offset(z_offset + k).unsafe_load()
                    ) + (Float64(b[k]) if self.fit_intercept else 0.0)
                    var exp_val = exp(lk - max_logit)
                    e_ptr.unsafe_offset(z_offset + k).unsafe_store(
                        Scalar[Self.compute_dtype](exp_val)
                    )
                    sum_exp += exp_val

                var inv_sum_exp = 1.0 / sum_exp
                var target = y_idx[i]
                for k in range(K):
                    var prob = (
                        Float64(e_ptr.unsafe_offset(z_offset + k).unsafe_load())
                        * inv_sum_exp
                    )
                    var err = Scalar[Self.compute_dtype](
                        prob - (1.0 if target == k else 0.0)
                    )
                    e_ptr.unsafe_offset(z_offset + k).unsafe_store(err)
                    if self.fit_intercept:
                        grad_b[k] += err

            var E_T = E.transpose()
            var grad_W = gemm(E_T, X_comp)
            var gw_ptr = grad_W.data.unsafe_ptr()
            var w_ptr = W.data.unsafe_ptr()

            # Check convergence
            var max_grad: Scalar[Self.compute_dtype] = 0
            for k in range(K):
                var gw_row_offset = k * D
                var w_row_offset = k * D
                for j in range(D):
                    var g_w = (
                        gw_ptr.unsafe_offset(gw_row_offset + j).unsafe_load()
                        / n_samples
                        + lambda_reg
                        * w_ptr.unsafe_offset(w_row_offset + j).unsafe_load()
                    )
                    var abs_gw = g_w if g_w >= 0 else -g_w
                    if abs_gw > max_grad:
                        max_grad = abs_gw
                if self.fit_intercept:
                    var g_b = grad_b[k] / n_samples
                    var abs_gb = g_b if g_b >= 0 else -g_b
                    if abs_gb > max_grad:
                        max_grad = abs_gb

            if max_grad < self.tol:
                break

            for k in range(K):
                var gw_row_offset = k * D
                var w_row_offset = k * D
                for j in range(D):
                    var g_w = (
                        gw_ptr.unsafe_offset(gw_row_offset + j).unsafe_load()
                        / n_samples
                        + lambda_reg
                        * w_ptr.unsafe_offset(w_row_offset + j).unsafe_load()
                    )
                    var cur_w = w_ptr.unsafe_offset(
                        w_row_offset + j
                    ).unsafe_load()
                    w_ptr.unsafe_offset(w_row_offset + j).unsafe_store(
                        cur_w - self.learning_rate * g_w
                    )
                if self.fit_intercept:
                    b[k] -= self.learning_rate * (grad_b[k] / n_samples)

        self.coef_ = W^
        self.intercept_ = b^
        self.is_fitted = True

    def predict_proba[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> Matrix[feat_dtype]:
        """Predict class probability distributions for samples in X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            Matrix[feat_dtype]: Probability matrix of shape $(N, K)$, where row $i$
                contains the normalized probability distribution over $K$ classes.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            DimensionMismatchError: If the feature dimension of $X$ does not match `coef_.cols`.
        """
        check_is_fitted("LogisticRegression", self.is_fitted)
        check_array[feat_dtype](X)
        var D = self.coef_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "LogisticRegression.predict_proba",
            )

        var N = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()
        var W_T = self.coef_.transpose()
        var Z = gemm(X_comp, W_T)
        var z_ptr = Z.data.unsafe_ptr()

        var probs_data = List[Scalar[feat_dtype]](capacity=N * K)
        for i in range(N):
            var z_offset = i * K
            var max_logit: Float64 = Float64(
                z_ptr.unsafe_offset(z_offset).unsafe_load()
            ) + (Float64(self.intercept_[0]) if self.fit_intercept else 0.0)
            for k in range(1, K):
                var lk: Float64 = Float64(
                    z_ptr.unsafe_offset(z_offset + k).unsafe_load()
                ) + (Float64(self.intercept_[k]) if self.fit_intercept else 0.0)
                if lk > max_logit:
                    max_logit = lk

            var sum_exp: Float64 = 0.0
            for k in range(K):
                var lk: Float64 = Float64(
                    z_ptr.unsafe_offset(z_offset + k).unsafe_load()
                ) + (Float64(self.intercept_[k]) if self.fit_intercept else 0.0)
                sum_exp += exp(lk - max_logit)

            var inv_sum_exp = 1.0 / sum_exp
            for k in range(K):
                var lk: Float64 = Float64(
                    z_ptr.unsafe_offset(z_offset + k).unsafe_load()
                ) + (Float64(self.intercept_[k]) if self.fit_intercept else 0.0)
                var p = exp(lk - max_logit) * inv_sum_exp
                probs_data.append(Scalar[feat_dtype](p))

        return Matrix[feat_dtype](N, K, probs_data^)

    def predict[
        feat_dtype: DType
    ](self, X: Matrix[feat_dtype]) raises -> List[Int]:
        """Predict discrete class labels for samples in X.

        Args:
            X: Feature matrix of shape $(N, D)$.

        Returns:
            List[Int]: Predicted class labels vector of length $N$.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            DimensionMismatchError: If the feature dimension of $X$ does not match `coef_.cols`.
        """
        check_is_fitted("LogisticRegression", self.is_fitted)

        check_array[feat_dtype](X)
        var D = self.coef_.cols
        if X.cols != D:
            raise DimensionMismatchError.error(
                "X.cols == " + String(D),
                "X.cols == " + String(X.cols),
                "LogisticRegression.predict",
            )

        var N = X.rows
        var K = len(self.classes_)
        var X_comp = X.cast[Self.compute_dtype]()

        var W_T = self.coef_.transpose()
        var Z = gemm(X_comp, W_T)
        var z_ptr = Z.data.unsafe_ptr()

        var preds = List[Int](capacity=N)
        for i in range(N):
            var z_offset = i * K
            var best_k = 0
            var max_logit: Float64 = Float64(
                z_ptr.unsafe_offset(z_offset).unsafe_load()
            ) + (Float64(self.intercept_[0]) if self.fit_intercept else 0.0)
            for k in range(1, K):
                var lk: Float64 = Float64(
                    z_ptr.unsafe_offset(z_offset + k).unsafe_load()
                ) + (Float64(self.intercept_[k]) if self.fit_intercept else 0.0)
                if lk > max_logit:
                    max_logit = lk
                    best_k = k
            preds.append(self.classes_[best_k])

        return preds^

    def serialize(self, mut writer: BufferWriter):
        """Serializes LogisticRegression parameters and fitted state into BufferWriter.
        """
        write_header(writer, "LogisticRegression")
        writer.write_bool(self.is_fitted)
        writer.write_string(self.penalty)
        writer.write_float64(self.C.cast[DType.float64]())
        writer.write_bool(self.fit_intercept)
        writer.write_int(self.max_iter)
        writer.write_float64(self.tol.cast[DType.float64]())
        writer.write_float64(self.learning_rate.cast[DType.float64]())
        writer.write_int_list(self.classes_)
        writer.write_matrix[Self.compute_dtype](self.coef_)
        writer.write_float_list[Self.compute_dtype](self.intercept_)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes LogisticRegression from BufferReader."""
        check_header(reader, "LogisticRegression")
        var is_fitted = reader.read_bool()
        var penalty = reader.read_string()
        var C = Scalar[Self.compute_dtype](reader.read_float64())
        var fit_intercept = reader.read_bool()
        var max_iter = reader.read_int()
        var tol = Scalar[Self.compute_dtype](reader.read_float64())
        var learning_rate = Scalar[Self.compute_dtype](reader.read_float64())
        var classes_ = reader.read_int_list()
        var coef_ = reader.read_matrix[Self.compute_dtype]()
        var intercept_ = reader.read_float_list[Self.compute_dtype]()

        var model = Self(
            penalty=penalty,
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
        )
        model.is_fitted = is_fitted
        model.classes_ = classes_^
        model.coef_ = coef_^
        model.intercept_ = intercept_^
        return model^
