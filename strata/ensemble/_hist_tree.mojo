from std.math import max, min
from ..core.matrix import Matrix
from ._binning import BinnedMatrix
from ..exceptions.errors import InvalidParameterError


struct FeatureHistogram(Copyable, Movable):
    """Accumulated 1st and 2nd order statistics across discrete bins for a single feature.
    """

    var grad_sum: List[Float64]
    var hess_sum: List[Float64]
    var count: List[Int]

    def __init__(out self, n_bins: Int):
        """Initializes zeroed histogram bins."""
        self.grad_sum = List[Float64](capacity=n_bins)
        self.hess_sum = List[Float64](capacity=n_bins)
        self.count = List[Int](capacity=n_bins)
        for _ in range(n_bins):
            self.grad_sum.append(0.0)
            self.hess_sum.append(0.0)
            self.count.append(0)

    def __init__(out self, *, copy: Self):
        """Copies a FeatureHistogram instance."""
        self.grad_sum = copy.grad_sum.copy()
        self.hess_sum = copy.hess_sum.copy()
        self.count = copy.count.copy()

    def subtract(self, child: FeatureHistogram) -> FeatureHistogram:
        """Computes sibling histogram via $O(K)$ subtraction: $\\text{Parent} - \\text{Child}$.
        """
        var n_bins = len(self.grad_sum)
        var res = FeatureHistogram(n_bins)
        for b in range(n_bins):
            res.grad_sum[b] = self.grad_sum[b] - child.grad_sum[b]
            var h = self.hess_sum[b] - child.hess_sum[b]
            res.hess_sum[b] = 0.0 if h < 0.0 else h
            var c = self.count[b] - child.count[b]
            res.count[b] = 0 if c < 0 else c
        return res^


struct HistSplit(Copyable, Movable):
    """Candidate split evaluated on binned feature histograms."""

    var feature_idx: Int
    var bin_idx: UInt8
    var threshold: Float64
    var gain: Float64
    var grad_left: Float64
    var hess_left: Float64
    var count_left: Int
    var grad_right: Float64
    var hess_right: Float64
    var count_right: Int
    var is_valid: Bool

    def __init__(out self):
        """Initializes an invalid dummy split."""
        self.feature_idx = -1
        self.bin_idx = 0
        self.threshold = 0.0
        self.gain = -1e30
        self.grad_left = 0.0
        self.hess_left = 0.0
        self.count_left = 0
        self.grad_right = 0.0
        self.hess_right = 0.0
        self.count_right = 0
        self.is_valid = False

    def __init__(out self, *, copy: Self):
        """Copies a HistSplit instance."""
        self.feature_idx = copy.feature_idx
        self.bin_idx = copy.bin_idx
        self.threshold = copy.threshold
        self.gain = copy.gain
        self.grad_left = copy.grad_left
        self.hess_left = copy.hess_left
        self.count_left = copy.count_left
        self.grad_right = copy.grad_right
        self.hess_right = copy.hess_right
        self.count_right = copy.count_right
        self.is_valid = copy.is_valid


struct HistNode(Copyable, Movable):
    """A decision or leaf node in a Histogram Gradient Boosted Tree."""

    var is_leaf: Bool
    var feature_idx: Int
    var bin_threshold: UInt8
    var float_threshold: Float64
    var left_child: Int
    var right_child: Int
    var value: Float64
    var n_samples: Int
    var gain: Float64

    def __init__(out self, value: Float64, n_samples: Int):
        """Constructs a leaf node."""
        self.is_leaf = True
        self.feature_idx = -1
        self.bin_threshold = 0
        self.float_threshold = 0.0
        self.left_child = -1
        self.right_child = -1
        self.value = value
        self.n_samples = n_samples
        self.gain = 0.0

    def __init__(
        out self,
        feature_idx: Int,
        bin_threshold: UInt8,
        float_threshold: Float64,
        left_child: Int,
        right_child: Int,
        n_samples: Int,
        gain: Float64,
    ):
        """Constructs a split node."""
        self.is_leaf = False
        self.feature_idx = feature_idx
        self.bin_threshold = bin_threshold
        self.float_threshold = float_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value = 0.0
        self.n_samples = n_samples
        self.gain = gain

    def __init__(out self, *, copy: Self):
        """Copies a HistNode instance."""
        self.is_leaf = copy.is_leaf
        self.feature_idx = copy.feature_idx
        self.bin_threshold = copy.bin_threshold
        self.float_threshold = copy.float_threshold
        self.left_child = copy.left_child
        self.right_child = copy.right_child
        self.value = copy.value
        self.n_samples = copy.n_samples
        self.gain = copy.gain


def _build_node_histograms(
    binned_matrix: BinnedMatrix,
    sample_indices: List[Int],
    gradients: List[Float64],
    hessians: List[Float64],
) -> List[FeatureHistogram]:
    """Constructs histogram buffers for all features across sample subset."""
    var D = binned_matrix.cols
    var hists = List[FeatureHistogram](capacity=D)
    for j in range(D):
        hists.append(FeatureHistogram(binned_matrix.n_bins_per_feature[j]))

    var n_samples = len(sample_indices)
    var b_ptr = binned_matrix.unsafe_ptr()
    var g_ptr = gradients.unsafe_ptr()
    var h_ptr = hessians.unsafe_ptr()

    for idx in range(n_samples):
        var i = sample_indices[idx]
        var g = g_ptr.unsafe_offset(i).unsafe_load()
        var h = h_ptr.unsafe_offset(i).unsafe_load()
        var row_offset = i * D

        for j in range(D):
            var b = Int(b_ptr.unsafe_offset(row_offset + j).unsafe_load())
            hists[j].grad_sum[b] += g
            hists[j].hess_sum[b] += h
            hists[j].count[b] += 1

    return hists^


def _find_best_split_in_histograms(
    hists: List[FeatureHistogram],
    bin_thresholds: List[List[Float64]],
    l2_regularization: Float64,
    min_samples_leaf: Int,
    min_gain_to_split: Float64,
) -> HistSplit:
    """Finds the optimal split maximizing Gain = 0.5 * (GL^2/(HL+lam) + GR^2/(HR+lam) - GP^2/(HP+lam)).
    """
    var D = len(hists)
    var best_split = HistSplit()
    var max_gain = min_gain_to_split

    for j in range(D):
        var n_bins = len(hists[j].grad_sum)
        if n_bins <= 1 or len(bin_thresholds[j]) == 0:
            continue

        var G_tot: Float64 = 0.0
        var H_tot: Float64 = 0.0
        var C_tot = 0
        for b in range(n_bins):
            G_tot += hists[j].grad_sum[b]
            H_tot += hists[j].hess_sum[b]
            C_tot += hists[j].count[b]

        if C_tot < 2 * min_samples_leaf:
            continue

        var parent_score = (G_tot * G_tot) / (H_tot + l2_regularization)

        var G_L: Float64 = 0.0
        var H_L: Float64 = 0.0
        var C_L = 0

        # Scan potential split bins
        var n_thresh = len(bin_thresholds[j])
        for b in range(n_thresh):
            G_L += hists[j].grad_sum[b]
            H_L += hists[j].hess_sum[b]
            C_L += hists[j].count[b]

            var C_R = C_tot - C_L
            if C_L < min_samples_leaf or C_R < min_samples_leaf:
                continue

            var G_R = G_tot - G_L
            var H_R = H_tot - H_L

            var left_score = (G_L * G_L) / (H_L + l2_regularization)
            var right_score = (G_R * G_R) / (H_R + l2_regularization)
            var gain = 0.5 * (left_score + right_score - parent_score)

            if gain > max_gain:
                max_gain = gain
                best_split.feature_idx = j
                best_split.bin_idx = UInt8(b)
                best_split.threshold = bin_thresholds[j][b]
                best_split.gain = gain
                best_split.grad_left = G_L
                best_split.hess_left = H_L
                best_split.count_left = C_L
                best_split.grad_right = G_R
                best_split.hess_right = H_R
                best_split.count_right = C_R
                best_split.is_valid = True

    return best_split^


struct HistTree(Copyable, Movable):
    """Histogram Gradient Boosted Decision Tree."""

    var nodes: List[HistNode]
    var max_depth: Int
    var max_leaf_nodes: Int
    var min_samples_leaf: Int
    var l2_regularization: Float64
    var min_gain_to_split: Float64
    var shrinkage: Float64

    def __init__(
        out self,
        max_depth: Int = 6,
        max_leaf_nodes: Int = 31,
        min_samples_leaf: Int = 20,
        l2_regularization: Float64 = 0.0,
        min_gain_to_split: Float64 = 0.0,
        shrinkage: Float64 = 0.1,
    ):
        """Initializes a HistTree with growth constraints."""
        self.nodes = List[HistNode]()
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.min_gain_to_split = min_gain_to_split
        self.shrinkage = shrinkage

    def __init__(out self, *, copy: Self):
        """Copies a HistTree instance."""
        self.nodes = List[HistNode](capacity=len(copy.nodes))
        for i in range(len(copy.nodes)):
            self.nodes.append(copy.nodes[i].copy())
        self.max_depth = copy.max_depth
        self.max_leaf_nodes = copy.max_leaf_nodes
        self.min_samples_leaf = copy.min_samples_leaf
        self.l2_regularization = copy.l2_regularization
        self.min_gain_to_split = copy.min_gain_to_split
        self.shrinkage = copy.shrinkage

    def _compute_leaf_value(self, sum_g: Float64, sum_h: Float64) -> Float64:
        """Computes optimal regularized leaf step: $v = -\\frac{\\sum g}{\\sum h + \\lambda} \\times \\text{shrinkage}$.
        """
        var denom = sum_h + self.l2_regularization
        if denom <= 0.0:
            return 0.0
        return (-sum_g / denom) * self.shrinkage

    def build(
        mut self,
        binned_matrix: BinnedMatrix,
        gradients: List[Float64],
        hessians: List[Float64],
    ):
        """Builds the tree using histogram aggregation and the histogram subtraction trick.
        """
        self.nodes.clear()
        var N = binned_matrix.rows
        if N == 0:
            self.nodes.append(HistNode(0.0, 0))
            return

        var all_indices = List[Int](capacity=N)
        var total_g: Float64 = 0.0
        var total_h: Float64 = 0.0
        for i in range(N):
            all_indices.append(i)
            total_g += gradients[i]
            total_h += hessians[i]

        var root_hists = _build_node_histograms(
            binned_matrix, all_indices, gradients, hessians
        )

        var n_leaves = 1
        _ = self._grow_node(
            binned_matrix,
            all_indices,
            gradients,
            hessians,
            root_hists,
            total_g,
            total_h,
            depth=0,
            n_leaves=n_leaves,
        )

    def _grow_node(
        mut self,
        binned_matrix: BinnedMatrix,
        indices: List[Int],
        gradients: List[Float64],
        hessians: List[Float64],
        hists: List[FeatureHistogram],
        sum_g: Float64,
        sum_h: Float64,
        depth: Int,
        mut n_leaves: Int,
    ) -> Int:
        var n_samples = len(indices)
        var leaf_val = self._compute_leaf_value(sum_g, sum_h)

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < 2 * self.min_samples_leaf
            or n_leaves >= self.max_leaf_nodes
        ):
            var node_idx = len(self.nodes)
            self.nodes.append(HistNode(leaf_val, n_samples))
            return node_idx

        var best_split = _find_best_split_in_histograms(
            hists,
            binned_matrix.bin_thresholds,
            self.l2_regularization,
            self.min_samples_leaf,
            self.min_gain_to_split,
        )

        if not best_split.is_valid:
            var node_idx = len(self.nodes)
            self.nodes.append(HistNode(leaf_val, n_samples))
            return node_idx

        # Partition indices into Left and Right child subsets
        var left_indices = List[Int](capacity=best_split.count_left)
        var right_indices = List[Int](capacity=best_split.count_right)
        var feat = best_split.feature_idx
        var split_bin = best_split.bin_idx

        for idx in range(n_samples):
            var i = indices[idx]
            var b = binned_matrix.get(i, feat)
            if b <= split_bin:
                left_indices.append(i)
            else:
                right_indices.append(i)

        if (
            len(left_indices) < self.min_samples_leaf
            or len(right_indices) < self.min_samples_leaf
        ):
            var node_idx = len(self.nodes)
            self.nodes.append(HistNode(leaf_val, n_samples))
            return node_idx

        # Allocate placeholder node for current split
        var curr_node_idx = len(self.nodes)
        self.nodes.append(
            HistNode(
                best_split.feature_idx,
                best_split.bin_idx,
                best_split.threshold,
                left_child=-1,
                right_child=-1,
                n_samples=n_samples,
                gain=best_split.gain,
            )
        )
        n_leaves += 1

        # Histogram Subtraction Trick: compute smaller child from data, sibling by subtraction
        var left_hists: List[FeatureHistogram]
        var right_hists: List[FeatureHistogram]

        if len(left_indices) <= len(right_indices):
            left_hists = _build_node_histograms(
                binned_matrix, left_indices, gradients, hessians
            )
            right_hists = List[FeatureHistogram](capacity=len(hists))
            for j in range(len(hists)):
                right_hists.append(hists[j].subtract(left_hists[j]))
        else:
            right_hists = _build_node_histograms(
                binned_matrix, right_indices, gradients, hessians
            )
            left_hists = List[FeatureHistogram](capacity=len(hists))
            for j in range(len(hists)):
                left_hists.append(hists[j].subtract(right_hists[j]))

        var left_child_idx = self._grow_node(
            binned_matrix,
            left_indices,
            gradients,
            hessians,
            left_hists,
            best_split.grad_left,
            best_split.hess_left,
            depth + 1,
            n_leaves,
        )

        var right_child_idx = self._grow_node(
            binned_matrix,
            right_indices,
            gradients,
            hessians,
            right_hists,
            best_split.grad_right,
            best_split.hess_right,
            depth + 1,
            n_leaves,
        )

        self.nodes[curr_node_idx].left_child = left_child_idx
        self.nodes[curr_node_idx].right_child = right_child_idx

        return curr_node_idx

    def predict_row[
        dtype: DType = DType.float64
    ](self, X: Matrix[dtype], r: Int) -> Float64:
        """Traverses the tree to predict the scalar leaf value for a single sample row.
        """
        if len(self.nodes) == 0:
            return 0.0

        var curr = 0
        while not self.nodes[curr].is_leaf:
            var feat = self.nodes[curr].feature_idx
            var val = Float64(X[r, feat])
            if val <= self.nodes[curr].float_threshold:
                curr = self.nodes[curr].left_child
            else:
                curr = self.nodes[curr].right_child

        return self.nodes[curr].value

    def predict[
        dtype: DType = DType.float64
    ](self, X: Matrix[dtype]) -> List[Float64]:
        """Predicts scalar leaf updates for all rows in continuous matrix X."""
        var N = X.rows
        var preds = List[Float64](capacity=N)
        for r in range(N):
            preds.append(self.predict_row[dtype](X, r))
        return preds^

    def predict_binned(self, binned_matrix: BinnedMatrix, r: Int) -> Float64:
        """Fast prediction traversal using integer UInt8 comparisons on binned data.
        """
        if len(self.nodes) == 0:
            return 0.0

        var curr = 0
        while not self.nodes[curr].is_leaf:
            var feat = self.nodes[curr].feature_idx
            var b = binned_matrix.get(r, feat)
            if b <= self.nodes[curr].bin_threshold:
                curr = self.nodes[curr].left_child
            else:
                curr = self.nodes[curr].right_child

        return self.nodes[curr].value
