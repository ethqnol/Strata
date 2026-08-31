from ..core.matrix import Matrix
from ..io.serializer import BufferWriter, BufferReader


struct Node(Copyable, Movable):
    """A single split or leaf node in a decision tree."""

    var feature_idx: Int
    var threshold: Float64
    var left_child: Int
    var right_child: Int
    var value_reg: Float64
    var class_counts: List[Int]
    var class_probabilities: List[Float64]
    var impurity: Float64
    var n_node_samples: Int
    var is_leaf: Bool

    def __init__(out self):
        self.feature_idx = -1
        self.threshold = 0.0
        self.left_child = -1
        self.right_child = -1
        self.value_reg = 0.0
        self.class_counts = List[Int]()
        self.class_probabilities = List[Float64]()
        self.impurity = 0.0
        self.n_node_samples = 0
        self.is_leaf = True

    def __init__(
        out self,
        value_reg: Float64,
        impurity: Float64,
        n_node_samples: Int,
    ):
        """Constructs a regression leaf node."""
        self.feature_idx = -1
        self.threshold = 0.0
        self.left_child = -1
        self.right_child = -1
        self.value_reg = value_reg
        self.class_counts = List[Int]()
        self.class_probabilities = List[Float64]()
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.is_leaf = True

    def __init__(
        out self,
        var class_counts: List[Int],
        var class_probabilities: List[Float64],
        impurity: Float64,
        n_node_samples: Int,
    ):
        """Constructs a classification leaf node."""
        self.feature_idx = -1
        self.threshold = 0.0
        self.left_child = -1
        self.right_child = -1
        self.value_reg = 0.0
        self.class_counts = class_counts^
        self.class_probabilities = class_probabilities^
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.is_leaf = True

    def __init__(
        out self,
        feature_idx: Int,
        threshold: Float64,
        left_child: Int,
        right_child: Int,
        impurity: Float64,
        n_node_samples: Int,
        value_reg: Float64 = 0.0,
        var class_counts: List[Int] = List[Int](),
        var class_probabilities: List[Float64] = List[Float64](),
    ):
        """Constructs an internal decision split node."""
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.value_reg = value_reg
        self.class_counts = class_counts^
        self.class_probabilities = class_probabilities^
        self.impurity = impurity
        self.n_node_samples = n_node_samples
        self.is_leaf = False

    def __init__(out self, *, copy: Self):
        self.feature_idx = copy.feature_idx
        self.threshold = copy.threshold
        self.left_child = copy.left_child
        self.right_child = copy.right_child
        self.value_reg = copy.value_reg
        self.class_counts = copy.class_counts.copy()
        self.class_probabilities = copy.class_probabilities.copy()
        self.impurity = copy.impurity
        self.n_node_samples = copy.n_node_samples
        self.is_leaf = copy.is_leaf


struct Tree(Copyable, Movable):
    """Flat array-backed binary decision tree."""

    var nodes: List[Node]
    var n_classes: Int
    var classes_: List[Int]

    def __init__(out self):
        self.nodes = List[Node]()
        self.n_classes = 0
        self.classes_ = List[Int]()

    def __init__(out self, n_classes: Int, var classes_: List[Int]):
        self.nodes = List[Node]()
        self.n_classes = n_classes
        self.classes_ = classes_^

    def __init__(out self, *, copy: Self):
        self.nodes = copy.nodes.copy()
        self.n_classes = copy.n_classes
        self.classes_ = copy.classes_.copy()

    def node_count(self) -> Int:
        """Returns total number of nodes in the tree."""
        return len(self.nodes)

    def add_node(mut self, var node: Node) -> Int:
        """Appends node to storage array and returns its index."""
        var idx = len(self.nodes)
        self.nodes.append(node^)
        return idx

    def max_depth(self) -> Int:
        """Returns maximum depth (edges from root) in the tree."""
        if len(self.nodes) == 0:
            return 0
        return self._compute_depth(0)

    def _compute_depth(self, node_idx: Int) -> Int:
        if node_idx < 0 or node_idx >= len(self.nodes):
            return 0
        if self.nodes[node_idx].is_leaf:
            return 0
        var left_d = self._compute_depth(self.nodes[node_idx].left_child)
        var right_d = self._compute_depth(self.nodes[node_idx].right_child)
        var max_child = left_d if left_d > right_d else right_d
        return 1 + max_child

    def predict_regression[
        in_dtype: DType, out_dtype: DType
    ](self, X: Matrix[in_dtype]) -> List[Scalar[out_dtype]]:
        """Evaluates regression predictions for input matrix X."""
        var n_rows = X.rows
        var preds = List[Scalar[out_dtype]](capacity=n_rows)

        if len(self.nodes) == 0:
            for _ in range(n_rows):
                preds.append(Scalar[out_dtype](0.0))
            return preds^

        for r in range(n_rows):
            var curr = 0
            while not self.nodes[curr].is_leaf:
                var f_idx = self.nodes[curr].feature_idx
                var val = X[r, f_idx].cast[DType.float64]()
                var next_idx = (
                    self.nodes[curr].left_child if val
                    <= self.nodes[curr]
                    .threshold else self.nodes[curr]
                    .right_child
                )
                if next_idx < 0 or next_idx >= len(self.nodes):
                    break
                curr = next_idx
            preds.append(Scalar[out_dtype](self.nodes[curr].value_reg))

        return preds^

    def predict_classification[
        in_dtype: DType
    ](self, X: Matrix[in_dtype]) -> List[Int]:
        """Evaluates discrete class predictions for input matrix X."""
        var n_rows = X.rows
        var preds = List[Int](capacity=n_rows)

        if len(self.nodes) == 0:
            for _ in range(n_rows):
                preds.append(0)
            return preds^

        for r in range(n_rows):
            var curr = 0
            while not self.nodes[curr].is_leaf:
                var f_idx = self.nodes[curr].feature_idx
                var val = X[r, f_idx].cast[DType.float64]()
                var next_idx = (
                    self.nodes[curr].left_child if val
                    <= self.nodes[curr]
                    .threshold else self.nodes[curr]
                    .right_child
                )
                if next_idx < 0 or next_idx >= len(self.nodes):
                    break
                curr = next_idx

            var best_c = 0
            var n_probs = len(self.nodes[curr].class_probabilities)
            if n_probs > 0:
                var best_prob: Float64 = -1.0
                for c in range(n_probs):
                    var p = self.nodes[curr].class_probabilities[c]
                    if p > best_prob:
                        best_prob = p
                        best_c = c
            else:
                var n_cnts = len(self.nodes[curr].class_counts)
                var best_cnt = -1
                for c in range(n_cnts):
                    var cnt = self.nodes[curr].class_counts[c]
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c

            if len(self.classes_) > best_c:
                preds.append(self.classes_[best_c])
            else:
                preds.append(best_c)

        return preds^

    def predict_proba[
        in_dtype: DType, out_dtype: DType
    ](self, X: Matrix[in_dtype]) -> Matrix[out_dtype]:
        """Evaluates class probabilities for input matrix X."""
        var n_rows = X.rows
        var n_cols = self.n_classes
        if n_cols <= 0:
            n_cols = 1

        var total_elements = n_rows * n_cols
        var data = List[Scalar[out_dtype]](capacity=total_elements)

        if len(self.nodes) == 0:
            for _ in range(total_elements):
                data.append(Scalar[out_dtype](0.0))
            return Matrix[out_dtype](n_rows, n_cols, data^)

        for r in range(n_rows):
            var curr = 0
            while not self.nodes[curr].is_leaf:
                var f_idx = self.nodes[curr].feature_idx
                var val = X[r, f_idx].cast[DType.float64]()
                var next_idx = (
                    self.nodes[curr].left_child if val
                    <= self.nodes[curr]
                    .threshold else self.nodes[curr]
                    .right_child
                )
                if next_idx < 0 or next_idx >= len(self.nodes):
                    break
                curr = next_idx

            var n_probs = len(self.nodes[curr].class_probabilities)
            if n_probs > 0:
                for c in range(n_cols):
                    if c < n_probs:
                        data.append(
                            Scalar[out_dtype](
                                self.nodes[curr].class_probabilities[c]
                            )
                        )
                    else:
                        data.append(Scalar[out_dtype](0.0))
            else:
                var n_cnts = len(self.nodes[curr].class_counts)
                var total_cnt = 0
                for c in range(n_cnts):
                    total_cnt += self.nodes[curr].class_counts[c]
                var total_f = Float64(total_cnt) if total_cnt > 0 else 1.0
                for c in range(n_cols):
                    if c < n_cnts:
                        data.append(
                            Scalar[out_dtype](
                                Float64(self.nodes[curr].class_counts[c])
                                / total_f
                            )
                        )
                    else:
                        data.append(Scalar[out_dtype](0.0))

        return Matrix[out_dtype](n_rows, n_cols, data^)

    def serialize(self, mut writer: BufferWriter):
        """Serializes tree nodes, class counts, and probabilities into BufferWriter.
        """
        writer.write_int(self.n_classes)
        writer.write_int_list(self.classes_)
        writer.write_int(len(self.nodes))
        for i in range(len(self.nodes)):
            var node = self.nodes[i].copy()
            writer.write_int(node.feature_idx)
            writer.write_float64(node.threshold)
            writer.write_int(node.left_child)
            writer.write_int(node.right_child)
            writer.write_float64(node.value_reg)
            writer.write_int_list(node.class_counts)
            writer.write_float64_list(node.class_probabilities)
            writer.write_float64(node.impurity)
            writer.write_int(node.n_node_samples)
            writer.write_bool(node.is_leaf)

    @staticmethod
    def deserialize(mut reader: BufferReader) raises -> Self:
        """Deserializes Tree from BufferReader."""
        var n_classes = reader.read_int()
        var classes_ = reader.read_int_list()
        var n_nodes = reader.read_int()
        var tree = Self(n_classes, classes_^)
        tree.nodes = List[Node](capacity=n_nodes)
        for _ in range(n_nodes):
            var node = Node()
            node.feature_idx = reader.read_int()
            node.threshold = reader.read_float64()
            node.left_child = reader.read_int()
            node.right_child = reader.read_int()
            node.value_reg = reader.read_float64()
            node.class_counts = reader.read_int_list()
            node.class_probabilities = reader.read_float64_list()
            node.impurity = reader.read_float64()
            node.n_node_samples = reader.read_int()
            node.is_leaf = reader.read_bool()
            tree.nodes.append(node^)
        return tree^
