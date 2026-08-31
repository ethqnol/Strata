# Sparse Matrix Operations & SpMM Kernels

This guide shows how to instantiate compressed sparse formats (`CSRMatrix` and `CSCMatrix`), convert between representations in $\mathcal{O}(\text{nnz})$ time, and execute sparse-dense matrix multiplication (SpMM).


---

## 1. Creating a Compressed Sparse Row (CSR) Matrix

A `CSRMatrix` stores non-zero elements across three flat vectors: `data`, `indices`, and `indptr`.

```mojo
from strata.core.csr_matrix import CSRMatrix
from strata.core.matrix import Matrix
from strata.core.sparse_ops import spmm

def main() raises:
    # 3x3 sparse matrix:
    # [ 1.0  0.0  2.0 ]
    # [ 0.0  3.0  0.0 ]
    # [ 4.0  0.0  5.0 ]

    var data = List[Float64](1.0, 2.0, 3.0, 4.0, 5.0)
    var indices = List[Int](0, 2, 1, 0, 2)
    var indptr = List[Int](0, 2, 3, 5)

    var csr = CSRMatrix[DType.float64](data, indices, indptr, rows=3, cols=3)
    print("Non-zeros:", csr.nnz)

    # Convert to CSC format in O(nnz) time
    var csc = csr.to_csc()
```

---

## 2. Sparse-Dense Matrix Multiplication (SpMM)

To multiply a sparse matrix $\mathbf{A} \in \mathbb{R}^{M \times K}$ by a dense matrix $\mathbf{B} \in \mathbb{R}^{K \times N}$:

```mojo
    var B = Matrix[DType.float64](3, 2)
    # Fill B...

    # Compute C = A * B
    var C = spmm(csr, B)
    print("Result shape:", C.rows, "x", C.cols)
```

---

## Related References
- [strata.core Reference](../reference/core.md)
- [strata.decomposition.TruncatedSVD](../reference/decomposition.md)
