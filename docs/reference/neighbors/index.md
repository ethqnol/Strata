# `strata.neighbors`

Distance metrics (Euclidean, Manhattan, Chebyshev, Minkowski, Cosine), nearest neighbors search, and k-NN classification and regression.

---

## Structs & Classes

| Struct | Description |
| :--- | :--- |
| [`NeighborDistIdx`](NeighborDistIdx.md) | Container holding a sample distance and its training dataset row index. |
| [`NearestNeighbors`](NearestNeighbors.md) | Unsupervised learner for implementing neighbor searches. |
| [`KNeighborsClassifier`](KNeighborsClassifier.md) | Classifier implementing the k-nearest neighbors vote. |
| [`KNeighborsRegressor`](KNeighborsRegressor.md) | Regression based on k-nearest neighbors. |
| [`KDNode`](KDNode.md) | Contiguous node in a flat KD-Tree buffer. |
| [`_AxisIndexPair`](_AxisIndexPair.md) | — |
| [`KDTree`](KDTree.md) | Fast spatial index for nearest neighbor and radius queries in low dimensions. |

## Functions

| Function | Description |
| :--- | :--- |
| [`sqeuclidean_distance`](sqeuclidean_distance.md) | Compute the squared Euclidean distance between row X[row_x] and row Y[row_y]. |
| [`euclidean_distance`](euclidean_distance.md) | Compute the Euclidean ($L_2$) distance between row X[row_x] and row Y[row_y]. |
| [`manhattan_distance`](manhattan_distance.md) | Compute the Manhattan ($L_1$ / taxicab / cityblock) distance between row X[row_x] and row Y[row_y]. |
| [`chebyshev_distance`](chebyshev_distance.md) | Compute the Chebyshev ($L_\infty$ / max) distance between row X[row_x] and row Y[row_y]. |
| [`minkowski_distance`](minkowski_distance.md) | Compute the Minkowski ($L_p$) distance between row X[row_x] and row Y[row_y]. |
| [`cosine_distance`](cosine_distance.md) | Compute the Cosine distance between row X[row_x] and row Y[row_y]. |
| [`row_distance`](row_distance.md) | Compute pairwise distance between sample row X[row_x] and sample row Y[row_y] under metric. |
| [`pairwise_distances`](pairwise_distances.md) | Compute the full pairwise distance matrix between rows of X and rows of Y. |
| [`pairwise_distances`](pairwise_distances.md) | Compute the self-pairwise distance matrix between all pairs of rows in X. |
