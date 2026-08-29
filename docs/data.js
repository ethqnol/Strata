window.STRATA_DOCS = {
  "navigation": [
    {
      "quadrant": "tutorials",
      "title": "Tutorials",
      "icon": "academic-cap",
      "description": "Learning-oriented step-by-step lessons for newcomers.",
      "items": [
        {
          "id": "tutorials/quickstart",
          "title": "Getting Started with Strata",
          "path": "tutorials/quickstart.md"
        },
        {
          "id": "tutorials/end_to_end_pipeline",
          "title": "End-to-End ML Pipelines",
          "path": "tutorials/end_to_end_pipeline.md"
        }
      ]
    },
    {
      "quadrant": "how_to",
      "title": "How-To Guides",
      "icon": "wrench-screwdriver",
      "description": "Problem-oriented guides and recipes for real-world tasks.",
      "items": [
        {
          "id": "how_to/hyperparameter_tuning",
          "title": "Hyperparameter Tuning",
          "path": "how_to/hyperparameter_tuning.md"
        },
        {
          "id": "how_to/sparse_matrix_ops",
          "title": "Sparse Matrix Operations",
          "path": "how_to/sparse_matrix_ops.md"
        },
        {
          "id": "how_to/out_of_bag_validation",
          "title": "Out-of-Bag Validation",
          "path": "how_to/out_of_bag_validation.md"
        }
      ]
    },
    {
      "quadrant": "reference",
      "title": "API Reference",
      "icon": "book-open",
      "description": "Information-oriented, auto-generated technical documentation of all modules.",
      "items": [
        {
          "id": "reference/core/index",
          "title": "strata.core",
          "path": "reference/core/index.md",
          "symbols": [
            {
              "id": "reference/core/Matrix",
              "title": "Matrix",
              "path": "reference/core/Matrix.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/MatrixView",
              "title": "MatrixView",
              "path": "reference/core/MatrixView.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/SparseMatrix",
              "title": "SparseMatrix",
              "path": "reference/core/SparseMatrix.md",
              "kind": "trait"
            },
            {
              "id": "reference/core/DatasetSplit",
              "title": "DatasetSplit",
              "path": "reference/core/DatasetSplit.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/Dataset",
              "title": "Dataset",
              "path": "reference/core/Dataset.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/SVDResult",
              "title": "SVDResult",
              "path": "reference/core/SVDResult.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/QRResult",
              "title": "QRResult",
              "path": "reference/core/QRResult.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/EigResult",
              "title": "EigResult",
              "path": "reference/core/EigResult.md",
              "kind": "struct"
            },
            {
              "id": "reference/core/matrix_to_numpy",
              "title": "matrix_to_numpy",
              "path": "reference/core/matrix_to_numpy.md",
              "kind": "function"
            },
            {
              "id": "reference/core/matrix_from_numpy",
              "title": "matrix_from_numpy",
              "path": "reference/core/matrix_from_numpy.md",
              "kind": "function"
            },
            {
              "id": "reference/core/csr_to_scipy",
              "title": "csr_to_scipy",
              "path": "reference/core/csr_to_scipy.md",
              "kind": "function"
            },
            {
              "id": "reference/core/csr_from_scipy",
              "title": "csr_from_scipy",
              "path": "reference/core/csr_from_scipy.md",
              "kind": "function"
            }
          ]
        },
        {
          "id": "reference/preprocessing/index",
          "title": "strata.preprocessing",
          "path": "reference/preprocessing/index.md",
          "symbols": [
            {
              "id": "reference/preprocessing/StandardScaler",
              "title": "StandardScaler",
              "path": "reference/preprocessing/StandardScaler.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/MinMaxScaler",
              "title": "MinMaxScaler",
              "path": "reference/preprocessing/MinMaxScaler.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/RobustScaler",
              "title": "RobustScaler",
              "path": "reference/preprocessing/RobustScaler.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/Normalizer",
              "title": "Normalizer",
              "path": "reference/preprocessing/Normalizer.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/Binarizer",
              "title": "Binarizer",
              "path": "reference/preprocessing/Binarizer.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/OneHotEncoder",
              "title": "OneHotEncoder",
              "path": "reference/preprocessing/OneHotEncoder.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/OrdinalEncoder",
              "title": "OrdinalEncoder",
              "path": "reference/preprocessing/OrdinalEncoder.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/LabelEncoder",
              "title": "LabelEncoder",
              "path": "reference/preprocessing/LabelEncoder.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/SimpleImputer",
              "title": "SimpleImputer",
              "path": "reference/preprocessing/SimpleImputer.md",
              "kind": "struct"
            },
            {
              "id": "reference/preprocessing/PolynomialFeatures",
              "title": "PolynomialFeatures",
              "path": "reference/preprocessing/PolynomialFeatures.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/linear_model/index",
          "title": "strata.linear_model",
          "path": "reference/linear_model/index.md",
          "symbols": [
            {
              "id": "reference/linear_model/LinearRegression",
              "title": "LinearRegression",
              "path": "reference/linear_model/LinearRegression.md",
              "kind": "struct"
            },
            {
              "id": "reference/linear_model/Ridge",
              "title": "Ridge",
              "path": "reference/linear_model/Ridge.md",
              "kind": "struct"
            },
            {
              "id": "reference/linear_model/LogisticRegression",
              "title": "LogisticRegression",
              "path": "reference/linear_model/LogisticRegression.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/tree/index",
          "title": "strata.tree",
          "path": "reference/tree/index.md",
          "symbols": [
            {
              "id": "reference/tree/DecisionTreeClassifier",
              "title": "DecisionTreeClassifier",
              "path": "reference/tree/DecisionTreeClassifier.md",
              "kind": "struct"
            },
            {
              "id": "reference/tree/DecisionTreeRegressor",
              "title": "DecisionTreeRegressor",
              "path": "reference/tree/DecisionTreeRegressor.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/ensemble/index",
          "title": "strata.ensemble",
          "path": "reference/ensemble/index.md",
          "symbols": [
            {
              "id": "reference/ensemble/RandomForestRegressor",
              "title": "RandomForestRegressor",
              "path": "reference/ensemble/RandomForestRegressor.md",
              "kind": "struct"
            },
            {
              "id": "reference/ensemble/RandomForestClassifier",
              "title": "RandomForestClassifier",
              "path": "reference/ensemble/RandomForestClassifier.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/cluster/index",
          "title": "strata.cluster",
          "path": "reference/cluster/index.md",
          "symbols": [
            {
              "id": "reference/cluster/KMeans",
              "title": "KMeans",
              "path": "reference/cluster/KMeans.md",
              "kind": "struct"
            },
            {
              "id": "reference/cluster/MiniBatchKMeans",
              "title": "MiniBatchKMeans",
              "path": "reference/cluster/MiniBatchKMeans.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/decomposition/index",
          "title": "strata.decomposition",
          "path": "reference/decomposition/index.md",
          "symbols": [
            {
              "id": "reference/decomposition/PCA",
              "title": "PCA",
              "path": "reference/decomposition/PCA.md",
              "kind": "struct"
            },
            {
              "id": "reference/decomposition/TruncatedSVD",
              "title": "TruncatedSVD",
              "path": "reference/decomposition/TruncatedSVD.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/neighbors/index",
          "title": "strata.neighbors",
          "path": "reference/neighbors/index.md",
          "symbols": [
            {
              "id": "reference/neighbors/sqeuclidean_distance",
              "title": "sqeuclidean_distance",
              "path": "reference/neighbors/sqeuclidean_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/euclidean_distance",
              "title": "euclidean_distance",
              "path": "reference/neighbors/euclidean_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/manhattan_distance",
              "title": "manhattan_distance",
              "path": "reference/neighbors/manhattan_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/chebyshev_distance",
              "title": "chebyshev_distance",
              "path": "reference/neighbors/chebyshev_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/minkowski_distance",
              "title": "minkowski_distance",
              "path": "reference/neighbors/minkowski_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/cosine_distance",
              "title": "cosine_distance",
              "path": "reference/neighbors/cosine_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/row_distance",
              "title": "row_distance",
              "path": "reference/neighbors/row_distance.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/pairwise_distances",
              "title": "pairwise_distances",
              "path": "reference/neighbors/pairwise_distances.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/pairwise_distances",
              "title": "pairwise_distances",
              "path": "reference/neighbors/pairwise_distances.md",
              "kind": "function"
            },
            {
              "id": "reference/neighbors/NeighborDistIdx",
              "title": "NeighborDistIdx",
              "path": "reference/neighbors/NeighborDistIdx.md",
              "kind": "struct"
            },
            {
              "id": "reference/neighbors/NearestNeighbors",
              "title": "NearestNeighbors",
              "path": "reference/neighbors/NearestNeighbors.md",
              "kind": "struct"
            },
            {
              "id": "reference/neighbors/KNeighborsClassifier",
              "title": "KNeighborsClassifier",
              "path": "reference/neighbors/KNeighborsClassifier.md",
              "kind": "struct"
            },
            {
              "id": "reference/neighbors/KNeighborsRegressor",
              "title": "KNeighborsRegressor",
              "path": "reference/neighbors/KNeighborsRegressor.md",
              "kind": "struct"
            },
            {
              "id": "reference/neighbors/KDNode",
              "title": "KDNode",
              "path": "reference/neighbors/KDNode.md",
              "kind": "struct"
            },
            {
              "id": "reference/neighbors/_AxisIndexPair",
              "title": "_AxisIndexPair",
              "path": "reference/neighbors/_AxisIndexPair.md",
              "kind": "struct"
            },
            {
              "id": "reference/neighbors/KDTree",
              "title": "KDTree",
              "path": "reference/neighbors/KDTree.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/model_selection/index",
          "title": "strata.model_selection",
          "path": "reference/model_selection/index.md",
          "symbols": [
            {
              "id": "reference/model_selection/train_test_split",
              "title": "train_test_split",
              "path": "reference/model_selection/train_test_split.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/train_test_split",
              "title": "train_test_split",
              "path": "reference/model_selection/train_test_split.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/Split",
              "title": "Split",
              "path": "reference/model_selection/Split.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/KFold",
              "title": "KFold",
              "path": "reference/model_selection/KFold.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/StratifiedKFold",
              "title": "StratifiedKFold",
              "path": "reference/model_selection/StratifiedKFold.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/TimeSeriesSplit",
              "title": "TimeSeriesSplit",
              "path": "reference/model_selection/TimeSeriesSplit.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/ShuffleSplit",
              "title": "ShuffleSplit",
              "path": "reference/model_selection/ShuffleSplit.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/StratifiedShuffleSplit",
              "title": "StratifiedShuffleSplit",
              "path": "reference/model_selection/StratifiedShuffleSplit.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/CrossValidateResult",
              "title": "CrossValidateResult",
              "path": "reference/model_selection/CrossValidateResult.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/cross_val_score",
              "title": "cross_val_score",
              "path": "reference/model_selection/cross_val_score.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_score",
              "title": "cross_val_score",
              "path": "reference/model_selection/cross_val_score.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_score",
              "title": "cross_val_score",
              "path": "reference/model_selection/cross_val_score.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_score",
              "title": "cross_val_score",
              "path": "reference/model_selection/cross_val_score.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_predict",
              "title": "cross_val_predict",
              "path": "reference/model_selection/cross_val_predict.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_predict",
              "title": "cross_val_predict",
              "path": "reference/model_selection/cross_val_predict.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_predict",
              "title": "cross_val_predict",
              "path": "reference/model_selection/cross_val_predict.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/cross_val_predict",
              "title": "cross_val_predict",
              "path": "reference/model_selection/cross_val_predict.md",
              "kind": "function"
            },
            {
              "id": "reference/model_selection/GridSearchRegressor",
              "title": "GridSearchRegressor",
              "path": "reference/model_selection/GridSearchRegressor.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/GridSearchClassifier",
              "title": "GridSearchClassifier",
              "path": "reference/model_selection/GridSearchClassifier.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/RandomizedSearchRegressor",
              "title": "RandomizedSearchRegressor",
              "path": "reference/model_selection/RandomizedSearchRegressor.md",
              "kind": "struct"
            },
            {
              "id": "reference/model_selection/RandomizedSearchClassifier",
              "title": "RandomizedSearchClassifier",
              "path": "reference/model_selection/RandomizedSearchClassifier.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/metrics/index",
          "title": "strata.metrics",
          "path": "reference/metrics/index.md",
          "symbols": [
            {
              "id": "reference/metrics/mean_squared_error",
              "title": "mean_squared_error",
              "path": "reference/metrics/mean_squared_error.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/root_mean_squared_error",
              "title": "root_mean_squared_error",
              "path": "reference/metrics/root_mean_squared_error.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/mean_absolute_error",
              "title": "mean_absolute_error",
              "path": "reference/metrics/mean_absolute_error.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/r2_score",
              "title": "r2_score",
              "path": "reference/metrics/r2_score.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/unique_labels",
              "title": "unique_labels",
              "path": "reference/metrics/unique_labels.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/accuracy_score",
              "title": "accuracy_score",
              "path": "reference/metrics/accuracy_score.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/confusion_matrix",
              "title": "confusion_matrix",
              "path": "reference/metrics/confusion_matrix.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/precision_score",
              "title": "precision_score",
              "path": "reference/metrics/precision_score.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/recall_score",
              "title": "recall_score",
              "path": "reference/metrics/recall_score.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/f1_score",
              "title": "f1_score",
              "path": "reference/metrics/f1_score.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/log_loss",
              "title": "log_loss",
              "path": "reference/metrics/log_loss.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/roc_auc_score",
              "title": "roc_auc_score",
              "path": "reference/metrics/roc_auc_score.md",
              "kind": "function"
            },
            {
              "id": "reference/metrics/silhouette_score",
              "title": "silhouette_score",
              "path": "reference/metrics/silhouette_score.md",
              "kind": "function"
            }
          ]
        },
        {
          "id": "reference/base/index",
          "title": "strata.base",
          "path": "reference/base/index.md",
          "symbols": [
            {
              "id": "reference/base/Estimator",
              "title": "Estimator",
              "path": "reference/base/Estimator.md",
              "kind": "trait"
            },
            {
              "id": "reference/base/Transformer",
              "title": "Transformer",
              "path": "reference/base/Transformer.md",
              "kind": "trait"
            },
            {
              "id": "reference/base/Regressor",
              "title": "Regressor",
              "path": "reference/base/Regressor.md",
              "kind": "trait"
            },
            {
              "id": "reference/base/Classifier",
              "title": "Classifier",
              "path": "reference/base/Classifier.md",
              "kind": "trait"
            },
            {
              "id": "reference/base/Clusterer",
              "title": "Clusterer",
              "path": "reference/base/Clusterer.md",
              "kind": "trait"
            },
            {
              "id": "reference/base/PipelineTransformer",
              "title": "PipelineTransformer",
              "path": "reference/base/PipelineTransformer.md",
              "kind": "struct"
            },
            {
              "id": "reference/base/PipelineRegressor",
              "title": "PipelineRegressor",
              "path": "reference/base/PipelineRegressor.md",
              "kind": "struct"
            },
            {
              "id": "reference/base/PipelineClassifier",
              "title": "PipelineClassifier",
              "path": "reference/base/PipelineClassifier.md",
              "kind": "struct"
            }
          ]
        },
        {
          "id": "reference/utils/index",
          "title": "strata.utils",
          "path": "reference/utils/index.md",
          "symbols": [
            {
              "id": "reference/utils/PRNG",
              "title": "PRNG",
              "path": "reference/utils/PRNG.md",
              "kind": "struct"
            },
            {
              "id": "reference/utils/sigmoid",
              "title": "sigmoid",
              "path": "reference/utils/sigmoid.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/softmax",
              "title": "softmax",
              "path": "reference/utils/softmax.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/log_sum_exp",
              "title": "log_sum_exp",
              "path": "reference/utils/log_sum_exp.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_is_fitted",
              "title": "check_is_fitted",
              "path": "reference/utils/check_is_fitted.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_floating_dtype",
              "title": "check_floating_dtype",
              "path": "reference/utils/check_floating_dtype.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_array",
              "title": "check_array",
              "path": "reference/utils/check_array.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_X_y",
              "title": "check_X_y",
              "path": "reference/utils/check_X_y.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_finite",
              "title": "check_finite",
              "path": "reference/utils/check_finite.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_consistent_length",
              "title": "check_consistent_length",
              "path": "reference/utils/check_consistent_length.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_consistent_length",
              "title": "check_consistent_length",
              "path": "reference/utils/check_consistent_length.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/check_sparse",
              "title": "check_sparse",
              "path": "reference/utils/check_sparse.md",
              "kind": "function"
            },
            {
              "id": "reference/utils/NotFittedError",
              "title": "NotFittedError",
              "path": "reference/utils/NotFittedError.md",
              "kind": "struct"
            },
            {
              "id": "reference/utils/DimensionMismatchError",
              "title": "DimensionMismatchError",
              "path": "reference/utils/DimensionMismatchError.md",
              "kind": "struct"
            },
            {
              "id": "reference/utils/ConvergenceError",
              "title": "ConvergenceError",
              "path": "reference/utils/ConvergenceError.md",
              "kind": "struct"
            },
            {
              "id": "reference/utils/InvalidParameterError",
              "title": "InvalidParameterError",
              "path": "reference/utils/InvalidParameterError.md",
              "kind": "struct"
            },
            {
              "id": "reference/utils/DataConversionError",
              "title": "DataConversionError",
              "path": "reference/utils/DataConversionError.md",
              "kind": "struct"
            }
          ]
        }
      ]
    },
    {
      "quadrant": "explanation",
      "title": "Explanation",
      "icon": "light-bulb",
      "description": "Understanding-oriented discussions on architecture, algorithms, and Mojo internals.",
      "items": [
        {
          "id": "explanation/memory_and_simd",
          "title": "Memory Model & SIMD",
          "path": "explanation/memory_and_simd.md"
        },
        {
          "id": "explanation/estimator_traits",
          "title": "Estimator Traits & Polymorphism",
          "path": "explanation/estimator_traits.md"
        },
        {
          "id": "explanation/tree_algorithms",
          "title": "Tree Splitting & Histograms",
          "path": "explanation/tree_algorithms.md"
        }
      ]
    }
  ],
  "documents": {
    "tutorials/quickstart": "# Getting Started with Strata\n\nStrata is a high-performance machine learning library written in native Mojo. It provides scikit-learn compatible estimator APIs while taking advantage of Mojo's compile-time optimizations and SIMD acceleration.\n\n---\n\n## 1. Installation\n\nStrata uses [`pixi`](https://pixi.sh) to manage the Mojo toolchain and C linear algebra dependencies (LAPACK/BLAS).\n\nClone the repository and set up the environment:\n\n```bash\ngit clone https://github.com/ethqnol/Strata.git\ncd Strata\npixi install\n```\n\nRun the test suite to verify your setup:\n\n```bash\npixi run test-ensemble\n```\n\n---\n\n## 2. Training Your First Model\n\nHere is a complete example of creating a dataset, training a `RandomForestClassifier`, and evaluating predictions.\n\nCreate a file named `main.mojo`:\n\n```mojo\nfrom strata.core.matrix import Matrix\nfrom strata.ensemble.forest import RandomForestClassifier\nfrom strata.metrics.classification import accuracy_score\n\ndef main() raises:\n    # Allocate an 8x2 floating-point feature matrix\n    var X = Matrix[DType.float64](8, 2)\n    \n    # Class 0 samples (clustered near negative values)\n    X[0, 0] = -3.0\n    X[0, 1] = -2.0\n    X[1, 0] = -2.0\n    X[1, 1] = -3.0\n    X[2, 0] = -4.0\n    X[2, 1] = -2.5\n    X[3, 0] = -2.5\n    X[3, 1] = -4.0\n\n    # Class 1 samples (clustered near positive values)\n    X[4, 0] = 3.0\n    X[4, 1] = 2.0\n    X[5, 0] = 2.0\n    X[5, 1] = 3.0\n    X[6, 0] = 4.0\n    X[6, 1] = 2.5\n    X[7, 0] = 2.5\n    X[7, 1] = 4.0\n\n    # Class labels for the 8 samples\n    var y = List[Scalar[DType.int32]]()\n    for _ in range(4):\n        y.append(0)\n    for _ in range(4):\n        y.append(1)\n\n    # Initialize the random forest classifier\n    var rf = RandomForestClassifier[DType.float64](\n        n_estimators=20,\n        max_depth=4,\n        random_state=42\n    )\n\n    # Fit model parameters on training data\n    rf.fit(X, y)\n\n    # Predict class labels and probabilities\n    var preds = rf.predict(X)\n    var proba = rf.predict_proba(X)\n    \n    # Evaluate accuracy\n    var acc = accuracy_score(y, preds)\n    print(\"Training Accuracy:\", acc)\n```\n\nRun the script:\n\n```bash\npixi run mojo run -I . main.mojo\n```\n\n---\n\n## Key Concepts\n\n- **`Matrix[dtype]`**: Strata's contiguous 2D dense matrix format. Specifying `[DType.float64]` configures numeric precision at compile time.\n- **`fit(X, y)` and `predict(X)`**: Standard estimator methods used across all classification and regression models in Strata.\n- **`predict_proba(X)`**: Returns an `(N, C)` matrix containing predicted probabilities for each class across all samples.\n\nNext, read [Composing End-to-End ML Pipelines](end_to_end_pipeline.md) to learn how to pair preprocessors and models into single reusable pipelines.\n",
    "tutorials/end_to_end_pipeline": "# Composing End-to-End ML Pipelines\n\nStrata provides pipeline wrappers like `PipelineRegressor` and `PipelineClassifier` to chain data transformers (such as `StandardScaler`) and estimators into a single object.\n\n---\n\n## How Pipelines Work in Mojo\n\nIn Python libraries like scikit-learn, pipelines use dynamic runtime dispatch to pass data between steps. In Mojo, Strata pipelines use compile-time generic types:\n\n```mojo\nPipelineRegressor[\n    TransformerT: Transformer,\n    RegressorT: Regressor,\n    target_dtype: DType\n]\n```\n\nBecause concrete types are known at compile time, calls to `transform()` and `predict()` are inlined directly without virtual table lookup overhead.\n\n---\n\n## Step-by-Step Example\n\nCreate `pipeline_demo.mojo`:\n\n```mojo\nfrom strata.core.matrix import Matrix\nfrom strata.preprocessing.scaler import StandardScaler\nfrom strata.linear_model.ridge import Ridge\nfrom strata.base.pipeline import PipelineRegressor\nfrom strata.metrics.regression import mean_squared_error, r2_score\n\ndef main() raises:\n    # 10 samples with 3 features on different scales\n    var X = Matrix[DType.float64](10, 3)\n    var y = List[Scalar[DType.float64]]()\n\n    for i in range(10):\n        X[i, 0] = Float64(i * 1000)      # High-scale feature\n        X[i, 1] = Float64(i) * 0.01      # Low-scale feature\n        X[i, 2] = Float64(i % 2)         # Binary feature\n        y.append(Float64(i * 5 + 2))     # Linear target\n\n    # Instantiate the scaler and regressor\n    var scaler = StandardScaler[DType.float64]()\n    var ridge = Ridge[DType.float64](alpha=1.0)\n\n    # Compose into a pipeline. Note the transfer operator (^) passing ownership\n    var pipe = PipelineRegressor[\n        StandardScaler[DType.float64],\n        Ridge[DType.float64],\n        DType.float64\n    ](scaler^, ridge^)\n\n    # Fit the pipeline: standardizes X, then fits Ridge on scaled features\n    pipe.fit(X, y)\n\n    # Predict: automatically scales input features before running inference\n    var preds = pipe.predict(X)\n\n    # Calculate metrics\n    var mse = mean_squared_error(y, preds)\n    var r2 = r2_score(y, preds)\n\n    print(\"MSE:\", mse)\n    print(\"R\u00b2:\", r2)\n```\n\nRun the script:\n\n```bash\npixi run mojo run -I . -Xlinker -L$CONDA_PREFIX/lib -Xlinker -llapack pipeline_demo.mojo\n```\n\n---\n\n## Ownership and Transfer Operator (`^`)\n\nIn Mojo, passing an object into a struct constructor requires explicit ownership handling:\n- `scaler^` uses Mojo's transfer operator `^` to move ownership of `scaler` into `PipelineRegressor`.\n- Once moved, the original `scaler` variable can no longer be accessed, preventing unintended mutations or double-free errors.\n",
    "how_to/hyperparameter_tuning": "# Hyperparameter Tuning with Grid & Randomized Search\n\nThis guide demonstrates how to tune model hyperparameters in Strata using `GridSearchRegressor` and `RandomizedSearchClassifier` across $K$-Fold cross-validation splits.\n\n\n---\n\n## 1. Tuning a Decision Tree with Grid Search\n\n`GridSearchRegressor` exhaustively evaluates candidate hyperparameter parameter grids:\n\n```mojo\nfrom strata.core.matrix import Matrix\nfrom strata.tree.regressor import DecisionTreeRegressor\nfrom strata.model_selection.grid_search import GridSearchRegressor\n\ndef main() raises:\n    var X = Matrix[DType.float64](20, 2)\n    var y = List[Scalar[DType.float64]](capacity=20)\n    for i in range(20):\n        X[i, 0] = Float64(i)\n        X[i, 1] = Float64(i * 2)\n        y.append(Float64(i * 3 + 1))\n\n    # Define hyperparameter grid\n    var max_depth_candidates = List[Int](1, 2, 3, 5)\n    var min_samples_split_candidates = List[Int](2, 4)\n\n    # Perform 5-fold cross-validated grid search\n    var grid_search = GridSearchRegressor[DType.float64](\n        estimator=DecisionTreeRegressor[DType.float64](),\n        cv=5,\n        max_depth_grid=max_depth_candidates,\n        min_samples_split_grid=min_samples_split_candidates\n    )\n\n    grid_search.fit(X, y)\n\n    print(\"Best Score (MSE):\", grid_search.best_score_)\n    print(\"Best max_depth:\", grid_search.best_params_[\"max_depth\"])\n\n    # Predict using the refitted best model\n    var best_preds = grid_search.predict(X)\n```\n\n---\n\n## 2. Randomized Search for High-Dimensional Parameter Spaces\n\nWhen evaluating wide continuous ranges or large forests, `RandomizedSearchClassifier` randomly samples $N$ configurations from the parameter distributions:\n\n```mojo\nfrom strata.ensemble.forest import RandomForestClassifier\nfrom strata.model_selection.randomized_search import RandomizedSearchClassifier\n\ndef run_random_search() raises:\n    # Set n_iter to evaluate a fixed budget of configurations\n    var random_search = RandomizedSearchClassifier[DType.float64](\n        estimator=RandomForestClassifier[DType.float64](),\n        n_iter=10,\n        cv=3,\n        random_state=42\n    )\n```\n\n---\n\n## Related References\n- [strata.model_selection Reference](../reference/model_selection.md)\n- [strata.tree Reference](../reference/tree.md)\n",
    "how_to/sparse_matrix_ops": "# Sparse Matrix Operations & SpMM Kernels\n\nThis guide shows how to instantiate compressed sparse formats (`CSRMatrix` and `CSCMatrix`), convert between representations in $\\mathcal{O}(\\text{nnz})$ time, and execute sparse-dense matrix multiplication (SpMM).\n\n\n---\n\n## 1. Creating a Compressed Sparse Row (CSR) Matrix\n\nA `CSRMatrix` stores non-zero elements across three flat vectors: `data`, `indices`, and `indptr`.\n\n```mojo\nfrom strata.core.csr_matrix import CSRMatrix\nfrom strata.core.matrix import Matrix\nfrom strata.core.sparse_ops import spmm\n\ndef main() raises:\n    # 3x3 sparse matrix:\n    # [ 1.0  0.0  2.0 ]\n    # [ 0.0  3.0  0.0 ]\n    # [ 4.0  0.0  5.0 ]\n\n    var data = List[Float64](1.0, 2.0, 3.0, 4.0, 5.0)\n    var indices = List[Int](0, 2, 1, 0, 2)\n    var indptr = List[Int](0, 2, 3, 5)\n\n    var csr = CSRMatrix[DType.float64](data, indices, indptr, rows=3, cols=3)\n    print(\"Non-zeros:\", csr.nnz)\n\n    # Convert to CSC format in O(nnz) time\n    var csc = csr.to_csc()\n```\n\n---\n\n## 2. Sparse-Dense Matrix Multiplication (SpMM)\n\nTo multiply a sparse matrix $\\mathbf{A} \\in \\mathbb{R}^{M \\times K}$ by a dense matrix $\\mathbf{B} \\in \\mathbb{R}^{K \\times N}$:\n\n```mojo\n    var B = Matrix[DType.float64](3, 2)\n    # Fill B...\n    \n    # Compute C = A * B\n    var C = spmm(csr, B)\n    print(\"Result shape:\", C.rows, \"x\", C.cols)\n```\n\n---\n\n## Related References\n- [strata.core Reference](../reference/core.md)\n- [strata.decomposition.TruncatedSVD](../reference/decomposition.md)\n",
    "how_to/out_of_bag_validation": "# Out-of-Bag Validation & Feature Importances\n\nIn Random Forests, every tree is trained on a bootstrap sample drawn with replacement ($\\approx 63.2\\%$ unique samples). The remaining $\\approx 36.8\\%$ of samples are **Out-of-Bag (OOB)** and can be used as an integrated test set without requiring a separate train/test split.\n\n\n---\n\n## 1. Enabling OOB Estimation\n\nSet `bootstrap=True` and `oob_score=True` in `RandomForestClassifier` or `RandomForestRegressor`:\n\n```mojo\nfrom strata.core.matrix import Matrix\nfrom strata.ensemble.forest import RandomForestClassifier\n\ndef main() raises:\n    var X = Matrix[DType.float64](50, 4)\n    var y = List[Scalar[DType.int32]](capacity=50)\n    # Populate X and y...\n\n    # Enable OOB evaluation\n    var rf = RandomForestClassifier[DType.float64](\n        n_estimators=50,\n        bootstrap=True,\n        oob_score=True,\n        random_state=42\n    )\n\n    rf.fit(X, y)\n\n    # Retrieve out-of-bag classification accuracy\n    var oob_acc = rf.get_oob_score()\n    print(\"Out-of-Bag Accuracy:\", oob_acc)\n```\n\n---\n\n## 2. Inspecting MDI Feature Importances\n\nMean Decrease in Impurity (MDI) importances measure the total normalized reduction in Gini/Entropy/MSE brought by each feature:\n\n```mojo\n    var importances = rf.get_feature_importances()\n    for j in range(X.cols):\n        print(\"Feature\", j, \"Importance:\", importances[j])\n```\n\n---\n\n## Related References\n- [strata.ensemble Reference](../reference/ensemble.md)\n- [Explanation: Tree Algorithms & MDI Mechanics](../explanation/tree_algorithms.md)\n",
    "reference/core/index": "# `strata.core`\n\nFundamental 2D dense Matrix, MatrixView, CSRMatrix, CSCMatrix sparse representations, Dataset containers, and hardware-accelerated BLAS/LAPACK solvers.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`Matrix`](Matrix.md) | Dense 2D row-major matrix container with striding and view support. |\n| [`MatrixView`](MatrixView.md) | Non-owning 2D view over a contiguous or strided matrix memory buffer. |\n| [`DatasetSplit`](DatasetSplit.md) | Container holding train and test partitions of a Dataset. |\n| [`Dataset`](Dataset.md) | Machine learning dataset container pairing a feature matrix with targets. |\n| [`SVDResult`](SVDResult.md) | Result of Singular Value Decomposition ($A = U \\Sigma V^T$). |\n| [`QRResult`](QRResult.md) | Result of QR Decomposition ($A = Q R$). |\n| [`EigResult`](EigResult.md) | Result of Symmetric Eigenvalue Decomposition ($A V = V \\Lambda$). |\n\n## Traits\n\n| Trait | Description |\n| :--- | :--- |\n| [`SparseMatrix`](SparseMatrix.md) | Base interface trait for 2D sparse matrix representations. |\n\n## Functions\n\n| Function | Description |\n| :--- | :--- |\n| [`matrix_to_numpy`](matrix_to_numpy.md) | Converts a Strata Matrix[dtype] to a NumPy 2D array. |\n| [`matrix_from_numpy`](matrix_from_numpy.md) | Converts a 2D NumPy ndarray to a Strata Matrix[dtype]. |\n| [`csr_to_scipy`](csr_to_scipy.md) | Converts a Strata CSRMatrix[dtype] to a scipy.sparse.csr_matrix. |\n| [`csr_from_scipy`](csr_from_scipy.md) | Converts a scipy.sparse.csr_matrix to a Strata CSRMatrix[dtype]. |\n",
    "reference/core/Matrix": "# `Matrix`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `ArrayLike, Copyable, Movable, Writable`  \n**Source**: [`strata/core/matrix.mojo`](file:////home/ewu/Code/Strata/strata/core/matrix.mojo)\n\n```mojo\nstruct Matrix[dtype: DType = DType.float64](ArrayLike, Copyable, Movable, Writable)\n```\n\n```mojo\nfrom strata.core import Matrix\n```\n\n**Dense 2D row-major matrix container with striding and view support.**\n\nProvides contiguous buffer allocation, SIMD-compatible row-major layout,\nslicing, element-wise arithmetic, and BLAS/LAPACK interop.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`dtype`** | Numerical data type of matrix elements. Default DType.float64. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`rows`** | Number of matrix rows ($N$). |\n| **`cols`** | Number of matrix columns ($D$). |\n| **`data`** | Flat 1D buffer of matrix elements in row-major order. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Matrix.zeros()`](#zeros) | Create a zero-filled Matrix of shape (rows, cols). |\n| [`Matrix.ones()`](#ones) | Create a one-filled Matrix of shape (rows, cols). |\n| [`Matrix.eye()`](#eye) | Create an identity Matrix of shape (n, n). |\n| [`Matrix.from_numpy()`](#from_numpy) | \u2014 |\n| [`Matrix.to_numpy()`](#to_numpy) | \u2014 |\n| [`Matrix.num_rows()`](#num_rows) | \u2014 |\n| [`Matrix.num_cols()`](#num_cols) | \u2014 |\n| [`Matrix.num_elements()`](#num_elements) | \u2014 |\n| [`Matrix.shape()`](#shape) | \u2014 |\n| [`Matrix.view()`](#view) | \u2014 |\n| [`Matrix.slice_rows()`](#slice_rows) | \u2014 |\n| [`Matrix.slice_cols()`](#slice_cols) | \u2014 |\n| [`Matrix.slice_2d()`](#slice_2d) | \u2014 |\n| [`Matrix.row()`](#row) | \u2014 |\n| [`Matrix.col()`](#col) | \u2014 |\n| [`Matrix.cast()`](#cast) | Promotes or converts the Matrix elements to target_dtype. |\n| [`Matrix.transpose()`](#transpose) | \u2014 |\n| [`Matrix.dot()`](#dot) | \u2014 |\n| [`Matrix.dot_vec()`](#dot_vec) | \u2014 |\n| [`Matrix.mean_along_axis_0()`](#mean_along_axis_0) | \u2014 |\n| [`Matrix.std_along_axis_0()`](#std_along_axis_0) | \u2014 |\n| [`Matrix.write_to()`](#write_to) | \u2014 |\n\n---\n\n## Method Details\n\n### `Matrix.zeros()`\n\n```mojo\ndef zeros(rows: Int, cols: Int) -> Self\n```\n\nCreate a zero-filled Matrix of shape (rows, cols).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`rows`** | `Int` | \u2014 |\n| **`cols`** | `Int` | \u2014 |\n\n**Returns**: `Self`\n\n---\n\n### `Matrix.ones()`\n\n```mojo\ndef ones(rows: Int, cols: Int) -> Self\n```\n\nCreate a one-filled Matrix of shape (rows, cols).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`rows`** | `Int` | \u2014 |\n| **`cols`** | `Int` | \u2014 |\n\n**Returns**: `Self`\n\n---\n\n### `Matrix.eye()`\n\n```mojo\ndef eye(n: Int) -> Self\n```\n\nCreate an identity Matrix of shape (n, n).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`n`** | `Int` | \u2014 |\n\n**Returns**: `Self`\n\n---\n\n### `Matrix.from_numpy()`\n\n```mojo\ndef from_numpy(np_arr: PythonObject) -> Self\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`np_arr`** | `PythonObject` | \u2014 |\n\n**Returns**: `Self`\n\n---\n\n### `Matrix.to_numpy()`\n\n```mojo\ndef to_numpy(self) -> PythonObject\n```\n\n**Returns**: `PythonObject`\n\n---\n\n### `Matrix.num_rows()`\n\n```mojo\ndef num_rows(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `Matrix.num_cols()`\n\n```mojo\ndef num_cols(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `Matrix.num_elements()`\n\n```mojo\ndef num_elements(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `Matrix.shape()`\n\n```mojo\ndef shape(self) -> Tuple[Int, Int]\n```\n\n**Returns**: `Tuple[Int, Int]`\n\n---\n\n### `Matrix.view()`\n\n```mojo\ndef view(ref self) -> MatrixView[Self.dtype, origin_of(self.data)]\n```\n\n**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`\n\n---\n\n### `Matrix.slice_rows()`\n\n```mojo\ndef slice_rows(ref self, start_row: Int, end_row: Int) -> MatrixView[Self.dtype, origin_of(self.data)]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`start_row`** | `Int` | \u2014 |\n| **`end_row`** | `Int` | \u2014 |\n\n**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`\n\n---\n\n### `Matrix.slice_cols()`\n\n```mojo\ndef slice_cols(ref self, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, origin_of(self.data)]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`start_col`** | `Int` | \u2014 |\n| **`end_col`** | `Int` | \u2014 |\n\n**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`\n\n---\n\n### `Matrix.slice_2d()`\n\n```mojo\ndef slice_2d(ref self, start_row: Int, end_row: Int, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, origin_of(self.data)]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`start_row`** | `Int` | \u2014 |\n| **`end_row`** | `Int` | \u2014 |\n| **`start_col`** | `Int` | \u2014 |\n| **`end_col`** | `Int` | \u2014 |\n\n**Returns**: `MatrixView[Self.dtype, origin_of(self.data)]`\n\n---\n\n### `Matrix.row()`\n\n```mojo\ndef row(self, r: Int) -> List[Scalar[Self.dtype]]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`r`** | `Int` | \u2014 |\n\n**Returns**: `List[Scalar[Self.dtype]]`\n\n---\n\n### `Matrix.col()`\n\n```mojo\ndef col(self, c: Int) -> List[Scalar[Self.dtype]]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`c`** | `Int` | \u2014 |\n\n**Returns**: `List[Scalar[Self.dtype]]`\n\n---\n\n### `Matrix.cast()`\n\n```mojo\ndef cast[target_dtype: DType](self) -> Matrix[target_dtype]\n```\n\nPromotes or converts the Matrix elements to target_dtype.\n\n**Returns**: `Matrix[target_dtype]`\n\n---\n\n### `Matrix.transpose()`\n\n```mojo\ndef transpose(self) -> Self\n```\n\n**Returns**: `Self`\n\n---\n\n### `Matrix.dot()`\n\n```mojo\ndef dot(self, other: Self) -> Self\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`other`** | `Self` | \u2014 |\n\n**Returns**: `Self`\n\n---\n\n### `Matrix.dot_vec()`\n\n```mojo\ndef dot_vec(self, vec: List[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`vec`** | `List[Scalar[Self.dtype]]` | \u2014 |\n\n**Returns**: `List[Scalar[Self.dtype]]`\n\n---\n\n### `Matrix.mean_along_axis_0()`\n\n```mojo\ndef mean_along_axis_0(self) -> List[Scalar[Self.dtype]]\n```\n\n**Returns**: `List[Scalar[Self.dtype]]`\n\n---\n\n### `Matrix.std_along_axis_0()`\n\n```mojo\ndef std_along_axis_0(self, means: List[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`means`** | `List[Scalar[Self.dtype]]` | \u2014 |\n\n**Returns**: `List[Scalar[Self.dtype]]`\n\n---\n\n### `Matrix.write_to()`\n\n```mojo\ndef write_to(self, mut writer: Some[Writer])\n```\n---\n\n## Example\n\n```mojo\nfrom strata.core import Matrix\n\nvar A = Matrix[DType.float64](2, 3, fill=1.0)\nvar B = Matrix[DType.float64].eye(3)\nvar C = A.dot(B)\n```\n",
    "reference/core/MatrixView": "# `MatrixView`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `ArrayLike, Copyable, Movable`  \n**Source**: [`strata/core/view.mojo`](file:////home/ewu/Code/Strata/strata/core/view.mojo)\n\n```mojo\nstruct MatrixView[dtype: DType, origin: Origin](ArrayLike, Copyable, Movable)\n```\n\n```mojo\nfrom strata.core import MatrixView\n```\n\n**Non-owning 2D view over a contiguous or strided matrix memory buffer.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`MatrixView.num_rows()`](#num_rows) | \u2014 |\n| [`MatrixView.num_cols()`](#num_cols) | \u2014 |\n| [`MatrixView.num_elements()`](#num_elements) | \u2014 |\n| [`MatrixView.shape()`](#shape) | \u2014 |\n| [`MatrixView.slice_rows()`](#slice_rows) | \u2014 |\n| [`MatrixView.slice_cols()`](#slice_cols) | \u2014 |\n| [`MatrixView.slice_2d()`](#slice_2d) | \u2014 |\n| [`MatrixView.to_matrix()`](#to_matrix) | \u2014 |\n\n---\n\n## Method Details\n\n### `MatrixView.num_rows()`\n\n```mojo\ndef num_rows(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `MatrixView.num_cols()`\n\n```mojo\ndef num_cols(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `MatrixView.num_elements()`\n\n```mojo\ndef num_elements(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `MatrixView.shape()`\n\n```mojo\ndef shape(self) -> Tuple[Int, Int]\n```\n\n**Returns**: `Tuple[Int, Int]`\n\n---\n\n### `MatrixView.slice_rows()`\n\n```mojo\ndef slice_rows(self, start_row: Int, end_row: Int) -> MatrixView[Self.dtype, Self.origin]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`start_row`** | `Int` | \u2014 |\n| **`end_row`** | `Int` | \u2014 |\n\n**Returns**: `MatrixView[Self.dtype, Self.origin]`\n\n---\n\n### `MatrixView.slice_cols()`\n\n```mojo\ndef slice_cols(self, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, Self.origin]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`start_col`** | `Int` | \u2014 |\n| **`end_col`** | `Int` | \u2014 |\n\n**Returns**: `MatrixView[Self.dtype, Self.origin]`\n\n---\n\n### `MatrixView.slice_2d()`\n\n```mojo\ndef slice_2d(self, start_row: Int, end_row: Int, start_col: Int, end_col: Int) -> MatrixView[Self.dtype, Self.origin]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`start_row`** | `Int` | \u2014 |\n| **`end_row`** | `Int` | \u2014 |\n| **`start_col`** | `Int` | \u2014 |\n| **`end_col`** | `Int` | \u2014 |\n\n**Returns**: `MatrixView[Self.dtype, Self.origin]`\n\n---\n\n### `MatrixView.to_matrix()`\n\n```mojo\ndef to_matrix(self) -> Matrix[Self.dtype]\n```\n\n**Returns**: `Matrix[Self.dtype]`\n\n---\n",
    "reference/core/SparseMatrix": "# `SparseMatrix`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `trait`  \n**Source**: [`strata/core/sparse.mojo`](file:////home/ewu/Code/Strata/strata/core/sparse.mojo)\n\n```mojo\ntrait SparseMatrix\n```\n\n```mojo\nfrom strata.core import SparseMatrix\n```\n\n**Base interface trait for 2D sparse matrix representations.**\n\nProvides common dimension queries (`num_rows`, `num_cols`) and structural\nsparsity counts (`nnz`) for Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) formats.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`SparseMatrix.num_rows()`](#num_rows) | \u2014 |\n| [`SparseMatrix.num_cols()`](#num_cols) | \u2014 |\n| [`SparseMatrix.nnz()`](#nnz) | \u2014 |\n\n---\n\n## Method Details\n\n### `SparseMatrix.num_rows()`\n\n```mojo\ndef num_rows(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `SparseMatrix.num_cols()`\n\n```mojo\ndef num_cols(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `SparseMatrix.nnz()`\n\n```mojo\ndef nnz(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n",
    "reference/core/DatasetSplit": "# `DatasetSplit`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/core/dataset.mojo`](file:////home/ewu/Code/Strata/strata/core/dataset.mojo)\n\n```mojo\nstruct DatasetSplit[feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Movable)\n```\n\n```mojo\nfrom strata.core import DatasetSplit\n```\n\n**Container holding train and test partitions of a Dataset.**\n",
    "reference/core/Dataset": "# `Dataset`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/core/dataset.mojo`](file:////home/ewu/Code/Strata/strata/core/dataset.mojo)\n\n```mojo\nstruct Dataset[feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.core import Dataset\n```\n\n**Machine learning dataset container pairing a feature matrix with targets.**\n\nEncapsulates 2D feature observations, 1D target labels/values, and optional\nfeature/target name metadata.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`feat_dtype`** | Data type of the feature matrix (default: Float64). |\n| **`target_dtype`** | Data type of the target values (default: Float64). |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`records`** | Feature matrix of shape $(N, D)$. |\n| **`targets`** | Target label/value vector of length $N$. |\n| **`feature_names`** | List of feature names of length $D$. |\n| **`target_names`** | List of class or target variable names. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Dataset.n_samples()`](#n_samples) | \u2014 |\n| [`Dataset.n_features()`](#n_features) | \u2014 |\n| [`Dataset.split_with_ratio()`](#split_with_ratio) | \u2014 |\n\n---\n\n## Method Details\n\n### `Dataset.n_samples()`\n\n```mojo\ndef n_samples(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `Dataset.n_features()`\n\n```mojo\ndef n_features(self) -> Int\n```\n\n**Returns**: `Int`\n\n---\n\n### `Dataset.split_with_ratio()`\n\n```mojo\ndef split_with_ratio(self, ratio: Float64 = 0.25, shuffle: Bool = True, seed: Int = 42) -> DatasetSplit[Self.feat_dtype, Self.target_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`ratio`** | `Float64` | \u2014 |\n| **`shuffle`** | `Bool` | \u2014 |\n| **`seed`** | `Int` | \u2014 |\n\n**Returns**: `DatasetSplit[Self.feat_dtype, Self.target_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.core import Matrix, Dataset\n\nvar X = Matrix[DType.float64](100, 4, fill=1.0)\nvar y = List[Scalar[DType.float64]](capacity=100)\nvar ds = Dataset(X^, y^)\nvar split = ds.split_with_ratio(ratio=0.2)\n```\n",
    "reference/core/SVDResult": "# `SVDResult`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)\n\n```mojo\nstruct SVDResult[dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.core import SVDResult\n```\n\n**Result of Singular Value Decomposition ($A = U \\Sigma V^T$).**\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`U`** | Left singular vectors matrix of shape $(M, K)$. |\n| **`S`** | Singular values vector of length $K$ in descending order. |\n| **`Vt`** | Right singular vectors transposed matrix of shape $(K, N)$. |\n",
    "reference/core/QRResult": "# `QRResult`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)\n\n```mojo\nstruct QRResult[dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.core import QRResult\n```\n\n**Result of QR Decomposition ($A = Q R$).**\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`Q`** | Orthogonal matrix of shape $(M, K)$. |\n| **`R`** | Upper triangular matrix of shape $(K, N)$. |\n",
    "reference/core/EigResult": "# `EigResult`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/core/linalg.mojo`](file:////home/ewu/Code/Strata/strata/core/linalg.mojo)\n\n```mojo\nstruct EigResult[dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.core import EigResult\n```\n\n**Result of Symmetric Eigenvalue Decomposition ($A V = V \\Lambda$).**\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`eigenvalues`** | Real eigenvalues vector of length $N$ in ascending order. |\n| **`eigenvectors`** | Eigenvector matrix of shape $(N, N)$ with columns representing eigenvectors. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`EigResult.gemm()`](#gemm) | Compute dense matrix product $C = A B$. |\n| [`EigResult.dense_dot_vec()`](#dense_dot_vec) | Dense matrix-vector product: y = A @ x + bias. |\n| [`EigResult.svd()`](#svd) | Computes the Singular Value Decomposition (SVD) of a matrix A = U * S * Vt. |\n| [`EigResult.qr()`](#qr) | Computes the QR Decomposition of matrix A = Q * R using LAPACK Householder reflectors (dgeqrf/dorgqr). |\n| [`EigResult.cholesky()`](#cholesky) | Computes the Cholesky factorization of a symmetric positive-definite matrix A = L * L^T. |\n| [`EigResult.lstsq()`](#lstsq) | Solve linear least-squares problem $\\min_x \\|A x - b\\|_2$ using SVD. |\n| [`EigResult.solve()`](#solve) | Solves a square linear system A * x = b using LU decomposition (dgesv/sgesv). |\n| [`EigResult.solve_cholesky()`](#solve_cholesky) | Solves a symmetric positive definite linear system A * x = b using Cholesky (dposv/sposv). |\n| [`EigResult.inv()`](#inv) | Computes the multiplicative inverse of a square matrix A using LU decomposition (dgetrf/dgetri). |\n| [`EigResult.norm()`](#norm) | Computes the Frobenius matrix norm: ||A||_F = sqrt(sum(A_ij^2)). |\n| [`EigResult.eigh()`](#eigh) | Computes the eigenvalues and eigenvectors of a real symmetric matrix. |\n\n---\n\n## Method Details\n\n### `EigResult.gemm()`\n\n```mojo\ndef gemm[dtype: DType = DType.float64](A: Matrix[dtype], B: Matrix[dtype]) -> Matrix[dtype]\n```\n\nCompute dense matrix product $C = A B$.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`B`** | `Matrix[dtype]` | \u2014 |\n\n**Returns**: `Matrix[dtype]` \u2014 Matrix[dtype]: Output matrix product $C$ of shape $(M, N)$.\n\n---\n\n### `EigResult.dense_dot_vec()`\n\n```mojo\ndef dense_dot_vec[dtype: DType = DType.float64](A: Matrix[dtype], x: List[Scalar[dtype]], bias: Scalar[dtype] = 0) -> List[Scalar[dtype]]\n```\n\nDense matrix-vector product: y = A @ x + bias.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`x`** | `List[Scalar[dtype]]` | \u2014 |\n| **`bias`** | `Scalar[dtype]` | \u2014 |\n\n**Returns**: `List[Scalar[dtype]]`\n\n---\n\n### `EigResult.svd()`\n\n```mojo\ndef svd[dtype: DType = DType.float64](A: Matrix[dtype], full_matrices: Bool = False) -> SVDResult[dtype]\n```\n\nComputes the Singular Value Decomposition (SVD) of a matrix A = U * S * Vt.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`full_matrices`** | `Bool` | \u2014 |\n\n**Returns**: `SVDResult[dtype]`\n\n---\n\n### `EigResult.qr()`\n\n```mojo\ndef qr[dtype: DType = DType.float64](A: Matrix[dtype]) -> QRResult[dtype]\n```\n\nComputes the QR Decomposition of matrix A = Q * R using LAPACK Householder reflectors (dgeqrf/dorgqr).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n\n**Returns**: `QRResult[dtype]`\n\n---\n\n### `EigResult.cholesky()`\n\n```mojo\ndef cholesky[dtype: DType = DType.float64](A: Matrix[dtype], lower: Bool = True) -> Matrix[dtype]\n```\n\nComputes the Cholesky factorization of a symmetric positive-definite matrix A = L * L^T.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`lower`** | `Bool` | \u2014 |\n\n**Returns**: `Matrix[dtype]`\n\n---\n\n### `EigResult.lstsq()`\n\n```mojo\ndef lstsq[dtype: DType = DType.float64](A: Matrix[dtype], b: List[Scalar[dtype]], rcond: Float64 = -1.0) -> List[Scalar[dtype]]\n```\n\nSolve linear least-squares problem $\\min_x \\|A x - b\\|_2$ using SVD.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`b`** | `List[Scalar[dtype]]` | \u2014 |\n| **`rcond`** | `Float64` | \u2014 |\n\n**Returns**: `List[Scalar[dtype]]` \u2014 List[Scalar[dtype]]: Least-squares solution vector $x$ of length $N$.\n\n---\n\n### `EigResult.solve()`\n\n```mojo\ndef solve[dtype: DType = DType.float64](A: Matrix[dtype], b: List[Scalar[dtype]]) -> List[Scalar[dtype]]\n```\n\nSolves a square linear system A * x = b using LU decomposition (dgesv/sgesv).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`b`** | `List[Scalar[dtype]]` | \u2014 |\n\n**Returns**: `List[Scalar[dtype]]`\n\n---\n\n### `EigResult.solve_cholesky()`\n\n```mojo\ndef solve_cholesky[dtype: DType = DType.float64](A: Matrix[dtype], b: List[Scalar[dtype]], lower: Bool = True) -> List[ Scalar[dtype] ]\n```\n\nSolves a symmetric positive definite linear system A * x = b using Cholesky (dposv/sposv).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`b`** | `List[Scalar[dtype]]` | \u2014 |\n| **`lower`** | `Bool` | \u2014 |\n\n**Returns**: `List[ Scalar[dtype] ]`\n\n---\n\n### `EigResult.inv()`\n\n```mojo\ndef inv[dtype: DType = DType.float64](A: Matrix[dtype]) -> Matrix[dtype]\n```\n\nComputes the multiplicative inverse of a square matrix A using LU decomposition (dgetrf/dgetri).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n\n**Returns**: `Matrix[dtype]`\n\n---\n\n### `EigResult.norm()`\n\n```mojo\ndef norm[dtype: DType = DType.float64](A: Matrix[dtype], ord: String = \"fro\") -> Float64\n```\n\nComputes the Frobenius matrix norm: ||A||_F = sqrt(sum(A_ij^2)).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`ord`** | `String` | \u2014 |\n\n**Returns**: `Float64`\n\n---\n\n### `EigResult.eigh()`\n\n```mojo\ndef eigh[dtype: DType = DType.float64](A: Matrix[dtype], UPLO: String = \"L\") -> EigResult[dtype]\n```\n\nComputes the eigenvalues and eigenvectors of a real symmetric matrix.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`A`** | `Matrix[dtype]` | \u2014 |\n| **`UPLO`** | `String` | \u2014 |\n\n**Returns**: `EigResult[dtype]`\n\n---\n",
    "reference/core/matrix_to_numpy": "# `matrix_to_numpy`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)\n\n```mojo\ndef matrix_to_numpy[dtype: DType](matrix: Matrix[dtype]) -> PythonObject\n```\n\n```mojo\nfrom strata.core import matrix_to_numpy\n```\n\n**Converts a Strata Matrix[dtype] to a NumPy 2D array.**\n",
    "reference/core/matrix_from_numpy": "# `matrix_from_numpy`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)\n\n```mojo\ndef matrix_from_numpy[dtype: DType = DType.float64](np_arr: PythonObject) -> Matrix[dtype]\n```\n\n```mojo\nfrom strata.core import matrix_from_numpy\n```\n\n**Converts a 2D NumPy ndarray to a Strata Matrix[dtype].**\n",
    "reference/core/csr_to_scipy": "# `csr_to_scipy`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)\n\n```mojo\ndef csr_to_scipy[dtype: DType](csr: CSRMatrix[dtype]) -> PythonObject\n```\n\n```mojo\nfrom strata.core import csr_to_scipy\n```\n\n**Converts a Strata CSRMatrix[dtype] to a scipy.sparse.csr_matrix.**\n",
    "reference/core/csr_from_scipy": "# `csr_from_scipy`\n\n**Module**: [`strata.core`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/core/interop.mojo`](file:////home/ewu/Code/Strata/strata/core/interop.mojo)\n\n```mojo\ndef csr_from_scipy[dtype: DType = DType.float64](sp_arr: PythonObject) -> CSRMatrix[dtype]\n```\n\n```mojo\nfrom strata.core import csr_from_scipy\n```\n\n**Converts a scipy.sparse.csr_matrix to a Strata CSRMatrix[dtype].**\n",
    "reference/preprocessing/index": "# `strata.preprocessing`\n\nStandardScaler, MinMaxScaler, RobustScaler, Normalizer, Binarizer, OneHotEncoder, OrdinalEncoder, LabelEncoder, SimpleImputer, and PolynomialFeatures with streaming SIMD statistics.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`StandardScaler`](StandardScaler.md) | Standardize features by removing the mean and scaling to unit variance. |\n| [`MinMaxScaler`](MinMaxScaler.md) | Transform features by scaling each feature to a specified range. |\n| [`RobustScaler`](RobustScaler.md) | Scale features using statistics that are robust to outliers. |\n| [`Normalizer`](Normalizer.md) | Normalize samples individually to unit norm. |\n| [`Binarizer`](Binarizer.md) | Binarize feature values according to a threshold. |\n| [`OneHotEncoder`](OneHotEncoder.md) | Encode categorical features as a one-hot numeric array. |\n| [`OrdinalEncoder`](OrdinalEncoder.md) | Encode categorical features as an integer array. |\n| [`LabelEncoder`](LabelEncoder.md) | Encode target labels with value between 0 and n_classes-1. |\n| [`SimpleImputer`](SimpleImputer.md) | Univariate imputer for completing missing values with simple statistics. |\n| [`PolynomialFeatures`](PolynomialFeatures.md) | Generate polynomial and interaction features. |\n",
    "reference/preprocessing/StandardScaler": "# `StandardScaler`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)\n\n```mojo\nstruct StandardScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import StandardScaler\n```\n\n**Standardize features by removing the mean and scaling to unit variance.**\n\nThe standard score of a sample $x$ is calculated as:\n$$\nz = \\frac{x - \\mu}{\\sigma}\n$$\nwhere $\\mu$ is the mean of the training samples and $\\sigma$ is the standard deviation.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`with_mean`** | If True, center the data before scaling. Default True. |\n| **`with_std`** | If True, scale the data to unit variance (unit standard deviation). Default True. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`mean_`** | Mean value for each feature in the training set. |\n| **`scale_`** | Per-feature standard deviation scaling factor. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`StandardScaler.fit()`](#fit) | \u2014 |\n| [`StandardScaler.transform()`](#transform) | \u2014 |\n| [`StandardScaler.fit_transform()`](#fit_transform) | \u2014 |\n\n---\n\n## Method Details\n\n### `StandardScaler.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `StandardScaler.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `StandardScaler.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import StandardScaler\nfrom strata.core import Matrix\n\nvar scaler = StandardScaler[DType.float64]()\nscaler.fit(X_train)\nvar X_scaled = scaler.transform(X_train)\n```\n",
    "reference/preprocessing/MinMaxScaler": "# `MinMaxScaler`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)\n\n```mojo\nstruct MinMaxScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import MinMaxScaler\n```\n\n**Transform features by scaling each feature to a specified range.**\n\nScales and translates each feature individually such that it is in the given\nrange on the training set, e.g. between zero and one:\n$$\nx_{\\text{scaled}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} \\cdot (\\text{max} - \\text{min}) + \\text{min}\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`feature_range_min`** | Lower bound of the desired transformed range. Default 0.0. |\n| **`feature_range_max`** | Upper bound of the desired transformed range. Default 1.0. |\n| **`clip`** | Whether to clip transformed values to the feature range. Default False. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`data_min_`** | Per-feature minimum seen in the training data. |\n| **`data_max_`** | Per-feature maximum seen in the training data. |\n| **`data_range_`** | Per-feature range ($x_{\\max} - x_{\\min}$) seen in the data. |\n| **`scale_`** | Per-feature relative scaling factor. |\n| **`min_`** | Per-feature minimum adjustment. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`MinMaxScaler.fit()`](#fit) | \u2014 |\n| [`MinMaxScaler.transform()`](#transform) | \u2014 |\n| [`MinMaxScaler.fit_transform()`](#fit_transform) | \u2014 |\n| [`MinMaxScaler.inverse_transform()`](#inverse_transform) | Undoes the scaling of X according to the fitted feature range. |\n\n---\n\n## Method Details\n\n### `MinMaxScaler.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `MinMaxScaler.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `MinMaxScaler.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `MinMaxScaler.inverse_transform()`\n\n```mojo\ndef inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\nUndoes the scaling of X according to the fitted feature range.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import MinMaxScaler\nfrom strata.core import Matrix\n\nvar scaler = MinMaxScaler[DType.float64](feature_range_min=0.0, feature_range_max=1.0)\nscaler.fit(X_train)\nvar X_scaled = scaler.transform(X_train)\n```\n",
    "reference/preprocessing/RobustScaler": "# `RobustScaler`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/scaler.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/scaler.mojo)\n\n```mojo\nstruct RobustScaler[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import RobustScaler\n```\n\n**Scale features using statistics that are robust to outliers.**\n\nCenters the data on the median and scales by the Interquartile Range (IQR):\n$$\nx_{\\text{scaled}} = \\frac{x - \\text{median}}{\\text{IQR}}\n$$\nwhere $\\text{IQR} = Q_3 - Q_1$ (by default 75th percentile minus 25th percentile).\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`with_centering`** | If True, center the data before scaling by subtracting the median. Default True. |\n| **`with_scaling`** | If True, scale the data to interquartile range. Default True. |\n| **`quantile_min`** | Lower quantile percentage of the scaling range ($0 <= q_{\\min} < q_{\\max} <= 100$). Default 25.0. |\n| **`quantile_max`** | Upper quantile percentage of the scaling range. Default 75.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`center_`** | Median value for each feature in the training set. |\n| **`scale_`** | Interquartile range scaling factor for each feature. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`RobustScaler.fit()`](#fit) | \u2014 |\n| [`RobustScaler.transform()`](#transform) | \u2014 |\n| [`RobustScaler.fit_transform()`](#fit_transform) | \u2014 |\n| [`RobustScaler.inverse_transform()`](#inverse_transform) | Undoes the centering and scaling of X. |\n\n---\n\n## Method Details\n\n### `RobustScaler.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `RobustScaler.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `RobustScaler.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `RobustScaler.inverse_transform()`\n\n```mojo\ndef inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\nUndoes the centering and scaling of X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import RobustScaler\nfrom strata.core import Matrix\n\nvar scaler = RobustScaler[DType.float64]()\nscaler.fit(X_train)\nvar X_scaled = scaler.transform(X_train)\n```\n",
    "reference/preprocessing/Normalizer": "# `Normalizer`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/normalizer.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/normalizer.mojo)\n\n```mojo\nstruct Normalizer[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import Normalizer\n```\n\n**Normalize samples individually to unit norm.**\n\nEach sample (i.e. each row of the data matrix) with at least one non-zero\ncomponent is rescaled independently of other samples so that its norm\n($L_1$, $L_2$ or $\\text{max}$) equals one.\n$$\nx_{\\text{norm}} = \\frac{x}{\\|x\\|_p}\n$$\nwhere $\\|x\\|_p$ is the chosen vector norm:\n- $L_1$: $\\|x\\|_1 = \\sum_j |x_j|$\n- $L_2$: $\\|x\\|_2 = \\sqrt{\\sum_j x_j^2}$\n- $\\text{max}$: $\\|x\\|_\\infty = \\max_j |x_j|$\nRows of all zeros remain all zeros.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`norm`** | The norm to use to normalize each non-zero sample ('l1', 'l2', or 'max'). Default 'l2'. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Normalizer.fit()`](#fit) | Fit the transformer on feature matrix X. |\n| [`Normalizer.transform()`](#transform) | Scale each non-zero sample in X to unit norm. |\n| [`Normalizer.fit_transform()`](#fit_transform) | Fit to data, then transform it. |\n\n---\n\n## Method Details\n\n### `Normalizer.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the transformer on feature matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `Normalizer.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nScale each non-zero sample in X to unit norm.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Normalized feature matrix.\n\n---\n\n### `Normalizer.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nFit to data, then transform it.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Normalized feature matrix.\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import Normalizer\nfrom strata.core import Matrix\n\nvar normalizer = Normalizer[DType.float64](norm=\"l2\")\nnormalizer.fit(X_train)\nvar X_norm = normalizer.transform(X_train)\n```\n",
    "reference/preprocessing/Binarizer": "# `Binarizer`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/binarizer.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/binarizer.mojo)\n\n```mojo\nstruct Binarizer[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import Binarizer\n```\n\n**Binarize feature values according to a threshold.**\n\nValues strictly greater than the threshold map to 1, while values less than\nor equal to the threshold map to 0:\n$$\nx_{\\text{bin}} = \\begin{cases} 1 & \\text{if } x > \\text{threshold} \\\\ 0 & \\text{otherwise} \\end{cases}\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`threshold`** | Feature values greater than this are mapped to 1. Default 0.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Binarizer.fit()`](#fit) | \u2014 |\n| [`Binarizer.transform()`](#transform) | \u2014 |\n| [`Binarizer.fit_transform()`](#fit_transform) | \u2014 |\n\n---\n\n## Method Details\n\n### `Binarizer.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `Binarizer.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `Binarizer.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import Binarizer\nfrom strata.core import Matrix\n\nvar binarizer = Binarizer[DType.float64](threshold=0.5)\nbinarizer.fit(X_train)\nvar X_bin = binarizer.transform(X_train)\n```\n",
    "reference/preprocessing/OneHotEncoder": "# `OneHotEncoder`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/encoders.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/encoders.mojo)\n\n```mojo\nstruct OneHotEncoder[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import OneHotEncoder\n```\n\n**Encode categorical features as a one-hot numeric array.**\n\nThe input to this transformer should be a 2D matrix of integer or float\ncategorical features. The features are encoded using a one-hot (also known as\n'one-of-K' or 'dummy') encoding scheme.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`drop`** | Category dropping strategy ('none', 'first', 'if_binary'). Default 'none'. |\n| **`handle_unknown`** | Behavior for unseen categories during transform ('error', 'ignore'). Default 'error'. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`categories_`** | Categories of each feature determined during fitting. |\n| **`drop_idx_`** | Indices of dropped categories for each feature. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`OneHotEncoder.n_features_out()`](#n_features_out) | Number of indicator columns produced by transform. |\n| [`OneHotEncoder.fit()`](#fit) | \u2014 |\n| [`OneHotEncoder.transform()`](#transform) | \u2014 |\n| [`OneHotEncoder.get_feature_names_out()`](#get_feature_names_out) | Output column names as '<feature>_<category>' pairs. |\n| [`OneHotEncoder.fit_transform()`](#fit_transform) | \u2014 |\n| [`OneHotEncoder.inverse_transform()`](#inverse_transform) | Recovers the original categorical values from a one-hot matrix. |\n\n---\n\n## Method Details\n\n### `OneHotEncoder.n_features_out()`\n\n```mojo\ndef n_features_out(self) -> Int\n```\n\nNumber of indicator columns produced by transform.\n\n**Returns**: `Int`\n\n---\n\n### `OneHotEncoder.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `OneHotEncoder.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `OneHotEncoder.get_feature_names_out()`\n\n```mojo\ndef get_feature_names_out(self, input_features: List[String] = List[String]()) -> List[String]\n```\n\nOutput column names as '<feature>_<category>' pairs.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`input_features`** | `List[String]` | \u2014 |\n\n**Returns**: `List[String]`\n\n---\n\n### `OneHotEncoder.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `OneHotEncoder.inverse_transform()`\n\n```mojo\ndef inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\nRecovers the original categorical values from a one-hot matrix.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import OneHotEncoder\nfrom strata.core import Matrix\n\nvar encoder = OneHotEncoder[DType.float64](drop=\"if_binary\")\nencoder.fit(X_cat)\nvar X_encoded = encoder.transform(X_cat)\n```\n",
    "reference/preprocessing/OrdinalEncoder": "# `OrdinalEncoder`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/encoders.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/encoders.mojo)\n\n```mojo\nstruct OrdinalEncoder[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import OrdinalEncoder\n```\n\n**Encode categorical features as an integer array.**\n\nThe input to this transformer should be a 2D matrix of integer or float\ncategorical features. The features are converted to ordinal integers\n($0, 1, \\dots, K_c - 1$) corresponding to the sorted order of unique\ncategories per feature.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`handle_unknown`** | When set to 'error' an error is raised on unknown categories. When set to 'use_encoded_value', unknown categories are set to unknown_value. Default 'error'. |\n| **`unknown_value`** | Value used when handle_unknown='use_encoded_value'. Default -1.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`categories_`** | Categories of each feature determined during fitting. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`OrdinalEncoder.fit()`](#fit) | Fit the OrdinalEncoder on feature matrix X. |\n| [`OrdinalEncoder.transform()`](#transform) | Transform X to ordinal integer codes. |\n| [`OrdinalEncoder.fit_transform()`](#fit_transform) | Fit to data, then transform it. |\n| [`OrdinalEncoder.inverse_transform()`](#inverse_transform) | Convert the ordinal codes back to original category values. |\n\n---\n\n## Method Details\n\n### `OrdinalEncoder.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the OrdinalEncoder on feature matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `OrdinalEncoder.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nTransform X to ordinal integer codes.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Encoded ordinal matrix.\n\n---\n\n### `OrdinalEncoder.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nFit to data, then transform it.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Encoded ordinal matrix.\n\n---\n\n### `OrdinalEncoder.inverse_transform()`\n\n```mojo\ndef inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\nConvert the ordinal codes back to original category values.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Decoded matrix containing reconstructed categories.\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import OrdinalEncoder\nfrom strata.core import Matrix\n\nvar encoder = OrdinalEncoder[DType.float64]()\nencoder.fit(X_cat)\nvar X_ord = encoder.transform(X_cat)\n```\n",
    "reference/preprocessing/LabelEncoder": "# `LabelEncoder`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/preprocessing/encoders.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/encoders.mojo)\n\n```mojo\nstruct LabelEncoder[compute_dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.preprocessing import LabelEncoder\n```\n\n**Encode target labels with value between 0 and n_classes-1.**\n\nUsed to transform non-numerical or non-consecutive 1D target labels\ninto continuous integer labels for classification tasks.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Precision used for internal class representation. Default DType.float64. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`classes_`** | Distinct classes seen during fit, in sorted order. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`LabelEncoder.fit()`](#fit) | Fit label encoder on target vector y. |\n| [`LabelEncoder.transform()`](#transform) | Transform target labels to normalized encoding indices. |\n| [`LabelEncoder.fit_transform()`](#fit_transform) | Fit label encoder and return encoded integer labels. |\n| [`LabelEncoder.inverse_transform()`](#inverse_transform) | Transform integer labels back to original encoding. |\n\n---\n\n## Method Details\n\n### `LabelEncoder.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, y: List[Scalar[in_dtype]])\n```\n\nFit label encoder on target vector y.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`y`** | `List[Scalar[in_dtype]]` | Target vector / class labels. |\n\n---\n\n### `LabelEncoder.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, y: List[Scalar[in_dtype]]) -> List[Int]\n```\n\nTransform target labels to normalized encoding indices.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`y`** | `List[Scalar[in_dtype]]` | Target vector / class labels. |\n\n**Returns**: `List[Int]` \u2014 List[Int]: Encoded integer labels in range [0, n_classes - 1].\n\n---\n\n### `LabelEncoder.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, y: List[Scalar[in_dtype]]) -> List[Int]\n```\n\nFit label encoder and return encoded integer labels.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`y`** | `List[Scalar[in_dtype]]` | Target vector / class labels. |\n\n**Returns**: `List[Int]` \u2014 List[Int]: Encoded integer labels.\n\n---\n\n### `LabelEncoder.inverse_transform()`\n\n```mojo\ndef inverse_transform[out_dtype: DType = Self.compute_dtype](self, y: List[Int]) -> List[Scalar[out_dtype]]\n```\n\nTransform integer labels back to original encoding.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`out_dtype`** | \u2014 | Output precision data type. Default compute_dtype. |\n\n**Returns**: `List[Scalar[out_dtype]]` \u2014 List[Scalar[out_dtype]]: Reconstructed original labels.\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import LabelEncoder\n\nvar encoder = LabelEncoder[DType.float64]()\nencoder.fit(y_train)\nvar y_encoded = encoder.transform(y_train)\nvar y_original = encoder.inverse_transform(y_encoded)\n```\n",
    "reference/preprocessing/SimpleImputer": "# `SimpleImputer`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/imputer.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/imputer.mojo)\n\n```mojo\nstruct SimpleImputer[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import SimpleImputer\n```\n\n**Univariate imputer for completing missing values with simple statistics.**\n\nReplaces missing values (`NaN` or a specified sentinel value) using a\nchosen statistical strategy along each column.\nStrategies:\n- `\"mean\"`: Replace missing values using the mean along each column.\n- `\"median\"`: Replace missing values using the median along each column.\n- `\"most_frequent\"`: Replace missing values using the most frequent value (mode) along each column.\n- `\"constant\"`: Replace missing values with `fill_value`.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`missing_values`** | The placeholder for missing values. Can be `NaN` or a scalar value. Default `NaN`. |\n| **`strategy`** | The imputation strategy ('mean', 'median', 'most_frequent', 'constant'). Default 'mean'. |\n| **`fill_value`** | Value used when strategy='constant'. Default 0.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`statistics_`** | The imputation fill value for each feature column calculated during fit. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`SimpleImputer.fit()`](#fit) | Fit the imputer on feature matrix X. |\n| [`SimpleImputer.transform()`](#transform) | Impute all missing values in X. |\n| [`SimpleImputer.fit_transform()`](#fit_transform) | Fit to data, then transform it. |\n\n---\n\n## Method Details\n\n### `SimpleImputer.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the imputer on feature matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `SimpleImputer.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nImpute all missing values in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Matrix with missing values imputed.\n\n---\n\n### `SimpleImputer.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nFit to data, then transform it.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Matrix with missing values imputed.\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import SimpleImputer\nfrom strata.core import Matrix\n\nvar imputer = SimpleImputer[DType.float64](strategy=\"mean\")\nimputer.fit(X_train)\nvar X_imputed = imputer.transform(X_train)\n```\n",
    "reference/preprocessing/PolynomialFeatures": "# `PolynomialFeatures`\n\n**Module**: [`strata.preprocessing`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/preprocessing/polynomial.mojo`](file:////home/ewu/Code/Strata/strata/preprocessing/polynomial.mojo)\n\n```mojo\nstruct PolynomialFeatures[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.preprocessing import PolynomialFeatures\n```\n\n**Generate polynomial and interaction features.**\n\nGenerates a new feature matrix consisting of all polynomial combinations\nof the features with degree less than or equal to the specified degree.\nFor example, if an input sample is 2D and of the form $[a, b]$, the\ndegree-2 polynomial features with bias are $[1, a, b, a^2, ab, b^2]$.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`degree`** | The maximal degree of polynomial features. Default 2. |\n| **`interaction_only`** | If True, only interaction features are produced: products of at most `degree` distinct input features. Default False. |\n| **`include_bias`** | If True, includes a bias column (all 1s) acting as an intercept term. Default True. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`powers_`** | Exponent matrix with shape (n_output_features_, n_features_in_). |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`n_output_features_`** | Total number of polynomial output features. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`PolynomialFeatures.fit()`](#fit) | Compute the number of output features and combination powers. |\n| [`PolynomialFeatures.transform()`](#transform) | Transform data matrix X to polynomial feature combinations. |\n| [`PolynomialFeatures.fit_transform()`](#fit_transform) | Fit to data, then transform it. |\n\n---\n\n## Method Details\n\n### `PolynomialFeatures.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nCompute the number of output features and combination powers.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `PolynomialFeatures.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nTransform data matrix X to polynomial feature combinations.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Expanded polynomial matrix with shape (n_samples, n_output_features_).\n\n---\n\n### `PolynomialFeatures.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nFit to data, then transform it.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |\n\n**Returns**: `Matrix[in_dtype]` \u2014 Matrix[in_dtype]: Expanded polynomial matrix.\n---\n\n## Example\n\n```mojo\nfrom strata.preprocessing import PolynomialFeatures\nfrom strata.core import Matrix\n\nvar poly = PolynomialFeatures[DType.float64](degree=2)\npoly.fit(X)\nvar X_poly = poly.transform(X)\n```\n",
    "reference/linear_model/index": "# `strata.linear_model`\n\nLinearRegression via OLS SVD, Ridge Regression via L2 regularization, and LogisticRegression with L-BFGS and SGD optimizers.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`LinearRegression`](LinearRegression.md) | Ordinary Least Squares Linear Regression. |\n| [`Ridge`](Ridge.md) | Ridge regression with L2 regularization. |\n| [`LogisticRegression`](LogisticRegression.md) | Logistic Regression classifier with L2 regularization. |\n",
    "reference/linear_model/LinearRegression": "# `LinearRegression`\n\n**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/linear_model/linear_regression.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/linear_regression.mojo)\n\n```mojo\nstruct LinearRegression[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.linear_model import LinearRegression\n```\n\n**Ordinary Least Squares Linear Regression.**\n\nFits a linear model with coefficients $w = (w_1, \\dots, w_D)$ and intercept $b$\nto minimize the residual sum of squares between observed targets and predictions:\n$$\n\\min_{w, b} \\frac{1}{2N} \\|y - (Xw + b)\\|_2^2\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`fit_intercept`** | Whether to calculate the intercept bias term. Default True. |\n| **`solver`** | Solver algorithm to use ('lstsq', 'qr', 'cholesky', 'solve'). Default 'lstsq'. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`coef_`** | Weight vector coefficients of length $D$. |\n| **`intercept_`** | Independent bias intercept term. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`LinearRegression.fit()`](#fit) | Fit the linear model from training data. |\n| [`LinearRegression.predict()`](#predict) | Predict continuous target values using the fitted linear model. |\n\n---\n\n## Method Details\n\n### `LinearRegression.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the linear model from training data.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `LinearRegression.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nPredict continuous target values using the fitted linear model.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[feat_dtype]]` \u2014 List[Scalar[feat_dtype]]: Predicted target vector of length $N$.\n---\n\n## Example\n\n```mojo\nfrom strata.linear_model import LinearRegression\nfrom strata.core import Matrix\n\nvar reg = LinearRegression[DType.float64](solver=\"cholesky\")\nreg.fit(X_train, y_train)\nvar preds = reg.predict(X_test)\n```\n",
    "reference/linear_model/Ridge": "# `Ridge`\n\n**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/linear_model/ridge.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/ridge.mojo)\n\n```mojo\nstruct Ridge[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.linear_model import Ridge\n```\n\n**Ridge regression with L2 regularization.**\n\nMinimizes the penalized objective function:\n$$\n\\min_{w} \\|y - Xw\\|_2^2 + \\alpha \\|w\\|_2^2\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`alpha`** | Regularization strength ($\\alpha \\ge 0$). Larger values enforce stronger shrinkage. Default 1.0. |\n| **`fit_intercept`** | Whether to calculate the intercept bias term. Default True. |\n| **`solver`** | Solver algorithm to use ('auto', 'cholesky', 'svd', 'solve'). Default 'auto'. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`coef_`** | Weight vector coefficients of length $D$. |\n| **`intercept_`** | Independent bias intercept term. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Ridge.fit()`](#fit) | Fit the Ridge regression model from training data. |\n| [`Ridge.predict()`](#predict) | Predict continuous target values using the fitted linear model. |\n\n---\n\n## Method Details\n\n### `Ridge.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the Ridge regression model from training data.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `Ridge.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nPredict continuous target values using the fitted linear model.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[feat_dtype]]` \u2014 List[Scalar[feat_dtype]]: Predicted target vector of length $N$.\n---\n\n## Example\n\n```mojo\nfrom strata.linear_model import Ridge\nfrom strata.core import Matrix\n\nvar model = Ridge[DType.float64](alpha=0.5)\nmodel.fit(X_train, y_train)\nvar preds = model.predict(X_test)\n```\n",
    "reference/linear_model/LogisticRegression": "# `LogisticRegression`\n\n**Module**: [`strata.linear_model`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/linear_model/logistic_regression.mojo`](file:////home/ewu/Code/Strata/strata/linear_model/logistic_regression.mojo)\n\n```mojo\nstruct LogisticRegression[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.linear_model import LogisticRegression\n```\n\n**Logistic Regression classifier with L2 regularization.**\n\nSupports binary and multiclass (multinomial) classification by minimizing\nthe regularized cross-entropy loss with gradient optimization:\n$$\n\\min_{W, b} -\\frac{1}{N} \\sum_{i=1}^{N} \\ln P(y_i \\mid x_i; W, b) + \\frac{1}{2C} \\|W\\|_F^2\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`penalty`** | Regularization norm ('l2' or 'none'). Default 'l2'. |\n| **`C`** | Inverse regularization strength ($C > 0$). Smaller values specify stronger regularization. Default 1.0. |\n| **`fit_intercept`** | Whether to calculate the intercept bias vector. Default True. |\n| **`max_iter`** | Maximum number of gradient optimization iterations. Default 100. |\n| **`tol`** | Tolerance threshold for stopping criterion based on gradient norm. Default 1e-4. |\n| **`learning_rate`** | Step size for gradient descent optimization updates. Default 0.1. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`classes_`** | Sorted list of unique class labels seen during fit. |\n| **`coef_`** | Learned weight coefficient matrix of shape $(K, D)$. |\n| **`intercept_`** | Learned bias intercept vector of length $K$. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`LogisticRegression.fit()`](#fit) | Fits the logistic regression model on training data (X, y). |\n| [`LogisticRegression.predict_proba()`](#predict_proba) | Predict class probability distributions for samples in X. |\n| [`LogisticRegression.predict()`](#predict) | Predict discrete class labels for samples in X. |\n\n---\n\n## Method Details\n\n### `LogisticRegression.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the logistic regression model on training data (X, y).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |\n\n---\n\n### `LogisticRegression.predict_proba()`\n\n```mojo\ndef predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]\n```\n\nPredict class probability distributions for samples in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `Matrix[feat_dtype]` \u2014 Matrix[feat_dtype]: Probability matrix of shape $(N, K)$, where row $i$ contains the normalized probability distribution over $K$ classes.\n\n---\n\n### `LogisticRegression.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredict discrete class labels for samples in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]` \u2014 List[Int]: Predicted class labels vector of length $N$.\n---\n\n## Example\n\n```mojo\nfrom strata.linear_model import LogisticRegression\nfrom strata.core import Matrix\n\nvar clf = LogisticRegression[DType.float64](C=1.0, max_iter=200)\nclf.fit(X_train, y_train)\nvar probs = clf.predict_proba(X_test)\nvar preds = clf.predict(X_test)\n```\n",
    "reference/tree/index": "# `strata.tree`\n\nFast recursive classification (Gini, Entropy, Log-Loss) and regression (MSE, Friedman MSE, MAE) trees with streaming histogram split calculations.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`DecisionTreeClassifier`](DecisionTreeClassifier.md) | Decision Tree Classifier for non-parametric supervised classification. |\n| [`DecisionTreeRegressor`](DecisionTreeRegressor.md) | Decision Tree Regressor for non-parametric continuous target regression. |\n",
    "reference/tree/DecisionTreeClassifier": "# `DecisionTreeClassifier`\n\n**Module**: [`strata.tree`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/tree/classifier.mojo`](file:////home/ewu/Code/Strata/strata/tree/classifier.mojo)\n\n```mojo\nstruct DecisionTreeClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.tree import DecisionTreeClassifier\n```\n\n**Decision Tree Classifier for non-parametric supervised classification.**\n\nSplits internal nodes to maximize impurity reduction based on Gini impurity\nor Shannon entropy:\n$$\nH_{\\text{gini}}(Q) = 1 - \\sum_{k=1}^{K} p_k^2, \\quad H_{\\text{entropy}}(Q) = -\\sum_{k=1}^{K} p_k \\log_2(p_k)\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`criterion`** | The function to measure the quality of a split ('gini', 'entropy', 'log_loss'). Default 'gini'. |\n| **`splitter`** | Strategy used to choose the split at each node ('best', 'random'). Default 'best'. |\n| **`max_depth`** | Maximum tree depth. -1 indicates unlimited depth. Default -1. |\n| **`min_samples_split`** | Minimum samples required to split an internal node. Default 2. |\n| **`min_samples_leaf`** | Minimum samples required to be at a leaf node. Default 1. |\n| **`min_impurity_decrease`** | Split threshold if impurity decrease >= this value. Default 0.0. |\n| **`max_features`** | Number of features to consider when looking for best split ('all', 'sqrt', 'log2', 'custom'). Default 'all'. |\n| **`max_features_count`** | Explicit number of features to evaluate when max_features='custom'. Default -1. |\n| **`max_features_ratio`** | Proportion of features to evaluate when max_features='custom'. Default 0.0. |\n| **`random_state`** | PRNG seed for deterministic feature and split selection. Default 42. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`classes_`** | Sorted list of unique class labels observed during fit. |\n| **`n_classes_`** | Number of unique classes observed. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`DecisionTreeClassifier.fit()`](#fit) | Fits the decision tree classifier on (X, y). |\n| [`DecisionTreeClassifier.predict()`](#predict) | Generates discrete class predictions for input matrix X. |\n| [`DecisionTreeClassifier.predict_proba()`](#predict_proba) | Generates class probability estimates for input matrix X. |\n| [`DecisionTreeClassifier.get_depth()`](#get_depth) | Returns the maximum depth of the fitted tree. |\n| [`DecisionTreeClassifier.get_n_leaves()`](#get_n_leaves) | Returns the total number of leaf nodes in the fitted tree. |\n\n---\n\n## Method Details\n\n### `DecisionTreeClassifier.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the decision tree classifier on (X, y).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |\n\n---\n\n### `DecisionTreeClassifier.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nGenerates discrete class predictions for input matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `DecisionTreeClassifier.predict_proba()`\n\n```mojo\ndef predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]\n```\n\nGenerates class probability estimates for input matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `Matrix[feat_dtype]`\n\n---\n\n### `DecisionTreeClassifier.get_depth()`\n\n```mojo\ndef get_depth(self) -> Int\n```\n\nReturns the maximum depth of the fitted tree.\n\n**Returns**: `Int`\n\n---\n\n### `DecisionTreeClassifier.get_n_leaves()`\n\n```mojo\ndef get_n_leaves(self) -> Int\n```\n\nReturns the total number of leaf nodes in the fitted tree.\n\n**Returns**: `Int`\n---\n\n## Example\n\n```mojo\nfrom strata.tree import DecisionTreeClassifier\nfrom strata.core import Matrix\n\nvar tree = DecisionTreeClassifier[DType.float64](max_depth=5, criterion=\"gini\")\ntree.fit(X_train, y_train)\nvar preds = tree.predict(X_test)\n```\n",
    "reference/tree/DecisionTreeRegressor": "# `DecisionTreeRegressor`\n\n**Module**: [`strata.tree`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/tree/regressor.mojo`](file:////home/ewu/Code/Strata/strata/tree/regressor.mojo)\n\n```mojo\nstruct DecisionTreeRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.tree import DecisionTreeRegressor\n```\n\n**Decision Tree Regressor for non-parametric continuous target regression.**\n\nBuilds a regression tree by minimizing sample variance (mean squared error)\nor mean absolute deviation across recursive binary splits:\n$$\nH_{\\text{MSE}}(Q) = \\frac{1}{|Q|} \\sum_{i \\in Q} (y_i - \\bar{y}_Q)^2\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`criterion`** | The function to measure split quality ('squared_error', 'friedman_mse', 'absolute_error'). Default 'squared_error'. |\n| **`splitter`** | Strategy used to choose the split at each node ('best', 'random'). Default 'best'. |\n| **`max_depth`** | Maximum tree depth. -1 indicates unlimited depth. Default -1. |\n| **`min_samples_split`** | Minimum samples required to split an internal node. Default 2. |\n| **`min_samples_leaf`** | Minimum samples required to be at a leaf node. Default 1. |\n| **`min_impurity_decrease`** | Split threshold if impurity decrease >= this value. Default 0.0. |\n| **`max_features`** | Number of features to consider when looking for best split ('all', 'sqrt', 'log2', 'custom'). Default 'all'. |\n| **`max_features_count`** | Explicit number of features to evaluate when max_features='custom'. Default -1. |\n| **`max_features_ratio`** | Proportion of features to evaluate when max_features='custom'. Default 0.0. |\n| **`random_state`** | PRNG seed for deterministic feature and split selection. Default 42. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`DecisionTreeRegressor.fit()`](#fit) | Fits the decision tree regressor on (X, y). |\n| [`DecisionTreeRegressor.predict()`](#predict) | Generates regression predictions for input matrix X. |\n| [`DecisionTreeRegressor.get_depth()`](#get_depth) | Returns the maximum depth of the fitted tree. |\n| [`DecisionTreeRegressor.get_n_leaves()`](#get_n_leaves) | Returns the total number of leaf nodes in the fitted tree. |\n\n---\n\n## Method Details\n\n### `DecisionTreeRegressor.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the decision tree regressor on (X, y).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `DecisionTreeRegressor.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nGenerates regression predictions for input matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[feat_dtype]]`\n\n---\n\n### `DecisionTreeRegressor.get_depth()`\n\n```mojo\ndef get_depth(self) -> Int\n```\n\nReturns the maximum depth of the fitted tree.\n\n**Returns**: `Int`\n\n---\n\n### `DecisionTreeRegressor.get_n_leaves()`\n\n```mojo\ndef get_n_leaves(self) -> Int\n```\n\nReturns the total number of leaf nodes in the fitted tree.\n\n**Returns**: `Int`\n---\n\n## Example\n\n```mojo\nfrom strata.tree import DecisionTreeRegressor\nfrom strata.core import Matrix\n\nvar reg = DecisionTreeRegressor[DType.float64](max_depth=4)\nreg.fit(X_train, y_train)\nvar preds = reg.predict(X_test)\n```\n",
    "reference/ensemble/index": "# `strata.ensemble`\n\nRandom Forest Regressors and Classifiers with bootstrap aggregation, decoupled PRNG streams, batched OOB scoring, and MDI feature importances.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`RandomForestRegressor`](RandomForestRegressor.md) | Random Forest Regressor ensemble estimator. |\n| [`RandomForestClassifier`](RandomForestClassifier.md) | Random Forest Classifier ensemble estimator. |\n",
    "reference/ensemble/RandomForestRegressor": "# `RandomForestRegressor`\n\n**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/ensemble/forest.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/forest.mojo)\n\n```mojo\nstruct RandomForestRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.ensemble import RandomForestRegressor\n```\n\n**Random Forest Regressor ensemble estimator.**\n\nAn ensemble of decision trees trained via bootstrap aggregation (bagging).\nPredictions are computed as the arithmetic mean of individual tree predictions.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_estimators`** | Number of trees in the forest. Default 100. |\n| **`criterion`** | Impurity split criterion ('squared_error', 'friedman_mse', 'absolute_error'). Default 'squared_error'. |\n| **`max_depth`** | Maximum tree depth. -1 means unlimited. Default -1. |\n| **`min_samples_split`** | Minimum samples required to split an internal node. Default 2. |\n| **`min_samples_leaf`** | Minimum samples required to be a leaf node. Default 1. |\n| **`min_impurity_decrease`** | Split threshold if impurity decrease >= this value. Default 0.0. |\n| **`max_features`** | Number of features to consider per split ('all', 'sqrt', 'log2'). Default 'sqrt'. |\n| **`max_features_count`** | Exact number of features per split. Default -1 (disabled). |\n| **`max_features_ratio`** | Proportion of features per split. Default 0.0 (disabled). |\n| **`bootstrap`** | Whether to use bootstrap sampling. Default True. |\n| **`max_samples_ratio`** | Proportion of samples drawn per tree when bootstrap=True. Default 1.0. |\n| **`max_samples_count`** | Exact number of samples drawn per tree. Default -1 (disabled). |\n| **`oob_score`** | Whether to compute out-of-bag $R^2$ score after fitting. Default False. |\n| **`random_state`** | PRNG seed for deterministic tree builds. Default 42. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`feature_importances_`** | Normalized impurity feature importance vector. |\n| **`oob_score_`** | Out-of-bag $R^2$ score (available when oob_score=True). |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`RandomForestRegressor.fit()`](#fit) | Fits the random forest on (X, y). |\n| [`RandomForestRegressor.predict()`](#predict) | Predicts regression targets as the arithmetic mean across all tree predictions. |\n| [`RandomForestRegressor.get_n_estimators()`](#get_n_estimators) | Returns the number of fitted trees. |\n| [`RandomForestRegressor.get_feature_importances()`](#get_feature_importances) | Returns normalized MDI feature importances (sums to 1.0). |\n| [`RandomForestRegressor.get_oob_score()`](#get_oob_score) | Returns out-of-bag R\u00b2 score. Requires oob_score=True and bootstrap=True. |\n\n---\n\n## Method Details\n\n### `RandomForestRegressor.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the random forest on (X, y).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `RandomForestRegressor.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nPredicts regression targets as the arithmetic mean across all tree predictions.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[feat_dtype]]`\n\n---\n\n### `RandomForestRegressor.get_n_estimators()`\n\n```mojo\ndef get_n_estimators(self) -> Int\n```\n\nReturns the number of fitted trees.\n\n**Returns**: `Int`\n\n---\n\n### `RandomForestRegressor.get_feature_importances()`\n\n```mojo\ndef get_feature_importances(self) -> List[Float64]\n```\n\nReturns normalized MDI feature importances (sums to 1.0).\n\n**Returns**: `List[Float64]`\n\n---\n\n### `RandomForestRegressor.get_oob_score()`\n\n```mojo\ndef get_oob_score(self) -> Float64\n```\n\nReturns out-of-bag R\u00b2 score. Requires oob_score=True and bootstrap=True.\n\n**Returns**: `Float64`\n---\n\n## Example\n\n```mojo\nfrom strata.ensemble import RandomForestRegressor\nfrom strata.core import Matrix\n\nvar rf = RandomForestRegressor[DType.float64](n_estimators=50, max_depth=6)\nrf.fit(X_train, y_train)\nvar preds = rf.predict(X_test)\n```\n",
    "reference/ensemble/RandomForestClassifier": "# `RandomForestClassifier`\n\n**Module**: [`strata.ensemble`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/ensemble/forest.mojo`](file:////home/ewu/Code/Strata/strata/ensemble/forest.mojo)\n\n```mojo\nstruct RandomForestClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.ensemble import RandomForestClassifier\n```\n\n**Random Forest Classifier ensemble estimator.**\n\nAn ensemble of decision trees trained via bootstrap aggregation (bagging).\nPredictions are computed via soft voting (averaging predicted class probabilities\nacross all trees and selecting the argmax class).\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_estimators`** | Number of trees in the forest. Default 100. |\n| **`criterion`** | Impurity split criterion ('gini', 'entropy', 'log_loss'). Default 'gini'. |\n| **`max_depth`** | Maximum tree depth. -1 means unlimited. Default -1. |\n| **`min_samples_split`** | Minimum samples required to split an internal node. Default 2. |\n| **`min_samples_leaf`** | Minimum samples required to be a leaf node. Default 1. |\n| **`min_impurity_decrease`** | Split threshold if impurity decrease >= this value. Default 0.0. |\n| **`max_features`** | Number of features to consider per split ('all', 'sqrt', 'log2'). Default 'sqrt'. |\n| **`max_features_count`** | Exact number of features per split. Default -1 (disabled). |\n| **`max_features_ratio`** | Proportion of features per split. Default 0.0 (disabled). |\n| **`bootstrap`** | Whether to use bootstrap sampling. Default True. |\n| **`max_samples_ratio`** | Proportion of samples drawn per tree when bootstrap=True. Default 1.0. |\n| **`max_samples_count`** | Exact number of samples drawn per tree. Default -1 (disabled). |\n| **`oob_score`** | Whether to compute out-of-bag accuracy score after fitting. Default False. |\n| **`random_state`** | PRNG seed for deterministic tree builds. Default 42. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`classes_`** | Sorted list of unique class labels seen during fit. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`feature_importances_`** | Normalized impurity feature importance vector. |\n| **`oob_score_`** | Out-of-bag accuracy score (available when oob_score=True). |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`RandomForestClassifier.fit()`](#fit) | Fits the random forest classifier on (X, y). |\n| [`RandomForestClassifier.predict_proba()`](#predict_proba) | Generates class probability estimates for input matrix X by averaging tree probabilities. |\n| [`RandomForestClassifier.predict()`](#predict) | Generates discrete class predictions via soft-voting argmax over predicted class probabilities. |\n| [`RandomForestClassifier.get_n_estimators()`](#get_n_estimators) | Returns the number of fitted trees. |\n| [`RandomForestClassifier.get_feature_importances()`](#get_feature_importances) | Returns normalized MDI feature importances (sums to 1.0). |\n| [`RandomForestClassifier.get_oob_score()`](#get_oob_score) | Returns out-of-bag accuracy score. Requires oob_score=True and bootstrap=True. |\n| [`RandomForestClassifier.get_classes()`](#get_classes) | Returns the sorted list of known class labels. |\n\n---\n\n## Method Details\n\n### `RandomForestClassifier.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the random forest classifier on (X, y).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |\n\n---\n\n### `RandomForestClassifier.predict_proba()`\n\n```mojo\ndef predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]\n```\n\nGenerates class probability estimates for input matrix X by averaging tree probabilities.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `Matrix[feat_dtype]`\n\n---\n\n### `RandomForestClassifier.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nGenerates discrete class predictions via soft-voting argmax over predicted class probabilities.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `RandomForestClassifier.get_n_estimators()`\n\n```mojo\ndef get_n_estimators(self) -> Int\n```\n\nReturns the number of fitted trees.\n\n**Returns**: `Int`\n\n---\n\n### `RandomForestClassifier.get_feature_importances()`\n\n```mojo\ndef get_feature_importances(self) -> List[Float64]\n```\n\nReturns normalized MDI feature importances (sums to 1.0).\n\n**Returns**: `List[Float64]`\n\n---\n\n### `RandomForestClassifier.get_oob_score()`\n\n```mojo\ndef get_oob_score(self) -> Float64\n```\n\nReturns out-of-bag accuracy score. Requires oob_score=True and bootstrap=True.\n\n**Returns**: `Float64`\n\n---\n\n### `RandomForestClassifier.get_classes()`\n\n```mojo\ndef get_classes(self) -> List[Int]\n```\n\nReturns the sorted list of known class labels.\n\n**Returns**: `List[Int]`\n---\n\n## Example\n\n```mojo\nfrom strata.ensemble import RandomForestClassifier\nfrom strata.core import Matrix\n\nvar rf = RandomForestClassifier[DType.float64](n_estimators=50, max_depth=6)\nrf.fit(X_train, y_train)\nvar preds = rf.predict(X_test)\n```\n",
    "reference/cluster/index": "# `strata.cluster`\n\nSIMD Lloyd's algorithm K-Means with k-means++ initialization and streaming MiniBatchKMeans.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`KMeans`](KMeans.md) | K-Means clustering using Lloyd's or Elkan's algorithm. |\n| [`MiniBatchKMeans`](MiniBatchKMeans.md) | Mini-Batch K-Means clustering algorithm. |\n",
    "reference/cluster/KMeans": "# `KMeans`\n\n**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`  \n**Source**: [`strata/cluster/kmeans.mojo`](file:////home/ewu/Code/Strata/strata/cluster/kmeans.mojo)\n\n```mojo\nstruct KMeans[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)\n```\n\n```mojo\nfrom strata.cluster import KMeans\n```\n\n**K-Means clustering using Lloyd's or Elkan's algorithm.**\n\nClusters $N$ observations into $K$ disjoint geometric partitions by\nminimizing within-cluster inertia (sum-of-squared Euclidean distances):\n$$\n\\arg\\min_{C} \\sum_{i=1}^{N} \\min_{\\mu_j \\in C} \\|x_i - \\mu_j\\|_2^2\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_clusters`** | The number of clusters to form as well as the number of centroids to generate. Default 8. |\n| **`init`** | Method for initialization ('k-means++', 'random'). Default 'k-means++'. |\n| **`n_init`** | Number of times the k-means algorithm will be run with different centroid seeds. Default 10. |\n| **`max_iter`** | Maximum number of iterations of the k-means algorithm for a single run. Default 300. |\n| **`tol`** | Relative tolerance with regards to Frobenius norm of the difference in cluster centers. Default 1e-4. |\n| **`algorithm`** | K-means algorithm to use ('lloyd'). Default 'lloyd'. |\n| **`random_state`** | PRNG seed for centroid initialization. Default 42. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`cluster_centers_`** | Coordinates of cluster centers matrix of shape $(K, D)$. |\n| **`labels_`** | Labels of each point vector of length $N$. |\n| **`inertia_`** | Sum of squared distances of samples to their closest cluster center. |\n| **`n_iter_`** | Number of iterations run. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`KMeans.fit()`](#fit) | \u2014 |\n| [`KMeans.predict()`](#predict) | \u2014 |\n| [`KMeans.fit_predict()`](#fit_predict) | \u2014 |\n| [`KMeans.transform()`](#transform) | \u2014 |\n| [`KMeans.fit_transform()`](#fit_transform) | \u2014 |\n| [`KMeans.score()`](#score) | \u2014 |\n\n---\n\n## Method Details\n\n### `KMeans.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits cluster centroids on Dataset feature records.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `KMeans.predict()`\n\n```mojo\ndef predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredicts closest cluster assignments for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `KMeans.fit_predict()`\n\n```mojo\ndef fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `KMeans.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `KMeans.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `KMeans.score()`\n\n```mojo\ndef score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Scalar[Self.compute_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.cluster import KMeans\nfrom strata.core import Matrix\n\nvar kmeans = KMeans[DType.float64](n_clusters=3, init=\"k-means++\")\nkmeans.fit(X_data)\nvar labels = kmeans.predict(X_data)\nvar distances = kmeans.transform(X_data)\n```\n",
    "reference/cluster/MiniBatchKMeans": "# `MiniBatchKMeans`\n\n**Module**: [`strata.cluster`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Clusterer, Copyable, Movable`  \n**Source**: [`strata/cluster/minibatch_kmeans.mojo`](file:////home/ewu/Code/Strata/strata/cluster/minibatch_kmeans.mojo)\n\n```mojo\nstruct MiniBatchKMeans[compute_dtype: DType = DType.float64](Clusterer, Copyable, Movable)\n```\n\n```mojo\nfrom strata.cluster import MiniBatchKMeans\n```\n\n**Mini-Batch K-Means clustering algorithm.**\n\nMini-Batch K-Means uses mini-batches of samples to reduce computation time while\noptimizing the same objective function as full-batch K-Means:\n$$\n\\arg\\min_{C} \\sum_{i=1}^{N} \\min_{\\mu_j \\in C} \\|x_i - \\mu_j\\|_2^2\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_clusters`** | The number of clusters to form as well as the number of centroids to generate. Default 8. |\n| **`init`** | Method for initialization ('k-means++', 'random'). Default 'k-means++'. |\n| **`max_iter`** | Maximum number of mini-batch iterations. Default 100. |\n| **`batch_size`** | Size of mini-batches drawn per iteration. Default 1024. |\n| **`tol`** | Tolerance threshold for early stopping based on center shift. Default 1e-4. |\n| **`max_no_improvement`** | Early stopping iteration count without inertia improvement. Default 10. |\n| **`n_init`** | Number of random initialization attempts. Default 3. |\n| **`reassignment_ratio`** | Fraction of max count threshold for center reassignment. Default 0.01. |\n| **`random_state`** | PRNG seed for reproducible centroid initializations. Default 42. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`cluster_centers_`** | Coordinates of cluster centers matrix of shape $(K, D)$. |\n| **`labels_`** | Labels of each point vector of length $N$. |\n| **`inertia_`** | Sum of squared distances of samples to their closest cluster center. |\n| **`n_iter_`** | Number of iterations run during fitting. |\n| **`n_steps_`** | Total mini-batch update steps performed. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`MiniBatchKMeans.partial_fit()`](#partial_fit) | \u2014 |\n| [`MiniBatchKMeans.fit()`](#fit) | \u2014 |\n| [`MiniBatchKMeans.predict()`](#predict) | \u2014 |\n| [`MiniBatchKMeans.fit_predict()`](#fit_predict) | \u2014 |\n| [`MiniBatchKMeans.transform()`](#transform) | \u2014 |\n| [`MiniBatchKMeans.fit_transform()`](#fit_transform) | \u2014 |\n| [`MiniBatchKMeans.score()`](#score) | \u2014 |\n\n---\n\n## Method Details\n\n### `MiniBatchKMeans.partial_fit()`\n\n```mojo\ndef partial_fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n---\n\n### `MiniBatchKMeans.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits cluster centroids on Dataset feature records.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `MiniBatchKMeans.predict()`\n\n```mojo\ndef predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredicts closest cluster assignments for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `MiniBatchKMeans.fit_predict()`\n\n```mojo\ndef fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `MiniBatchKMeans.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `MiniBatchKMeans.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `MiniBatchKMeans.score()`\n\n```mojo\ndef score[in_dtype: DType](self, X: Matrix[in_dtype]) -> Scalar[Self.compute_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Scalar[Self.compute_dtype]`\n\n---\n",
    "reference/decomposition/index": "# `strata.decomposition`\n\nPrincipal Component Analysis (PCA) via exact SVD and TruncatedSVD with dense and sparse SpMM linear projection.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`PCA`](PCA.md) | Principal Component Analysis (PCA). |\n| [`TruncatedSVD`](TruncatedSVD.md) | Dimensionality reduction using truncated SVD. |\n",
    "reference/decomposition/PCA": "# `PCA`\n\n**Module**: [`strata.decomposition`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/decomposition/pca.mojo`](file:////home/ewu/Code/Strata/strata/decomposition/pca.mojo)\n\n```mojo\nstruct PCA[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.decomposition import PCA\n```\n\n**Principal Component Analysis (PCA).**\n\nLinear dimensionality reduction using Singular Value Decomposition of the\ncentered data matrix to project it to a lower dimensional subspace:\n$$\nX_{\\text{projected}} = (X - \\mu) V_k\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision data type. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_components`** | Number of components to keep. If 0, all components are kept. Default 0. |\n| **`whiten`** | When True, components vectors are divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. Default False. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`components_`** | Principal axes in feature space, representing directions of maximum variance matrix of shape $(K, D)$. |\n| **`explained_variance_`** | Variance explained by each selected component vector of length $K$. |\n| **`explained_variance_ratio_`** | Percentage of variance explained by each component. |\n| **`singular_values_`** | Singular values corresponding to each of the selected components. |\n| **`mean_`** | Per-feature empirical mean estimated from the training set. |\n| **`n_components_`** | Estimated number of components. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`PCA.fit()`](#fit) | Fits the PCA model on matrix X. |\n| [`PCA.transform()`](#transform) | Projects matrix X onto the principal components. |\n| [`PCA.fit_transform()`](#fit_transform) | Fits PCA to X and returns the projected data. |\n| [`PCA.inverse_transform()`](#inverse_transform) | Transforms data back to its original space. |\n\n---\n\n## Method Details\n\n### `PCA.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the PCA model on matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `PCA.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\nProjects matrix X onto the principal components.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `PCA.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\nFits PCA to X and returns the projected data.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `PCA.inverse_transform()`\n\n```mojo\ndef inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\nTransforms data back to its original space.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n---\n\n## Example\n\n```mojo\nfrom strata.decomposition import PCA\nfrom strata.core import Matrix\n\nvar pca = PCA[DType.float64](n_components=2)\npca.fit(X_train)\nvar X_proj = pca.transform(X_train)\n```\n",
    "reference/decomposition/TruncatedSVD": "# `TruncatedSVD`\n\n**Module**: [`strata.decomposition`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/decomposition/truncated_svd.mojo`](file:////home/ewu/Code/Strata/strata/decomposition/truncated_svd.mojo)\n\n```mojo\nstruct TruncatedSVD[compute_dtype: DType = DType.float64](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.decomposition import TruncatedSVD\n```\n\n**Dimensionality reduction using truncated SVD.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`TruncatedSVD.fit()`](#fit) | Fits TruncatedSVD on dense matrix X. |\n| [`TruncatedSVD.transform()`](#transform) | Projects dense matrix X onto the truncated components. |\n| [`TruncatedSVD.fit_transform()`](#fit_transform) | Fits TruncatedSVD to X and returns the projected data. |\n| [`TruncatedSVD.inverse_transform()`](#inverse_transform) | Transforms data back to its original space. |\n\n---\n\n## Method Details\n\n### `TruncatedSVD.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[in_dtype: DType](mut self, X: CSRMatrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits TruncatedSVD on dense matrix X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `TruncatedSVD.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[in_dtype: DType](self, X: CSRMatrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\nProjects dense matrix X onto the truncated components.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `TruncatedSVD.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[in_dtype: DType](mut self, X: CSRMatrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\nFits TruncatedSVD to X and returns the projected data.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `TruncatedSVD.inverse_transform()`\n\n```mojo\ndef inverse_transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\nTransforms data back to its original space.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n",
    "reference/neighbors/index": "# `strata.neighbors`\n\nDistance metrics (Euclidean, Manhattan, Chebyshev, Minkowski, Cosine), nearest neighbors search, and k-NN classification and regression.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`NeighborDistIdx`](NeighborDistIdx.md) | Container holding a sample distance and its training dataset row index. |\n| [`NearestNeighbors`](NearestNeighbors.md) | Unsupervised learner for implementing neighbor searches. |\n| [`KNeighborsClassifier`](KNeighborsClassifier.md) | Classifier implementing the k-nearest neighbors vote. |\n| [`KNeighborsRegressor`](KNeighborsRegressor.md) | Regression based on k-nearest neighbors. |\n| [`KDNode`](KDNode.md) | Contiguous node in a flat KD-Tree buffer. |\n| [`_AxisIndexPair`](_AxisIndexPair.md) | \u2014 |\n| [`KDTree`](KDTree.md) | Fast spatial index for nearest neighbor and radius queries in low dimensions. |\n\n## Functions\n\n| Function | Description |\n| :--- | :--- |\n| [`sqeuclidean_distance`](sqeuclidean_distance.md) | Compute the squared Euclidean distance between row X[row_x] and row Y[row_y]. |\n| [`euclidean_distance`](euclidean_distance.md) | Compute the Euclidean ($L_2$) distance between row X[row_x] and row Y[row_y]. |\n| [`manhattan_distance`](manhattan_distance.md) | Compute the Manhattan ($L_1$ / taxicab / cityblock) distance between row X[row_x] and row Y[row_y]. |\n| [`chebyshev_distance`](chebyshev_distance.md) | Compute the Chebyshev ($L_\\infty$ / max) distance between row X[row_x] and row Y[row_y]. |\n| [`minkowski_distance`](minkowski_distance.md) | Compute the Minkowski ($L_p$) distance between row X[row_x] and row Y[row_y]. |\n| [`cosine_distance`](cosine_distance.md) | Compute the Cosine distance between row X[row_x] and row Y[row_y]. |\n| [`row_distance`](row_distance.md) | Compute pairwise distance between sample row X[row_x] and sample row Y[row_y] under metric. |\n| [`pairwise_distances`](pairwise_distances.md) | Compute the full pairwise distance matrix between rows of X and rows of Y. |\n| [`pairwise_distances`](pairwise_distances.md) | Compute the self-pairwise distance matrix between all pairs of rows in X. |\n",
    "reference/neighbors/sqeuclidean_distance": "# `sqeuclidean_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef sqeuclidean_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64\n```\n\n```mojo\nfrom strata.neighbors import sqeuclidean_distance\n```\n\n**Compute the squared Euclidean distance between row X[row_x] and row Y[row_y].**\n\n$$\nd^2(u, v) = \\sum_{j=1}^D (u_j - v_j)^2\n$$\n\n**Returns**: `Float64` \u2014 Float64: Squared Euclidean distance.\n",
    "reference/neighbors/euclidean_distance": "# `euclidean_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef euclidean_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64\n```\n\n```mojo\nfrom strata.neighbors import euclidean_distance\n```\n\n**Compute the Euclidean ($L_2$) distance between row X[row_x] and row Y[row_y].**\n\n$$\nd(u, v) = \\sqrt{\\sum_{j=1}^D (u_j - v_j)^2}\n$$\n\n**Returns**: `Float64` \u2014 Float64: Euclidean distance.\n",
    "reference/neighbors/manhattan_distance": "# `manhattan_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef manhattan_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64\n```\n\n```mojo\nfrom strata.neighbors import manhattan_distance\n```\n\n**Compute the Manhattan ($L_1$ / taxicab / cityblock) distance between row X[row_x] and row Y[row_y].**\n\n$$\nd(u, v) = \\sum_{j=1}^D |u_j - v_j|\n$$\n\n**Returns**: `Float64` \u2014 Float64: Manhattan distance.\n",
    "reference/neighbors/chebyshev_distance": "# `chebyshev_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef chebyshev_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64\n```\n\n```mojo\nfrom strata.neighbors import chebyshev_distance\n```\n\n**Compute the Chebyshev ($L_\\infty$ / max) distance between row X[row_x] and row Y[row_y].**\n\n$$\nd(u, v) = \\max_{1 \\le j \\le D} |u_j - v_j|\n$$\n\n**Returns**: `Float64` \u2014 Float64: Chebyshev distance.\n",
    "reference/neighbors/minkowski_distance": "# `minkowski_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef minkowski_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int, p: Float64 = 2.0) -> Float64\n```\n\n```mojo\nfrom strata.neighbors import minkowski_distance\n```\n\n**Compute the Minkowski ($L_p$) distance between row X[row_x] and row Y[row_y].**\n\n$$\nd(u, v) = \\left( \\sum_{j=1}^D |u_j - v_j|^p \\right)^{1/p}\n$$\n\n**Returns**: `Float64` \u2014 Float64: Minkowski distance.\n",
    "reference/neighbors/cosine_distance": "# `cosine_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef cosine_distance[dtype_x: DType, dtype_y: DType = dtype_x](X: Matrix[dtype_x], row_x: Int, Y: Matrix[dtype_y], row_y: Int) -> Float64\n```\n\n```mojo\nfrom strata.neighbors import cosine_distance\n```\n\n**Compute the Cosine distance between row X[row_x] and row Y[row_y].**\n\n$$\nd(u, v) = 1 - \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}\n$$\n\n**Returns**: `Float64` \u2014 Float64: Cosine distance in range $[0, 2]$.\n",
    "reference/neighbors/row_distance": "# `row_distance`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef row_distance[dtype: DType](X: Matrix[dtype], row_x: Int, Y: Matrix[dtype], row_y: Int, metric: String = \"euclidean\", p: Float64 = 2.0) -> Scalar[dtype]\n```\n\n```mojo\nfrom strata.neighbors import row_distance\n```\n\n**Compute pairwise distance between sample row X[row_x] and sample row Y[row_y] under metric.**\n\nSupported metrics:\n- `\"euclidean\"` or `\"l2\"`\n- `\"sqeuclidean\"`\n- `\"manhattan\"`, `\"cityblock\"`, or `\"l1\"`\n- `\"chebyshev\"`, `\"infinity\"`, or `\"max\"`\n- `\"minkowski\"` (with order parameter `p >= 1.0`)\n- `\"cosine\"`\n\n**Returns**: `Scalar[dtype]` \u2014 Scalar[dtype]: Calculated distance scalar.\n",
    "reference/neighbors/pairwise_distances": "# `pairwise_distances`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/neighbors/distance.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/distance.mojo)\n\n```mojo\ndef pairwise_distances[dtype: DType](X: Matrix[dtype], metric: String = \"euclidean\", p: Float64 = 2.0) -> Matrix[dtype]\n```\n\n```mojo\nfrom strata.neighbors import pairwise_distances\n```\n\n**Compute the self-pairwise distance matrix between all pairs of rows in X.**\n\nExploits symmetry $D_{i, j} = D_{j, i}$ and $D_{i, i} = 0$ for symmetric distance metrics.\n\n**Returns**: `Matrix[dtype]` \u2014 Matrix[dtype]: Symmetric distance matrix of shape $(N, N)$.\n",
    "reference/neighbors/NeighborDistIdx": "# `NeighborDistIdx`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Comparable, Copyable, Movable`  \n**Source**: [`strata/neighbors/base.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/base.mojo)\n\n```mojo\nstruct NeighborDistIdx(Comparable, Copyable, Movable)\n```\n\n```mojo\nfrom strata.neighbors import NeighborDistIdx\n```\n\n**Container holding a sample distance and its training dataset row index.**\n",
    "reference/neighbors/NearestNeighbors": "# `NearestNeighbors`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/neighbors/base.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/base.mojo)\n\n```mojo\nstruct NearestNeighbors[compute_dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.neighbors import NearestNeighbors\n```\n\n**Unsupervised learner for implementing neighbor searches.**\n\nFinds the $k$-nearest neighbors or all neighbors within a given radius\nusing brute-force or index-backed spatial distance metrics.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Precision used for distance computations. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_neighbors`** | Number of neighbors to use by default for `kneighbors` queries. Default 5. |\n| **`radius`** | Range of parameter space to use by default for `radius_neighbors` queries. Default 1.0. |\n| **`algorithm`** | Algorithm used to compute nearest neighbors ('auto', 'brute'). Default 'auto'. |\n| **`metric`** | Distance metric to use ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'. |\n| **`p`** | Parameter for the Minkowski metric. Default 2.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_samples_fit_`** | Number of samples in the fitted data. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`NearestNeighbors.fit()`](#fit) | Fit the nearest neighbors estimator from the training dataset. |\n| [`NearestNeighbors.kneighbors()`](#kneighbors) | Find the K-neighbors of points in X. |\n| [`NearestNeighbors.radius_neighbors()`](#radius_neighbors) | Find the neighbors within a given radius of points in X. |\n\n---\n\n## Method Details\n\n### `NearestNeighbors.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\n```\n\nFit the nearest neighbors estimator from the training dataset.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n---\n\n### `NearestNeighbors.kneighbors()`\n\n```mojo\ndef kneighbors[in_dtype: DType](self, X: Matrix[in_dtype], n_neighbors: Int = -1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]\ndef kneighbors[in_dtype: DType = DType.float64](self, n_neighbors: Int = -1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]\n```\n\nFind the K-neighbors of points in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`n_neighbors`** | `Int` | \u2014 |\n\n**Returns**: `Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]` \u2014 Tuple of: - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, n_neighbors). - Matrix[DType.int32]: Indices of neighbors in the training dataset with shape (n_queries, n_neighbors).\n\n---\n\n### `NearestNeighbors.radius_neighbors()`\n\n```mojo\ndef radius_neighbors[in_dtype: DType](self, X: Matrix[in_dtype], radius: Float64 = -1.0) -> Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]\n```\n\nFind the neighbors within a given radius of points in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`radius`** | `Float64` | \u2014 |\n\n**Returns**: `Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]` \u2014 Tuple of: - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius. - List[List[Int]]: Indices of neighbors in the training dataset.\n---\n\n## Example\n\n```mojo\nfrom strata.neighbors import NearestNeighbors\nfrom strata.core import Matrix\n\nvar nn = NearestNeighbors[DType.float64](n_neighbors=2)\nnn.fit(X_train)\nvar res = nn.kneighbors(X_test)\nvar distances = res[0]\nvar indices = res[1]\n```\n",
    "reference/neighbors/KNeighborsClassifier": "# `KNeighborsClassifier`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/neighbors/classification.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/classification.mojo)\n\n```mojo\nstruct KNeighborsClassifier[compute_dtype: DType = DType.float64](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.neighbors import KNeighborsClassifier\n```\n\n**Classifier implementing the k-nearest neighbors vote.**\n\nSupports uniform voting and inverse-distance weighted voting:\n- **Uniform weights**:\n$$\nP(y = c \\mid x) = \\frac{1}{K} \\sum_{i \\in N_K(x)} \\mathbb{I}(y_i = c)\n$$\n- **Distance weights**:\n$$\nw_i = \\frac{1}{d(x, x_i)}, \\quad P(y = c \\mid x) = \\frac{\\sum_{i \\in N_K(x)} w_i \\mathbb{I}(y_i = c)}{\\sum_{i \\in N_K(x)} w_i}\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Precision used for internal distance calculations. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_neighbors`** | Number of neighbors to use for queries. Default 5. |\n| **`weights`** | Weight function used in prediction ('uniform', 'distance'). Default 'uniform'. |\n| **`algorithm`** | Algorithm used to compute nearest neighbors ('auto', 'brute'). Default 'auto'. |\n| **`metric`** | Distance metric to use. Default 'euclidean'. |\n| **`p`** | Power parameter for the Minkowski metric. Default 2.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`classes_`** | Distinct class labels matrix of shape (n_classes,). |\n| **`n_classes_`** | Number of distinct classes seen during fit. |\n| **`n_samples_fit_`** | Number of samples in the fitted data. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`KNeighborsClassifier.fit()`](#fit) | Fit the k-nearest neighbors classifier from the training dataset. |\n| [`KNeighborsClassifier.predict_proba()`](#predict_proba) | Return probability estimates for the test data X. |\n| [`KNeighborsClassifier.predict()`](#predict) | Predict the class labels for the provided data. |\n\n---\n\n## Method Details\n\n### `KNeighborsClassifier.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the k-nearest neighbors classifier from the training dataset.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `KNeighborsClassifier.predict_proba()`\n\n```mojo\ndef predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[ feat_dtype ]\n```\n\nReturn probability estimates for the test data X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[feat_dtype]` \u2014 Matrix[feat_dtype]: Probabilities of shape (n_queries, n_classes).\n\n---\n\n### `KNeighborsClassifier.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredict the class labels for the provided data.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `List[Int]` \u2014 List[Int]: Predicted class label for each sample.\n---\n\n## Example\n\n```mojo\nfrom strata.neighbors import KNeighborsClassifier\nfrom strata.core import Matrix\n\nvar clf = KNeighborsClassifier[DType.float64](n_neighbors=3, weights=\"distance\")\nclf.fit(X_train, y_train)\nvar y_pred = clf.predict(X_test)\nvar proba = clf.predict_proba(X_test)\n```\n",
    "reference/neighbors/KNeighborsRegressor": "# `KNeighborsRegressor`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/neighbors/regression.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/regression.mojo)\n\n```mojo\nstruct KNeighborsRegressor[compute_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.neighbors import KNeighborsRegressor\n```\n\n**Regression based on k-nearest neighbors.**\n\nPredicts the target value for query points by local interpolation:\n- **Uniform weights**:\n$$\n\\hat{y}(x) = \\frac{1}{K} \\sum_{i \\in N_K(x)} y_i\n$$\n- **Distance weights**:\n$$\nw_i = \\frac{1}{d(x, x_i)}, \\quad \\hat{y}(x) = \\frac{\\sum_{i \\in N_K(x)} w_i y_i}{\\sum_{i \\in N_K(x)} w_i}\n$$\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Computational precision for distance arithmetic. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_neighbors`** | Number of neighbors to use for prediction. Default 5. |\n| **`weights`** | Weight function used in prediction ('uniform', 'distance'). Default 'uniform'. |\n| **`algorithm`** | Neighbor search algorithm ('auto', 'brute'). Default 'auto'. |\n| **`metric`** | Distance metric to use ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'). Default 'euclidean'. |\n| **`p`** | Power parameter for the Minkowski metric. Default 2.0. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_samples_fit_`** | Number of samples in the fitted data. |\n| **`n_features_in_`** | Number of features seen during fit. |\n| **`is_fitted`** | Boolean flag indicating if estimator has been fitted. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`KNeighborsRegressor.fit()`](#fit) | Fit the k-nearest neighbors regressor from the training dataset. |\n| [`KNeighborsRegressor.predict()`](#predict) | Predict the continuous target values for the provided data. |\n\n---\n\n## Method Details\n\n### `KNeighborsRegressor.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFit the k-nearest neighbors regressor from the training dataset.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `KNeighborsRegressor.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[ Scalar[feat_dtype] ]\n```\n\nPredict the continuous target values for the provided data.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `List[Scalar[feat_dtype]]` \u2014 List[Scalar[feat_dtype]]: Predicted regression values.\n---\n\n## Example\n\n```mojo\nfrom strata.neighbors import KNeighborsRegressor\nfrom strata.core import Matrix\n\nvar reg = KNeighborsRegressor[DType.float64](n_neighbors=3, weights=\"distance\")\nreg.fit(X_train, y_train)\nvar y_pred = reg.predict(X_test)\n```\n",
    "reference/neighbors/KDNode": "# `KDNode`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/neighbors/kd_tree.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/kd_tree.mojo)\n\n```mojo\nstruct KDNode(Copyable, Movable)\n```\n\n```mojo\nfrom strata.neighbors import KDNode\n```\n\n**Contiguous node in a flat KD-Tree buffer.**\n",
    "reference/neighbors/_AxisIndexPair": "# `_AxisIndexPair`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Comparable, Copyable, Movable`  \n**Source**: [`strata/neighbors/kd_tree.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/kd_tree.mojo)\n\n```mojo\nstruct _AxisIndexPair(Comparable, Copyable, Movable)\n```\n\n```mojo\nfrom strata.neighbors import _AxisIndexPair\n```\n",
    "reference/neighbors/KDTree": "# `KDTree`\n\n**Module**: [`strata.neighbors`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/neighbors/kd_tree.mojo`](file:////home/ewu/Code/Strata/strata/neighbors/kd_tree.mojo)\n\n```mojo\nstruct KDTree[compute_dtype: DType = DType.float64](Copyable, Movable)\n```\n\n```mojo\nfrom strata.neighbors import KDTree\n```\n\n**Fast spatial index for nearest neighbor and radius queries in low dimensions.**\n\nOrganizes $N$ points in $D$-dimensional space into a binary space-partitioning\ntree for $O(K \\log N)$ neighbor lookups.\n\n---\n\n## Parameters (Compile-Time)\n\n| Parameter | Description |\n| :--- | :--- |\n| **`compute_dtype`** | Precision for spatial coordinate representation. Default DType.float64. |\n\n---\n\n## Arguments (Runtime)\n\n| Argument | Description |\n| :--- | :--- |\n| **`data`** | Matrix of training points with shape (n_samples, n_features). |\n| **`metric`** | Distance metric to use ('euclidean', 'manhattan', 'chebyshev'). Default 'euclidean'. |\n\n---\n\n## Attributes\n\n| Attribute | Description |\n| :--- | :--- |\n| **`n_samples_`** | Number of samples indexed in the tree. |\n| **`n_features_`** | Dimensionality of the indexed space. |\n| **`root_idx_`** | Index of the root node in the internal buffer. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`KDTree.query()`](#query) | Query the KDTree for the k-nearest neighbors of points in X. |\n| [`KDTree.query_radius()`](#query_radius) | Find all points within distance r of points in X. |\n\n---\n\n## Method Details\n\n### `KDTree.query()`\n\n```mojo\ndef query[in_dtype: DType](self, X: Matrix[in_dtype], k: Int = 1) -> Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]\n```\n\nQuery the KDTree for the k-nearest neighbors of points in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`k`** | `Int` | \u2014 |\n\n**Returns**: `Tuple[ Matrix[in_dtype], Matrix[DType.int32] ]` \u2014 Tuple of: - Matrix[in_dtype]: Distances to neighbors with shape (n_queries, k). - Matrix[DType.int32]: Indices of neighbors in original data with shape (n_queries, k).\n\n---\n\n### `KDTree.query_radius()`\n\n```mojo\ndef query_radius[in_dtype: DType](self, X: Matrix[in_dtype], r: Float64) -> Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]\n```\n\nFind all points within distance r of points in X.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`r`** | `Float64` | \u2014 |\n\n**Returns**: `Tuple[ List[List[Scalar[in_dtype]]], List[List[Int]] ]` \u2014 Tuple of: - List[List[Scalar[in_dtype]]]: Distances to each neighbor within radius. - List[List[Int]]: Dataset indices of neighbors within radius.\n---\n\n## Example\n\n```mojo\nfrom strata.neighbors import KDTree\nfrom strata.core import Matrix\n\nvar tree = KDTree[DType.float64](X_train, metric=\"euclidean\")\nvar res = tree.query(X_query, k=3)\nvar dists = res[0]\nvar idxs = res[1]\n```\n",
    "reference/model_selection/index": "# `strata.model_selection`\n\nCross-validation splitters (K-Fold, Stratified, TimeSeries, Shuffle), cross_val_score, cross_validate, and Grid/Randomized hyperparameter search.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`Split`](Split.md) | Pair of train and validation sample indices for a cross-validation fold. |\n| [`KFold`](KFold.md) | K-Fold cross-validator. |\n| [`StratifiedKFold`](StratifiedKFold.md) | Stratified K-Fold cross-validator for classification datasets. |\n| [`TimeSeriesSplit`](TimeSeriesSplit.md) | Time Series cross-validator. |\n| [`ShuffleSplit`](ShuffleSplit.md) | Random permutation cross-validator. |\n| [`StratifiedShuffleSplit`](StratifiedShuffleSplit.md) | Stratified random permutation cross-validator. |\n| [`CrossValidateResult`](CrossValidateResult.md) | Per-fold scores for one or more metrics from a cross-validation run. |\n| [`GridSearchRegressor`](GridSearchRegressor.md) | Exhaustive hyperparameter grid search for regression models. |\n| [`GridSearchClassifier`](GridSearchClassifier.md) | Exhaustive hyperparameter grid search for classification models. |\n| [`RandomizedSearchRegressor`](RandomizedSearchRegressor.md) | Randomized hyperparameter search for regression models. |\n| [`RandomizedSearchClassifier`](RandomizedSearchClassifier.md) | Randomized hyperparameter search for classification models. |\n\n## Functions\n\n| Function | Description |\n| :--- | :--- |\n| [`train_test_split`](train_test_split.md) | Split a Dataset container into random train and test partitions. |\n| [`train_test_split`](train_test_split.md) | Split feature matrix and target list into train and test partitions. |\n| [`cross_val_score`](cross_val_score.md) | Evaluate regression scores by cross-validation across K folds. |\n| [`cross_val_score`](cross_val_score.md) | Evaluate regression scores by cross-validation on predefined splits. |\n| [`cross_val_score`](cross_val_score.md) | Evaluates classification scores by stratified cross-validation across K folds. |\n| [`cross_val_score`](cross_val_score.md) | Evaluates classification scores by cross-validation on pre-defined splits. |\n| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold regression predictions across K folds. |\n| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold regression predictions on pre-defined splits. |\n| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold class label predictions across K stratified folds. |\n| [`cross_val_predict`](cross_val_predict.md) | Generates out-of-fold class label predictions on pre-defined splits. |\n",
    "reference/model_selection/train_test_split": "# `train_test_split`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/model_selection/split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/split.mojo)\n\n```mojo\ndef train_test_split[dtype: DType = DType.float64](X: Matrix[dtype], y: List[Scalar[dtype]], test_size: Float64 = 0.25, shuffle: Bool = True, seed: Int = 42) -> DatasetSplit[dtype, dtype]\n```\n\n```mojo\nfrom strata.model_selection import train_test_split\n```\n\n**Split feature matrix and target list into train and test partitions.**\n\n**Returns**: `DatasetSplit[dtype, dtype]` \u2014 DatasetSplit: Container holding partitioned training and testing datasets.\n",
    "reference/model_selection/Split": "# `Split`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/kfold.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/kfold.mojo)\n\n```mojo\nstruct Split(Movable)\n```\n\n```mojo\nfrom strata.model_selection import Split\n```\n\n**Pair of train and validation sample indices for a cross-validation fold.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Split.copy()`](#copy) | Returns a deep copy of this split pair. |\n\n---\n\n## Method Details\n\n### `Split.copy()`\n\n```mojo\ndef copy(self) -> Self\n```\n\nReturns a deep copy of this split pair.\n\n**Returns**: `Self`\n\n---\n",
    "reference/model_selection/KFold": "# `KFold`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/kfold.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/kfold.mojo)\n\n```mojo\nstruct KFold(Movable)\n```\n\n```mojo\nfrom strata.model_selection import KFold\n```\n\n**K-Fold cross-validator.**\n\nProvides train/validation indices to split data into $K$ consecutive or shuffled folds.\nEach fold is used once as a validation set while the remaining $K-1$ folds form the\ntraining set.\n\n---\n\n## Arguments\n\n| Argument | Description |\n| :--- | :--- |\n| **`n_splits`** | Number of folds ($K >= 2$). Default 5. |\n| **`shuffle`** | Whether to shuffle the data before splitting into batches. Default False. |\n| **`random_state`** | PRNG seed when shuffle=True. Default 42. |\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`KFold.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |\n| [`KFold.split()`](#split) | Generates indices to split data into training and test sets. |\n\n---\n\n## Method Details\n\n### `KFold.get_n_splits()`\n\n```mojo\ndef get_n_splits(self) -> Int\n```\n\nReturns the number of splitting iterations in the cross-validator.\n\n**Returns**: `Int`\n\n---\n\n### `KFold.split()`\n\n```mojo\ndef split(self, n_samples: Int) -> List[Split]\ndef split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]\n```\n\nGenerates indices to split data into training and test sets.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`n_samples`** | `Int` | \u2014 |\n| **`X`** | `Matrix[dtype]` | Feature matrix. |\n\n**Returns**: `List[Split]` \u2014 List of Split objects containing train and validation indices.\n---\n\n## Example\n\n```mojo\nfrom strata.model_selection import KFold\n\nvar kf = KFold(n_splits=5, shuffle=True, random_state=42)\nvar folds = kf.split(n_samples=100)\n```\n",
    "reference/model_selection/StratifiedKFold": "# `StratifiedKFold`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/stratified_kfold.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/stratified_kfold.mojo)\n\n```mojo\nstruct StratifiedKFold(Movable)\n```\n\n```mojo\nfrom strata.model_selection import StratifiedKFold\n```\n\n**Stratified K-Fold cross-validator for classification datasets.**\n\nSplits dataset into k folds such that each fold preserves approximately\nthe same percentage of samples for each target class.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`StratifiedKFold.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations. |\n| [`StratifiedKFold.split()`](#split) | Generates stratified train and test indices from target labels. |\n\n---\n\n## Method Details\n\n### `StratifiedKFold.get_n_splits()`\n\n```mojo\ndef get_n_splits(self) -> Int\n```\n\nReturns the number of splitting iterations.\n\n**Returns**: `Int`\n\n---\n\n### `StratifiedKFold.split()`\n\n```mojo\ndef split[target_dtype: DType](self, y: List[Scalar[target_dtype]]) -> List[Split]\ndef split[feat_dtype: DType, target_dtype: DType](self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) -> List[Split]\n```\n\nGenerates stratified train and test indices from target labels.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n\n**Returns**: `List[Split]` \u2014 List of Split instances containing train and validation indices.\n\n---\n",
    "reference/model_selection/TimeSeriesSplit": "# `TimeSeriesSplit`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/time_series_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/time_series_split.mojo)\n\n```mojo\nstruct TimeSeriesSplit(Movable)\n```\n\n```mojo\nfrom strata.model_selection import TimeSeriesSplit\n```\n\n**Time Series cross-validator.**\n\nProvides train/test indices to split time series data samples that are\nobserved at fixed time intervals. In each split, test indices must be\nhigher than before, so shuffling in cross-validator is inappropriate.\nSuccessive training sets are supersets of those that come before them.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`TimeSeriesSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |\n| [`TimeSeriesSplit.split()`](#split) | Generates indices to split time-ordered data into train and test sets. |\n\n---\n\n## Method Details\n\n### `TimeSeriesSplit.get_n_splits()`\n\n```mojo\ndef get_n_splits(self) -> Int\n```\n\nReturns the number of splitting iterations in the cross-validator.\n\n**Returns**: `Int`\n\n---\n\n### `TimeSeriesSplit.split()`\n\n```mojo\ndef split(self, n_samples: Int) -> List[Split]\ndef split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]\n```\n\nGenerates indices to split time-ordered data into train and test sets.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`n_samples`** | `Int` | \u2014 |\n| **`X`** | `Matrix[dtype]` | Feature matrix. |\n\n**Returns**: `List[Split]` \u2014 List of Split objects containing train and validation indices.\n\n---\n",
    "reference/model_selection/ShuffleSplit": "# `ShuffleSplit`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/shuffle_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/shuffle_split.mojo)\n\n```mojo\nstruct ShuffleSplit(Movable)\n```\n\n```mojo\nfrom strata.model_selection import ShuffleSplit\n```\n\n**Random permutation cross-validator.**\n\nYields indices to split data into training and test sets. Each split is an\nindependent random permutation of the samples, so successive test sets may\noverlap. Sizes are expressed as proportions of the total sample count.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`ShuffleSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |\n| [`ShuffleSplit.split()`](#split) | Generates randomly permuted train and test indices for each split. |\n\n---\n\n## Method Details\n\n### `ShuffleSplit.get_n_splits()`\n\n```mojo\ndef get_n_splits(self) -> Int\n```\n\nReturns the number of splitting iterations in the cross-validator.\n\n**Returns**: `Int`\n\n---\n\n### `ShuffleSplit.split()`\n\n```mojo\ndef split(self, n_samples: Int) -> List[Split]\ndef split[dtype: DType](self, X: Matrix[dtype]) -> List[Split]\n```\n\nGenerates randomly permuted train and test indices for each split.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`n_samples`** | `Int` | \u2014 |\n| **`X`** | `Matrix[dtype]` | Feature matrix. |\n\n**Returns**: `List[Split]` \u2014 List of Split objects containing train and validation indices.\n\n---\n",
    "reference/model_selection/StratifiedShuffleSplit": "# `StratifiedShuffleSplit`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/stratified_shuffle_split.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/stratified_shuffle_split.mojo)\n\n```mojo\nstruct StratifiedShuffleSplit(Movable)\n```\n\n```mojo\nfrom strata.model_selection import StratifiedShuffleSplit\n```\n\n**Stratified random permutation cross-validator.**\n\nYields indices to split data into training and test sets. Each split is an\nindependent random draw that preserves the percentage of samples for each\ntarget class, so successive test sets may overlap.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`StratifiedShuffleSplit.get_n_splits()`](#get_n_splits) | Returns the number of splitting iterations in the cross-validator. |\n| [`StratifiedShuffleSplit.split()`](#split) | Generates class-balanced random train and test indices for each split. |\n\n---\n\n## Method Details\n\n### `StratifiedShuffleSplit.get_n_splits()`\n\n```mojo\ndef get_n_splits(self) -> Int\n```\n\nReturns the number of splitting iterations in the cross-validator.\n\n**Returns**: `Int`\n\n---\n\n### `StratifiedShuffleSplit.split()`\n\n```mojo\ndef split[target_dtype: DType](self, y: List[Scalar[target_dtype]]) -> List[Split]\ndef split[feat_dtype: DType, target_dtype: DType](self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]]) -> List[Split]\n```\n\nGenerates class-balanced random train and test indices for each split.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n\n**Returns**: `List[Split]` \u2014 List of Split objects containing train and validation indices.\n\n---\n",
    "reference/model_selection/CrossValidateResult": "# `CrossValidateResult`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Movable`  \n**Source**: [`strata/model_selection/validation.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/validation.mojo)\n\n```mojo\nstruct CrossValidateResult(Movable)\n```\n\n```mojo\nfrom strata.model_selection import CrossValidateResult\n```\n\n**Per-fold scores for one or more metrics from a cross-validation run.**\n\nScores are stored as parallel lists: metrics[m] names the metric whose\nper-fold values are held in test_scores[m] and, when requested,\ntrain_scores[m].\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`CrossValidateResult.metric_index()`](#metric_index) | Returns the position of a metric name within this result. |\n| [`CrossValidateResult.test_scores_for()`](#test_scores_for) | Returns the per-fold validation scores for a named metric. |\n| [`CrossValidateResult.train_scores_for()`](#train_scores_for) | Returns the per-fold training scores for a named metric. |\n| [`CrossValidateResult.cross_validate()`](#cross_validate) | Evaluates several regression metrics across K folds in one pass. |\n\n---\n\n## Method Details\n\n### `CrossValidateResult.metric_index()`\n\n```mojo\ndef metric_index(self, metric: String) -> Int\n```\n\nReturns the position of a metric name within this result.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`metric`** | `String` | \u2014 |\n\n**Returns**: `Int` \u2014 Index into the metrics, test_scores, and train_scores lists.\n\n---\n\n### `CrossValidateResult.test_scores_for()`\n\n```mojo\ndef test_scores_for(self, metric: String) -> List[Float64]\n```\n\nReturns the per-fold validation scores for a named metric.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`metric`** | `String` | \u2014 |\n\n**Returns**: `List[Float64]` \u2014 One score per fold, in fold order.\n\n---\n\n### `CrossValidateResult.train_scores_for()`\n\n```mojo\ndef train_scores_for(self, metric: String) -> List[Float64]\n```\n\nReturns the per-fold training scores for a named metric.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`metric`** | `String` | \u2014 |\n\n**Returns**: `List[Float64]` \u2014 One score per fold, in fold order.\n\n---\n\n### `CrossValidateResult.cross_validate()`\n\n```mojo\ndef cross_validate[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], scoring: List[String], cv: Int = 5, return_train_score: Bool = False) -> CrossValidateResult\ndef cross_validate[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split], scoring: List[String], return_train_score: Bool = False) -> CrossValidateResult\ndef cross_validate[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], scoring: List[String], cv: Int = 5, return_train_score: Bool = False) -> CrossValidateResult\ndef cross_validate[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split], scoring: List[String], return_train_score: Bool = False) -> CrossValidateResult\n```\n\nEvaluates several regression metrics across K folds in one pass.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`estimator`** | `ModelType` | \u2014 |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n| **`scoring`** | `List[String]` | \u2014 |\n| **`cv`** | `Int` | \u2014 |\n| **`return_train_score`** | `Bool` | \u2014 |\n| **`splits`** | `List[Split]` | \u2014 |\n\n**Returns**: `CrossValidateResult` \u2014 Per-fold scores for every requested metric.\n\n---\n",
    "reference/model_selection/cross_val_score": "# `cross_val_score`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/model_selection/validation.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/validation.mojo)\n\n```mojo\ndef cross_val_score[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split], scoring: String = \"accuracy\") -> List[Float64]\n```\n\n```mojo\nfrom strata.model_selection import cross_val_score\n```\n\n**Evaluates classification scores by cross-validation on pre-defined splits.**\n",
    "reference/model_selection/cross_val_predict": "# `cross_val_predict`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/model_selection/validation.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/validation.mojo)\n\n```mojo\ndef cross_val_predict[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](estimator: ModelType, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], splits: List[Split]) -> List[Int]\n```\n\n```mojo\nfrom strata.model_selection import cross_val_predict\n```\n\n**Generates out-of-fold class label predictions on pre-defined splits.**\n",
    "reference/model_selection/GridSearchRegressor": "# `GridSearchRegressor`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/model_selection/grid_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/grid_search.mojo)\n\n```mojo\nstruct GridSearchRegressor[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.model_selection import GridSearchRegressor\n```\n\n**Exhaustive hyperparameter grid search for regression models.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`GridSearchRegressor.fit()`](#fit) | Runs cross-validation across all candidate models and fits the best one. |\n| [`GridSearchRegressor.predict()`](#predict) | Predicts targets using the best discovered model configuration. |\n\n---\n\n## Method Details\n\n### `GridSearchRegressor.fit()`\n\n```mojo\ndef fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nRuns cross-validation across all candidate models and fits the best one.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `GridSearchRegressor.predict()`\n\n```mojo\ndef predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Scalar[in_feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nPredicts targets using the best discovered model configuration.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[in_feat_dtype]]`\n\n---\n",
    "reference/model_selection/GridSearchClassifier": "# `GridSearchClassifier`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/model_selection/grid_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/grid_search.mojo)\n\n```mojo\nstruct GridSearchClassifier[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.model_selection import GridSearchClassifier\n```\n\n**Exhaustive hyperparameter grid search for classification models.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`GridSearchClassifier.fit()`](#fit) | Runs stratified cross-validation across all candidate models and fits the best one. |\n| [`GridSearchClassifier.predict()`](#predict) | Predicts class labels using the best discovered model configuration. |\n| [`GridSearchClassifier.predict_proba()`](#predict_proba) | Predicts class probabilities using the best discovered model configuration. |\n\n---\n\n## Method Details\n\n### `GridSearchClassifier.fit()`\n\n```mojo\ndef fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nRuns stratified cross-validation across all candidate models and fits the best one.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |\n\n---\n\n### `GridSearchClassifier.predict()`\n\n```mojo\ndef predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredicts class labels using the best discovered model configuration.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `GridSearchClassifier.predict_proba()`\n\n```mojo\ndef predict_proba[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> Matrix[in_feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]\n```\n\nPredicts class probabilities using the best discovered model configuration.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `Matrix[in_feat_dtype]`\n\n---\n",
    "reference/model_selection/RandomizedSearchRegressor": "# `RandomizedSearchRegressor`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/model_selection/randomized_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/randomized_search.mojo)\n\n```mojo\nstruct RandomizedSearchRegressor[ModelType: Regressor, feat_dtype: DType = DType.float64, target_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.model_selection import RandomizedSearchRegressor\n```\n\n**Randomized hyperparameter search for regression models.**\n\nEvaluates a random subset of the supplied candidate configurations rather\nthan the full grid, trading exhaustive coverage for a fixed search budget.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`RandomizedSearchRegressor.fit()`](#fit) | Cross-validates a random subset of candidates and fits the best one. |\n| [`RandomizedSearchRegressor.predict()`](#predict) | Predicts targets using the best discovered model configuration. |\n\n---\n\n## Method Details\n\n### `RandomizedSearchRegressor.fit()`\n\n```mojo\ndef fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nCross-validates a random subset of candidates and fits the best one.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `RandomizedSearchRegressor.predict()`\n\n```mojo\ndef predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Scalar[in_feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nPredicts targets using the best discovered model configuration.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[in_feat_dtype]]`\n\n---\n",
    "reference/model_selection/RandomizedSearchClassifier": "# `RandomizedSearchClassifier`\n\n**Module**: [`strata.model_selection`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/model_selection/randomized_search.mojo`](file:////home/ewu/Code/Strata/strata/model_selection/randomized_search.mojo)\n\n```mojo\nstruct RandomizedSearchClassifier[ModelType: Classifier, feat_dtype: DType = DType.float64, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.model_selection import RandomizedSearchClassifier\n```\n\n**Randomized hyperparameter search for classification models.**\n\nEvaluates a random subset of the supplied candidate configurations rather\nthan the full grid, trading exhaustive coverage for a fixed search budget.\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`RandomizedSearchClassifier.fit()`](#fit) | Cross-validates a random subset of candidates and fits the best one. |\n| [`RandomizedSearchClassifier.predict()`](#predict) | Predicts class labels using the best discovered model configuration. |\n| [`RandomizedSearchClassifier.predict_proba()`](#predict_proba) | Predicts class probabilities using the best model configuration. |\n\n---\n\n## Method Details\n\n### `RandomizedSearchClassifier.fit()`\n\n```mojo\ndef fit[in_feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[in_feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nCross-validates a random subset of candidates and fits the best one.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |\n\n---\n\n### `RandomizedSearchClassifier.predict()`\n\n```mojo\ndef predict[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredicts class labels using the best discovered model configuration.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `RandomizedSearchClassifier.predict_proba()`\n\n```mojo\ndef predict_proba[in_feat_dtype: DType](self, X: Matrix[in_feat_dtype]) -> Matrix[in_feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]\n```\n\nPredicts class probabilities using the best model configuration.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `Matrix[in_feat_dtype]`\n\n---\n",
    "reference/metrics/index": "# `strata.metrics`\n\nRegression metrics (MSE, RMSE, MAE, R\u00b2), classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, Log Loss, ROC AUC), and clustering metrics (Silhouette Score).\n\n---\n\n## Functions\n\n| Function | Description |\n| :--- | :--- |\n| [`mean_squared_error`](mean_squared_error.md) | Compute Mean Squared Error (MSE) regression loss. |\n| [`root_mean_squared_error`](root_mean_squared_error.md) | Compute Root Mean Squared Error (RMSE) regression loss. |\n| [`mean_absolute_error`](mean_absolute_error.md) | Compute Mean Absolute Error (MAE) regression loss. |\n| [`r2_score`](r2_score.md) | Compute $R^2$ (coefficient of determination) regression score function. |\n| [`unique_labels`](unique_labels.md) | Sorted list of the distinct labels appearing in y_true or y_pred. |\n| [`accuracy_score`](accuracy_score.md) | Fraction (or count, if normalize is False) of correctly classified samples. |\n| [`confusion_matrix`](confusion_matrix.md) | Confusion matrix C where C[i, j] counts samples of label i predicted as label j. |\n| [`precision_score`](precision_score.md) | Compute classification precision score. |\n| [`recall_score`](recall_score.md) | Compute classification recall (sensitivity) score. |\n| [`f1_score`](f1_score.md) | Compute classification F1 score (harmonic mean of precision and recall). |\n| [`log_loss`](log_loss.md) | Compute log loss (cross-entropy loss), the negative log-likelihood of true labels. |\n| [`roc_auc_score`](roc_auc_score.md) | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC). |\n| [`silhouette_score`](silhouette_score.md) | Compute the mean Silhouette Coefficient of all samples. |\n",
    "reference/metrics/mean_squared_error": "# `mean_squared_error`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/regression.mojo`](file:////home/ewu/Code/Strata/strata/metrics/regression.mojo)\n\n```mojo\ndef mean_squared_error[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Float64\n```\n\n```mojo\nfrom strata.metrics import mean_squared_error\n```\n\n**Compute Mean Squared Error (MSE) regression loss.**\n\n$$\n\\text{MSE}(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2\n$$\n\n**Returns**: `Float64` \u2014 Float64: Non-negative floating point mean squared error.\n",
    "reference/metrics/root_mean_squared_error": "# `root_mean_squared_error`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/regression.mojo`](file:////home/ewu/Code/Strata/strata/metrics/regression.mojo)\n\n```mojo\ndef root_mean_squared_error[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Float64\n```\n\n```mojo\nfrom strata.metrics import root_mean_squared_error\n```\n\n**Compute Root Mean Squared Error (RMSE) regression loss.**\n\n$$\n\\text{RMSE}(y, \\hat{y}) = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2}\n$$\n\n**Returns**: `Float64` \u2014 Float64: Non-negative square root of mean squared error.\n",
    "reference/metrics/mean_absolute_error": "# `mean_absolute_error`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/regression.mojo`](file:////home/ewu/Code/Strata/strata/metrics/regression.mojo)\n\n```mojo\ndef mean_absolute_error[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Float64\n```\n\n```mojo\nfrom strata.metrics import mean_absolute_error\n```\n\n**Compute Mean Absolute Error (MAE) regression loss.**\n\n$$\n\\text{MAE}(y, \\hat{y}) = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|\n$$\n\n**Returns**: `Float64` \u2014 Float64: Non-negative floating point mean absolute error.\n",
    "reference/metrics/r2_score": "# `r2_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/regression.mojo`](file:////home/ewu/Code/Strata/strata/metrics/regression.mojo)\n\n```mojo\ndef r2_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Float64\n```\n\n```mojo\nfrom strata.metrics import r2_score\n```\n\n**Compute $R^2$ (coefficient of determination) regression score function.**\n\n$$\nR^2(y, \\hat{y}) = 1 - \\frac{\\sum_{i=1}^N (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^N (y_i - \\bar{y})^2}\n$$\n\n**Returns**: `Float64` \u2014 Float64: $R^2$ score (best possible score is 1.0, can be negative for arbitrarily worse models).\n",
    "reference/metrics/unique_labels": "# `unique_labels`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef unique_labels[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> List[Float64]\n```\n\n```mojo\nfrom strata.metrics import unique_labels\n```\n\n**Sorted list of the distinct labels appearing in y_true or y_pred.**\n",
    "reference/metrics/accuracy_score": "# `accuracy_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef accuracy_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], normalize: Bool = True) -> Float64\n```\n\n```mojo\nfrom strata.metrics import accuracy_score\n```\n\n**Fraction (or count, if normalize is False) of correctly classified samples.**\n",
    "reference/metrics/confusion_matrix": "# `confusion_matrix`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef confusion_matrix[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]]) -> Matrix[DType.int64]\n```\n\n```mojo\nfrom strata.metrics import confusion_matrix\n```\n\n**Confusion matrix C where C[i, j] counts samples of label i predicted as label j.**\n\nRows and columns are indexed by the sorted distinct labels.\n",
    "reference/metrics/precision_score": "# `precision_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef precision_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], average: String = \"binary\", pos_label: Float64 = 1.0, zero_division: Float64 = 0.0) -> Float64\n```\n\n```mojo\nfrom strata.metrics import precision_score\n```\n\n**Compute classification precision score.**\n\n$$\n\\text{Precision} = \\frac{TP}{TP + FP}\n$$\n\n**Returns**: `Float64` \u2014 Float64: Precision score ratio.\n",
    "reference/metrics/recall_score": "# `recall_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef recall_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], average: String = \"binary\", pos_label: Float64 = 1.0, zero_division: Float64 = 0.0) -> Float64\n```\n\n```mojo\nfrom strata.metrics import recall_score\n```\n\n**Compute classification recall (sensitivity) score.**\n\n$$\n\\text{Recall} = \\frac{TP}{TP + FN}\n$$\n\n**Returns**: `Float64` \u2014 Float64: Recall score ratio.\n",
    "reference/metrics/f1_score": "# `f1_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef f1_score[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: List[Scalar[pred_dtype]], average: String = \"binary\", pos_label: Float64 = 1.0, zero_division: Float64 = 0.0) -> Float64\n```\n\n```mojo\nfrom strata.metrics import f1_score\n```\n\n**Compute classification F1 score (harmonic mean of precision and recall).**\n\n$$\nF_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2 TP}{2 TP + FP + FN}\n$$\n\n**Returns**: `Float64` \u2014 Float64: F1 score between 0.0 and 1.0.\n",
    "reference/metrics/log_loss": "# `log_loss`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef log_loss[true_dtype: DType = DType.float64, pred_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_pred: Matrix[pred_dtype], normalize: Bool = True) -> Float64\n```\n\n```mojo\nfrom strata.metrics import log_loss\n```\n\n**Compute log loss (cross-entropy loss), the negative log-likelihood of true labels.**\n\n$$\n\\text{Log Loss} = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{k=1}^K y_{i, k} \\log(p_{i, k})\n$$\n\n**Returns**: `Float64` \u2014 Float64: The mean (or total) cross-entropy between y_true and y_pred.\n",
    "reference/metrics/roc_auc_score": "# `roc_auc_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/classification.mojo`](file:////home/ewu/Code/Strata/strata/metrics/classification.mojo)\n\n```mojo\ndef roc_auc_score[true_dtype: DType = DType.float64, score_dtype: DType = DType.float64](y_true: List[Scalar[true_dtype]], y_score: List[Scalar[score_dtype]], pos_label: Float64 = 1.0) -> Float64\n```\n\n```mojo\nfrom strata.metrics import roc_auc_score\n```\n\n**Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).**\n\n$$\n\\text{ROC AUC} = \\frac{R_1 - \\frac{n_1(n_1 + 1)}{2}}{n_1 n_0}\n$$\n\n**Returns**: `Float64` \u2014 Float64: The area under the ROC curve, between 0.0 and 1.0.\n",
    "reference/metrics/silhouette_score": "# `silhouette_score`\n\n**Module**: [`strata.metrics`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/metrics/cluster.mojo`](file:////home/ewu/Code/Strata/strata/metrics/cluster.mojo)\n\n```mojo\ndef silhouette_score[dtype: DType = DType.float64](X: Matrix[dtype], labels: List[Int]) -> Float64\n```\n\n```mojo\nfrom strata.metrics import silhouette_score\n```\n\n**Compute the mean Silhouette Coefficient of all samples.**\n\n$$\ns(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}\n$$\nwhere $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance.\nSamples alone in their cluster score 0.0.\n\n**Returns**: `Float64` \u2014 Float64: Mean Silhouette Coefficient between -1.0 and 1.0.\n",
    "reference/base/index": "# `strata.base`\n\nUnified estimator traits (Transformer, Regressor, Classifier, Clusterer) and sequential Pipeline composition wrappers.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`PipelineTransformer`](PipelineTransformer.md) | Chains two data transformers into a single composite transformer. |\n| [`PipelineRegressor`](PipelineRegressor.md) | Sequentially applies a transformer pipeline before fitting a regressor. |\n| [`PipelineClassifier`](PipelineClassifier.md) | Sequentially applies a transformer pipeline before fitting a classifier. |\n\n## Traits\n\n| Trait | Description |\n| :--- | :--- |\n| [`Estimator`](Estimator.md) | Base marker trait for all Strata estimators. |\n| [`Transformer`](Transformer.md) | Interface for data preprocessing transformers. |\n| [`Regressor`](Regressor.md) | Interface for supervised regression models. |\n| [`Classifier`](Classifier.md) | Interface for supervised classification models. |\n| [`Clusterer`](Clusterer.md) | Interface for unsupervised clustering algorithms. |\n",
    "reference/base/Estimator": "# `Estimator`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Copyable, Deinitable, Movable`  \n**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)\n\n```mojo\ntrait Estimator(Copyable, Deinitable, Movable)\n```\n\n```mojo\nfrom strata.base import Estimator\n```\n\n**Base marker trait for all Strata estimators.**\n",
    "reference/base/Transformer": "# `Transformer`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`  \n**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)\n\n```mojo\ntrait Transformer(Estimator)\n```\n\n```mojo\nfrom strata.base import Transformer\n```\n\n**Interface for data preprocessing transformers.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Transformer.fit()`](#fit) | \u2014 |\n| [`Transformer.transform()`](#transform) | \u2014 |\n| [`Transformer.fit_transform()`](#fit_transform) | \u2014 |\n\n---\n\n## Method Details\n\n### `Transformer.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n---\n\n### `Transformer.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `Transformer.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n",
    "reference/base/Regressor": "# `Regressor`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`  \n**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)\n\n```mojo\ntrait Regressor(Estimator)\n```\n\n```mojo\nfrom strata.base import Regressor\n```\n\n**Interface for supervised regression models.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Regressor.fit()`](#fit) | \u2014 |\n| [`Regressor.predict()`](#predict) | \u2014 |\n\n---\n\n## Method Details\n\n### `Regressor.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n\n---\n\n### `Regressor.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n\n**Returns**: `List[Scalar[feat_dtype]]`\n\n---\n",
    "reference/base/Classifier": "# `Classifier`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`  \n**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)\n\n```mojo\ntrait Classifier(Estimator)\n```\n\n```mojo\nfrom strata.base import Classifier\n```\n\n**Interface for supervised classification models.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Classifier.fit()`](#fit) | \u2014 |\n| [`Classifier.predict()`](#predict) | \u2014 |\n| [`Classifier.predict_proba()`](#predict_proba) | \u2014 |\n\n---\n\n## Method Details\n\n### `Classifier.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[target_dtype]])\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[target_dtype]]` | Target vector / class labels. |\n\n---\n\n### `Classifier.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `Classifier.predict_proba()`\n\n```mojo\ndef predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n\n**Returns**: `Matrix[feat_dtype]`\n\n---\n",
    "reference/base/Clusterer": "# `Clusterer`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `trait` &bull; **Traits**: `Estimator`  \n**Source**: [`strata/base/estimator.mojo`](file:////home/ewu/Code/Strata/strata/base/estimator.mojo)\n\n```mojo\ntrait Clusterer(Estimator)\n```\n\n```mojo\nfrom strata.base import Clusterer\n```\n\n**Interface for unsupervised clustering algorithms.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`Clusterer.fit()`](#fit) | \u2014 |\n| [`Clusterer.predict()`](#predict) | \u2014 |\n| [`Clusterer.fit_predict()`](#fit_predict) | \u2014 |\n| [`Clusterer.predict_proba()`](#predict_proba) | Predicts class probabilities for a Dataset container. |\n| [`Clusterer.transform()`](#transform) | Transforms dataset records and returns a new Dataset preserving labels and names. |\n| [`Clusterer.fit_transform()`](#fit_transform) | Fits transformer and transforms dataset records in place. |\n\n---\n\n## Method Details\n\n### `Clusterer.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[R: Regressor, feat_dtype: DType, target_dtype: DType](mut model: R, dataset: Dataset[feat_dtype, target_dtype])\ndef fit[C: Classifier, feat_dtype: DType, target_dtype: DType](mut model: C, dataset: Dataset[feat_dtype, target_dtype])\ndef fit[T: Transformer, feat_dtype: DType, target_dtype: DType](mut model: T, dataset: Dataset[feat_dtype, target_dtype])\ndef fit[K: Clusterer, feat_dtype: DType, target_dtype: DType](mut model: K, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits a Regressor using a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n---\n\n### `Clusterer.predict()`\n\n```mojo\ndef predict[in_dtype: DType](self, X: Matrix[in_dtype]) -> List[Int]\ndef predict[R: Regressor, feat_dtype: DType, target_dtype: DType](model: R, dataset: Dataset[feat_dtype, target_dtype]) -> List[ Scalar[feat_dtype] ]\ndef predict[C: Classifier, feat_dtype: DType, target_dtype: DType](model: C, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\ndef predict[K: Clusterer, feat_dtype: DType, target_dtype: DType](model: K, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredicts regression targets for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`model`** | `R` | \u2014 |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `Clusterer.fit_predict()`\n\n```mojo\ndef fit_predict[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> List[Int]\n```\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `Clusterer.predict_proba()`\n\n```mojo\ndef predict_proba[C: Classifier, feat_dtype: DType, target_dtype: DType](model: C, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[ feat_dtype ]\n```\n\nPredicts class probabilities for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`model`** | `C` | \u2014 |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Matrix[ feat_dtype ]`\n\n---\n\n### `Clusterer.transform()`\n\n```mojo\ndef transform[T: Transformer, feat_dtype: DType, target_dtype: DType](model: T, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\nTransforms dataset records and returns a new Dataset preserving labels and names.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`model`** | `T` | \u2014 |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Dataset[ feat_dtype, target_dtype ]`\n\n---\n\n### `Clusterer.fit_transform()`\n\n```mojo\ndef fit_transform[T: Transformer, feat_dtype: DType, target_dtype: DType](mut model: T, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[ feat_dtype, target_dtype ]\n```\n\nFits transformer and transforms dataset records in place.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`dataset`** | `Dataset[feat_dtype` | Dataset container. |\n\n**Returns**: `Dataset[ feat_dtype, target_dtype ]`\n\n---\n",
    "reference/base/PipelineTransformer": "# `PipelineTransformer`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Transformer`  \n**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)\n\n```mojo\nstruct PipelineTransformer[T1: Transformer, T2: Transformer](Copyable, Movable, Transformer)\n```\n\n```mojo\nfrom strata.base import PipelineTransformer\n```\n\n**Chains two data transformers into a single composite transformer.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`PipelineTransformer.fit()`](#fit) | \u2014 |\n| [`PipelineTransformer.transform()`](#transform) | \u2014 |\n| [`PipelineTransformer.fit_transform()`](#fit_transform) | \u2014 |\n\n---\n\n## Method Details\n\n### `PipelineTransformer.fit()`\n\n```mojo\ndef fit[in_dtype: DType](mut self, X: Matrix[in_dtype])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the transformer on Dataset feature records.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n---\n\n### `PipelineTransformer.transform()`\n\n```mojo\ndef transform[in_dtype: DType](self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef transform[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nTransforms dataset records, preserving column names and target metadata.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to transform. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n\n### `PipelineTransformer.fit_transform()`\n\n```mojo\ndef fit_transform[in_dtype: DType](mut self, X: Matrix[in_dtype]) -> Matrix[in_dtype]\ndef fit_transform[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype]) -> Dataset[feat_dtype, target_dtype]\n```\n\nFits transformer and transforms dataset records.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[in_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container to fit and transform. |\n\n**Returns**: `Matrix[in_dtype]`\n\n---\n",
    "reference/base/PipelineRegressor": "# `PipelineRegressor`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Regressor`  \n**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)\n\n```mojo\nstruct PipelineRegressor[T: Transformer, R: Regressor, target_dtype: DType = DType.float64](Copyable, Movable, Regressor)\n```\n\n```mojo\nfrom strata.base import PipelineRegressor\n```\n\n**Sequentially applies a transformer pipeline before fitting a regressor.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`PipelineRegressor.fit()`](#fit) | \u2014 |\n| [`PipelineRegressor.predict()`](#predict) | \u2014 |\n\n---\n\n## Method Details\n\n### `PipelineRegressor.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the regressor using a unified Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and targets. |\n\n---\n\n### `PipelineRegressor.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Scalar[feat_dtype]]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Scalar[feat_dtype]]\n```\n\nPredicts continuous targets for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Scalar[feat_dtype]]`\n\n---\n",
    "reference/base/PipelineClassifier": "# `PipelineClassifier`\n\n**Module**: [`strata.base`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Classifier, Copyable, Movable`  \n**Source**: [`strata/base/pipeline.mojo`](file:////home/ewu/Code/Strata/strata/base/pipeline.mojo)\n\n```mojo\nstruct PipelineClassifier[T: Transformer, C: Classifier, target_dtype: DType = DType.int32](Classifier, Copyable, Movable)\n```\n\n```mojo\nfrom strata.base import PipelineClassifier\n```\n\n**Sequentially applies a transformer pipeline before fitting a classifier.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`PipelineClassifier.fit()`](#fit) | \u2014 |\n| [`PipelineClassifier.predict()`](#predict) | \u2014 |\n| [`PipelineClassifier.predict_proba()`](#predict_proba) | \u2014 |\n\n---\n\n## Method Details\n\n### `PipelineClassifier.fit()`\n\n```mojo\ndef fit[feat_dtype: DType, in_target_dtype: DType](mut self, X: Matrix[feat_dtype], y: List[Scalar[in_target_dtype]])\ndef fit[feat_dtype: DType, target_dtype: DType](mut self, dataset: Dataset[feat_dtype, target_dtype])\n```\n\nFits the classifier using a unified Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`y`** | `List[Scalar[in_target_dtype]]` | Target vector / class labels. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature matrix and target labels. |\n\n---\n\n### `PipelineClassifier.predict()`\n\n```mojo\ndef predict[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> List[Int]\ndef predict[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> List[Int]\n```\n\nPredicts class labels for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `List[Int]`\n\n---\n\n### `PipelineClassifier.predict_proba()`\n\n```mojo\ndef predict_proba[feat_dtype: DType](self, X: Matrix[feat_dtype]) -> Matrix[feat_dtype]\ndef predict_proba[feat_dtype: DType, target_dtype: DType](self, dataset: Dataset[feat_dtype, target_dtype]) -> Matrix[feat_dtype]\n```\n\nPredicts class probability distributions for a Dataset container.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`X`** | `Matrix[feat_dtype]` | Feature matrix. |\n| **`dataset`** | `Dataset[feat_dtype, target_dtype]` | Dataset container holding feature records. |\n\n**Returns**: `Matrix[feat_dtype]`\n\n---\n",
    "reference/utils/index": "# `strata.utils`\n\n64-bit SplitMix64 PRNG with Lemire unbiased sampling, mathematical activations (softmax, sigmoid, log_sum_exp), and validation helpers.\n\n---\n\n## Structs & Classes\n\n| Struct | Description |\n| :--- | :--- |\n| [`PRNG`](PRNG.md) | 64-bit SplitMix64 pseudo-random number generator with unbiased range generation. |\n| [`NotFittedError`](NotFittedError.md) | Exception raised when an estimator is used before calling `fit`. |\n| [`DimensionMismatchError`](DimensionMismatchError.md) | Exception raised when input matrix/vector dimensions do not match requirements. |\n| [`ConvergenceError`](ConvergenceError.md) | Exception raised when iterative optimization fails to converge within max iterations. |\n| [`InvalidParameterError`](InvalidParameterError.md) | Exception raised when an invalid hyperparameter value is supplied. |\n| [`DataConversionError`](DataConversionError.md) | Exception raised when data type conversion or matrix array formatting fails. |\n\n## Functions\n\n| Function | Description |\n| :--- | :--- |\n| [`sigmoid`](sigmoid.md) | Compute the logistic sigmoid function $\\sigma(x)$. |\n| [`softmax`](softmax.md) | Compute numerically stable softmax probability distribution. |\n| [`log_sum_exp`](log_sum_exp.md) | Compute numerically stable log-sum-exp: $\\text{LSE}(x) = \\ln \\sum_i e^{x_i}$. |\n| [`check_is_fitted`](check_is_fitted.md) | \u2014 |\n| [`check_floating_dtype`](check_floating_dtype.md) | Asserts at compile time that the specified dtype is a floating-point type. |\n| [`check_array`](check_array.md) | \u2014 |\n| [`check_X_y`](check_X_y.md) | \u2014 |\n| [`check_finite`](check_finite.md) | Rejects NaN and infinite entries in a target or prediction list. |\n| [`check_consistent_length`](check_consistent_length.md) | \u2014 |\n| [`check_consistent_length`](check_consistent_length.md) | \u2014 |\n| [`check_sparse`](check_sparse.md) | Validates CSR/CSC sparse matrix format invariants. |\n",
    "reference/utils/PRNG": "# `PRNG`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable`  \n**Source**: [`strata/utils/random.mojo`](file:////home/ewu/Code/Strata/strata/utils/random.mojo)\n\n```mojo\nstruct PRNG(Copyable, Movable)\n```\n\n```mojo\nfrom strata.utils import PRNG\n```\n\n**64-bit SplitMix64 pseudo-random number generator with unbiased range generation.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`PRNG.next_u64()`](#next_u64) | \u2014 |\n| [`PRNG.next_int()`](#next_int) | Returns a pseudo-random integer in [0, upper_bound). |\n| [`PRNG.permutation()`](#permutation) | Generates a pseudo-random permutation of range(0, n) using Fisher-Yates shuffle. |\n| [`PRNG.shuffle()`](#shuffle) | In-place Fisher-Yates shuffle on a List. |\n\n---\n\n## Method Details\n\n### `PRNG.next_u64()`\n\n```mojo\ndef next_u64(mut self) -> UInt64\n```\n\n**Returns**: `UInt64`\n\n---\n\n### `PRNG.next_int()`\n\n```mojo\ndef next_int(mut self, upper_bound: Int) -> Int\n```\n\nReturns a pseudo-random integer in [0, upper_bound).\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`upper_bound`** | `Int` | \u2014 |\n\n**Returns**: `Int`\n\n---\n\n### `PRNG.permutation()`\n\n```mojo\ndef permutation(n: Int, seed: Int = 42) -> List[Int]\n```\n\nGenerates a pseudo-random permutation of range(0, n) using Fisher-Yates shuffle.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`n`** | `Int` | \u2014 |\n| **`seed`** | `Int` | \u2014 |\n\n**Returns**: `List[Int]`\n\n---\n\n### `PRNG.shuffle()`\n\n```mojo\ndef shuffle[T: Deinitable & Copyable](mut list: List[T], seed: Int = 42)\n```\n\nIn-place Fisher-Yates shuffle on a List.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`seed`** | `Int` | \u2014 |\n\n---\n",
    "reference/utils/sigmoid": "# `sigmoid`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/math.mojo`](file:////home/ewu/Code/Strata/strata/utils/math.mojo)\n\n```mojo\ndef sigmoid[dtype: DType = DType.float64](x: Scalar[dtype]) -> Scalar[dtype]\n```\n\n```mojo\nfrom strata.utils import sigmoid\n```\n\n**Compute the logistic sigmoid function $\\sigma(x)$.**\n\n$$\n\\sigma(x) = \\frac{1}{1 + e^{-x}}\n$$\n\n**Returns**: `Scalar[dtype]` \u2014 Scalar[dtype]: Evaluated sigmoid activation in the range $(0, 1)$.\n",
    "reference/utils/softmax": "# `softmax`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/math.mojo`](file:////home/ewu/Code/Strata/strata/utils/math.mojo)\n\n```mojo\ndef softmax[dtype: DType = DType.float64](x: List[Scalar[dtype]]) -> List[Scalar[dtype]]\n```\n\n```mojo\nfrom strata.utils import softmax\n```\n\n**Compute numerically stable softmax probability distribution.**\n\n$$\n\\text{Softmax}(x)_i = \\frac{e^{x_i - \\max(x)}}{\\sum_j e^{x_j - \\max(x)}}\n$$\n\n**Returns**: `List[Scalar[dtype]]` \u2014 List[Scalar[dtype]]: Normalized probability distribution vector summing to 1.0.\n",
    "reference/utils/log_sum_exp": "# `log_sum_exp`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/math.mojo`](file:////home/ewu/Code/Strata/strata/utils/math.mojo)\n\n```mojo\ndef log_sum_exp[dtype: DType = DType.float64](x: List[Scalar[dtype]]) -> Scalar[dtype]\n```\n\n```mojo\nfrom strata.utils import log_sum_exp\n```\n\n**Compute numerically stable log-sum-exp: $\\text{LSE}(x) = \\ln \\sum_i e^{x_i}$.**\n\n**Returns**: `Scalar[dtype]` \u2014 Scalar[dtype]: Evaluated log-sum-exp scalar value.\n",
    "reference/utils/check_is_fitted": "# `check_is_fitted`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_is_fitted(estimator_name: String, is_fitted: Bool)\n```\n\n```mojo\nfrom strata.utils import check_is_fitted\n```\n",
    "reference/utils/check_floating_dtype": "# `check_floating_dtype`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_floating_dtype[dtype: DType, caller: StringLiteral = \"Estimator\"]()\n```\n\n```mojo\nfrom strata.utils import check_floating_dtype\n```\n\n**Asserts at compile time that the specified dtype is a floating-point type.**\n",
    "reference/utils/check_array": "# `check_array`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_array[dtype: DType](X: Matrix[dtype], allow_empty: Bool = False, force_all_finite: Bool = True)\n```\n\n```mojo\nfrom strata.utils import check_array\n```\n",
    "reference/utils/check_X_y": "# `check_X_y`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_X_y[feat_dtype: DType, target_dtype: DType](X: Matrix[feat_dtype], y: List[Scalar[target_dtype]], force_all_finite: Bool = True)\n```\n\n```mojo\nfrom strata.utils import check_X_y\n```\n",
    "reference/utils/check_finite": "# `check_finite`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_finite[dtype: DType](values: List[Scalar[dtype]], name: String, caller: String)\n```\n\n```mojo\nfrom strata.utils import check_finite\n```\n\n**Rejects NaN and infinite entries in a target or prediction list.**\n",
    "reference/utils/check_consistent_length": "# `check_consistent_length`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_consistent_length[dtype: DType, T: Copyable](X: Matrix[dtype], y: List[T])\n```\n\n```mojo\nfrom strata.utils import check_consistent_length\n```\n",
    "reference/utils/check_sparse": "# `check_sparse`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `function`  \n**Source**: [`strata/utils/validation.mojo`](file:////home/ewu/Code/Strata/strata/utils/validation.mojo)\n\n```mojo\ndef check_sparse[dtype: DType](rows: Int, cols: Int, data: List[Scalar[dtype]], indices: List[Int], indptr: List[Int], is_csr: Bool = True, allow_empty: Bool = True, caller: String = \"SparseMatrix.__init__\")\n```\n\n```mojo\nfrom strata.utils import check_sparse\n```\n\n**Validates CSR/CSC sparse matrix format invariants.**\n",
    "reference/utils/NotFittedError": "# `NotFittedError`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`  \n**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)\n\n```mojo\nstruct NotFittedError(Copyable, Movable, Writable)\n```\n\n```mojo\nfrom strata.utils import NotFittedError\n```\n\n**Exception raised when an estimator is used before calling `fit`.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`NotFittedError.error()`](#error) | Create a formatted NotFittedError message. |\n| [`NotFittedError.write_to()`](#write_to) | \u2014 |\n\n---\n\n## Method Details\n\n### `NotFittedError.error()`\n\n```mojo\ndef error(estimator_name: String, msg: String = \"\") -> Error\n```\n\nCreate a formatted NotFittedError message.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`estimator_name`** | `String` | \u2014 |\n| **`msg`** | `String` | \u2014 |\n\n**Returns**: `Error`\n\n---\n\n### `NotFittedError.write_to()`\n\n```mojo\ndef write_to(self, mut writer: Some[Writer])\n```\n\n---\n",
    "reference/utils/DimensionMismatchError": "# `DimensionMismatchError`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`  \n**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)\n\n```mojo\nstruct DimensionMismatchError(Copyable, Movable, Writable)\n```\n\n```mojo\nfrom strata.utils import DimensionMismatchError\n```\n\n**Exception raised when input matrix/vector dimensions do not match requirements.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`DimensionMismatchError.error()`](#error) | Create a formatted DimensionMismatchError message. |\n| [`DimensionMismatchError.write_to()`](#write_to) | \u2014 |\n\n---\n\n## Method Details\n\n### `DimensionMismatchError.error()`\n\n```mojo\ndef error(expected: String, actual: String, context: String = \"\") -> Error\n```\n\nCreate a formatted DimensionMismatchError message.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`expected`** | `String` | \u2014 |\n| **`actual`** | `String` | \u2014 |\n| **`context`** | `String` | \u2014 |\n\n**Returns**: `Error`\n\n---\n\n### `DimensionMismatchError.write_to()`\n\n```mojo\ndef write_to(self, mut writer: Some[Writer])\n```\n\n---\n",
    "reference/utils/ConvergenceError": "# `ConvergenceError`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`  \n**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)\n\n```mojo\nstruct ConvergenceError(Copyable, Movable, Writable)\n```\n\n```mojo\nfrom strata.utils import ConvergenceError\n```\n\n**Exception raised when iterative optimization fails to converge within max iterations.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`ConvergenceError.error()`](#error) | Create a formatted ConvergenceError message. |\n| [`ConvergenceError.write_to()`](#write_to) | \u2014 |\n\n---\n\n## Method Details\n\n### `ConvergenceError.error()`\n\n```mojo\ndef error(estimator_name: String, max_iter: Int, loss: Float64 = 0.0) -> Error\n```\n\nCreate a formatted ConvergenceError message.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`estimator_name`** | `String` | \u2014 |\n| **`max_iter`** | `Int` | \u2014 |\n| **`loss`** | `Float64` | \u2014 |\n\n**Returns**: `Error`\n\n---\n\n### `ConvergenceError.write_to()`\n\n```mojo\ndef write_to(self, mut writer: Some[Writer])\n```\n\n---\n",
    "reference/utils/InvalidParameterError": "# `InvalidParameterError`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`  \n**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)\n\n```mojo\nstruct InvalidParameterError(Copyable, Movable, Writable)\n```\n\n```mojo\nfrom strata.utils import InvalidParameterError\n```\n\n**Exception raised when an invalid hyperparameter value is supplied.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`InvalidParameterError.error()`](#error) | Create a formatted InvalidParameterError message. |\n| [`InvalidParameterError.write_to()`](#write_to) | \u2014 |\n\n---\n\n## Method Details\n\n### `InvalidParameterError.error()`\n\n```mojo\ndef error(param_name: String, reason: String) -> Error\n```\n\nCreate a formatted InvalidParameterError message.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`param_name`** | `String` | \u2014 |\n| **`reason`** | `String` | \u2014 |\n\n**Returns**: `Error`\n\n---\n\n### `InvalidParameterError.write_to()`\n\n```mojo\ndef write_to(self, mut writer: Some[Writer])\n```\n\n---\n",
    "reference/utils/DataConversionError": "# `DataConversionError`\n\n**Module**: [`strata.utils`](index.md) &bull; **Kind**: `struct` &bull; **Traits**: `Copyable, Movable, Writable`  \n**Source**: [`strata/exceptions/errors.mojo`](file:////home/ewu/Code/Strata/strata/exceptions/errors.mojo)\n\n```mojo\nstruct DataConversionError(Copyable, Movable, Writable)\n```\n\n```mojo\nfrom strata.utils import DataConversionError\n```\n\n**Exception raised when data type conversion or matrix array formatting fails.**\n\n---\n\n## Methods Overview\n\n| Method | Description |\n| :--- | :--- |\n| [`DataConversionError.error()`](#error) | Create a formatted DataConversionError message. |\n| [`DataConversionError.write_to()`](#write_to) | \u2014 |\n\n---\n\n## Method Details\n\n### `DataConversionError.error()`\n\n```mojo\ndef error(msg: String) -> Error\n```\n\nCreate a formatted DataConversionError message.\n\n| Parameter | Type | Description |\n| :--- | :--- | :--- |\n| **`msg`** | `String` | \u2014 |\n\n**Returns**: `Error`\n\n---\n\n### `DataConversionError.write_to()`\n\n```mojo\ndef write_to(self, mut writer: Some[Writer])\n```\n\n---\n",
    "explanation/memory_and_simd": "# Memory Model, SIMD Execution & Value Semantics\n\nThis document explains the low-level design choices underpinning Strata's linear algebra engine, data layouts, and memory safety model in Mojo.\n\n\n---\n\n## 1. Contiguous Row-Major Allocation\n\nTraditional Python libraries like NumPy and scikit-learn manage memory through the CPython heap with PyObject wrappers and reference counting.\n\nIn Strata, `Matrix[dtype]` manages a single contiguous heap buffer aligned to the hardware's SIMD vector width (e.g. 512-bit for AVX-512 / ARM Neon):\n\n```mojo\nstruct Matrix[dtype: DType](Copyable, Movable):\n    var rows: Int\n    var cols: Int\n    var data: UnsafePointer[Scalar[dtype]]\n```\n\n### Benefits:\n- **Spatial Locality**: Row-major traversal matches CPU cache line loading patterns (64 bytes per line).\n- **Direct SIMD Vectorization**: Inner loops load `simd_width = simdwidthof[dtype]()` elements per CPU cycle without memory shuffling.\n\n---\n\n## 2. Zero-Copy Views via `MatrixView`\n\nWhen slicing a matrix or passing a subset of features to an estimator, creating a deep copy introduces $\\mathcal{O}(M \\times N)$ memory allocation churn.\n\nStrata provides `MatrixView[dtype, origin]`:\n- Tracks a pointer offset, row stride, and column count.\n- Enforces compile-time origin lifetime tracking to guarantee that the view cannot outlive the parent matrix.\n\n---\n\n## 3. Mojo Value Semantics & Transfer Operators\n\nStrata data structures strictly adhere to Mojo's value lifecycle:\n- **`Copyable`**: Explicit `.copy()` when a new independent buffer is required.\n- **`Movable`**: Transfer operator (`^`) moves pointers directly into target structs with zero-cost destruction of the source handle.\n\n```mojo\n# Moves the memory buffer into the estimator without heap reallocation\nvar model = PipelineRegressor(scaler^, regressor^)\n```\n",
    "explanation/estimator_traits": "# Estimator Traits & Zero-Cost Composition\n\nThis document explains Strata's trait hierarchy and how compile-time parametric polymorphism enables type-safe, composable Machine Learning pipelines without runtime overhead.\n\n\n---\n\n## 1. The Trait Hierarchy\n\nStrata structures its Machine Learning interfaces around four fundamental traits:\n\n```mermaid\nclassDiagram\n    class Estimator {\n        <<trait>>\n        +var is_fitted: Bool\n    }\n    class Transformer {\n        <<trait>>\n        +fit(X, y)\n        +transform(X)\n        +fit_transform(X, y)\n    }\n    class Regressor {\n        <<trait>>\n        +fit(X, y)\n        +predict(X)\n    }\n    class Classifier {\n        <<trait>>\n        +fit(X, y)\n        +predict(X)\n        +predict_proba(X)\n    }\n    class Clusterer {\n        <<trait>>\n        +fit(X)\n        +predict(X)\n        +fit_predict(X)\n    }\n\n    Estimator <|-- Transformer\n    Estimator <|-- Regressor\n    Estimator <|-- Classifier\n    Estimator <|-- Clusterer\n```\n\n---\n\n## 2. Compile-Time Generics vs Runtime Virtual Tables\n\nIn languages like C++ or Python, polymorphism typically uses virtual method tables (`vtable`) or dynamic method lookup. This prevents loop vectorization and function inlining across pipeline stages.\n\nIn Strata, pipelines are generic structs parameterizing the exact concrete types:\n\n```mojo\nstruct PipelineRegressor[\n    TransformerT: Transformer,\n    RegressorT: Regressor,\n    target_dtype: DType\n](Copyable, Movable, Regressor):\n    var transformer: TransformerT\n    var regressor: RegressorT\n```\n\n### Compiler Optimization:\nBecause `TransformerT` and `RegressorT` are known at compile time, the Mojo compiler directly inlines `transformer.transform(X)` and `regressor.predict(X_trans)` into a unified execution stream.\n",
    "explanation/tree_algorithms": "# Decision Tree Splitting & MDI Mechanics\n\nThis document details how Strata computes recursive binary partition splits and Mean Decrease in Impurity (MDI) feature importances with $O(1)$ streaming histogram updates.\n\n\n---\n\n## 1. Streaming Histogram Split Search\n\nGiven a continuous feature vector $\\mathbf{x} \\in \\mathbb{R}^N$ and target labels $\\mathbf{y}$, finding the optimal threshold $s$ requires evaluating impurity for all candidate split points.\n\nNaive split evaluation recalculates class counts in $\\mathcal{O}(N)$ for each candidate threshold, yielding $\\mathcal{O}(N^2)$ complexity per node.\n\n### Strata's $\\mathcal{O}(N \\log N)$ Streaming Engine:\n1. Samples are sorted by feature values: $\\mathbf{x}_{(1)} \\le \\mathbf{x}_{(2)} \\le \\dots \\le \\mathbf{x}_{(N)}$.\n2. A single pass streams counts from the **Right Partition** into the **Left Partition**:\n   $$\\text{LeftCount}_c \\leftarrow \\text{LeftCount}_c + 1, \\quad \\text{RightCount}_c \\leftarrow \\text{RightCount}_c - 1$$\n3. Gini impurity or MSE is updated in $\\mathcal{O}(K)$ time for each candidate split point:\n   $$I_{\\text{Gini}}(L) = 1 - \\sum_{c=1}^K \\left(\\frac{\\text{LeftCount}_c}{N_L}\\right)^2$$\n\n---\n\n## 2. Flat-Buffer Tree Layout\n\nInstead of pointer-chasing tree structures where each node is a separate heap allocation, Strata stores tree topologies in contiguous flat array buffers:\n\n```mojo\nstruct Node(Copyable, Movable):\n    var feature: Int        # Split feature index (-1 for leaves)\n    var threshold: Float64  # Split threshold\n    var left: Int           # Index into nodes buffer\n    var right: Int          # Index into nodes buffer\n    var impurity: Float64   # Impurity at this node\n    var n_node_samples: Int # Sample count\n```\n\n### Advantages:\n- **Cache-Friendly Inference**: Evaluating predictions streams through sequential buffer indices.\n- **Zero-Fragmentation Garbage Collection**: Destroying a tree is a single buffer deallocation.\n"
  },
  "searchIndex": [
    {
      "module": "core",
      "name": "Matrix",
      "kind": "struct",
      "summary": "Dense 2D row-major matrix container with striding and view support.",
      "ref_file": "reference/core/Matrix",
      "traits": "ArrayLike, Copyable, Movable, Writable",
      "file": "strata/core/matrix.mojo"
    },
    {
      "module": "core",
      "name": "MatrixView",
      "kind": "struct",
      "summary": "Non-owning 2D view over a contiguous or strided matrix memory buffer.",
      "ref_file": "reference/core/MatrixView",
      "traits": "ArrayLike, Copyable, Movable",
      "file": "strata/core/view.mojo"
    },
    {
      "module": "core",
      "name": "SparseMatrix",
      "kind": "trait",
      "summary": "Base interface trait for 2D sparse matrix representations.",
      "ref_file": "reference/core/SparseMatrix",
      "traits": "",
      "file": "strata/core/sparse.mojo"
    },
    {
      "module": "core",
      "name": "DatasetSplit",
      "kind": "struct",
      "summary": "Container holding train and test partitions of a Dataset.",
      "ref_file": "reference/core/DatasetSplit",
      "traits": "Movable",
      "file": "strata/core/dataset.mojo"
    },
    {
      "module": "core",
      "name": "Dataset",
      "kind": "struct",
      "summary": "Machine learning dataset container pairing a feature matrix with targets.",
      "ref_file": "reference/core/Dataset",
      "traits": "Copyable, Movable",
      "file": "strata/core/dataset.mojo"
    },
    {
      "module": "core",
      "name": "SVDResult",
      "kind": "struct",
      "summary": "Result of Singular Value Decomposition ($A = U \\Sigma V^T$).",
      "ref_file": "reference/core/SVDResult",
      "traits": "Copyable, Movable",
      "file": "strata/core/linalg.mojo"
    },
    {
      "module": "core",
      "name": "QRResult",
      "kind": "struct",
      "summary": "Result of QR Decomposition ($A = Q R$).",
      "ref_file": "reference/core/QRResult",
      "traits": "Copyable, Movable",
      "file": "strata/core/linalg.mojo"
    },
    {
      "module": "core",
      "name": "EigResult",
      "kind": "struct",
      "summary": "Result of Symmetric Eigenvalue Decomposition ($A V = V \\Lambda$).",
      "ref_file": "reference/core/EigResult",
      "traits": "Copyable, Movable",
      "file": "strata/core/linalg.mojo"
    },
    {
      "module": "core",
      "name": "matrix_to_numpy",
      "kind": "function",
      "summary": "Converts a Strata Matrix[dtype] to a NumPy 2D array.",
      "ref_file": "reference/core/matrix_to_numpy",
      "traits": "",
      "file": "strata/core/interop.mojo"
    },
    {
      "module": "core",
      "name": "matrix_from_numpy",
      "kind": "function",
      "summary": "Converts a 2D NumPy ndarray to a Strata Matrix[dtype].",
      "ref_file": "reference/core/matrix_from_numpy",
      "traits": "",
      "file": "strata/core/interop.mojo"
    },
    {
      "module": "core",
      "name": "csr_to_scipy",
      "kind": "function",
      "summary": "Converts a Strata CSRMatrix[dtype] to a scipy.sparse.csr_matrix.",
      "ref_file": "reference/core/csr_to_scipy",
      "traits": "",
      "file": "strata/core/interop.mojo"
    },
    {
      "module": "core",
      "name": "csr_from_scipy",
      "kind": "function",
      "summary": "Converts a scipy.sparse.csr_matrix to a Strata CSRMatrix[dtype].",
      "ref_file": "reference/core/csr_from_scipy",
      "traits": "",
      "file": "strata/core/interop.mojo"
    },
    {
      "module": "preprocessing",
      "name": "StandardScaler",
      "kind": "struct",
      "summary": "Standardize features by removing the mean and scaling to unit variance.",
      "ref_file": "reference/preprocessing/StandardScaler",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/scaler.mojo"
    },
    {
      "module": "preprocessing",
      "name": "MinMaxScaler",
      "kind": "struct",
      "summary": "Transform features by scaling each feature to a specified range.",
      "ref_file": "reference/preprocessing/MinMaxScaler",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/scaler.mojo"
    },
    {
      "module": "preprocessing",
      "name": "RobustScaler",
      "kind": "struct",
      "summary": "Scale features using statistics that are robust to outliers.",
      "ref_file": "reference/preprocessing/RobustScaler",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/scaler.mojo"
    },
    {
      "module": "preprocessing",
      "name": "Normalizer",
      "kind": "struct",
      "summary": "Normalize samples individually to unit norm.",
      "ref_file": "reference/preprocessing/Normalizer",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/normalizer.mojo"
    },
    {
      "module": "preprocessing",
      "name": "Binarizer",
      "kind": "struct",
      "summary": "Binarize feature values according to a threshold.",
      "ref_file": "reference/preprocessing/Binarizer",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/binarizer.mojo"
    },
    {
      "module": "preprocessing",
      "name": "OneHotEncoder",
      "kind": "struct",
      "summary": "Encode categorical features as a one-hot numeric array.",
      "ref_file": "reference/preprocessing/OneHotEncoder",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/encoders.mojo"
    },
    {
      "module": "preprocessing",
      "name": "OrdinalEncoder",
      "kind": "struct",
      "summary": "Encode categorical features as an integer array.",
      "ref_file": "reference/preprocessing/OrdinalEncoder",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/encoders.mojo"
    },
    {
      "module": "preprocessing",
      "name": "LabelEncoder",
      "kind": "struct",
      "summary": "Encode target labels with value between 0 and n_classes-1.",
      "ref_file": "reference/preprocessing/LabelEncoder",
      "traits": "Copyable, Movable",
      "file": "strata/preprocessing/encoders.mojo"
    },
    {
      "module": "preprocessing",
      "name": "SimpleImputer",
      "kind": "struct",
      "summary": "Univariate imputer for completing missing values with simple statistics.",
      "ref_file": "reference/preprocessing/SimpleImputer",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/imputer.mojo"
    },
    {
      "module": "preprocessing",
      "name": "PolynomialFeatures",
      "kind": "struct",
      "summary": "Generate polynomial and interaction features.",
      "ref_file": "reference/preprocessing/PolynomialFeatures",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/preprocessing/polynomial.mojo"
    },
    {
      "module": "linear_model",
      "name": "LinearRegression",
      "kind": "struct",
      "summary": "Ordinary Least Squares Linear Regression.",
      "ref_file": "reference/linear_model/LinearRegression",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/linear_model/linear_regression.mojo"
    },
    {
      "module": "linear_model",
      "name": "Ridge",
      "kind": "struct",
      "summary": "Ridge regression with L2 regularization.",
      "ref_file": "reference/linear_model/Ridge",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/linear_model/ridge.mojo"
    },
    {
      "module": "linear_model",
      "name": "LogisticRegression",
      "kind": "struct",
      "summary": "Logistic Regression classifier with L2 regularization.",
      "ref_file": "reference/linear_model/LogisticRegression",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/linear_model/logistic_regression.mojo"
    },
    {
      "module": "tree",
      "name": "DecisionTreeClassifier",
      "kind": "struct",
      "summary": "Decision Tree Classifier for non-parametric supervised classification.",
      "ref_file": "reference/tree/DecisionTreeClassifier",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/tree/classifier.mojo"
    },
    {
      "module": "tree",
      "name": "DecisionTreeRegressor",
      "kind": "struct",
      "summary": "Decision Tree Regressor for non-parametric continuous target regression.",
      "ref_file": "reference/tree/DecisionTreeRegressor",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/tree/regressor.mojo"
    },
    {
      "module": "ensemble",
      "name": "RandomForestRegressor",
      "kind": "struct",
      "summary": "Random Forest Regressor ensemble estimator.",
      "ref_file": "reference/ensemble/RandomForestRegressor",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/ensemble/forest.mojo"
    },
    {
      "module": "ensemble",
      "name": "RandomForestClassifier",
      "kind": "struct",
      "summary": "Random Forest Classifier ensemble estimator.",
      "ref_file": "reference/ensemble/RandomForestClassifier",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/ensemble/forest.mojo"
    },
    {
      "module": "cluster",
      "name": "KMeans",
      "kind": "struct",
      "summary": "K-Means clustering using Lloyd's or Elkan's algorithm.",
      "ref_file": "reference/cluster/KMeans",
      "traits": "Clusterer, Copyable, Movable",
      "file": "strata/cluster/kmeans.mojo"
    },
    {
      "module": "cluster",
      "name": "MiniBatchKMeans",
      "kind": "struct",
      "summary": "Mini-Batch K-Means clustering algorithm.",
      "ref_file": "reference/cluster/MiniBatchKMeans",
      "traits": "Clusterer, Copyable, Movable",
      "file": "strata/cluster/minibatch_kmeans.mojo"
    },
    {
      "module": "decomposition",
      "name": "PCA",
      "kind": "struct",
      "summary": "Principal Component Analysis (PCA).",
      "ref_file": "reference/decomposition/PCA",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/decomposition/pca.mojo"
    },
    {
      "module": "decomposition",
      "name": "TruncatedSVD",
      "kind": "struct",
      "summary": "Dimensionality reduction using truncated SVD.",
      "ref_file": "reference/decomposition/TruncatedSVD",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/decomposition/truncated_svd.mojo"
    },
    {
      "module": "neighbors",
      "name": "sqeuclidean_distance",
      "kind": "function",
      "summary": "Compute the squared Euclidean distance between row X[row_x] and row Y[row_y].",
      "ref_file": "reference/neighbors/sqeuclidean_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "euclidean_distance",
      "kind": "function",
      "summary": "Compute the Euclidean ($L_2$) distance between row X[row_x] and row Y[row_y].",
      "ref_file": "reference/neighbors/euclidean_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "manhattan_distance",
      "kind": "function",
      "summary": "Compute the Manhattan ($L_1$ / taxicab / cityblock) distance between row X[row_x] and row Y[row_y].",
      "ref_file": "reference/neighbors/manhattan_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "chebyshev_distance",
      "kind": "function",
      "summary": "Compute the Chebyshev ($L_\\infty$ / max) distance between row X[row_x] and row Y[row_y].",
      "ref_file": "reference/neighbors/chebyshev_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "minkowski_distance",
      "kind": "function",
      "summary": "Compute the Minkowski ($L_p$) distance between row X[row_x] and row Y[row_y].",
      "ref_file": "reference/neighbors/minkowski_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "cosine_distance",
      "kind": "function",
      "summary": "Compute the Cosine distance between row X[row_x] and row Y[row_y].",
      "ref_file": "reference/neighbors/cosine_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "row_distance",
      "kind": "function",
      "summary": "Compute pairwise distance between sample row X[row_x] and sample row Y[row_y] under metric.",
      "ref_file": "reference/neighbors/row_distance",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "pairwise_distances",
      "kind": "function",
      "summary": "Compute the full pairwise distance matrix between rows of X and rows of Y.",
      "ref_file": "reference/neighbors/pairwise_distances",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "pairwise_distances",
      "kind": "function",
      "summary": "Compute the self-pairwise distance matrix between all pairs of rows in X.",
      "ref_file": "reference/neighbors/pairwise_distances",
      "traits": "",
      "file": "strata/neighbors/distance.mojo"
    },
    {
      "module": "neighbors",
      "name": "NeighborDistIdx",
      "kind": "struct",
      "summary": "Container holding a sample distance and its training dataset row index.",
      "ref_file": "reference/neighbors/NeighborDistIdx",
      "traits": "Comparable, Copyable, Movable",
      "file": "strata/neighbors/base.mojo"
    },
    {
      "module": "neighbors",
      "name": "NearestNeighbors",
      "kind": "struct",
      "summary": "Unsupervised learner for implementing neighbor searches.",
      "ref_file": "reference/neighbors/NearestNeighbors",
      "traits": "Copyable, Movable",
      "file": "strata/neighbors/base.mojo"
    },
    {
      "module": "neighbors",
      "name": "KNeighborsClassifier",
      "kind": "struct",
      "summary": "Classifier implementing the k-nearest neighbors vote.",
      "ref_file": "reference/neighbors/KNeighborsClassifier",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/neighbors/classification.mojo"
    },
    {
      "module": "neighbors",
      "name": "KNeighborsRegressor",
      "kind": "struct",
      "summary": "Regression based on k-nearest neighbors.",
      "ref_file": "reference/neighbors/KNeighborsRegressor",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/neighbors/regression.mojo"
    },
    {
      "module": "neighbors",
      "name": "KDNode",
      "kind": "struct",
      "summary": "Contiguous node in a flat KD-Tree buffer.",
      "ref_file": "reference/neighbors/KDNode",
      "traits": "Copyable, Movable",
      "file": "strata/neighbors/kd_tree.mojo"
    },
    {
      "module": "neighbors",
      "name": "_AxisIndexPair",
      "kind": "struct",
      "summary": "",
      "ref_file": "reference/neighbors/_AxisIndexPair",
      "traits": "Comparable, Copyable, Movable",
      "file": "strata/neighbors/kd_tree.mojo"
    },
    {
      "module": "neighbors",
      "name": "KDTree",
      "kind": "struct",
      "summary": "Fast spatial index for nearest neighbor and radius queries in low dimensions.",
      "ref_file": "reference/neighbors/KDTree",
      "traits": "Copyable, Movable",
      "file": "strata/neighbors/kd_tree.mojo"
    },
    {
      "module": "model_selection",
      "name": "train_test_split",
      "kind": "function",
      "summary": "Split a Dataset container into random train and test partitions.",
      "ref_file": "reference/model_selection/train_test_split",
      "traits": "",
      "file": "strata/model_selection/split.mojo"
    },
    {
      "module": "model_selection",
      "name": "train_test_split",
      "kind": "function",
      "summary": "Split feature matrix and target list into train and test partitions.",
      "ref_file": "reference/model_selection/train_test_split",
      "traits": "",
      "file": "strata/model_selection/split.mojo"
    },
    {
      "module": "model_selection",
      "name": "Split",
      "kind": "struct",
      "summary": "Pair of train and validation sample indices for a cross-validation fold.",
      "ref_file": "reference/model_selection/Split",
      "traits": "Movable",
      "file": "strata/model_selection/kfold.mojo"
    },
    {
      "module": "model_selection",
      "name": "KFold",
      "kind": "struct",
      "summary": "K-Fold cross-validator.",
      "ref_file": "reference/model_selection/KFold",
      "traits": "Movable",
      "file": "strata/model_selection/kfold.mojo"
    },
    {
      "module": "model_selection",
      "name": "StratifiedKFold",
      "kind": "struct",
      "summary": "Stratified K-Fold cross-validator for classification datasets.",
      "ref_file": "reference/model_selection/StratifiedKFold",
      "traits": "Movable",
      "file": "strata/model_selection/stratified_kfold.mojo"
    },
    {
      "module": "model_selection",
      "name": "TimeSeriesSplit",
      "kind": "struct",
      "summary": "Time Series cross-validator.",
      "ref_file": "reference/model_selection/TimeSeriesSplit",
      "traits": "Movable",
      "file": "strata/model_selection/time_series_split.mojo"
    },
    {
      "module": "model_selection",
      "name": "ShuffleSplit",
      "kind": "struct",
      "summary": "Random permutation cross-validator.",
      "ref_file": "reference/model_selection/ShuffleSplit",
      "traits": "Movable",
      "file": "strata/model_selection/shuffle_split.mojo"
    },
    {
      "module": "model_selection",
      "name": "StratifiedShuffleSplit",
      "kind": "struct",
      "summary": "Stratified random permutation cross-validator.",
      "ref_file": "reference/model_selection/StratifiedShuffleSplit",
      "traits": "Movable",
      "file": "strata/model_selection/stratified_shuffle_split.mojo"
    },
    {
      "module": "model_selection",
      "name": "CrossValidateResult",
      "kind": "struct",
      "summary": "Per-fold scores for one or more metrics from a cross-validation run.",
      "ref_file": "reference/model_selection/CrossValidateResult",
      "traits": "Movable",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_score",
      "kind": "function",
      "summary": "Evaluate regression scores by cross-validation across K folds.",
      "ref_file": "reference/model_selection/cross_val_score",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_score",
      "kind": "function",
      "summary": "Evaluate regression scores by cross-validation on predefined splits.",
      "ref_file": "reference/model_selection/cross_val_score",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_score",
      "kind": "function",
      "summary": "Evaluates classification scores by stratified cross-validation across K folds.",
      "ref_file": "reference/model_selection/cross_val_score",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_score",
      "kind": "function",
      "summary": "Evaluates classification scores by cross-validation on pre-defined splits.",
      "ref_file": "reference/model_selection/cross_val_score",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_predict",
      "kind": "function",
      "summary": "Generates out-of-fold regression predictions across K folds.",
      "ref_file": "reference/model_selection/cross_val_predict",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_predict",
      "kind": "function",
      "summary": "Generates out-of-fold regression predictions on pre-defined splits.",
      "ref_file": "reference/model_selection/cross_val_predict",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_predict",
      "kind": "function",
      "summary": "Generates out-of-fold class label predictions across K stratified folds.",
      "ref_file": "reference/model_selection/cross_val_predict",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "cross_val_predict",
      "kind": "function",
      "summary": "Generates out-of-fold class label predictions on pre-defined splits.",
      "ref_file": "reference/model_selection/cross_val_predict",
      "traits": "",
      "file": "strata/model_selection/validation.mojo"
    },
    {
      "module": "model_selection",
      "name": "GridSearchRegressor",
      "kind": "struct",
      "summary": "Exhaustive hyperparameter grid search for regression models.",
      "ref_file": "reference/model_selection/GridSearchRegressor",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/model_selection/grid_search.mojo"
    },
    {
      "module": "model_selection",
      "name": "GridSearchClassifier",
      "kind": "struct",
      "summary": "Exhaustive hyperparameter grid search for classification models.",
      "ref_file": "reference/model_selection/GridSearchClassifier",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/model_selection/grid_search.mojo"
    },
    {
      "module": "model_selection",
      "name": "RandomizedSearchRegressor",
      "kind": "struct",
      "summary": "Randomized hyperparameter search for regression models.",
      "ref_file": "reference/model_selection/RandomizedSearchRegressor",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/model_selection/randomized_search.mojo"
    },
    {
      "module": "model_selection",
      "name": "RandomizedSearchClassifier",
      "kind": "struct",
      "summary": "Randomized hyperparameter search for classification models.",
      "ref_file": "reference/model_selection/RandomizedSearchClassifier",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/model_selection/randomized_search.mojo"
    },
    {
      "module": "metrics",
      "name": "mean_squared_error",
      "kind": "function",
      "summary": "Compute Mean Squared Error (MSE) regression loss.",
      "ref_file": "reference/metrics/mean_squared_error",
      "traits": "",
      "file": "strata/metrics/regression.mojo"
    },
    {
      "module": "metrics",
      "name": "root_mean_squared_error",
      "kind": "function",
      "summary": "Compute Root Mean Squared Error (RMSE) regression loss.",
      "ref_file": "reference/metrics/root_mean_squared_error",
      "traits": "",
      "file": "strata/metrics/regression.mojo"
    },
    {
      "module": "metrics",
      "name": "mean_absolute_error",
      "kind": "function",
      "summary": "Compute Mean Absolute Error (MAE) regression loss.",
      "ref_file": "reference/metrics/mean_absolute_error",
      "traits": "",
      "file": "strata/metrics/regression.mojo"
    },
    {
      "module": "metrics",
      "name": "r2_score",
      "kind": "function",
      "summary": "Compute $R^2$ (coefficient of determination) regression score function.",
      "ref_file": "reference/metrics/r2_score",
      "traits": "",
      "file": "strata/metrics/regression.mojo"
    },
    {
      "module": "metrics",
      "name": "unique_labels",
      "kind": "function",
      "summary": "Sorted list of the distinct labels appearing in y_true or y_pred.",
      "ref_file": "reference/metrics/unique_labels",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "accuracy_score",
      "kind": "function",
      "summary": "Fraction (or count, if normalize is False) of correctly classified samples.",
      "ref_file": "reference/metrics/accuracy_score",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "confusion_matrix",
      "kind": "function",
      "summary": "Confusion matrix C where C[i, j] counts samples of label i predicted as label j.",
      "ref_file": "reference/metrics/confusion_matrix",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "precision_score",
      "kind": "function",
      "summary": "Compute classification precision score.",
      "ref_file": "reference/metrics/precision_score",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "recall_score",
      "kind": "function",
      "summary": "Compute classification recall (sensitivity) score.",
      "ref_file": "reference/metrics/recall_score",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "f1_score",
      "kind": "function",
      "summary": "Compute classification F1 score (harmonic mean of precision and recall).",
      "ref_file": "reference/metrics/f1_score",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "log_loss",
      "kind": "function",
      "summary": "Compute log loss (cross-entropy loss), the negative log-likelihood of true labels.",
      "ref_file": "reference/metrics/log_loss",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "roc_auc_score",
      "kind": "function",
      "summary": "Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).",
      "ref_file": "reference/metrics/roc_auc_score",
      "traits": "",
      "file": "strata/metrics/classification.mojo"
    },
    {
      "module": "metrics",
      "name": "silhouette_score",
      "kind": "function",
      "summary": "Compute the mean Silhouette Coefficient of all samples.",
      "ref_file": "reference/metrics/silhouette_score",
      "traits": "",
      "file": "strata/metrics/cluster.mojo"
    },
    {
      "module": "base",
      "name": "Estimator",
      "kind": "trait",
      "summary": "Base marker trait for all Strata estimators.",
      "ref_file": "reference/base/Estimator",
      "traits": "Copyable, Deinitable, Movable",
      "file": "strata/base/estimator.mojo"
    },
    {
      "module": "base",
      "name": "Transformer",
      "kind": "trait",
      "summary": "Interface for data preprocessing transformers.",
      "ref_file": "reference/base/Transformer",
      "traits": "Estimator",
      "file": "strata/base/estimator.mojo"
    },
    {
      "module": "base",
      "name": "Regressor",
      "kind": "trait",
      "summary": "Interface for supervised regression models.",
      "ref_file": "reference/base/Regressor",
      "traits": "Estimator",
      "file": "strata/base/estimator.mojo"
    },
    {
      "module": "base",
      "name": "Classifier",
      "kind": "trait",
      "summary": "Interface for supervised classification models.",
      "ref_file": "reference/base/Classifier",
      "traits": "Estimator",
      "file": "strata/base/estimator.mojo"
    },
    {
      "module": "base",
      "name": "Clusterer",
      "kind": "trait",
      "summary": "Interface for unsupervised clustering algorithms.",
      "ref_file": "reference/base/Clusterer",
      "traits": "Estimator",
      "file": "strata/base/estimator.mojo"
    },
    {
      "module": "base",
      "name": "PipelineTransformer",
      "kind": "struct",
      "summary": "Chains two data transformers into a single composite transformer.",
      "ref_file": "reference/base/PipelineTransformer",
      "traits": "Copyable, Movable, Transformer",
      "file": "strata/base/pipeline.mojo"
    },
    {
      "module": "base",
      "name": "PipelineRegressor",
      "kind": "struct",
      "summary": "Sequentially applies a transformer pipeline before fitting a regressor.",
      "ref_file": "reference/base/PipelineRegressor",
      "traits": "Copyable, Movable, Regressor",
      "file": "strata/base/pipeline.mojo"
    },
    {
      "module": "base",
      "name": "PipelineClassifier",
      "kind": "struct",
      "summary": "Sequentially applies a transformer pipeline before fitting a classifier.",
      "ref_file": "reference/base/PipelineClassifier",
      "traits": "Classifier, Copyable, Movable",
      "file": "strata/base/pipeline.mojo"
    },
    {
      "module": "utils",
      "name": "PRNG",
      "kind": "struct",
      "summary": "64-bit SplitMix64 pseudo-random number generator with unbiased range generation.",
      "ref_file": "reference/utils/PRNG",
      "traits": "Copyable, Movable",
      "file": "strata/utils/random.mojo"
    },
    {
      "module": "utils",
      "name": "sigmoid",
      "kind": "function",
      "summary": "Compute the logistic sigmoid function $\\sigma(x)$.",
      "ref_file": "reference/utils/sigmoid",
      "traits": "",
      "file": "strata/utils/math.mojo"
    },
    {
      "module": "utils",
      "name": "softmax",
      "kind": "function",
      "summary": "Compute numerically stable softmax probability distribution.",
      "ref_file": "reference/utils/softmax",
      "traits": "",
      "file": "strata/utils/math.mojo"
    },
    {
      "module": "utils",
      "name": "log_sum_exp",
      "kind": "function",
      "summary": "Compute numerically stable log-sum-exp: $\\text{LSE}(x) = \\ln \\sum_i e^{x_i}$.",
      "ref_file": "reference/utils/log_sum_exp",
      "traits": "",
      "file": "strata/utils/math.mojo"
    },
    {
      "module": "utils",
      "name": "check_is_fitted",
      "kind": "function",
      "summary": "",
      "ref_file": "reference/utils/check_is_fitted",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_floating_dtype",
      "kind": "function",
      "summary": "Asserts at compile time that the specified dtype is a floating-point type.",
      "ref_file": "reference/utils/check_floating_dtype",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_array",
      "kind": "function",
      "summary": "",
      "ref_file": "reference/utils/check_array",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_X_y",
      "kind": "function",
      "summary": "",
      "ref_file": "reference/utils/check_X_y",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_finite",
      "kind": "function",
      "summary": "Rejects NaN and infinite entries in a target or prediction list.",
      "ref_file": "reference/utils/check_finite",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_consistent_length",
      "kind": "function",
      "summary": "",
      "ref_file": "reference/utils/check_consistent_length",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_consistent_length",
      "kind": "function",
      "summary": "",
      "ref_file": "reference/utils/check_consistent_length",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "check_sparse",
      "kind": "function",
      "summary": "Validates CSR/CSC sparse matrix format invariants.",
      "ref_file": "reference/utils/check_sparse",
      "traits": "",
      "file": "strata/utils/validation.mojo"
    },
    {
      "module": "utils",
      "name": "NotFittedError",
      "kind": "struct",
      "summary": "Exception raised when an estimator is used before calling `fit`.",
      "ref_file": "reference/utils/NotFittedError",
      "traits": "Copyable, Movable, Writable",
      "file": "strata/exceptions/errors.mojo"
    },
    {
      "module": "utils",
      "name": "DimensionMismatchError",
      "kind": "struct",
      "summary": "Exception raised when input matrix/vector dimensions do not match requirements.",
      "ref_file": "reference/utils/DimensionMismatchError",
      "traits": "Copyable, Movable, Writable",
      "file": "strata/exceptions/errors.mojo"
    },
    {
      "module": "utils",
      "name": "ConvergenceError",
      "kind": "struct",
      "summary": "Exception raised when iterative optimization fails to converge within max iterations.",
      "ref_file": "reference/utils/ConvergenceError",
      "traits": "Copyable, Movable, Writable",
      "file": "strata/exceptions/errors.mojo"
    },
    {
      "module": "utils",
      "name": "InvalidParameterError",
      "kind": "struct",
      "summary": "Exception raised when an invalid hyperparameter value is supplied.",
      "ref_file": "reference/utils/InvalidParameterError",
      "traits": "Copyable, Movable, Writable",
      "file": "strata/exceptions/errors.mojo"
    },
    {
      "module": "utils",
      "name": "DataConversionError",
      "kind": "struct",
      "summary": "Exception raised when data type conversion or matrix array formatting fails.",
      "ref_file": "reference/utils/DataConversionError",
      "traits": "Copyable, Movable, Writable",
      "file": "strata/exceptions/errors.mojo"
    }
  ]
};
