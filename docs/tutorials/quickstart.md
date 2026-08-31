# Getting Started with Strata

Strata is a high-performance machine learning library written in native Mojo. It provides scikit-learn compatible estimator APIs while taking advantage of Mojo's compile-time optimizations and SIMD acceleration.

---

## 1. Installation

Strata uses [`pixi`](https://pixi.sh) to manage the Mojo toolchain and C linear algebra dependencies (LAPACK/BLAS).

Clone the repository and set up the environment:

```bash
git clone https://github.com/ethqnol/Strata.git
cd Strata
pixi install
```

Run the test suite to verify your setup:

```bash
pixi run test-ensemble
```

---

## 2. Training Your First Model

Here is a complete example of creating a dataset, training a `RandomForestClassifier`, and evaluating predictions.

Create a file named `main.mojo`:

```mojo
from strata.core.matrix import Matrix
from strata.ensemble.forest import RandomForestClassifier
from strata.metrics.classification import accuracy_score

def main() raises:
    # Allocate an 8x2 floating-point feature matrix
    var X = Matrix[DType.float64](8, 2)

    # Class 0 samples (clustered near negative values)
    X[0, 0] = -3.0
    X[0, 1] = -2.0
    X[1, 0] = -2.0
    X[1, 1] = -3.0
    X[2, 0] = -4.0
    X[2, 1] = -2.5
    X[3, 0] = -2.5
    X[3, 1] = -4.0

    # Class 1 samples (clustered near positive values)
    X[4, 0] = 3.0
    X[4, 1] = 2.0
    X[5, 0] = 2.0
    X[5, 1] = 3.0
    X[6, 0] = 4.0
    X[6, 1] = 2.5
    X[7, 0] = 2.5
    X[7, 1] = 4.0

    # Class labels for the 8 samples
    var y = List[Scalar[DType.int32]]()
    for _ in range(4):
        y.append(0)
    for _ in range(4):
        y.append(1)

    # Initialize the random forest classifier
    var rf = RandomForestClassifier[DType.float64](
        n_estimators=20,
        max_depth=4,
        random_state=42
    )

    # Fit model parameters on training data
    rf.fit(X, y)

    # Predict class labels and probabilities
    var preds = rf.predict(X)
    var proba = rf.predict_proba(X)

    # Evaluate accuracy
    var acc = accuracy_score(y, preds)
    print("Training Accuracy:", acc)
```

Run the script:

```bash
pixi run mojo run -I . main.mojo
```

---

## Key Concepts

- **`Matrix[dtype]`**: Strata's contiguous 2D dense matrix format. Specifying `[DType.float64]` configures numeric precision at compile time.
- **`fit(X, y)` and `predict(X)`**: Standard estimator methods used across all classification and regression models in Strata.
- **`predict_proba(X)`**: Returns an `(N, C)` matrix containing predicted probabilities for each class across all samples.

Next, read [Composing End-to-End ML Pipelines](end_to_end_pipeline.md) to learn how to pair preprocessors and models into single reusable pipelines.
