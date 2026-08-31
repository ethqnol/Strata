# `strata.base`

Unified estimator traits (Transformer, Regressor, Classifier, Clusterer) and sequential Pipeline composition wrappers.

---

## Structs & Classes

| Struct | Description |
| :--- | :--- |
| [`PipelineTransformer`](PipelineTransformer.md) | Chains an arbitrary variadic sequence of data transformers into a single composite transformer. |
| [`PipelineRegressor`](PipelineRegressor.md) | Sequentially applies a transformer pipeline before fitting a regressor. |
| [`PipelineClassifier`](PipelineClassifier.md) | Sequentially applies a transformer pipeline before fitting a classifier. |

## Traits

| Trait | Description |
| :--- | :--- |
| [`Estimator`](Estimator.md) | Base marker trait for all Strata estimators. |
| [`Transformer`](Transformer.md) | Interface for data preprocessing transformers. |
| [`Regressor`](Regressor.md) | Interface for supervised regression models. |
| [`Classifier`](Classifier.md) | Interface for supervised classification models. |
| [`Clusterer`](Clusterer.md) | Interface for unsupervised clustering algorithms. |
