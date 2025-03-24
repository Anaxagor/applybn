# Estimators, pipelines and bayesian networks (Advanced)

The purpose of estimators in the core is supporting developers by creating 
universal wrapper on bamt with sklearn format. Thus, we have to implement 
variability of estimators' interfaces. 

The applybn has the following structure:
```
Bayessian networks (core: bamt)
         |
         |
         ↓ 
  BN Estimators (wrapper on bamt's networks) [developer level]
         |
         |
         ↓ 
     Pipelines (applybn interface for bamt) [user level]
```

## Bayesian Networks

In bamt there are 3 types of networks: `HybridBN`, `ContinousBN`, `DiscreteBN`.
And there are also autotyping of dataframe. 

Please consider supported types:
```python
disc = ["str", "O", "b", "categorical", "object", "bool"]
disc_numerical = ["int32", "int64"]
cont = ["float32", "float64"]
```
!!! danger

    **Please do not use this level unless necessary.**


## Estimators

Estimators are designed to be used by developers. 
They inherit sklearn `BaseEstimator`.

### Detecting bn type

This method is static and can be used to determine bn_type depending on user's data.

```python
estimator = BNEstimator()

data = pd.read_csv(DATA_PATH)
estimator.detect_bn(data)
```

### Fit method

```python
estimator = BNEstimator()

data = pd.read_csv(DATA_PATH)
descriptor = {"types": {...}, "signs": {...}}
pp_data = preprocessor([...]).apply(data) # use any preprocessing method

fit_package = (pp_data, descriptor, data)

estimator.fit(fit_package)
```

#### Learning structure and parameters (partial=False; default value)
Low-level behaviour:
```python
bn = HybridBN(use_mixture=False, has_logit=True) # can be any net
bn.add_nodes(descriptor)
bn.add_edges(X, scoring_function=("MI", ))
bn.fit_parameters(clean_data)
```
Corresponding applybn way:
```python
estimator = BNEstimator(use_mixture=False, has_logit=True,
                        learning_params={"scoring_function": ("MI", )})

<...>

estimator.fit(fit_package)
```

#### Learning only structure (partial="structure")
Low-level behaviour:
```python
bn.add_nodes(descriptor)
bn.add_edges(X)
```

Corresponding applybn way:
```python
estimator = BNEstimator(partial="structure")

<...>

estimator.fit(fit_package)
```

#### Learning only parameters (partial="parameters")
This way requires a pretrained structure, so one need to set `estimator.bn_` 
and `estimator.bn_.edges` must not be empty, otherwise `NotFittedError` is raised.

Low-level behaviour:
```python
bn.fit_parameters(clean_data)
```
Corresponding applybn way:
```python
estimator = BNEstimator(partial="parameters")

<...>

estimator.fit(fit_package)
```

### Customizing estimators

You can create your own estimator object by using metaprogramming inheritance:

```python
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory
import inspect

factory = EstimatorPipelineFactory(task_type="classification")
estimator_with_default_interface = factory.estimator.__class__

class StaticEstimator(estimator_with_default_interface):
    def __init__(self):
        pass

my_estimator = StaticEstimator()

print(*inspect.getmembers(my_estimator), sep="\n") # check that all methods are in
```

### Delegation strategy

An unknown to `BNEstimator` method will be delegated to `self.bn_` if any, 
otherwise `NotFittedError` or `AttributeError` will be thrown.

## Pipelines

Pipelines are combinations of `bamt_preprocessor` and `BNEstimator`. If you want you can change 
preprocessor on anything compatible with scikit-learn interface.

Pipelines follow a factory pattern, any `getattr` call will be delegated
to the last step of a pipeline (usually `BNEstimator`).

### Initializing factory
On initialization one must specify a solving task. It can be classification or regression.
This is done for assigning proper interface to `BaseEstimators`:

```python
interfaces = {"classification": ClassifierMixin,
              "regression": RegressorMixin}
```

```python
import pandas as pd
from applybn.core.estimators.estimator_factory import EstimatorPipelineFactory

X = pd.read_csv(DATA_PATH)
# y = X.pop("anomaly") # pop target if any

factory = EstimatorPipelineFactory(task_type="classification")
```

### Creating pipeline
```python
# pipeline with default preprocessor
pipeline = factory()
# pipeline = factory(preprocessor) 

pipeline.fit(X)
```

### Managing pipeline attributes
Let's say we want to learn structure first, do something else, and then learn parameters.
```python
# making only structure with MI sf
pipeline = factory(partial="structure", learning_params=dict(scoring_function="MI"))
pipeline.fit(X)

# doing some stuff here
<...>

# learning parameters then
pipeline.set_params(bn_estimator__partial="parameters")
pipeline.fit(X)

print(pipeline.bn_.edges)
```

Notice how we have set parameters. There are default names in factory:
```python
CorePipeline([("preprocessor", wrapped_preprocessor),
              ("bn_estimator", self.estimator_)])
```

So one can set attributes to preprocessors as well by `preprocessor__attrName`.

!!! tip

    You can set parameters on initializing pipeline and after this.

### Delegation strategy

User can use `bamt` methods on `pipeline`, their calls will be delegated to the last step of 
pipeline's chain. 

```python
factory = EstimatorPipelineFactory(task_type="classification")

pipeline = factory()
pipeline.fit(X)

pipeline.get_info(as_df=False) 
pipeline.save("mybn")
```

## Creating your own preprocessors

If you want to create your own preprocessor you must make it scikit-learn compatible.

Consider inheritance from `BaseEstimator` and `TransformerMixin` and
don't forget to implement `transform` method.
```python
class PreprocessorWrapper(BaseEstimator, TransformerMixin):
       <...>
       
       def transform(self, X):
            df = do_smth(X)
        
            return df
```