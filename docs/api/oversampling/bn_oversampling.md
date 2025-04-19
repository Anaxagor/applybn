# Bayesian Network oversampling

[User Guide reference](../../user-guide/oversampling_module/bn_oversampling.md)

::: applybn.imbalanced.over_sampling.BNOverSampler

# Example

```python
from applybn.imbalanced.over_sampling import BNOverSampler

# Initialize with GMM-based BN (auto-configured via use_mixture=True)
oversampler = BNOverSampler(
    class_column='target', 
    strategy='max_class'  # Match largest class size
)

# Generates samples using P(X|class) from learned BN
X_res, y_res = oversampler.fit_resample(X, y)
```