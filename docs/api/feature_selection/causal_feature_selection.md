# CausalFeatureSelector: Feature Selection via Causal analysis

[User Guide reference](../../user-guide/feature_selection/causal_feature_selection.md)

::: applybn.feature_selection.ce_feature_selector.CausalFeatureSelector


# Example

```python
from applybn.feature_selection.ce_feature_selector import CausalFeatureSelector

causal_selector = CausalFeatureSelector(n_bins=10)
causal_selector.fit(X_train, y_train)
X_train_selected = causal_selector.transform(X_train)
```