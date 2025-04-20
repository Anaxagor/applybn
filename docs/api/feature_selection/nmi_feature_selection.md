# NMIFeatureSelector: Feature Selection via Normalized Mutual Information

[User Guide reference](../../user-guide/feature_selection/nmi_feature_selection.md)

::: applybn.feature_selection.bn_nmi_feature_selector.NMIFeatureSelector


# Example

```python
from applybn.feature_selection.bn_nmi_feature_selector import NMIFeatureSelector
from sklearn.ensemble import RandomForestClassifier

selector = NMIFeatureSelector(threshold=0.5, n_bins=20)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
```