# NMIFeatureSelector: Feature Selection via Normalized Mutual Information

## Overview
The [`NMIFeatureSelector`](../../api/feature_selection/nmi_feature_selection.md) class performs feature selection by evaluating the Normalized Mutual Information (NMI) between features and a target variable based on the paper [`Local Bayesian Network Structure Learning for High-Dimensional Data`](https://ieeexplore.ieee.org/document/10589754). This method identifies features with strong statistical dependencies on the target while reducing redundancy among selected features. It is particularly effective for capturing non-linear relationships and is compatible with scikit-learn's `SelectorMixin` API.

---

## Mathematical Foundation

### Normalized Mutual Information (NMI)
NMI measures the dependency between two variables, normalized to the range [0, 1]. For features \(a\) and \(b\), NMI is computed as:
$$
\text{NMI}(a, b) = \frac{H(a) + H(b) - H(a, b)}{\min(H(a), H(b))}
$$
where:
- \(H(a)\) and \(H(b)\) are the entropies of \(a\) and \(b\),
- \(H(a, b)\) is their joint entropy.

Higher NMI values indicate stronger dependencies.


## Data Discretization
The algorithm discretizes features to compute entropy efficiently:
- **Integer/string columns**: Mapped to enumerated categories.
- **Float columns**: Discretized using uniform discretization with a given number of bins.

---

## Feature Selection Process
1. **First Stage**:
   - Compute NMI between each feature and the target.
   - Select features with NMI > `threshold`.

2. **Second Stage**:
   - For each pair of selected features, compute pairwise NMI.
   - Remove feature \(f_j\) if:
     - Another feature \(f_i\) has higher NMI with the target, and
     - NMI(\(f_i\), \(f_j\)) > NMI(\(f_j\), target).

This reduces redundancy while prioritizing features more relevant to the target.

---

## Example

``` py title="examples/feature_selection/nmi_fs_example.py"
--8<-- "examples/feature_selection/nmi_fs_example.py"
```
