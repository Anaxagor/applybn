# CausalFeatureSelector

## Overview
The [`CausalFeatureSelector`](../../api/feature_selection/causal_feature_selection.md) class implements a causal model-inspired feature selection method for identifying features with significant causal effects on a target variable. This approach is based on the methodology described in the paper ["A Causal Model-Inspired Automatic Feature-Selection Method for Developing Data-Driven Soft Sensors in Complex Industrial Processes"](https://www.sciencedirect.com/science/article/pii/S2095809922005641). Instead of relying on correlation, it selects features by quantifying their causal influence through entropy reduction, ensuring interpretable and causally relevant feature subsets.


## Key Features
- **Causal Effect Estimation**: Evaluates features based on their ability to reduce uncertainty in the target variable.
- **IQR Discretization**: Automatically bins continuous features using the Interquartile Range (IQR) rule.
- **Scikit-Learn Integration**: Compatible with scikit-learn pipelines and transformers.


## Mathematical Background

### Discretization
Features and the target variable are discretized using IQR-based binning. The number of bins `n_bins` is calculated as:
$$
n_{\text{bins}} = \max\left(2, \left\lceil \frac{R}{2 \cdot \text{IQR} \cdot n^{1/3}} \cdot \log_2(n + 1) \right\rceil \right)
$$
where \( R \) is the data range, \( \text{IQR} \) is the interquartile range, and \( n \) is the sample size.

### Causal Effect Calculation
The causal effect of feature \( X_i \) on target \( Y \) is computed as the reduction in conditional entropy:
$$
\text{CE}(X_i \rightarrow Y) = H(Y \mid \text{other features}) - H(Y \mid X_i, \text{other features})
$$
where \( H \) denotes entropy. Features with \( \text{CE} > 0 \) are retained.

## Example
``` py title="examples/feature_selection/causal_fs_example.py"
--8<-- "examples/feature_selection/causal_fs_example.py"
```
