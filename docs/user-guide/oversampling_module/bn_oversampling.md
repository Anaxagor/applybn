# Oversampling based Bayesian networks

## Overview
The [`BNOverSampler`](../../api/oversampling/bn_oversampling.md) implements a Bayesian Network-based approach to address class imbalance by learning feature distributions through **mixtures of Gaussians** and generating synthetic samples. This method captures complex feature relationships and multimodal distributions, offering statistically rigorous oversampling.

## Mathematical Foundation

### Bayesian Network with Mixture Models
The oversampler employs a Bayesian Network (BN) where nodes represent features and edges encode conditional dependencies. For flexible distribution modeling:
1. **Continuous variables** are modeled using **Gaussian Mixture Models (GMM)**:
   $$
   P(X_i | \text{Pa}(X_i)) = \sum_{k=1}^K \pi_k \mathcal{N}(X_i | \mu_k, \Sigma_k)
   $$
   where:
   - $\pi_k$: Mixture weights ($\sum \pi_k = 1$)
   - $\mu_k, \Sigma_k$: Mean and covariance of the $k$-th Gaussian component
   - $\text{Pa}(X_i)$: Parent nodes of $X_i$ in the BN

2. **Discrete variables** use multinomial distributions conditioned on parent states.

### Parameter Learning
The BN structure and parameters are learned via:
- **Structure Learning**: Hill Climbing algorithm
- **Parameter Estimation**: Expectation-Maximization (EM) algorithm for GMM components:
  $$
  \theta^* = \arg\max_\theta \mathbb{E}_{Z|X,\theta^{old}}[\log P(X,Z|\theta)]
  $$
  where $Z$ represents latent mixture component assignments.

### Synthetic Generation
For minority class $C$, samples are drawn from the conditional distribution:
$$
P(\mathbf{X} | C) = \prod_{i=1}^d P(X_i | \text{Pa}(X_i), C)
$$
using ancestral sampling through the BN, with evidence fixed on the target class.

## Key Features
- **Multimodal Modeling**: GMMs capture complex distributions (e.g., multiple peaks in feature values)
- **Conditional Sampling**: Generates samples respecting feature dependencies via BN structure
- **Discrete-Continuous Hybrid**: Handles mixed data types natively through:
  - Quantile discretization (preprocessing)
  - Type-aware sampling (_disc_num_ columns cast to integers post-generation)

## Advantages Over Traditional Methods
1. Preserves **non-linear correlations** between features
2. Avoids interpolation artifacts (common in SMOTE) through probabilistic sampling
3. Adapts to heterogeneous distributions via mixture components

## Example


``` py title="examples/imbalanced/iris-example.py"
--8<-- "examples/imbalanced/iris-example.py"
```
