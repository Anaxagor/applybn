# applybn

---

![License](https://img.shields.io/github/license/Anaxagor/applybn?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![python](https://img.shields.io/badge/Python-3776AB.svg?style={0}&logo=Python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style={0}&logo=scikit-learn&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)

---

## Overview

applybn is a data analysis framework that leverages causal and Bayesian modeling to deliver interpretable insights. It helps users detect anomalies, generate synthetic data for imbalanced datasets, select impactful features, and understand the reasoning behind machine learning model predictions.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---
## Core features

1. **Anomaly Detection**: Provides methods for detecting anomalies in both time-series and tabular data, utilizing Bayesian network conditional distributions and dynamic Bayesian networks (DBNs). It captures density outliers and dependency violations.
2. **Synthetic Data Generation**: Generates synthetic training data to address class imbalance problems, employing hybrid Bayesian networks with Gaussian mixture models to create balanced datasets that improve model training.
3. **Causal Feature Selection**: Offers feature selection techniques based on causal effects quantified through information theory and post-nonlinear models, identifying features with a non-zero causal impact on KPIs.
4. **Scikit-learn Compatibility**: All estimators and data transformers are designed to be compatible with the scikit-learn framework, facilitating seamless integration into existing machine learning pipelines.
5. **Bayesian Network Modeling**: Core functionality revolves around utilizing Bayesian Networks (BNs) – including Discrete, Continuous and Hybrid types - for data analysis, modeling dependencies, and generating insights.

---

## Installation

**Prerequisites:** requires Python ^3.11

Install applybn using one of the following methods:

**Build from source:**

1. Clone the applybn repository:
```sh
git clone https://github.com/Anaxagor/applybn
```

2. Navigate to the project directory:
```sh
cd applybn
```

---

## Getting Started

To get started with `applybn`, clone the repository and install the dependencies using Poetry:

```bash
git clone https://github.com/Anaxagor/applybn.git
cd applybn
poetry install
```

The documentation provides examples for various functionalities, including anomaly detection, explainable AI, feature extraction, feature selection, and handling imbalanced datasets. You can find the documentation [here](https://anaxagor.github.io/applybn/).  Example code is available in the `examples` directory.

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/Anaxagor/applybn/tree/main/docs/examples).

---

## Documentation

A detailed applybn description is available [here](https://anaxagor.github.io/applybn/).

---

## Contributing

- **[Report Issues](https://github.com/Anaxagor/applybn/issues)**: Submit bugs found or log feature requests for the project.

---

## License

This project is protected under the MIT License. For more details, refer to the [LICENSE](https://github.com/Anaxagor/applybn/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    Anaxagor (2024). applybn repository [Computer software]. https://github.com/Anaxagor/applybn

### BibTeX format:

    @misc{applybn,

        author = {Anaxagor},

        title = {applybn repository},

        year = {2024},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/Anaxagor/applybn.git}},

        url = {https://github.com/Anaxagor/applybn.git}

    }

---
