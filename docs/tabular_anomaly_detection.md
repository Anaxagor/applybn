# Tabular anomaly detection

## Overview

Tabular anomaly detection in `applybn` is built on the [article](). 
It utilizes scores from different methods to identify anomalies in tabular data.

We have slightly modified the original code to fit our needs. 
The main changes include:
- Added support for mixed data
- Added several new methods such as IQR, Cond Ratio that based on arbitrary bayesian network.

## Data types vs. Allowed methods
| Data type  | Allowed methods                                   |
|------------|---------------------------------------------------|
| Continuous | `iqr`, `original_modified`                        |
| Discrete   | `cond_ratio`, `original_modified`                 |
| Mixed      | `cond_ratio + iqr (default)`, `original_modified` |

## Usage

### With target variable
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y")
detector.fit(data)

detector.get_info(as_df=False)

print(detector.predict(data))
```

### Without target
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector()
detector.fit(data)

detector.get_info(as_df=False)

# print(detector.predict(data)) # raise an error
print(detector.predict_scores(data))
```
### Plotting result
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y")
detector.fit(data)

# detector.get_info(as_df=False)

preds = detector.predict(data)
detector.plot_result(preds) # scores or labels
```

### Changing methods
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y", model_estimation_method="iqr") # works because ecoli cont (as bams log says as well)
detector.fit(data)
preds = detector.predict_scores(data)
print(preds[:5])
```

#### Wrong usage
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y", model_estimation_method="cond_ratio") # works because ecoli cont (as bams log says as well)
detector.fit(data)
preds = detector.predict_scores(data)  # error
```

### Original modified
```python
import pandas as pd
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector

# run from applybn root or change path here
data = pd.read_csv("applybn/anomaly_detection/data/tabular/ecoli.csv")

detector = TabularDetector(target_name="y", 
                           model_estimation_method="original_modified")
detector.fit(data)
preds = detector.predict_scores(data)
```

