"""
Example usage script for the Tabular Detector class defined in tabular_detector.py.

This script demonstrates how to:
1. Load the E.coli Adult dataset (as an example).
2. Create and configure the Tabular Detector.
3. Detect anomalies in the E.coli dataset, get scores, plot the result.

Run this script to see how the methods can be chained together for end-to-end analysis.
"""

import pandas as pd
from sklearn.metrics import classification_report
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import (
    TabularDetector,
)


def main():
    # run from applybn root or change path here
    data = pd.read_csv("data/anomaly_detection/tabular/ecoli.csv")

    detector_default = TabularDetector(target_name="y")
    detector_iqr = TabularDetector(target_name="y", model_estimation_method="iqr")
    detector_IF = TabularDetector(target_name="y", additional_score="IF")

    detector_default.fit(data)
    detector_iqr.fit(data)
    detector_IF.fit(data)

    preds_default = detector_default.predict(data)
    preds_iqr = detector_iqr.predict(data)
    preds_IF = detector_IF.predict(data)

    # let's compare the result of different methods
    print(classification_report(data["y"], preds_default))
    print("___")
    print(classification_report(data["y"], preds_iqr))
    print("___")
    print(classification_report(data["y"], preds_IF))


if __name__ == "__main__":
    main()
