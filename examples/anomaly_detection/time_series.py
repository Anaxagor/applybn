from applybn.anomaly_detection.dynamic_anomaly_detector.fast_time_series_detector import (
    FastTimeSeriesDetector,
)
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def add_anomalies(df, anomaly_fraction=0.05, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    df_anomaly = df.copy()
    total = df.shape[0]

    # Initialize label matrix with zeros
    anomaly_labels = np.zeros(total, dtype=int)

    n_anomalies = int(df.shape[0] * anomaly_fraction)

    # Generate random positions
    rows = np.random.randint(0, df.shape[0], size=n_anomalies)

    for row_idx in rows:
        new_value = np.random.choice(["a", "b", "c"], size=df.shape[1] - 1)
        anomaly_labels[row_idx] = 1

        df_anomaly.iloc[row_idx, 1:] = new_value

    return df_anomaly, anomaly_labels


df_discrete = pd.read_csv(
    "data/anomaly_detection/ts/meteor_discrete_example_data.csv"
)

df_anomaly, anomalies = add_anomalies(
    df_discrete, anomaly_fraction=0.1, random_state=42
)

detector = FastTimeSeriesDetector(markov_lag=1, num_parents=1)

detector.fit(df_anomaly)
detector.calibrate(anomalies)
preds_cont = detector.predict(df_anomaly)

print(f1_score(anomalies, preds_cont))  # 0.7352941176470589 (may vary)
