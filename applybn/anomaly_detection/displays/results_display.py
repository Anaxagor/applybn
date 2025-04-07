import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

class ResultsDisplay:
    def __init__(self, outlier_scores=None, y_true = None):
        self.outlier_scores = outlier_scores
        self.y_true = y_true

    def show(self):
        outlier_scores = np.array(self.outlier_scores)
        y_true = np.array(self.y_true)

        final = pd.DataFrame(np.hstack([outlier_scores.reshape(-1, 1),
                                        y_true.reshape(-1, 1).astype(int)]),
                         columns=["score", "anomaly"])
        plt.figure(figsize=(20, 12))
        sns.scatterplot(data=final, x=range(final.shape[0]), s=20,
                        y="score", hue="anomaly")

        plt.show()