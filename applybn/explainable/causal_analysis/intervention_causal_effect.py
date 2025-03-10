from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)

from applybn.core.data_iq import DataIQSKLearn
from applybn.core import get_logger, track

logger = get_logger(__name__)


class InterventionCausalExplainer:
    def __init__(self, n_estimators=10):
        """Initialize the ModelInterpreter.

        Attributes:
            n_estimators: Number of estimators for Data-IQ.
        """
        logger.info("Initializing InterventionCausalExplainer with %d estimators", n_estimators)
        self.n_estimators = n_estimators
        self.clf = None
        self.dataiq_train = None
        self.dataiq_test = None
        self.confidence_train = None
        self.confidence_test = None
        self.aleatoric_uncertainty_train = None
        self.aleatoric_uncertainty_test = None
        self.feature_effects = None
        self.confidence_test_before_intervention = None
        self.aleatoric_uncertainty_test_before_intervention = None

    def train_model(self, model: Union[BaseEstimator, ClassifierMixin], X, y):
        """Train the model on the training data.

        Args:
            model: The model to train
            X: Training data
            y: Training labels
        """
        logger.info("Training the model with %d samples", X.shape[0])
        self.clf = model
        self.clf.fit(X, y)
        logger.info("Model training complete.")

    def _compute_confidence_uncertainty(self, X, y, suffix: str):
        """Helper to compute confidence and uncertainty for train/test data using Data-IQ."""
        data_type = "training" if suffix == "train" else "test"
        logger.info(
            f"Computing confidence and uncertainty on {data_type} data using Data-IQ."
        )

        dataiq = DataIQSKLearn(X=X, y=y)
        dataiq.on_epoch_end(clf=self.clf, iteration=self.n_estimators)

        setattr(self, f"dataiq_{suffix}", dataiq)
        setattr(self, f"confidence_{suffix}", dataiq.confidence)
        setattr(self, f"aleatoric_uncertainty_{suffix}", dataiq.aleatoric)

    def compute_confidence_uncertainty_train(self, X, y):
        """Compute model confidence and aleatoric uncertainty on training data using Data-IQ."""
        self._compute_confidence_uncertainty(X, y, "train")

    def compute_confidence_uncertainty_test(self, X, y):
        """Compute model confidence and aleatoric uncertainty on test data using Data-IQ."""
        self._compute_confidence_uncertainty(X, y, "test")

    def estimate_feature_impact(self, X, random_state=42):
        """Estimate the causal effect of each feature on the model's confidence using training data."""
        logger.info(
            "Estimating feature impact using causal inference on training data."
        )
        self.feature_effects = {}
        for feature in track(X.columns, description="Estimating feature impacts"):
            logger.debug(f"Estimating effect of feature '{feature}'.")
            treatment = X[feature].values
            outcome = self.confidence_train
            covariates = X.drop(columns=[feature])

            est = CausalForestDML(
                model_y=RandomForestRegressor(),
                model_t=RandomForestRegressor(),
                discrete_treatment=False,
                random_state=random_state,
            )
            est.fit(Y=outcome, T=treatment, X=covariates)
            te = est.const_marginal_effect(covariates).mean()
            self.feature_effects[feature] = te

        # Convert to Series and sort
        self.feature_effects = (
            pd.Series(self.feature_effects).abs().sort_values(ascending=False)
        )
        logger.info("Feature effects estimated.")

    def plot_aleatoric_uncertainty(self, before_intervention: bool = True):
        """Plot aleatoric uncertainty for test data before and after intervention."""
        if before_intervention:
            plt.figure(figsize=(10, 5))
            plt.hist(
                self.aleatoric_uncertainty_test_before_intervention,
                bins=30,
                alpha=0.5,
                label="Uncertainty Before Intervention",
            )
            plt.hist(
                self.aleatoric_uncertainty_test,
                bins=30,
                alpha=0.5,
                color="red",
                label="Uncertainty After Intervention",
            )
            plt.title("Test Data: Aleatoric Uncertainty Before and After Intervention")
            plt.xlabel("Aleatoric Uncertainty")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

    def plot_top_feature_effects(self, top_n: int = 10):
        """Plot a bin plot of the top N most impactful features with their causal effects.

        Args:
            top_n: Number of top features to plot.
        """
        top_features = self.feature_effects.head(top_n)
        plt.figure(figsize=(10, 8))
        top_features.plot(kind="bar", color="skyblue")
        plt.title(f"Top {top_n} Most Impactful Features by Causal Effect")
        plt.xlabel("Features")
        plt.ylabel("Causal Effect")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def perform_intervention(self, X_test, y_test):
        """Perform an intervention on the top 5 most impactful features in the test data and observe changes."""
        if self.feature_effects is None:
            raise ValueError("Feature effects have not been estimated yet.")

        top_features = self.feature_effects.head(5).index.tolist()
        logger.info(f"Top {len(top_features)} most impactful features: {top_features}")

        # Compute confidence on test data before intervention
        self.compute_confidence_uncertainty_test(X=X_test, y=y_test)
        self.confidence_test_before_intervention = self.confidence_test.copy()
        self.aleatoric_uncertainty_test_before_intervention = (
            self.aleatoric_uncertainty_test.copy()
        )

        original_feature_values_test = X_test[top_features].copy()

        for feature in track(top_features, description="Performing interventions"):
            plt.figure(figsize=(10, 5))
            plt.hist(
                original_feature_values_test[feature],
                bins=30,
                alpha=0.5,
                label="Before Intervention",
            )

            logger.debug(f"Performing intervention on '{feature}' in test data.")
            min_val = original_feature_values_test[feature].min()
            max_val = original_feature_values_test[feature].max()
            np.random.seed(42)
            new_values = np.random.uniform(
                low=min_val, high=max_val, size=X_test.shape[0]
            )
            X_test[feature] = new_values

            plt.hist(
                X_test[feature],
                bins=30,
                alpha=0.5,
                color="orange",
                label="After Intervention",
            )
            plt.title(
                f"Test Data: Distribution of '{feature}' Before and After Intervention"
            )
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        self.compute_confidence_uncertainty_test(X=X_test, y=y_test)

        plt.figure(figsize=(10, 5))
        plt.hist(
            self.confidence_test_before_intervention,
            bins=30,
            alpha=0.5,
            label="Confidence Before Intervention",
        )
        plt.hist(
            self.confidence_test,
            bins=30,
            alpha=0.5,
            color="green",
            label="Confidence After Intervention",
        )
        plt.title(
            f"Test Data: Model Confidence Before and After Intervention on {len(top_features)} features"
        )
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

        self.plot_aleatoric_uncertainty()

        logger.info(
            "Intervention complete. Observed changes in model confidence on test data."
        )

    def interpret(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """Run the full interpretation process."""
        self.train_model(model=model, X=X_train, y=y_train)
        self.compute_confidence_uncertainty_train(X=X_train, y=y_train)
        self.estimate_feature_impact(X=X_train)
        self.plot_top_feature_effects()
        self.perform_intervention(X_test=X_test, y_test=y_test)

