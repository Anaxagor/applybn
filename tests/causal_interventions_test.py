import unittest
import logging
import pandas as pd
from unittest.mock import patch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from applybn.explainable.causal_analysis import InterventionCausalExplainer


class TestInterventionCausalExplainer(unittest.TestCase):
    """
    Tests the InterventionCausalExplainer class."""

    @classmethod
    def setUpClass(cls):
        """
        Loads and splits the breast cancer dataset for testing.

            Args:
                None

            Returns:
                None
        """
        # Load sample data using breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
        ) = train_test_split(X, y, test_size=0.2, random_state=42)

    def setUp(self):
        """
        Initializes the object under test before each test.

            Args:
                None

            Returns:
                None
        """
        # Initialize the object under test before each test
        self.explainer = InterventionCausalExplainer(n_estimators=5)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_train_model(self):
        """Test that the model is trained and stored properly."""
        self.explainer.train_model(self.model, self.X_train, self.y_train)
        self.assertIsNotNone(
            self.explainer.clf, "The classifier should not be None after training."
        )
        self.assertTrue(
            hasattr(self.explainer.clf, "predict"),
            "Trained model should have a predict method.",
        )

    def test_compute_confidence_uncertainty_train(self):
        """Test that confidence and uncertainty are computed on training data."""
        self.explainer.train_model(self.model, self.X_train, self.y_train)
        self.explainer.compute_confidence_uncertainty_train(self.X_train, self.y_train)
        self.assertIsNotNone(
            self.explainer.confidence_train,
            "Confidence for training data should be computed.",
        )
        self.assertIsNotNone(
            self.explainer.aleatoric_uncertainty_train,
            "Aleatoric uncertainty for training data should be computed.",
        )

    def test_compute_confidence_uncertainty_test(self):
        """Test that confidence and uncertainty are computed on test data."""
        self.explainer.train_model(self.model, self.X_train, self.y_train)
        self.explainer.compute_confidence_uncertainty_test(self.X_test, self.y_test)
        self.assertIsNotNone(
            self.explainer.confidence_test,
            "Confidence for test data should be computed.",
        )
        self.assertIsNotNone(
            self.explainer.aleatoric_uncertainty_test,
            "Aleatoric uncertainty for test data should be computed.",
        )

    def test_estimate_feature_impact(self):
        """Test that feature impact is estimated correctly."""
        self.explainer.train_model(self.model, self.X_train, self.y_train)
        self.explainer.compute_confidence_uncertainty_train(self.X_train, self.y_train)
        self.explainer.estimate_feature_impact(self.X_train)
        self.assertIsNotNone(
            self.explainer.feature_effects,
            "Feature effects should be available after estimation.",
        )
        self.assertGreater(
            len(self.explainer.feature_effects),
            0,
            "There should be at least one feature effect estimated.",
        )

    @patch.object(
        InterventionCausalExplainer, "plot_aleatoric_uncertainty", autospec=True
    )
    @patch.object(
        InterventionCausalExplainer, "plot_top_feature_effects", autospec=True
    )
    def test_interpret_runs(self, mock_plot_feature_effects, mock_plot_uncertainty):
        """Test the full interpret pipeline runs end-to-end without error."""
        # We patch plotting methods to avoid rendering figures in a test environment.
        self.explainer.interpret(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )
        # Check that the model was trained
        self.assertIsNotNone(
            self.explainer.clf, "Model was not trained in interpret method."
        )
        # Check that confidence was computed
        self.assertIsNotNone(
            self.explainer.confidence_train,
            "Confidence was not computed on train data.",
        )
        # Check that feature impact was estimated
        self.assertIsNotNone(
            self.explainer.feature_effects, "Feature effects were not estimated."
        )

    def test_perform_intervention_without_estimated_effects(self):
        """Test that performing intervention without estimating feature effects raises an error."""
        with self.assertRaises(ValueError):
            self.explainer.perform_intervention(self.X_test, self.y_test)

    @patch.object(
        InterventionCausalExplainer, "plot_aleatoric_uncertainty", autospec=True
    )
    def test_perform_intervention(self, mock_plot_uncertainty):
        """Test that intervention runs after feature effects are estimated."""
        # Train, compute confidence, estimate feature effects
        self.explainer.train_model(self.model, self.X_train, self.y_train)
        self.explainer.compute_confidence_uncertainty_train(self.X_train, self.y_train)
        self.explainer.estimate_feature_impact(self.X_train)

        # Perform intervention
        self.explainer.perform_intervention(self.X_test.copy(), self.y_test)
        self.assertIsNotNone(
            self.explainer.confidence_test_before_intervention,
            "Confidence test before intervention should be recorded.",
        )
        self.assertIsNotNone(
            self.explainer.aleatoric_uncertainty_test_before_intervention,
            "Aleatoric uncertainty test before intervention should be recorded.",
        )
        self.assertIsNotNone(
            self.explainer.confidence_test,
            "Confidence test after intervention should be computed.",
        )
        self.assertIsNotNone(
            self.explainer.aleatoric_uncertainty_test,
            "Aleatoric uncertainty test after intervention should be computed.",
        )
        mock_plot_uncertainty.assert_called_once()


if __name__ == "__main__":
    unittest.main()
