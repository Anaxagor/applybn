import pytest
from pandas import Index
from unittest.mock import MagicMock

from sklearn.exceptions import NotFittedError
from applybn.core.estimators.base_estimator import BNEstimator
from applybn.core.exceptions.estimator_exc import NodesAutoTypingError
from bamt.log import logger_preprocessor
from bamt.networks import HybridBN

logger_preprocessor.disabled = True


@pytest.fixture
def estimator():
    """Fixture for initializing BNEstimator with mocked parameters."""
    estimator = BNEstimator()
    estimator.use_mixture = True
    estimator.has_logit = False
    return estimator


@pytest.fixture
def mock_estimator(mocker):
    """Fixture to create a BNEstimator with patched dependencies."""
    estimator = BNEstimator()

    # Mock detect_bn to return 'hybrid'
    mocker.patch.object(estimator, "detect_bn", return_value="hybrid")

    # Mock init_bn to return a MagicMock BN object
    mock_bn = MagicMock()
    mocker.patch.object(estimator, "init_bn", return_value=mock_bn)

    return estimator, mock_bn


def generate_case(t="hybrid"):
    """
    Generates a mock data fixture with specified column types.

        This function creates a pytest fixture that returns a MagicMock object
        representing a DataFrame with columns and dtypes defined based on the input 't'.

        Args:
            t: A string indicating the desired case for column type definitions.
               Valid values are "hybrid", "cont", "disc", "empty", and "invalid".
               Defaults to "hybrid".

        Returns:
            A pytest fixture function that returns a MagicMock DataFrame object.
    """
    match t:
        case "hybrid":
            column_types = {"node1": "float64", "node2": "int64", "node3": "object"}
        case "cont":
            column_types = {
                "node1": "float64",
            }
        case "disc":
            column_types = {"node2": "int64", "node3": "object"}
        case "empty":
            column_types = {}
        case "invalid":
            column_types = {"node1": "float64", "node2": "int64", "node3": "blablalba"}

    @pytest.fixture
    def mock_data():
        data = MagicMock()
        data.columns = Index(list(column_types.keys()))

        def mock_getitem(c):
            mock_column = MagicMock()
            mock_column.dtype = MagicMock()
            mock_column.dtype.name = column_types[c]
            return mock_column

        data.__getitem__.side_effect = mock_getitem
        return data

    return mock_data


mock_data_hybrid = generate_case("hybrid")
mock_data_cont = generate_case("cont")
mock_data_disc = generate_case("disc")
mock_data_invalid = generate_case("invalid")
mock_data_empty = generate_case("empty")


def test_detect_bn_hybrid(mock_data_hybrid):
    """
    Tests the detect_bn method with hybrid batch normalization data.

        Args:
            mock_data_hybrid: Mock data representing a hybrid batch normalization scenario.

        Returns:
            None: This function asserts that the result of BNEstimator.detect_bn is "hybrid".
    """
    result = BNEstimator.detect_bn(mock_data_hybrid)
    assert result == "hybrid"


def test_detect_bn_disc(mock_data_disc):
    """
    Tests that detect_bn correctly identifies 'disc' data.

        Args:
            mock_data_disc: The mock discrete data to be tested with.

        Returns:
            None
    """
    result = BNEstimator.detect_bn(mock_data_disc)
    assert result == "disc"


def test_detect_bn_cont(mock_data_cont):
    """
    Tests the detect_bn method with continuous data.

        Args:
            mock_data_cont: Mock continuous data for testing.

        Returns:
            None
            Asserts that the result of BNEstimator.detect_bn is "cont".
    """
    result = BNEstimator.detect_bn(mock_data_cont)
    assert result == "cont"


def test_detect_bn_invalid_node_types(mock_data_invalid):
    """
    Tests that detect_bn raises an error for invalid node types.

        Args:
            mock_data_invalid: Mock data containing nodes with invalid types.

        Returns:
            None
            Raises NodesAutoTypingError if the function does not behave as expected.
    """
    with pytest.raises(NodesAutoTypingError, match="{'node3'}"):
        BNEstimator.detect_bn(mock_data_invalid)


def test_detect_bn_empty(mock_data_empty):
    """
    Tests that detect_bn returns None for empty data.

        Args:
            mock_data_empty: A mock dataset representing empty input data.

        Returns:
            None: This is a test function and does not explicitly return a value,
                  but asserts that the BNEstimator.detect_bn method returns None.
    """
    assert BNEstimator.detect_bn(mock_data_empty) is None


def test_init_bn_hybrid(estimator, mocker):
    """Test init_bn for hybrid network."""
    mock_hybrid_bn = mocker.patch(
        "applybn.core.estimators.base_estimator.HybridBN", return_value=MagicMock()
    )
    bn = estimator.init_bn("hybrid")
    mock_hybrid_bn.assert_called_once_with(use_mixture=True, has_logit=False)

    assert bn is mock_hybrid_bn.return_value


def test_init_bn_cont(estimator, mocker):
    """Test init_bn for continuous network."""
    mock_cont_bn = mocker.patch(
        "applybn.core.estimators.base_estimator.ContinuousBN", return_value=MagicMock()
    )

    bn = estimator.init_bn("cont")

    mock_cont_bn.assert_called_once_with(use_mixture=True)
    assert bn is mock_cont_bn.return_value


def test_init_bn_disc(estimator, mocker):
    """Test init_bn for discrete network."""
    mock_disc_bn = mocker.patch(
        "applybn.core.estimators.base_estimator.DiscreteBN", return_value=MagicMock()
    )

    bn = estimator.init_bn("disc")

    mock_disc_bn.assert_called_once_with()
    assert bn is mock_disc_bn.return_value


def test_init_bn_invalid(estimator):
    """Test init_bn raises TypeError for invalid bn_type."""
    with pytest.raises(TypeError, match="Invalid bn_type, obtained bn_type: unknown"):
        estimator.init_bn("unknown")


def test_fit_initializes_bn(mock_estimator):
    """Test that fit initializes the Bayesian Network correctly."""
    estimator, mock_bn = mock_estimator

    mock_X = MagicMock(name="X")
    descriptor = MagicMock(name="descriptor")
    clean_data = MagicMock(name="clean_data")

    estimator.fit((mock_X, descriptor, clean_data))

    # Ensure detect_bn was called when bn_type is None
    estimator.detect_bn.assert_called_once_with(clean_data)

    # Ensure init_bn was called
    estimator.init_bn.assert_called_once_with("hybrid")

    # Ensure bn_ was set
    assert estimator.bn_ is mock_bn


def test_fit_parameters_case(mock_estimator):
    """Test that fit calls fit_parameters when partial='parameters'."""
    estimator, mock_bn = mock_estimator
    estimator.partial = "parameters"

    # Ensure bn_ is already set with edges
    mock_bn.edges = ["edge1", "edge2"]

    mock_X = MagicMock(name="X")
    descriptor = MagicMock(name="descriptor")
    clean_data = MagicMock(name="clean_data")

    estimator.bn_ = mock_bn  # Manually set to bypass initialization
    estimator.fit((mock_X, descriptor, clean_data))

    # Ensure fit_parameters was called
    mock_bn.fit_parameters.assert_called_once_with(clean_data)


def test_fit_parameters_raises_error(mock_estimator):
    """Test that fit raises NotFittedError if partial='parameters' but the network has no edges."""
    estimator, mock_bn = mock_estimator
    estimator.partial = "parameters"

    mock_bn.edges = []  # Simulating an unfitted BN

    mock_X = MagicMock(name="X")
    descriptor = MagicMock(name="descriptor")
    clean_data = MagicMock(name="clean_data")

    estimator.bn_ = mock_bn  # Manually set to bypass initialization

    with pytest.raises(
        NotFittedError, match="Trying to learn parameters on unfitted estimator"
    ):
        estimator.fit((mock_X, descriptor, clean_data))


def test_fit_structure_case(mock_estimator):
    """Test that fit calls add_nodes and add_edges when partial='structure'."""
    estimator, mock_bn = mock_estimator
    estimator.partial = "structure"

    mock_X = MagicMock(name="X")
    descriptor = MagicMock(name="descriptor")
    clean_data = MagicMock(name="clean_data")

    estimator.fit((mock_X, descriptor, clean_data))

    # Ensure add_nodes and add_edges were called
    mock_bn.add_nodes.assert_called_once_with(descriptor)
    mock_bn.add_edges.assert_called_once_with(mock_X, progress_bar=False)


def test_fit_full_case(mock_estimator):
    """Test that fit calls add_nodes, add_edges, and fit_parameters when partial=False."""
    estimator, mock_bn = mock_estimator
    estimator.partial = False

    mock_X = MagicMock(name="X")
    descriptor = MagicMock(name="descriptor")
    clean_data = MagicMock(name="clean_data")

    estimator.fit((mock_X, descriptor, clean_data))

    # Ensure all methods were called
    mock_bn.add_nodes.assert_called_once_with(descriptor)
    mock_bn.add_edges.assert_called_once_with(mock_X, progress_bar=False)
    mock_bn.fit_parameters.assert_called_once_with(clean_data)


def test_attribute_passing(estimator):
    """
    Tests that necessary attributes are passed to the estimator.

        Args:
            estimator: The estimator object being tested.

        Returns:
            None
    """
    estimator.bn_ = MagicMock(name="bn", spec=HybridBN)

    assert hasattr(estimator, "get_info")
    assert hasattr(estimator, "add_edges")
