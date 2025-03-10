import pytest
from applybn.synthetics.bn_synthetic_generator import BNSyntheticGenerator
import pandas as pd

def test_data_generator_initialization():
    dg = BNSyntheticGenerator()
    assert dg.bn is None

def test_data_generator_fit(imbalanced_data):
    dg = BNSyntheticGenerator()
    dg.fit(imbalanced_data)
    
    assert dg.bn is not None
    assert hasattr(dg.bn, 'sample')
    assert len(dg.bn.nodes) == 4  # 3 features + 1 class
