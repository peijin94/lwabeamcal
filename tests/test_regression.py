"""Actual held-out 2026-08-26 observations; fixture has no synthetic flux."""
from pathlib import Path
import numpy as np
import pytest
from lwabeamcal import calibrate


@pytest.mark.parametrize('beam',['memo178','lwapb'])
def test_matches_approved_frozen_model_real_data(beam):
    with np.load(Path(__file__).parent/'data/real_20260826.npz',allow_pickle=False) as z:
        result=calibrate(z['raw_jy'],z['frequencies_mhz'],input_unit='jy',
                         fast_jy=z['fast_jy'],gain=z['gain_'+beam],input_mask=z['mask_'+beam])
        np.testing.assert_allclose(result.corrected_jy,z['expected_'+beam],rtol=2e-9,atol=2e-5,equal_nan=True)
