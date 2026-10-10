import numpy as np
import pytest
from lwabeamcal.beam import Memo178Beam, solar_azel


def test_operational_coefficients_match_research_reference():
    expected=[[0.08138792788748181,0.547219958789577,1.],
              [0.0888239536039914,0.5504534725181134,1.],
              [0.10249754049440901,0.5490677705437969,1.]]
    np.testing.assert_allclose(Memo178Beam().gain([40.,55.,70.],[0.,90.,180.],[12.,45.,90.]),expected,rtol=1e-12)


def test_zenith_normalized_gain():
    b=Memo178Beam()
    np.testing.assert_allclose(b.gain([40.,55.,70.],[0.,90.,180.],[90.,90.,90.]),1.,atol=1e-12)


def test_below_horizon_not_clamped_and_bad_frequency_rejected():
    assert np.isnan(Memo178Beam().gain([50.],[0.],[-1.])).all()
    with pytest.raises(ValueError): Memo178Beam().gain([500.],[0.],[45.])


def test_solar_geometry_shapes_without_network():
    az,alt=solar_azel([61278.8,61278.9])
    assert az.shape==alt.shape==(2,)
    assert np.isfinite(az).all() and np.isfinite(alt).all()
