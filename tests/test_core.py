import numpy as np
import pytest
from lwabeamcal.core import calibrate, FrozenBackground, Mask


def test_subtract_background_before_dividing_gain():
    result = calibrate([[1., .5]], [50.], fast_jy=[[1000., 1000.]],
                       gain=[[.5, .25]], input_unit="sfu")
    expected = (np.array([[10000., 5000.]]) - 1.1*1000 - 288.95515757076816)/[[.5,.25]]
    np.testing.assert_allclose(result.corrected_jy, expected)
    assert result.valid.all()
    assert FrozenBackground().floor_jy([50.])[0] == 288.95515757076816


def test_masked_signed_values_and_terrain_without_clipping():
    result = calibrate(np.zeros((1, 6)), [50.], fast_jy=np.ones((1, 6))*100,
                       gain=[[1, 0, -1, np.nan, 1, 1]], input_unit='jy',
                       sun_alt_deg=[25, 25, 25, 25, 11, 15], input_mask=[[0,0,0,0,0,512]])
    assert result.corrected_jy[0,0] < 0
    assert np.isnan(result.corrected_jy[0,1:]).all()
    assert result.mask[0,1] & Mask.INVALID_GAIN
    assert result.mask[0,4] & Mask.TERRAIN
    assert result.mask[0,5] & Mask.USER
    assert result.input_mask[0,5] == 512
    assert result.caution[0,5] != 0


def test_frequency_policy_and_experimental_opt_in():
    x = np.ones((3,1))*5000
    base = calibrate(x, [30,50,84], fast_jy=x/5, gain=1, input_unit='jy')
    assert base.mask[0,0] & Mask.UNSUPPORTED_FREQUENCY
    assert base.valid[1:,0].all()
    experimental = calibrate(x, [30,50,84], fast_jy=x/5, gain=1, input_unit='jy', experimental=True)
    assert experimental.valid.all()
    assert experimental.caution[0,0] and experimental.caution[2,0]


@pytest.mark.parametrize('unit', ['kelvin', '', None])
def test_units_must_be_explicit_supported(unit):
    with pytest.raises(ValueError):
        calibrate([[1]], [50], fast_jy=[[1]], gain=1, input_unit=unit)


@pytest.mark.parametrize('freq', [[0], [np.nan], [np.inf], [-50]])
def test_floor_rejects_bad_frequencies(freq):
    with pytest.raises(ValueError): FrozenBackground().floor_jy(freq)


def test_sfu_jy_equivalence_and_no_input_mutation():
    x=np.array([[1., 0.]]); saved=x.copy()
    a=calibrate(x, [50], fast_jy=100, gain=.5, input_unit='sfu')
    b=calibrate(x*1e4, [50], fast_jy=100, gain=.5, input_unit='jy')
    np.testing.assert_array_equal(a.corrected_jy,b.corrected_jy)
    np.testing.assert_array_equal(x,saved)


def test_bad_shape_and_nonfinite_background():
    with pytest.raises(ValueError):
        calibrate([1,2], [50], fast_jy=1, gain=1, input_unit='jy')
    r=calibrate([[1]], [50], fast_jy=np.nan, gain=1, input_unit='jy')
    assert r.mask[0,0] & Mask.UNSUPPORTED_BACKGROUND
    assert np.isnan(r.corrected_jy).all()


def test_negative_fast_is_not_a_physical_background():
    r=calibrate([[100]], [50], fast_jy=[[-1000]], gain=1, input_unit='jy')
    assert r.mask[0,0] & Mask.UNSUPPORTED_BACKGROUND
    assert np.isnan(r.corrected_jy[0,0])


def test_custom_model_does_not_claim_default_training():
    meta=FrozenBackground(k=1.2).metadata()
    assert meta['model_id']=='custom'
    assert meta['training_days']==[] and meta['held_out_days']==[]


def test_off_pointing_masks_are_per_day_explicit():
    r=calibrate([[1,1]], [50], fast_jy=1, gain=1, input_unit='jy', off_mask=[False,True])
    assert r.mask[0,1] & Mask.OFF_POINTING
    assert r.valid[0,0]


@pytest.mark.parametrize('params', [{'k':0}, {'n50_jy':-1}, {'exponent':np.nan}, {'reference_mhz':0}])
def test_model_parameters_validation(params):
    with pytest.raises(ValueError): FrozenBackground(**params)
