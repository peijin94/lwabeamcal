import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from lwabeamcal.cache import ModelCache, interpolate_grid


def test_bilinear_exact_interpolation_and_no_endpoint_clamping():
    v=np.array([[0.,2.],[10.,12.]])
    out=interpolate_grid([40,60], [60000,60002], v, [30,40,50,60,70], [60000,60001,60002])
    np.testing.assert_allclose(out[1:4],[[0,1,2],[5,6,7],[10,11,12]])
    assert np.isnan(out[[0,4]]).all()
    assert np.isnan(interpolate_grid([40,60],[60000,60002],v,[50],[59999,60003])).all()


def test_exact_good_node_survives_nan_neighbor_but_hole_not_bridged():
    v=np.array([[1.,np.nan,3.],[4.,5.,6.]])
    out=interpolate_grid([40,60],[0.,1.,2.],v,[40],[0.,.5,1.,2.])
    assert out[0,0]==1 and out[0,3]==3
    assert np.isnan(out[0,1:3]).all()


def test_large_time_gap_can_be_declared_unsupported():
    out=interpolate_grid([50],[60000,60001],[[1,3]],[50],[60000,60000.5,60001],max_time_gap_seconds=300)
    np.testing.assert_allclose(out,[[1,np.nan,3]],equal_nan=True)


def test_portable_cache_round_trip_hash_and_no_clobber(tmp_path):
    c=ModelCache([40,60],[60000,60001],[[100,200],[300,400]],metadata={'origin':'test fixture'})
    p=tmp_path/'model.npz'; c.save(p)
    loaded=ModelCache.load(p)
    assert loaded.metadata['origin']=='test fixture'
    assert len(loaded.sha256)==64
    np.testing.assert_allclose(loaded.fast_at([50],[60000.5]),[[250]])
    with pytest.raises(FileExistsError): c.save(p)


@pytest.mark.parametrize('meta',[{'model_kind':'MASTER'},{'model_state':'calibrated'},{'flux_unit':'sfu'}, {'frequency_unit':'Hz'}, {'schema':999}])
def test_rejects_wrong_model_units_or_schema(meta):
    with pytest.raises(ValueError): ModelCache([50],[60000],[[1]],metadata=meta)


def test_duplicate_axes_and_gain_normalization_rejected():
    with pytest.raises(ValueError): ModelCache([50],[60000,60000],[[1,1]])
    with pytest.raises(ValueError): ModelCache([50],[60000],[[1]],gain=[[1]],metadata={'gain_normalization':'raw'})


def test_legacy_audit_import_records_raw_fast_and_real_off_window(tmp_path):
    p=tmp_path/'audit.npz'
    np.savez(p,model_kind='FAST',flux_unit='Jy',model_freqs_mhz=[40.,60.],
             model_mjd=[60000.,60000.,60001.],model_jy=[[1.,2.],[1.,2.],[3.,4.]],
             gain_freqs_mhz=[40.,60.],gain_mjd=[60000.,60001.],gain_grid=[[.5,.6],[.7,.8]],
             gain_beam='memo178',day='20260826',off_window_frame='ut',off_window=[20.3583333333,20.4083333333])
    c=ModelCache.from_td_audit(p)
    np.testing.assert_array_equal(c.times_mjd,[60000,60001])
    assert c.metadata['model_state']=='raw'
    assert c.metadata['gain_model']=='memo178'
    assert len(c.metadata['off_windows_mjd'])==1
    assert c.source_path==p.resolve()
    assert c.sha256==c.metadata['source_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert c.gain_at([50],[60000.5])[0,0]==pytest.approx(.65)
    before=p.read_bytes()
    with pytest.raises(ValueError): c.save(p,overwrite=True)
    assert p.read_bytes()==before


@pytest.mark.parametrize('source',['constructor','npz'])
def test_gain_requires_explicit_normalization(source,tmp_path):
    metadata=dict(schema=1,model_kind='FAST',model_state='raw',flux_unit='Jy',
                  frequency_unit='MHz',time_unit='MJD UTC',gain_model='memo178')
    if source=='constructor':
        with pytest.raises(ValueError,match='zenith-1'):
            ModelCache([50],[60000],[[1]],gain=[[.5]],metadata=metadata)
    else:
        path=tmp_path/'undeclared.npz'
        np.savez(path,frequencies_mhz=[50],times_mjd=[60000],fast_jy=[[1]],
                 gain=[[.5]],metadata_json=json.dumps(metadata))
        with pytest.raises(ValueError,match='zenith-1'): ModelCache.load(path)


@pytest.mark.parametrize('operation',['fast_at','gain_at','save'])
@pytest.mark.parametrize('field,value',[
    ('model_kind','MASTER'),('model_state','calibrated'),('flux_unit','sfu'),
    ('frequency_unit','Hz'),('time_unit','TAI'),('schema',999),
    ('model_state',None),('gain_normalization',None),('gain_model','unknown'),
    ('frequencies_mhz',[0.,60.]),('times_mjd',[60000.,60000.]),
    ('fast_jy',[[1.]]),('gain_frequencies_mhz',[40.,np.nan]),
    ('gain_times_mjd',[60001.,60000.]),('gain',[[.5]])])
def test_mutated_cache_revalidated_before_use_or_save(field,value,operation,tmp_path):
    cache=ModelCache([40,60],[60000,60001],[[1,2],[3,4]],gain=[[.5,.5],[.5,.5]],
                     metadata={'gain_model':'memo178','gain_normalization':'zenith-1'})
    if field in cache.metadata:
        if value is None: del cache.metadata[field]
        else: cache.metadata[field]=value
    else: setattr(cache,field,np.array(value))
    output=tmp_path/'invalid.npz'
    with pytest.raises(ValueError):
        if operation=='save': cache.save(output)
        else: getattr(cache,operation)([50],[60000.5])
    assert not output.exists()


def test_loaded_cache_retains_absolute_source_path(tmp_path,monkeypatch):
    path=tmp_path/'cache.npz'
    ModelCache([50],[60000],[[1]]).save(path)
    monkeypatch.chdir(tmp_path)
    cache=ModelCache.load('cache.npz')
    assert Path(cache.source_path)==path.resolve()
