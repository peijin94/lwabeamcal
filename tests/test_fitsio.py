from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from astropy.io import fits
import lwabeamcal.fitsio as fitsio
from lwabeamcal.cache import ModelCache
from lwabeamcal.fitsio import calibrate_fits


@pytest.fixture
def inputs(tmp_path):
    f=np.array([40.,50.,70.]); t=np.array([61278.8,61278.8001,61278.8002])
    primary=fits.PrimaryHDU(np.ones((2,1,3,3),dtype=np.float32))
    primary.header['BUNIT']='sfu'
    freq=fits.BinTableHDU.from_columns([fits.Column(name='SFREQ',format='D',unit='GHz',array=f/1000)])
    time=fits.BinTableHDU.from_columns([fits.Column(name='mjd',format='D',array=np.floor(t)),
            fits.Column(name='time',format='D',unit='ms',array=(t-np.floor(t))*86400000)])
    p=tmp_path/'raw.fits'; fits.HDUList([primary,freq,time]).writeto(p)
    cache=ModelCache(f,t,np.ones((3,3))*1000,gain=np.ones((3,3))*.5,
             metadata={'gain_model':'memo178','gain_normalization':'zenith-1',
                       'off_windows_mjd':[[t[1],t[1]+1e-8]]})
    return p,cache,f,t


def test_fits_corrects_i_converts_v_preserves_tables_provenance(inputs,tmp_path):
    p,cache,f,t=inputs; before=p.read_bytes(); out=tmp_path/'cal.fits'
    report=calibrate_fits(p,out,cache=cache,chunk_size=1)
    with fits.open(out) as h, fits.open(p) as raw:
        assert h[0].header['BUNIT']=='jy' and h[0].header['LWACAL']
        np.testing.assert_allclose(h[0].data[1,0],10000)
        assert np.isnan(h[0].data[0,0,:,1]).all()
        expected=(10000-1100-288.95515757076816)/.5
        assert h[0].data[0,0,1,0]==pytest.approx(expected)
        assert h[1].data.tobytes()==raw[1].data.tobytes()
        assert h[2].data.tobytes()==raw[2].data.tobytes()
        assert all(name in h for name in ('TD_MASK','TD_BKG','TD_GAIN','LWA_CAUTION','LWA_INPUT_MASK','LWA_PROVENANCE'))
        prov=json.loads(h['LWA_PROVENANCE'].data['JSON'][0])
        assert prov['model']['parameters']['k']==1.1
        assert h[0].header['LWACSHA']==prov['cache_effective_sha256']
        assert len(h[0].header.cards['LWACSHA'].image)==80
        assert prov['stokes_scope']=='I corrected; other planes unit-converted only'
    assert p.read_bytes()==before
    assert report['valid_samples']==6
    with pytest.raises(FileExistsError): calibrate_fits(p,out,cache=cache)
    with pytest.raises(ValueError): calibrate_fits(p,p,cache=cache,overwrite=True)
    with pytest.raises(ValueError): calibrate_fits(out,tmp_path/'double.fits',cache=cache)


def test_input_units_override_and_missing_unit_rejected(inputs,tmp_path):
    p,cache,f,t=inputs
    with fits.open(p,mode='update') as h: del h[0].header['BUNIT']
    with pytest.raises(ValueError): calibrate_fits(p,tmp_path/'bad.fits',cache=cache)
    calibrate_fits(p,tmp_path/'ok.fits',cache=cache,input_unit='sfu')


def test_external_mask_preserved_and_wrong_axes_rejected(inputs,tmp_path):
    p,cache,f,t=inputs; maskfile=tmp_path/'mask.fits'
    with fits.open(p) as h:
        a=fits.HDUList([x.copy() for x in h]); m=np.zeros((3,3),dtype='int32');m[0,0]=512
        a.append(fits.ImageHDU(m,name='TD_MASK'));a.writeto(maskfile)
    out=tmp_path/'cal.fits'; calibrate_fits(p,out,cache=cache,mask_fits=maskfile)
    with fits.open(out) as h:
        assert h['LWA_INPUT_MASK'].data[0,0]==512
        assert np.isnan(h[0].data[0,0,0,0])
    with fits.open(maskfile,mode='update') as h: h[1].data['SFREQ'][0]+=.001
    with pytest.raises(ValueError): calibrate_fits(p,tmp_path/'wrong.fits',cache=cache,mask_fits=maskfile)


def test_gain_model_mismatch_is_loud(inputs,tmp_path):
    p,cache,f,t=inputs
    with pytest.raises(ValueError): calibrate_fits(p,tmp_path/'wrong.fits',cache=cache,beam='lwapb')


def test_atomic_failure_no_incomplete_output_or_temp_files(inputs,tmp_path):
    p,cache,f,t=inputs
    cache.metadata['off_windows_mjd']=[['not a time',1]]
    with pytest.raises((ValueError,TypeError)): calibrate_fits(p,tmp_path/'bad.fits',cache=cache)
    assert not (tmp_path/'bad.fits').exists()
    assert not list(tmp_path.glob('.lwabeamcal-*'))


@pytest.mark.parametrize('alias',['same_path','hardlink','relative_load'])
def test_loaded_object_cache_cannot_be_output(inputs,tmp_path,monkeypatch,alias):
    p,cache,f,t=inputs
    cache_path=tmp_path/'cache.npz'; cache.save(cache_path)
    if alias=='relative_load':
        monkeypatch.chdir(tmp_path)
        loaded=ModelCache.load('cache.npz')
        monkeypatch.chdir(tmp_path.parent)
    else: loaded=ModelCache.load(cache_path)
    target=cache_path
    if alias=='hardlink':
        target=tmp_path/'cache-alias.npz'; target.hardlink_to(cache_path)
    before=cache_path.read_bytes()
    with pytest.raises(ValueError,match='alias cache'):
        calibrate_fits(p,target,cache=loaded,overwrite=True)
    assert cache_path.read_bytes()==target.read_bytes()==before


@pytest.mark.parametrize('alias',['same_path','hardlink'])
def test_imported_audit_cache_cannot_be_output(inputs,tmp_path,alias):
    p,cache,f,t=inputs
    audit_path=tmp_path/'audit.npz'
    np.savez(audit_path,model_kind='FAST',flux_unit='Jy',model_freqs_mhz=f,
             model_mjd=t,model_jy=cache.fast_jy.T,
             gain_freqs_mhz=f,gain_mjd=t,gain_grid=cache.gain,gain_beam='memo178')
    imported=ModelCache.from_td_audit(audit_path)
    target=audit_path
    if alias=='hardlink':
        target=tmp_path/'audit-alias.npz'; target.hardlink_to(audit_path)
    before=audit_path.read_bytes()
    with pytest.raises(ValueError,match='alias cache'):
        calibrate_fits(p,target,cache=imported,overwrite=True)
    assert audit_path.read_bytes()==target.read_bytes()==before


def test_calibrated_cache_mutation_rejected_before_fits_use(inputs,tmp_path):
    p,cache,f,t=inputs
    cache.fast_jy *= 1.1
    cache.metadata['model_state']='calibrated'
    target=tmp_path/'invalid.fits'
    with pytest.raises(ValueError,match='model_state'):
        calibrate_fits(p,target,cache=cache)
    assert not target.exists()


@pytest.mark.parametrize('mutation',['arrays','metadata'])
def test_provenance_fingerprints_effective_cache_and_original_file(inputs,tmp_path,mutation):
    p,cache,f,t=inputs
    path=tmp_path/'cache.npz'; cache.save(path)
    cache=ModelCache.load(path)
    original_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    first=tmp_path/'first.fits'; calibrate_fits(p,first,cache=cache)
    with fits.open(first) as h:
        before=json.loads(h['LWA_PROVENANCE'].data['JSON'][0])
    if mutation=='arrays': cache.fast_jy += 100
    else: cache.metadata['origin']='modified in memory'
    second=tmp_path/'second.fits'; calibrate_fits(p,second,cache=cache)
    with fits.open(second) as h:
        after=json.loads(h['LWA_PROVENANCE'].data['JSON'][0])
        assert h[0].header['LWACSHA']==after['cache_effective_sha256']
        if mutation=='arrays':
            np.testing.assert_allclose(h['TD_BKG'].data[1,0],1.1*1100+288.95515757076816)
    assert before['cache_effective_sha256']!=after['cache_effective_sha256']
    assert before['cache_file_sha256']==after['cache_file_sha256']==original_sha
    assert after['cache_sha256']==after['cache_effective_sha256']
    assert fitsio._cache_fingerprint(cache)==after['cache_effective_sha256']
    assert cache.sha256==original_sha
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original_sha


@pytest.mark.parametrize('location',['embedded','external','both'])
def test_unsigned_uint32_masks_preserve_all_bits(inputs,tmp_path,location):
    p,cache,f,t=inputs
    embedded=np.zeros((3,3),dtype=np.uint32)
    embedded[0,0]=2**31; embedded[2,2]=2**32-1
    external=np.zeros((3,3),dtype=np.uint32)
    external[0,0]=1; external[1,0]=2**31+512
    maskfile=None
    if location in ('external','both'):
        maskfile=tmp_path/'mask.fits'
        with fits.open(p) as h, fits.HDUList([x.copy() for x in h]) as output:
            output.append(fits.ImageHDU(external,name='TD_MASK'))
            output.writeto(maskfile)
    if location in ('embedded','both'):
        with fits.open(p,mode='update',memmap=False) as h:
            h.append(fits.ImageHDU(embedded,name='TD_MASK'))
    expected=np.zeros((3,3),dtype=np.uint32)
    if location in ('embedded','both'): expected |= embedded
    if location in ('external','both'): expected |= external
    before=p.read_bytes()
    target=tmp_path/'cal.fits'
    calibrate_fits(p,target,cache=cache,mask_fits=maskfile,chunk_size=1)
    with fits.open(target) as h:
        np.testing.assert_array_equal(h['LWA_INPUT_MASK'].data,expected.astype(np.int64))
        assert np.isnan(h[0].data[0,0][expected!=0]).all()
    assert p.read_bytes()==before


@pytest.mark.parametrize('crval,crpix,cdelt,accepted',[
    (1,2,-1,False),(2,2,1,True),(1,1,1,True)])
def test_stokes_first_plane_uses_reference_pixel(inputs,tmp_path,crval,crpix,cdelt,accepted):
    p,cache,f,t=inputs
    with fits.open(p,mode='update') as h:
        h[0].header['CTYPE4']='STOKES'
        h[0].header['CRVAL4']=crval; h[0].header['CRPIX4']=crpix; h[0].header['CDELT4']=cdelt
    target=tmp_path/'cal.fits'
    if accepted:
        calibrate_fits(p,target,cache=cache)
        assert target.exists()
    else:
        with pytest.raises(ValueError,match='first Stokes plane must be I'):
            calibrate_fits(p,target,cache=cache)
        assert not target.exists()


@pytest.mark.parametrize('existing_target',[False,True])
def test_failure_during_write_preserves_target_and_closes_resources(inputs,tmp_path,monkeypatch,existing_target):
    p,cache,f,t=inputs
    target=tmp_path/'cal.fits'; before=p.read_bytes()
    if existing_target: target.write_bytes(b'existing target must survive')
    initial_paths=set(tmp_path.iterdir())
    mappings=[]; opened=[]; closed=[]; attempted=[]
    real_memmap=np.memmap; real_open=fits.open; real_close=fits.HDUList.close
    def track_memmap(*args,**kwargs):
        array=real_memmap(*args,**kwargs); mappings.append(array); return array
    def track_open(*args,**kwargs):
        hdus=real_open(*args,**kwargs); opened.append(hdus); return hdus
    def track_close(hdus,*args,**kwargs):
        closed.append(hdus); return real_close(hdus,*args,**kwargs)
    def fail_write(hdus,path,*args,**kwargs):
        attempted.append(hdus)
        Path(path).write_bytes(b'partial FITS output')
        raise OSError('simulated failure during writing')
    monkeypatch.setattr(fitsio.np,'memmap',track_memmap)
    monkeypatch.setattr(fits,'open',track_open)
    monkeypatch.setattr(fits.HDUList,'close',track_close)
    monkeypatch.setattr(fits.HDUList,'writeto',fail_write)
    with pytest.raises(OSError,match='during writing'):
        calibrate_fits(p,target,cache=cache,overwrite=existing_target)
    assert len(attempted)==1 and mappings and opened
    assert all(any(hdus is item for item in closed) for hdus in opened+attempted)
    assert all(array._mmap.closed for array in mappings)
    assert p.read_bytes()==before
    if existing_target: assert target.read_bytes()==b'existing target must survive'
    else: assert not target.exists()
    assert set(tmp_path.iterdir())==initial_paths
