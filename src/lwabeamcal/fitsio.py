"""Read-only spec_v2 adapter and atomic FITS correction products.

The FITS adapter is I-only for background/PB correction. Other Stokes
planes receive unit conversion only. Input spectral/UT tables are copied.
"""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from astropy.io import fits
from .beam import Memo178Beam, solar_azel
from .cache import ModelCache
from .core import FrozenBackground, calibrate, unit_scale, Mask, Caution
from .storage import atomic_target, sha256_file


def read_axes(hdus):
    """spec_v2 SFREQ: GHz by convention; UT time is milliseconds in its MJD day."""
    freq=np.asarray(hdus[1].data['SFREQ'],float).reshape(-1)
    declared=hdus[1].columns['SFREQ'].unit
    units={None:1000.,'ghz':1000.,'mhz':1.,'khz':.001,'hz':1e-6}
    key=None if declared is None else declared.strip().lower()
    if key not in units: raise ValueError('unsupported SFREQ unit')
    f=freq*units[key]
    time_unit=hdus[2].columns['time'].unit
    if time_unit is not None and time_unit.strip().lower() not in ('ms','millisecond','milliseconds'):
        raise ValueError('spec_v2 time column must be milliseconds')
    t=np.asarray(hdus[2].data['mjd'],float).reshape(-1)+np.asarray(hdus[2].data['time'],float).reshape(-1)/86400000.
    if f.ndim!=1 or not len(f) or not np.isfinite(f).all() or np.any(f<=0) or np.any(np.diff(f)<=0):
        raise ValueError('positive increasing frequencies required')
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite increasing native times required')
    return f,t


def _already_corrected(header):
    for key in ('LWACAL','PBCOR','TDBCOR'):
        value=header.get(key,False)
        if isinstance(value,str): value=value.strip().lower() not in ('','false','f','no','0','none')
        if value: return True
    return False


def _cache_fingerprint(cache):
    cache.validate()
    h=hashlib.sha256(json.dumps(cache.metadata,sort_keys=True,allow_nan=False).encode())
    for key in ('frequencies_mhz','times_mjd','fast_jy','gain','gain_frequencies_mhz','gain_times_mjd'):
        value=getattr(cache,key)
        h.update(key.encode())
        if value is not None:
            array=np.asarray(value,dtype='<f8')
            h.update(json.dumps(array.shape).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def _close_hdus(hdus):
    # Astropy retains mappings while local data arrays still reference them.
    mapping=getattr(getattr(hdus,'_file',None),'_mmap',None)
    try: hdus.close()
    finally:
        if mapping is not None and not mapping.closed: mapping.close()


def _open_fits(stack,path,*,memmap=True):
    hdus=fits.open(path,memmap=memmap)
    stack.callback(_close_hdus,hdus)
    return hdus


def _read_mask(stack,path,hdus):
    mask=hdus['TD_MASK']
    if any(key in mask.header for key in ('BZERO','BSCALE','BLANK')):
        # Decode scaled/unsigned masks with Astropy, keeping the primary mapped.
        mask=_open_fits(stack,path,memmap=False)['TD_MASK']
    return mask.data


def _close_memmap(array):
    try: array.flush()
    finally: array._mmap.close()


def calibrate_fits(input_path,output_path,*,cache,beam='memo178',input_unit=None,
                   output_unit='jy',model=None,experimental=False,mask_fits=None,
                   off_windows_mjd=None,terrain_min_alt_deg=12.,min_gain=None,
                   max_time_gap_seconds=None,chunk_size=4096,overwrite=False,
                   use_cached_gain=True):
    """Calibrate spec_v2 FITS while preserving original files and table payloads.

    `cache` is a ModelCache or portable cache path. By default cached gains
    must match `beam`; without gains Memo178 is computed at native times.
    lwapb is supported via supplied zenith-1 gain grids, not bundled model
    coefficients. Off-point exclusions come from cache metadata or explicit
    MJD intervals, never an assumed daily UT window.
    """
    source=Path(input_path); target=Path(output_path)
    if target.is_symlink(): raise ValueError('refusing symlink output')
    if source.resolve()==target.resolve() or (target.exists() and os.path.samefile(source,target)):
        raise ValueError('output must not alias input')
    if target.exists() and not overwrite: raise FileExistsError(target)
    if mask_fits and (Path(mask_fits).resolve()==target.resolve() or (target.exists() and os.path.samefile(mask_fits,target))):
        raise ValueError('output must not alias mask source')
    if not isinstance(chunk_size,int) or isinstance(chunk_size,bool) or chunk_size<=0:
        raise ValueError('chunk_size must be a positive integer')
    if beam not in ('memo178','lwapb'): raise ValueError('beam must be memo178 or lwapb')
    cache_path=None
    if not isinstance(cache,ModelCache):
        cache_path=Path(cache)
        if target.resolve()==cache_path.resolve() or (target.exists() and os.path.samefile(cache_path,target)):
            raise ValueError('output must not alias cache')
        cache=ModelCache.load(cache_path)
    if cache.source_path is not None:
        cache_path=cache.source_path
        if target.resolve()==cache_path.resolve() or (cache_path.exists() and target.exists() and os.path.samefile(cache_path,target)):
            raise ValueError('output must not alias cache')
    cache.validate()
    if cache.gain is not None and use_cached_gain:
        if cache.metadata['gain_model']!=beam: raise ValueError('requested beam disagrees with cached gain_model')
    elif beam!='memo178': raise ValueError('lwapb requires a supplied matching normalized gain grid')
    out_scale=unit_scale(output_unit)
    model=model or FrozenBackground()
    windows=cache.metadata.get('off_windows_mjd',[]) if off_windows_mjd is None else off_windows_mjd
    checked_windows=[]
    for a,b in windows:
        a=float(a);b=float(b)
        if not np.isfinite([a,b]).all() or a>=b: raise ValueError('finite ordered off windows required')
        checked_windows.append([a,b])
    target.parent.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        hdus=_open_fits(stack,source)
        if _already_corrected(hdus[0].header) or 'TD_BKG' in hdus or 'TD_GAIN' in hdus:
            raise ValueError('refusing to calibrate an already corrected FITS')
        unit=input_unit if input_unit is not None else hdus[0].header.get('BUNIT')
        in_scale=unit_scale(unit)
        f,t=read_axes(hdus)
        data=hdus[0].data
        if data.ndim!=4 or data.shape[1]!=1 or data.shape[2:]!=(len(f),len(t)):
            raise ValueError('spec_v2 primary shape must be (nStokes,1,nfreq,ntime)')
        if 'CRVAL4' in hdus[0].header and hdus[0].header.get('CTYPE4','').strip().upper()=='STOKES':
            first_stokes=hdus[0].header['CRVAL4']+(1-hdus[0].header.get('CRPIX4',1))*hdus[0].header.get('CDELT4',1)
            if first_stokes!=1: raise ValueError('first Stokes plane must be I')
        masks=[]
        if 'TD_MASK' in hdus: masks.append(_read_mask(stack,source,hdus))
        if mask_fits:
            mh=_open_fits(stack,mask_fits)
            mf,mt=read_axes(mh)
            if not np.array_equal(mf,f) or not np.array_equal(mt,t): raise ValueError('mask axes do not exactly match input')
            if 'TD_MASK' not in mh: raise ValueError('mask FITS requires TD_MASK extension')
            masks.append(_read_mask(stack,mask_fits,mh))
        for m in masks:
            if m.shape!=(len(f),len(t)) or m.dtype.kind not in 'biu' or np.any(m<0) or np.any(m>2**32-1):
                raise ValueError('source mask must be a nonnegative integer (nfreq,ntime) bitmask')
        effective_sha256=_cache_fingerprint(cache)
        provenance=dict(package='lwabeamcal',version='0.1.0',model=model.metadata(),
                        input_unit=str(unit).strip().lower(),output_unit=output_unit.strip().lower(),
                        input_sha256=sha256_file(source),cache_sha256=effective_sha256,
                        cache_effective_sha256=effective_sha256,cache_file_sha256=cache.sha256,
                        cache_provenance=cache.metadata,beam=beam,gain_normalization='zenith-1',
                        gain_source='cache' if cache.gain is not None and use_cached_gain else 'direct Memo178',
                        off_windows_mjd=checked_windows,source_path=str(source.resolve()),
                        mask_source_sha256=sha256_file(mask_fits) if mask_fits else None,
                        stokes_scope='I corrected; other planes unit-converted only',
                        terrain_min_alt_deg=terrain_min_alt_deg,experimental=bool(experimental),
                        min_gain=min_gain,max_time_gap_seconds=max_time_gap_seconds,
                        bit_names={int(v):v.name for v in Mask},caution_bits={int(v):v.name for v in Caution},
                        original_mask_namespace='provider-specific; unchanged in LWA_INPUT_MASK')
        report=dict(output=str(target.resolve()),total_samples=len(f)*len(t),valid_samples=0,
                    nonpositive_valid_samples=0,mask_counts={v.name:0 for v in Mask})
        work=stack.enter_context(tempfile.TemporaryDirectory(prefix='.lwabeamcal-',dir=target.parent))
        def allocate(name,dtype,shape):
            a=np.memmap(Path(work)/name,mode='w+',dtype=dtype,shape=shape)
            stack.callback(_close_memmap,a)
            return a
        result_data=allocate('primary','>f8',data.shape)
        bg=allocate('background','>f8',(len(f),len(t)))
        gains=allocate('gain','>f8',(len(f),len(t)))
        flags=allocate('mask','>i4',(len(f),len(t)))
        cautions=allocate('caution','>i4',(len(f),len(t)))
        original=allocate('original','>i8',(len(f),len(t)))
        for start in range(0,len(t),chunk_size):
            end=min(start+chunk_size,len(t)); sl=slice(start,end)
            az,alt=solar_azel(t[sl])
            fast=cache.fast_at(f,t[sl],max_time_gap_seconds=max_time_gap_seconds)
            gain=cache.gain_at(f,t[sl],max_time_gap_seconds=max_time_gap_seconds) if cache.gain is not None and use_cached_gain else Memo178Beam().gain(f,az,alt)
            mask=np.zeros((len(f),end-start),np.uint32)
            for m in masks: mask |= m[:,sl].astype(np.uint32)
            off=np.zeros(end-start,bool)
            for a,b in checked_windows: off |= (t[sl]>=a)&(t[sl]<=b)
            r=calibrate(data[0,0,:,sl],f,fast_jy=fast,gain=gain,input_unit=unit,model=model,
                        sun_alt_deg=alt,input_mask=mask,off_mask=off,experimental=experimental,
                        terrain_min_alt_deg=terrain_min_alt_deg,min_gain=min_gain)
            result_data[:,:,:,sl]=data[:,:,:,sl].astype(float)*(in_scale/out_scale)
            result_data[0,0,:,sl]=r.corrected_jy/out_scale
            bg[:,sl]=r.background_jy;gains[:,sl]=r.gain;flags[:,sl]=r.mask
            cautions[:,sl]=r.caution;original[:,sl]=r.input_mask
            report['valid_samples']+=int(r.valid.sum())
            report['nonpositive_valid_samples']+=int((r.valid&(r.corrected_jy<=0)).sum())
            for v in Mask: report['mask_counts'][v.name]+=int(((r.mask&int(v))!=0).sum())
        header=hdus[0].header.copy()
        for card in ('CHECKSUM','DATASUM'): header.remove(card,ignore_missing=True)
        header['BUNIT']=output_unit.strip().lower();header['LWACAL']=True;header['PBCOR']=True;header['TDBCOR']=True
        header['HIERARCH LWA CAL VERSION']='0.1.0';header['HIERARCH LWA BEAM']=beam
        header['HIERARCH LWA FLAT K']=model.k;header['HIERARCH LWA N50 JY']=model.n50_jy
        header['HIERARCH LWA N EXPONENT']=model.exponent;header['HIERARCH LWA N REF MHZ']=model.reference_mhz
        # A short FITS keyword holds the full hash in one 80-byte card, also on Astropy 6.
        header['LWACSHA']=provenance['cache_sha256']
        header['HIERARCH LWA STOKES SCOPE']='I only; other planes unit-converted'
        header.add_history('lwabeamcal: I=(raw_Jy-K*FAST-n(nu))/g; signed residuals preserved')
        reserved={'TD_MASK','LWA_CAUTION','LWA_INPUT_MASK','LWA_PROVENANCE'}
        output=fits.HDUList([fits.PrimaryHDU(result_data,header=header)]+[h.copy() for h in hdus[1:] if h.name not in reserved])
        stack.callback(_close_hdus,output)
        for name,a,unitname in [('TD_MASK',flags,None),('TD_BKG',bg,'jy'),('TD_GAIN',gains,'1'),('LWA_CAUTION',cautions,None),('LWA_INPUT_MASK',original,None)]:
            image=fits.ImageHDU(a,name=name)
            if unitname: image.header['BUNIT']=unitname
            output.append(image)
        text=json.dumps(provenance,sort_keys=True,allow_nan=False)
        output.append(fits.BinTableHDU.from_columns([fits.Column(name='JSON',format=f'{len(text.encode("ascii"))}A',array=[text])],name='LWA_PROVENANCE'))
        with atomic_target(target,overwrite=overwrite) as temporary:
            output.writeto(temporary,overwrite=True,checksum=True)
    return report
