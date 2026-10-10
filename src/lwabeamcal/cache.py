"""Portable raw FAST caches, optional gains, and bounded gap-aware interpolation."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from .storage import sha256_file, atomic_target

_CONTRACT = dict(schema=1, model_kind='FAST', model_state='raw', flux_unit='Jy',
                 frequency_unit='MHz', time_unit='MJD UTC')


def _axis(x, name):
    x=np.array(x,dtype=float,copy=True)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all() or np.any(np.diff(x)<=0):
        raise ValueError(f'{name} must be nonempty finite strictly increasing 1D')
    return x


def _interp_axis(axis, values, targets, dimension, max_gap=None):
    y=np.moveaxis(values,dimension,0)
    q=np.asarray(targets,dtype=float)
    if q.ndim!=1: raise ValueError('target axes must be 1D')
    hi=np.clip(np.searchsorted(axis,q,side='left'),0,len(axis)-1)
    exact=axis[hi]==q
    lo=np.where(exact,hi,np.maximum(hi-1,0))
    span=axis[hi]-axis[lo]
    w=np.zeros(len(q))
    np.divide(q-axis[lo],span,out=w,where=span>0)
    expand=(len(q),)+(1,)*(y.ndim-1)
    ww=w.reshape(expand)
    # Exact nodes do not require an adjacent (potentially missing) node.
    with np.errstate(invalid='ignore',over='ignore'):
        result=np.where(ww==0,y[lo],np.where(ww==1,y[hi],y[lo]*(1-ww)+y[hi]*ww))
    invalid=~np.isfinite(q)|(q<axis[0])|(q>axis[-1])
    if max_gap is not None: invalid |= (span>max_gap)&~exact
    result[invalid]=np.nan
    return np.moveaxis(result,0,dimension)


def interpolate_grid(frequencies_mhz,times_mjd,values,query_frequencies_mhz,query_times_mjd,
                     *,max_time_gap_seconds=None):
    """Bilinear interpolation, exact-node NaN semantics, no endpoint extrapolation.

    Input/output values have (frequency,time) shape. Missing nodes are never
    skipped. max_time_gap_seconds explicitly refuses long interpolation spans.
    """
    f=_axis(frequencies_mhz,'frequency grid'); t=_axis(times_mjd,'time grid')
    v=np.asarray(values,dtype=float)
    if v.shape!=(len(f),len(t)): raise ValueError('grid values must have (frequency,time) shape')
    if max_time_gap_seconds is not None and (not np.isfinite(max_time_gap_seconds) or max_time_gap_seconds<=0):
        raise ValueError('max_time_gap_seconds must be positive finite or None')
    first=_interp_axis(t,v,query_times_mjd,1,
                       None if max_time_gap_seconds is None else max_time_gap_seconds/86400.)
    return _interp_axis(f,first,query_frequencies_mhz,0)


@dataclass
class ModelCache:
    """Raw FAST model in Jy; never a K-scaled or MASTER product.

    Metadata declarations are checked, not independently certified. The caller
    remains responsible for the sky/array state from which the cache was made.
    """
    frequencies_mhz: np.ndarray
    times_mjd: np.ndarray
    fast_jy: np.ndarray
    gain: np.ndarray|None=None
    gain_frequencies_mhz: np.ndarray|None=None
    gain_times_mjd: np.ndarray|None=None
    metadata: dict=field(default_factory=dict)
    # Original loaded-file identity; mutable effective content is hashed separately.
    sha256: str|None=field(default=None,init=False)
    source_path: Path|None=field(default=None,init=False)

    def __post_init__(self):
        self.frequencies_mhz=_axis(self.frequencies_mhz,'frequency grid')
        self.times_mjd=_axis(self.times_mjd,'time grid')
        if np.any(self.frequencies_mhz<=0): raise ValueError('positive frequencies required')
        self.fast_jy=np.array(self.fast_jy,dtype=float,copy=True)
        if self.fast_jy.shape!=(len(self.frequencies_mhz),len(self.times_mjd)):
            raise ValueError('FAST must have (frequency,time) shape')
        supplied=dict(self.metadata)
        for k,v in _CONTRACT.items():
            if k in supplied and supplied[k]!=v: raise ValueError(f'cache {k} must be {v!r}')
        self.metadata={**_CONTRACT,**supplied}
        if self.gain is not None:
            self.gain_frequencies_mhz=_axis(self.frequencies_mhz if self.gain_frequencies_mhz is None else self.gain_frequencies_mhz,'gain frequency grid')
            self.gain_times_mjd=_axis(self.times_mjd if self.gain_times_mjd is None else self.gain_times_mjd,'gain time grid')
            self.gain=np.array(self.gain,dtype=float,copy=True)
        self.validate()

    def validate(self):
        """Check the current mutable cache without restoring missing declarations."""
        for k,v in _CONTRACT.items():
            if self.metadata.get(k)!=v: raise ValueError(f'cache {k} must be {v!r}')
        # Validate metadata as JSON without arbitrary Python-object/pickle support.
        json.dumps(self.metadata,allow_nan=False)
        f=_axis(self.frequencies_mhz,'frequency grid')
        t=_axis(self.times_mjd,'time grid')
        if np.any(f<=0): raise ValueError('positive frequencies required')
        if np.asarray(self.fast_jy,dtype=float).shape!=(len(f),len(t)):
            raise ValueError('FAST must have (frequency,time) shape')
        if self.gain is not None:
            if self.metadata.get('gain_normalization')!='zenith-1':
                raise ValueError('gain must be declared zenith-1 normalized')
            if self.metadata.get('gain_model') not in ('memo178','lwapb'):
                raise ValueError('cache gain_model must be explicitly memo178 or lwapb')
            gf=_axis(self.gain_frequencies_mhz,'gain frequency grid')
            gt=_axis(self.gain_times_mjd,'gain time grid')
            if np.any(gf<=0): raise ValueError('positive gain frequencies required')
            if np.asarray(self.gain,dtype=float).shape!=(len(gf),len(gt)):
                raise ValueError('gain must have (frequency,time) shape')
        windows=self.metadata.get('off_windows_mjd',[])
        for a,b in windows:
            if not np.isfinite([a,b]).all() or a>=b: raise ValueError('off windows must be finite start/end MJD pairs')

    def fast_at(self,frequencies_mhz,times_mjd,*,max_time_gap_seconds=None):
        self.validate()
        return interpolate_grid(self.frequencies_mhz,self.times_mjd,self.fast_jy,
                                frequencies_mhz,times_mjd,max_time_gap_seconds=max_time_gap_seconds)

    def gain_at(self,frequencies_mhz,times_mjd,*,max_time_gap_seconds=None):
        self.validate()
        if self.gain is None: raise ValueError('cache contains no gain grid')
        return interpolate_grid(self.gain_frequencies_mhz,self.gain_times_mjd,self.gain,
                                frequencies_mhz,times_mjd,max_time_gap_seconds=max_time_gap_seconds)

    def save(self,path,*,overwrite=False):
        self.validate()
        # Audit import is read-only even with an explicit overwrite request.
        import os
        source = self.metadata.get('source_path')
        if source is not None:
            original = Path(source)
            output = Path(path)
            if original.resolve() == output.resolve() or (original.exists() and output.exists() and os.path.samefile(original,output)):
                raise ValueError('cache output must not alias original audit')
        data=dict(frequencies_mhz=self.frequencies_mhz,times_mjd=self.times_mjd,
                  fast_jy=self.fast_jy,metadata_json=np.array(json.dumps(self.metadata,sort_keys=True,allow_nan=False)))
        if self.gain is not None:
            data.update(gain=self.gain,gain_frequencies_mhz=self.gain_frequencies_mhz,gain_times_mjd=self.gain_times_mjd)
        with atomic_target(path,overwrite) as temporary:
            with open(temporary,'wb') as stream: np.savez_compressed(stream,**data)

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json'].item()))
            if not isinstance(meta,dict) or any(k not in meta for k in _CONTRACT):
                raise ValueError('cache contract metadata missing')
            kwargs={k:z[k].copy() for k in ('gain','gain_frequencies_mhz','gain_times_mjd') if k in z}
            c=cls(z['frequencies_mhz'],z['times_mjd'],z['fast_jy'],metadata=meta,**kwargs)
        c.sha256=sha256_file(path)
        c.source_path=path
        return c

    @classmethod
    def from_td_audit(cls,path):
        """Import legacy raw FAST/gain fields, not the old seasonal K/n table.

        Exact duplicate time nodes may be collapsed only when values agree.
        The original source mask must be supplied separately via --mask-fits.
        """
        path=Path(path).resolve()
        def unique(t,v):
            t=np.asarray(t,float); v=np.asarray(v,float)
            order=np.argsort(t,kind='stable'); t=t[order]; v=v[:,order]
            keep=[]
            for i,x in enumerate(t):
                if keep and t[keep[-1]]==x:
                    if not np.array_equal(v[:,keep[-1]],v[:,i],equal_nan=True):
                        raise ValueError('inconsistent duplicate grid times')
                else: keep.append(i)
            return t[keep],v[:,keep]
        with np.load(path,allow_pickle=False) as z:
            if str(z['model_kind'].item())!='FAST' or str(z['flux_unit'].item()).lower()!='jy':
                raise ValueError('audit must declare raw FAST and Jy')
            t,fast=unique(z['model_mjd'],z['model_jy'].T)
            gt,gain=unique(z['gain_mjd'],z['gain_grid'])
            beam=str(z['gain_beam'].item())
            meta=dict(source_format='legacy TD audit (raw model_jy only)',source_sha256=sha256_file(path),
                      gain_model=beam,gain_normalization='zenith-1',mask_policy='source flags must be supplied separately',
                      source_model_kind='FAST',source_path=str(path))
            if 'off_window' in z:
                from astropy.time import Time
                import datetime
                if str(z['off_window_frame'].item()).lower()!='ut': raise ValueError('unsupported audit off-window frame')
                date=datetime.datetime.strptime(str(z['day'].item()),'%Y%m%d')
                epoch=float(Time(date,scale='utc').mjd)
                meta['off_windows_mjd']=[(epoch+np.asarray(z['off_window'],float)/24).tolist()]
            c=cls(z['model_freqs_mhz'],t,fast,gain,z['gain_freqs_mhz'],gt,meta)
        c.sha256=meta['source_sha256']
        c.source_path=path
        return c
