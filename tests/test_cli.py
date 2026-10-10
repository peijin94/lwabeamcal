import json
import numpy as np
from astropy.io import fits
from lwabeamcal.cli import main
from lwabeamcal import FrozenBackground, calibrate, ModelCache


def test_model_info_reports_exact_frozen_coefficients(capsys):
    assert main(['model-info'])==0
    d=json.loads(capsys.readouterr().out)
    assert d['parameters']['n50_jy']==288.95515757076816
    assert d['parameters']['k']==1.1


def test_cli_cache_info_and_failure_exit_status(tmp_path,capsys):
    p=tmp_path/'cache.npz';ModelCache([50],[60000],[[1]]).save(p)
    assert main(['cache-info',str(p)])==0
    assert json.loads(capsys.readouterr().out)['metadata']['model_state']=='raw'
    assert main(['cache-info',str(tmp_path/'notthere')])==2
    assert 'error:' in capsys.readouterr().err


def test_cli_runs_fits_correction(tmp_path,capsys):
    p=tmp_path/'raw.fits'; c=tmp_path/'cache.npz';out=tmp_path/'cal.fits'
    primary=fits.PrimaryHDU(np.ones((1,1,1,2)));primary.header['BUNIT']='sfu'
    freq=fits.BinTableHDU.from_columns([fits.Column(name='SFREQ',format='D',unit='GHz',array=[.05])])
    times=fits.BinTableHDU.from_columns([fits.Column(name='mjd',format='D',array=[61278.,61278.]),
         fits.Column(name='time',format='D',array=[20.*3600000,20.*3600000+1000])])
    fits.HDUList([primary,freq,times]).writeto(p)
    t=61278+np.array([20.*3600000,20.*3600000+1000])/86400000.
    ModelCache([50.],t,[[1000.,1000.]]).save(c)
    assert main(['correct',str(p),'--cache',str(c),'--output',str(out)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['valid_samples']==2 and out.exists()
    assert main(['correct',str(p),'--cache',str(c),'--output',str(out)])==2
