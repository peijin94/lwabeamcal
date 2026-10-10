"""Operational Memo-178 Stokes-I gain, same coefficients as lwasolarproc.

Coefficient provenance: pbcorr/beam_memo178.py (spec-pb-corr), originally
lwasolarproc/beammodel.py. Redistribution rights have not been assigned
here: see THIRD_PARTY_NOTICES.md before public release.
"""
import numpy as np
from .core import _frequencies

_E178 = np.array(
    [
        [-4.529931167425190e01, -3.066691727279789e01, +7.111192148086860e01, +1.131338637814271e01],
        [+1.723596273204143e02, +1.372536555724785e02, -2.664504470520252e02, -3.493942140370373e01],
        [-2.722311453669980e02, -2.368121452949910e02, +4.343953141004854e02, +5.159758406047006e01],
        [+2.408402047219155e02, +2.302040670131884e02, -3.993175413350526e02, -4.152532958513045e01],
        [-1.334589043702679e02, -1.414814115813401e02, +2.312015990805256e02, +2.016753495756661e01],
        [+4.917278320096442e01, +5.820262041648866e01, -8.952191930147437e01, -6.170044495806915e00],
        [-1.244683117802972e01, -1.652458005311876e01, +2.396407975769994e01, +1.185153343629695e00],
        [+2.195017889732819e00, +3.281357788749097e00, -4.500463893128127e00, -1.311234364955677e-01],
        [-2.690339381925372e-01, -4.547844648614167e-01, +5.919912952153034e-01, +4.850030701166225e-03],
        [+2.243604641077215e-02, +4.311447324076417e-02, -5.345256010140458e-02, +7.056536178294128e-04],
        [-1.211643367267544e-03, -2.665185970775750e-03, +3.157455237232649e-03, -1.102532965104807e-04],
        [+3.810179416998095e-05, +9.681257383030349e-05, -1.099243217364324e-04, +6.250327364174989e-06],
        [-5.277176362640874e-07, -1.567746596003443e-06, +1.710529173897111e-06, -1.350506188926788e-07],
    ],
    dtype=float,
)

_H178 = np.array(
    [
        [+4.062920357822495e02, +3.038713068453467e01],
        [-1.706845366994521e03, -1.337217221068207e02],
        [+3.095438045596764e03, +2.567017051330929e02],
        [-3.164514198869798e03, -2.757538828204631e02],
        [+2.035008485840167e03, +1.850998851042284e02],
        [-8.713893456840954e02, -8.227855478515090e01],
        [+2.564477521133275e02, +2.502479051323083e01],
        [-5.262251671400085e01, -5.287316195660030e00],
        [+7.519512104996896e00, +7.754610944702455e-01],
        [-7.337746902310935e-01, -7.744604652809532e-02],
        [+4.663414310713179e-02, +5.024254273102379e-03],
        [-1.740005497709271e-03, -1.908989166200470e-04],
        [+2.892116885178882e-05, +3.223985512686652e-06],
    ],
    dtype=float,
)

class Memo178Beam:
    name='memo178'
    normalization='zenith-1'

    def gain(self,frequencies_mhz,azimuth_deg,elevation_deg):
        """Gain shape (nfreq, ntime), angles in degrees N through E.

        The dual-dipole average is 0.5*Tr(J J^dagger). Values below the
        horizon are unsupported (NaN), not elevation-clamped.
        """
        f=_frequencies(frequencies_mhz)
        if np.any((f<10)|(f>100)): raise ValueError('Memo178 coverage is 10-100 MHz')
        az,el=np.broadcast_arrays(np.atleast_1d(azimuth_deg).astype(float),np.atleast_1d(elevation_deg).astype(float))
        if az.ndim!=1: raise ValueError('azimuth and elevation must be scalar or 1D')
        good=np.isfinite(az)&np.isfinite(el)&(el>=0)&(el<=90)
        theta=np.pi/2-np.deg2rad(np.where(good,el,45.))
        z=theta/(np.pi/2); cos=np.cos(theta)
        out=np.full((len(f),len(az)),np.nan)
        for i,nu in enumerate(f):
            alpha,beta,gamma,delta=[np.polynomial.polynomial.polyval(nu/10.,_E178[:,j]) for j in range(4)]
            h_alpha,h_beta=[np.polynomial.polynomial.polyval(nu/10.,_H178[:,j]) for j in range(2)]
            with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                e=(1-z**alpha)*cos**beta+gamma*z*cos**delta
                h=(1-z**h_alpha)*cos**h_beta
                phi=np.deg2rad(np.mod(az,360.))
                def pattern(p): return np.sqrt((e*np.cos(p))**2+(h*np.sin(p))**2)
                out[i]=np.where(good,.5*(pattern(phi)+pattern(phi+np.pi/2)),np.nan)
        return out


def solar_azel(times_mjd):
    """Geometric Sun az/el at OVRO. No online site registry or IERS downloads."""
    from astropy import units as u
    from astropy.coordinates import AltAz,EarthLocation,get_sun
    from astropy.time import Time
    from astropy.utils import iers
    t=np.asarray(times_mjd,float)
    if t.ndim!=1 or not len(t) or not np.isfinite(t).all(): raise ValueError('finite 1D UTC MJD times required')
    site=EarthLocation.from_geodetic(-118.2817*u.deg,37.23317*u.deg,1183*u.m)
    times=Time(t,format='mjd',scale='utc')
    with iers.conf.set_temp('auto_download',False):
        sun=get_sun(times).transform_to(AltAz(location=site,obstime=times,pressure=0*u.hPa))
    return np.asarray(sun.az.deg),np.asarray(sun.alt.deg)
