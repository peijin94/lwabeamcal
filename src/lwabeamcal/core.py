"""Format-independent array calibration; all spectral arrays use (frequency, time)."""
from dataclasses import dataclass, asdict
from enum import IntFlag
import numpy as np


class Mask(IntFlag):
    """Fatal validity bits (separate from unmodified caller/provider mask)."""
    NONFINITE_RAW = 1
    UNSUPPORTED_BACKGROUND = 2
    INVALID_GAIN = 4
    TERRAIN = 8
    USER = 16
    OFF_POINTING = 32
    UNSUPPORTED_FREQUENCY = 64
    NUMERICAL_OVERFLOW = 128


class Caution(IntFlag):
    """Nonfatal scientific caveats; no data are hidden by these bits."""
    UPPER_BAND = 1
    BELOW_40 = 2
    LOW_ELEVATION = 4
    CUSTOM_MODEL = 8
    NO_GEOMETRY = 16


def unit_scale(unit):
    """Scale explicitly declared Jy or SFU into Jy; never infer amplitudes."""
    if not isinstance(unit, str) or unit.strip().lower() not in ('jy', 'sfu'):
        raise ValueError('unit must be explicitly Jy or SFU')
    return 10000. if unit.strip().lower() == 'sfu' else 1.


def _frequencies(values):
    f = np.asarray(values, dtype=float)
    if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or np.any(f <= 0):
        raise ValueError('frequencies_mhz must be a nonempty positive finite 1D array')
    if np.any(np.diff(f) <= 0):
        raise ValueError('frequencies_mhz must be strictly increasing')
    return f


@dataclass(frozen=True)
class FrozenBackground:
    """Raw FAST sky model times flat K plus an analytic imaging-trained floor."""
    k: float = 1.1
    n50_jy: float = 288.95515757076816
    exponent: float = 3.5547074905328397
    reference_mhz: float = 50.0

    def __post_init__(self):
        if not all(np.isscalar(x) and np.isfinite(x) for x in asdict(self).values()):
            raise ValueError('model coefficients must be finite scalars')
        if self.k <= 0 or self.n50_jy < 0 or self.reference_mhz <= 0:
            raise ValueError('positive K/reference and nonnegative floor amplitude required')

    def floor_jy(self, frequencies_mhz):
        """Evaluate n exactly at the requested frequencies, not at coarse knots."""
        f = np.asarray(frequencies_mhz, dtype=float)
        if np.any(~np.isfinite(f)) or np.any(f <= 0):
            raise ValueError('positive finite frequencies required')
        with np.errstate(over='ignore', invalid='ignore'):
            floor = self.n50_jy * (f / self.reference_mhz) ** self.exponent
        if not np.isfinite(floor).all():
            raise ValueError('floor overflows at supplied frequency')
        return floor

    def evaluate(self, fast_jy, frequencies_mhz):
        """Return B in Jy. FAST must be raw/unscaled, shape compatible with (f,t)."""
        f = _frequencies(frequencies_mhz)
        with np.errstate(over='ignore', invalid='ignore'):
            return self.k * np.asarray(fast_jy, dtype=float) + self.floor_jy(f)[:, None]

    def metadata(self):
        custom = self != FrozenBackground()
        return dict(model_id='custom' if custom else 'flatk-imaging-20261006', formula='B=K*FAST+n50*(nu/reference_MHz)**exponent',
                    parameters=asdict(self), calibration_status='empirical; absolute calibration unverified',
                    training_days=[] if custom else ['20260615', '20260715', '20260915'],
                    held_out_days=[] if custom else ['20260825', '20260826'], fit_domain_mhz=None if custom else [40., 85.],
                    benchmark_mhz=[40., 80.], raw_fast_required=True,
                    custom_parameters=self != FrozenBackground())


@dataclass(frozen=True)
class CalibrationResult:
    corrected_jy: np.ndarray
    background_jy: np.ndarray
    gain: np.ndarray
    mask: np.ndarray
    caution: np.ndarray
    input_mask: np.ndarray
    metadata: dict

    @property
    def valid(self):
        return self.mask == 0


def calibrate(raw, frequencies_mhz, *, fast_jy, gain, input_unit, model=None,
              sun_alt_deg=None, input_mask=None, off_mask=None, experimental=False,
              terrain_min_alt_deg=12., min_gain=None):
    """Subtract background then divide by zenith-normalized Stokes-I gain.

    `raw` has shape (nfreq, ntime). FAST, gain and masks may broadcast to
    that shape (use (nfreq,1) for per-frequency arrays). Both input and
    background are interpreted in the documented units, never normalized
    from observed intensities. Missing/invalid support is NaN, signed valid
    residuals survive. Low-elevation/experimental flags are cautions only.
    """
    f = _frequencies(frequencies_mhz)
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != len(f) or not raw.shape[1]:
        raise ValueError('raw must have shape (len(frequencies_mhz), nonzero ntime)')
    if not np.isfinite(terrain_min_alt_deg) or not 0 <= terrain_min_alt_deg <= 90:
        raise ValueError('terrain_min_alt_deg must be between 0 and 90')
    if min_gain is not None and (not np.isfinite(min_gain) or min_gain <= 0):
        raise ValueError('min_gain must be a positive finite value or None')
    scale = unit_scale(input_unit)
    model = model or FrozenBackground()
    background = np.broadcast_to(model.evaluate(fast_jy, f), raw.shape).copy()
    g = np.broadcast_to(np.asarray(gain, dtype=float), raw.shape).copy()
    original = np.zeros(raw.shape, np.uint32)
    if input_mask is not None:
        m = np.asarray(input_mask)
        if m.dtype.kind not in 'biu' or np.any(m < 0) or np.any(m > np.iinfo(np.uint32).max):
            raise ValueError('input_mask must contain nonnegative integer bitmasks')
        original = np.broadcast_to(m, raw.shape).astype(np.uint32, copy=True)
    flags = np.zeros(raw.shape, np.uint16)
    caution = np.zeros(raw.shape, np.uint16)
    flags[~np.isfinite(raw)] |= int(Mask.NONFINITE_RAW)
    fast = np.broadcast_to(np.asarray(fast_jy, dtype=float), raw.shape)
    flags[~np.isfinite(background) | ~np.isfinite(fast) | (fast < 0)] |= int(Mask.UNSUPPORTED_BACKGROUND)
    bad_gain = ~np.isfinite(g) | (g <= 0)
    if min_gain is not None:
        bad_gain |= g < min_gain
    flags[bad_gain] |= int(Mask.INVALID_GAIN)
    flags[original != 0] |= int(Mask.USER)
    domain = (15.,90.) if experimental else (40.,85.)
    flags[np.broadcast_to(((f < domain[0]) | (f > domain[1]))[:,None], raw.shape)] |= int(Mask.UNSUPPORTED_FREQUENCY)
    caution[f > 80.,:] |= int(Caution.UPPER_BAND)
    caution[f < 40.,:] |= int(Caution.BELOW_40)
    if model != FrozenBackground():
        caution[:] |= int(Caution.CUSTOM_MODEL)
    if sun_alt_deg is None:
        caution[:] |= int(Caution.NO_GEOMETRY)
    else:
        alt = np.asarray(sun_alt_deg, dtype=float)
        if alt.shape != (raw.shape[1],):
            raise ValueError('sun_alt_deg must have shape (ntime,)')
        flags[:,~np.isfinite(alt) | (alt < terrain_min_alt_deg)] |= int(Mask.TERRAIN)
        caution[:,np.isfinite(alt) & (alt >= terrain_min_alt_deg) & (alt < 20.)] |= int(Caution.LOW_ELEVATION)
    if off_mask is not None:
        m = np.asarray(off_mask)
        if m.dtype.kind != 'b':
            raise ValueError('off_mask must be boolean')
        flags[np.broadcast_to(m, raw.shape)] |= int(Mask.OFF_POINTING)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        corrected = (raw * scale - background) / g
    flags[(flags == 0) & ~np.isfinite(corrected)] |= int(Mask.NUMERICAL_OVERFLOW)
    corrected = np.where(flags == 0, corrected, np.nan)
    return CalibrationResult(corrected, background, g, flags, caution, original,
                             dict(model=model.metadata(), input_unit=input_unit.strip().lower(),
                                  output_unit='jy', ordering='background subtraction then gain division',
                                  terrain_min_alt_deg=terrain_min_alt_deg, geometry_supplied=sun_alt_deg is not None,
                                  experimental=bool(experimental), min_gain=min_gain))
