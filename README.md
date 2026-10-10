# lwabeamcal

Standalone Python package for **OVRO-LWA beamformed solar spectra**. Version 0.1.0.
The correction is always applied in this order:

```text
B(t,nu) = 1.1 * FAST_raw(t,nu) + 288.95515757076816 * (nu_MHz/50)**3.5547074905328397 Jy
I_cal(t,nu) = (I_raw_Jy(t,nu) - B(t,nu)) / g(t,nu)
```

Primary-beam correction is **division by gain**, not subtraction. No intensity
normalization, daily amplitude fitting, clipping of negative residuals, or
extra empirical scaling is applied.

## Install

Python >=3.10; dependencies NumPy and Astropy. From this directory:

```bash
python -m pip install .
# Development/test installation:
python -m pip install -e '.[test]'
lwabeamcal --help
```

Built wheel and source archives are in `dist/`. No dependency on the research
checkout (`spec-pb-corr`), its scripts, `lwasolarproc`, or its absolute paths.

## Python API

All spectral arrays have shape **(frequency, time)**. Frequencies are explicitly
MHz; cache/native times are **UTC MJD**; input flux unit is explicitly `jy` or
`sfu` (1 SFU = 10000 Jy). Gains are zenith-normalized Stokes-I power gains.

```python
from lwabeamcal import calibrate

result = calibrate(
    raw_spectrum, frequencies_mhz,
    input_unit="sfu",
    fast_jy=raw_fast_background,     # NOT K-scaled; not MASTER
    gain=normalized_gain,
    sun_alt_deg=solar_altitudes_deg,
    input_mask=original_flags,      # optional integer (f,t) mask, preserved separately
)
corrected_jy = result.corrected_jy
valid = result.valid
background_jy = result.background_jy
```

`FrozenBackground().floor_jy(frequencies_mhz)` evaluates the frozen analytic
floor exactly at native channels, not by interpolating floor knots.
`Memo178Beam().gain(frequencies_mhz, azimuth_deg, elevation_deg)` computes the
operational dual-dipole gain. `solar_azel(times_mjd)` uses the fixed OVRO site
and geometric elevations, with IERS downloads disabled. Stale local IERS
  tables can emit accuracy warnings; update the environment's offline table
  for precision work rather than silently suppressing them.

`result.mask` contains fatal validity reasons; `result.caution` contains
nonfatal low-elevation / exploratory-band cautions. `result.input_mask`
preserves provider flags separately, because their bit namespace may differ.
Negative **valid** solar residuals survive; invalid samples become NaN.
Without supplied solar elevation, the array API cannot apply terrain masks
and marks every cell with `NO_GEOMETRY`. FITS correction computes geometry.

See `examples/arrays.py` for a runnable explicitly synthetic demonstration.

## Background caches (required physical inputs)

**This release consumes a precomputed raw FAST sky background. It does not
calculate GSM sky integrals or provide an annual background cache.** The
sky/element/array calculation remains an external upstream product. The frozen
noise-floor formula alone is NOT a background predictor. A cache must match
your observing date, frequencies, array/element configuration and sky model.
The file hash provides provenance, not physical certification.

Portable NPZ caches use `allow_pickle=False`, versioned JSON metadata and
explicit axes. Create from known unscaled FAST values:

```python
from lwabeamcal import ModelCache
cache = ModelCache(
    model_frequencies_mhz, model_times_mjd, fast_jy,  # fast_jy shape (f,t)
    metadata={"origin": "describe your raw FAST calculation and array state"},
)
cache.save("fast-cache.npz")
```

The loader requires metadata `schema=1`, `model_kind=FAST`, `model_state=raw`,
`flux_unit=Jy`, `frequency_unit=MHz`, `time_unit=MJD UTC`. Creation fills these
declarations; the caller remains responsible for their truth. MASTER,
pre-scaled/calibrated FAST and mismatched units are rejected when declared.
It cannot detect a falsely labeled pre-scaled array from numbers alone.

Optional gain grids have `(gain_frequency,gain_time)` shape and must explicitly
declare `gain_model` (`memo178` or `lwapb`) and `gain_normalization=zenith-1`.
Separate gain axes are supported. **lwapb support is via supplied gain grids**;
its coefficients/evaluator are not bundled. Memo178 can be computed directly
when a gain cache is absent or with `--compute-gain`.

Interpolation is bounded and bilinear. Exact nodes remain usable next to a
NaN node; interior missing-node support is never skipped or bridged. There is
no endpoint clamping/extrapolation. Set `--max-time-gap-seconds` to disallow
interpolation across unusually long gaps; numerical cache gaps must otherwise
be represented explicitly by missing nodes. Observational acquisition gaps
are not filled by this package.

### Import the existing research audit

```bash
lwabeamcal import-audit 20260826_memo178_td.td_inputs.npz --output fast-cache.npz
lwabeamcal cache-info fast-cache.npz
```

The importer extracts **raw** `model_jy` and gain grids only, drops the old
seasonal Ks/n parameters, verifies duplicate nodes agree, records the source
SHA256, and retains the actual off-pointing interval from the audit. Its old
native sample flags are NOT present in an audit NPZ: supply the corresponding
FITS `TD_MASK` separately using `--mask-fits` if those flags must be retained.

## FITS CLI

```bash
lwabeamcal correct 20260826.fits \
    --cache fast-cache.npz \
    --mask-fits 20260826_memo178_td.fits \
    --beam memo178 \
    --output 20260826_lwabeamcal.fits
```

The adapter accepts the spec_v2 primary layout `(nStokes,1,nfrequency,ntime)`.
`SFREQ` defaults to its established GHz convention only if its column unit is
absent; explicit Hz/kHz/MHz/GHz units are recognized. The time column is
milliseconds into each row's MJD day. Frequencies/times must increase strictly.

- Background subtraction and gain division apply to **I only**.
- Other Stokes planes receive flux-unit conversion only; no polarization
  calibration or V background/PB correction is claimed.
- All output Stokes planes match `BUNIT` (Jy by default, optional SFU).
- Frequency, timing and other table payloads remain unchanged.
- `TD_MASK`, `TD_BKG` (Jy), `TD_GAIN`, `LWA_CAUTION`, `LWA_INPUT_MASK`, and
  JSON `LWA_PROVENANCE` extensions describe the applied correction. Original
  cache-file and effective in-memory content hashes are recorded separately;
  `LWACSHA` contains the full effective fingerprint in a standard FITS card.
- Per-day off intervals are explicitly supplied by cache metadata or repeated
  `--off-window START_MJD END_MJD`. No daily off-pointing time is assumed.
- Existing products, source/cache/mask aliases, symlink targets and repeated
  corrections are refused. `--overwrite` is explicit and still cannot touch
  an input. Writes are atomic; temporary arrays are disk-backed, processed
  in time chunks (default 4096 native samples).
- `--input-unit sfu` is an explicit override for a **known** mislabeled legacy
  file, not an amplitude-based unit guess. Missing/unknown units fail otherwise.

## Scientific scope and caveats

The adopted floor was trained on imaging agreement on June 15, July 15 and
September 15, 2026, with August 25/26 held out. This is a smooth empirical
background model, **not a separately measured receiver-noise floor** or an
independent absolute calibration. Training range 40-85 MHz; benchmark 40-80 MHz.

Default usable frequency domain is 40-85 MHz, with 80-85 flagged exploratory.
`--experimental` permits 15-40 and 85-90 MHz **only where the supplied FAST/gain
cache supports them**. It does not extrapolate FAST, create missing support or
remove shortwave interference. Floor extrapolation below 40 remains unvalidated.

Terrain elevation <12 degrees is invalid by default. Elevations 12-20 degrees
are retained with a caution because the residual edge sag is unresolved. There
is no hidden gain cutoff; optional `--min-gain` is explicit and recorded. This
package retains existing RFI masks but **does not invent a new RFI detector**.
Raw-minus-background can be negative near the edges or during dropouts; signed
values remain visible and should not be interpreted as physical negative solar
flux. Day-to-day offsets and Memo178/lwapb systematics remain.

## Development and release

```bash
python -m pytest tests -q
python -m build
```

`notes.md` records verification; `mem.md` records maintenance conventions.
No PyPI upload/GitHub publication has been performed. Redistribution licensing,
especially the upstream Memo178 coefficient provenance, must be decided before
public release; see `THIRD_PARTY_NOTICES.md`.
