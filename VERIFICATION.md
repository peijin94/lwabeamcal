# Release verification — lwabeamcal 0.1.0

Verified 2026-10-10 with real command output:

| Environment | Result |
|---|---|
| Local source, Python 3.14 / NumPy 2.5.3 / Astropy 8.0.1 | 108 passed |
| Fresh, noneditable installed wheel, outside source checkout | 108 passed |
| inti, Python 3.10 / NumPy 1.26.4 / Astropy 6.1.3 | 108 passed; 28 stale offline IERS warnings |
| pip dependency check | No broken requirements |
| Independent final static code review | Passed; no outstanding logic/security findings |
| Wheel resource check | Frozen model JSON and third-party notice present |

Real held-out 2026-08-26 data at 40/55/70 MHz:
- Entire-day array extraction parity: 453975 valid cells per beam, signed
  residuals/flags preserved; maximum difference 1.82e-11 Jy. This roundtrip
  uses stored B to recover FAST and is not independent sky-model validation.
- Independent raw FITS plus original unscaled FAST/gain audits: 310 native
  times, 906 valid cells per beam; max difference 2.1828e-11 Jy Memo178 and
  3.4107e-11 Jy lwapb against the approved plotting product.
- FITS checksums verified; native table payloads preserved; observational
  inputs untouched.

Wheel and source archive successfully built. Installation was exercised in a
new scratch virtual environment with fresh dependencies, not an editable
research-repository import. CLI help/version, synthetic runnable API example,
real audit-to-cache conversion, and cache-info all executed successfully.

The initial review identified data/provenance edge cases; focused test-first
fixes addressed normalization declarations, mutable-cache hashes/validation,
portable/direct-audit alias protection, unsigned FITS masks, Stokes reference
pixels and failure-time resource cleanup. Cross-version testing found and
fixed a full-SHA256/long-HIERARCH card compatibility problem with Astropy 6.

This verifies software extraction and numerical parity, not independent
absolute calibration or unresolved low-elevation/shortwave behavior. FAST
cache production remains external. No PyPI or GitHub publication occurred;
licensing is unassigned pending upstream review.
