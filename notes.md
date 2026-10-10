# Verification and development notes

## 2026-10-10 — standalone extraction, version 0.1.0

Created a src-layout Python distribution with array API, raw FAST/gain NPZ
cache, read-only legacy-audit importer, standalone Memo178 beam, offline OVRO
geometry, FITS I-only correction and a CLI. No research-repo imports, hardcoded
data paths, production changes or manuscript edits. Background inputs stay
external and have not been re-fit or regenerated here.

### Tests and scientific regression

Test-first vertical slices were observed RED (missing modules/features and
explicit failures) then GREEN for array arithmetic, validity/unit policy,
cache/interpolation, beam/geometry, FITS I/O and CLI. The runtime NumPy
unsigned-array/IntFlag casting failure was corrected with explicit integer
bit values. Tests cover source immutability, no-clobber/alias/refusal of repeat
correction, unit conversion of every Stokes plane, cache state/units, missing
support, gain model mismatch, exact native-channel floor evaluation, provider
mask retention, atomic cleanup, and real-data regressions.

Final suite: 108 passed on local Python 3.14 / NumPy 2.5.3 / Astropy 8.0.1,
and 108 passed on inti Python 3.10 / NumPy 1.26.4 / Astropy 6.1.3. The latter
reports 28 stale offline IERS-table accuracy warnings; they were not suppressed.
An independent local Codex static review initially identified cache alias,
normalization-declaration, mutable provenance, unsigned mask and shifted Stokes
WCS problems. Separate fix contexts supplied test-first regressions; final
independent review passed with no outstanding security or logic findings.
The build and installed-wheel checks are recorded in release verification.

Actual held-out August 26 data, 3 channels:
- 453975 valid entire-day cells per beam round-tripped against the approved
  frozen model, errors <=1.82e-11 Jy (this check reconstructs FAST from stored B).
- Independent actual FITS samples plus raw FAST/gain audits: 310 native times,
  906 valid cells per beam, max discrepancy 2.19e-11 Jy Memo178 / 3.42e-11 Jy
  lwapb. This check does NOT reconstruct FAST from the fitted B. Full output
  FITS checksums verified. Original observational files unchanged.
- A small real, nonsynthetic fixture is retained in tests/data, not a full day
  FITS or an annual cache. The expected values are from the approved plotting
  product. This is numerical extraction parity, not absolute calibration.
- Direct Memo178 coefficients match the old implementation at 40/55/70 MHz,
  representative azimuths/elevations, to 1e-12 relative tolerance.

Package implementation preserves masks and signs but does not solve the
low-elevation negative residuals, shortwave/RFI pollution, or daily offsets.
Primary-beam correction is gain division. Licensing remains unassigned;
no public upload or remote repository has been created.

Python 3.10 / Astropy 6 exposed a real FITS compatibility bug: a full SHA256
under a long HIERARCH keyword could not fit that version's card formatter.
Changed to standard short `LWACSHA` (one 80-byte card, full digest); the JSON
provenance retains separate original-file and effective-content hashes. Added
an exact-card regression, then reran both environments successfully.

The generic Hermes helper dispatch initially failed because its configured
model was unsupported by the provider; direct implementation plus explicitly
pinned local Codex review/fix contexts replaced it. No codex-inti delegation
was used for this package task. Tests used disposable scratch directories;
fix-context sandbox tests used a Git-ignored repo-local .cache fallback.
