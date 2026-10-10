# Maintenance context

- Package correction is background subtraction THEN gain division, not PB subtraction.
- Frozen coefficients: K=1.1, n50=288.95515757076816 Jy, b=3.5547074905328397, reference=50 MHz.
- Never introduce intensity normalization or clip signed valid residuals.
- FAST cache must be raw/unscaled Jy, never MASTER or already K-calibrated.
- Core shape (frequency,time); MHz, UTC MJD, explicit Jy/SFU.
- Primary FITS I-only correction; other planes unit conversion only.
- Preserve provider bit masks separately from the package reason namespace.
- Python package uses no research-repo imports or absolute dependency paths.
- For regressions use canonical scratch; heavy full-day work belongs on inti.
- Licensing is unassigned pending upstream coefficient/implementation review; do not publish by default.
