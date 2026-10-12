# Third-party provenance and release status

The Memo-178 coefficient arrays in `src/lwabeamcal/beam.py` come from
`spec-pb-corr/pbcorr/beam_memo178.py`, which attributes its constants/formula to
`peijin94/lwasolarproc/lwasolarproc/beammodel.py` and the LWA Memo-178 dipole model.
The constants have been retained, not refitted. Original implementation and
numerical reference are documented in that research file and its tests.

No new license has been assigned to the combined distribution, and no claim
is made that the upstream code/coefficients are MIT/BSD or public domain.
The repository was made public on GitHub on 2026-10-12 at the owner's explicit
request. Public visibility is not an upstream licensing clearance or a grant
of redistribution rights. Ownership and redistribution terms remain under
review before choosing a distribution license or publishing to PyPI.
NumPy and Astropy are dependencies, not copied into this distribution.
