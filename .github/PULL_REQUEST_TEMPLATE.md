## What this changes

<!-- One paragraph. If it fixes a bug, say what the failure looked like. -->

## The number

<!--
This project's rule: a claim without a command that reproduces it does not go
in. Fill in whichever applies and delete the rest.

- `pytest -q` before / after:
- `snndsl deploy <file> --verify` max |error| before / after:
- third-party probe status before / after:
-->

## Checklist

- [ ] `python -m pytest -q` passes (73 passed, 4 skipped on a clean install)
- [ ] `python -m ruff check snndsl tests examples` is clean
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] every number I changed is updated everywhere it appears (README, `docs/`,
      docstrings, test assertions)
- [ ] if I removed a limitation, I said what it actually was — "it was our bug,
      not the standard's" is a valid and welcome answer here
- [ ] no new dependency without a reason next to it in `pyproject.toml`

## Anything I could not verify

<!--
Be explicit. "Verified on CPU only", "no vendor interpreter available for the
sinabs probe", "the residual case still has no NIR merge primitive". An honest
gap is fine; an unmarked one is not.
-->
