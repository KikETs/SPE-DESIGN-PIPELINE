# Vendor

This directory contains small local dependencies needed by the release.

- `canonicalize_psmiles-0.1.2-py3-none-any.whl`: wheel built from
  `Ramprasad-Group/canonicalize_psmiles` commit
  `729dfd20b0e909df7cb245ffbfa5ab2187ffbf8b`.
- `psmiles_local/`: unpacked `psmiles` 0.6.10 source with dependency metadata
  rewritten to use the local canonicalization wheel instead of a VCS URL.
