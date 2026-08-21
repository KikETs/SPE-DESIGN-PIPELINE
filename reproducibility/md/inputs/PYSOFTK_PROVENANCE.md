# Vendored PySoftK provenance

Verified against the official upstream repository on 2026-08-21:
`https://github.com/alejandrosantanabonilla/pysoftk`.

## Source identity

- Upstream commit: `a3458567d9b61e5caf6d891861e5fa016dfe97b5`
- Commit date/message: 2025-10-20, `Implement dendrimer monomer and builder classes`
- Git description in upstream history: `v.1.0.1-151-ga345856`
- Upstream license at that commit: BSD-3-Clause
- Historical installed-environment metadata: `pysoftk 1.0.0`

The installed package version and source-tree commit description are both
reported because the vendored tree itself had no embedded package metadata.

## Byte-level comparison

The local vendored directory contains 44 tracked source files. Compared with
the upstream commit tree:

- 43 of 44 files are byte-identical Git blobs;
- there are no local-only or upstream-only source files;
- only `pysoftk/linear_polymer/linear_polymer.py` differs.

## Local modification

The local `linear_polymer.py` adds an optional multicore construction path:

- RDKit ETKDGv3 multi-conformer embedding with deterministic
  `randomSeed = 0xF00D`;
- UFF/MMFF conformer optimization and energy ranking;
- parallel OpenBabel refinement through `ProcessPoolExecutor`;
- environment controls `GROMACS_PYSOFTK_MULTICORE`,
  `GROMACS_PYSOFTK_INTERNAL_THREADS`, `GROMACS_PYSOFTK_NUM_CONFS`, and
  `GROMACS_PYSOFTK_OB_WORKERS`;
- fallback to the original serial PySoftK path on failure.

The full modified source needed for polymer construction is included at
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/pysoftk/`. The upstream
BSD-3-Clause license is retained in that directory as `LICENSE.md`.

