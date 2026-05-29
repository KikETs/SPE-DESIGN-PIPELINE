# ff_sources (shared across Traj_*)

This folder is used by `gromacs_opls_ligpargen_boss_patched.ipynb`.
(Currently configured for **PolyParGen-style polymer + ion FF sources** by default.)

Expected files:

- `ions/tfsi_clp_GMX.itp` (if `OPLS_TFSI_SOURCE=CLP`)
- `ions/tfsi_oplsil_GMX.itp` (if `OPLS_TFSI_SOURCE=OPLSIL`)
- `ions/li_opls_GMX.itp`
- `ligpargen/polymer_GMX.itp` (optional, if `OPLS_POLYMER_BACKEND=LIGPARGEN`)
- `polypargen/polymer_GMX.itp` (auto-generated output when `OPLS_POLYMER_BACKEND=POLYPARGEN`)

Notebook search order for this folder:

1. `$OPLS_FFSRC_DIR`
2. `ff_sources` in this package
3. `Traj_xxxx/ff_sources` inside the current run folder

## Typical provenance

- polymer (`polypargen/polymer_GMX.itp`): auto-built by notebook with PolyParGen-like flow (fragment split -> LigParGen/BOSS typing -> merged ITP)
- polymer (`ligpargen/polymer_GMX.itp`): optional direct LigParGen/BOSS output
- TFSI (`tfsi_clp_GMX.itp`): CL&P/ILFF source converted/exported to GROMACS .itp
- TFSI (`tfsi_oplsil_GMX.itp`): OPLS-IL/2009IL source (NTf2) converted/exported to GROMACS .itp
- Li (`li_opls_GMX.itp`): Li+ monatomic itp from the same ion FF family used for TFSI

## Notes

- Keep one consistent FF family in one simulation setup.
- If using OPLS-IL VSIL parameters, do not mix with 2009IL non-VSIL parameters.
- Notebook enforces `spec.li_charge_scale = 0.7` for Li/TFSI.
- Default polymer backend is `OPLS_POLYMER_BACKEND=POLYPARGEN`.
- Large polymer chains can exceed BOSS QM atom limits with direct LigParGen typing.
- PolyParGen-like fragment settings can be tuned via env vars:
  - `OPLS_POLYPARGEN_REBUILD` (default `1`)
  - `OPLS_FRAG_CORE_ATOMS` (default `130`)
  - `OPLS_FRAG_OVERLAP_ATOMS` (default `50`)
  - `OPLS_FRAG_MIN_CORE_ATOMS` (default `90`)
  - `OPLS_FRAG_MAX_COUNT` (default `10`)
