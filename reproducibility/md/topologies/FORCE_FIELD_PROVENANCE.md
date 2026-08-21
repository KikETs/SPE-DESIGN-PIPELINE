# Force-field and charge provenance

Evidence checked: 2026-08-21.

## Polymer and TFSI atom typing

The archived command forms in
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/ACPYPE_INVOCATIONS.md`
are:

```bash
acpype -i <polymer-fixed.pdb> -b polymer -c gas -o gmx -n 0
acpype -i tfsi.pdb -b tfsi -c gas -o gmx -n -1
```

No `-a` option is present. Output headers identify ACPYPE 2023.10.27. The CLI
source for that exact release defines the omitted `--atom_type/-a` default as
`gaff2` (`https://github.com/alanwilter/acpype/blob/2023.10.27/acpype/parser_args.py`,
checked 2026-08-21). Therefore the supported force-field identification is:

**GAFF2 atom typing through ACPYPE 2023.10.27's versioned default.**

Verification status is `INFERRED_FROM_VERSIONED_DEFAULT_AND_ARCHIVED_COMMAND`,
not `DIRECT_EXPLICIT_ARGUMENT`, because no historical command line containing
`-a gaff2` was recovered. AmberTools package metadata reports 23.3, while the
archived Antechamber banner reports 22.0.

## Charges and custom modifications

- Polymer: ACPYPE gas-charge route with requested net charge zero. Endpoint Br
  handling, geometry repair, and direct/fallback topology logic are implemented
  in `.../static_cne0_release/workflow/phase_scripts/gromacs_new_phase_atomtyping.py`.
- TFSI: ACPYPE supplies bonded/atom-type parameters, after which atomic charges
  are replaced by the archived canonical `lammps_fq07` values summing to
  `-0.7 e`. The modified charges are visible in representative `tfsi_clean.itp`.
- Li: monatomic local topology labeled `forced amber99sb-ildn Li fallback` with
  mass 6.94100, sigma 0.202590 nm, and epsilon 0.0765672 kJ/mol. Production
  charge is `+0.7 e`; values are defined by `LI_AMBER99SB_PARAMS` in the same
  atom-typing script and preserved in representative `li_GMX.itp`/`li_clean.itp`.
- Ionic scaling: Li `+1 -> +0.7 e` and TFSI `-1 -> -0.7 e`; the exact structured
  record is `.../static_cne0_release/inputs/charge_scaling_parameters.csv`.
- Neutrality: 54 of 60 generated systems were neutral as built. Six used
  polymer-chain charge repair, then reran topology and charge sanity checks.
- Topology route: 47 direct ACPYPE and 13 trimer/tetramer bonded fallback.

The archived repository does not contain a specific upstream file citation for
the Li fallback values. That limitation remains unresolved. The complete
fallback procedure and a real example are in
`.../static_cne0_release/FALLBACK_TOPOLOGY_PROCEDURE.md` and
`.../static_cne0_release/examples/fallback_Traj_912122/`.

