# MD input index

Original inputs are retained without copies under the static cNE0 release:

- Generated representative: `examples/generated_Traj_912118_rep3/`
- Reference representative: `examples/reference_Traj_13430/`
- Bonded fallback: `examples/fallback_Traj_912122/`
- Charge repair: `examples/charge_repair_Traj_911758/`

The generated/reference examples contain full MDP stage sets, Packmol inputs
for seeds 456789/457789/458789, construction PDBs, topology diagnostics, and
representative production structures. The pipeline scripts that created them
are in `static_cne0_release/workflow/`.

Candidate-level DP, composition, topology route, neutrality status, and ion
counts are in
`static_cne0_release/data/generated/generated_candidate_composition_60.csv`.

Vendored PySoftK upstream commit, license, and the single locally modified file
are documented in `PYSOFTK_PROVENANCE.md`.
