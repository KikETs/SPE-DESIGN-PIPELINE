# Representative topology index

Use the topology folders in the canonical static release:

- `examples/generated_Traj_912118_rep3/topology/`
- `examples/reference_Traj_13430/topology/`
- `examples/fallback_Traj_912122/`
- `examples/charge_repair_Traj_911758/`

They preserve `topol.top`, merged atom types, polymer, Li, and TFSI ITPs,
position restraints where applicable, and structured charge diagnostics. The
generated example is direct ACPYPE; `Traj_912122` demonstrates the bonded
fallback; `Traj_911758` demonstrates polymer-charge repair.

See `FORCE_FIELD_PROVENANCE.md` before reusing parameters.

