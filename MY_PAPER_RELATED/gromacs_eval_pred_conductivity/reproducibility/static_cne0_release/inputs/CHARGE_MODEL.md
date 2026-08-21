# Charge model and neutrality handling

- Li charge: formal `+1 e` multiplied by `0.7`, giving `+0.7 e` per ion.
- TFSI charge: formal `-1 e` multiplied by `0.7`, giving `-0.7 e` per ion.
- Each system contains 100 Li and 100 TFSI, so their totals cancel at `+70 e` and `-70 e`.
- Polymer charges originate from the ACPYPE gas-charge topology.
- A system is accepted as built when the full topology is neutral within the configured numerical tolerance.
- When polymer charge assignment leaves a nonzero integer-like system charge, the polymer atomic charges are repaired to make each polymer chain neutral; topology and charge sanity checks are then rerun before MD.

Across the 60 generated candidates, 54 systems were neutral as built and 6 were neutralized after polymer charge repair. `examples/charge_repair_Traj_911758/` provides a representative repaired polymer topology and its before/after structured diagnostics.
