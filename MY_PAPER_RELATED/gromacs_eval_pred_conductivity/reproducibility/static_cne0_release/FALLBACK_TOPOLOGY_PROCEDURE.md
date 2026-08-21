# Polymer bonded-topology fallback

The fallback is entered only when direct polymer ACPYPE fails with an antechamber atom-typing signature. It is not an atoms-only substitution.

1. Build trimer and tetramer probe oligomers with PySoftK/RDKit, using deterministic retry configurations for geometry repair and local optimization.
2. Run ACPYPE independently on the neutral trimer and tetramer using gas-phase charges.
3. Infer the repeat-unit insertion block from the trimer/tetramer atom correspondence.
4. Expand the typed atom block to the requested full-chain DP and rebalance the polymer total charge to zero.
5. Reconstruct the full-chain molecular graph from `chain.mol`.
6. Build bonded-parameter lookups from the ACPYPE trimer and tetramer ITP files.
7. Generate full-chain bonds, pairs, angles, proper dihedrals, and improper dihedrals from that graph and the parameter lookups.
8. Write `polymer_trimer_fallback_full_GMX.itp`; the run proceeds only when full bonded reconstruction succeeds.

`examples/fallback_Traj_912122/` contains a representative full bonded fallback ITP, the cleaned production ITP, atom types, topology, and structured charge diagnostics. The complete implementation and retry settings are in `workflow/phase_scripts/gromacs_new_phase_atomtyping.py`.
