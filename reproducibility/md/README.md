# Molecular-dynamics reproducibility index

The full compact source bundle is
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/`.
This index deliberately avoids duplicating its 160 tracked files.

## Generated reassessment

- 60 candidates in six archived `HIGH_*`/`LOW_*` selection groups.
- Three production replicas per candidate, 180 replica result records.
- 50 ns at 353 K.
- Archived generated representative MDP: 1 fs production timestep.
- Saved spacing in results: 120 replicas at 0.5 ps and 60 at 1.0 ps; all are
  analyzed on a 1 ps grid.

The requested TT/LT group names are not substituted here because no explicit
TT/LT-to-HIGH/LOW mapping is archived.

## Reference reassessment

- 120 selected systems; 108 completed and 12 failed.
- One 50 ns production trajectory per completed system.
- Archived reference representative MDP: 2 fs timestep and 1 ps output.

## Manuscript static cNE0 protocol

The released manuscript protocol is Li-TFSI oxygen RDF first minimum, at least
two simultaneous oxygen contacts per Li-TFSI pair, a static association graph
without a persistence filter, maximum cluster size 101, and formal cluster
charge `z_ij = i - j`. Ionic charges are scaled to `+0.7/-0.7`.

`rdf_peak_tau0_replica_eval.py` is a separately labeled sensitivity analysis and
must not be substituted for the manuscript source.

