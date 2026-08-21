# Generated-candidate MD tables

- `generated_md_replica_180.csv`: one row per production replica, preserving
  static cNE0, NE, RDF first minimum, diffusivities, protocol metadata,
  candidate structure/prediction/DP, topology route, and charge status.
- `generated_md_candidate_60.csv`: deterministic arithmetic aggregation over
  the three replica-level linear conductivity values per candidate.

Canonical inputs are the static release's
`generated_static_cNE0_replica_180.csv`, `generated_md_results_60.csv`, and
`generated_candidate_composition_60.csv`, plus the tracked selection and
weighted-prediction tables. No group-level mean replaces replica rows.

