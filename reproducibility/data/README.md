# Consolidated data index

The CSVs in this directory are deterministic projections/joins of tracked
source tables. They do not replace canonical source data. Every generated table
is listed in `TABLE_PROVENANCE.csv` with its builder.

Run:

```bash
python scripts/build_jcim_reproducibility_package.py
```

The builder checks that its 60 candidate-level static-cNE0 arithmetic means
match the archived `generated_md_results_60.csv` values before writing.

