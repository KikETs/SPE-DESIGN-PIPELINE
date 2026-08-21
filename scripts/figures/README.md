# Figure Scripts

Final figure preview is handled by `final_figures_overview.ipynb`. The notebook
contains the actual plotting cells copied from `plot.ipynb`, exports the
generated source figures to final `Fig*.svg` and `Fig*.tif` names, and displays
the resulting SVG image as notebook output.

## Main entry points

- `python scripts/figures/export_final_figures.py`
  - Collects the requested figure set into `figures/`.
  - Writes final figure names as `Fig1A`, `Fig1B`, `Fig2`, `Fig6`, `Fig7`, `Fig8`, and `Fig9`.
  - Keeps final exports to `*.svg` and `*.tif`; TIF files are saved at 600 dpi.

- `python scripts/figures/generate_figure.py Fig1A`
  - Regenerates/exports one final figure by name.
  - Use `--all` to regenerate/export the full final set.

- `python scripts/figures/export_notebook_code.py`
  - Exports code cells from `plot.ipynb` into `scripts/figures/notebook_exports/`.

- `python scripts/figures/build_actual_plot_overview.py`
  - Rebuilds `final_figures_overview.ipynb` by copying the selected plotting
    cells from `plot.ipynb` without rewriting the plotting logic.
  - Figures without a current plotting cell in `plot.ipynb` are previewed from
    existing `Fig*.svg`/`Fig*.tif` outputs rather than reconstructed.

## Final figure order

1. `figures/Fig1A.svg` and `figures/Fig1A.tif`
2. `figures/Fig1B.svg` and `figures/Fig1B.tif`
3. `figures/Fig2.svg` and `figures/Fig2.tif`
4. `figures/Fig6.svg` and `figures/Fig6.tif`
5. `figures/Fig7.svg` and `figures/Fig7.tif`
6. `figures/Fig8.svg` and `figures/Fig8.tif`
7. `figures/Fig9.svg` and `figures/Fig9.tif`
