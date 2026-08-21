# Manuscript Figure Specs

These JSON files are parsed, coordinate-preserving specs exported from the
PowerPoint SVG files in `manuscript_figure/`.

Runtime figure generation for Fig1, Fig3, Fig4, Fig5, and GraphicalAbstract
uses these specs directly and does not require the source `.SVG` files.

To regenerate the specs after editing the source SVG slides, run:

```bash
python -m scripts.manuscript_figures.export_svg_specs
python -m scripts.manuscript_figures.apply_spec_edits
```

`apply_spec_edits` reapplies manuscript-specific adjustments such as the Fig1
label color/weight edits and the Graphical Abstract Fig9B embedded panel.

If `CAMBRIA.TTC` is available in the repository root or its parent directory,
the plotting code extracts the `Cambria Math` face to
`assets/manuscript_figures/fonts/CambriaMath.ttf` and uses it for Cambria Math
SVG text.
