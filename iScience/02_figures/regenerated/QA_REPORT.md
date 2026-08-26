# iScience Figure 1-9 regeneration QA

The exact Windows-side `iScience/02_figures/Figure_N.tiff` files and `manuscript_iScience_v5.docx` were not present in this checkout. Read-only comparisons therefore use the tracked `figures/FigN.tif` and editable `figures/FigN.svg` artifacts as the repository baselines.

| Figure | Status | Panels | Data/geometry identity | Min body text / pt | TIFF pixels | PDF points | DPI | RGB/LZW | Forbidden exponent tokens |
|---|---|---|---|---:|---:|---:|---:|---|---|
| Figure 1 | PASS | A-C | non-text SVG geometry SHA-256 unchanged | 6.0 | 3720x1722 | 446.4x206.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 2 | PASS | A-C | non-text SVG geometry SHA-256 unchanged | 8.0 | 3720x2588 | 446.4x310.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 3 | PASS | A-E | non-text SVG geometry SHA-256 unchanged | 8.0 | 3720x2877 | 446.4x345.3 | 600x600 | RGB/tiff_lzw | none |
| Figure 4 | PASS | A-C | non-text SVG geometry SHA-256 unchanged | 6.0 | 3720x2439 | 446.4x292.7 | 600x600 | RGB/tiff_lzw | none |
| Figure 5 | PASS | single panel | non-text SVG geometry SHA-256 unchanged | 6.5 | 3720x1261 | 446.4x151.3 | 600x600 | RGB/tiff_lzw | none |
| Figure 6 | PASS | A-D | baseline SVG key values and 14 source-code points matched | 7.0 | 3720x3480 | 446.4x417.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 7 | PASS | A-B | source CSV rows verified: OOF=6270, model selection=85 | 7.0 | 3720x1830 | 446.4x219.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 8 | PASS | A-C | source CSV completed rows verified: 108 | 7.0 | 3720x1830 | 446.4x219.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 9 | PASS | A-B | source CSV verified: 60 candidates, 6x10 groups, 180 replicas | 7.0 | 3720x1800 | 446.4x216.0 | 600x600 | RGB/tiff_lzw | none |

## Scope and limitations

- Figures 1-5 were edited from tracked SVG sources. Non-text nodes, embedded molecular images, paths, connectors, colors, and their order are byte-normalized and SHA-256 checked before export.
- Figure 6 uses cells 0-4 of the specified notebook-export script. Its 14 plotted model/condition points and key baseline labels were checked before regeneration.
- Figures 7-9 retain their existing CSV loading, validation, data coordinates, statistics, axis ranges, group order, and color mappings. Only typography, annotation formatting, layout, and export handling changed.
- Minimum body-text values exclude mathematical sub/superscripts and chemical atom labels. For plotted figures the listed minimum is the smallest intentional legend/statistical annotation size; axes and tick labels meet the requested 9 pt and 8 pt targets.
- The side-by-side PNG files under `qa/` were manually reviewed at the 6.2-inch comparison scale; no generated panel was clipped and the corrected labels, annotations, and legends do not overlap incoherently.
- This report does not claim pixel identity because typography and layout intentionally changed.
