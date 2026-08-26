# iScience Figure 1-9 regeneration QA

The exact Windows-side `iScience/02_figures/Figure_N.tiff` files and `manuscript_iScience_v5.docx` were not present in this checkout. Read-only comparisons therefore use the tracked `figures/FigN.tif` and editable `figures/FigN.svg` artifacts as the repository baselines.

| Figure | Status | Panels | Data/geometry identity | Visual layout QC | Edge clearance L/T/R/B px | Min body text / pt | TIFF pixels | PDF points | DPI | RGB/LZW | Forbidden exponent tokens |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| Figure 1 | PASS | A-C | non-text SVG geometry SHA-256 unchanged | PASS: schematic labels and connectors clear (reviewed TIFF hash matched) | 23/5/29/30 | 6.0 | 3720x1722 | 446.4x206.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 2 | PASS | A-C | non-text SVG geometry SHA-256 unchanged | PASS: axes, thresholds, and table clear (reviewed TIFF hash matched) | 48/38/44/38 | 8.0 | 3720x2588 | 446.4x310.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 3 | PASS | A-E | non-text SVG geometry SHA-256 unchanged | PASS: five-panel workflow labels clear (reviewed TIFF hash matched) | 43/41/43/42 | 8.0 | 3720x2877 | 446.4x345.3 | 600x600 | RGB/tiff_lzw | none |
| Figure 4 | PASS | A-C | non-text SVG geometry SHA-256 unchanged | PASS: equations and cross-attention labels clear (reviewed TIFF hash matched) | 33/7/33/30 | 6.0 | 3720x2439 | 446.4x292.7 | 600x600 | RGB/tiff_lzw | none |
| Figure 5 | PASS | single panel | non-text SVG geometry SHA-256 unchanged | PASS: topology bullets and workflow arrows clear (reviewed TIFF hash matched) | 29/29/29/29 | 6.5 | 3720x1261 | 446.4x151.3 | 600x600 | RGB/tiff_lzw | none |
| Figure 6 | PASS | A-D | baseline SVG key values and 14 source-code points matched | PASS: annotations clear bars; legend and titles clear (reviewed TIFF hash matched) | 162/143/106/66 | 7.0 | 3720x3480 | 446.4x417.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 7 | PASS | A-B | source CSV rows verified: OOF=6270, model selection=85 | PASS: metric box, bars, labels, and legend clear (reviewed TIFF hash matched) | 52/95/108/103 | 7.0 | 3720x1830 | 446.4x219.6 | 600x600 | RGB/tiff_lzw | none |
| Figure 8 | PASS | A-C | source CSV completed rows verified: 108 | PASS: annotations outside data axes; legend outside axes (reviewed TIFF hash matched) | 38/78/89/84 | 7.0 | 3720x2100 | 446.4x252.0 | 600x600 | RGB/tiff_lzw | none |
| Figure 9 | PASS | A-B | source CSV verified: 60 candidates, 6x10 groups, 180 replicas | PASS: legend clears points; labels inside canvas (reviewed TIFF hash matched) | 98/98/43/227 | 7.0 | 3720x1860 | 446.4x223.2 | 600x600 | RGB/tiff_lzw | none |

## Scope and limitations

- Figures 1-5 were edited from tracked SVG sources. Non-text nodes, embedded molecular images, paths, connectors, colors, and their order are byte-normalized and SHA-256 checked before export.
- Figure 6 uses cells 0-4 of the specified notebook-export script. Its 14 plotted model/condition points and key baseline labels were checked before regeneration.
- Figures 7-9 retain their existing CSV loading, validation, data coordinates, statistics, axis ranges, group order, and color mappings. Only typography, annotation formatting, layout, and export handling changed.
- Minimum body-text values exclude mathematical sub/superscripts and chemical atom labels. For plotted figures the listed minimum is the smallest intentional legend/statistical annotation size; axes and tick labels meet the requested 9 pt and 8 pt targets.
- The side-by-side PNG files under `qa/` were manually reviewed at the 6.2-inch comparison scale on 2026-08-26; no generated panel was clipped and the corrected labels, annotations, and legends do not overlap incoherently.
- Each manual visual PASS is bound to the reviewed TIFF SHA-256 in `VISUAL_QC_APPROVED_SHA256`. A rendering change therefore fails QC until the new comparison is reviewed and its digest is explicitly approved.
- This report does not claim pixel identity because typography and layout intentionally changed.
