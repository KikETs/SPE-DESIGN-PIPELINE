from __future__ import annotations

from scripts.figures.make_fig8_reference_stratified_static_cne0 import main as make_fig8
from scripts.manuscript_figures.common import display_or_print, require_outputs


def main(display_inline: bool = False) -> dict[str, str]:
    make_fig8()
    row = require_outputs("Fig8")
    return display_or_print(row, display_inline=display_inline)


if __name__ == "__main__":
    main(display_inline=False)

