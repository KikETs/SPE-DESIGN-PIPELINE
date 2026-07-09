from __future__ import annotations

from scripts.figures.make_fig7_surrogate_reliability import main as make_fig7
from scripts.manuscript_figures.common import display_or_print, require_outputs


def main(display_inline: bool = False) -> dict[str, str]:
    make_fig7()
    row = require_outputs("Fig7")
    return display_or_print(row, display_inline=display_inline)


if __name__ == "__main__":
    main(display_inline=False)

