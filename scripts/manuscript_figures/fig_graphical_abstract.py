from __future__ import annotations

from scripts.manuscript_figures.fig_graphical_abstract_redesign import main as make_redesign


def main(display_inline: bool = False) -> dict[str, str]:
    return make_redesign(display_inline=display_inline, output_name="GraphicalAbstract")


if __name__ == "__main__":
    main(display_inline=False)
