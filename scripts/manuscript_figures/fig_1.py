from __future__ import annotations

from scripts.manuscript_figures.svg_rebuilder import rebuild_spec_figure


def main(display_inline: bool = False) -> dict[str, str]:
    return rebuild_spec_figure("fig1", "Fig1", display_inline=display_inline)


if __name__ == "__main__":
    main(display_inline=False)
