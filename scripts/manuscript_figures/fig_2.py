from __future__ import annotations

from scripts.manuscript_figures.fig_2_hybrid import main as build_hybrid_fig2


def main(display_inline: bool = False) -> dict[str, str]:
    return build_hybrid_fig2(display_inline=display_inline)


if __name__ == "__main__":
    main(display_inline=False)
