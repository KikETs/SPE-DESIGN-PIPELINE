from __future__ import annotations

from scripts.manuscript_figures.plot_ipynb_runner import run_plot_ipynb_figure


def main(display_inline: bool = False) -> dict[str, str]:
    return run_plot_ipynb_figure("Fig6", display_inline=display_inline)


if __name__ == "__main__":
    main(display_inline=False)

