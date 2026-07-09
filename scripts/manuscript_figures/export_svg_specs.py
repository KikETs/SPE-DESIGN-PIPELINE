from __future__ import annotations

from scripts.manuscript_figures.svg_rebuilder import write_slide_spec


SPECS = {
    "fig1": 1,
    "fig3": 3,
    "fig4": 4,
    "fig5": 5,
    "graphical_abstract": 10,
}


def main() -> int:
    for spec_name, slide_number in SPECS.items():
        out_path = write_slide_spec(slide_number, spec_name)
        print(f"{spec_name}: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
