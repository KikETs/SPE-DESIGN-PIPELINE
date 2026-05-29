#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def _jsonify(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _run(cmd: list[str], *, cwd: Optional[Path] = None, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


@dataclass
class GKInputs:
    run_dir: Path
    tpr: Path
    xtc: Path
    trr: Optional[Path]
    output_dir: Path
    group: str
    begin_ps: float
    end_ps: Optional[float]
    sample_dt_ps: float
    temperature_k: float


def _default_paths(run_dir: Path) -> tuple[Path, Path, Optional[Path]]:
    prod_dir = run_dir / "md" / "production"
    tpr = prod_dir / "production.tpr"
    xtc = prod_dir / "production.xtc"
    trr = prod_dir / "production.trr"
    return tpr, xtc, trr if trr.exists() else None


def _resolve_inputs(args: argparse.Namespace) -> GKInputs:
    run_dir = Path(args.run_dir).expanduser().resolve()
    tpr, xtc, trr = _default_paths(run_dir)
    if args.tpr:
        tpr = Path(args.tpr).expanduser().resolve()
    if args.xtc:
        xtc = Path(args.xtc).expanduser().resolve()
    if args.trr:
        trr = Path(args.trr).expanduser().resolve()

    missing = [str(p) for p in [tpr, xtc] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required GK input files: {missing}")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "analysis_gk"
    out_dir.mkdir(parents=True, exist_ok=True)
    return GKInputs(
        run_dir=run_dir,
        tpr=tpr,
        xtc=xtc,
        trr=trr if trr and trr.exists() else None,
        output_dir=out_dir,
        group=args.group,
        begin_ps=float(args.begin_ns) * 1000.0,
        end_ps=None if args.end_ns is None else float(args.end_ns) * 1000.0,
        sample_dt_ps=float(args.sample_dt_ps),
        temperature_k=float(args.temperature_k),
    )


def _build_current_cmd(inp: GKInputs, *, traj: Path, use_caf: bool, prefix: str) -> list[str]:
    cmd = [
        shutil.which("gmx") or "gmx",
        "current",
        "-s",
        str(inp.tpr),
        "-f",
        str(traj),
        "-b",
        f"{inp.begin_ps:g}",
        "-dt",
        f"{inp.sample_dt_ps:g}",
        "-temp",
        f"{inp.temperature_k:g}",
        "-nojump",
        "-xvg",
        "none",
        "-o",
        str(inp.output_dir / f"{prefix}_current.xvg"),
        "-dsp",
        str(inp.output_dir / f"{prefix}_dsp.xvg"),
        "-md",
        str(inp.output_dir / f"{prefix}_md.xvg"),
        "-mj",
        str(inp.output_dir / f"{prefix}_mj.xvg"),
    ]
    if inp.end_ps is not None:
        cmd.extend(["-e", f"{inp.end_ps:g}"])
    if use_caf:
        cmd.extend(
            [
                "-caf",
                str(inp.output_dir / f"{prefix}_caf.xvg"),
                "-mc",
                str(inp.output_dir / f"{prefix}_mc.xvg"),
            ]
        )
    return cmd


def _run_mode(inp: GKInputs, *, mode: str) -> dict:
    if mode == "acf":
        if inp.trr is None:
            raise FileNotFoundError("ACF mode requires production.trr with saved velocities.")
        traj = inp.trr
        use_caf = True
    else:
        traj = inp.trr if (mode == "eh" and inp.trr is not None) else inp.xtc
        use_caf = False

    cmd = _build_current_cmd(inp, traj=traj, use_caf=use_caf, prefix=mode)
    proc = _run(cmd, cwd=inp.output_dir, input_text=f"{inp.group}\n")
    log_path = inp.output_dir / f"{mode}_gmx_current.log"
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(cmd),
                "",
                "STDOUT",
                proc.stdout or "",
                "",
                "STDERR",
                proc.stderr or "",
            ]
        )
    )
    return {
        "mode": mode,
        "returncode": int(proc.returncode),
        "trajectory": str(traj),
        "used_caf": bool(use_caf),
        "log_path": str(log_path),
        "outputs": {
            "current_xvg": str(inp.output_dir / f"{mode}_current.xvg"),
            "dsp_xvg": str(inp.output_dir / f"{mode}_dsp.xvg"),
            "md_xvg": str(inp.output_dir / f"{mode}_md.xvg"),
            "mj_xvg": str(inp.output_dir / f"{mode}_mj.xvg"),
            "caf_xvg": str(inp.output_dir / f"{mode}_caf.xvg") if use_caf else None,
            "mc_xvg": str(inp.output_dir / f"{mode}_mc.xvg") if use_caf else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Green-Kubo / Einstein-Helfand conductivity post-analysis outside the notebook."
    )
    ap.add_argument("--run-dir", required=True, help="Trajectory run directory, e.g. .../runs/Traj_14748")
    ap.add_argument("--tpr", help="Override production.tpr path")
    ap.add_argument("--xtc", help="Override production.xtc path")
    ap.add_argument("--trr", help="Override production.trr path")
    ap.add_argument("--output-dir", help="Output directory; default: <run_dir>/analysis_gk")
    ap.add_argument("--group", default="System", help="Neutral index group for gmx current")
    ap.add_argument("--begin-ns", type=float, default=0.0, help="Begin time in ns")
    ap.add_argument("--end-ns", type=float, help="End time in ns")
    ap.add_argument("--sample-dt-ps", type=float, default=1.0, help="Sampling interval passed to gmx current")
    ap.add_argument("--temperature-k", type=float, default=353.0, help="Temperature for GK fit metadata")
    ap.add_argument(
        "--mode",
        choices=["eh", "acf", "both"],
        default="eh",
        help="eh: Einstein-Helfand style using nojump trajectory, acf: current ACF with velocities, both: run both if TRR exists",
    )
    args = ap.parse_args()

    inp = _resolve_inputs(args)
    summary = {
        "inputs": _jsonify(asdict(inp)),
        "modes": [],
    }

    modes = ["eh", "acf"] if args.mode == "both" else [args.mode]
    for mode in modes:
        try:
            summary["modes"].append(_run_mode(inp, mode=mode))
        except Exception as exc:
            summary["modes"].append(
                {
                    "mode": mode,
                    "returncode": -1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_path = inp.output_dir / "gk_summary.json"
    summary_path.write_text(json.dumps(_jsonify(summary), ensure_ascii=False, indent=2))
    print(f"Saved GK summary: {summary_path}")

    if any(int(m.get("returncode", 1)) != 0 for m in summary["modes"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
