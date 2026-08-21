#!/usr/bin/env python3
"""Create/resume a Zenodo draft and upload the verified representative MD bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
STATIC = (
    ROOT
    / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release"
)
DEFAULT_MANIFEST = STATIC / "external_data/representative_trajectory_files.csv"
DEFAULT_METADATA = ROOT / "reproducibility/zenodo/metadata.json"
DEFAULT_STATE = ROOT / "reproducibility/zenodo/zenodo_draft_state.json"


SOURCE_FILES = {
    "generated_Traj_912118_rep3_50ns.xtc": (
        ROOT
        / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/runs/Traj_912118"
        / "md/production_rep3/production_rep3.xtc"
    ),
    "generated_Traj_912118_rep3.tpr": (
        STATIC
        / "examples/generated_Traj_912118_rep3/representative_structure/production_rep3.tpr"
    ),
    "generated_Traj_912118_rep3_final.gro": (
        STATIC
        / "examples/generated_Traj_912118_rep3/representative_structure/production_rep3.gro"
    ),
    "generated_Traj_912118_rep3_production.mdp": (
        STATIC / "examples/generated_Traj_912118_rep3/mdp/production_rep3.mdp"
    ),
    "generated_Traj_912118_rep3_index.ndx": (
        STATIC / "examples/generated_Traj_912118_rep3/representative_structure/index.ndx"
    ),
    "reference_reassessment_Traj_13430_rep1_50ns.xtc": (
        Path("/home/user/\ubc14\ud0d5\ud654\uba74/DL/gromacs/eval_top10_bottom10_stratified100")
        / "runs/Traj_13430/md/production/production.xtc"
    ),
    "reference_reassessment_Traj_13430_rep1.tpr": (
        STATIC / "examples/reference_Traj_13430/representative_structure/production.tpr"
    ),
    "reference_reassessment_Traj_13430_rep1_final.gro": (
        STATIC / "examples/reference_Traj_13430/representative_structure/production.gro"
    ),
    "reference_reassessment_Traj_13430_rep1_production.mdp": (
        STATIC / "examples/reference_Traj_13430/mdp/production.mdp"
    ),
    "reference_reassessment_Traj_13430_rep1_index.ndx": (
        STATIC / "examples/reference_Traj_13430/representative_structure/index.ndx"
    ),
}


@dataclass(frozen=True)
class UploadFile:
    dataset: str
    archive_filename: str
    source_path: Path
    size_bytes: int
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_files(manifest: Path, datasets: set[str]) -> list[UploadFile]:
    files: list[UploadFile] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = str(row["dataset"])
            if dataset not in datasets:
                continue
            filename = str(row["archive_filename"])
            source = SOURCE_FILES.get(filename)
            if source is None:
                raise RuntimeError(f"No source mapping for {filename}")
            files.append(
                UploadFile(
                    dataset=dataset,
                    archive_filename=filename,
                    source_path=source,
                    size_bytes=int(row["size_bytes"]),
                    sha256=str(row["sha256"]),
                )
            )
    if not files:
        raise RuntimeError(f"No manifest rows selected for datasets={sorted(datasets)}")
    return files


def preflight(files: list[UploadFile]) -> None:
    total = sum(item.size_bytes for item in files)
    print(f"[preflight] files={len(files)} bytes={total}", flush=True)
    for index, item in enumerate(files, start=1):
        path = item.source_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing source: {path}")
        actual_size = path.stat().st_size
        if actual_size != item.size_bytes:
            raise RuntimeError(
                f"Size mismatch for {item.archive_filename}: "
                f"expected={item.size_bytes} actual={actual_size}"
            )
        print(
            f"[preflight] sha256 {index}/{len(files)} {item.archive_filename}",
            flush=True,
        )
        actual_sha = file_sha256(path)
        if actual_sha != item.sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {item.archive_filename}: "
                f"expected={item.sha256} actual={actual_sha}"
            )
    print("[preflight] all files match manifest", flush=True)


class ZenodoClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        headers = dict(self.headers)
        headers.update(kwargs.pop("headers", {}))
        response = requests.request(method, url, headers=headers, **kwargs)
        if not response.ok:
            raise RuntimeError(
                f"Zenodo API {method} {url} failed: "
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )
        return response

    def create_draft(self, metadata: dict) -> dict:
        url = f"{self.base_url}/api/deposit/depositions"
        return self.request(
            "POST",
            url,
            headers={"Content-Type": "application/json"},
            json={"metadata": metadata},
            timeout=60,
        ).json()

    def get_draft(self, deposition_id: int) -> dict:
        url = f"{self.base_url}/api/deposit/depositions/{deposition_id}"
        return self.request("GET", url, timeout=60).json()

    def upload(self, bucket_url: str, item: UploadFile) -> dict:
        url = f"{bucket_url.rstrip('/')}/{quote(item.archive_filename)}"
        with item.source_path.open("rb") as handle:
            return self.request(
                "PUT",
                url,
                data=handle,
                headers={"Content-Type": "application/octet-stream"},
                timeout=(30, 12 * 60 * 60),
            ).json()


def existing_files(draft: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in draft.get("files", []):
        name = item.get("filename") or item.get("key")
        size = item.get("filesize") if item.get("filesize") is not None else item.get("size")
        if name is not None and size is not None:
            result[str(name)] = int(size)
    return result


def write_state(path: Path, draft: dict, selected: list[UploadFile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "deposition_id": int(draft["id"]),
        "state": draft.get("state"),
        "submitted": bool(draft.get("submitted", False)),
        "html": draft.get("links", {}).get("html"),
        "latest_draft_html": draft.get("links", {}).get("latest_draft_html"),
        "selected_files": [item.archive_filename for item in selected],
        "uploaded_files": existing_files(draft),
        "publish_called": False,
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("generated", "reference_reassessment"),
        help="Dataset to upload; repeat for both. Default: generated only.",
    )
    parser.add_argument("--deposition-id", type=int, help="Resume an existing draft.")
    parser.add_argument(
        "--create-draft",
        action="store_true",
        help="Create a new draft. Mutually exclusive with --deposition-id.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZENODO_BASE_URL", "https://zenodo.org"),
    )
    parser.add_argument("--token-env", default="ZENODO_ACCESS_TOKEN")
    args = parser.parse_args()
    if args.create_draft and args.deposition_id is not None:
        parser.error("Use either --create-draft or --deposition-id, not both")
    if not args.preflight_only and not args.create_draft and args.deposition_id is None:
        parser.error("Upload requires --create-draft or --deposition-id")
    return args


def main() -> int:
    args = parse_args()
    datasets = set(args.dataset or ["generated"])
    files = load_files(args.manifest.resolve(), datasets)
    preflight(files)
    if args.preflight_only:
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Environment variable {args.token_env} is not set")
    client = ZenodoClient(args.base_url, token)
    if args.create_draft:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        draft = client.create_draft(metadata)
        print(f"[draft] created deposition_id={draft['id']}", flush=True)
        write_state(args.state_file, draft, files)
    else:
        draft = client.get_draft(args.deposition_id)
        print(f"[draft] resumed deposition_id={draft['id']}", flush=True)

    if draft.get("submitted"):
        raise RuntimeError("Refusing upload: deposition is already published")
    bucket_url = draft.get("links", {}).get("bucket")
    if not bucket_url:
        raise RuntimeError("Draft response does not contain a bucket URL")

    for index, item in enumerate(files, start=1):
        current = existing_files(draft)
        current_size = current.get(item.archive_filename)
        if current_size == item.size_bytes:
            print(f"[upload] skip existing {item.archive_filename}", flush=True)
            continue
        if current_size is not None:
            raise RuntimeError(
                f"Existing remote file has wrong size: {item.archive_filename} "
                f"expected={item.size_bytes} remote={current_size}"
            )
        print(
            f"[upload] {index}/{len(files)} {item.archive_filename} "
            f"bytes={item.size_bytes}",
            flush=True,
        )
        response = client.upload(bucket_url, item)
        remote_size = response.get("size")
        if remote_size is not None and int(remote_size) != item.size_bytes:
            raise RuntimeError(
                f"Remote size mismatch after upload for {item.archive_filename}: "
                f"expected={item.size_bytes} remote={remote_size}"
            )
        draft = client.get_draft(int(draft["id"]))
        write_state(args.state_file, draft, files)

    draft = client.get_draft(int(draft["id"]))
    uploaded = existing_files(draft)
    for item in files:
        if uploaded.get(item.archive_filename) != item.size_bytes:
            raise RuntimeError(f"Remote verification failed for {item.archive_filename}")
    write_state(args.state_file, draft, files)
    print(f"[done] draft={draft['id']} files_verified={len(files)} publish_called=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; resume with --deposition-id from the state file.", file=sys.stderr)
        raise SystemExit(130)
