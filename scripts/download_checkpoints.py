#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


ASSET_URL = (
    "https://github.com/KikETs/test/releases/download/v0.1.0/"
    "paper-model-checkpoints-v0.1.0.tar.gz"
)
ASSET_SHA256 = "9bd1cf640e9519e18b4919c8b1248ee7d39b854b79d0d29fb3eb6c205053ffd6"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    request = Request(url, headers={"User-Agent": "paper-repro-checkpoint-downloader"})
    with urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def safe_extract(archive: Path, dest: Path) -> None:
    base = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        supports_filter = "filter" in inspect.signature(tar.extract).parameters
        for original in tar.getmembers():
            member = copy.copy(original)
            member.name = member.name.lstrip("/")
            if member.name in {"", "."}:
                continue
            target = (dest / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"Archive member escapes destination: {original.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing to extract link member: {original.name}")
            if supports_filter:
                tar.extract(member, dest, filter="fully_trusted")
            else:
                tar.extract(member, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract release checkpoint weights.")
    parser.add_argument("--url", default=ASSET_URL, help="Checkpoint archive URL.")
    parser.add_argument("--sha256", default=ASSET_SHA256, help="Expected archive SHA256.")
    parser.add_argument("--dest", type=Path, default=repo_root(), help="Repository root to extract into.")
    parser.add_argument("--archive", type=Path, help="Use an existing local archive instead of downloading.")
    args = parser.parse_args()

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="paper-checkpoints-") as tmp:
        archive = args.archive.resolve() if args.archive else Path(tmp) / "checkpoints.tar.gz"
        if args.archive is None:
            print(f"downloading={args.url}")
            download(args.url, archive)

        actual = sha256_file(archive)
        if actual != args.sha256:
            raise RuntimeError(f"SHA256 mismatch: expected {args.sha256}, got {actual}")

        safe_extract(archive, dest)

    print(f"checkpoints_extracted_to={dest}")
    print("sha256_ok=1")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
