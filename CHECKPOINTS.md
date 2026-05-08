# Checkpoint Distribution

Large learned weights are distributed as a GitHub Release asset instead of
being committed to Git history. This keeps normal clones small and avoids the
regular repository file-size limit.

## Release Asset

- Release: `v0.1.0`
- Asset: `paper-model-checkpoints-v0.1.0.tar.gz`
- URL: `https://github.com/KikETs/test/releases/download/v0.1.0/paper-model-checkpoints-v0.1.0.tar.gz`
- Size: `415524028` bytes
- SHA256: `9bd1cf640e9519e18b4919c8b1248ee7d39b854b79d0d29fb3eb6c205053ffd6`

Download and extract into the repository root:

```bash
python scripts/download_checkpoints.py
```

The archive restores files under their original repository-relative paths.

## Included

- `MY_PAPER_RELATED/MODELS/checkpoints/**/*.pth`
- `MY_PAPER_RELATED/selfies-psmiles/outputs/experiments/pi1m_endpoint_baseline_20260306_073534_UTC/checkpoints/best.pt`
- `MY_PAPER_RELATED/selfies-psmiles/outputs/experiments/pi1m_endpoint_baseline_20260306_073534_UTC/vocab.pt`

## Excluded

- `MY_PAPER_RELATED/MODELS/checkpoints/cache/`
- Regenerable cache tensors and pickle files.
- Unfinished atomistic-simulation batch pipeline artifacts.

The largest excluded cache file is larger than the current per-asset GitHub
Release limit, so it remains a regeneration artifact rather than a published
asset.

## Basis

Checked on 2026-05-08 against GitHub Docs:

- [About releases](https://docs.github.com/articles/about-releases): release
  assets are limited per file, while total release size and bandwidth are not
  capped by the release feature.
- [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github):
  regular Git repository files are subject to GitHub file-size limits.
