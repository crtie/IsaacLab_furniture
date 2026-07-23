# Sharpawave asset installation

Binary Sharpawave assets are not stored in Git. The expected directory is:

```text
source/isaaclab_assets/isaaclab_assets/robots/sharpa-wave-description/
```

Install an authorized asset tree with:

```bash
python scripts/environments/install_sharpawave_assets.py \
  --source /path/to/sharpa-wave-description --mode symlink
python scripts/environments/install_sharpawave_assets.py --verify-only
```

The requirement file is
`configs/chair_assembly/sharpawave_asset_requirement.json`. Verification checks
the required member files, 93 files, total byte count and tree SHA256.

Missing assets return `MISSING_ASSET` with the expected path and installation
hint. A different tree returns `ASSET_VERSION_MISMATCH`. No Wuji asset is used
as a substitute.

After installation, run:

```bash
make asset-verify
make sharpawave-runtime
```

The binary tree's redistribution license must be confirmed separately before
including it in any distribution.
