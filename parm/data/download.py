"""Fetch MMSA pre-extracted features and verify against official SHA-256.

The canonical files live on Google Drive / BaiduYun (see thuiar/MMSA). Those
links are awkward to automate, so we pull byte-identical copies from public
HuggingFace mirrors and verify every file against the hashes published in the
MMSA README. A mirror we cannot verify is refused, not used.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# SHA-256 as published in the official MMSA README.
SPECS = {
    "MOSI": dict(
        repo="tamb2203579/CMU-MOSI",
        remote="Processed/unaligned_50.pkl",
        local="MOSI/unaligned_50.pkl",
        sha256="78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524",
    ),
    "MOSEI": dict(
        repo="tamb2203579/CMU-MOSEI",
        remote="Processed/unaligned_50.pkl",
        local="MOSEI/unaligned_50.pkl",
        sha256="ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f",
    ),
    "SIMS": dict(
        repo="tamb2203579/CH-SIMS",
        remote="Processed/unaligned_39.pkl",
        local="SIMS/unaligned_39.pkl",
        sha256="c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f",
    ),
}


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def fetch(name: str, root: Path, allow_unverified: bool = False) -> bool:
    from huggingface_hub import hf_hub_download

    spec = SPECS[name]
    dest = root / spec["local"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        got = sha256_of(dest)
        if got == spec["sha256"]:
            print(f"[{name}] already present and verified")
            return True
        print(f"[{name}] present but hash mismatch -- refetching")
        dest.unlink()

    print(f"[{name}] downloading {spec['repo']}/{spec['remote']} ...", flush=True)
    cached = hf_hub_download(repo_id=spec["repo"], filename=spec["remote"],
                             repo_type="dataset")
    shutil.copyfile(cached, dest)

    got = sha256_of(dest)
    mb = dest.stat().st_size / 1e6
    if got == spec["sha256"]:
        print(f"[{name}] OK  {mb:.1f} MB  sha256 matches official MMSA hash")
        return True

    print(f"[{name}] HASH MISMATCH  ({mb:.1f} MB)")
    print(f"  expected {spec['sha256']}")
    print(f"  got      {got}")
    if not allow_unverified:
        dest.unlink()
        print(f"[{name}] refused (use --allow-unverified to keep)")
        return False
    print(f"[{name}] kept as UNVERIFIED -- must be disclosed in the paper")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--datasets", nargs="+", default=list(SPECS))
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    results = {n: fetch(n, root, a.allow_unverified) for n in a.datasets}
    print("\n=== summary ===")
    for n, ok in results.items():
        print(f"  {n:6s} {'verified' if ok else 'FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
