#!/usr/bin/env python3
"""Fetch and verify the official EPIC garage map linked by its authors."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

URL = "https://drive.usercontent.google.com/download?id=1nvXdB-uCoQqOKGD0TCK1jATvwqg9Nckq&export=download&confirm=t"
SHA256 = "1f87e77d3e07395426ce216660ecd26473be45f42240e13e147490eb32c88bca"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "src/MARSIM/map_generator/resource/garage.pcd"
        ),
    )
    a = p.parse_args()
    target = Path(a.output).resolve()
    if target.exists():
        if digest(target) != SHA256:
            p.error(
                "Existing file differs from the verified official map; choose a new output"
            )
        print("Verified existing official garage.pcd")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target.parent, prefix=".garage-download-"
    ) as tmp:
        candidate = Path(tmp) / "garage.pcd"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
                URL,
                "--output",
                str(candidate),
            ],
            check=True,
        )
        if digest(candidate) != SHA256:
            raise RuntimeError(
                "Download did not match the verified official map SHA-256"
            )
        # Hard-link publishes a complete verified file without replacing existing data.
        os.link(candidate, target)
    print("Downloaded and verified " + str(target))


if __name__ == "__main__":
    main()
