"""Record the local control implementation without collecting host identifiers."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = ["parm/train.py", "parm/seeding.py", "parm/data/datasets.py",
         "parm/data/masking.py", "parm/data/download.py", "parm/metrics.py",
         "parm/models/backbones/fusion.py", "requirements-controls-local.txt"]


def main():
    destination = ROOT / "paper/data/control_implementation.json"
    record = dict(
        files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in FILES},
        git_base=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip(),
        description="File hashes identify the working implementation; Git base alone "
                    "does not include the uncommitted control changes.")
    if destination.exists():
        # A later commit or a release checkout may have a different Git base
        # while containing exactly the implementation that produced the runs.
        if json.loads(destination.read_text())["files"] != record["files"]:
            raise RuntimeError("Control implementation differs from recorded provenance")
        print("Control implementation hashes verified")
        return
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print("Recorded control implementation hashes")


if __name__ == "__main__":
    main()
