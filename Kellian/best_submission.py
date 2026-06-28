from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


EXPECTED_SHA256 = "778140DF46741F9EBC7BC5EC6B4F6FB50191203FBBB43851BE39FBE2101A8700"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the preserved ~0.84 submission artifact.")
    parser.add_argument("--backup", type=Path, default=Path("Kellian/submission_084_backup.csv"))
    parser.add_argument("--out", type=Path, default=Path("Kellian/submission.csv"))
    args = parser.parse_args()

    if not args.backup.exists():
        raise SystemExit(f"Missing backup: {args.backup}")
    backup_hash = sha256(args.backup)
    if backup_hash != EXPECTED_SHA256:
        raise SystemExit(f"Backup hash mismatch: {backup_hash}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.backup, args.out)
    out_hash = sha256(args.out)
    if out_hash != EXPECTED_SHA256:
        raise SystemExit(f"Restored file hash mismatch: {out_hash}")
    print(f"Restored {args.out} ({out_hash})")


if __name__ == "__main__":
    main()
