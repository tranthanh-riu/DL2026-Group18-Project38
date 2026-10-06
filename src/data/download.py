"""Download the CWRU Bearing files listed in data/manifest.csv.

Every file is fetched from the official CWRU Bearing Data Center, checked by
actually loading its drive-end signal (truncated downloads are common), and its
SHA-256 is written to data/raw/SHA256SUMS so the exact data version can be
verified later.

Usage:
    python -m src.data.download                # download missing files
    python -m src.data.download --verify-only  # only check files already on disk
    python -m src.data.download --force        # re-download everything
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import scipy.io as sio

from src.utils import load_config, resolve


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path, key: str) -> tuple[bool, int]:
    """A file is valid if scipy can read the expected drive-end signal."""
    try:
        sig = sio.loadmat(path, variable_names=[key])[key]
        return True, int(sig.size)
    except Exception:
        return False, 0


def fetch(url: str, dest, retries: int = 4, timeout: int = 180) -> None:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:  # network errors, HTTP errors
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)["data"]
    raw_dir = resolve(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(cfg["manifest"]))

    records, failed = [], []
    for row in manifest.itertuples():
        dest = raw_dir / f"{row.file_id}.mat"
        url = f"{cfg['base_url']}/{row.file_id}.mat"
        ok, n = (is_valid(dest, row.signal_key) if dest.exists() else (False, 0))

        if not args.verify_only and (args.force or not ok):
            for attempt in range(3):  # re-download if the file arrives truncated
                print(f"downloading {url}")
                fetch(url, dest)
                ok, n = is_valid(dest, row.signal_key)
                if ok:
                    break
                print(f"  {dest.name} is corrupted/truncated, retrying")

        status = "ok" if ok else "MISSING_OR_CORRUPT"
        print(f"{row.file_id:>4}  {row.fault_type:<10} {row.load_hp} HP  {n:>7} samples  {status}")
        if not ok:
            failed.append(row.file_id)
            continue
        records.append({"file_id": int(row.file_id), "url": url, "samples": n,
                        "bytes": dest.stat().st_size, "sha256": sha256(dest)})

    with open(raw_dir / "SHA256SUMS", "w") as f:
        for r in records:
            f.write(f"{r['sha256']}  {r['file_id']}.mat\n")
    log = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": "Case Western Reserve University Bearing Data Center",
           "files": records}
    with open(raw_dir / "download_log.json", "w") as f:
        json.dump(log, f, indent=2)

    if failed:
        raise SystemExit(f"{len(failed)} file(s) missing or corrupt: {failed}")
    print(f"all {len(records)} files valid; checksums in {raw_dir / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
