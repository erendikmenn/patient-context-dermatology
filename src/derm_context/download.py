"""Download immutable challenge files from the official ISIC download links."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2024/"
FILES = {
    "input.zip": "ISIC_2024_Training_Input.zip",
    "supplement.csv": "ISIC_2024_Training_Supplement.csv",
    "ground_truth.csv": "ISIC_2024_Training_GroundTruth.csv",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {
        "dataset": "SLICE-3D 2024 full training set",
        "license": "CC-BY-NC-4.0",
        "source": "https://challenge.isic-archive.com/data/",
        "doi": "https://doi.org/10.34970/2024-slice-3d",
        "files": {},
    }
    for local_name, remote_name in FILES.items():
        destination = raw_dir / local_name
        old = previous.get("files", {}).get(local_name, {})
        if destination.exists() and old.get("sha256") == sha256_file(destination):
            print(f"Verified existing {local_name}", flush=True)
            manifest["files"][local_name] = old
            continue
        url = BASE_URL + remote_name
        partial = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        request = Request(url, headers={"User-Agent": "derm-context-research/0.1"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as target:
            expected = int(response.headers.get("Content-Length", 0))
            total = 0
            last = time.monotonic()
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                target.write(chunk)
                total += len(chunk)
                if time.monotonic() - last > 15:
                    print(f"  {total / 1e6:.1f} / {expected / 1e6:.1f} MB", flush=True)
                    last = time.monotonic()
            if expected and total != expected:
                raise RuntimeError(f"Incomplete download: {local_name}: {total} != {expected}")
        partial.replace(destination)
        manifest["files"][local_name] = {
            "url": url,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {local_name}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    download(parser.parse_args().raw_dir)
