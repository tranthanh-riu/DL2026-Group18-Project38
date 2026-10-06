from pathlib import Path
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR.parent / "data" / "raw" / "CWRU"


FILES = [
    "97.mat",
    "98.mat",
    "99.mat",
    "100.mat",
    "105.mat",
    "106.mat",
    "107.mat",
    "108.mat",
    "118.mat",
    "119.mat",
    "120.mat",
    "121.mat",
    "130.mat",
    "131.mat",
    "132.mat",
    "133.mat",
]


BASE_URL = "https://engineering.case.edu/sites/default/files"


def download_file(filename):
    output_path = RAW_DIR / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skip: {filename}")
        return

    url = f"{BASE_URL}/{filename}"

    print(f"Downloading: {filename}")

    with urlopen(url, timeout=60) as response:
        data = response.read()

    if len(data) == 0:
        raise RuntimeError(f"Downloaded file is empty: {filename}")

    output_path.write_bytes(data)

    print(f"Saved: {output_path} ({len(data)} bytes)")


def download_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        download_file(filename)


if __name__ == "__main__":
    download_all()