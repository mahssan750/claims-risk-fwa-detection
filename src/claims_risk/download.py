import hashlib
import urllib.request
from pathlib import Path

from claims_risk.config import DATA_FILES, DATASET_RECORD, PATHS


def file_md5(file_path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(file_name: str) -> str:
    return f"https://zenodo.org/api/records/{DATASET_RECORD}/files/{file_name}/content"


def download_file(file_name: str, expected_md5: str) -> Path:
    destination = PATHS.raw / file_name
    if destination.exists() and file_md5(destination) == expected_md5:
        return destination

    # A temporary suffix prevents an interrupted transfer from looking like a valid source file.
    partial = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(download_url(file_name), partial)
    if file_md5(partial) != expected_md5:
        partial.unlink()
        raise ValueError(f"Checksum mismatch for {file_name}")
    partial.replace(destination)
    return destination


def download_dataset() -> list[Path]:
    PATHS.raw.mkdir(parents=True, exist_ok=True)
    return [download_file(name, checksum) for name, checksum in DATA_FILES.items()]
