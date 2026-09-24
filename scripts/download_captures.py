"""Download the paper's attention captures instead of recapturing them.

    uv run python -m scripts.download_captures                        # all four models
    uv run python -m scripts.download_captures --models phi4-mini,qwen3.5-9b

Fetches one archive per model from the GitHub release, checks its SHA-256,
and unpacks it to captures/<model>/<corpus>/<split>/, the layout
scripts.capture writes. scripts.train and scripts.evaluate then run on it
unchanged. These are the eager captures behind the paper's tables.
"""

import hashlib
import pathlib
import tarfile
import urllib.request
from dataclasses import dataclass

RELEASE_URL = "https://github.com/citeprobe/citeprobe/releases/download/v1.0"


@dataclass(frozen=True)
class Asset:
    """One model's capture archive and the digest it must match."""

    model: str
    sha256: str

    @property
    def filename(self) -> str:
        return f"captures-{self.model}.tar"


ASSETS = [
    Asset("phi4-mini", "5bf6c16b77e4844583064272ab788786bd69764127490e2b4e5e0f958fa31d5c"),
    Asset("ministral-8b", "50d58c999dfe048eb6013c7546f9a1f771292702cbfc28888503340cfc1fcbc6"),
    Asset("qwen3.5-9b", "d749fea3e5efc8cbcf518382517a7c68f64574965e0292e0be395bfa7d410e8a"),
    Asset("gemma4-12b", "dc7cca5e9ae4d0843acd41b59c2f5159f24c663d04ab5b69b1094222beecb549"),
]


def main(models: str = "", directory: str = ".", keep_archives: bool = False) -> None:
    """models takes comma-separated names (all by default). Archives unpack into directory/captures/."""
    chosen = select_assets(models)
    root = pathlib.Path(directory)
    for asset in chosen:
        archive = root / asset.filename
        if not archive.exists() or digest(archive) != asset.sha256:
            download(f"{RELEASE_URL}/{asset.filename}", archive)
        if digest(archive) != asset.sha256:
            raise ValueError(f"{archive} does not match its published SHA-256; delete it and retry")
        print(f"Unpacking {archive.name} into {root / 'captures' / asset.model}", flush=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
        if not keep_archives:
            archive.unlink()


def select_assets(models: str) -> list[Asset]:
    """The assets for the requested models, rejecting unknown names."""
    if not models:
        return ASSETS
    by_model = {asset.model: asset for asset in ASSETS}
    unknown = [name for name in models.split(",") if name not in by_model]
    if unknown:
        raise ValueError(f"unknown models {unknown}; known: {sorted(by_model)}")
    return [by_model[name] for name in models.split(",")]


def download(url: str, target: pathlib.Path) -> None:
    """Stream url to target through a .part file, printing progress."""
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while chunk := response.read(1 << 20):
            file.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r  {received / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    partial.rename(target)


def digest(path: pathlib.Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
