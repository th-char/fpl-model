"""Local file cache for raw downloaded data."""

import shutil
from pathlib import Path


class FileCache:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _path(self, source: str, season: str, filename: str) -> Path:
        return self.base_dir / source / season / filename

    def has(self, source: str, season: str, filename: str) -> bool:
        return self._path(source, season, filename).exists()

    def get(self, source: str, season: str, filename: str) -> bytes | None:
        path = self._path(source, season, filename)
        if path.exists():
            return path.read_bytes()
        return None

    def put(self, source: str, season: str, filename: str, content: bytes) -> None:
        path = self._path(source, season, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def clear(self, source: str, season: str | None = None) -> None:
        if season:
            target = self.base_dir / source / season
        else:
            target = self.base_dir / source
        if target.exists():
            shutil.rmtree(target)
