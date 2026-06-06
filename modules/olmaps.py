from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import OLMAPS_DIR

REGION_BLOCKS = 512
REGION_PNG_RE = re.compile(r"^r\.(-?\d+)\.(-?\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class OLMap:
    name: str
    root: Path

    @property
    def cache_dir(self) -> Path:
        return self.root / "_cache"

    def iter_region_files(self) -> list[tuple[int, int, Path]]:
        rows: list[tuple[int, int, Path]] = []
        if not self.cache_dir.is_dir():
            return rows
        for child in self.cache_dir.iterdir():
            if not child.is_file():
                continue
            match = REGION_PNG_RE.match(child.name)
            if match:
                rows.append((int(match.group(1)), int(match.group(2)), child))
        return sorted(rows)

    def to_dict(self) -> dict:
        regions = self.iter_region_files()
        return {"name": self.name, "regions": len(regions)}


def list_olmaps(root: Path = OLMAPS_DIR) -> list[OLMap]:
    if not root.is_dir():
        return []
    maps: list[OLMap] = []
    for child in sorted(root.iterdir()):
        candidate = OLMap(child.name, child)
        if child.is_dir() and candidate.iter_region_files():
            maps.append(candidate)
    return maps


def get_olmap(name: str, root: Path = OLMAPS_DIR) -> Optional[OLMap]:
    candidate = root / name
    if not candidate.is_dir():
        return None
    olmap = OLMap(name, candidate)
    if not olmap.cache_dir.is_dir():
        return None
    return olmap


class TileCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mem: dict[tuple[str, int, int], tuple[float, bytes]] = {}

    def get(self, olmap: OLMap, rx: int, rz: int) -> Optional[bytes]:
        cache_path = olmap.cache_dir / f"r.{rx}.{rz}.png"
        if not cache_path.is_file():
            return None
        mtime = cache_path.stat().st_mtime
        key = (olmap.name, rx, rz)
        with self._lock:
            cached = self._mem.get(key)
            if cached and cached[0] >= mtime:
                return cached[1]
        data = cache_path.read_bytes()
        with self._lock:
            self._mem[key] = (mtime, data)
        return data
