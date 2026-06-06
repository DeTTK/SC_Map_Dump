from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
DATA_DIR = ROOT / "data"
EXPORTS_DIR = ROOT / "exports"
TMP_DIR = ROOT / "tmp"
OLMAPS_DIR = ROOT / "OLMaps"

MDATV2_TOOLS = ROOT / "tools"
MDAT_EXPORT_SCRIPT = MDATV2_TOOLS / "mdat_obj_export.py"
LOCAL_MAP_BLOCKS = ROOT / "assets" / "configs" / "assets" / "pda" / "map_blocks.json"
LOCAL_WEATHER_PALETTES = ROOT / "assets" / "configs" / "assets" / "effects" / "weather" / "palettes.json"

DEFAULT_BLENDER_EXE = Path("blender")


def ensure_dirs() -> None:
    for path in (DATA_DIR, EXPORTS_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def find_map_cache(game_dir: Path) -> Path:
    direct = [
        game_dir / "map_cache",
        game_dir / "cache" / "map_cache",
        game_dir / "game" / "map_cache",
    ]
    for candidate in direct:
        if candidate.is_dir():
            return resolve_map_cache_version(candidate)
    for candidate in game_dir.rglob("map_cache"):
        if candidate.is_dir():
            return resolve_map_cache_version(candidate)
    raise FileNotFoundError(f"map_cache not found under {game_dir}")


def resolve_map_cache_version(map_cache: Path) -> Path:
    versions = []
    for child in map_cache.iterdir():
        if not child.is_dir():
            continue
        try:
            version = tuple(int(part) for part in child.name.split("."))
        except ValueError:
            continue
        versions.append((version, child.stat().st_mtime, child))
    if versions:
        return sorted(versions, reverse=True)[0][2]
    return map_cache


def game_texture_paths(game_dir: Path) -> dict[str, Path]:
    assets = game_dir / "modassets" / "assets"
    return {
        "texarr": assets / "stalcraft" / "textures" / "blockMap.texarr",
        "ctm_dir": assets / "stalcraft" / "ctmpatcher" / "ctm",
        "map_blocks": LOCAL_MAP_BLOCKS,
        "weather_palettes": LOCAL_WEATHER_PALETTES,
    }


def find_blender(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("BLENDER_EXE")
    if env:
        candidates.append(Path(env))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    candidates.extend([
        Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"),
        DEFAULT_BLENDER_EXE,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else DEFAULT_BLENDER_EXE
