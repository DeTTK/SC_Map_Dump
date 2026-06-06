from __future__ import annotations

import json
import math
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SECTOR_SIZE = 4096
REGION_BITS = 5
REGION_SIZE = 1 << REGION_BITS
REGION_CHUNKS = REGION_SIZE * REGION_SIZE
HEADER_INTS = 6
HEADER_ENTRY_SIZE = HEADER_INTS * 4
HEADER_SIZE = REGION_CHUNKS * HEADER_ENTRY_SIZE
REGION_RE = re.compile(r"^reg\.(-?\d+)\.(-?\d+)\.mdat$")


@dataclass(frozen=True)
class ChunkRef:
    x: int
    z: int
    world: str
    region_x: int
    region_z: int
    local_index: int
    sector: int
    sectors: int
    file: Path

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "z": self.z,
            "world": self.world,
            "region_x": self.region_x,
            "region_z": self.region_z,
            "local_index": self.local_index,
            "file": str(self.file),
        }


def parse_region_name(path: Path) -> tuple[int, int] | None:
    match = REGION_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def iter_region_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("reg.*.*.mdat") if parse_region_name(path))


def list_worlds(map_cache: Path) -> list[dict]:
    worlds: list[dict] = []
    if not map_cache.is_dir():
        return worlds
    for child in sorted(map_cache.iterdir()):
        if not child.is_dir():
            continue
        regions = iter_region_files(child)
        if not regions:
            continue
        worlds.append({
            "name": child.name,
            "path": str(child),
            "regions": len(regions),
            "mtime": max((p.stat().st_mtime for p in regions), default=child.stat().st_mtime),
        })
    return worlds


def scan_region(path: Path, world: str) -> list[ChunkRef]:
    coords = parse_region_name(path)
    if coords is None:
        return []
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        return []
    rx, rz = coords
    chunks: list[ChunkRef] = []
    min_sector = math.ceil(HEADER_SIZE / SECTOR_SIZE)
    for local_index in range(REGION_CHUNKS):
        offset = local_index * HEADER_ENTRY_SIZE
        sector, sector_count, *_ = struct.unpack_from(">6i", data, offset)
        if sector < min_sector or sector_count <= 0:
            continue
        payload_pos = sector * SECTOR_SIZE
        if payload_pos + 4 > len(data):
            continue
        local_x = local_index & (REGION_SIZE - 1)
        local_z = local_index >> REGION_BITS
        chunks.append(ChunkRef(
            x=rx * REGION_SIZE + local_x,
            z=rz * REGION_SIZE + local_z,
            world=world,
            region_x=rx,
            region_z=rz,
            local_index=local_index,
            sector=sector,
            sectors=sector_count,
            file=path,
        ))
    return chunks


def scan_map_cache(map_cache: Path, *, excluded_worlds: set[str] | None = None) -> dict:
    excluded_worlds = excluded_worlds or set()
    worlds = list_worlds(map_cache)
    source_chunks: list[ChunkRef] = []
    errors: list[dict] = []
    world_summaries: list[dict] = []
    for world in worlds:
        name = world["name"]
        regions = iter_region_files(Path(world["path"]))
        raw_chunks: list[ChunkRef] = []
        for region in regions:
            try:
                raw_chunks.extend(scan_region(region, name))
            except Exception as exc:
                errors.append({"world": name, "file": str(region), "error": str(exc)})
        world_summaries.append({
            **world,
            "chunks": len(raw_chunks),
            "excluded": name in excluded_worlds,
        })
        source_chunks.extend(raw_chunks)

    chunks = [chunk for chunk in source_chunks if chunk.world not in excluded_worlds]

    by_coord: dict[tuple[int, int], list[ChunkRef]] = {}
    for chunk in chunks:
        by_coord.setdefault((chunk.x, chunk.z), []).append(chunk)

    deduped: list[dict] = []
    overlap_count = 0
    for (_x, _z), refs in sorted(by_coord.items()):
        # Prefer newest source file for overlapping worlds; excluding worlds
        # from the UI changes this choice without mutating source data.
        refs = sorted(refs, key=lambda c: (c.file.stat().st_mtime, c.world), reverse=True)
        chosen = refs[0]
        row = chosen.to_dict()
        row["overlaps"] = [ref.world for ref in refs]
        row["overlap_count"] = len(refs)
        if len(refs) > 1:
            overlap_count += 1
        deduped.append(row)

    xs = [c["x"] for c in deduped]
    zs = [c["z"] for c in deduped]
    return {
        "map_cache": str(map_cache),
        "worlds": world_summaries,
        "regions": sum(w["regions"] for w in world_summaries if not w["excluded"]),
        "source_chunks": [c.to_dict() for c in source_chunks],
        "chunks": deduped,
        "raw_chunk_count": len(chunks),
        "chunk_count": len(deduped),
        "overlap_chunks": overlap_count,
        "excluded_worlds": sorted(excluded_worlds),
        "bounds": {
            "min_x": min(xs) if xs else 0,
            "max_x": max(xs) if xs else 0,
            "min_z": min(zs) if zs else 0,
            "max_z": max(zs) if zs else 0,
        },
        "errors": errors,
    }


def write_filtered_scan_chunks(scan: dict, out_dir: Path, selected: Iterable[tuple[int, int]]) -> int:
    wanted = set(selected)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows = {
        (int(row["x"]), int(row["z"])): row
        for row in scan.get("chunks") or []
        if (int(row["x"]), int(row["z"])) in wanted
    }
    by_file: dict[Path, list[dict]] = {}
    for row in chunk_rows.values():
        by_file.setdefault(Path(row["file"]), []).append(row)

    staged: dict[tuple[int, int], list[tuple[int, int, bytes]]] = {}
    for source, rows in by_file.items():
        coords = parse_region_name(source)
        if coords is None:
            continue
        data = source.read_bytes()
        rx, rz = coords
        for row in sorted(rows, key=lambda item: (int(item["x"]), int(item["z"]))):
            x = int(row["x"])
            z = int(row["z"])
            local_x = x - rx * REGION_SIZE
            local_z = z - rz * REGION_SIZE
            if not (0 <= local_x < REGION_SIZE and 0 <= local_z < REGION_SIZE):
                continue
            local_index = local_z * REGION_SIZE + local_x
            entry_offset = local_index * HEADER_ENTRY_SIZE
            sector, sector_count, *_uuid = struct.unpack_from(">6i", data, entry_offset)
            if sector <= 0 or sector_count <= 0:
                continue
            start = sector * SECTOR_SIZE
            end = start + sector_count * SECTOR_SIZE
            if end > len(data):
                continue
            staged.setdefault(coords, []).append((local_index, sector_count, data[start:end]))

    written = 0
    for (rx, rz), payload_rows in staged.items():
        header = bytearray(HEADER_SIZE)
        payloads = bytearray()
        next_sector = math.ceil(HEADER_SIZE / SECTOR_SIZE)
        seen_local: set[int] = set()
        for local_index, sector_count, payload in sorted(payload_rows, key=lambda item: item[0]):
            if local_index in seen_local:
                continue
            seen_local.add(local_index)
            struct.pack_into(">6i", header, local_index * HEADER_ENTRY_SIZE, next_sector, sector_count, 0, 0, 0, 0)
            payloads.extend(payload)
            next_sector += sector_count
            written += 1
        if payloads:
            (out_dir / f"reg.{rx}.{rz}.mdat").write_bytes(bytes(header) + bytes(payloads))
    return written


def write_filtered_map_cache(source_map_cache: Path, out_dir: Path, selected: Iterable[tuple[int, int]]) -> int:
    scan = scan_map_cache(source_map_cache)
    return write_filtered_scan_chunks(scan, out_dir, selected)


def save_scan(path: Path, scan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
