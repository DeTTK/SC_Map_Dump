#!/usr/bin/env python3
"""
Standalone exporter for .mdat region files.

The goal of this tool is to keep the decoded format obvious:
region header -> zstd chunk payload -> section arrays -> AABB mesh -> OBJ.

It emits render geometry from decoded terrain chunks. Unknown shape mappings
fall back to full cubes, which keeps partial exports predictable.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import struct
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import zstandard as zstd
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Missing dependency: pip install zstandard") from exc


SECTOR_SIZE = 4096
REGION_BITS = 5
REGION_SIZE = 1 << REGION_BITS
REGION_CHUNKS = REGION_SIZE * REGION_SIZE
HEADER_INTS = 6
HEADER_ENTRY_SIZE = HEADER_INTS * 4
HEADER_SIZE = REGION_CHUNKS * HEADER_ENTRY_SIZE

AABB = Tuple[float, float, float, float, float, float]
Face = Tuple[Tuple[float, float, float], ...]
UV = Tuple[float, float]
BBox = Tuple[int, int, int, int, int, int]
FULL_CUBE: AABB = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_REGISTRY = ROOT / "docs" / "external_config_block_registry.json"
DEFAULT_GEOMETRY_REPORT = ROOT / "docs" / "north_mines_geometry_by_block_id.json"
DEFAULT_BLOCK_REGISTRY = ROOT / "docs" / "block_registry.json"
DEFAULT_GAME_DIR = Path(".")
DEFAULT_UNPACKED_DIR = ROOT / "assets"
DEFAULT_MAP_BLOCKS = DEFAULT_UNPACKED_DIR / "configs" / "assets" / "pda" / "map_blocks.json"
DEFAULT_WEATHER_PALETTES = DEFAULT_UNPACKED_DIR / "configs" / "assets" / "effects" / "weather" / "palettes.json"
DEFAULT_BLOCK_TEXARR = DEFAULT_GAME_DIR / "modassets" / "assets" / "stalcraft" / "textures" / "blockMap.texarr"
DEFAULT_CTM_DIR = DEFAULT_GAME_DIR / "modassets" / "assets" / "stalcraft" / "ctmpatcher" / "ctm"
TEXARR_TA_AES_KEY = bytes([11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 12, 13, 14, 15])
TEXARR_TA_AES_IV = b"0123456789ABCDFE"


@dataclass(frozen=True)
class BlockState:
    block_id: int
    meta: int


@dataclass
class ProBuilderTile:
    world_x: int
    y: int
    world_z: int
    tile_type: int
    covers: List[int]
    overlays: List[int]  # Game NBT/stream color_short_* values, kept signed.
    colors: List[int]  # Game NBT/stream overlay bytes: grass/snow/vine pass id.
    data: int
    data_ext: int = 0
    rot: int = 0
    color_exponent: int = 0
    packcovers: int = 0
    side_extra: List[int] = None  # type: ignore[assignment]
    flags: int = 0
    has_all_covers: bool = False
    has_overlays: bool = False
    has_colors: bool = False
    has_color_exp: bool = False
    has_rot: bool = False
    has_packcovers: bool = False
    has_extra_side: bool = False


@dataclass
class ChunkData:
    chunk_x: int
    chunk_z: int
    primary_mask: int
    add_mask: int
    blocks: Dict[Tuple[int, int, int], BlockState]
    tiles: Dict[Tuple[int, int, int], ProBuilderTile]
    biomes: bytes = b""


@dataclass
class TypeTextureInfo:
    block_id: int
    type_id: int
    base_icon: Optional[str] = None
    pane_icon: Optional[str] = None
    pane_edge_icon: Optional[str] = None
    wrapper_base_id: Optional[int] = None
    wrapper_meta: int = 0


@dataclass(frozen=True)
class ModelFace:
    vertices: Face
    uvs: Tuple[UV, ...]


@dataclass(frozen=True)
class ExternalModelMesh:
    source: Path
    faces: Tuple[ModelFace, ...]


class ByteReader:
    def __init__(self, data: bytes, offset: int = 0) -> None:
        self.data = data
        self.offset = offset

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def read_byte(self) -> int:
        value = struct.unpack_from(">b", self.data, self.offset)[0]
        self.offset += 1
        return value

    def read_short(self) -> int:
        value = struct.unpack_from(">h", self.data, self.offset)[0]
        self.offset += 2
        return value

    def read_int(self) -> int:
        value = struct.unpack_from(">i", self.data, self.offset)[0]
        self.offset += 4
        return value


def parse_region_coords(path: Path) -> Tuple[int, int]:
    parts = path.stem.split(".")
    if len(parts) != 3 or parts[0] != "reg":
        raise ValueError(f"Unexpected region filename: {path.name}")
    return int(parts[1]), int(parts[2])


def normalize_bbox(values: Sequence[int]) -> BBox:
    if len(values) != 6:
        raise ValueError("bbox requires exactly six integers: x1 y1 z1 x2 y2 z2")
    x1, y1, z1, x2, y2, z2 = values
    return (min(x1, x2), min(y1, y2), min(z1, z2), max(x1, x2), max(y1, y2), max(z1, z2))


def expand_bbox(bbox: Optional[BBox], padding: int) -> Optional[BBox]:
    if bbox is None or padding <= 0:
        return bbox
    x0, y0, z0, x1, y1, z1 = bbox
    return (x0 - padding, y0 - padding, z0 - padding, x1 + padding, y1 + padding, z1 + padding)


def pos_in_bbox(pos: Tuple[int, int, int], bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return True
    x0, y0, z0, x1, y1, z1 = bbox
    x, y, z = pos
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1


def chunk_intersects_bbox(chunk_x: int, chunk_z: int, bbox: Optional[BBox]) -> bool:
    if bbox is None:
        return True
    x0, _y0, z0, x1, _y1, z1 = bbox
    chunk_min_x = chunk_x << 4
    chunk_min_z = chunk_z << 4
    chunk_max_x = chunk_min_x + 15
    chunk_max_z = chunk_min_z + 15
    return not (chunk_max_x < x0 or chunk_min_x > x1 or chunk_max_z < z0 or chunk_min_z > z1)


def nibble_at(buf: bytes, index: int) -> int:
    value = buf[index >> 1]
    if index & 1:
        return (value >> 4) & 0xF
    return value & 0xF


def parse_pb_tile_payload(reader: ByteReader, tile_type: int, x: int, y: int, z: int) -> ProBuilderTile:
    flags = reader.read_byte()
    has_extra_side = bool(flags & 0x40)
    has_packcovers = bool(flags & 0x20)
    has_rot = bool(flags & 0x10)
    has_all_covers = bool(flags & 0x08)
    has_color_exp = bool(flags & 0x04)
    has_overlays = bool(flags & 0x02)
    has_colors = bool(flags & 0x01)

    covers = [-1] * 7
    if has_all_covers:
        covers = [reader.read_short() for _ in range(7)]
    else:
        covers[6] = reader.read_short()

    overlays = [-1] * 7
    if has_overlays:
        overlays = [reader.read_short() for _ in range(7)]

    color_exponent = reader.read_short() if has_color_exp else 0

    colors = [0] * 7
    if has_colors:
        colors = [reader.read_byte() for _ in range(7)]

    rot = reader.read_short() if has_rot else 0
    packcovers = reader.read_byte() if has_packcovers else 0

    side_extra = [0] * 6
    if has_extra_side:
        side_extra = [reader.read_byte() for _ in range(6)]

    data = reader.read_int()
    data_ext = reader.read_int() if tile_type == 1 else 0

    return ProBuilderTile(
        world_x=x,
        y=y,
        world_z=z,
        tile_type=tile_type,
        covers=covers,
        overlays=overlays,
        colors=colors,
        data=data,
        data_ext=data_ext,
        rot=rot,
        color_exponent=color_exponent,
        packcovers=packcovers,
        side_extra=side_extra,
        flags=flags,
        has_all_covers=has_all_covers,
        has_overlays=has_overlays,
        has_colors=has_colors,
        has_color_exp=has_color_exp,
        has_rot=has_rot,
        has_packcovers=has_packcovers,
        has_extra_side=has_extra_side,
    )


def parse_tiles(decoded: bytes, tile_offset: int) -> Dict[Tuple[int, int, int], ProBuilderTile]:
    if tile_offset + 4 > len(decoded):
        return {}
    reader = ByteReader(decoded, tile_offset)
    count = reader.read_int()
    tiles: Dict[Tuple[int, int, int], ProBuilderTile] = {}
    for _ in range(count):
        tile_type = reader.read_byte()
        x = reader.read_int()
        y = reader.read_int()
        z = reader.read_int()
        tile = parse_pb_tile_payload(reader, tile_type, x, y, z)
        tiles[(x, y, z)] = tile
    return tiles


def decode_section_blocks(raw: bytes, primary_mask: int, add_mask: int) -> Tuple[Dict[Tuple[int, int, int], BlockState], bytes]:
    sections = [section for section in range(16) if primary_mask & (1 << section)]
    offset = 0

    low_by_section: Dict[int, bytes] = {}
    for section in sections:
        low_by_section[section] = raw[offset : offset + 4096]
        offset += 4096

    meta_by_section: Dict[int, bytes] = {}
    for section in sections:
        meta_by_section[section] = raw[offset : offset + 2048]
        offset += 2048

    # The next three nibble groups are light/aux data. They are irrelevant for
    # geometry, but their presence is controlled by the primary mask.
    offset += len(sections) * 2048 * 3

    add_by_section: Dict[int, bytes] = {}
    for section in range(16):
        if add_mask & (1 << section):
            add_by_section[section] = raw[offset : offset + 2048]
            offset += 2048

    # Final 256 bytes are biome ids when the full chunk is applied. Geometry
    # ignores them, but render texturing uses them for block tint.
    biome_bytes = raw[offset : offset + 256] if len(raw) - offset >= 256 else b""

    blocks: Dict[Tuple[int, int, int], BlockState] = {}
    for section in sections:
        low = low_by_section[section]
        meta = meta_by_section[section]
        add = add_by_section.get(section)
        base_y = section << 4
        for local_index, low_id in enumerate(low):
            high = nibble_at(add, local_index) if add is not None else 0
            block_id = low_id | (high << 8)
            if block_id == 0:
                continue
            y = base_y + (local_index >> 8)
            z = (local_index >> 4) & 0xF
            x = local_index & 0xF
            blocks[(x, y, z)] = BlockState(block_id, nibble_at(meta, local_index))
    return blocks, biome_bytes


def read_region(path: Path) -> Iterator[ChunkData]:
    rx, rz = parse_region_coords(path)
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        return

    dctx = zstd.ZstdDecompressor()
    for local_index in range(REGION_CHUNKS):
        entry_offset = local_index * HEADER_ENTRY_SIZE
        sector, sector_count, *_uuid = struct.unpack_from(">6i", data, entry_offset)
        if sector == 0 or sector_count == 0:
            continue
        if sector < HEADER_SIZE // SECTOR_SIZE:
            continue

        payload_pos = sector * SECTOR_SIZE
        if payload_pos + 4 > len(data):
            continue

        declared_len = struct.unpack_from(">i", data, payload_pos)[0]
        payload_len = declared_len - 1
        payload = data[payload_pos + 4 : payload_pos + 4 + payload_len]
        if len(payload) < 16:
            continue

        primary_mask, add_mask, raw_len, compressed_len = struct.unpack_from(">4i", payload, 0)
        compressed = payload[16 : 16 + compressed_len]
        decoded = dctx.decompress(compressed)
        raw = decoded[:raw_len]

        local_x = local_index & (REGION_SIZE - 1)
        local_z = local_index >> REGION_BITS
        chunk_x = rx * REGION_SIZE + local_x
        chunk_z = rz * REGION_SIZE + local_z

        blocks, biomes = decode_section_blocks(raw, primary_mask, add_mask)
        yield ChunkData(
            chunk_x=chunk_x,
            chunk_z=chunk_z,
            primary_mask=primary_mask,
            add_mask=add_mask,
            blocks=blocks,
            tiles=parse_tiles(decoded, raw_len),
            biomes=biomes,
        )


def extract_method_body(src: str, signature_pattern: str) -> str:
    match = re.search(signature_pattern, src)
    if not match:
        return ""
    start = src.find("{", match.end())
    if start < 0:
        return ""
    depth = 0
    for pos in range(start, len(src)):
        ch = src[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1 : pos]
    return ""


def parse_id_constants(src: str) -> Dict[str, int]:
    body = extract_method_body(src, r"public static void char\(\)\s*")
    out: Dict[str, int] = {}
    next_id: Optional[int] = None
    for raw_line in body.splitlines():
        line = raw_line.strip().rstrip(";")
        if not line:
            continue
        m = re.match(r"int\s+n\s*=\s*(-?\d+)$", line)
        if m:
            next_id = int(m.group(1))
            continue
        if line == "++n":
            if next_id is not None:
                next_id += 1
            continue
        m = re.match(r"n\s*\+=\s*(-?\d+)$", line)
        if m and next_id is not None:
            next_id += int(m.group(1))
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)$", line)
        if m:
            out[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*n\+\+$", line)
        if m and next_id is not None:
            out[m.group(1)] = next_id
            next_id += 1
    return out


# Material constants from class 288455 that 111004 treats as "snow above" for
# the side-snowed grass icon. `void` (ice) is intentionally excluded because
# 111004.do checks against `new` and `for` only.
SNOW_LIKE_MATERIAL_NAMES: Set[str] = {"new", "for"}

# Vanilla blocks that the registrar maps onto the base 275700 class without a
# dedicated subclass. They never carry their own `super(_, 288455.X)` call, so
# parsing the class file cannot recover their material. Listed manually as a
# backstop. Block 80 is the full snow block; matches `288455.new` in game.
SNOW_LIKE_BLOCK_ID_OVERRIDES: Set[int] = {80}


def parse_class_super_material(src: str) -> Optional[str]:
    """Return the `288455.<material>` name from a constructor `super(...)` call."""

    match = re.search(r"super\([^()]*?,\s*288455\.([A-Za-z_][A-Za-z0-9_]*)\s*\)", src)
    if match is None:
        return None
    return match.group(1)


def load_type_materials(source_tables_dir: Path, type_ids: Iterable[int]) -> Dict[int, str]:
    materials: Dict[int, str] = {}
    for type_id in type_ids:
        path = source_tables_dir / f"{type_id}.java"
        if not path.exists():
            continue
        material = parse_class_super_material(path.read_text(encoding="utf-8", errors="ignore"))
        if material is not None:
            materials[type_id] = material
    return materials


def load_snow_like_block_ids(
    source_tables_dir: Path,
    block_type_by_id: Dict[int, int],
) -> Set[int]:
    """Recover the block ids whose class material is `new` or `for`.

    These are the materials that `111004.do(world,x,y,z,side)` recognises as
    "snow above" when picking `*_side_snowed` instead of `*_side`. Replaces
    the legacy hard-coded `{78, 79, 80}` set: 79 (ice) is removed because the
    game does NOT treat it as snow above grass, while 80 (snow block) stays
    via `SNOW_LIKE_BLOCK_ID_OVERRIDES` since it falls back to the base class.
    """

    materials = load_type_materials(source_tables_dir, set(block_type_by_id.values()))
    snow: Set[int] = set(SNOW_LIKE_BLOCK_ID_OVERRIDES)
    for block_id, type_id in block_type_by_id.items():
        if materials.get(type_id) in SNOW_LIKE_MATERIAL_NAMES:
            snow.add(block_id)
    return snow


def load_block_registry(source_tables_dir: Path) -> Dict[int, int]:
    registry: Dict[int, int] = {}

    registrar = source_tables_dir / "17525.java"
    if registrar.exists():
        src = registrar.read_text(encoding="utf-8", errors="ignore")
        const_to_id = parse_id_constants(src)
        body = extract_method_body(src, r"public static void do\(\)\s*")
        for type_id, arg in re.findall(r"new\s+([0-9]+)\(([A-Za-z_][A-Za-z0-9_]*)\)", body):
            block_id = const_to_id.get(arg)
            if block_id is not None:
                registry[block_id] = int(type_id)

        slope_src = source_tables_dir / "131839.java"
        slope_count = 0
        if slope_src.exists():
            slope_text = slope_src.read_text(encoding="utf-8", errors="ignore")
            count_match = re.search(r"new\s+275700\[(\d+)\]", slope_text)
            if count_match:
                slope_count = int(count_match.group(1))
        if slope_count:
            # 17525.do() registers slope wrapper blocks with computed ids:
            #   new 131839(2415 + n2++, baseBlock)
            #   new 168350(2445 + n2++, baseBlock)
            # CFR sometimes prints the second post-increment as pre-increment,
            # but javap shows both are post-increment ranges.
            for offset in range(slope_count):
                registry.setdefault(2415 + offset, 131839)
                registry.setdefault(2445 + offset, 168350)

    # Vanilla-like blocks are registered in the main block class via direct
    # constructors such as new 106044(102, ...). Parse only this file to avoid
    # scanning the whole source table on every export.
    block_src = source_tables_dir / "275700.java"
    if block_src.exists():
        src = block_src.read_text(encoding="utf-8", errors="ignore")
        for type_id, block_id in re.findall(r"new\s+([0-9]+)\((-?\d+)\s*[,)]", src):
            registry.setdefault(int(block_id), int(type_id))

    return registry


def split_java_args(arg_text: str) -> List[str]:
    args: List[str] = []
    start = 0
    in_string = False
    escaped = False
    for index, ch in enumerate(arg_text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == ",":
            args.append(arg_text[start:index].strip())
            start = index + 1
    tail = arg_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def unquote_java_string(token: str) -> Optional[str]:
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return bytes(token[1:-1], "utf-8").decode("unicode_escape")
    return None


def load_type_texture_registry(source_tables_dir: Path, block_type_by_id: Dict[int, int]) -> Dict[int, TypeTextureInfo]:
    """Recover the in-game block icon registration table from 275700/88276.

    map_blocks.json is a PDA/minimap table. The render path first lets every
    275700 block register icons through 88276, then asks the block class for a
    side/meta-specific 149882 icon. This registry captures that class-side
    state in a compact form for the OBJ exporter.
    """

    infos: Dict[int, TypeTextureInfo] = {}
    field_to_id: Dict[str, int] = {}
    pending_wrappers: List[Tuple[int, str, int]] = []

    block_src = source_tables_dir / "275700.java"
    if block_src.exists():
        for raw_line in block_src.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "275700." not in raw_line or "new " not in raw_line:
                continue
            line = raw_line.strip()
            field_match = re.search(r"275700\.([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            ctor_match = re.search(r"new\s+([0-9]+)\(([^)]*)\)", line)
            if not field_match or not ctor_match:
                continue
            args = split_java_args(ctor_match.group(2))
            if not args or not args[0].lstrip("-").isdigit():
                continue
            field_name = field_match.group(1)
            block_id = int(args[0])
            type_id = int(ctor_match.group(1))
            field_to_id[field_name] = block_id

            icon_matches = re.findall(r"\.char\(\"([^\"]+)\"\)", line)
            info = TypeTextureInfo(
                block_id=block_id,
                type_id=type_id,
                base_icon=icon_matches[-1] if icon_matches else None,
            )
            if type_id == 106044 and len(args) >= 3:
                info.pane_icon = unquote_java_string(args[1])
                info.pane_edge_icon = unquote_java_string(args[2])
                info.base_icon = info.pane_icon
            if type_id == 216192 and len(args) >= 3:
                base_match = re.match(r"275700\.([A-Za-z_][A-Za-z0-9_]*)$", args[1])
                wrapper_meta = int(args[2]) if args[2].lstrip("-").isdigit() else 0
                if base_match:
                    pending_wrappers.append((block_id, base_match.group(1), wrapper_meta))
            infos[block_id] = info

    for block_id, base_field, wrapper_meta in pending_wrappers:
        info = infos.get(block_id)
        base_id = field_to_id.get(base_field)
        if info is not None and base_id is not None:
            info.wrapper_base_id = base_id
            info.wrapper_meta = wrapper_meta & 0xF

    # SMT slope wrappers are generated from 131839.byte. Their class delegates
    # texture selection straight back to the wrapped base block.
    slope_src = source_tables_dir / "131839.java"
    if slope_src.exists():
        slope_text = slope_src.read_text(encoding="utf-8", errors="ignore")
        base_fields: Dict[int, str] = {}
        for index, field in re.findall(r"nullArray\[(\d+)\]\s*=\s*275700\.([A-Za-z_][A-Za-z0-9_]*)", slope_text):
            base_fields[int(index)] = field
        for offset in sorted(base_fields):
            base_id = field_to_id.get(base_fields[offset])
            if base_id is None:
                continue
            for block_id in (2415 + offset, 2445 + offset):
                type_id = block_type_by_id.get(block_id)
                if type_id is None:
                    continue
                infos[block_id] = TypeTextureInfo(
                    block_id=block_id,
                    type_id=type_id,
                    wrapper_base_id=base_id,
                    wrapper_meta=0,
                )

    type_default_icons: Dict[int, str] = {}
    type_extends: Dict[int, int] = {}
    for type_id in set(block_type_by_id.values()):
        path = source_tables_dir / f"{type_id}.java"
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        extends_match = re.search(r"public\s+(?:abstract\s+)?class\s+[0-9]+\s+extends\s+([0-9]+)", src)
        if extends_match:
            type_extends[type_id] = int(extends_match.group(1))
        constructor_icons = re.findall(r"\.char\(\"([^\"]+)\"\)", src)
        literal_for_icons = re.findall(r"\.for\(\"([^\"]+)\"\)", src)
        if constructor_icons:
            preferred = next((icon for icon in constructor_icons if ":" in icon), constructor_icons[-1])
            type_default_icons[type_id] = preferred
        elif literal_for_icons:
            type_default_icons[type_id] = literal_for_icons[0]

    def inherited_type_icon(type_id: int) -> Optional[str]:
        seen: Set[int] = set()
        current = type_id
        while current not in seen:
            seen.add(current)
            icon = type_default_icons.get(current)
            if icon is not None:
                return icon
            parent = type_extends.get(current)
            if parent is None:
                return None
            current = parent
        return None

    for block_id, type_id in block_type_by_id.items():
        info = infos.setdefault(block_id, TypeTextureInfo(block_id=block_id, type_id=type_id))
        if info.base_icon is None and info.wrapper_base_id is None:
            info.base_icon = inherited_type_icon(type_id)
    return infos


def merge_block_registry_json(registry: Dict[int, int], registry_json: Path) -> Dict[int, int]:
    if not registry_json.exists():
        return registry
    raw = json.loads(registry_json.read_text(encoding="utf-8"))
    merged = dict(registry)
    for block_id, item in raw.items():
        if not str(block_id).lstrip("-").isdigit():
            continue
        type_id = (item.get("type_id", item.get("type_id")) if isinstance(item, dict) else item)
        if isinstance(type_id, int):
            merged[int(block_id)] = type_id
    return merged


def load_external_block_configs(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, Any]] = {}

    for block_id, item in raw.items():
        if not str(block_id).lstrip("-").isdigit() or int(block_id) < 0:
            continue
        if not isinstance(item, dict):
            continue
        cfg = item.get("config")
        if isinstance(cfg, dict):
            out[int(block_id)] = cfg
    return out


def load_geometry_report(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["block_id"]): row for row in raw.get("by_id", []) if "block_id" in row}


def java_rgb_to_float(value: int) -> Tuple[float, float, float]:
    value &= 0xFFFFFF
    return (
        ((value >> 16) & 0xFF) / 255.0,
        ((value >> 8) & 0xFF) / 255.0,
        (value & 0xFF) / 255.0,
    )


def load_biome_color_table(path: Path) -> Dict[int, Tuple[float, float, float]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    palettes = raw.values() if isinstance(raw, dict) else raw
    palette_list = [item for item in palettes if isinstance(item, dict) and isinstance(item.get("biomes"), list)]
    if not palette_list:
        return {}

    chosen = next((item for item in palette_list if item.get("isDefault") is True), None)
    if chosen is None:
        chosen = next((item for item in palette_list if str(item.get("name", "")).lower() == "default palette"), None)
    if chosen is None:
        chosen = max(palette_list, key=lambda item: len(item.get("biomes") or []))

    out: Dict[int, Tuple[float, float, float]] = {}
    for row in chosen.get("biomes") or []:
        if not isinstance(row, dict):
            continue
        biome_id = row.get("id")
        color = row.get("color")
        if isinstance(biome_id, int) and isinstance(color, int):
            out[biome_id & 0xFF] = java_rgb_to_float(color)
    return out


def parse_switch_bounds_table(src: str) -> Dict[int, List[AABB]]:
    body = extract_method_body(src, r"public void char\(143968\s+[^)]*\)\s*")
    if not body:
        return {}

    switch_match = re.search(r"\bswitch\s*\(", body)
    if not switch_match:
        return {}
    switch_start = body.find("{", switch_match.end())
    if switch_start < 0:
        return {}

    depth = 0
    switch_end = -1
    for pos in range(switch_start, len(body)):
        ch = body[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                switch_end = pos
                break
    if switch_end < 0:
        return {}

    switch_body = body[switch_start + 1 : switch_end]
    table: Dict[int, List[AABB]] = {}
    case_matches = list(re.finditer(r"\bcase\s+(-?\d+)\s*:", switch_body))
    boundary_matches = sorted(
        [
            *case_matches,
            *re.finditer(r"\bdefault\s*:", switch_body),
        ],
        key=lambda item: item.start(),
    )
    for case_match in case_matches:
        key = int(case_match.group(1))
        following_boundaries = [item.start() for item in boundary_matches if item.start() > case_match.start()]
        end = following_boundaries[0] if following_boundaries else len(switch_body)
        case_body = switch_body[case_match.end() : end]
        bounds_match = re.search(r"this\.char\(([^)]*)\);", case_body)
        if not bounds_match:
            continue
        values = parse_float_args(bounds_match.group(1))
        if len(values) == 6:
            table[key] = [tuple(values)]  # type: ignore[list-item]
    return table


def parse_float_args(arg_text: str) -> List[float]:
    values: List[float] = []
    for raw in arg_text.split(","):
        token = raw.strip().rstrip("fFdD")
        try:
            values.append(float(token))
        except ValueError:
            return []
    return values


def parse_static_part_table(src: str, type_id: int) -> Dict[int, List[AABB]]:
    body = extract_method_body(src, r"public static float\[\] for\(int n, int n2\)\s*")
    if not body:
        return {}

    table: Dict[int, List[AABB]] = {}
    switch_stack: List[Tuple[str, int]] = []
    current_shape: Optional[int] = None
    depth = 1

    for line in body.splitlines():
        stripped = line.strip()
        pending_switch: Optional[str] = None
        switch_match = re.search(r"switch\s*\((n2|n)\)", stripped)
        if switch_match:
            pending_switch = switch_match.group(1)

        case_match = re.search(r"case\s+(-?\d+)\s*:", stripped)
        if case_match and switch_stack:
            active_switch = switch_stack[-1][0]
            if active_switch == "n2":
                current_shape = int(case_match.group(1))
                table.setdefault(current_shape, [])

        return_match = re.search(rf"return\s+{type_id}\.do\(([^)]*)\);", stripped)
        if return_match and current_shape is not None:
            values = parse_float_args(return_match.group(1))
            if len(values) == 6:
                table.setdefault(current_shape, []).append(tuple(values))  # type: ignore[arg-type]

        new_depth = depth + stripped.count("{") - stripped.count("}")
        if pending_switch is not None:
            switch_stack.append((pending_switch, depth + stripped.count("{")))
        while switch_stack and switch_stack[-1][1] > new_depth:
            switch_stack.pop()
        depth = new_depth

    return {key: value for key, value in table.items() if value}


def load_shape_tables(source_tables_dir: Path, registry: Dict[int, int]) -> Dict[int, Dict[int, List[AABB]]]:
    type_ids = set(registry.values())
    tables: Dict[int, Dict[int, List[AABB]]] = {}
    for type_id in type_ids:
        path = source_tables_dir / f"{type_id}.java"
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        table = parse_static_part_table(src, type_id)
        if not table:
            table = parse_switch_bounds_table(src)
        if table:
            tables[type_id] = table
    extra_tables = load_payload_shape_tables(source_tables_dir, type_ids)
    for type_id, table in extra_tables.items():
        tables.setdefault(type_id, {}).update(table)
    return tables


def parse_nested_part_table(src: str, signature_pattern: str, type_id: int, part_offset: int = 0) -> Dict[int, List[AABB]]:
    body = extract_method_body(src, signature_pattern)
    if not body:
        return {}
    table: Dict[int, Dict[int, AABB]] = {}
    switch_stack: List[Tuple[str, int]] = []
    current_outer: Optional[int] = None
    current_inner: Optional[int] = None
    depth = 1

    def switch_role(expr: str) -> Optional[str]:
        expr = expr.strip()
        if expr == "n":
            return "inner"
        if expr == "n2" or expr.endswith(".break"):
            return "outer"
        return None

    for line in body.splitlines():
        stripped = line.strip()
        pending_switch: Optional[str] = None
        switch_match = re.search(r"switch\s*\(([^)]*)\)", stripped)
        if switch_match:
            pending_switch = switch_role(switch_match.group(1))

        case_match = re.search(r"case\s+(-?\d+)\s*:", stripped)
        if case_match and switch_stack:
            value = int(case_match.group(1))
            active_switch = switch_stack[-1][0]
            if active_switch == "outer":
                current_outer = value
                current_inner = None
                table.setdefault(current_outer, {})
            elif active_switch == "inner":
                current_inner = value - part_offset
        if stripped.startswith("default:"):
            current_inner = None
        if stripped.startswith("return false") or stripped.startswith("break"):
            current_inner = None
        return_match = re.search(rf"return\s+{type_id}\.do\(([^)]*)\);", stripped)
        if return_match and current_outer is not None and current_inner is not None:
            values = parse_float_args(return_match.group(1))
            if len(values) == 6:
                table.setdefault(current_outer, {})[current_inner] = tuple(values)  # type: ignore[assignment]

        new_depth = depth + stripped.count("{") - stripped.count("}")
        if pending_switch is not None:
            switch_stack.append((pending_switch, depth + stripped.count("{")))
        while switch_stack and switch_stack[-1][1] > new_depth:
            switch_stack.pop()
        depth = new_depth

    out: Dict[int, List[AABB]] = {}
    for key, parts in table.items():
        out[key] = [box for _, box in sorted(parts.items())]
    return out


def parse_static_float_arrays(src: str) -> Dict[str, AABB]:
    arrays: Dict[str, List[Optional[float]]] = {}
    aliases: Dict[str, AABB] = {}
    current: Optional[str] = None
    for raw_line in src.splitlines():
        line = raw_line.strip().rstrip(";")
        new_match = re.match(r"float\[\]\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*new float\[6\]", line)
        if new_match:
            current = new_match.group(1)
            arrays[current] = [None] * 6
            continue
        assign_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\s*=\s*([0-9.]+)f?", line)
        if assign_match:
            name = assign_match.group(1)
            index = int(assign_match.group(2))
            value = float(assign_match.group(3))
            if name in arrays and 0 <= index < 6:
                arrays[name][index] = value
            continue
        alias_match = re.match(r"(?:[0-9]+\.)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if alias_match:
            alias = alias_match.group(1)
            source = alias_match.group(2)
            values = arrays.get(source)
            if values and all(value is not None for value in values):
                aliases[alias] = tuple(float(value) for value in values)  # type: ignore[arg-type]
                current = None
    return aliases


def parse_int_method_table(src: str) -> Dict[int, List[AABB]]:
    body = extract_method_body(src, r"public float\[\] int\(int n\)\s*")
    if not body:
        return {}

    static_arrays = parse_static_float_arrays(src)
    table: Dict[int, AABB] = {}
    default_box: Optional[AABB] = None
    current_cases: List[Optional[int]] = []
    local_values: Dict[str, List[Optional[float]]] = {}

    def assign_box(box: AABB) -> None:
        nonlocal default_box
        if not current_cases:
            return
        for case in current_cases:
            if case is None:
                default_box = box
            else:
                table[case] = box

    for raw_line in body.splitlines():
        line = raw_line.strip().rstrip(";")
        case_match = re.match(r"case\s+(-?\d+)\s*:", line)
        if case_match:
            current_cases.append(int(case_match.group(1)))
            continue
        if line.startswith("default:"):
            current_cases = [None]
            continue
        new_match = re.match(r"float\[\]\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*new float\[6\]", line)
        if new_match:
            local_values[new_match.group(1)] = [None] * 6
            continue
        assign_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\s*=\s*([0-9.]+)f?", line)
        if assign_match:
            name = assign_match.group(1)
            index = int(assign_match.group(2))
            value = float(assign_match.group(3))
            if name in local_values and 0 <= index < 6:
                local_values[name][index] = value
            continue
        return_local = re.match(r"return\s+(?:(?:[0-9]+)\.)?([A-Za-z_][A-Za-z0-9_]*)$", line)
        if return_local:
            name = return_local.group(1)
            if name in static_arrays:
                assign_box(static_arrays[name])
            else:
                values = local_values.get(name)
                if values and all(value is not None for value in values):
                    assign_box(tuple(float(value) for value in values))  # type: ignore[arg-type]
            current_cases = []
            continue
    out = {key: [box] for key, box in table.items()}
    if default_box is not None:
        for key in range(0, 256):
            out.setdefault(key, [default_box])
    return out


def load_payload_shape_tables(source_tables_dir: Path, type_ids: Set[int]) -> Dict[int, Dict[int, List[AABB]]]:
    tables: Dict[int, Dict[int, List[AABB]]] = {}
    nested_specs = {
        189260: (r"public float\[\] char\(int n, 153941\s+[^)]*\)\s*", 1),
        3124: (r"public float\[\] for\(int n, int n2\)\s*", 0),
        26120: (r"public float\[\] for\(int n, int n2\)\s*", 0),
    }
    for type_id, (pattern, offset) in nested_specs.items():
        if type_id not in type_ids:
            continue
        path = source_tables_dir / f"{type_id}.java"
        if path.exists():
            parsed = parse_nested_part_table(path.read_text(encoding="utf-8", errors="ignore"), pattern, type_id, offset)
            if parsed:
                tables[type_id] = parsed

    for type_id in (85288, 299262, 256114, 297013):
        if type_id not in type_ids:
            continue
        path = source_tables_dir / f"{type_id}.java"
        if path.exists():
            parsed = parse_int_method_table(path.read_text(encoding="utf-8", errors="ignore"))
            if parsed:
                tables[type_id] = parsed
    return tables


def is_full_aabb(box: AABB) -> bool:
    return box == FULL_CUBE


def pane_boxes(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    block_type_by_id: Dict[int, int],
    x: int,
    y: int,
    z: int,
) -> List[AABB]:
    def connectable(dx: int, dz: int) -> bool:
        other = world_blocks.get((x + dx, y, z + dz))
        if other is None:
            return False
        other_class = block_type_by_id.get(other.block_id)
        # Class 106044 is the pane/iron-bars style block. Glass (102 in
        # vanilla) and normal solid blocks are connectable in the original code.
        return other_class == 106044 or other.block_id != 0

    north = connectable(0, -1)
    south = connectable(0, 1)
    west = connectable(-1, 0)
    east = connectable(1, 0)

    boxes: List[AABB] = []
    if (west and east) or not (west or east or north or south):
        boxes.append((0.0, 0.0, 0.4375, 1.0, 1.0, 0.5625))
    else:
        if west:
            boxes.append((0.0, 0.0, 0.4375, 0.5, 1.0, 0.5625))
        if east:
            boxes.append((0.5, 0.0, 0.4375, 1.0, 1.0, 0.5625))

    if (north and south) or not (west or east or north or south):
        boxes.append((0.4375, 0.0, 0.0, 0.5625, 1.0, 1.0))
    else:
        if north:
            boxes.append((0.4375, 0.0, 0.0, 0.5625, 1.0, 0.5))
        if south:
            boxes.append((0.4375, 0.0, 0.5, 0.5625, 1.0, 1.0))
    return boxes


SLOPE_META_TO_J = {
    0: 34,
    1: 38,
    2: 0,
    3: 9,
    4: 32,
    5: 36,
    6: 3,
    7: 10,
    8: 30,
    9: 42,
    10: 1,
    11: 8,
    12: 28,
    13: 40,
    14: 2,
    15: 11,
}


def shift_box(box: AABB, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> AABB:
    x0, y0, z0, x1, y1, z1 = box
    return (x0 + dx, y0 + dy, z0 + dz, x1 + dx, y1 + dy, z1 + dz)


def slope_piece_for_j(j: int, layer: int, layers: int, index: int, width: int) -> Optional[AABB]:
    step0 = index / float(width)
    step1 = (index + 1) / float(width)
    layer0 = layer / float(layers)
    layer1 = (layer + 1) / float(layers)

    if j == 0:
        return (0.0, 0.0, 0.0, step1, 1.0, 1.0 - step0)
    if j == 1:
        return (step0, 0.0, 1.0 - step1, 1.0, 1.0, 1.0)
    if j == 2:
        return (0.0, 0.0, step0, step1, 1.0, 1.0)
    if j == 3:
        return (step0, 0.0, 0.0, 1.0, 1.0, step1)
    if j == 4:
        return (0.0, 1.0 - step1, step0, 1.0, 1.0, 1.0)
    if j == 5:
        return (0.0, step0, 0.0, 1.0, 1.0, step1)
    if j == 6:
        return (step0, 1.0 - step1, 0.0, 1.0, 1.0, 1.0)
    if j == 7:
        return (0.0, step0, 0.0, step1, 1.0, 1.0)
    if j == 8:
        return (0.0, 0.0, step0, 1.0, step1, 1.0)
    if j == 9:
        return (0.0, 0.0, 0.0, 1.0, 1.0 - step0, step1)
    if j == 10:
        return (step0, 0.0, 0.0, 1.0, step1, 1.0)
    if j == 11:
        return (0.0, 0.0, 0.0, step1, 1.0 - step0, 1.0)

    if j == 12:
        if layer == 0:
            return (0.0, 0.0, 0.0, step1, 1.0 - step0, 1.0)
        return (0.0, 0.0, step0, 1.0, step1, 1.0)
    if j == 13:
        if layer == 0:
            return (0.0, step0, 0.0, step1, 1.0, 1.0)
        return (0.0, 1.0 - step1, step0, 1.0, 1.0, 1.0)
    if j == 14:
        if layer == 0:
            return (step0, 0.0, 0.0, 1.0, step1, 1.0)
        return (0.0, 0.0, step0, 1.0, step1, 1.0)
    if j == 15:
        if layer == 0:
            return (step0, 1.0 - step1, 0.0, 1.0, 1.0, 1.0)
        return (0.0, 1.0 - step1, step0, 1.0, 1.0, 1.0)
    if j == 16:
        if layer == 0:
            return (step0, 0.0, 0.0, 1.0, step1, 1.0)
        return (0.0, 0.0, 0.0, 1.0, 1.0 - step0, step1)
    if j == 17:
        if layer == 0:
            return (step0, 1.0 - step1, 0.0, 1.0, 1.0, 1.0)
        return (0.0, step0, 0.0, 1.0, 1.0, step1)
    if j == 18:
        if layer == 0:
            return (0.0, 0.0, 0.0, step1, 1.0 - step0, 1.0)
        return (0.0, 0.0, 0.0, 1.0, 1.0 - step0, step1)
    if j == 19:
        if layer == 0:
            return (0.0, step0, 0.0, step1, 1.0, 1.0)
        return (0.0, step0, 0.0, 1.0, 1.0, step1)
    if j == 20:
        return (step0, 0.0, 0.0, 1.0, step1, 1.0 - step0)
    if j == 21:
        return (step0, 1.0 - step1, 0.0, 1.0, 1.0, 1.0 - step0)
    if j == 22:
        return (0.0, 0.0, 0.0, 1.0 - step0, step1, 1.0 - step0)
    if j == 23:
        return (0.0, 1.0 - step1, 0.0, 1.0 - step0, 1.0, 1.0 - step0)
    if j == 24:
        return (0.0, 0.0, step0, 1.0 - step0, step1, 1.0)
    if j == 25:
        return (0.0, 1.0 - step1, step0, 1.0 - step0, 1.0, 1.0)
    if j == 26:
        return (step0, 0.0, step0, 1.0, step1, 1.0)
    if j == 27:
        return (step0, 1.0 - step1, step0, 1.0, 1.0, 1.0)

    # Oblique internal/external collision path still uses the 8-step slices from
    # 161216; render mode replaces these with inferred smooth heightfields.
    if j == 28:
        if layer == 0:
            return (0.0, 0.0, step0, step1, 1.0, 1.0)
        return (0.4 * step0, 0.0, 0.4 * step0, 1.0 - 0.4 * step0, step1, 1.0 - 0.4 * step0)
    if j == 29:
        if layer == 0:
            return (0.0, 0.0, step0, step1, 1.0, 1.0)
        if layer == 1:
            return (0.0, step0, 0.0, step1, 1.0, 1.0)
        return (0.0, 1.0 - step1, step0, 1.0, 1.0, 1.0)
    if j == 30:
        if layer == 0:
            return (step0, 0.0, 1.0 - step1, 1.0, 1.0, 1.0)
        return (0.4 * step0, 0.0, 0.4 * step0, 1.0 - 0.4 * step0, step1, 1.0 - 0.4 * step0)
    if j == 31:
        if layer == 0:
            return (step0, 0.0, 1.0 - step1, 1.0, 1.0, 1.0)
        if layer == 1:
            return (step0, 1.0 - step1, 0.0, 1.0, 1.0, 1.0)
        return (0.0, 1.0 - step1, step0, 1.0, 1.0, 1.0)
    if j == 32:
        if layer == 0:
            return (step0, 0.0, 0.0, 1.0, 1.0, step1)
        return (0.4 * step0, 0.0, 0.4 * step0, 1.0 - 0.4 * step0, step1, 1.0 - 0.4 * step0)
    if j == 33:
        if layer == 0:
            return (step0, 0.0, 0.0, 1.0, 1.0, step1)
        if layer == 1:
            return (step0, 1.0 - step1, 0.0, 1.0, 1.0, 1.0)
        return (0.0, step0, 0.0, 1.0, 1.0, step1)
    if j == 34:
        if layer == 0:
            return (0.0, 0.0, 0.0, step1, 1.0, 1.0 - step0)
        return (0.4 * step0, 0.0, 0.4 * step0, 1.0 - 0.4 * step0, step1, 1.0 - 0.4 * step0)
    if j == 35:
        if layer == 0:
            return (0.0, 0.0, 0.0, step1, 1.0, 1.0 - step0)
        if layer == 1:
            return (0.0, step0, 0.0, step1, 1.0, 1.0)
        return (0.0, step0, 0.0, 1.0, 1.0, step1)

    if j == 36:
        return (layer0 + step0 * (1.0 - layer0), 0.0, 0.0, 1.0, layer1, step1 * (1.0 - layer0))
    if j == 38:
        return (0.0, 0.0, 0.0, step1 * (1.0 - layer0), layer1, 1.0 - layer0 - step0 * (1.0 - layer0))
    if j == 40:
        return (0.0, 0.0, layer0 + step0 * (1.0 - layer0), step1 * (1.0 - layer0), layer1, 1.0)
    if j == 42:
        return (layer0 + step0 * (1.0 - layer0), 0.0, 1.0 - step1 * (1.0 - layer0), 1.0, layer1, 1.0)
    if j == 43:
        return (layer0 + step0 * (1.0 - layer0), 1.0 - layer1, 1.0 - step1 * (1.0 - layer0), 1.0, 1.0, 1.0)
    if j == 44:
        return (0.5 * step0, 0.0, 0.5 * step0, 1.0 - 0.5 * step0, step1 * 0.5, 1.0 - 0.5 * step0)
    if j == 45:
        return (0.5 * step0, 1.0 - step1 * 0.5, 0.5 * step0, 1.0 - 0.5 * step0, 1.0, 1.0 - 0.5 * step0)
    return None


def slope_boxes_for_j(j: int) -> List[AABB]:
    if 12 <= j <= 19:
        layers = 2
        initial_width = 8
    elif 28 <= j <= 35:
        layers = 2
        initial_width = 8
    elif 36 <= j <= 43:
        layers = 8
        initial_width = 8
    elif j in {44, 45}:
        layers = 1
        initial_width = 4
    else:
        layers = 1
        initial_width = 8

    boxes: List[AABB] = []
    width = initial_width
    for layer in range(layers):
        for index in range(max(width, 1)):
            box = slope_piece_for_j(j, layer, layers, index, max(width, 1))
            if box is not None:
                boxes.append(box)
        if 36 <= j <= 43 and width > 1:
            width -= 1
    return boxes or [FULL_CUBE]


def slope_boxes(meta: int, double_slope: bool = False) -> List[AABB]:
    boxes = slope_boxes_for_j(SLOPE_META_TO_J.get(meta & 0xF, 0))

    if double_slope:
        return [FULL_CUBE] + [shift_box(box, dy=1.0) for box in boxes]
    return boxes


def rotate_external_box(box: AABB, rotation: int) -> AABB:
    x0, y0, z0, x1, y1, z1 = box
    rotation &= 3
    if rotation == 1:
        return (z0, y0, 1.0 - x1, z1, y1, 1.0 - x0)
    if rotation == 2:
        return (1.0 - x1, y0, 1.0 - z1, 1.0 - x0, y1, 1.0 - z0)
    if rotation == 3:
        return (1.0 - z1, y0, x0, 1.0 - z0, y1, x1)
    return box


def external_config_box(cfg: Dict[str, Any], meta: int) -> AABB:
    box = (
        float(cfg.get("min_x", 0.0)),
        float(cfg.get("min_y", 0.0)),
        float(cfg.get("min_z", 0.0)),
        float(cfg.get("max_x", 1.0)),
        float(cfg.get("max_y", 1.0)),
        float(cfg.get("max_z", 1.0)),
    )
    if cfg.get("rotatable"):
        return rotate_external_box(box, meta)
    return box


def model_cross_faces() -> List[Face]:
    return [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)),
    ]


def face_has_area(face: Face, eps: float = 1.0e-9) -> bool:
    unique = {tuple(round(coord, 9) for coord in vertex) for vertex in face}
    if len(unique) < 3:
        return False
    ax, ay, az = face[0]
    for i in range(1, len(face) - 1):
        bx, by, bz = face[i]
        cx, cy, cz = face[i + 1]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        cross = (
            uy * vz - uz * vy,
            uz * vx - ux * vz,
            ux * vy - uy * vx,
        )
        if cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2] > eps:
            return True
    return False


def translated_faces(faces: Iterable[Face], dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> List[Face]:
    return [
        tuple((x + dx, y + dy, z + dz) for x, y, z in face)
        for face in faces
    ]


def triangular_prism_faces(footprint: Sequence[Tuple[float, float]]) -> List[Face]:
    faces: List[Face] = []
    # Keep horizontal winding consistent with the real face direction. The
    # WEDGE_XZ double-slope wrapper translates these faces to y=1..2, where
    # material-side detection falls back to normals instead of y=0/y=1 bounds.
    bottom = tuple((x, 0.0, z) for x, z in footprint)
    if polygon_normal(bottom)[1] > 0.0:
        bottom = tuple(reversed(bottom))
    top = tuple((x, 1.0, z) for x, z in footprint)
    if polygon_normal(top)[1] < 0.0:
        top = tuple(reversed(top))
    faces.extend([bottom, top])
    for index, (x0, z0) in enumerate(footprint):
        x1, z1 = footprint[(index + 1) % len(footprint)]
        faces.append(((x0, 0.0, z0), (x1, 0.0, z1), (x1, 1.0, z1), (x0, 1.0, z0)))
    return [face for face in faces if face_has_area(face)]


def heightfield_faces(
    height_fn: Any,
    solid_below: bool,
    segments: int = 8,
) -> List[Face]:
    """Emit a smooth render surface for ProBuilder slope height fields.

    `161216` collision slices the same shapes into stair-step boxes. The real
    renderer (`175229`) writes sloped polygons; this helper keeps the OBJ mesh
    close to that render path without depending on texture/light state.
    """
    faces: List[Face] = []
    step = 1.0 / float(segments)

    def h(x: float, z: float) -> float:
        return max(0.0, min(1.0, float(height_fn(x, z))))

    if solid_below:
        faces.append(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    else:
        faces.append(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)))

    for zi in range(segments):
        z0 = zi * step
        z1 = (zi + 1) * step
        for xi in range(segments):
            x0 = xi * step
            x1 = (xi + 1) * step
            p00 = (x0, h(x0, z0), z0)
            p10 = (x1, h(x1, z0), z0)
            p11 = (x1, h(x1, z1), z1)
            p01 = (x0, h(x0, z1), z1)
            if solid_below:
                faces.extend([(p00, p10, p11), (p00, p11, p01)])
            else:
                faces.extend([(p00, p11, p10), (p00, p01, p11)])

    for i in range(segments):
        a = i * step
        b = (i + 1) * step
        north0 = h(a, 0.0)
        north1 = h(b, 0.0)
        south0 = h(a, 1.0)
        south1 = h(b, 1.0)
        west0 = h(0.0, a)
        west1 = h(0.0, b)
        east0 = h(1.0, a)
        east1 = h(1.0, b)

        if solid_below:
            faces.extend(
                [
                    ((a, 0.0, 0.0), (b, 0.0, 0.0), (b, north1, 0.0), (a, north0, 0.0)),
                    ((b, 0.0, 1.0), (a, 0.0, 1.0), (a, south0, 1.0), (b, south1, 1.0)),
                    ((0.0, 0.0, b), (0.0, 0.0, a), (0.0, west0, a), (0.0, west1, b)),
                    ((1.0, 0.0, a), (1.0, 0.0, b), (1.0, east1, b), (1.0, east0, a)),
                ]
            )
        else:
            faces.extend(
                [
                    ((a, north0, 0.0), (b, north1, 0.0), (b, 1.0, 0.0), (a, 1.0, 0.0)),
                    ((b, south1, 1.0), (a, south0, 1.0), (a, 1.0, 1.0), (b, 1.0, 1.0)),
                    ((0.0, west1, b), (0.0, west0, a), (0.0, 1.0, a), (0.0, 1.0, b)),
                    ((1.0, east0, a), (1.0, east1, b), (1.0, 1.0, b), (1.0, 1.0, a)),
                ]
            )

    return [face for face in faces if face_has_area(face)]


def transform_faces(
    faces: Iterable[Face],
    mapper: Any,
) -> List[Face]:
    return [
        tuple(mapper(x, y, z) for x, y, z in face)
        for face in faces
    ]


def scale_faces(
    faces: Iterable[Face],
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    x1: float = 1.0,
    y1: float = 1.0,
    z1: float = 1.0,
) -> List[Face]:
    sx = x1 - x0
    sy = y1 - y0
    sz = z1 - z0
    return transform_faces(faces, lambda x, y, z: (x0 + x * sx, y0 + y * sy, z0 + z * sz))


def polygon_normal(face: Face) -> Tuple[float, float, float]:
    nx = ny = nz = 0.0
    for index, (x0, y0, z0) in enumerate(face):
        x1, y1, z1 = face[(index + 1) % len(face)]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return nx, ny, nz


def polygon_center(face: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    count = len(face)
    if count == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(x for x, _y, _z in face) / count,
        sum(y for _x, y, _z in face) / count,
        sum(z for _x, _y, z in face) / count,
    )


def face_points_outward(
    face: Sequence[Tuple[float, float, float]],
    reference_center: Tuple[float, float, float],
    eps: float = 1.0e-7,
) -> bool:
    if len(face) < 3:
        return False
    nx, ny, nz = polygon_normal(tuple(face))
    cx, cy, cz = polygon_center(face)
    rx, ry, rz = reference_center
    dot = nx * (cx - rx) + ny * (cy - ry) + nz * (cz - rz)
    return dot > eps


def face_points_inward(
    face: Sequence[Tuple[float, float, float]],
    reference_center: Tuple[float, float, float],
    eps: float = 1.0e-7,
) -> bool:
    if len(face) < 3:
        return False
    nx, ny, nz = polygon_normal(tuple(face))
    cx, cy, cz = polygon_center(face)
    rx, ry, rz = reference_center
    dot = nx * (cx - rx) + ny * (cy - ry) + nz * (cz - rz)
    return dot < -eps


def face_points_against_axis(
    face: Sequence[Tuple[float, float, float]],
    axis: Tuple[float, float, float],
    eps: float = 1.0e-7,
) -> bool:
    if len(face) < 3:
        return False
    nx, ny, nz = polygon_normal(tuple(face))
    ax, ay, az = axis
    return nx * ax + ny * ay + nz * az < -eps


def clip_polygon_to_halfspace(face: Face, plane_value: Any, eps: float = 1.0e-7) -> Face:
    out: List[Tuple[float, float, float]] = []
    points = list(face)
    for index, current in enumerate(points):
        previous = points[index - 1]
        current_value = plane_value(*current)
        previous_value = plane_value(*previous)
        current_inside = current_value <= eps
        previous_inside = previous_value <= eps
        if current_inside != previous_inside:
            denom = previous_value - current_value
            t = 0.0 if abs(denom) < eps else previous_value / denom
            out.append(
                (
                    previous[0] + (current[0] - previous[0]) * t,
                    previous[1] + (current[1] - previous[1]) * t,
                    previous[2] + (current[2] - previous[2]) * t,
                )
            )
        if current_inside:
            out.append(current)
    return tuple(out)


def unique_points(points: Iterable[Tuple[float, float, float]], eps: float = 1.0e-6) -> List[Tuple[float, float, float]]:
    unique: List[Tuple[float, float, float]] = []
    for point in points:
        if not any(
            abs(point[0] - other[0]) <= eps and abs(point[1] - other[1]) <= eps and abs(point[2] - other[2]) <= eps
            for other in unique
        ):
            unique.append(point)
    return unique


def clean_face(face: Face, eps: float = 1.0e-6) -> Face:
    points: List[Tuple[float, float, float]] = []
    for point in face:
        if not points or any(abs(point[i] - points[-1][i]) > eps for i in range(3)):
            points.append(point)
    if len(points) > 1 and all(abs(points[0][i] - points[-1][i]) <= eps for i in range(3)):
        points.pop()

    changed = True
    while changed and len(points) >= 3:
        changed = False
        for index, point in enumerate(points):
            prev = points[index - 1]
            nxt = points[(index + 1) % len(points)]
            ax, ay, az = point[0] - prev[0], point[1] - prev[1], point[2] - prev[2]
            bx, by, bz = nxt[0] - point[0], nxt[1] - point[1], nxt[2] - point[2]
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            if cx * cx + cy * cy + cz * cz <= eps * eps:
                del points[index]
                changed = True
                break
    return tuple(points)


def plane_basis(normal: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    nx, ny, nz = normal
    if abs(ny) < 0.9:
        ref = (0.0, 1.0, 0.0)
    else:
        ref = (1.0, 0.0, 0.0)
    ux = ref[1] * nz - ref[2] * ny
    uy = ref[2] * nx - ref[0] * nz
    uz = ref[0] * ny - ref[1] * nx
    length = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    u = (ux / length, uy / length, uz / length)
    vx = ny * u[2] - nz * u[1]
    vy = nz * u[0] - nx * u[2]
    vz = nx * u[1] - ny * u[0]
    return u, (vx, vy, vz)


def clipped_cube_faces_for_plane(solid_below: bool, a: float, b: float, c: float) -> List[Face]:
    # Exact convex polyhedron for y <= a+b*x+c*z (or the inverted solid).
    # This avoids the visible 8x8 "ribbing" produced by sampling a planar slope.
    cube_faces: List[Face] = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)),
    ]

    if solid_below:
        normal = (-b, 1.0, -c)

        def plane_value(x: float, y: float, z: float) -> float:
            return y - (a + b * x + c * z)
    else:
        normal = (b, -1.0, c)

        def plane_value(x: float, y: float, z: float) -> float:
            return (a + b * x + c * z) - y

    faces: List[Face] = []
    cap_points: List[Tuple[float, float, float]] = []
    cube_edges = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
        ((1.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
        ((1.0, 1.0, 0.0), (1.0, 1.0, 1.0)),
    ]

    for face in cube_faces:
        clipped = clean_face(clip_polygon_to_halfspace(face, plane_value))
        if len(clipped) >= 3 and face_has_area(clipped):
            faces.append(clipped)

    for start, end in cube_edges:
        start_value = plane_value(*start)
        end_value = plane_value(*end)
        if abs(start_value) <= 1.0e-7:
            cap_points.append(start)
        if abs(end_value) <= 1.0e-7:
            cap_points.append(end)
        if start_value * end_value < 0.0:
            t = start_value / (start_value - end_value)
            cap_points.append(
                (
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t,
                    start[2] + (end[2] - start[2]) * t,
                )
            )

    cap = unique_points(cap_points)
    if len(cap) >= 3:
        center = (
            sum(p[0] for p in cap) / len(cap),
            sum(p[1] for p in cap) / len(cap),
            sum(p[2] for p in cap) / len(cap),
        )
        u, v = plane_basis(normal)
        cap.sort(
            key=lambda p: math.atan2(
                (p[0] - center[0]) * v[0] + (p[1] - center[1]) * v[1] + (p[2] - center[2]) * v[2],
                (p[0] - center[0]) * u[0] + (p[1] - center[1]) * u[1] + (p[2] - center[2]) * u[2],
            )
        )
        cap_face: Face = clean_face(tuple(cap))
        nx, ny, nz = polygon_normal(cap_face)
        if nx * normal[0] + ny * normal[1] + nz * normal[2] < 0.0:
            cap_face = tuple(reversed(cap_face))
        if face_has_area(cap_face):
            faces.append(cap_face)

    return [face for face in faces if face_has_area(face)]


def smooth_slope_spec(j: int) -> Optional[Tuple[bool, Any]]:
    # solid_below=True means y <= h(x,z); False means y >= h(x,z).
    match j:
        case 4:
            return False, lambda x, z: 1.0 - z
        case 5:
            return False, lambda x, z: z
        case 6:
            return False, lambda x, z: 1.0 - x
        case 7:
            return False, lambda x, z: x
        case 8:
            return True, lambda x, z: z
        case 9:
            return True, lambda x, z: 1.0 - z
        case 10:
            return True, lambda x, z: x
        case 11:
            return True, lambda x, z: 1.0 - x
        case 12:
            return True, lambda x, z: max(1.0 - x, z)
        case 13:
            return False, lambda x, z: min(x, 1.0 - z)
        case 14:
            return True, lambda x, z: max(x, z)
        case 15:
            return False, lambda x, z: min(1.0 - x, 1.0 - z)
        case 16:
            return True, lambda x, z: max(x, 1.0 - z)
        case 17:
            return False, lambda x, z: min(1.0 - x, z)
        case 18:
            return True, lambda x, z: max(1.0 - x, 1.0 - z)
        case 19:
            return False, lambda x, z: min(x, z)
        case 20:
            return True, lambda x, z: min(x, 1.0 - z)
        case 21:
            return False, lambda x, z: max(1.0 - x, z)
        case 22:
            return True, lambda x, z: min(1.0 - x, 1.0 - z)
        case 23:
            return False, lambda x, z: max(x, z)
        case 24:
            return True, lambda x, z: min(1.0 - x, z)
        case 25:
            return False, lambda x, z: max(x, 1.0 - z)
        case 26:
            return True, lambda x, z: min(x, z)
        case 27:
            return False, lambda x, z: max(1.0 - x, 1.0 - z)
        case 28:
            return True, lambda x, z: 1.0 + z - x
        case 29:
            return False, lambda x, z: x - z
        case 30:
            return True, lambda x, z: x + z
        case 31:
            return False, lambda x, z: 1.0 - x - z
        case 32:
            return True, lambda x, z: 1.0 + x - z
        case 33:
            return False, lambda x, z: z - x
        case 34:
            return True, lambda x, z: 2.0 - x - z
        case 35:
            return False, lambda x, z: x + z - 1.0
        case 36:
            return True, lambda x, z: x - z
        case 37:
            return False, lambda x, z: 1.0 + z - x
        case 38:
            return True, lambda x, z: 1.0 - x - z
        case 39:
            return False, lambda x, z: x + z
        case 40:
            return True, lambda x, z: z - x
        case 41:
            return False, lambda x, z: 1.0 + x - z
        case 42:
            return True, lambda x, z: x + z - 1.0
        case 43:
            return False, lambda x, z: 2.0 - x - z
        case 44:
            return True, lambda x, z: 0.5 - max(abs(x - 0.5), abs(z - 0.5))
        case 45:
            return False, lambda x, z: 0.5 + max(abs(x - 0.5), abs(z - 0.5))
    return None


def smooth_slope_plane_spec(j: int) -> Optional[Tuple[bool, float, float, float]]:
    # y = a + b*x + c*z. These shapes are a single clipped plane and should not
    # be tessellated into an 8x8 heightfield, otherwise OBJ viewers show ribs.
    match j:
        case 4:
            return False, 1.0, 0.0, -1.0
        case 5:
            return False, 0.0, 0.0, 1.0
        case 6:
            return False, 1.0, -1.0, 0.0
        case 7:
            return False, 0.0, 1.0, 0.0
        case 8:
            return True, 0.0, 0.0, 1.0
        case 9:
            return True, 1.0, 0.0, -1.0
        case 10:
            return True, 0.0, 1.0, 0.0
        case 11:
            return True, 1.0, -1.0, 0.0
        case 28:
            return True, 1.0, -1.0, 1.0
        case 29:
            return False, 0.0, 1.0, -1.0
        case 30:
            return True, 0.0, 1.0, 1.0
        case 31:
            return False, 1.0, -1.0, -1.0
        case 32:
            return True, 1.0, 1.0, -1.0
        case 33:
            return False, 0.0, -1.0, 1.0
        case 34:
            return True, 2.0, -1.0, -1.0
        case 35:
            return False, -1.0, 1.0, 1.0
        case 36:
            return True, 0.0, 1.0, -1.0
        case 37:
            return False, 1.0, -1.0, 1.0
        case 38:
            return True, 1.0, -1.0, -1.0
        case 39:
            return False, 0.0, 1.0, 1.0
        case 40:
            return True, 0.0, -1.0, 1.0
        case 41:
            return False, 1.0, 1.0, -1.0
        case 42:
            return True, -1.0, 1.0, 1.0
        case 43:
            return False, 2.0, -1.0, -1.0
    return None


def smooth_slope_faces_for_j(j: int) -> List[Face]:
    if j == 0:
        return triangular_prism_faces([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    if j == 1:
        return triangular_prism_faces([(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    if j == 2:
        return triangular_prism_faces([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    if j == 3:
        return triangular_prism_faces([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])

    plane_spec = smooth_slope_plane_spec(j)
    if plane_spec is not None:
        solid_below, a, b, c = plane_spec
        return clipped_cube_faces_for_plane(solid_below, a, b, c)

    spec = smooth_slope_spec(j)
    if spec is None:
        return []
    solid_below, height_fn = spec
    return heightfield_faces(height_fn, solid_below)


def smooth_slope_supported(j: int) -> bool:
    return 0 <= j <= 45


def cover_id(raw_cover: int) -> int:
    return (raw_cover & 0xFFFF) & 0x0FFF


def cover_meta(raw_cover: int) -> int:
    return ((raw_cover & 0xFFFF) >> 12) & 0xF


MASK64 = (1 << 64) - 1


def tile_payload_u64(tile: ProBuilderTile) -> int:
    return ((tile.data_ext & 0xFFFFFFFF) << 32) | (tile.data & 0xFFFFFFFF)


def grid_bit(payload: int, x: int, y: int, z: int) -> bool:
    if x < 0 or x > 3 or y < 0 or y > 3 or z < 0 or z > 3:
        return False
    return bool((payload & MASK64) & (1 << (y * 16 + z * 4 + x)))


def grid_side_scan(side: str, payload: int, a: int, b: int) -> int:
    # Port of 103413.do(25940,long,int,int): find the first occupied voxel on
    # a 4x4x4 ProBuilder grid along one side.
    if side == "UP":
        for y in range(3, -1, -1):
            if grid_bit(payload, a, y, b):
                return y
        return 0
    if side == "DOWN":
        for y in range(4):
            if grid_bit(payload, a, y, b):
                return y
        return 3
    if side == "SOUTH":
        for z in range(3, -1, -1):
            if grid_bit(payload, a, b, z):
                return z
        return 0
    if side == "NORTH":
        for z in range(4):
            if grid_bit(payload, a, b, z):
                return z
        return 3
    if side == "EAST":
        for x in range(3, -1, -1):
            if grid_bit(payload, x, b, a):
                return x
        return 0
    if side == "WEST":
        for x in range(4):
            if grid_bit(payload, x, b, a):
                return x
        return 3
    return 0


def grid_side_value(side: str, payload: int, a: int, b: int) -> int:
    value = grid_side_scan(side, payload, a, b)
    if side in {"UP", "SOUTH", "EAST"}:
        return 3 - value
    return value


def grid_min_side(side: str, payload: int) -> int:
    value = 3
    for b in range(4):
        for a in range(4):
            value = min(value, grid_side_value(side, payload, a, b))
    return value


def probuilder_grid_boxes(tile: ProBuilderTile) -> List[AABB]:
    """Port 132769.else(): merge a 4x4x4 141081 voxel payload into AABBs."""
    payload = tile_payload_u64(tile)
    if payload == 0:
        return []
    if (payload & 0xFFFF000000000000) == 0xFFFF000000000000 and (payload & 0xFFFF) == 0xFFFF:
        return [FULL_CUBE]

    z0 = grid_min_side("NORTH", payload)
    z1 = 4 - grid_min_side("SOUTH", payload)
    x0 = grid_min_side("WEST", payload)
    x1 = 4 - grid_min_side("EAST", payload)
    if x1 <= x0 or z1 <= z0:
        return []

    width = x1 - x0
    visited = 0
    cursor = 0
    boxes: List[AABB] = []
    z = z0
    while z < z1:
        x = x0
        while x < x1:
            bit = 1 << cursor
            if visited & bit:
                cursor += 1
                x += 1
                continue

            down = grid_side_scan("DOWN", payload, x, z)
            up = grid_side_scan("UP", payload, x, z)
            if down == 3 and up == 0 and not any(grid_bit(payload, x, y, z) for y in range(4)):
                cursor += 1
                x += 1
                continue

            run_w = 1
            while x + run_w < x1:
                if visited & (1 << (cursor + run_w)):
                    break
                if grid_side_scan("DOWN", payload, x + run_w, z) != down:
                    break
                if grid_side_scan("UP", payload, x + run_w, z) != up:
                    break
                run_w += 1

            run_d = 1
            while z + run_d < z1:
                stop = False
                for dx in range(run_w):
                    next_cursor = cursor + dx + run_d * width
                    if visited & (1 << next_cursor):
                        stop = True
                        break
                    if grid_side_scan("DOWN", payload, x + dx, z + run_d) != down:
                        stop = True
                        break
                    if grid_side_scan("UP", payload, x + dx, z + run_d) != up:
                        stop = True
                        break
                if stop:
                    break
                run_d += 1

            boxes.append((x * 0.25, down * 0.25, z * 0.25, (x + run_w) * 0.25, (up + 1) * 0.25, (z + run_d) * 0.25))

            for dz in range(run_d):
                for dx in range(run_w):
                    visited |= 1 << (cursor + dx + dz * width)
            x += run_w
            cursor += run_w
        z += 1
    return boxes


GRID_FACE_DEFS = [
    ("DOWN", (0, -1, 0), lambda x0, y0, z0, x1, y1, z1: ((x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0))),
    ("UP", (0, 1, 0), lambda x0, y0, z0, x1, y1, z1: ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))),
    ("NORTH", (0, 0, -1), lambda x0, y0, z0, x1, y1, z1: ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0))),
    ("SOUTH", (0, 0, 1), lambda x0, y0, z0, x1, y1, z1: ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
    ("WEST", (-1, 0, 0), lambda x0, y0, z0, x1, y1, z1: ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))),
    ("EAST", (1, 0, 0), lambda x0, y0, z0, x1, y1, z1: ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1))),
]
GRID_FACE_BY_NAME = {name: (offset, make_face) for name, offset, make_face in GRID_FACE_DEFS}


def probuilder_grid_render_faces(tile: ProBuilderTile) -> List[Face]:
    payload = tile_payload_u64(tile)
    if payload == 0:
        return []
    step = 0.25
    faces: List[Face] = []
    for y in range(4):
        for z in range(4):
            for x in range(4):
                if not grid_bit(payload, x, y, z):
                    continue
                x0, y0, z0 = x * step, y * step, z * step
                x1, y1, z1 = x0 + step, y0 + step, z0 + step
                for _name, (dx, dy, dz), make_face in GRID_FACE_DEFS:
                    if not grid_bit(payload, x + dx, y + dy, z + dz):
                        faces.append(make_face(x0, y0, z0, x1, y1, z1))
    return faces


SIDE_NAMES = ("DOWN", "UP", "NORTH", "SOUTH", "WEST", "EAST", "UNKNOWN")
SIDE_INDEX = {name: index for index, name in enumerate(SIDE_NAMES[:6])}
SIDE_OFFSET = {name: offset for name, offset, _make_face in GRID_FACE_DEFS}
SIDE_AXIS = {
    "DOWN": (0.0, -1.0, 0.0),
    "UP": (0.0, 1.0, 0.0),
    "NORTH": (0.0, 0.0, -1.0),
    "SOUTH": (0.0, 0.0, 1.0),
    "WEST": (-1.0, 0.0, 0.0),
    "EAST": (1.0, 0.0, 0.0),
}
OPPOSITE_SIDE = {
    "DOWN": "UP",
    "UP": "DOWN",
    "NORTH": "SOUTH",
    "SOUTH": "NORTH",
    "WEST": "EAST",
    "EAST": "WEST",
}

# 143634 WEDGE_XZ shapes split a block cell diagonally in X/Z. The listed
# sides are the empty half-space directions for the vertical diagonal cut.
WEDGE_XZ_EMPTY_SIDES = {
    0: ("SOUTH", "EAST"),
    1: ("NORTH", "WEST"),
    2: ("NORTH", "EAST"),
    3: ("SOUTH", "WEST"),
}
# 200242.goto(WEDGE_XZ) emits the vertical diagonal split through a NORTH
# helper when the shape's switch list contains NORTH, otherwise through SOUTH.
WEDGE_XZ_DIAGONAL_RENDER_SIDE = {
    0: "SOUTH",
    1: "NORTH",
    2: "NORTH",
    3: "SOUTH",
}
# Grass neighbor graft is a terrain blending quirk, not a general rule for
# all ProBuilder wedges. Keep it away from explicit custom/decorative covers
# such as customitems:tiles_concrete.
GRASS_NEIGHBOR_GRAFT_BASE_IDS = {1, 2, 3, 12, 13, 87, 3558}
THICK_COVER_BLOCK_IDS = {78, 80}  # 275700.H snow and 275700.public snow block.
INVISIBLE_MATERIAL_CLASS_IDS = {273574, 97976, 158138, 40409}
OBJ_MATERIAL_VISIBLE = "stc_visible"
OBJ_MATERIAL_COVER = "stc_cover"
OBJ_MATERIAL_INVISIBLE = "stc_invisible_carrier"
OBJ_INVISIBLE_MATERIAL_BY_BLOCK_ID = {
    3928: "stc_invisible_3928_fake_light",
    3929: "stc_invisible_3929_fake_light",
    4018: "stc_invisible_4018_impblock",
    4023: "stc_invisible_4023_interference",
    4024: "stc_invisible_4024_interference",
    4025: "stc_invisible_4025_inv",
}
INVISIBLE_MATERIAL_ROLE_DIRECT = "direct"
INVISIBLE_MATERIAL_ROLE_COVER_CARRIER = "cover_carrier"
INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE = "packed_surface"
INVISIBLE_MATERIAL_ROLE_PACKED_BODY = "packed_body"


@dataclass(frozen=True)
class CTMRule:
    source: Path
    base_path: str
    match_blocks: Set[int]
    match_metadata_by_block: Dict[int, Set[int]]
    metadata: Optional[Set[int]]
    tiles: List[str]
    width: int
    height: int
    min_height: int = -1
    max_height: int = 1024


@dataclass
class MaterialDefinition:
    name: str
    texture_rel: Optional[str] = None
    alpha_map_rel: Optional[str] = None
    color: Tuple[float, float, float] = (0.72, 0.72, 0.72)
    alpha: float = 1.0
    note: str = ""


@dataclass
class FaceMaterialDescriptor:
    """Resolved face material for the glTF writer.

    icon is the canonical sprite path (`namespace:path/to/sprite`) or `None`
    for fallback materials. png_bytes carry raw embeddable PNG bytes
    when the sprite was found in `blockMap.texarr`. alpha_mode is the glTF
    alphaMode value (`OPAQUE` / `MASK` / `BLEND`). tint is a multiplicative
    RGB factor applied as a per-vertex `COLOR_0` attribute.
    """

    icon: Optional[str]
    png_bytes: Optional[bytes]
    alpha_mode: str
    tint: Optional[Tuple[float, float, float]]
    emissive: Optional[Tuple[float, float, float]] = None
    note: str = ""


def decrypt_texarr_ta(data: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "blockMap.texarr is missing and blockMap.ta fallback requires cryptography. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    decryptor = Cipher(
        algorithms.AES(TEXARR_TA_AES_KEY),
        modes.CBC(TEXARR_TA_AES_IV),
    ).decryptor()
    padded = decryptor.update(data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def ensure_texarr_available(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".ta":
        ta_path = path
        texarr_path = path.with_suffix(".texarr")
    else:
        texarr_path = path
        ta_path = path.with_suffix(".ta")

    if texarr_path.exists() and texarr_path.stat().st_size > 4:
        return texarr_path
    if not ta_path.exists():
        return path

    print(f"[texarr] {texarr_path} missing or empty; decrypting fallback {ta_path}")
    plain = decrypt_texarr_ta(ta_path.read_bytes())
    texarr_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = texarr_path.with_name(f"{texarr_path.name}.tmp")
    temp_path.write_bytes(plain)
    temp_path.replace(texarr_path)
    return texarr_path


class TextureArrayIndex:
    """Lazy index/extractor for STALCRAFT blockMap.texarr DDS entries."""

    def __init__(self, path: Path) -> None:
        self.path = ensure_texarr_available(path)
        self.entries: Dict[str, Tuple[int, int]] = {}
        if self.path.exists():
            self._build()

    def _build(self) -> None:
        with self.path.open("rb") as handle:
            count_data = handle.read(4)
            if len(count_data) != 4:
                return
            count = struct.unpack(">I", count_data)[0]
            for _ in range(count):
                length_data = handle.read(2)
                if len(length_data) != 2:
                    break
                name_len = struct.unpack(">H", length_data)[0]
                name = handle.read(name_len).decode("utf-8", errors="replace")
                size_data = handle.read(4)
                if len(size_data) != 4:
                    break
                size = struct.unpack(">I", size_data)[0]
                data_offset = handle.tell()
                self.entries[name] = (data_offset, size)
                handle.seek(size, 1)

    def has(self, icon_name: str) -> bool:
        return icon_name in self.entries

    def read(self, icon_name: str) -> Optional[bytes]:
        entry = self.entries.get(icon_name)
        if entry is None:
            return None
        offset, size = entry
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(size)


def parse_int_set(text: Optional[str]) -> Optional[Set[int]]:
    if not text:
        return None
    values: Set[int] = set()
    for raw_part in re.split(r"[\s,]+", text.strip()):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            left, right = part.split("-", 1)
            if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
                lo = int(left)
                hi = int(right)
                if lo <= hi:
                    values.update(range(lo, hi + 1))
                continue
        if part.lstrip("-").isdigit():
            values.add(int(part))
    return values


def parse_ctm_match_blocks(text: Optional[str]) -> Tuple[Set[int], Dict[int, Set[int]]]:
    blocks: Set[int] = set()
    metadata_by_block: Dict[int, Set[int]] = {}
    if not text:
        return blocks, metadata_by_block
    for raw_part in re.split(r"[\s,]+", text.strip()):
        part = raw_part.strip()
        if not part:
            continue
        meta: Optional[int] = None
        if "\\" in part:
            block_part, meta_part = part.split("\\", 1)
            part = block_part
            if meta_part.lstrip("-").isdigit():
                meta = int(meta_part)
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            left, right = part.split("-", 1)
            if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
                for block_id in range(int(left), int(right) + 1):
                    blocks.add(block_id)
                    if meta is not None:
                        metadata_by_block.setdefault(block_id, set()).add(meta)
                continue
        if part.lstrip("-").isdigit():
            block_id = int(part)
            blocks.add(block_id)
            if meta is not None:
                metadata_by_block.setdefault(block_id, set()).add(meta)
    return blocks, metadata_by_block


def parse_properties_file(path: Path) -> Dict[str, str]:
    props: Dict[str, str] = {}
    pending = ""
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if pending:
            line = pending + line
            pending = ""
        if line.endswith("\\"):
            pending = line[:-1]
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        props[key.strip()] = value.strip()
    return props


def expand_tile_tokens(text: Optional[str]) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    for raw_part in re.split(r"[\s,]+", text.strip()):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            left, right = part.split("-", 1)
            if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
                lo = int(left)
                hi = int(right)
                if lo <= hi:
                    out.extend(str(value) for value in range(lo, hi + 1))
                    continue
        out.append(part)
    return out


def normalize_ctm_tile_icon(base_path: str, token: str) -> str:
    tile = token.strip().replace("\\", "/")
    if tile.endswith(".png"):
        tile = tile[:-4]
    if tile.startswith("textures/blocks/"):
        tile = tile[len("textures/blocks/") :]
    elif tile.startswith("textures/"):
        tile = tile[len("textures/") :]
    elif not tile.startswith("ctmpatcher/"):
        tile = f"{base_path}/{tile}"
    tile = tile.lstrip("/")
    return f"stalcraft:{tile}"


def rotate_square_tiles(tiles: List[str], width: int, height: int, rotations: int) -> List[str]:
    if width != height or width <= 1:
        return tiles
    current = list(tiles)
    size = width
    for _ in range(rotations % 4):
        rotated = list(current)
        for y in range(size):
            for x in range(size):
                rotated[y * size + x] = current[(size - 1 - x) * size + y]
        current = rotated
    return current


def safe_material_name(prefix: str, raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not cleaned:
        cleaned = "unnamed"
    return f"{prefix}_{cleaned[:180]}"


def icon_to_asset_path(icon_name: str, suffix: str) -> Path:
    namespace, _, rest = icon_name.partition(":")
    if not rest:
        namespace = "unknown"
        rest = icon_name
    safe_parts = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in rest.replace("\\", "/").split("/") if part]
    return Path(namespace, *safe_parts).with_suffix(suffix)


def rgb565_to_rgb(value: int) -> Tuple[float, float, float]:
    value &= 0xFFFF
    r = ((value & 0xF800) >> 11) / 31.0
    g = ((value & 0x07E0) >> 5) / 63.0
    b = (value & 0x001F) / 31.0
    return (r, g, b)


class JavaRandom:
    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self.seed = 0
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = (int(seed) ^ self.MULTIPLIER) & self.MASK

    def next_bits(self, bits: int) -> int:
        self.seed = (self.seed * self.MULTIPLIER + self.ADDEND) & self.MASK
        return self.seed >> (48 - bits)

    def next_signed_int32(self) -> int:
        value = self.next_bits(32)
        return value - (1 << 32) if value >= (1 << 31) else value

    def next_long(self) -> int:
        return (self.next_signed_int32() << 32) + self.next_signed_int32()

    def next_float(self) -> float:
        return self.next_bits(24) / float(1 << 24)

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & -bound) == bound:
            return (bound * self.next_bits(31)) >> 31
        while True:
            bits = self.next_bits(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value


class TextureResolver:
    def __init__(
        self,
        map_blocks_path: Path,
        texarr_path: Path,
        ctm_dir: Path,
        output_path: Path,
        external_configs: Optional[Dict[int, Dict[str, Any]]] = None,
        type_textures: Optional[Dict[int, TypeTextureInfo]] = None,
        block_type_by_id: Optional[Dict[int, int]] = None,
        world_blocks: Optional[Dict[Tuple[int, int, int], BlockState]] = None,
        world_biomes: Optional[Dict[Tuple[int, int], int]] = None,
        biome_color_table: Optional[Dict[int, Tuple[float, float, float]]] = None,
        texture_format: str = "png",
        enable_ctm: bool = True,
        snow_block_ids: Optional[Set[int]] = None,
    ) -> None:
        self.map_blocks_path = map_blocks_path
        self.texarr = TextureArrayIndex(texarr_path)
        self.ctm_dir = ctm_dir
        self.texture_root = output_path.with_name(f"{output_path.stem}_textures")
        self.texture_format = texture_format
        self.enable_ctm = enable_ctm
        self.block_icons = self._load_block_icons(map_blocks_path)
        self.external_configs = external_configs or {}
        self.type_textures = type_textures or {}
        self.block_type_by_id = block_type_by_id or {}
        self.world_blocks = world_blocks or {}
        self.world_biomes = world_biomes or {}
        self.biome_color_table = biome_color_table or {}
        self.ctm_rules_by_block = self._load_ctm_rules(ctm_dir) if enable_ctm else {}
        self.materials: Dict[str, MaterialDefinition] = {}
        self.icon_materials: Dict[Tuple[str, Optional[Tuple[int, int, int]]], str] = {}
        self.extracted_icons: Dict[str, str] = {}
        self.tinted_icons: Dict[Tuple[str, Tuple[int, int, int]], str] = {}
        self.alpha_icons: Set[str] = set()
        # Cutout sprites have alpha histogram dominated by 0/255; translucent
        # sprites carry intermediate alpha values (water, glass, fog). The
        # split drives glTF `alphaMode` MASK vs BLEND.
        self.translucent_icons: Set[str] = set()
        self.alpha_texture_rels: Set[str] = set()
        self.missing_icons: Set[str] = set()
        self.convert_fallbacks: Set[str] = set()
        self._png_bytes_cache: Dict[str, Optional[bytes]] = {}
        # Class-derived snow-like materials (`288455.new` / `288455.for`).
        # Falls back to vanilla snow id 80 if no source-table scan was provided so
        # legacy callers keep working without passing the parameter.
        self.snow_block_ids: Set[int] = set(snow_block_ids) if snow_block_ids else set(SNOW_LIKE_BLOCK_ID_OVERRIDES)

    @staticmethod
    def _load_block_icons(path: Path) -> Dict[Tuple[int, int], str]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[Tuple[int, int], str] = {}
        for row in raw:
            if not isinstance(row, dict):
                continue
            block_id = row.get("id")
            meta = row.get("meta", 0)
            icon = row.get("iconName")
            if isinstance(block_id, int) and isinstance(meta, int) and isinstance(icon, str):
                out[(block_id, meta & 0xF)] = icon
        return out

    def _load_ctm_rules(self, ctm_dir: Path) -> Dict[int, List[CTMRule]]:
        rules_by_block: Dict[int, List[CTMRule]] = {}
        if not ctm_dir.exists():
            return rules_by_block
        for path in sorted(ctm_dir.rglob("*.properties"), key=lambda item: item.as_posix().lower()):
            props = parse_properties_file(path)
            if props.get("method", "repeat") != "repeat":
                continue
            match_blocks, match_meta = parse_ctm_match_blocks(props.get("matchBlocks"))
            if not match_blocks:
                continue
            width = int(props.get("width", "-1")) if props.get("width", "-1").lstrip("-").isdigit() else -1
            height = int(props.get("height", "-1")) if props.get("height", "-1").lstrip("-").isdigit() else -1
            if width <= 0 or height <= 0:
                continue
            rel_parent = path.relative_to(ctm_dir).parent.as_posix()
            base_path = "ctmpatcher/ctm" if rel_parent == "." else f"ctmpatcher/ctm/{rel_parent}"
            tiles = [normalize_ctm_tile_icon(base_path, token) for token in expand_tile_tokens(props.get("tiles"))]
            if len(tiles) < width * height:
                continue
            metadata = parse_int_set(props.get("metadata"))
            min_height = int(props.get("minHeight", "-1")) if props.get("minHeight", "-1").lstrip("-").isdigit() else -1
            max_height = int(props.get("maxHeight", "1024")) if props.get("maxHeight", "1024").lstrip("-").isdigit() else 1024
            rule = CTMRule(
                source=path,
                base_path=base_path,
                match_blocks=match_blocks,
                match_metadata_by_block=match_meta,
                metadata=metadata,
                tiles=tiles,
                width=width,
                height=height,
                min_height=min_height,
                max_height=max_height,
            )
            for block_id in match_blocks:
                rules_by_block.setdefault(block_id, []).append(rule)
        return rules_by_block

    def canonical_icon_name(self, icon_name: Optional[str]) -> Optional[str]:
        if not icon_name:
            return None
        candidates = [icon_name]
        if ":" not in icon_name:
            candidates.insert(0, f"stalcraft:{icon_name}")
        for candidate in candidates:
            if self.texarr.has(candidate):
                return candidate
        # 88276.catch(String) namespaces bare block icons as stalcraft:* before
        # lookup. Return that canonical name even if the texture is currently
        # missing so diagnostics point at the same key the game would use.
        if ":" not in icon_name:
            return f"stalcraft:{icon_name}"
        return icon_name

    def block_above_is_snow_like(self, pos: Optional[Tuple[int, int, int]]) -> bool:
        if pos is None:
            return False
        x, y, z = pos
        above = self.world_blocks.get((x, y + 1, z))
        if above is None:
            return False
        return above.block_id in self.snow_block_ids

    @staticmethod
    def axis_uses_end_texture(meta: int, side: str) -> bool:
        side_index = SIDE_INDEX.get(side.upper(), -1)
        orientation = meta & 0xC
        return (
            (orientation == 0 and side_index in {0, 1})
            or (orientation == 4 and side_index in {4, 5})
            or (orientation == 8 and side_index in {2, 3})
        )

    def slab_icon_name(self, meta: int, side: str, pos: Optional[Tuple[int, int, int]]) -> Optional[str]:
        slab_kind = meta & 7
        if slab_kind == 0:
            if side.upper() in {"UP", "DOWN"}:
                return self.canonical_icon_name("stone_slab_top")
            return self.canonical_icon_name("stone_slab_side")
        delegates = {
            1: (24, meta),
            2: (5, 0),
            3: (4, 0),
            4: (45, 0),
            5: (98, 0),
            6: (112, 0),
            7: (155, 0),
        }
        delegate = delegates.get(slab_kind)
        if delegate is None:
            return self.canonical_icon_name("stone_slab_top")
        delegate_id, delegate_meta = delegate
        delegate_side = "UP" if slab_kind == 6 else side
        return self.block_icon_name(delegate_id, delegate_meta, delegate_side, pos)

    def b7_icon_name(self, base_icon: Optional[str], meta: int, side: str) -> Optional[str]:
        """Port of 156120.char(side, meta): b7 has top/bottom/side/meta variants."""
        base = base_icon or "b7"
        side_index = SIDE_INDEX.get(side.upper(), 1)
        m = meta & 0xF
        if m not in {2, 3, 4}:
            if side_index == 1 or (side_index == 0 and m == 1):
                suffix = "1_top" if m == 1 else "top"
                return self.canonical_icon_name(f"{base}_{suffix}")
            if side_index == 0:
                return self.canonical_icon_name(f"{base}_bottom")
            suffix_by_meta = {0: "side", 1: "1", 2: "2", 3: "2", 4: "2"}
            return self.canonical_icon_name(f"{base}_{suffix_by_meta.get(m, 'side')}")

        if (
            (m == 2 and side_index in {0, 1})
            or (m == 3 and side_index in {4, 5})
            or (m == 4 and side_index in {2, 3})
        ):
            return self.canonical_icon_name(f"{base}_2_top")
        return self.canonical_icon_name(f"{base}_2")

    def pane_icon_name(self, info: TypeTextureInfo, base_icon: Optional[str], side: str) -> Optional[str]:
        # 106044 registers both the broad pane sprite and the thin edge/top sprite.
        # The renderer asks char() for edge passes; for our box geometry the
        # horizontal caps are the important place where the edge texture matters.
        if side.upper() in {"UP", "DOWN"}:
            return self.canonical_icon_name(info.pane_edge_icon or info.pane_icon or base_icon)
        return self.canonical_icon_name(info.pane_icon or base_icon)

    def wall_icon_name(self, meta: int, side: str, pos: Optional[Tuple[int, int, int]]) -> Optional[str]:
        # 196560.char(side, meta): metadata 1 delegates to mossy cobble (J),
        # otherwise to cobblestone. The type itself registers no icon.
        delegate_id = 48 if (meta & 0xF) == 1 else 4
        return self.block_icon_name(delegate_id, 0, side, pos)

    def monster_egg_icon_name(self, meta: int, side: str, pos: Optional[Tuple[int, int, int]]) -> Optional[str]:
        # 285096.char(side, meta): silverfish/monster egg copies one of the
        # normal block textures instead of registering its own sprites.
        m = meta & 0xF
        if m == 1:
            return self.block_icon_name(4, 0, side, pos)
        if m == 2:
            return self.block_icon_name(98, 0, side, pos)
        return self.block_icon_name(1, 0, side, pos)

    def wooden_slab_icon_name(self, meta: int, side: str, pos: Optional[Tuple[int, int, int]]) -> Optional[str]:
        # 36148.char(side, meta): both single/double wooden slabs delegate to
        # planks with the lower three metadata bits.
        return self.block_icon_name(5, meta & 0x7, side, pos)

    def door_combined_meta(self, block_id: int, meta: int, pos: Optional[Tuple[int, int, int]]) -> int:
        # 230253.const(world,x,y,z) combines lower orientation/open bits with the
        # upper half hinge bit. If the paired half is outside the export, fall back
        # to the local metadata so standalone ranges still texture deterministically.
        upper = (meta & 0x8) != 0
        lower_meta = meta
        upper_meta = meta
        if pos is not None:
            x, y, z = pos
            if upper:
                lower = self.world_blocks.get((x, y - 1, z))
                if lower is not None and lower.block_id == block_id:
                    lower_meta = lower.meta
            else:
                above = self.world_blocks.get((x, y + 1, z))
                if above is not None and above.block_id == block_id:
                    upper_meta = above.meta
        hinge = (upper_meta & 0x1) != 0
        return (lower_meta & 0x7) | (0x8 if upper else 0) | (0x10 if hinge else 0)

    def door_icon_name(
        self,
        block_id: int,
        meta: int,
        side: str,
        pos: Optional[Tuple[int, int, int]],
        base_icon: Optional[str],
    ) -> Optional[str]:
        base = base_icon or "doori"
        side_index = SIDE_INDEX.get(side.upper(), 1)
        if side_index in {0, 1}:
            return self.canonical_icon_name(f"{base}l")
        combined = self.door_combined_meta(block_id, meta, pos)
        # The game may mirror UVs through 37058 for one face orientation. Material
        # choice is still upper/lower; UV mirroring can be layered in separately.
        suffix = "u" if (combined & 0x8) else "l"
        return self.canonical_icon_name(f"{base}{suffix}")

    def net_block_icon_name(self, block_id: int, meta: int, side: str) -> Optional[str]:
        variant = 2 if block_id == 100 else 1
        side_index = SIDE_INDEX.get(side.upper(), 1)
        m = meta & 0xF
        use_outer = (
            (1 <= m <= 9 and side_index == 1)
            or (1 <= m <= 3 and side_index == 2)
            or (7 <= m <= 9 and side_index == 3)
            or (m in {1, 4, 7} and side_index == 4)
            or (m in {3, 6, 9} and side_index == 5)
            or m == 14
        )
        if m == 10 and side_index > 1:
            return self.canonical_icon_name("net_block_stem")
        if m == 15:
            return self.canonical_icon_name("net_block_stem")
        if use_outer:
            return self.canonical_icon_name(f"net_block_{variant}")
        return self.canonical_icon_name("net_block_inside")

    def class_block_icon_name(self, block_id: int, meta: int, side: str, pos: Optional[Tuple[int, int, int]]) -> Optional[str]:
        info = self.type_textures.get(block_id)
        type_id = info.type_id if info is not None else self.block_type_by_id.get(block_id)
        base_icon = info.base_icon if info is not None else None
        side = side.upper()
        m = meta & 0xF

        if info is not None and info.wrapper_base_id is not None:
            return self.block_icon_name(info.wrapper_base_id, info.wrapper_meta, side, pos)

        if block_id == 31 or type_id == 125838:
            plant_icons = {
                0: "deadbush",
                1: "tallgrass",
                2: "fern",
            }
            return self.canonical_icon_name(plant_icons.get(m, "deadbush"))
        if block_id == 32 or type_id == 8358:
            return self.canonical_icon_name("deadbush")

        if type_id == 111004:
            if side == "DOWN":
                return self.block_icon_name(3, 0, side, pos)
            if side == "UP":
                return self.canonical_icon_name(f"{base_icon or 'grass'}_top")
            if self.block_above_is_snow_like(pos):
                return self.canonical_icon_name(f"{base_icon or 'grass'}_side_snowed")
            return self.canonical_icon_name(f"{base_icon or 'grass'}_side")

        if type_id == 120208:
            if side == "DOWN":
                return self.block_icon_name(3, 0, side, pos)
            if side == "UP":
                return self.canonical_icon_name(f"{base_icon or 'dirt2'}_top")
            if self.block_above_is_snow_like(pos):
                return self.canonical_icon_name("grass_side_snowed")
            return self.canonical_icon_name(f"{base_icon or 'dirt2'}_side")

        if type_id == 222639:
            log_index = (m & 0x3) + 1
            suffix = "_top" if self.axis_uses_end_texture(m, side) else ""
            return self.canonical_icon_name(f"{base_icon or 'log'}_{log_index}{suffix}")

        if type_id == 149960:
            suffix = "top" if self.axis_uses_end_texture(m, side) else "side"
            return self.canonical_icon_name(f"{base_icon or 'comp'}_{suffix}")

        if type_id == 10357 and base_icon:
            return self.canonical_icon_name(f"{base_icon}_{(~m) & 0xF}")

        if type_id == 201816:
            return self.canonical_icon_name(f"leaves_{(m & 0x3) + 1}")

        if type_id == 284366:
            return self.slab_icon_name(m, side, pos)

        if type_id == 36148:
            return self.wooden_slab_icon_name(meta, side, pos)

        if type_id == 156120:
            return self.b7_icon_name(base_icon, m, side)

        if type_id == 196560:
            return self.wall_icon_name(m, side, pos)

        if type_id == 285096:
            return self.monster_egg_icon_name(m, side, pos)

        if type_id == 27862:
            return self.block_icon_name(1, 0, "UP", pos)

        if type_id == 111389:
            return self.block_icon_name(5, 0, "UP", pos)

        if type_id == 230253:
            return self.door_icon_name(block_id, meta, side, pos, base_icon)

        if type_id == 54696:
            if side == "UP" or (side == "DOWN" and m in {1, 2}):
                return self.canonical_icon_name(f"{base_icon or 'sandstone'}_top")
            if side == "DOWN":
                return self.canonical_icon_name(f"{base_icon or 'sandstone'}_bottom")
            return self.canonical_icon_name(f"{base_icon or 'sandstone'}_{min(m + 1, 3)}")

        if type_id == 80107:
            if side in {"UP", "DOWN"}:
                return self.canonical_icon_name(f"{base_icon or 'tv'}_top")
            face_side_by_meta = {2: "NORTH", 3: "EAST", 0: "SOUTH", 1: "WEST"}
            is_on = block_id == 91
            if face_side_by_meta.get(m & 0x3) == side:
                return self.canonical_icon_name(f"{base_icon or 'tv'}_face_{'on' if is_on else 'off'}")
            return self.canonical_icon_name(f"{base_icon or 'tv'}_side")

        if type_id == 55631 and base_icon:
            return self.canonical_icon_name(f"{base_icon}_{m + 1 if 0 <= m < 4 else 1}")

        if type_id == 151717 and base_icon:
            if m in {1, 2, 3}:
                return self.canonical_icon_name(f"{base_icon}_{m}")
            return self.canonical_icon_name(base_icon)

        if type_id == 165483:
            if side in {"UP", "DOWN"}:
                return self.block_icon_name(5, 0, side, pos)
            return self.canonical_icon_name(base_icon)

        if type_id == 212292 and base_icon:
            suffix = "top" if side in {"UP", "DOWN"} else "side"
            return self.canonical_icon_name(f"{base_icon}_{suffix}")

        if type_id == 222549:
            return self.net_block_icon_name(block_id, m, side)

        if type_id == 106044 and info is not None:
            return self.pane_icon_name(info, base_icon, side)

        if base_icon:
            return self.canonical_icon_name(base_icon)
        return None

    def block_icon_name(self, block_id: int, meta: int, side: str, pos: Optional[Tuple[int, int, int]] = None) -> Optional[str]:
        side = side.upper()
        class_icon = self.class_block_icon_name(block_id, meta, side, pos)
        if class_icon is not None:
            return class_icon
        external_icon = self.external_block_icon_name(block_id)
        if external_icon is not None:
            return external_icon
        # Last-resort compatibility only: map_blocks.json is a PDA/minimap
        # table and must not override class-side render rules.
        icon = self.block_icons.get((block_id, meta & 0xF)) or self.block_icons.get((block_id, 0))
        return self.canonical_icon_name(icon)

    def external_block_icon_name(self, block_id: int) -> Optional[str]:
        cfg = self.external_configs.get(block_id)
        if not cfg:
            return None
        config = cfg.get("config") if isinstance(cfg.get("config"), dict) else cfg
        icon = config.get("icon") if isinstance(config, dict) else None
        if not isinstance(icon, str) or not icon:
            return None

        candidates = [icon]
        if ":" not in icon:
            # Custom block definitions store bare icon names, while blockMap.texarr
            # keeps them namespaced. Prefer the customitems namespace because that
            # is how the external block registry references its model assets.
            candidates.extend(
                [
                    f"customitems:{icon}",
                    f"stalcraft:{icon}",
                    f"vegetation:{icon}",
                ]
            )
        for candidate in candidates:
            if self.texarr.has(candidate):
                return candidate
        return None

    @staticmethod
    def _ctm_col(rule: CTMRule, x: int, y: int, z: int, side_index: int) -> int:
        if side_index in {0, 1, 3}:  # DOWN, UP, SOUTH
            raw = x
        elif side_index == 2:  # NORTH
            raw = -x - 1
        elif side_index == 4:  # WEST
            raw = z
        elif side_index == 5:  # EAST
            raw = -z - 1
        else:
            raw = x
        return raw % rule.width

    @staticmethod
    def _ctm_row(rule: CTMRule, x: int, y: int, z: int, side_index: int) -> int:
        if side_index in {0, 1}:  # DOWN, UP
            raw = z
        elif side_index in {2, 3, 4, 5}:  # horizontal sides
            raw = -y
        else:
            raw = z
        return raw % rule.height

    @staticmethod
    def _side_extra(tile: Optional[ProBuilderTile], side_index: int) -> int:
        if tile is None or tile.side_extra is None or side_index < 0 or side_index >= len(tile.side_extra):
            return 0
        return tile.side_extra[side_index] & 0xFF

    def ctm_icon_name(
        self,
        block_id: int,
        meta: int,
        pos: Tuple[int, int, int],
        side: str,
        tile: Optional[ProBuilderTile],
        fallback_icon: Optional[str],
    ) -> Optional[str]:
        if not self.enable_ctm:
            return fallback_icon
        rules = self.ctm_rules_by_block.get(block_id)
        if not rules:
            return fallback_icon
        x, y, z = pos
        side_index = SIDE_INDEX.get(side.upper(), 6)
        for rule in rules:
            if rule.min_height >= 0 and y < rule.min_height:
                continue
            if y > rule.max_height:
                continue
            block_specific_meta = rule.match_metadata_by_block.get(block_id)
            if block_specific_meta is not None and (meta & 0xF) not in block_specific_meta:
                continue
            if rule.metadata is not None and (meta & 0xF) not in rule.metadata:
                continue
            side_extra = self._side_extra(tile, side_index)
            rotation = (side_extra & 0xC0) >> 6
            if (side_extra & 0x03) == 0x03:
                col = (side_extra & 0x0C) >> 2
                row = (side_extra & 0x30) >> 4
            else:
                col = self._ctm_col(rule, x, y, z, side_index)
                row = self._ctm_row(rule, x, y, z, side_index)
            tiles = rotate_square_tiles(rule.tiles, rule.width, rule.height, rotation)
            index = row * rule.width + col
            if index >= len(tiles):
                index = self._ctm_row(rule, x, y, z, side_index) * rule.width + self._ctm_col(rule, x, y, z, side_index)
            if 0 <= index < len(tiles):
                return tiles[index]
        return fallback_icon

    def biome_grass_tint(self, pos: Tuple[int, int, int]) -> Optional[Tuple[float, float, float]]:
        if not self.world_biomes or not self.biome_color_table:
            return None
        x, _y, z = pos
        r = g = b = 0.0
        count = 0
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                biome_id = self.world_biomes.get((x + dx, z + dz))
                if biome_id is None:
                    continue
                color = self.biome_color_table.get(biome_id)
                if color is None:
                    continue
                r += color[0]
                g += color[1]
                b += color[2]
                count += 1
        if count == 0:
            return None
        return (r / count, g / count, b / count)

    def biome_tint_for_block(
        self,
        block_id: int,
        meta: int,
        pos: Tuple[int, int, int],
        side: str,
        icon_name: Optional[str],
    ) -> Optional[Tuple[float, float, float]]:
        info = self.type_textures.get(block_id)
        type_id = info.type_id if info is not None else self.block_type_by_id.get(block_id)
        side = side.upper()
        if type_id == 111004:
            # 111004.else(world,x,y,z) returns the 9-biome average grass color
            # for both the top face and the side overlay pass.
            if side == "UP" or (icon_name is not None and icon_name.endswith("grass_side_overlay")):
                return self.biome_grass_tint(pos)
            return None
        if type_id == 125838 and (meta & 0xF) != 0:
            # 125838 (tallgrass/fern): biome tint only when meta != 0 (deadbush stays white).
            return self.biome_grass_tint(pos)
        if type_id == 201816 and (meta & 0x3) not in {1, 2}:
            # 201816 (leaves): biome tint when (meta & 3) is not one of the
            # statically-coloured oak/spruce variants.
            return self.biome_grass_tint(pos)
        if type_id == 80251:
            # 80251 (vines, block 106): else(world,x,y,z) routes directly to the
            # biome chunk colour helper without metadata gating.
            return self.biome_grass_tint(pos)
        if icon_name and icon_name.endswith(("grass_top", "tallgrass", "fern", "grass_side_overlay")):
            return self.biome_grass_tint(pos)
        return None

    def material_for_block(
        self,
        block_id: int,
        meta: int,
        pos: Tuple[int, int, int],
        side: str,
        tile: Optional[ProBuilderTile] = None,
        allow_ctm: bool = True,
        tint: Optional[Tuple[float, float, float]] = None,
    ) -> str:
        icon = self.block_icon_name(block_id, meta, side, pos)
        if allow_ctm:
            icon = self.ctm_icon_name(block_id, meta, pos, side, tile, icon)
        if tint is None:
            tint = self.biome_tint_for_block(block_id, meta, pos, side, icon)
        if icon is None:
            material = safe_material_name("stc_missing_block", f"{block_id}_{meta}_{side}")
            self.materials.setdefault(material, MaterialDefinition(material, color=(1.0, 0.1, 0.1), alpha=1.0, note="missing block icon"))
            return material
        return self.material_for_icon(icon, tint)

    @staticmethod
    def _tint_key(tint: Optional[Tuple[float, float, float]]) -> Optional[Tuple[int, int, int]]:
        if tint is None:
            return None
        return tuple(max(0, min(255, round(channel * 255.0))) for channel in tint)  # type: ignore[return-value]

    def material_for_icon(self, icon_name: str, tint: Optional[Tuple[float, float, float]] = None) -> str:
        tint_key = self._tint_key(tint)
        material_key = (icon_name, tint_key)
        existing = self.icon_materials.get(material_key)
        if existing is not None:
            return existing
        tint_suffix = "" if tint_key is None else f"_tint_{tint_key[0]:02x}{tint_key[1]:02x}{tint_key[2]:02x}"
        material = safe_material_name("stc_tex", f"{icon_name}{tint_suffix}")
        baked_tint = False
        texture_rel: Optional[str] = None
        alpha_map_rel: Optional[str] = None
        if tint_key is not None and self.texture_format == "png":
            texture_rel = self.extract_tinted_texture(icon_name, tint_key)
            if texture_rel is not None:
                baked_tint = True
                if texture_rel in self.alpha_texture_rels:
                    alpha_map_rel = texture_rel
        if texture_rel is None:
            texture_rel = self.extract_texture(icon_name)
            if texture_rel is not None and icon_name in self.alpha_icons:
                alpha_map_rel = texture_rel
        if texture_rel is None:
            self.missing_icons.add(icon_name)
            self.materials[material] = MaterialDefinition(
                name=material,
                texture_rel=None,
                alpha_map_rel=None,
                color=(1.0, 0.0, 0.0),
                alpha=1.0,
                note=f"missing texture icon {icon_name}",
            )
        else:
            # Blender's OBJ importer often lets map_Kd override Kd, so baked
            # tinted PNG variants are required for visible biome/payload color.
            color = (1.0, 1.0, 1.0) if baked_tint else (tint if tint is not None else (1.0, 1.0, 1.0))
            self.materials[material] = MaterialDefinition(
                name=material,
                texture_rel=texture_rel,
                alpha_map_rel=alpha_map_rel,
                color=color,
                alpha=1.0,
                note=icon_name
                if tint_key is None
                else f"{icon_name}; {'baked ' if baked_tint else ''}tint=#{tint_key[0]:02x}{tint_key[1]:02x}{tint_key[2]:02x}",
            )
        self.icon_materials[material_key] = material
        return material

    def extract_tinted_texture(self, icon_name: str, tint_key: Tuple[int, int, int]) -> Optional[str]:
        cached = self.tinted_icons.get((icon_name, tint_key))
        if cached is not None:
            return cached
        data = self.texarr.read(icon_name)
        if data is None:
            return None
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(data)).convert("RGBA")
            r_ch, g_ch, b_ch, a_ch = image.split()
            r_mul, g_mul, b_mul = tint_key
            r_ch = r_ch.point(lambda value: (value * r_mul) // 255)
            g_ch = g_ch.point(lambda value: (value * g_mul) // 255)
            b_ch = b_ch.point(lambda value: (value * b_mul) // 255)
            tinted = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))
            alpha_is_partial = a_ch.getextrema()[0] < 255

            base_asset = icon_to_asset_path(icon_name, ".png")
            asset_path = base_asset.with_name(f"{base_asset.stem}_tint_{tint_key[0]:02x}{tint_key[1]:02x}{tint_key[2]:02x}.png")
            out_path = self.texture_root / asset_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tinted.save(out_path)
            rel = out_path.relative_to(self.texture_root.parent).as_posix()
            if alpha_is_partial:
                self.alpha_texture_rels.add(rel)
            self.tinted_icons[(icon_name, tint_key)] = rel
            return rel
        except Exception:
            self.convert_fallbacks.add(f"{icon_name}#tint")
            return None

    def extract_texture(self, icon_name: str) -> Optional[str]:
        cached = self.extracted_icons.get(icon_name)
        if cached is not None:
            return cached
        data = self.texarr.read(icon_name)
        if data is None:
            return None
        suffix = ".png" if self.texture_format == "png" else ".dds"
        asset_path = icon_to_asset_path(icon_name, suffix)
        out_path = self.texture_root / asset_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.texture_format == "png":
            try:
                from PIL import Image

                image = Image.open(io.BytesIO(data))
                if "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255:
                    self.alpha_icons.add(icon_name)
                image.save(out_path)
            except Exception:
                self.convert_fallbacks.add(icon_name)
                asset_path = icon_to_asset_path(icon_name, ".dds")
                out_path = self.texture_root / asset_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
        else:
            out_path.write_bytes(data)
        rel = out_path.relative_to(self.texture_root.parent).as_posix()
        self.extracted_icons[icon_name] = rel
        return rel

    def png_bytes_for_icon(self, icon_name: str) -> Optional[bytes]:
        """Return raw PNG bytes for an icon for glTF embedding.

        DDS entries from `blockMap.texarr` are converted to PNG in memory via
        Pillow. Existing PNG entries are returned as-is. Alpha presence is
        recorded into `self.alpha_icons` so the glTF writer can pick the right
        `alphaMode`.
        """

        if icon_name in self._png_bytes_cache:
            return self._png_bytes_cache[icon_name]
        data = self.texarr.read(icon_name)
        if data is None:
            self._png_bytes_cache[icon_name] = None
            return None
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(data))
            if image.mode not in ("RGBA", "RGB"):
                image = image.convert("RGBA")
            if "A" in image.getbands():
                alpha = image.getchannel("A")
                alpha_min, _alpha_max = alpha.getextrema()
                if alpha_min < 255:
                    self.alpha_icons.add(icon_name)
                    # Histogram-based cutout vs translucent split: if at least
                    # 99% of alpha values are at the extremes (0 or 255), treat
                    # as cutout (`MASK`); otherwise the sprite carries real
                    # intermediate transparency and needs `BLEND`.
                    histogram = alpha.histogram()
                    cutout = histogram[0] + histogram[255]
                    total = sum(histogram)
                    if total > 0 and cutout / total < 0.99:
                        self.translucent_icons.add(icon_name)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            png_bytes = buffer.getvalue()
        except Exception:
            self.convert_fallbacks.add(icon_name)
            self._png_bytes_cache[icon_name] = None
            return None
        self._png_bytes_cache[icon_name] = png_bytes
        return png_bytes

    def face_material_descriptor(
        self,
        block_id: int,
        meta: int,
        pos: Tuple[int, int, int],
        side: str,
        tile: Optional[ProBuilderTile],
        *,
        allow_ctm: bool = True,
        explicit_tint: Optional[Tuple[float, float, float]] = None,
    ) -> "FaceMaterialDescriptor":
        """Resolve sprite + alpha mode + tint for a face without baking PNGs.

        Combines `block_icon_name` + `ctm_icon_name` + biome/tile tint sources.
        The returned descriptor is consumed by the glTF writer:
        the icon decides the material, the alpha flag picks `alphaMode`, and
        the tint becomes a per-vertex `COLOR_0` multiplier.
        """

        side = side.upper()
        icon = self.block_icon_name(block_id, meta, side, pos)
        if allow_ctm:
            icon = self.ctm_icon_name(block_id, meta, pos, side, tile, icon)
        png_bytes = self.png_bytes_for_icon(icon) if icon else None
        tint = explicit_tint
        if tint is None:
            tint = self.biome_tint_for_block(block_id, meta, pos, side, icon)
        if tint is None and tile is not None:
            tint = GeometryResolver.tile_tint(tile, side)
        alpha_mode = self._alpha_mode_for_icon(icon)
        return FaceMaterialDescriptor(
            icon=icon,
            png_bytes=png_bytes,
            alpha_mode=alpha_mode,
            tint=tint,
        )

    def overlay_material_descriptor(
        self,
        icon: Optional[str],
        tint: Optional[Tuple[float, float, float]],
    ) -> "FaceMaterialDescriptor":
        png_bytes = self.png_bytes_for_icon(icon) if icon else None
        if icon is None:
            alpha_mode = "OPAQUE"
        elif icon in self.translucent_icons:
            alpha_mode = "BLEND"
        elif icon in self.alpha_icons:
            alpha_mode = "MASK"
        else:
            # Overlay sprites without baked alpha still need BLEND so the base
            # face shows through where the overlay carries no contribution.
            alpha_mode = "BLEND"
        return FaceMaterialDescriptor(
            icon=icon,
            png_bytes=png_bytes,
            alpha_mode=alpha_mode,
            tint=tint,
        )

    def _alpha_mode_for_icon(self, icon: Optional[str]) -> str:
        if icon is None:
            return "OPAQUE"
        if icon in self.translucent_icons:
            return "BLEND"
        if icon in self.alpha_icons:
            return "MASK"
        return "OPAQUE"

    def block_overlay_icon_name(
        self,
        block_id: int,
        meta: int,
        side: str,
        pos: Tuple[int, int, int],
    ) -> Optional[str]:
        info = self.type_textures.get(block_id)
        type_id = info.type_id if info is not None else self.block_type_by_id.get(block_id)
        side = side.upper()
        if type_id == 111004 and side in {"NORTH", "SOUTH", "WEST", "EAST"} and not self.block_above_is_snow_like(pos):
            return self.canonical_icon_name("grass_side_overlay")
        return None

    def probuilder_overlay_icon_name(self, overlay_value: int, side: str) -> Optional[str]:
        side = side.upper()
        overlay_value &= 0xFF
        if overlay_value == 0:
            return None
        if overlay_value == 1:
            if side == "DOWN":
                return None
            if side == "UP":
                return "stalcraft:grass_top"
            # 188933.char(int) uses overlay_fast_grass_side only when
            # 117842.b is enabled in normal runtime data, so
            # side overlays route through 111004.char() = grass_side_overlay.
            return self.canonical_icon_name("grass_side_overlay")
        if overlay_value == 2:
            if side == "DOWN":
                return None
            if side == "UP":
                return "stalcraft:snow"
            return "probuilder:overlay/overlay_snow_side"
        if overlay_value == 4:
            if side == "DOWN":
                return None
            return "stalcraft:vine"
        if overlay_value == 5:
            if side == "DOWN":
                return None
            if side == "UP":
                return "stalcraft:comp_top"
            return "probuilder:overlay/overlay_hay_side"
        if overlay_value == 6:
            if side == "DOWN":
                return None
            if side == "UP":
                return "stalcraft:dirt2_top"
            return "probuilder:overlay/overlay_mycelium_side"
        return None


def side_name_from_data(data: int) -> str:
    side = data & 7
    if 0 <= side < len(SIDE_NAMES):
        return SIDE_NAMES[side]
    return "UNKNOWN"


def box_faces_for_sides(box: AABB, include_sides: Optional[Set[str]] = None) -> List[Face]:
    x0, y0, z0, x1, y1, z1 = box
    faces: List[Face] = []
    for side, _offset, make_face in GRID_FACE_DEFS:
        if include_sides is not None and side not in include_sides:
            continue
        face = make_face(x0, y0, z0, x1, y1, z1)
        if face_has_area(face):
            faces.append(face)
    return faces


def merge_4x4_mask_rectangles(mask: Sequence[bool]) -> List[Tuple[int, int, int, int]]:
    visited = [False] * 16
    rectangles: List[Tuple[int, int, int, int]] = []
    for row in range(4):
        col = 0
        while col < 4:
            index = row * 4 + col
            if visited[index] or not mask[index]:
                col += 1
                continue

            width = 1
            while col + width < 4 and mask[index + width] and not visited[index + width]:
                width += 1

            height = 1
            while row + height < 4:
                if all(mask[(row + height) * 4 + col + dx] and not visited[(row + height) * 4 + col + dx] for dx in range(width)):
                    height += 1
                    continue
                break

            for dy in range(height):
                for dx in range(width):
                    visited[(row + dy) * 4 + col + dx] = True
            rectangles.append((col, row, width, height))
            col += width
    return rectangles


def grid_exposed_cell_for_side(payload: int, side: str, layer: int, col: int, row: int) -> bool:
    if side == "DOWN":
        return grid_bit(payload, col, layer, row) and not grid_bit(payload, col, layer - 1, row)
    if side == "UP":
        y = 3 - layer
        return grid_bit(payload, col, y, row) and not grid_bit(payload, col, y + 1, row)
    if side == "NORTH":
        return grid_bit(payload, col, row, layer) and not grid_bit(payload, col, row, layer - 1)
    if side == "SOUTH":
        z = 3 - layer
        return grid_bit(payload, col, row, z) and not grid_bit(payload, col, row, z + 1)
    if side == "WEST":
        return grid_bit(payload, layer, row, col) and not grid_bit(payload, layer - 1, row, col)
    if side == "EAST":
        x = 3 - layer
        return grid_bit(payload, x, row, col) and not grid_bit(payload, x + 1, row, col)
    return False


def grid_exposed_rectangle_box(side: str, layer: int, col: int, row: int, width: int, height: int) -> AABB:
    step = 0.25
    if side == "DOWN":
        return (col * step, layer * step, row * step, (col + width) * step, (layer + 1) * step, (row + height) * step)
    if side == "UP":
        y = 3 - layer
        return (col * step, y * step, row * step, (col + width) * step, (y + 1) * step, (row + height) * step)
    if side == "NORTH":
        return (col * step, row * step, layer * step, (col + width) * step, (row + height) * step, (layer + 1) * step)
    if side == "SOUTH":
        z = 3 - layer
        return (col * step, row * step, z * step, (col + width) * step, (row + height) * step, (z + 1) * step)
    if side == "WEST":
        return (layer * step, row * step, col * step, (layer + 1) * step, (row + height) * step, (col + width) * step)
    if side == "EAST":
        x = 3 - layer
        return (x * step, row * step, col * step, (x + 1) * step, (row + height) * step, (col + width) * step)
    return FULL_CUBE


def probuilder_grid_exposed_rectangles(payload: int) -> List[Tuple[str, AABB]]:
    rectangles: List[Tuple[str, AABB]] = []
    for side in SIDE_NAMES[:6]:
        for layer in range(4):
            mask = [
                grid_exposed_cell_for_side(payload, side, layer, col, row)
                for row in range(4)
                for col in range(4)
            ]
            for col, row, width, height in merge_4x4_mask_rectangles(mask):
                rectangles.append((side, grid_exposed_rectangle_box(side, layer, col, row, width, height)))
    return rectangles


def cover_plate_box(surface_box: AABB, side: str, cover_block_id: int, eps: float = 0.002) -> AABB:
    x0, y0, z0, x1, y1, z1 = surface_box
    thickness = 0.125 if side == "UP" and cover_block_id in THICK_COVER_BLOCK_IDS else 0.0625
    if side == "DOWN":
        return (x0 - eps, y0 - thickness, z0 - eps, x1 + eps, y0, z1 + eps)
    if side == "UP":
        return (x0 - eps, y1, z0 - eps, x1 + eps, y1 + thickness, z1 + eps)
    if side == "NORTH":
        return (x0 - eps, y0 - eps, z0 - thickness, x1 + eps, y1 + eps, z0)
    if side == "SOUTH":
        return (x0 - eps, y0 - eps, z1, x1 + eps, y1 + eps, z1 + thickness)
    if side == "WEST":
        return (x0 - thickness, y0 - eps, z0 - eps, x0, y1 + eps, z1 + eps)
    if side == "EAST":
        return (x1, y0 - eps, z0 - eps, x1 + thickness, y1 + eps, z1 + eps)
    return surface_box


def cover_plate_faces(surface_box: AABB, side: str, cover_block_id: int, closed: bool = True) -> List[Face]:
    plate = cover_plate_box(surface_box, side, cover_block_id)
    include = set(SIDE_NAMES[:6])
    back_side = OPPOSITE_SIDE.get(side)
    if not closed and back_side is not None:
        include.discard(back_side)
    return box_faces_for_sides(plate, include)


def face_lies_on_block_boundary(face: Face, side: str, eps: float = 1.0e-7) -> bool:
    if side == "DOWN":
        return all(abs(y - 0.0) <= eps for _x, y, _z in face)
    if side == "UP":
        return all(abs(y - 1.0) <= eps for _x, y, _z in face)
    if side == "NORTH":
        return all(abs(z - 0.0) <= eps for _x, _y, z in face)
    if side == "SOUTH":
        return all(abs(z - 1.0) <= eps for _x, _y, z in face)
    if side == "WEST":
        return all(abs(x - 0.0) <= eps for x, _y, _z in face)
    if side == "EAST":
        return all(abs(x - 1.0) <= eps for x, _y, _z in face)
    return False


def cover_side_for_face(face: Face, eps: float = 1.0e-7) -> Optional[str]:
    for side in SIDE_NAMES[:6]:
        if face_lies_on_block_boundary(face, side, eps):
            return side

    nx, ny, nz = polygon_normal(face)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= eps:
        return None
    nx, ny, nz = nx / length, ny / length, nz / length
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ay >= ax and ay >= az:
        return "UP" if ny >= 0.0 else "DOWN"
    if az >= ax:
        return "SOUTH" if nz >= 0.0 else "NORTH"
    return "EAST" if nx >= 0.0 else "WEST"


def uv_for_face(face: Face, side: str) -> List[Tuple[float, float]]:
    side = side.upper()
    uvs: List[Tuple[float, float]] = []
    for x, y, z in face:
        if side in {"DOWN", "UP"}:
            u, v = x, z
        elif side in {"NORTH", "SOUTH"}:
            u, v = x, y
        elif side in {"WEST", "EAST"}:
            u, v = z, y
        else:
            u, v = x, z
        uvs.append((u, v))
    return uvs


def rotate_uvs(uvs: List[Tuple[float, float]], rotations: int) -> List[Tuple[float, float]]:
    if not uvs or rotations % 4 == 0:
        return uvs
    us = [uv[0] for uv in uvs]
    vs = [uv[1] for uv in uvs]
    min_u, max_u = min(us), max(us)
    min_v, max_v = min(vs), max(vs)
    width = max_u - min_u
    height = max_v - min_v
    if abs(width) < 1.0e-9 or abs(height) < 1.0e-9:
        return uvs
    rotated = list(uvs)
    for _ in range(rotations % 4):
        rotated = [(min_u + (v - min_v) * width / height, max_v - (u - min_u) * height / width) for u, v in rotated]
    return rotated


def cover_prism_faces(surface_face: Face, side: str, cover_block_id: int, closed: bool = True) -> List[Face]:
    thickness = 0.125 if side == "UP" and cover_block_id in THICK_COVER_BLOCK_IDS else 0.0625
    ax, ay, az = SIDE_AXIS[side]
    offset_face = tuple((x + ax * thickness, y + ay * thickness, z + az * thickness) for x, y, z in surface_face)
    faces: List[Face] = [offset_face]
    if closed:
        faces.append(tuple(reversed(surface_face)))
    for index, p0 in enumerate(surface_face):
        p1 = surface_face[(index + 1) % len(surface_face)]
        q0 = offset_face[index]
        q1 = offset_face[(index + 1) % len(offset_face)]
        faces.append((p0, p1, q1, q0))
    return [face for face in faces if face_has_area(face)]


def offset_face_along_side(face: Face, side: str, distance: float = 0.003) -> Face:
    ax, ay, az = SIDE_AXIS[side]
    return tuple((x + ax * distance, y + ay * distance, z + az * distance) for x, y, z in face)


def collapsible_value(data: int, index: int) -> int:
    if index == 0:
        value = (data & 0x7C0000) >> 18
    elif index == 1:
        value = (data & 0x3E000) >> 13
    elif index == 2:
        value = (data & 0x1F00) >> 8
    elif index == 3:
        value = (data & 0xF8) >> 3
    else:
        value = 0
    return min(value, 16)


def collapsible_is_curved(data: int) -> bool:
    side = side_name_from_data(data)
    if side in {"UP", "DOWN", "UNKNOWN"}:
        return False
    v0 = collapsible_value(data, 0)
    v2 = collapsible_value(data, 2)
    v1 = collapsible_value(data, 1)
    v3 = collapsible_value(data, 3)
    return not (abs(v1 - v0) > 10 and abs(v3 - v2) > 10)


def triangle_split(v0: float, v1: float, v2: float, v3: float) -> bool:
    return abs(v0 - v3) < abs(v2 - v1)


def barycentric_height(
    ax: float,
    ah: float,
    ay: float,
    bx: float,
    bh: float,
    by: float,
    cx: float,
    ch: float,
    cy: float,
    px: float,
    py: float,
) -> float:
    bx -= ax
    by -= ay
    cx -= ax
    cy -= ay
    px -= ax
    py -= ay
    dot00 = bx * bx + by * by
    dot01 = bx * cx + by * cy
    dot02 = bx * px + by * py
    dot11 = cx * cx + cy * cy
    dot12 = cx * px + cy * py
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1.0e-9:
        return ah
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v
    return u * bh + v * ch + w * ah


def collapsible_interp(split: bool, v0: float, v1: float, v2: float, v3: float, u: float, v: float) -> float:
    # Direct port of 21985.char(boolean,...): interpolate one of two possible
    # triangle splits across four corner heights.
    if split:
        if u + v <= 1.0:
            return barycentric_height(0.0, v0, 0.0, 1.0, v2, 0.0, 0.0, v1, 1.0, u, v)
        return barycentric_height(1.0, v2, 0.0, 1.0, v3, 1.0, 0.0, v1, 1.0, u, v)
    if u - v <= 0.0:
        return barycentric_height(0.0, v0, 0.0, 1.0, v3, 1.0, 0.0, v1, 1.0, u, v)
    return barycentric_height(0.0, v0, 0.0, 1.0, v2, 0.0, 1.0, v3, 1.0, u, v)


def collapsible_height_grid(data: int) -> List[float]:
    curved = collapsible_is_curved(data)
    v0 = collapsible_value(data, 0) / 16.0
    v2 = collapsible_value(data, 2) / 16.0
    v1 = collapsible_value(data, 1) / 16.0
    v3 = collapsible_value(data, 3) / 16.0
    split = triangle_split(v0, v1, v2, v3)
    cols = 8
    rows = 2 if curved else 8
    out: List[float] = []
    for row in range(rows):
        for col in range(cols):
            u0 = col / cols
            u1 = (col + 1) / cols
            v_row0 = row / rows
            v_row1 = (row + 1) / rows
            heights = [
                collapsible_interp(split, v0, v1, v2, v3, u0, v_row0),
                collapsible_interp(split, v0, v1, v2, v3, u1, v_row0),
                collapsible_interp(split, v0, v1, v2, v3, u0, v_row1),
                collapsible_interp(split, v0, v1, v2, v3, u1, v_row1),
            ]
            if curved and row + 1 <= rows / 2:
                out.append(min(heights))
            else:
                out.append(max(heights))
    return out


def box_has_volume(box: AABB) -> bool:
    x0, y0, z0, x1, y1, z1 = box
    return x1 > x0 and y1 > y0 and z1 > z0


def collapsible_boxes(tile: ProBuilderTile) -> List[AABB]:
    data = tile.data
    side = side_name_from_data(data)
    heights = collapsible_height_grid(data)
    cols = 8
    rows = 2 if collapsible_is_curved(data) else 8
    dx = 1.0 / cols
    dz = 1.0 / rows
    boxes: List[AABB] = []
    for index, height in enumerate(heights):
        col = index % cols
        row = index // cols
        x0 = col * dx
        x1 = (col + 1) * dx
        t0 = row * dz
        t1 = (row + 1) * dz
        h = float(height)
        if side == "UP":
            box = (x0, 0.0, t0, x1, h, t1)
        elif side == "DOWN":
            box = (x0, 1.0 - h, t0, x1, 1.0, t1)
        elif side == "NORTH":
            rev_row = len(heights) // cols - 1 - row
            rev_col = len(heights) // rows - 1 - col
            h = float(heights[rev_row * cols + rev_col])
            box = (x0, t0, 1.0 - h, x1, t1, 1.0)
        elif side == "SOUTH":
            rev_row = len(heights) // cols - 1 - row
            h = float(heights[rev_row * cols + col])
            box = (x0, t0, 0.0, x1, t1, h)
        elif side == "EAST":
            rev_row = len(heights) // cols - 1 - row
            rev_col = len(heights) // rows - 1 - col
            h = float(heights[rev_row * cols + rev_col])
            box = (0.0, t0, x0, h, t1, x1)
        elif side == "WEST":
            rev_row = len(heights) // cols - 1 - row
            h = float(heights[rev_row * cols + col])
            box = (1.0 - h, t0, x0, 1.0, t1, x1)
        else:
            box = FULL_CUBE
        if box_has_volume(box):
            boxes.append(box)
    return boxes


def collapsible_corner_height(data: int, u: float, v: float) -> float:
    v0 = collapsible_value(data, 0) / 16.0
    v2 = collapsible_value(data, 2) / 16.0
    v1 = collapsible_value(data, 1) / 16.0
    v3 = collapsible_value(data, 3) / 16.0
    return collapsible_interp(triangle_split(v0, v1, v2, v3), v0, v1, v2, v3, u, v)


def collapsible_oriented_height(data: int, side: str, u: float, v: float) -> float:
    if side in {"UP", "DOWN"}:
        return collapsible_corner_height(data, u, v)
    if side in {"NORTH", "EAST"}:
        return collapsible_corner_height(data, 1.0 - u, 1.0 - v)
    if side in {"SOUTH", "WEST"}:
        return collapsible_corner_height(data, u, 1.0 - v)
    return collapsible_corner_height(data, u, v)


def variable_depth_faces(
    surface_point: Any,
    base_point: Any,
    segments_u: int = 8,
    segments_v: int = 8,
) -> List[Face]:
    faces: List[Face] = []
    du = 1.0 / float(segments_u)
    dv = 1.0 / float(segments_v)

    for vi in range(segments_v):
        v0 = vi * dv
        v1 = (vi + 1) * dv
        for ui in range(segments_u):
            u0 = ui * du
            u1 = (ui + 1) * du
            p00 = surface_point(u0, v0)
            p10 = surface_point(u1, v0)
            p11 = surface_point(u1, v1)
            p01 = surface_point(u0, v1)
            faces.extend([(p00, p10, p11), (p00, p11, p01)])

    faces.append((base_point(0.0, 1.0), base_point(1.0, 1.0), base_point(1.0, 0.0), base_point(0.0, 0.0)))

    for ui in range(segments_u):
        u0 = ui * du
        u1 = (ui + 1) * du
        faces.append((base_point(u0, 0.0), base_point(u1, 0.0), surface_point(u1, 0.0), surface_point(u0, 0.0)))
        faces.append((surface_point(u0, 1.0), surface_point(u1, 1.0), base_point(u1, 1.0), base_point(u0, 1.0)))
    for vi in range(segments_v):
        v0 = vi * dv
        v1 = (vi + 1) * dv
        faces.append((base_point(0.0, v1), base_point(0.0, v0), surface_point(0.0, v0), surface_point(0.0, v1)))
        faces.append((surface_point(1.0, v0), base_point(1.0, v0), base_point(1.0, v1), surface_point(1.0, v1)))

    return [face for face in faces if face_has_area(face)]


def collapsible_smooth_faces(tile: ProBuilderTile) -> List[Face]:
    data = tile.data
    side = side_name_from_data(data)

    def h(u: float, v: float) -> float:
        return max(0.0, min(1.0, collapsible_oriented_height(data, side, u, v)))

    if side == "UP":
        return variable_depth_faces(lambda u, v: (u, h(u, v), v), lambda u, v: (u, 0.0, v))
    if side == "DOWN":
        return variable_depth_faces(lambda u, v: (u, 1.0 - h(u, v), v), lambda u, v: (u, 1.0, v))
    if side == "SOUTH":
        return variable_depth_faces(lambda u, v: (u, v, h(u, v)), lambda u, v: (u, v, 0.0))
    if side == "NORTH":
        return variable_depth_faces(lambda u, v: (u, v, 1.0 - h(u, v)), lambda u, v: (u, v, 1.0))
    if side == "EAST":
        return variable_depth_faces(lambda u, v: (h(u, v), v, u), lambda u, v: (0.0, v, u))
    if side == "WEST":
        return variable_depth_faces(lambda u, v: (1.0 - h(u, v), v, u), lambda u, v: (1.0, v, u))
    return []


def triangular_panel_slice_box(shape: int, index: int, steps: int = 8) -> AABB:
    f = float(index) / float(steps)
    f2 = float(index + 1) / float(steps)
    match shape:
        case 0:
            return (0.0, 0.5, 0.0, f2, 1.0, 1.0 - f)
        case 1:
            return (f, 0.5, 1.0 - f2, 1.0, 1.0, 1.0)
        case 2:
            return (0.0, 0.5, f, f2, 1.0, 1.0)
        case 3:
            return (f, 0.5, 0.0, 1.0, 1.0, f2)
        case 4:
            return (0.0, 0.0, 0.0, f2, 0.5, 1.0 - f)
        case 5:
            return (f, 0.0, 1.0 - f2, 1.0, 0.5, 1.0)
        case 6:
            return (0.0, 0.0, f, f2, 0.5, 1.0)
        case 7:
            return (f, 0.0, 0.0, 1.0, 0.5, f2)
        case 8:
            return (0.5, 0.0, 0.0, 1.0, 1.0 - f, f2)
        case 9:
            return (0.5, 0.0, f, 1.0, f2, 1.0)
        case 10:
            return (0.5, 1.0 - f2, f, 1.0, 1.0, 1.0)
        case 11:
            return (0.5, f, 0.0, 1.0, 1.0, f2)
        case 12:
            return (f, 0.0, 0.5, 1.0, f2, 1.0)
        case 13:
            return (0.0, 0.0, 0.5, f2, 1.0 - f, 1.0)
        case 14:
            return (0.0, f, 0.5, f2, 1.0, 1.0)
        case 15:
            return (f, 1.0 - f2, 0.5, 1.0, 1.0, 1.0)
        case 16:
            return (0.0, 0.0, 0.0, 0.5, 1.0 - f, f2)
        case 17:
            return (0.0, f, 0.0, 0.5, 1.0, f2)
        case 18:
            return (0.0, 1.0 - f2, f, 0.5, 1.0, 1.0)
        case 19:
            return (0.0, 0.0, f, 0.5, f2, 1.0)
        case 20:
            return (f, 0.0, 0.0, 1.0, f2, 0.5)
        case 21:
            return (f, 1.0 - f2, 0.0, 1.0, 1.0, 0.5)
        case 22:
            return (0.0, f, 0.0, f2, 1.0, 0.5)
        case 23:
            return (0.0, 0.0, 0.0, f2, 1.0 - f, 0.5)
    return FULL_CUBE


def triangular_panel_boxes(shape: int, steps: int = 8) -> List[AABB]:
    return [
        box
        for index in range(steps)
        if box_has_volume(box := triangular_panel_slice_box(shape, index, steps))
    ]


def triangular_panel_base_box(shape: int) -> AABB:
    match shape // 4:
        case 0:
            return (0.0, 0.0, 0.0, 1.0, 0.5, 1.0)
        case 1:
            return (0.0, 0.5, 0.0, 1.0, 1.0, 1.0)
        case 2:
            return (0.0, 0.0, 0.0, 0.5, 1.0, 1.0)
        case 3:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 0.5)
        case 4:
            return (0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
        case 5:
            return (0.0, 0.0, 0.5, 1.0, 1.0, 1.0)
    return FULL_CUBE


TRIANGULAR_PANEL_FOOTPRINTS: Dict[int, List[Tuple[float, float]]] = {
    0: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    1: [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    2: [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    3: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
}


def triangular_panel_smooth_faces(shape: int) -> List[Face]:
    shape %= 24
    if 0 <= shape <= 3:
        return scale_faces(triangular_prism_faces(TRIANGULAR_PANEL_FOOTPRINTS[shape]), y0=0.5, y1=1.0)
    if 4 <= shape <= 7:
        return scale_faces(smooth_slope_faces_for_j(shape - 4), y0=0.0, y1=0.5)

    specs: Dict[int, Tuple[float, float, float, float, bool, Any]] = {
        8: (0.5, 1.0, 0.0, 1.0, True, lambda x, z: 1.0 - z),
        9: (0.5, 1.0, 0.0, 1.0, True, lambda x, z: z),
        10: (0.5, 1.0, 0.0, 1.0, False, lambda x, z: 1.0 - z),
        11: (0.5, 1.0, 0.0, 1.0, False, lambda x, z: z),
        12: (0.0, 1.0, 0.5, 1.0, True, lambda x, z: x),
        13: (0.0, 1.0, 0.5, 1.0, True, lambda x, z: 1.0 - x),
        14: (0.0, 1.0, 0.5, 1.0, False, lambda x, z: x),
        15: (0.0, 1.0, 0.5, 1.0, False, lambda x, z: 1.0 - x),
        16: (0.0, 0.5, 0.0, 1.0, True, lambda x, z: 1.0 - z),
        17: (0.0, 0.5, 0.0, 1.0, False, lambda x, z: z),
        18: (0.0, 0.5, 0.0, 1.0, False, lambda x, z: 1.0 - z),
        19: (0.0, 0.5, 0.0, 1.0, True, lambda x, z: z),
        20: (0.0, 1.0, 0.0, 0.5, True, lambda x, z: x),
        21: (0.0, 1.0, 0.0, 0.5, False, lambda x, z: 1.0 - x),
        22: (0.0, 1.0, 0.0, 0.5, False, lambda x, z: x),
        23: (0.0, 1.0, 0.0, 0.5, True, lambda x, z: 1.0 - x),
    }
    x0, x1, z0, z1, solid_below, height_fn = specs[shape]
    faces = heightfield_faces(height_fn, solid_below)
    return scale_faces(faces, x0=x0, x1=x1, z0=z0, z1=z1)


TRIANGULAR_HALFSTEP_BASE_BOXES: Dict[int, Tuple[AABB, AABB]] = {
    0: ((0.0, 0.0, 0.0, 1.0, 0.5, 0.5), (0.0, 0.0, 0.5, 0.5, 0.5, 1.0)),
    1: ((0.0, 0.0, 0.5, 1.0, 0.5, 1.0), (0.5, 0.0, 0.0, 1.0, 0.5, 0.5)),
    2: ((0.0, 0.0, 0.5, 1.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.5, 0.5, 0.5)),
    3: ((0.5, 0.0, 0.0, 1.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.5, 0.5, 0.5)),
    4: ((0.0, 0.5, 0.0, 1.0, 1.0, 0.5), (0.0, 0.5, 0.5, 0.5, 1.0, 1.0)),
    5: ((0.0, 0.5, 0.5, 1.0, 1.0, 1.0), (0.5, 0.5, 0.0, 1.0, 1.0, 0.5)),
    6: ((0.0, 0.5, 0.5, 1.0, 1.0, 1.0), (0.0, 0.5, 0.0, 0.5, 1.0, 0.5)),
    7: ((0.5, 0.5, 0.0, 1.0, 1.0, 1.0), (0.0, 0.5, 0.0, 0.5, 1.0, 0.5)),
    8: ((0.0, 0.0, 0.0, 0.5, 0.5, 1.0), (0.0, 0.5, 0.0, 0.5, 1.0, 0.5)),
    9: ((0.0, 0.0, 0.0, 0.5, 0.5, 1.0), (0.0, 0.5, 0.5, 0.5, 1.0, 1.0)),
    10: ((0.0, 0.5, 0.0, 0.5, 1.0, 1.0), (0.0, 0.0, 0.5, 0.5, 0.5, 1.0)),
    11: ((0.0, 0.5, 0.0, 0.5, 1.0, 1.0), (0.0, 0.0, 0.0, 0.5, 0.5, 0.5)),
    12: ((0.0, 0.0, 0.0, 1.0, 0.5, 0.5), (0.5, 0.5, 0.0, 1.0, 1.0, 0.5)),
    13: ((0.0, 0.0, 0.0, 1.0, 0.5, 0.5), (0.0, 0.5, 0.0, 0.5, 1.0, 0.5)),
    14: ((0.0, 0.5, 0.0, 1.0, 1.0, 0.5), (0.0, 0.0, 0.0, 0.5, 0.5, 0.5)),
    15: ((0.0, 0.5, 0.0, 1.0, 1.0, 0.5), (0.5, 0.0, 0.0, 1.0, 0.5, 0.5)),
    16: ((0.5, 0.0, 0.0, 1.0, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0, 1.0, 0.5)),
    17: ((0.5, 0.5, 0.0, 1.0, 1.0, 1.0), (0.5, 0.0, 0.0, 1.0, 0.5, 0.5)),
    18: ((0.5, 0.5, 0.0, 1.0, 1.0, 1.0), (0.5, 0.0, 0.5, 1.0, 0.5, 1.0)),
    19: ((0.5, 0.0, 0.0, 1.0, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0, 1.0, 1.0)),
    20: ((0.0, 0.0, 0.5, 1.0, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0, 1.0, 1.0)),
    21: ((0.0, 0.5, 0.5, 1.0, 1.0, 1.0), (0.5, 0.0, 0.5, 1.0, 0.5, 1.0)),
    22: ((0.0, 0.5, 0.5, 1.0, 1.0, 1.0), (0.0, 0.0, 0.5, 0.5, 0.5, 1.0)),
    23: ((0.0, 0.0, 0.5, 1.0, 0.5, 1.0), (0.0, 0.5, 0.5, 0.5, 1.0, 1.0)),
}


def triangular_halfstep_base_boxes(shape: int) -> List[AABB]:
    return [box for box in TRIANGULAR_HALFSTEP_BASE_BOXES.get(shape % 24, (FULL_CUBE,)) if box_has_volume(box)]


def button_box(state: BlockState, tile: Optional[ProBuilderTile]) -> AABB:
    data = tile.data if tile is not None else 0
    direction = data & 7
    if direction == 0:
        direction = state.meta & 7
    thickness = 0.0625 if ((data & 8) >> 3) == 1 else 0.125
    if direction == 1:
        return (0.0, 0.375, 0.3125, thickness, 0.625, 0.6875)
    if direction == 2:
        return (1.0 - thickness, 0.375, 0.3125, 1.0, 0.625, 0.6875)
    if direction == 3:
        return (0.3125, 0.375, 0.0, 0.6875, 0.625, thickness)
    if direction == 4:
        return (0.3125, 0.375, 1.0 - thickness, 0.6875, 0.625, 1.0)
    return FULL_CUBE


def ladder_faces(meta: int) -> List[Face]:
    if meta == 2:
        return [
            ((0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (1.0, 1.0, 0.8), (0.0, 1.0, 0.8)),
            ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        ]
    if meta == 3:
        return [
            ((0.0, 0.0, 0.2), (1.0, 0.0, 0.2), (1.0, 1.0, 0.2), (0.0, 1.0, 0.2)),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        ]
    if meta == 4:
        return [
            ((0.8, 0.0, 0.0), (0.8, 0.0, 1.0), (0.8, 1.0, 1.0), (0.8, 1.0, 0.0)),
            ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 0.0)),
        ]
    if meta == 5:
        return [
            ((0.2, 0.0, 0.0), (0.2, 0.0, 1.0), (0.2, 1.0, 1.0), (0.2, 1.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
        ]
    return []


def plant_cross_faces(radius: float, height: float) -> List[Face]:
    lo = 0.5 - radius
    hi = 0.5 + radius
    return [
        ((lo, 0.0, lo), (hi, 0.0, hi), (hi, height, hi), (lo, height, lo)),
        ((lo, 0.0, hi), (hi, 0.0, lo), (hi, height, lo), (lo, height, hi)),
    ]


LIQUID_TYPE_IDS: Set[int] = {238471, 47312, 93367}
# Horizontal phantom-liquid positions: cells adjacent to real liquid blocks
# that should be treated as liquid even though `world_blocks` carries a
# different (or no) block there. Populated by `compute_liquid_phantom_positions`
# at startup when `--water-expand N > 0`. Consulted by `_liquid_top_y` so the
# corner-height / culling logic naturally extends the lake surface into the
# adjacent terrain, hiding the visible water-side-wall strip inside opaque
# neighbour blocks.
LIQUID_PHANTOM_POSITIONS: Set[Tuple[int, int, int]] = set()
# Sprite name pattern used to auto-detect liquid block types at startup.
# STALCRAFT renames the vanilla 1.7.10 liquid types (8/9/10/11) so the
# hardcoded ids above miss the live mapping; auto-detection picks them up
# from the class registry by inspecting the base icon attached to each class.
LIQUID_ICON_RE = re.compile(r"^(water|lava)(_still|_flow|_static|_overlay)?$", re.IGNORECASE)
# Matches the in-engine "invisible technical material" sprite regardless of
# whether the registry serves it as `inv_mask`, `inv-mask`, `inv.mask`, etc.
# The whole-word match (`(?:^|[^a-z0-9])inv[-_. ]?mask(?:$|[^a-z0-9])`) avoids
# false positives on icons that merely contain `inv` somewhere in their path.
_INV_MASK_ICON_RE = re.compile(r"(?:^|[^a-z0-9])inv[\W_]*mask(?:$|[^a-z0-9])", re.IGNORECASE)
# Final-material-name match for the in-engine invisible mask sprite. Its
# material name has the form `stc_tex_<sanitized-icon>_mask` (the trailing
# `_mask` comes from `alphaMode=MASK` via `_material_name_for_icon`). The
# user-facing icon paths involved are `<ns>:inv` (literal "inv" basename) or
# any explicit `inv_mask`/`inv-mask` spelling. Both collapse to a name ending
# in `..._inv_mask` after sanitization, so a single suffix match is enough.
_INV_MASK_NAME_RE = re.compile(r"_inv_mask$", re.IGNORECASE)


def auto_detect_liquid_type_ids(type_textures: Dict[int, "TypeTextureInfo"]) -> Set[int]:
    detected: Set[int] = set()
    for info in type_textures.values():
        icon = info.base_icon
        if icon and LIQUID_ICON_RE.match(icon):
            detected.add(info.type_id)
    return detected


def compute_liquid_phantom_positions(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    class_for: Callable[[int], Optional[int]],
    radius: int,
) -> Set[Tuple[int, int, int]]:
    """Dilate every real liquid cell by ``radius`` blocks in the 4 horizontal
    directions and return the set of phantom positions (positions not already
    holding a real liquid block). Vertical layers are NOT dilated -- water
    surfaces stay at their original Y so the lake silhouette is preserved.
    Phantoms intentionally include positions that hold a real non-liquid
    block (dirt, slope, ProBuilder tile, etc.); the renderer overlays liquid
    geometry on those cells so the visible water-side-wall strip becomes
    hidden inside the opaque neighbour.
    """

    if radius <= 0:
        return set()
    seeds: Set[Tuple[int, int, int]] = set()
    for pos, state in world_blocks.items():
        if is_liquid_type(class_for(state.block_id)):
            seeds.add(pos)
    if not seeds:
        return set()
    frontier: Set[Tuple[int, int, int]] = set(seeds)
    visited: Set[Tuple[int, int, int]] = set(seeds)
    phantoms: Set[Tuple[int, int, int]] = set()
    for _ in range(radius):
        next_frontier: Set[Tuple[int, int, int]] = set()
        for x, y, z in frontier:
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y, z + dz)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                if neighbor not in seeds:
                    phantoms.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return phantoms


def is_liquid_type(type_id: Optional[int]) -> bool:
    return type_id in LIQUID_TYPE_IDS


def _liquid_top_y(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    pos: Tuple[int, int, int],
    class_for: Callable[[int], Optional[int]],
) -> float:
    """Top Y in [0, 1] for a liquid cell, or -1 if pos is not liquid.

    Mirrors ``RenderBlocks.getLiquidHeightPercent``: when the column directly
    above is also liquid the cell is rendered as a full cube; otherwise the
    surface sits at ``1 - (level + 1) / 9`` below the top, matching the
    vanilla water/lava height table (source = 8/9, level 7 = 1/9).
    """

    cell = world_blocks.get(pos)
    if cell is not None and is_liquid_type(class_for(cell.block_id)):
        above = world_blocks.get((pos[0], pos[1] + 1, pos[2]))
        above_is_liquid = above is not None and is_liquid_type(class_for(above.block_id))
        if not above_is_liquid and (pos[0], pos[1] + 1, pos[2]) in LIQUID_PHANTOM_POSITIONS:
            above_is_liquid = True
        if above_is_liquid:
            return 1.0
        level = cell.meta & 0xF
        if level >= 8:
            # 8..15 are flowing-distance variants of a source block; treat as full level 0.
            level = 0
        return 1.0 - (level + 1) / 9.0
    if pos in LIQUID_PHANTOM_POSITIONS:
        # Phantom liquid: acts like a source block. If a real liquid (or another
        # phantom) sits directly above, it is rendered as a full cube to match
        # the column behaviour; otherwise its surface sits at 8/9 = source level.
        above_pos = (pos[0], pos[1] + 1, pos[2])
        above = world_blocks.get(above_pos)
        if (above is not None and is_liquid_type(class_for(above.block_id))) or above_pos in LIQUID_PHANTOM_POSITIONS:
            return 1.0
        return 8.0 / 9.0
    return -1.0


def liquid_corner_height(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    x: int,
    y: int,
    z: int,
    dx: int,
    dz: int,
    class_for: Callable[[int], Optional[int]],
) -> float:
    """Top Y at corner (x+dx, z+dz) of a liquid cell.

    Reproduces vanilla's 4-column average of liquid heights touching the
    corner. If any of the four neighbouring columns has liquid stacked
    above it, the corner snaps to 1.0 (full surface) just like in game.
    """

    has_full = False
    total = 0.0
    count = 0
    for ox in (-1, 0):
        for oz in (-1, 0):
            h = _liquid_top_y(world_blocks, (x + dx + ox, y, z + dz + oz), class_for)
            if h < 0:
                continue
            if h >= 1.0:
                has_full = True
            total += h
            count += 1
    if has_full:
        return 1.0
    if count == 0:
        h = _liquid_top_y(world_blocks, (x, y, z), class_for)
        return h if h > 0 else 8.0 / 9.0
    return total / count


def liquid_render_faces(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    pos: Tuple[int, int, int],
    class_for: Callable[[int], Optional[int]],
) -> List[Face]:
    """Render-mode surface mesh for a liquid cell with vanilla corner tilt.

    Side faces are culled only against fully-filled liquid columns so that
    water sitting next to slopes/half-blocks keeps its visible vertical face
    instead of being hidden behind the legacy `liquid_box` full cube. Top
    corners follow ``liquid_corner_height`` to match in-game tilt.
    """

    x, y, z = pos
    h00 = liquid_corner_height(world_blocks, x, y, z, 0, 0, class_for)
    h10 = liquid_corner_height(world_blocks, x, y, z, 1, 0, class_for)
    h11 = liquid_corner_height(world_blocks, x, y, z, 1, 1, class_for)
    h01 = liquid_corner_height(world_blocks, x, y, z, 0, 1, class_for)

    above_is_liquid = _liquid_top_y(world_blocks, (x, y + 1, z), class_for) >= 0.0
    faces: List[Face] = []

    if not above_is_liquid:
        faces.append((
            (0.0, h00, 0.0),
            (1.0, h10, 0.0),
            (1.0, h11, 1.0),
            (0.0, h01, 1.0),
        ))

    below_h = _liquid_top_y(world_blocks, (x, y - 1, z), class_for)
    if below_h < 0.0:
        faces.append((
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ))

    def neighbor_full(nx: int, ny: int, nz: int) -> bool:
        return _liquid_top_y(world_blocks, (nx, ny, nz), class_for) >= 1.0

    if not neighbor_full(x, y, z - 1):
        faces.append((
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, h00, 0.0),
            (1.0, h10, 0.0),
        ))
    if not neighbor_full(x, y, z + 1):
        faces.append((
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, h11, 1.0),
            (0.0, h01, 1.0),
        ))
    if not neighbor_full(x - 1, y, z):
        faces.append((
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, h01, 1.0),
            (0.0, h00, 0.0),
        ))
    if not neighbor_full(x + 1, y, z):
        faces.append((
            (1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, h10, 0.0),
            (1.0, h11, 1.0),
        ))
    return faces


def liquid_box(meta: int) -> AABB:
    """Legacy collision-mode AABB for a liquid cell (no neighbor context).

    Source/level-0 cells now resolve to height 8/9 instead of a misleading
    full cube so collision exports stop overlapping nearby slope geometry.
    Use :func:`liquid_render_faces` for the actual render mesh.
    """

    level = meta & 0xF
    if level >= 8:
        level = 0
    height = 1.0 - (level + 1) / 9.0
    return (0.0, 0.0, 0.0, 1.0, height, 1.0)


def connected_post_box(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    pos: Tuple[int, int, int],
    state: BlockState,
) -> AABB:
    def connects(dx: int, dz: int) -> bool:
        other = world_blocks.get((pos[0] + dx, pos[1], pos[2] + dz))
        return other is not None and other.block_id != 0

    x0, x1 = 0.375, 0.625
    z0, z1 = 0.375, 0.625
    if connects(0, -1):
        z0 = 0.0
    if connects(0, 1):
        z1 = 1.0
    if connects(-1, 0):
        x0 = 0.0
    if connects(1, 0):
        x1 = 1.0
    return (x0, 0.0, z0, x1, 1.0, z1)


def fence_boxes(
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    block_type_by_id: Dict[int, int],
    world_tiles: Dict[Tuple[int, int, int], ProBuilderTile],
    pos: Tuple[int, int, int],
    state: BlockState,
) -> List[AABB]:
    tile = world_tiles.get(pos)
    data = tile.data if tile is not None else 0
    fence_kind = data & 0xF
    high_kind = data >> 4

    def connects(dx: int, dy: int, dz: int) -> bool:
        other_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        other = world_blocks.get(other_pos)
        if other is None:
            return False
        other_class = block_type_by_id.get(other.block_id)
        if other.block_id in {state.block_id, 2404} or other_class == 142993:
            return True
        if dy == 1:
            return True
        if high_kind == 1:
            return False
        return other.block_id != 0

    south = connects(0, 0, -1)
    north = connects(0, 0, 1)
    west = connects(-1, 0, 0)
    east = connects(1, 0, 0)

    if fence_kind == 5 and not any((south, north, west, east)):
        x0, x1 = 0.4375, 0.5625
        z0, z1 = 0.4375, 0.5625
        y1 = 1.0 if connects(0, 1, 0) else 0.875
    elif fence_kind <= 5:
        x0, x1 = 0.375, 0.625
        z0, z1 = 0.375, 0.625
        y1 = 1.0
    else:
        x0, x1 = 0.25, 0.75
        z0, z1 = 0.25, 0.75
        y1 = 1.0

    if south:
        z0 = 0.0
    if north:
        z1 = 1.0
    if west:
        x0 = 0.0
    if east:
        x1 = 1.0
    if fence_kind > 3:
        if south and north and not west and not east:
            x0, x1 = 0.3125, 0.6875
        elif not south and not north and west and east:
            z0, z1 = 0.3125, 0.6875

    return [(x0, 0.0, z0, x1, y1, z1)]


class GeometryResolver:
    def __init__(
        self,
        block_type_by_id: Dict[int, int],
        shape_tables: Dict[int, Dict[int, List[AABB]]],
        world_tiles: Dict[Tuple[int, int, int], ProBuilderTile],
        world_blocks: Dict[Tuple[int, int, int], BlockState],
        external_configs: Optional[Dict[int, Dict[str, Any]]] = None,
        geometry_report: Optional[Dict[int, Dict[str, Any]]] = None,
        model_placeholder: str = "cross",
        mesh_mode: str = "collision",
    ) -> None:
        self.block_type_by_id = block_type_by_id
        self.shape_tables = shape_tables
        self.world_tiles = world_tiles
        self.world_blocks = world_blocks
        self.external_configs = external_configs or {}
        self.geometry_report = geometry_report or {}
        self.model_placeholder = model_placeholder
        self.mesh_mode = mesh_mode

    def class_for(self, block_id: int) -> Optional[int]:
        type_id = self.block_type_by_id.get(block_id)
        if type_id is not None:
            return type_id
        row = self.geometry_report.get(block_id)
        if row and isinstance(row.get("type_id", row.get("type_id")), int):
            return row.get("type_id", row.get("type_id"))
        return None

    def external_config_for_block(self, block_id: int) -> Optional[Dict[str, Any]]:
        cfg = self.external_configs.get(block_id)
        if not isinstance(cfg, dict):
            return None
        config = cfg.get("config") if isinstance(cfg.get("config"), dict) else cfg
        return config if isinstance(config, dict) else None

    def block_id_is_simple_external_model(self, block_id: int) -> bool:
        config = self.external_config_for_block(block_id)
        if config is None:
            return False
        return bool(config.get("model")) and bool(config.get("simple_render"))

    def external_model_mesh_for_block(self, block_id: int) -> Optional[ExternalModelMesh]:
        return None

    @staticmethod
    def external_model_rotation(meta: int) -> float:
        # 246003.char(n6): 1=270 deg, 2=180 deg, 3=90 deg, else identity.
        return {1: 1.5 * math.pi, 2: math.pi, 3: 0.5 * math.pi}.get(meta & 0x3, 0.0)

    @staticmethod
    def transform_external_model_face(
        face: Face,
        tx: float,
        tz: float,
        scale: float,
        rotation: float,
    ) -> Face:
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        out: List[Tuple[float, float, float]] = []
        for x, y, z in face:
            sx = x * scale
            sy = y * scale
            sz = z * scale
            # Converted mcsb/mcsa OBJ/GLB meshes are handed opposite to the
            # world-space X/Z rotation used by the game renderer. Mirroring the
            # rotation here keeps metadata-rotated side models (lianas, wall
            # decor) attached to the same block side as in-game.
            rx = sx * cos_r - sz * sin_r
            rz = sx * sin_r + sz * cos_r
            out.append((tx + rx, sy, tz + rz))
        return tuple(out)

    def external_model_face_entries_for(
        self,
        pos: Tuple[int, int, int],
        state: BlockState,
    ) -> List[ModelFace]:
        if self.mesh_mode != "render":
            return []
        mesh = self.external_model_mesh_for_block(state.block_id)
        if mesh is None:
            return []
        config = self.external_config_for_block(state.block_id) or {}

        min_instances = max(1, int(config.get("min_instances", 1) or 1))
        max_instances = max(min_instances, int(config.get("max_instances", min_instances) or min_instances))
        min_scale = float(config.get("min_scale", 1.0) or 1.0)
        max_scale = float(config.get("max_scale", min_scale) or min_scale)
        if max_scale < min_scale:
            max_scale = min_scale
        max_pos_offset = float(config.get("max_pos_offset", 0.0) or 0.0)
        random_rotation = bool(config.get("random_rotation"))

        x, y, z = pos
        rng = JavaRandom(x * 31 + y * 23 + z * 37)
        rng.set_seed(rng.next_long())
        count = min_instances + rng.next_int(max_instances - min_instances + 1)

        entries: List[ModelFace] = []
        for _index in range(count):
            tx = 0.5
            tz = 0.5
            if max_pos_offset != 0.0:
                tx += (rng.next_float() - 0.5) * max_pos_offset * 2.0
                tz += (rng.next_float() - 0.5) * max_pos_offset * 2.0
            scale = 1.0
            if min_scale != 1.0 or max_scale != 1.0:
                scale = min_scale + rng.next_float() * (max_scale - min_scale)
            rotation = rng.next_float() * math.tau if random_rotation else self.external_model_rotation(state.meta)
            for model_face in mesh.faces:
                transformed = self.transform_external_model_face(model_face.vertices, tx, tz, scale, rotation)
                entries.append(ModelFace(transformed, model_face.uvs))
        return entries

    def strategy_for(self, block_id: int) -> str:
        row = self.geometry_report.get(block_id)
        if row:
            geometry = row.get("geometry", {})
            strategy = geometry.get("strategy")
            if isinstance(strategy, str):
                return strategy
        if block_id in self.external_configs:
            return "external_config_block"
        type_id = self.class_for(block_id)
        if type_id == 131839:
            return "smt_slope_wrapper"
        if type_id == 168350:
            return "smt_double_slope_wrapper"
        if type_id in self.shape_tables:
            return "java_shape_table"
        return "base_full_cube"

    def payload_shape_key(self, pos: Tuple[int, int, int], state: BlockState) -> int:
        tile = self.world_tiles.get(pos)
        if tile is None:
            return state.meta
        # 288803.class() returns low 32 bits; 141081.new() combines data_ext.
        # Current parsed static AABB tables are int-keyed, so use the low word
        # for those tables but keep data_ext in the report/prototype metadata.
        return tile.data

    def payload_data_key(self, pos: Tuple[int, int, int]) -> Optional[int]:
        tile = self.world_tiles.get(pos)
        if tile is None:
            return None
        if tile.tile_type == 1:
            return tile_payload_u64(tile)
        return tile.data

    def tile_cover_block_id(self, tile: ProBuilderTile, side_index: int) -> Optional[int]:
        if side_index < 0 or side_index >= len(tile.covers):
            return None
        block_id = cover_id(tile.covers[side_index])
        if block_id <= 0:
            return None
        if self.class_for(block_id) is None and block_id not in self.external_configs:
            return None
        return block_id

    def tile_cover_block_state(self, tile: ProBuilderTile, side_index: int) -> Optional[BlockState]:
        block_id = self.tile_cover_block_id(tile, side_index)
        if block_id is None:
            return None
        if self.block_id_is_simple_external_model(block_id):
            # 153279 registers these through the .mcsa model renderer (246003).
            # Their config `icon` is an inventory/model texture, not a flat
            # ProBuilder cover material. Using it here paints walls with plant
            # item icons such as kiddney_weed/Kamish_Base.
            return None
        return BlockState(block_id, cover_meta(tile.covers[side_index]))

    def tile_cover_is_packed(self, tile: ProBuilderTile, side_index: int) -> bool:
        return bool(tile.packcovers & (1 << side_index))

    def tile_packed_cover_block_id(self, tile: ProBuilderTile, side_index: int) -> Optional[int]:
        if side_index < 0 or side_index >= 6:
            return None
        if not self.tile_cover_is_packed(tile, side_index):
            return None
        block_id = self.tile_cover_block_id(tile, side_index)
        if block_id is not None and self.block_id_is_simple_external_model(block_id):
            return None
        return block_id

    def tile_has_packed_side_covers(self, tile: ProBuilderTile) -> bool:
        for side_index in range(6):
            if self.tile_packed_cover_block_id(tile, side_index) is not None:
                return True
        return False

    def block_id_is_invisible_material(self, block_id: int) -> bool:
        return self.class_for(block_id) in INVISIBLE_MATERIAL_CLASS_IDS

    def invisible_surface_role_for_block(self, pos: Tuple[int, int, int], state: BlockState) -> Optional[Tuple[int, str]]:
        tile = self.world_tiles.get(pos)
        if tile is not None:
            cover6 = self.tile_cover_block_id(tile, 6)
            if cover6 is not None and self.block_id_is_invisible_material(cover6):
                if self.tile_has_packed_side_covers(tile):
                    return cover6, INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE
                return cover6, INVISIBLE_MATERIAL_ROLE_COVER_CARRIER
        if self.block_id_is_invisible_material(state.block_id):
            return state.block_id, INVISIBLE_MATERIAL_ROLE_DIRECT
        return None

    def invisible_material_block_id_for_surface(self, pos: Tuple[int, int, int], state: BlockState) -> Optional[int]:
        role = self.invisible_surface_role_for_block(pos, state)
        if role is None:
            return None
        return role[0]

    def invisible_material_name(self, block_id: int, role: Optional[str] = None) -> str:
        material = OBJ_INVISIBLE_MATERIAL_BY_BLOCK_ID.get(block_id, f"stc_invisible_block_{block_id}")
        if role:
            return f"{material}_{role}"
        return material

    def block_surface_is_invisible(self, pos: Tuple[int, int, int], state: BlockState) -> bool:
        role = self.invisible_surface_role_for_block(pos, state)
        if role is None:
            return False
        # Packed covers are the game's visible, no-collision surface baked onto
        # an invisible base material. Treat them as visible for material fallback
        # purposes, while still giving the unpainted faces their own material.
        return role[1] != INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE

    def material_for_block_surface(self, pos: Tuple[int, int, int], state: BlockState) -> str:
        role = self.invisible_surface_role_for_block(pos, state)
        if role is not None:
            invisible_block_id, material_role = role
            if material_role == INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE:
                return self.invisible_material_name(invisible_block_id, INVISIBLE_MATERIAL_ROLE_PACKED_BODY)
            return self.invisible_material_name(invisible_block_id, material_role)
        return OBJ_MATERIAL_VISIBLE

    def material_for_box_face(self, pos: Tuple[int, int, int], state: BlockState, face_name: str) -> str:
        role = self.invisible_surface_role_for_block(pos, state)
        if role is None:
            return OBJ_MATERIAL_VISIBLE

        invisible_block_id, material_role = role
        if material_role != INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE:
            return self.invisible_material_name(invisible_block_id, material_role)

        tile = self.world_tiles.get(pos)
        side = face_name.upper()
        side_index = SIDE_INDEX.get(side)
        if tile is not None and side_index is not None and self.tile_packed_cover_block_id(tile, side_index) is not None:
            return self.invisible_material_name(invisible_block_id, f"packed_{face_name.lower()}")
        return self.invisible_material_name(invisible_block_id, INVISIBLE_MATERIAL_ROLE_PACKED_BODY)

    def texture_block_state_for_surface(self, pos: Tuple[int, int, int], state: BlockState, side: str) -> Optional[BlockState]:
        role = self.invisible_surface_role_for_block(pos, state)
        tile = self.world_tiles.get(pos)
        side_index = SIDE_INDEX.get(side.upper())
        if role is not None:
            invisible_block_id, material_role = role
            if material_role == INVISIBLE_MATERIAL_ROLE_DIRECT and invisible_block_id == 4025:
                return state
            if material_role != INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE:
                return None
            if tile is not None and side_index is not None:
                packed_state = self.tile_cover_block_state(tile, side_index)
                if packed_state is not None and self.tile_cover_is_packed(tile, side_index):
                    return packed_state
            return None

        if tile is not None:
            if side_index is not None:
                packed_state = self.tile_cover_block_state(tile, side_index)
                if packed_state is not None and self.tile_cover_is_packed(tile, side_index):
                    return packed_state
            cover6_state = self.tile_cover_block_state(tile, 6)
            if cover6_state is not None and not self.block_id_is_invisible_material(cover6_state.block_id):
                return cover6_state
        return state

    def texture_material_side_for_surface(self, pos: Tuple[int, int, int], state: BlockState, side: str) -> Optional[str]:
        role = self.invisible_surface_role_for_block(pos, state)
        tile = self.world_tiles.get(pos)
        side = side.upper()
        side_index = SIDE_INDEX.get(side)
        if role is not None:
            invisible_block_id, material_role = role
            if material_role == INVISIBLE_MATERIAL_ROLE_DIRECT and invisible_block_id == 4025:
                return side
            if material_role != INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE:
                return None
            if tile is not None and side_index is not None:
                packed_state = self.tile_cover_block_state(tile, side_index)
                if packed_state is not None and self.tile_cover_is_packed(tile, side_index):
                    return side
            return None

        if tile is not None:
            if side_index is not None:
                packed_state = self.tile_cover_block_state(tile, side_index)
                if packed_state is not None and self.tile_cover_is_packed(tile, side_index):
                    return side
            cover6_state = self.tile_cover_block_state(tile, 6)
            if cover6_state is not None and not self.block_id_is_invisible_material(cover6_state.block_id):
                return "UNKNOWN"
        return side

    def wedge_xz_diagonal_render_side_for_surface(
        self,
        pos: Tuple[int, int, int],
        state: BlockState,
        local_face: Face,
    ) -> Optional[str]:
        """Return the native render side for a 143634 WEDGE_XZ diagonal face."""
        if self.class_for(state.block_id) != 284865:
            return None
        tile = self.world_tiles.get(pos)
        if tile is None:
            return None
        shape = tile.data & 0xFF
        render_side = WEDGE_XZ_DIAGONAL_RENDER_SIDE.get(shape)
        if render_side is None:
            return None
        if len(local_face) != 4:
            return None
        if any(face_lies_on_block_boundary(local_face, side) for side in SIDE_NAMES[:6]):
            return None
        ys = [vertex[1] for vertex in local_face]
        if abs(min(ys) - 0.0) > 1.0e-7 or abs(max(ys) - 1.0) > 1.0e-7:
            return None
        xz_points = {(round(vertex[0], 7), round(vertex[2], 7)) for vertex in local_face}
        if len(xz_points) != 2:
            return None
        xs = {point[0] for point in xz_points}
        zs = {point[1] for point in xz_points}
        if len(xs) < 2 or len(zs) < 2:
            return None
        return render_side

    def diagonal_grass_neighbor_state_for_surface(
        self,
        pos: Tuple[int, int, int],
        state: BlockState,
        local_face: Face,
    ) -> Optional[BlockState]:
        """Native-like material graft for WEDGE_XZ diagonal split faces.

        Some 284865 ProBuilder wedge cells visually borrow the neighboring
        grass block as the face material for the diagonal split. This is not a
        288803.overlay[] pass: the diagonal face itself is textured as grass,
        then the normal grass-side overlay path can add the green alpha layer.
        """
        if self.wedge_xz_diagonal_render_side_for_surface(pos, state, local_face) is None:
            return None
        tile = self.world_tiles.get(pos)
        if tile is None:
            return None
        shape = tile.data & 0xFF
        empty_sides = WEDGE_XZ_EMPTY_SIDES.get(shape)
        if empty_sides is None:
            return None
        cover6_block_id = self.tile_cover_block_id(tile, 6)
        if cover6_block_id not in GRASS_NEIGHBOR_GRAFT_BASE_IDS:
            return None

        grass_neighbor: Optional[BlockState] = None
        for empty_side in empty_sides:
            dx, dy, dz = SIDE_OFFSET[empty_side]
            neighbor = self.world_blocks.get((pos[0] + dx, pos[1] + dy, pos[2] + dz))
            if neighbor is not None and self.class_for(neighbor.block_id) == 111004:
                grass_neighbor = neighbor
                break
        return grass_neighbor

    @staticmethod
    def tile_uv_rotation(tile: Optional[ProBuilderTile], side: str) -> int:
        if tile is None:
            return 0
        side_index = SIDE_INDEX.get(side.upper())
        if side_index is None:
            return 0
        side_extra = 0
        if tile.side_extra is not None and side_index < len(tile.side_extra):
            side_extra = (tile.side_extra[side_index] & 0xC0) >> 6
        return (((tile.rot >> (side_index * 2)) & 0x3) + side_extra) % 4

    @staticmethod
    def tile_layer_index(side: str) -> Optional[int]:
        side = side.upper()
        if side in {"UNKNOWN", "DEFAULT", "COVER6"}:
            return 6
        return SIDE_INDEX.get(side)

    @staticmethod
    def tile_tint(tile: Optional[ProBuilderTile], side: str) -> Optional[Tuple[float, float, float]]:
        if tile is None:
            return None
        side_index = GeometryResolver.tile_layer_index(side)
        if side_index is None or side_index >= len(tile.overlays):
            return None
        value = tile.overlays[side_index]
        # 288803 stores color_short_* as signed Java shorts, but 80080.char(short)
        # decodes them as unsigned RGB565 via bit masks. Only -1 is the empty
        # sentinel; other negative values are valid packed colors.
        if value == -1:
            return None
        return rgb565_to_rgb(value)

    @staticmethod
    def tile_overlay_value(tile: Optional[ProBuilderTile], side: str) -> int:
        if tile is None:
            return 0
        side_index = GeometryResolver.tile_layer_index(side)
        if side_index is None or side_index >= len(tile.colors):
            return 0
        return tile.colors[side_index] & 0xFF

    def non_packed_tile_side_covers(self, tile: ProBuilderTile) -> List[Tuple[str, int]]:
        covers: List[Tuple[str, int]] = []
        for side in SIDE_NAMES[:6]:
            side_index = SIDE_INDEX[side]
            cover_block_id = self.tile_cover_block_id(tile, side_index)
            if cover_block_id is None:
                continue
            if self.tile_cover_is_packed(tile, side_index):
                continue
            covers.append((side, cover_block_id))
        return covers

    def non_packed_tile_side_cover_states(self, tile: ProBuilderTile) -> List[Tuple[str, BlockState]]:
        covers: List[Tuple[str, BlockState]] = []
        for side in SIDE_NAMES[:6]:
            side_index = SIDE_INDEX[side]
            cover_state = self.tile_cover_block_state(tile, side_index)
            if cover_state is None:
                continue
            if self.tile_cover_is_packed(tile, side_index):
                continue
            covers.append((side, cover_state))
        return covers

    def probuilder_grid_cover_faces(self, tile: ProBuilderTile) -> List[Face]:
        return [face for face, _side, _cover_state in self.probuilder_grid_cover_face_entries(tile)]

    def probuilder_grid_cover_face_entries(self, tile: ProBuilderTile) -> List[Tuple[Face, str, BlockState]]:
        payload = tile_payload_u64(tile)
        if payload == 0:
            return []

        entries: List[Tuple[Face, str, BlockState]] = []
        for side, surface_box in probuilder_grid_exposed_rectangles(payload):
            side_index = SIDE_INDEX[side]
            cover_state = self.tile_cover_block_state(tile, side_index)
            if cover_state is None:
                continue
            if self.tile_cover_is_packed(tile, side_index):
                continue
            entries.extend((face, side, cover_state) for face in cover_plate_faces(surface_box, side, cover_state.block_id))
        return entries

    def box_face_visible_for_cover(self, pos: Tuple[int, int, int], box: AABB, side: str) -> bool:
        # 188933.char(x,y,z,side): partial/internal faces are renderable; outer
        # faces ask the world whether the neighbour blocks that side.
        if not face_on_block_boundary(side.lower(), box):
            return True
        dx, dy, dz = SIDE_OFFSET[side]
        x, y, z = pos
        return not self.is_full_cell((x + dx, y + dy, z + dz), include_invisible=False)

    def generic_tile_cover_faces(self, pos: Tuple[int, int, int], state: BlockState, tile: ProBuilderTile) -> List[Face]:
        return [face for face, _side, _cover_state in self.generic_tile_cover_face_entries(pos, state, tile)]

    def generic_tile_cover_face_entries(
        self,
        pos: Tuple[int, int, int],
        state: BlockState,
        tile: ProBuilderTile,
    ) -> List[Tuple[Face, str, BlockState]]:
        if self.mesh_mode != "render":
            return []
        if self.class_for(state.block_id) == 132769:
            # 92433 has a dedicated 4x4 voxel cover pass; the generic AABB pass
            # would duplicate and flatten its per-cell cover rectangles.
            return []

        side_covers = self.non_packed_tile_side_cover_states(tile)
        if not side_covers:
            return []

        entries: List[Tuple[Face, str, BlockState]] = []
        for box in self.boxes_for(pos, state):
            for side, cover_state in side_covers:
                if self.box_face_visible_for_cover(pos, box, side):
                    entries.extend((face, side, cover_state) for face in cover_plate_faces(box, side, cover_state.block_id))
        return entries

    def smooth_tile_cover_faces(self, pos: Tuple[int, int, int], tile: ProBuilderTile, surface_faces: Iterable[Face]) -> List[Face]:
        return [face for face, _side, _cover_state in self.smooth_tile_cover_face_entries(pos, tile, surface_faces)]

    def smooth_tile_cover_face_entries(
        self,
        pos: Tuple[int, int, int],
        tile: ProBuilderTile,
        surface_faces: Iterable[Face],
    ) -> List[Tuple[Face, str, BlockState]]:
        if self.mesh_mode != "render":
            return []
        side_covers = dict(self.non_packed_tile_side_cover_states(tile))
        if not side_covers:
            return []

        x, y, z = pos
        entries: List[Tuple[Face, str, BlockState]] = []
        for surface_face in surface_faces:
            side = cover_side_for_face(surface_face)
            if side is None:
                continue
            cover_state = side_covers.get(side)
            if cover_state is None:
                continue
            if face_lies_on_block_boundary(surface_face, side):
                dx, dy, dz = SIDE_OFFSET[side]
                if self.is_full_cell((x + dx, y + dy, z + dz), include_invisible=False):
                    continue
            entries.extend((face, side, cover_state) for face in cover_prism_faces(surface_face, side, cover_state.block_id))
        return entries

    def cover_faces_for(self, pos: Tuple[int, int, int], state: BlockState) -> List[Face]:
        return [face for face, _side, _cover_state in self.cover_face_entries_for(pos, state)]

    def cover_face_entries_for(self, pos: Tuple[int, int, int], state: BlockState) -> List[Tuple[Face, str, BlockState]]:
        if self.mesh_mode != "render":
            return []
        tile = self.world_tiles.get(pos)
        if tile is None:
            return []

        type_id = self.class_for(state.block_id)
        if type_id == 284865:
            return self.smooth_tile_cover_face_entries(pos, tile, smooth_slope_faces_for_j(tile.data & 0xFF))
        if type_id == 132769 and tile.tile_type == 1:
            return self.probuilder_grid_cover_face_entries(tile)
        if type_id == 151250:
            return self.smooth_tile_cover_face_entries(pos, tile, collapsible_smooth_faces(tile))
        if type_id in {3448, 25546, 40761}:
            surface_faces = triangular_panel_smooth_faces(tile.data)
            return [
                *self.generic_tile_cover_face_entries(pos, state, tile),
                *self.smooth_tile_cover_face_entries(pos, tile, surface_faces),
            ]
        return self.generic_tile_cover_face_entries(pos, state, tile)

    def boxes_for(self, pos: Tuple[int, int, int], state: BlockState) -> List[AABB]:
        cfg = self.external_configs.get(state.block_id)
        if cfg is not None:
            if cfg.get("model") and cfg.get("simple_render") and self.model_placeholder in {"cross", "none"}:
                return []
            return [external_config_box(cfg, state.meta)]

        type_id = self.class_for(state.block_id)
        if type_id == 106044:
            return pane_boxes(self.world_blocks, self.block_type_by_id, *pos)
        if is_liquid_type(type_id):
            # Render mode emits the tilted surface mesh from local_faces_for so
            # neighbouring slopes/half-blocks no longer clash with a phantom
            # full-cube water hitbox. Collision mode keeps a flat AABB.
            if self.mesh_mode == "render":
                return []
            above = self.world_blocks.get((pos[0], pos[1] + 1, pos[2]))
            if above is not None and is_liquid_type(self.class_for(above.block_id)):
                return [FULL_CUBE]
            return [liquid_box(state.meta)]
        if type_id == 216960:
            return [connected_post_box(self.world_blocks, pos, state)]
        if type_id in {131093, 125838, 8358}:
            return []
        if type_id in {273574, 97976, 158138}:
            if self.mesh_mode == "render":
                return [FULL_CUBE]
            return []
        if type_id == 95760:
            return [(-0.1, -0.8, -0.1, 1.1, 1.0, 1.1)]
        if type_id == 131839:
            j = SLOPE_META_TO_J.get(state.meta & 0xF, 0)
            if self.mesh_mode == "render" and smooth_slope_supported(j):
                return []
            return slope_boxes(state.meta, double_slope=False)
        if type_id == 168350:
            j = SLOPE_META_TO_J.get(state.meta & 0xF, 0)
            if self.mesh_mode == "render" and smooth_slope_supported(j):
                return [FULL_CUBE]
            return slope_boxes(state.meta, double_slope=True)
        if type_id == 284865:
            tile = self.world_tiles.get(pos)
            if tile is not None:
                j = tile.data & 0xFF
                if self.mesh_mode == "render" and smooth_slope_supported(j):
                    return []
                return slope_boxes_for_j(tile.data & 0xFF)
        if type_id == 132769:
            tile = self.world_tiles.get(pos)
            if tile is not None and tile.tile_type == 1:
                if self.mesh_mode == "render":
                    return []
                return probuilder_grid_boxes(tile)
        if type_id == 151250:
            tile = self.world_tiles.get(pos)
            if tile is not None:
                if self.mesh_mode == "render":
                    return []
                return collapsible_boxes(tile)
        if type_id == 142993:
            return fence_boxes(self.world_blocks, self.block_type_by_id, self.world_tiles, pos, state)
        if type_id == 30886:
            return [button_box(state, self.world_tiles.get(pos))]
        if type_id == 213644:
            return []
        if type_id == 3448:
            tile = self.world_tiles.get(pos)
            shape = tile.data if tile is not None else state.meta
            if self.mesh_mode == "render":
                return triangular_halfstep_base_boxes(shape)
        if type_id == 25546:
            tile = self.world_tiles.get(pos)
            if tile is not None:
                if self.mesh_mode == "render":
                    return []
                return triangular_panel_boxes(tile.data)
        if type_id == 40761:
            tile = self.world_tiles.get(pos)
            if tile is not None:
                if self.mesh_mode == "render":
                    return [triangular_panel_base_box(tile.data)]
                return [triangular_panel_base_box(tile.data), *triangular_panel_boxes(tile.data)]

        shape_key = self.payload_shape_key(pos, state)
        payload_key = self.payload_data_key(pos)
        table = self.shape_tables.get(type_id or -1)
        if table:
            if payload_key is not None and payload_key in table:
                return table[payload_key]
            return table.get(shape_key) or table.get(shape_key & 0xF) or [FULL_CUBE]
        return [FULL_CUBE]

    def local_faces_for(self, pos: Tuple[int, int, int], state: BlockState, include_covers: bool = True) -> List[Face]:
        cfg = self.external_configs.get(state.block_id)
        if not cfg or not cfg.get("model") or not cfg.get("simple_render"):
            type_id = self.class_for(state.block_id)
            if self.mesh_mode == "render":
                if is_liquid_type(type_id):
                    return liquid_render_faces(self.world_blocks, pos, self.class_for)
                if type_id == 131839:
                    j = SLOPE_META_TO_J.get(state.meta & 0xF, 0)
                    return smooth_slope_faces_for_j(j)
                if type_id == 168350:
                    j = SLOPE_META_TO_J.get(state.meta & 0xF, 0)
                    return translated_faces(smooth_slope_faces_for_j(j), dy=1.0)
                if type_id == 284865:
                    tile = self.world_tiles.get(pos)
                    if tile is not None:
                        surface_faces = smooth_slope_faces_for_j(tile.data & 0xFF)
                        if include_covers:
                            return [*surface_faces, *self.smooth_tile_cover_faces(pos, tile, surface_faces)]
                        return surface_faces
                if type_id == 132769:
                    tile = self.world_tiles.get(pos)
                    if tile is not None and tile.tile_type == 1:
                        surface_faces = probuilder_grid_render_faces(tile)
                        if include_covers:
                            return [*surface_faces, *self.probuilder_grid_cover_faces(tile)]
                        return surface_faces
                if type_id == 151250:
                    tile = self.world_tiles.get(pos)
                    if tile is not None:
                        surface_faces = collapsible_smooth_faces(tile)
                        if include_covers:
                            return [*surface_faces, *self.smooth_tile_cover_faces(pos, tile, surface_faces)]
                        return surface_faces
                if type_id == 3448:
                    tile = self.world_tiles.get(pos)
                    shape = tile.data if tile is not None else state.meta
                    surface_faces = triangular_panel_smooth_faces(shape)
                    cover_faces = []
                    if include_covers and tile is not None:
                        cover_faces = [
                            *self.generic_tile_cover_faces(pos, state, tile),
                            *self.smooth_tile_cover_faces(pos, tile, surface_faces),
                        ]
                    return [*surface_faces, *cover_faces]
                if type_id == 25546:
                    tile = self.world_tiles.get(pos)
                    if tile is not None:
                        surface_faces = triangular_panel_smooth_faces(tile.data)
                        if not include_covers:
                            return surface_faces
                        return [
                            *surface_faces,
                            *self.generic_tile_cover_faces(pos, state, tile),
                            *self.smooth_tile_cover_faces(pos, tile, surface_faces),
                        ]
                if type_id == 40761:
                    tile = self.world_tiles.get(pos)
                    if tile is not None:
                        surface_faces = triangular_panel_smooth_faces(tile.data)
                        if not include_covers:
                            return surface_faces
                        return [
                            *surface_faces,
                            *self.generic_tile_cover_faces(pos, state, tile),
                            *self.smooth_tile_cover_faces(pos, tile, surface_faces),
                        ]
                tile = self.world_tiles.get(pos)
                if include_covers and tile is not None:
                    return self.generic_tile_cover_faces(pos, state, tile)
            if type_id == 213644:
                return ladder_faces(state.meta)
            if type_id in {125838, 8358}:
                return plant_cross_faces(0.4, 0.8)
            if type_id == 131093:
                return plant_cross_faces(0.2, 0.6)
            return []
        if self.model_placeholder == "none":
            return []
        if self.model_placeholder == "cube":
            return []
        return model_cross_faces()

    def group_name_for(self, pos: Tuple[int, int, int], state: BlockState, group_by_cover: bool) -> str:
        if not group_by_cover:
            return f"block_{state.block_id}"
        tile = self.world_tiles.get(pos)
        if tile is None:
            return f"block_{state.block_id}"
        payload_key = self.payload_data_key(pos)
        if payload_key is None:
            payload_key = tile.data
        cover_parts = []
        for side_index in range(min(7, len(tile.covers))):
            cover_block_id = self.tile_cover_block_id(tile, side_index)
            if cover_block_id is None:
                continue
            packed = "p" if side_index < 6 and self.tile_cover_is_packed(tile, side_index) else "u"
            cover_parts.append(f"{side_index}_{cover_block_id}_{cover_meta(tile.covers[side_index])}_{packed}")
        cover_key = "-".join(cover_parts) if cover_parts else "none"
        return f"block_{state.block_id}_covers_{cover_key}_data_{payload_key:x}"

    def is_full_cell(self, pos: Tuple[int, int, int], include_invisible: bool = True) -> bool:
        state = self.world_blocks.get(pos)
        if state is None:
            return False
        if not include_invisible and self.block_surface_is_invisible(pos, state):
            return False
        boxes = self.boxes_for(pos, state)
        return len(boxes) == 1 and is_full_aabb(boxes[0])


FACE_DEFS = [
    ("down", (0, -1, 0), lambda x0, y0, z0, x1, y1, z1: [(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)]),
    ("up", (0, 1, 0), lambda x0, y0, z0, x1, y1, z1: [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]),
    ("north", (0, 0, -1), lambda x0, y0, z0, x1, y1, z1: [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)]),
    ("south", (0, 0, 1), lambda x0, y0, z0, x1, y1, z1: [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
    ("west", (-1, 0, 0), lambda x0, y0, z0, x1, y1, z1: [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
    ("east", (1, 0, 0), lambda x0, y0, z0, x1, y1, z1: [(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)]),
]


def face_on_block_boundary(face_name: str, box: AABB) -> bool:
    x0, y0, z0, x1, y1, z1 = box
    return (
        (face_name == "down" and y0 == 0.0)
        or (face_name == "up" and y1 == 1.0)
        or (face_name == "north" and z0 == 0.0)
        or (face_name == "south" and z1 == 1.0)
        or (face_name == "west" and x0 == 0.0)
        or (face_name == "east" and x1 == 1.0)
    )


def iter_region_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("reg.*.*.mdat"))


def load_world(
    region_files: Sequence[Path],
    max_chunks: Optional[int],
    empty_ids: Set[int],
    include_ids: Optional[Set[int]] = None,
    max_blocks: Optional[int] = None,
    bbox: Optional[BBox] = None,
    collect_biomes: bool = False,
) -> Any:
    world_blocks: Dict[Tuple[int, int, int], BlockState] = {}
    world_tiles: Dict[Tuple[int, int, int], ProBuilderTile] = {}
    world_biomes: Dict[Tuple[int, int], int] = {}
    chunks_loaded = 0
    for region in region_files:
        for chunk in read_region(region):
            chunks_loaded += 1
            if not chunk_intersects_bbox(chunk.chunk_x, chunk.chunk_z, bbox):
                if max_chunks is not None and chunks_loaded >= max_chunks:
                    if collect_biomes:
                        return world_blocks, world_tiles, world_biomes, chunks_loaded
                    return world_blocks, world_tiles, chunks_loaded
                continue
            base_x = chunk.chunk_x << 4
            base_z = chunk.chunk_z << 4
            if collect_biomes and len(chunk.biomes) >= 256:
                for local_z in range(16):
                    for local_x in range(16):
                        world_x = base_x + local_x
                        world_z = base_z + local_z
                        if bbox is not None:
                            x0, _y0, z0, x1, _y1, z1 = bbox
                            if world_x < x0 or world_x > x1 or world_z < z0 or world_z > z1:
                                continue
                        world_biomes[(world_x, world_z)] = chunk.biomes[(local_z << 4) | local_x] & 0xFF
            for (x, y, z), state in chunk.blocks.items():
                if state.block_id in empty_ids:
                    continue
                if include_ids is not None and state.block_id not in include_ids:
                    continue
                world_pos = (base_x + x, y, base_z + z)
                if not pos_in_bbox(world_pos, bbox):
                    continue
                world_blocks[world_pos] = state
                if max_blocks is not None and len(world_blocks) >= max_blocks:
                    world_tiles.update(chunk.tiles)
                    if collect_biomes:
                        return world_blocks, world_tiles, world_biomes, chunks_loaded
                    return world_blocks, world_tiles, chunks_loaded
            world_tiles.update(chunk.tiles)
            if max_chunks is not None and chunks_loaded >= max_chunks:
                if collect_biomes:
                    return world_blocks, world_tiles, world_biomes, chunks_loaded
                return world_blocks, world_tiles, chunks_loaded
    if collect_biomes:
        return world_blocks, world_tiles, world_biomes, chunks_loaded
    return world_blocks, world_tiles, chunks_loaded


def load_world_from_chunk_iter(
    chunks: Iterable[ChunkData],
    max_chunks: Optional[int],
    empty_ids: Set[int],
    include_ids: Optional[Set[int]] = None,
    max_blocks: Optional[int] = None,
    bbox: Optional[BBox] = None,
    collect_biomes: bool = False,
    replace_duplicate_chunks: bool = True,
) -> Any:
    world_blocks: Dict[Tuple[int, int, int], BlockState] = {}
    world_tiles: Dict[Tuple[int, int, int], ProBuilderTile] = {}
    world_biomes: Dict[Tuple[int, int], int] = {}
    chunk_block_positions: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    chunk_tile_positions: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    chunk_biome_positions: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    chunks_loaded = 0

    for chunk in chunks:
        chunks_loaded += 1
        chunk_key = (chunk.chunk_x, chunk.chunk_z)

        if replace_duplicate_chunks and chunk_key in chunk_block_positions:
            for pos in chunk_block_positions.pop(chunk_key, []):
                world_blocks.pop(pos, None)
            for pos in chunk_tile_positions.pop(chunk_key, []):
                world_tiles.pop(pos, None)
            for pos in chunk_biome_positions.pop(chunk_key, []):
                world_biomes.pop(pos, None)

        if not chunk_intersects_bbox(chunk.chunk_x, chunk.chunk_z, bbox):
            if max_chunks is not None and chunks_loaded >= max_chunks:
                break
            continue

        base_x = chunk.chunk_x << 4
        base_z = chunk.chunk_z << 4
        added_blocks: List[Tuple[int, int, int]] = []
        added_biomes: List[Tuple[int, int]] = []

        if collect_biomes and len(chunk.biomes) >= 256:
            for local_z in range(16):
                for local_x in range(16):
                    world_x = base_x + local_x
                    world_z = base_z + local_z
                    if bbox is not None:
                        x0, _y0, z0, x1, _y1, z1 = bbox
                        if world_x < x0 or world_x > x1 or world_z < z0 or world_z > z1:
                            continue
                    biome_pos = (world_x, world_z)
                    world_biomes[biome_pos] = chunk.biomes[(local_z << 4) | local_x] & 0xFF
                    added_biomes.append(biome_pos)

        for (x, y, z), state in chunk.blocks.items():
            if state.block_id in empty_ids:
                continue
            if include_ids is not None and state.block_id not in include_ids:
                continue
            world_pos = (base_x + x, y, base_z + z)
            if not pos_in_bbox(world_pos, bbox):
                continue
            world_blocks[world_pos] = state
            added_blocks.append(world_pos)
            if max_blocks is not None and len(world_blocks) >= max_blocks:
                world_tiles.update(chunk.tiles)
                chunk_block_positions[chunk_key] = added_blocks
                chunk_tile_positions[chunk_key] = list(chunk.tiles.keys())
                chunk_biome_positions[chunk_key] = added_biomes
                if collect_biomes:
                    return world_blocks, world_tiles, world_biomes, chunks_loaded
                return world_blocks, world_tiles, chunks_loaded

        world_tiles.update(chunk.tiles)
        chunk_block_positions[chunk_key] = added_blocks
        chunk_tile_positions[chunk_key] = list(chunk.tiles.keys())
        chunk_biome_positions[chunk_key] = added_biomes

        if max_chunks is not None and chunks_loaded >= max_chunks:
            break

    if collect_biomes:
        return world_blocks, world_tiles, world_biomes, chunks_loaded
    return world_blocks, world_tiles, chunks_loaded


def compute_center_offset(positions: Iterable[Tuple[int, int, int]]) -> Tuple[float, float, float]:
    pos_list = list(positions)
    if not pos_list:
        return (0.0, 0.0, 0.0)
    min_x = min(p[0] for p in pos_list)
    max_x = max(p[0] for p in pos_list)
    min_y = min(p[1] for p in pos_list)
    max_y = max(p[1] for p in pos_list)
    min_z = min(p[2] for p in pos_list)
    max_z = max(p[2] for p in pos_list)
    return ((min_x + max_x + 1) / 2.0, (min_y + max_y + 1) / 2.0, (min_z + max_z + 1) / 2.0)


class GltfPrimitive:
    """Per-material vertex/index accumulator for one glTF mesh primitive."""

    __slots__ = ("position", "uv", "color", "indices", "vertex_count", "index_count", "pmin", "pmax")

    def __init__(self) -> None:
        self.position = bytearray()
        self.uv = bytearray()
        self.color = bytearray()
        self.indices = bytearray()
        self.vertex_count = 0
        self.index_count = 0
        self.pmin = [float("inf"), float("inf"), float("inf")]
        self.pmax = [float("-inf"), float("-inf"), float("-inf")]

    def add_polygon(
        self,
        positions: Sequence[Tuple[float, float, float]],
        uvs: Sequence[Tuple[float, float]],
        color_rgba_u8: Tuple[int, int, int, int],
    ) -> None:
        n = len(positions)
        if n < 3 or len(uvs) != n:
            return
        start = self.vertex_count
        cr, cg, cb, ca = color_rgba_u8
        for (x, y, z), (u, v) in zip(positions, uvs):
            self.position += struct.pack("<fff", x, y, z)
            # glTF UV origin is top-left (V=0 at top of image); our `uv_for_face`
            # follows the OBJ convention (V=0 at bottom). Flip V here so that
            # asymmetric sprites (foliage, flowers, signs) render right-side-up.
            self.uv += struct.pack("<ff", u, 1.0 - v)
            self.color += struct.pack("<BBBB", cr, cg, cb, ca)
            if x < self.pmin[0]:
                self.pmin[0] = x
            if y < self.pmin[1]:
                self.pmin[1] = y
            if z < self.pmin[2]:
                self.pmin[2] = z
            if x > self.pmax[0]:
                self.pmax[0] = x
            if y > self.pmax[1]:
                self.pmax[1] = y
            if z > self.pmax[2]:
                self.pmax[2] = z
        for i in range(1, n - 1):
            self.indices += struct.pack("<III", start, start + i, start + i + 1)
            self.index_count += 3
        self.vertex_count += n


class GltfBuilder:
    """Collects images, materials, and per-material primitive geometry."""

    def __init__(self) -> None:
        self.images: List[Tuple[str, bytes]] = []  # (sanitized_name, png_bytes)
        self.image_indices: Dict[str, int] = {}    # icon_name -> image index
        self.materials: List[Dict[str, Any]] = []
        self.material_indices: Dict[Tuple[Any, ...], int] = {}
        self.primitives: Dict[int, GltfPrimitive] = {}

    def _image_for_icon(self, icon: str, png_bytes: bytes) -> int:
        cached = self.image_indices.get(icon)
        if cached is not None:
            return cached
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", icon).strip("_") or f"img_{len(self.images)}"
        index = len(self.images)
        self.images.append((safe[:120], png_bytes))
        self.image_indices[icon] = index
        return index

    def material_for_descriptor(self, descriptor: FaceMaterialDescriptor, *, double_sided: bool = True, force_mask: bool = False, hide_invisible: bool = False) -> int:
        alpha_mode = descriptor.alpha_mode
        # Identify the in-engine "invisible technical material" sprite. STALCRAFT
        # registers the same `inv_mask` icon for barriers, fake-light carriers,
        # interference markers, and packed-surface fallback sides. Matching by
        # icon name catches every emission path (DIRECT role, COVER_CARRIER,
        # PACKED_SURFACE leftover side) with a single regex. When detected and
        # `hide_invisible` is on, return the SKIP sentinel so callers drop the
        # face entirely -- glTF viewers handle BLEND/alpha=0 materials
        # inconsistently, so dropping geometry is the only viewer-agnostic way
        # to hide the technical surface.
        # Detect the invisible-mask sprite by the FINAL material name, not by
        # the raw icon. The in-engine sprite is registered as a bare icon
        # (e.g. `stalcraft:inv`) and `_material_name_for_icon` appends `_mask`
        # ONLY when the resolved alpha mode is MASK, so the literal substring
        # `inv_mask` never appears in the icon path itself -- earlier raw-icon
        # checks were missing the material entirely. Computing the prospective
        # material name first catches every namespace + alpha-mode combination
        # in a single comparison.
        if hide_invisible:
            prospective_alpha_mode = alpha_mode
            if force_mask and prospective_alpha_mode == "BLEND":
                prospective_alpha_mode = "MASK"
            prospective_descriptor = descriptor
            if prospective_alpha_mode != descriptor.alpha_mode:
                prospective_descriptor = replace(descriptor, alpha_mode=prospective_alpha_mode)
            prospective_name = self._material_name_for_icon(prospective_descriptor)
            if _INV_MASK_NAME_RE.search(prospective_name):
                return SKIP_MATERIAL_INDEX
        if force_mask and alpha_mode == "BLEND":
            # glTF BLEND materials don't write to depth, so plant/foliage sprites
            # show through each other in viewers. Promoting to MASK with a 0.5
            # cutoff produces correct depth-sorted opaque-cutout rendering.
            alpha_mode = "MASK"
        emissive_key = None if descriptor.emissive is None else tuple(round(c, 4) for c in descriptor.emissive)
        key = ("desc", descriptor.icon, alpha_mode, emissive_key, bool(double_sided))
        existing = self.material_indices.get(key)
        if existing is not None:
            return existing
        material: Dict[str, Any] = {
            "name": self._material_name_for_icon(descriptor),
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": alpha_mode,
            "doubleSided": bool(double_sided),
        }
        if alpha_mode == "MASK":
            material["alphaCutoff"] = 0.5
        if descriptor.icon and descriptor.png_bytes is not None:
            image_index = self._image_for_icon(descriptor.icon, descriptor.png_bytes)
            material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": image_index}
        elif descriptor.icon is None:
            # Missing texture fallback. Bright magenta to make it obvious.
            material["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 0.0, 1.0, 1.0]
        if descriptor.emissive is not None:
            material["emissiveFactor"] = [float(descriptor.emissive[0]), float(descriptor.emissive[1]), float(descriptor.emissive[2])]
        index = len(self.materials)
        self.materials.append(material)
        self.material_indices[key] = index
        return index

    def fallback_material(
        self,
        name: str,
        color: Tuple[float, float, float],
        *,
        alpha: float = 1.0,
        alpha_mode: str = "OPAQUE",
        emissive: Optional[Tuple[float, float, float]] = None,
        double_sided: bool = True,
    ) -> int:
        key = ("fallback", name, alpha_mode, alpha, emissive, bool(double_sided))
        existing = self.material_indices.get(key)
        if existing is not None:
            return existing
        material: Dict[str, Any] = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [color[0], color[1], color[2], alpha],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "alphaMode": alpha_mode,
            "doubleSided": bool(double_sided),
        }
        if alpha_mode == "MASK":
            material["alphaCutoff"] = 0.5
        if emissive is not None:
            material["emissiveFactor"] = list(emissive)
        index = len(self.materials)
        self.materials.append(material)
        self.material_indices[key] = index
        return index

    def primitive_for_material(self, material_index: int) -> GltfPrimitive:
        prim = self.primitives.get(material_index)
        if prim is None:
            prim = GltfPrimitive()
            self.primitives[material_index] = prim
        return prim

    @staticmethod
    def _material_name_for_icon(descriptor: FaceMaterialDescriptor) -> str:
        if not descriptor.icon:
            return f"stc_missing_{descriptor.alpha_mode.lower()}"
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", descriptor.icon).strip("_")
        suffix = "" if descriptor.alpha_mode == "OPAQUE" else f"_{descriptor.alpha_mode.lower()}"
        return f"stc_tex_{safe}{suffix}"


# Fallback palette used for invisible
# carriers, fake-light blocks, and packed-cover role surfaces. The glTF writer
# resolves these names to materials lazily through `_resolve_fallback_material`.
GLB_FALLBACK_PALETTE: Dict[str, Tuple[Tuple[float, float, float], float, str, Optional[Tuple[float, float, float]]]] = {
    OBJ_MATERIAL_VISIBLE: ((0.72, 0.72, 0.72), 1.0, "OPAQUE", None),
    OBJ_MATERIAL_COVER: ((0.86, 0.78, 0.56), 1.0, "OPAQUE", None),
    OBJ_MATERIAL_INVISIBLE: ((1.0, 0.1, 0.1), 0.18, "BLEND", None),
}
GLB_INVISIBLE_BLOCK_BASE: Dict[int, Tuple[Tuple[float, float, float], Optional[Tuple[float, float, float]]]] = {
    3928: ((0.95, 0.25, 0.10), (1.0, 0.4, 0.1)),
    3929: ((0.95, 0.45, 0.10), (1.0, 0.5, 0.1)),
    4018: ((0.95, 0.05, 0.95), None),
    4023: ((0.10, 0.45, 1.00), None),
    4024: ((0.10, 0.75, 1.00), None),
    4025: ((0.10, 1.00, 0.30), None),
}
GLB_INVISIBLE_ROLE_PALETTE: Dict[str, Tuple[Tuple[float, float, float], float]] = {
    INVISIBLE_MATERIAL_ROLE_DIRECT: ((1.00, 0.10, 0.10), 0.35),
    INVISIBLE_MATERIAL_ROLE_COVER_CARRIER: ((1.00, 0.00, 0.75), 0.20),
    INVISIBLE_MATERIAL_ROLE_PACKED_SURFACE: ((0.10, 1.00, 0.30), 1.0),
    INVISIBLE_MATERIAL_ROLE_PACKED_BODY: ((0.45, 0.45, 0.45), 0.30),
    "packed_down": ((0.15, 0.45, 1.00), 1.0),
    "packed_up": ((0.10, 1.00, 0.30), 1.0),
    "packed_north": ((1.00, 0.85, 0.10), 1.0),
    "packed_south": ((1.00, 0.55, 0.10), 1.0),
    "packed_west": ((0.85, 0.20, 1.00), 1.0),
    "packed_east": ((0.20, 1.00, 0.90), 1.0),
}


# Sentinel returned by material lookups when ``hide_invisible`` is on and the
# requested material identifies an invisible technical surface (in-engine
# `inv_mask` sprite or any `stc_invisible_*_direct`/`_cover_carrier` fallback
# material). `emit_world_face` detects this value and drops the face
# entirely -- glTF viewers don't honour BLEND/alpha=0 uniformly so removing
# geometry is the only viewer-agnostic fix.
SKIP_MATERIAL_INDEX: int = -1

# Invisible-material roles whose faces should vanish when ``hide_invisible``
# is on. PACKED_SURFACE / packed_<side> are intentionally excluded: those
# names are emitted for packed-cover sides that DO carry a visible game
# surface (they only carry the invisible-base for layering / grouping).
INVISIBLE_HIDDEN_ROLES: Set[str] = {
    INVISIBLE_MATERIAL_ROLE_DIRECT,
    INVISIBLE_MATERIAL_ROLE_COVER_CARRIER,
}


def _resolve_fallback_material(builder: "GltfBuilder", material_name: str, *, hide_invisible: bool = False) -> int:
    # When ``hide_invisible`` is on, replace the fallback palette for every
    # invisible-technical-material name with a fully-transparent BLEND
    # material (alpha=0). PACKED_SURFACE / packed_<side> variants stay
    # untouched -- those carry visible packed-cover game surfaces.
    # Only the texture-driven `inv_mask` icon (handled in
    # `material_for_descriptor`) drops faces outright via SKIP.
    def _transparent_stub() -> int:
        return builder.fallback_material(
            material_name + "_transparent",
            (1.0, 1.0, 1.0),
            alpha=0.0,
            alpha_mode="BLEND",
        )

    if hide_invisible:
        if material_name == OBJ_MATERIAL_INVISIBLE:
            return _transparent_stub()
        for _block_id, base_name in OBJ_INVISIBLE_MATERIAL_BY_BLOCK_ID.items():
            if material_name == base_name:
                return _transparent_stub()
            prefix = base_name + "_"
            if material_name.startswith(prefix):
                role = material_name[len(prefix):]
                if role in INVISIBLE_HIDDEN_ROLES:
                    return _transparent_stub()
                break
        if (
            material_name.startswith("stc_invisible")
            and "_packed_" not in material_name
            and "_packed_body" not in material_name
        ):
            return _transparent_stub()
    palette = GLB_FALLBACK_PALETTE.get(material_name)
    if palette is not None:
        color, alpha, alpha_mode, emissive = palette
        return builder.fallback_material(material_name, color, alpha=alpha, alpha_mode=alpha_mode, emissive=emissive)

    for block_id, base_name in OBJ_INVISIBLE_MATERIAL_BY_BLOCK_ID.items():
        if material_name == base_name:
            color, emissive = GLB_INVISIBLE_BLOCK_BASE.get(block_id, ((1.0, 0.1, 0.1), None))
            return builder.fallback_material(base_name, color, alpha=0.35, alpha_mode="BLEND", emissive=emissive)
        prefix = base_name + "_"
        if material_name.startswith(prefix):
            role = material_name[len(prefix):]
            base_color, _alpha = GLB_INVISIBLE_ROLE_PALETTE.get(role, ((1.0, 0.1, 0.1), 0.35))
            block_color, emissive = GLB_INVISIBLE_BLOCK_BASE.get(block_id, ((1.0, 0.1, 0.1), None))
            color = (
                (base_color[0] + block_color[0]) * 0.5,
                (base_color[1] + block_color[1]) * 0.5,
                (base_color[2] + block_color[2]) * 0.5,
            )
            alpha = _alpha
            alpha_mode = "OPAQUE" if alpha >= 0.999 else "BLEND"
            return builder.fallback_material(material_name, color, alpha=alpha, alpha_mode=alpha_mode, emissive=emissive)

    if material_name.startswith("stc_missing_block"):
        return builder.fallback_material(material_name, (1.0, 0.1, 0.1))
    return builder.fallback_material(material_name, (0.72, 0.72, 0.72))


def _tint_to_rgba_u8(tint: Optional[Tuple[float, float, float]], alpha: float = 1.0) -> Tuple[int, int, int, int]:
    if tint is None:
        return (255, 255, 255, max(0, min(255, round(alpha * 255.0))))
    return (
        max(0, min(255, round(tint[0] * 255.0))),
        max(0, min(255, round(tint[1] * 255.0))),
        max(0, min(255, round(tint[2] * 255.0))),
        max(0, min(255, round(alpha * 255.0))),
    )


def _add_buffer_chunk(buffer: bytearray, data: bytes, alignment: int = 4) -> Tuple[int, int]:
    pad = (-len(buffer)) & (alignment - 1)
    if pad:
        buffer += b"\x00" * pad
    offset = len(buffer)
    buffer += data
    return offset, len(data)


def _serialize_glb(builder: "GltfBuilder", output_path: Path) -> Tuple[int, int, int, int, int]:
    bin_buffer = bytearray()
    buffer_views: List[Dict[str, Any]] = []
    accessors: List[Dict[str, Any]] = []
    primitives_meta: List[Dict[str, Any]] = []
    total_faces = 0
    total_vertices = 0

    sorted_materials = sorted(builder.primitives.keys())
    for material_index in sorted_materials:
        prim = builder.primitives[material_index]
        if prim.vertex_count == 0:
            continue
        total_vertices += prim.vertex_count
        total_faces += prim.index_count // 3

        offset, size = _add_buffer_chunk(bin_buffer, bytes(prim.position), 4)
        bv_pos = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": size, "target": 34962})
        acc_pos = len(accessors)
        accessors.append({
            "bufferView": bv_pos,
            "byteOffset": 0,
            "componentType": 5126,
            "count": prim.vertex_count,
            "type": "VEC3",
            "min": prim.pmin,
            "max": prim.pmax,
        })

        offset, size = _add_buffer_chunk(bin_buffer, bytes(prim.uv), 4)
        bv_uv = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": size, "target": 34962})
        acc_uv = len(accessors)
        accessors.append({
            "bufferView": bv_uv,
            "byteOffset": 0,
            "componentType": 5126,
            "count": prim.vertex_count,
            "type": "VEC2",
        })

        offset, size = _add_buffer_chunk(bin_buffer, bytes(prim.color), 4)
        bv_col = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": size, "target": 34962})
        acc_col = len(accessors)
        accessors.append({
            "bufferView": bv_col,
            "byteOffset": 0,
            "componentType": 5121,
            "count": prim.vertex_count,
            "type": "VEC4",
            "normalized": True,
        })

        offset, size = _add_buffer_chunk(bin_buffer, bytes(prim.indices), 4)
        bv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": size, "target": 34963})
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": 5125,
            "count": prim.index_count,
            "type": "SCALAR",
        })

        primitives_meta.append({
            "attributes": {
                "POSITION": acc_pos,
                "TEXCOORD_0": acc_uv,
                "COLOR_0": acc_col,
            },
            "indices": acc_idx,
            "material": material_index,
            "mode": 4,
        })

    images_json: List[Dict[str, Any]] = []
    textures_json: List[Dict[str, Any]] = []
    samplers_json = [{"magFilter": 9728, "minFilter": 9728, "wrapS": 10497, "wrapT": 10497}]
    for image_index, (name, png_bytes) in enumerate(builder.images):
        offset, size = _add_buffer_chunk(bin_buffer, png_bytes, 4)
        bv_img = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": size})
        images_json.append({"bufferView": bv_img, "mimeType": "image/png", "name": name})
        textures_json.append({"sampler": 0, "source": image_index})

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "tools/mdat_obj_export.py glTF writer"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "world"}],
        "meshes": [{"primitives": primitives_meta, "name": "world"}],
        "materials": builder.materials,
        "samplers": samplers_json,
        "buffers": [{"byteLength": len(bin_buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    if images_json:
        gltf["images"] = images_json
    if textures_json:
        gltf["textures"] = textures_json

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) & 3
    json_chunk = json_bytes + b" " * json_pad
    bin_pad = (-len(bin_buffer)) & 3
    bin_chunk = bytes(bin_buffer) + b"\x00" * bin_pad

    total_size = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = bytearray()
    glb += struct.pack("<III", 0x46546C67, 2, total_size)
    glb += struct.pack("<II", len(json_chunk), 0x4E4F534A)
    glb += json_chunk
    glb += struct.pack("<II", len(bin_chunk), 0x004E4942)
    glb += bin_chunk

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(glb))

    return total_vertices, total_faces, len(builder.materials), len(builder.images), total_size


def write_glb(
    output_path: Path,
    world_blocks: Dict[Tuple[int, int, int], BlockState],
    resolver: GeometryResolver,
    center: bool,
    cull: bool,
    group_by_cover: bool = False,
    texture_resolver: Optional[TextureResolver] = None,
    force_double_sided: bool = False,
    force_mask: bool = False,
    hide_invisible: bool = False,
) -> Tuple[int, int, int, int, int]:
    """Write a GLB file representing the resolved world geometry.

    Mirrors the OBJ pipeline that lived in `write_obj`, but routes face material
    selection through `TextureResolver.face_material_descriptor`. Per-side game
    tints (RGB565, biome grass tint, ProBuilder overlay tint) are stored as
    per-vertex `COLOR_0` and multiplied with `baseColorTexture` per the glTF
    2.0 spec, instead of baking tinted PNG variants.
    """

    offset = compute_center_offset(world_blocks.keys()) if center else (0.0, 0.0, 0.0)
    builder = GltfBuilder()

    def fallback_material_index(name: str) -> int:
        return _resolve_fallback_material(builder, name, hide_invisible=hide_invisible)

    def descriptor_material_index(descriptor: FaceMaterialDescriptor, *, double_sided: bool) -> int:
        # ``force_double_sided`` overrides the per-descriptor choice so that
        # glTF viewers do not backface-cull faces whose winding the world
        # mesher emitted as CW. OBJ tools tolerate mixed winding because they
        # default to two-sided rendering; glTF strictly enforces CCW front
        # faces unless the material opts in via ``doubleSided: true``.
        if force_double_sided:
            double_sided = True
        return builder.material_for_descriptor(descriptor, double_sided=double_sided, force_mask=force_mask, hide_invisible=hide_invisible)

    def emit_world_face(
        material_index: int,
        world_face: Face,
        local_face: Optional[Face],
        uv_side: Optional[str],
        uv_rotation: int,
        explicit_uvs: Optional[Tuple[UV, ...]],
        tint: Optional[Tuple[float, float, float]],
        alpha: float = 1.0,
        outward_reference: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        # SKIP sentinel: drops the face entirely. Used by the inv_mask path
        # in `material_for_descriptor` when `hide_invisible` is on.
        if material_index == SKIP_MATERIAL_INDEX:
            return
        if explicit_uvs is not None and len(explicit_uvs) == len(world_face):
            uvs = list(explicit_uvs)
        else:
            uv_source = local_face if local_face is not None else world_face
            side = uv_side or cover_side_for_face(uv_source) or "UP"
            uvs = rotate_uvs(uv_for_face(uv_source, side), uv_rotation)
        side_axis = SIDE_AXIS.get((uv_side or "").upper())
        should_flip = (
            face_points_against_axis(world_face, side_axis)
            if side_axis is not None
            else outward_reference is not None and face_points_inward(world_face, outward_reference)
        )
        if should_flip:
            world_face = tuple(reversed(world_face))
            uvs = list(reversed(uvs))
        prim = builder.primitive_for_material(material_index)
        prim.add_polygon(world_face, uvs, _tint_to_rgba_u8(tint, alpha))

    def texture_state_for_surface_face(
        pos: Tuple[int, int, int],
        state: BlockState,
        local_face: Face,
        side: str,
    ) -> Tuple[Optional[BlockState], Optional[str]]:
        diagonal_grass_state = resolver.diagonal_grass_neighbor_state_for_surface(pos, state, local_face)
        if diagonal_grass_state is not None:
            return diagonal_grass_state, side
        return (
            resolver.texture_block_state_for_surface(pos, state, side),
            resolver.texture_material_side_for_surface(pos, state, side),
        )

    def textured_surface_descriptor(
        pos: Tuple[int, int, int],
        state: BlockState,
        local_face: Face,
        fallback_name: str,
        side_hint: Optional[str] = None,
    ) -> Tuple[int, str, int, Optional[str], Optional[Tuple[float, float, float]]]:
        side = (
            side_hint
            or resolver.wedge_xz_diagonal_render_side_for_surface(pos, state, local_face)
            or cover_side_for_face(local_face)
            or "UP"
        )
        tile = resolver.world_tiles.get(pos)
        if texture_resolver is None:
            return fallback_material_index(fallback_name), side, 0, side, None
        texture_state, material_side = texture_state_for_surface_face(pos, state, local_face, side)
        if texture_state is None:
            return fallback_material_index(fallback_name), side, 0, material_side, None
        explicit_tint = GeometryResolver.tile_tint(tile, material_side or side)
        descriptor = texture_resolver.face_material_descriptor(
            texture_state.block_id,
            texture_state.meta,
            pos,
            side,
            tile,
            explicit_tint=explicit_tint,
        )
        material_index = descriptor_material_index(descriptor, double_sided=descriptor.alpha_mode != "OPAQUE")
        return material_index, side, GeometryResolver.tile_uv_rotation(tile, side), material_side, descriptor.tint

    def cover_descriptor(
        pos: Tuple[int, int, int],
        tile: Optional[ProBuilderTile],
        cover_side: str,
        texture_side: str,
        cover_state: BlockState,
    ) -> Tuple[int, str, int, Optional[Tuple[float, float, float]]]:
        if texture_resolver is None:
            return fallback_material_index(OBJ_MATERIAL_COVER), texture_side, 0, None
        explicit_tint = GeometryResolver.tile_tint(tile, cover_side)
        descriptor = texture_resolver.face_material_descriptor(
            cover_state.block_id,
            cover_state.meta,
            pos,
            texture_side,
            tile,
            explicit_tint=explicit_tint,
        )
        material_index = descriptor_material_index(descriptor, double_sided=descriptor.alpha_mode != "OPAQUE")
        return material_index, texture_side, GeometryResolver.tile_uv_rotation(tile, texture_side), descriptor.tint

    def overlay_tint_for_icon(
        pos: Tuple[int, int, int],
        tile: Optional[ProBuilderTile],
        side: str,
        icon: str,
    ) -> Optional[Tuple[float, float, float]]:
        if texture_resolver is None:
            return None
        tint = GeometryResolver.tile_tint(tile, side)
        if tint is not None:
            return tint
        if icon.endswith(("grass_top", "overlay_fast_grass_side", "grass_side_overlay")):
            return texture_resolver.biome_grass_tint(pos)
        return None

    def overlay_descriptor(
        pos: Tuple[int, int, int],
        tile: Optional[ProBuilderTile],
        side: str,
        material_side: Optional[str] = None,
    ) -> Optional[Tuple[int, str, int, Optional[Tuple[float, float, float]]]]:
        if texture_resolver is None:
            return None
        overlay_value = GeometryResolver.tile_overlay_value(tile, material_side or side)
        icon = texture_resolver.probuilder_overlay_icon_name(overlay_value, side)
        if icon is None:
            return None
        tint = overlay_tint_for_icon(pos, tile, material_side or side, icon)
        descriptor = texture_resolver.overlay_material_descriptor(icon, tint)
        material_index = descriptor_material_index(descriptor, double_sided=True)
        return material_index, side, GeometryResolver.tile_uv_rotation(tile, side), descriptor.tint

    def block_overlay_descriptor(
        pos: Tuple[int, int, int],
        texture_state: Optional[BlockState],
        side: str,
        tile: Optional[ProBuilderTile],
        tint_side: str,
    ) -> Optional[Tuple[int, str, int, Optional[Tuple[float, float, float]]]]:
        if texture_resolver is None or texture_state is None:
            return None
        icon = texture_resolver.block_overlay_icon_name(texture_state.block_id, texture_state.meta, side, pos)
        if icon is None:
            return None
        tint = GeometryResolver.tile_tint(tile, tint_side) or texture_resolver.biome_tint_for_block(
            texture_state.block_id,
            texture_state.meta,
            pos,
            side,
            icon,
        )
        descriptor = texture_resolver.overlay_material_descriptor(icon, tint)
        material_index = descriptor_material_index(descriptor, double_sided=True)
        return material_index, side, GeometryResolver.tile_uv_rotation(tile, side), descriptor.tint

    for pos in sorted(world_blocks):
        state = world_blocks[pos]
        wx, wy, wz = pos
        block_center = (wx + 0.5 - offset[0], wy + 0.5 - offset[1], wz + 0.5 - offset[2])
        base_material_name = resolver.material_for_block_surface(pos, state)
        tile = resolver.world_tiles.get(pos)

        external_model_entries = resolver.external_model_face_entries_for(pos, state)
        if external_model_entries:
            if texture_resolver is not None:
                config = resolver.external_config_for_block(state.block_id) or {}
                explicit_tint = texture_resolver.biome_grass_tint(pos) if config.get("color_multiplier") else None
                descriptor = texture_resolver.face_material_descriptor(
                    state.block_id,
                    state.meta,
                    pos,
                    "UP",
                    tile,
                    explicit_tint=explicit_tint,
                )
                model_material_index = descriptor_material_index(descriptor, double_sided=True)
                model_tint = descriptor.tint
            else:
                model_material_index = fallback_material_index(base_material_name)
                model_tint = None
            for model_face in external_model_entries:
                world_face = tuple(
                    (wx + lx - offset[0], wy + ly - offset[1], wz + lz - offset[2])
                    for lx, ly, lz in model_face.vertices
                )
                emit_world_face(
                    model_material_index,
                    world_face,
                    model_face.vertices,
                    None,
                    0,
                    model_face.uvs,
                    model_tint,
                )
            continue

        local_faces = resolver.local_faces_for(pos, state, include_covers=False)
        for face in local_faces:
            world_face = tuple(
                (wx + lx - offset[0], wy + ly - offset[1], wz + lz - offset[2]) for lx, ly, lz in face
            )
            material_index, uv_side, uv_rotation, material_side, surface_tint = textured_surface_descriptor(
                pos, state, face, base_material_name
            )
            emit_world_face(
                material_index,
                world_face,
                face,
                uv_side,
                uv_rotation,
                None,
                surface_tint,
                outward_reference=block_center,
            )
            texture_state = (
                texture_state_for_surface_face(pos, state, face, uv_side)[0]
                if texture_resolver is not None
                else None
            )
            block_overlay = block_overlay_descriptor(pos, texture_state, uv_side, tile, material_side or uv_side)
            if block_overlay is not None:
                overlay_face = offset_face_along_side(world_face, uv_side)
                ovl_idx, ovl_side, ovl_rot, ovl_tint = block_overlay
                emit_world_face(
                    ovl_idx,
                    overlay_face,
                    face,
                    ovl_side,
                    ovl_rot,
                    None,
                    ovl_tint,
                    outward_reference=block_center,
                )
            overlay = overlay_descriptor(pos, tile, uv_side, material_side)
            if overlay is not None:
                overlay_face = offset_face_along_side(world_face, uv_side)
                ovl_idx, ovl_side, ovl_rot, ovl_tint = overlay
                emit_world_face(
                    ovl_idx,
                    overlay_face,
                    face,
                    ovl_side,
                    ovl_rot,
                    None,
                    ovl_tint,
                    outward_reference=block_center,
                )

        for face, cover_side, cover_state in resolver.cover_face_entries_for(pos, state):
            world_face = tuple(
                (wx + lx - offset[0], wy + ly - offset[1], wz + lz - offset[2]) for lx, ly, lz in face
            )
            texture_side = cover_side_for_face(face) or cover_side
            material_index, uv_side, uv_rotation, cover_tint = cover_descriptor(
                pos, tile, cover_side, texture_side, cover_state
            )
            emit_world_face(
                material_index,
                world_face,
                face,
                uv_side,
                uv_rotation,
                None,
                cover_tint,
                outward_reference=block_center,
            )
            block_overlay = block_overlay_descriptor(pos, cover_state, uv_side, tile, cover_side)
            if block_overlay is not None:
                overlay_face = offset_face_along_side(world_face, uv_side)
                ovl_idx, ovl_side, ovl_rot, ovl_tint = block_overlay
                emit_world_face(
                    ovl_idx,
                    overlay_face,
                    face,
                    ovl_side,
                    ovl_rot,
                    None,
                    ovl_tint,
                    outward_reference=block_center,
                )
            if cover_side_for_face(face) == cover_side:
                overlay = overlay_descriptor(pos, tile, cover_side)
                if overlay is not None:
                    overlay_face = offset_face_along_side(world_face, cover_side)
                    ovl_idx, ovl_side, ovl_rot, ovl_tint = overlay
                    emit_world_face(
                        ovl_idx,
                        overlay_face,
                        face,
                        ovl_side,
                        ovl_rot,
                        None,
                        ovl_tint,
                        outward_reference=block_center,
                    )

        for box in resolver.boxes_for(pos, state):
            x0, y0, z0, x1, y1, z1 = box
            bx0 = wx + x0 - offset[0]
            by0 = wy + y0 - offset[1]
            bz0 = wz + z0 - offset[2]
            bx1 = wx + x1 - offset[0]
            by1 = wy + y1 - offset[1]
            bz1 = wz + z1 - offset[2]
            box_center = ((bx0 + bx1) * 0.5, (by0 + by1) * 0.5, (bz0 + bz1) * 0.5)
            for face_name, (dx, dy, dz), make_face in FACE_DEFS:
                if cull and face_on_block_boundary(face_name, box):
                    neighbor_pos = (wx + dx, wy + dy, wz + dz)
                    if neighbor_pos in world_blocks and resolver.is_full_cell(neighbor_pos, include_invisible=False):
                        continue
                local_verts = tuple(make_face(x0, y0, z0, x1, y1, z1))
                verts = tuple(make_face(bx0, by0, bz0, bx1, by1, bz1))
                fallback_name = resolver.material_for_box_face(pos, state, face_name)
                material_index, uv_side, uv_rotation, material_side, surface_tint = textured_surface_descriptor(
                    pos, state, local_verts, fallback_name, face_name.upper()
                )
                emit_world_face(
                    material_index,
                    verts,
                    local_verts,
                    uv_side,
                    uv_rotation,
                    None,
                    surface_tint,
                    outward_reference=box_center,
                )
                texture_state = (
                    texture_state_for_surface_face(pos, state, local_verts, uv_side)[0]
                    if texture_resolver is not None
                    else None
                )
                block_overlay = block_overlay_descriptor(pos, texture_state, uv_side, tile, material_side or uv_side)
                if block_overlay is not None:
                    overlay_face = offset_face_along_side(verts, uv_side)
                    ovl_idx, ovl_side, ovl_rot, ovl_tint = block_overlay
                    emit_world_face(
                        ovl_idx,
                        overlay_face,
                        local_verts,
                        ovl_side,
                        ovl_rot,
                        None,
                        ovl_tint,
                        outward_reference=box_center,
                    )
                overlay = overlay_descriptor(pos, tile, uv_side, material_side)
                if overlay is not None:
                    overlay_face = offset_face_along_side(verts, uv_side)
                    ovl_idx, ovl_side, ovl_rot, ovl_tint = overlay
                    emit_world_face(
                        ovl_idx,
                        overlay_face,
                        local_verts,
                        ovl_side,
                        ovl_rot,
                        None,
                        ovl_tint,
                        outward_reference=box_center,
                    )

    # Phantom-liquid emission: replays `liquid_render_faces` for every cell in
    # `LIQUID_PHANTOM_POSITIONS` using a real liquid block as the material
    # prototype. The phantom set was built by dilating each real liquid cell
    # horizontally, so this overlays water geometry on top of whichever
    # terrain/ProBuilder block already sits at the phantom position. The
    # original block's faces are untouched -- viewers see the water surface
    # tucked inside the opaque neighbour, hiding the side-wall strip that
    # used to be visible at the lake edge.
    if LIQUID_PHANTOM_POSITIONS:
        prototype_pos: Optional[Tuple[int, int, int]] = None
        prototype_state: Optional[BlockState] = None
        for liquid_pos, liquid_state in world_blocks.items():
            if is_liquid_type(resolver.class_for(liquid_state.block_id)):
                prototype_pos = liquid_pos
                prototype_state = liquid_state
                break
        if prototype_state is not None:
            phantom_state = BlockState(prototype_state.block_id, 0)
            for phantom_pos in sorted(LIQUID_PHANTOM_POSITIONS):
                if phantom_pos in world_blocks and is_liquid_type(
                    resolver.class_for(world_blocks[phantom_pos].block_id)
                ):
                    continue
                wx, wy, wz = phantom_pos
                faces = liquid_render_faces(world_blocks, phantom_pos, resolver.class_for)
                if not faces:
                    continue
                if texture_resolver is not None:
                    descriptor = texture_resolver.face_material_descriptor(
                        phantom_state.block_id,
                        phantom_state.meta,
                        phantom_pos,
                        "UP",
                        None,
                    )
                    material_index = descriptor_material_index(descriptor, double_sided=True)
                    tint = descriptor.tint
                else:
                    material_index = fallback_material_index(
                        resolver.material_for_block_surface(prototype_pos or phantom_pos, prototype_state)
                    )
                    tint = None
                phantom_center = (wx + 0.5 - offset[0], wy + 0.5 - offset[1], wz + 0.5 - offset[2])
                for face in faces:
                    world_face = tuple(
                        (wx + lx - offset[0], wy + ly - offset[1], wz + lz - offset[2])
                        for lx, ly, lz in face
                    )
                    emit_world_face(
                        material_index,
                        world_face,
                        face,
                        None,
                        0,
                        None,
                        tint,
                        outward_reference=phantom_center,
                    )

    return _serialize_glb(builder, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export .mdat terrain geometry to glTF (.glb).")
    parser.add_argument("input", type=Path, help="Region .mdat file or directory.")
    parser.add_argument("output", type=Path, help="Output .glb path.")
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_BLOCK_REGISTRY)
    parser.add_argument("--external-registry", type=Path, default=DEFAULT_EXTERNAL_REGISTRY)
    parser.add_argument("--geometry-report", type=Path, default=DEFAULT_GEOMETRY_REPORT)
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunks for tests/small exports.")
    parser.add_argument("--max-blocks", type=int, default=None, help="Stop after exporting this many selected blocks.")
    parser.add_argument(
        "--double-sided",
        action="store_true",
        help="Force all glTF materials to doubleSided=true.",
    )
    parser.add_argument(
        "--force-mask",
        action="store_true",
        help="Promote glTF alphaMode BLEND -> MASK (cutoff 0.5).",
    )
    parser.add_argument(
        "--hide-invisible",
        action="store_true",
        help="Make technical invisible-material faces transparent while preserving geometry.",
    )
    parser.add_argument(
        "--liquid-type-id",
        dest="liquid_type_id",
        type=int,
        action="append",
        default=[],
        help="Extra type id(s) to treat as liquid (water/lava).",
    )
    parser.add_argument(
        "--no-auto-liquid",
        action="store_true",
        help="Disable auto-detection of liquid types by icon name.",
    )
    parser.add_argument(
        "--water-expand",
        type=int,
        default=1,
        help="Horizontally dilate every liquid cell by N blocks of phantom water (default 1). Phantom water overlays whichever terrain/ProBuilder block sits at the neighbour cell so the visible water-side-wall strip at the lake edge tucks inside the opaque neighbour. Pass 0 to disable.",
    )
    parser.add_argument("--id-min", type=int, default=None, help="Inclusive minimum block id to export.")
    parser.add_argument("--id-max", type=int, default=None, help="Inclusive maximum block id to export.")
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=6,
        metavar=("X1", "Y1", "Z1", "X2", "Y2", "Z2"),
        help="Inclusive world-space box to export. Coordinate order can be reversed.",
    )
    parser.add_argument(
        "--context-padding",
        type=int,
        default=1,
        help=(
            "Extra block radius loaded around --bbox for neighbor-dependent render "
            "geometry/textures/CTM. Geometry is still emitted only for --bbox."
        ),
    )
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated block ids and ranges, e.g. 2401,2405,2480-2499. Combined with id-min/id-max.",
    )
    parser.add_argument("--group-by-cover", action="store_true", help="For ProBuilder tiles, include cover/data in OBJ groups.")
    parser.add_argument(
        "--model-placeholder",
        choices=("cross", "cube", "none"),
        default="cross",
        help="How to export external simple_render model blocks when .mcsa meshes are unavailable.",
    )
    parser.add_argument("--map-blocks", type=Path, default=DEFAULT_MAP_BLOCKS, help="map_blocks.json with id/meta -> iconName.")
    parser.add_argument("--weather-palettes", type=Path, default=DEFAULT_WEATHER_PALETTES, help="Weather palettes.json with biome tint colors.")
    parser.add_argument("--texarr", type=Path, default=DEFAULT_BLOCK_TEXARR, help="blockMap.texarr texture array.")
    parser.add_argument("--ctm-dir", type=Path, default=DEFAULT_CTM_DIR, help="Directory with stalcraft ctmpatcher .properties.")
    parser.add_argument("--no-ctm", action="store_true", help="Disable CTM repeat rule resolution.")
    parser.add_argument(
        "--empty-id",
        type=int,
        action="append",
        default=[0],
        help="Block id to treat as empty. Can be repeated. Default: 0.",
    )
    args = parser.parse_args()
    args.input_kind = "mdat"
    args.center = True
    args.no_cull = False
    args.mesh_mode = "render"
    args.texture_mode = "game"
    args.texture_format = "png"
    args.source_tables_dir = Path("__standalone_disabled__")
    return args


def parse_id_selector(args: argparse.Namespace) -> Optional[Set[int]]:
    ids: Set[int] = set()
    if args.id_min is not None or args.id_max is not None:
        lo = args.id_min if args.id_min is not None else 0
        hi = args.id_max if args.id_max is not None else lo
        if hi < lo:
            raise SystemExit("--id-max must be >= --id-min")
        ids.update(range(lo, hi + 1))
    for part in str(args.ids or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise SystemExit(f"Invalid id range: {part}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(part))
    return ids or None


def main() -> None:
    args = parse_args()
    include_ids = parse_id_selector(args)
    bbox = normalize_bbox(args.bbox) if args.bbox is not None else None
    context_padding = max(0, args.context_padding)
    context_bbox = expand_bbox(bbox, context_padding)
    context_loads_extra = context_bbox != bbox
    input_kind = args.input_kind
    if input_kind == "auto":
        input_kind = "mdat"

    block_registry = merge_block_registry_json(load_block_registry(args.source_tables_dir), args.registry_json)
    type_textures = load_type_texture_registry(args.source_tables_dir, block_registry)
    # Extend the liquid type registry from icon-name auto-detection and manual
    # overrides so custom water/lava blocks use the liquid surface path.
    if not args.no_auto_liquid:
        auto_liquid = auto_detect_liquid_type_ids(type_textures)
        new_auto = auto_liquid - LIQUID_TYPE_IDS
        if new_auto:
            print(f"[liquid] auto-detected liquid type ids by icon: {sorted(new_auto)}")
            LIQUID_TYPE_IDS.update(new_auto)
    if args.liquid_type_id:
        new_cli = set(args.liquid_type_id) - LIQUID_TYPE_IDS
        if new_cli:
            print(f"[liquid] CLI-added liquid type ids: {sorted(new_cli)}")
            LIQUID_TYPE_IDS.update(new_cli)
    external_configs = load_external_block_configs(args.external_registry)
    biome_color_table = load_biome_color_table(args.weather_palettes)
    geometry_report = load_geometry_report(args.geometry_report)
    shape_tables = load_shape_tables(args.source_tables_dir, block_registry)
    snow_block_ids = load_snow_like_block_ids(args.source_tables_dir, block_registry)

    region_files: Sequence[Path] = ()
    region_files = iter_region_files(args.input)
    if not region_files:
        raise SystemExit(f"No region files found: {args.input}")
    world_blocks, world_tiles, world_biomes, chunks_loaded = load_world(
        region_files=region_files,
        max_chunks=args.max_chunks,
        empty_ids=set(args.empty_id),
        include_ids=None if context_loads_extra else include_ids,
        max_blocks=None if context_loads_extra else args.max_blocks,
        bbox=context_bbox,
        collect_biomes=True,
    )

    export_blocks = {
        pos: state
        for pos, state in world_blocks.items()
        if pos_in_bbox(pos, bbox) and (include_ids is None or state.block_id in include_ids)
    }
    if args.max_blocks is not None and context_loads_extra:
        export_blocks = dict(sorted(export_blocks.items())[: args.max_blocks])

    resolver = GeometryResolver(
        block_type_by_id=block_registry,
        shape_tables=shape_tables,
        world_tiles=world_tiles,
        world_blocks=world_blocks,
        external_configs=external_configs,
        geometry_report=geometry_report,
        model_placeholder=args.model_placeholder,
        mesh_mode=args.mesh_mode,
    )
    texture_resolver = None
    if args.texture_mode == "game":
        texture_resolver = TextureResolver(
            map_blocks_path=args.map_blocks,
            texarr_path=args.texarr,
            ctm_dir=args.ctm_dir,
            output_path=args.output,
            external_configs=external_configs,
            type_textures=type_textures,
            block_type_by_id=block_registry,
            world_blocks=world_blocks,
            world_biomes=world_biomes,
            biome_color_table=biome_color_table,
            texture_format=args.texture_format,
            enable_ctm=not args.no_ctm,
            snow_block_ids=snow_block_ids,
        )
    # Phantom-liquid dilation needs the final export block set (after bbox/id
    # filters) plus the resolver's class lookup. Populate the module-level
    # set now so the helpers inside ``write_glb`` and the per-cell liquid
    # height functions see the dilated water footprint.
    LIQUID_PHANTOM_POSITIONS.clear()
    if args.water_expand > 0:
        phantoms = compute_liquid_phantom_positions(export_blocks, resolver.class_for, args.water_expand)
        if phantoms:
            print(f"[liquid] phantom water cells: {len(phantoms)} (radius={args.water_expand})")
            LIQUID_PHANTOM_POSITIONS.update(phantoms)
    vertices, faces, glb_materials, glb_images, glb_size = write_glb(
        output_path=args.output,
        world_blocks=export_blocks,
        resolver=resolver,
        center=args.center,
        cull=not args.no_cull,
        group_by_cover=args.group_by_cover,
        texture_resolver=texture_resolver,
        force_double_sided=args.double_sided,
        force_mask=args.force_mask,
        hide_invisible=args.hide_invisible,
    )

    strategy_counts: Dict[str, int] = {}
    block_id_counts: Dict[int, int] = {}
    for state in export_blocks.values():
        strategy = resolver.strategy_for(state.block_id)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        block_id_counts[state.block_id] = block_id_counts.get(state.block_id, 0) + 1

    print(f"regions={len(region_files)} chunks={chunks_loaded} blocks={len(export_blocks)}")
    if len(world_blocks) != len(export_blocks):
        print(f"context_blocks={len(world_blocks)} context_padding={context_padding}")
    print(
        f"block_types={len(block_registry)} external_configs={len(external_configs)} "
        f"shape_tables={len(shape_tables)} tiles={len(world_tiles)}"
    )
    print(f"selected_ids={len(include_ids) if include_ids is not None else 'all'}")
    if texture_resolver is not None:
        print(
            "textures="
            + json.dumps(
                {
                    "materials": len(texture_resolver.materials),
                    "extracted": len(texture_resolver.extracted_icons),
                    "tinted": len(texture_resolver.tinted_icons),
                    "texarr_entries": len(texture_resolver.texarr.entries),
                    "ctm_blocks": len(texture_resolver.ctm_rules_by_block),
                    "type_textures": len(texture_resolver.type_textures),
                    "biome_columns": len(texture_resolver.world_biomes),
                    "biome_colors": len(texture_resolver.biome_color_table),
                    "missing": sorted(texture_resolver.missing_icons)[:20],
                    "dds_fallbacks": sorted(texture_resolver.convert_fallbacks)[:20],
                },
                ensure_ascii=False,
            )
        )
    print("strategies=" + json.dumps(dict(sorted(strategy_counts.items())), ensure_ascii=False))
    print("top_ids=" + json.dumps(dict(sorted(block_id_counts.items(), key=lambda item: item[1], reverse=True)[:20])))
    print(
        f"glb={args.output} vertices={vertices} faces={faces} "
        f"materials={glb_materials} images={glb_images} bytes={glb_size}"
    )
    sys.stdout.flush()

if __name__ == "__main__":
    main()
