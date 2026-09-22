from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .mdat_scan import write_filtered_scan_chunks
from .paths import EXPORTS_DIR, MDAT_EXPORT_SCRIPT, ROOT, TMP_DIR, find_blender, game_texture_paths


@dataclass
class ExportJob:
    job_id: str
    name: str
    game_dir: str
    map_cache: str
    scan: dict[str, Any]
    selected: list[tuple[int, int]]
    out_dir: Path
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    glb: Path | None = None
    blend: Path | None = None

    @property
    def running(self) -> bool:
        return self.finished_at is None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "running": self.running,
            "success": self.finished_at is not None and self.error is None,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "chunks": len(self.selected),
            "glb": str(self.glb) if self.glb else None,
            "blend": str(self.blend) if self.blend else None,
            "log": self.log[-80:],
        }


class ExportRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ExportJob] = {}

    def list(self) -> list[ExportJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.started_at, reverse=True)
        return jobs

    def submit(self, *, game_dir: Path, map_cache: Path, scan: dict[str, Any], chunks: list[tuple[int, int]], options: dict[str, Any]) -> ExportJob:
        if not chunks:
            raise ValueError("select at least one chunk")
        if not MDAT_EXPORT_SCRIPT.is_file():
            raise FileNotFoundError(str(MDAT_EXPORT_SCRIPT))
        name = safe_name(str(options.get("name") or f"mdat_{int(time.time())}"))
        job_id = uuid.uuid4().hex[:12]
        out_dir = EXPORTS_DIR / f"{name}_{job_id}"
        job = ExportJob(
            job_id=job_id,
            name=name,
            game_dir=str(game_dir),
            map_cache=str(map_cache),
            scan=scan,
            selected=chunks,
            out_dir=out_dir,
        )
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job, options), daemon=True)
        thread.start()
        return job

    def _run(self, job: ExportJob, options: dict[str, Any]) -> None:
        try:
            run_export(job, options)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(job.error)
        finally:
            job.finished_at = time.time()
            job.out_dir.mkdir(parents=True, exist_ok=True)
            (job.out_dir / "job.json").write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in value).strip(" .")
    return cleaned or "mdat_export"


def run_export(job: ExportJob, options: dict[str, Any]) -> None:
    game_dir = Path(job.game_dir)
    map_cache = Path(job.map_cache)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_cache = TMP_DIR / f"{job.job_id}_map_cache"
    written = write_filtered_scan_chunks(job.scan, tmp_cache, job.selected)
    if written <= 0:
        raise RuntimeError("selected chunks were not found in map_cache")
    job.log.append(f"filtered_map_cache={tmp_cache} chunks={written}")

    glb = job.out_dir / f"{job.name}.glb"
    blend = job.out_dir / f"{job.name}.blend"
    paths = game_texture_paths(game_dir)
    cmd = [
        sys.executable,
        str(MDAT_EXPORT_SCRIPT),
        str(tmp_cache),
        str(glb),
        "--double-sided",
        "--force-mask",
        "--hide-invisible",
        "--texarr", str(paths["texarr"]),
        "--ctm-dir", str(paths["ctm_dir"]),
        "--map-blocks", str(paths["map_blocks"]),
        "--weather-palettes", str(paths["weather_palettes"]),
    ]
    job.log.append(" ".join(quote_part(part) for part in cmd))
    run_logged(cmd, cwd=MDAT_EXPORT_SCRIPT.parent, log=job.log)
    if not glb.is_file():
        raise RuntimeError(f"GLB was not written: {glb}")
    job.glb = glb

    blender = find_blender(str(options.get("blender") or "").strip() or None)
    if not blender.is_file():
        raise FileNotFoundError(f"Blender not found: {blender}")
    blender_script = ROOT / "modules" / "glb_to_blend.py"
    blender_cmd = [
        str(blender),
        "--background",
        "--python", str(blender_script),
        "--",
        "--input", str(glb),
        "--output", str(blend),
    ]
    job.log.append(" ".join(quote_part(part) for part in blender_cmd))
    run_logged(blender_cmd, cwd=job.out_dir, log=job.log)
    if not blend.is_file():
        raise RuntimeError(f"Blend was not written: {blend}")
    job.blend = blend
    job.log.append(f"blend={blend}")


def run_logged(cmd: list[str], *, cwd: Path, log: list[str]) -> None:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.append(line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}")


def quote_part(value: object) -> str:
    text = str(value)
    if " " in text or "\t" in text:
        return f'"{text}"'
    return text
