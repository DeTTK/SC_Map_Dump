from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from modules.exporter import ExportRunner
from modules.mdat_scan import list_worlds, save_scan, scan_map_cache
from modules.olmaps import TileCache, get_olmap, list_olmaps
from modules.paths import DATA_DIR, UI_DIR, ensure_dirs, find_map_cache, game_texture_paths


CONFIG_PATH = DATA_DIR / "config.json"


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    try:
        return json.loads(handler.rfile.read(length).decode("utf-8"))
    except Exception:
        return {}


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "MapDumpMDAT/1.0"
    exporter: ExportRunner = ExportRunner()
    tile_cache: TileCache = TileCache()
    last_scan: dict[str, Any] | None = None

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.serve_static("index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            self.serve_static(path[len("/static/"):])
            return
        if path == "/config":
            config = load_config()
            payload = {"config": config}
            game_dir = Path(config["game_dir"]) if config.get("game_dir") else None
            if game_dir:
                try:
                    map_cache = find_map_cache(game_dir)
                    payload["map_cache"] = str(map_cache)
                    payload["worlds"] = list_worlds(map_cache)
                    payload["textures"] = {k: {"path": str(v), "exists": v.exists()} for k, v in game_texture_paths(game_dir).items()}
                except Exception as exc:
                    payload["error"] = str(exc)
            json_response(self, payload)
            return
        if path == "/maps":
            json_response(self, [m.to_dict() for m in list_olmaps()])
            return
        if path.startswith("/maps/"):
            self.handle_tile(path[len("/maps/"):])
            return
        if path == "/scan":
            if self.last_scan is not None:
                json_response(self, self.last_scan)
                return
            self.handle_scan()
            return
        if path == "/jobs":
            json_response(self, {"jobs": [job.to_dict() for job in self.exporter.list()]})
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        body = read_json(self)
        if path == "/config":
            game_dir = str(body.get("game_dir") or "").strip()
            if not game_dir:
                json_response(self, {"error": "game_dir is required"}, 400)
                return
            config = load_config()
            config["game_dir"] = game_dir
            if body.get("blender"):
                config["blender"] = str(body.get("blender")).strip()
            if body.get("olmap") is not None:
                config["olmap"] = str(body.get("olmap") or "").strip()
            if isinstance(body.get("excluded_worlds"), list):
                config["excluded_worlds"] = [str(x) for x in body.get("excluded_worlds") if str(x).strip()]
            save_config(config)
            self.last_scan = None
            json_response(self, {"ok": True, "config": config})
            return
        if path == "/scan":
            self.handle_scan(force=True)
            return
        if path == "/export":
            config = load_config()
            if not config.get("game_dir"):
                json_response(self, {"error": "set game dir first"}, 400)
                return
            try:
                game_dir = Path(config["game_dir"]).resolve()
                map_cache = find_map_cache(game_dir)
                chunks = [(int(c[0]), int(c[1])) for c in (body.get("chunks") or [])]
                options = body.get("options") or {}
                if config.get("blender") and not options.get("blender"):
                    options["blender"] = config["blender"]
                if self.last_scan is None:
                    excluded = set(str(x) for x in config.get("excluded_worlds") or [])
                    self.last_scan = scan_map_cache(map_cache, excluded_worlds=excluded)
                    self.last_scan["game_dir"] = str(game_dir)
                    self.last_scan["olmap"] = config.get("olmap") or ""
                job = self.exporter.submit(game_dir=game_dir, map_cache=map_cache, scan=self.last_scan, chunks=chunks, options=options)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, 400)
                return
            json_response(self, job.to_dict(), 202)
            return
        self.send_error(404)

    def handle_scan(self, *, force: bool = False) -> None:
        config = load_config()
        if not config.get("game_dir"):
            json_response(self, {"error": "set game dir first"}, 400)
            return
        try:
            game_dir = Path(config["game_dir"]).resolve()
            map_cache = find_map_cache(game_dir)
            excluded = set(str(x) for x in config.get("excluded_worlds") or [])
            scan = scan_map_cache(map_cache, excluded_worlds=excluded)
            scan["game_dir"] = str(game_dir)
            scan["olmap"] = config.get("olmap") or ""
            scan["textures"] = {k: {"path": str(v), "exists": v.exists()} for k, v in game_texture_paths(game_dir).items()}
            self.last_scan = scan
            save_scan(DATA_DIR / "last_scan.json", scan)
            json_response(self, scan)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, 400)

    def handle_tile(self, suffix: str) -> None:
        try:
            name, kind, rx_s, rz_s = suffix.split("/", 3)
        except ValueError:
            self.send_error(404)
            return
        if kind != "tile" or not rz_s.endswith(".png"):
            self.send_error(404)
            return
        try:
            rx = int(rx_s)
            rz = int(rz_s[:-4])
        except ValueError:
            self.send_error(400)
            return
        olmap = get_olmap(name)
        if olmap is None:
            self.send_error(404)
            return
        try:
            data = self.tile_cache.get(olmap, rx, rz)
        except Exception as exc:
            print(f"TILE_ERROR {name} {rx} {rz}: {exc}", flush=True)
            self.send_error(500)
            return
        if data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def serve_static(self, name: str, default_type: str | None = None) -> None:
        clean = name.replace("\\", "/")
        if clean.startswith("/") or ".." in clean.split("/"):
            self.send_error(400)
            return
        path = UI_DIR / clean
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", default_type or guess_type(path))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def guess_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone map_cache .mdat scanner/exporter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39300)
    args = parser.parse_args()
    ensure_dirs()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MAPDUMP_MDAT_HTTP http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("MAPDUMP_MDAT_SHUTDOWN")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
