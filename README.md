# Map_Dump_MDAT

Standalone app for existing `.mdat` `map_cache` data.

The user points it at the game directory. The app finds the latest versioned
cache such as `<game_dir>\map_cache\5.0`, scans world folders with
`reg.*.*.mdat`, draws chunks over the local PNG map underlay, and exports
selected chunks to a `.blend` file.

## Run

```powershell
pip install -r requirements.txt
python mdat_app.py
```

Open <http://127.0.0.1:39300/>.

World folders can be excluded in the sidebar. When multiple included worlds
overlap at the same chunk coordinate, the scan keeps one active source chunk
for export without drawing overlaps differently.

## External tools

- Local `tools\mdat_obj_export.py` copied into this app
- Blender, selected in the UI or auto-detected from `BLENDER_EXE`, `PATH`, or common install paths

The exporter uses textures from:

```text
<game_dir>\modassets\assets\stalcraft\textures\blockMap.texarr
<game_dir>\modassets\assets\stalcraft\ctmpatcher\ctm
```

Exports are written into `exports/`.
