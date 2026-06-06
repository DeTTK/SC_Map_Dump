from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input))
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output), relative_remap=True)
    print(f"blend={args.output}", flush=True)


if __name__ == "__main__":
    main()
