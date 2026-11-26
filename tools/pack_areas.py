# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Pack a built area package into the archives a release carries and the list the repository keeps.

Usage: python tools/pack_areas.py PACKAGE_ROOT OUT_FOLDER [JOBS] --release TAG

PACKAGE_ROOT holds areas.json, one folder an area and the shared folder. Every folder becomes OUT_FOLDER/<name>.tar.xz,
links followed, with its files in sorted order and no owner, time or mode of this machine in it.
Each area archive is self-contained, and the shared archive holds horizon, rock and tree assets from the shared folder.
OUT_FOLDER/areas.lock.json lists the archive release and each archive's size and SHA-256, and areas.json is
copied beside it. Copy those two files into swarm_worlds/maps/andorra/ and attach the archives to the release:
swarm_worlds.areas fetches them from there and holds each one to the list.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LOCK_NAME = "areas.lock.json"
CATALOGUE_NAME = "areas.json"
SHARED = "shared"
PRESET = 6


def plain(member: tarfile.TarInfo) -> tarfile.TarInfo:
    """A member with nothing of this machine in it: no owner, no time, one mode a kind."""
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mtime = 0
    member.mode = 0o755 if member.isdir() else 0o644
    return member


def pack(job: tuple) -> tuple:
    """Write one folder's archive; returns (name, its entry for the list)."""
    root, out, name = Path(job[0]), Path(job[1]), job[2]
    files = _visible_files(root / name)
    assert files, f"{name} holds no file"
    target = out / f"{name}.tar.xz"
    with tarfile.open(target.with_name(target.name + ".part"), "w:xz", preset=PRESET, format=tarfile.GNU_FORMAT, dereference=True) as tar:
        for path in files:
            tar.add(path, arcname=str(Path(name) / path.relative_to(root / name)), recursive=False, filter=plain)
    os.replace(target.with_name(target.name + ".part"), target)
    digest = hashlib.sha256()
    with open(target, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return name, {"file": target.name, "sha256": digest.hexdigest(), "bytes": target.stat().st_size, "files": len(files)}


def _visible_files(root: Path) -> list[Path]:
    """Collect sorted asset files, following directory links and rejecting ancestor cycles."""
    files = []
    ancestors = {}
    for folder, dirs, leaves in os.walk(root, followlinks=True):
        path = Path(folder)
        info = path.stat()
        identity = (info.st_dev, info.st_ino)
        parents = ancestors[path.parent] if path != root else set()
        if identity in parents:
            raise ValueError(f"directory link cycle at {path}")
        ancestors[path] = parents | {identity}
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        files.extend(path / leaf for leaf in leaves if not leaf.startswith("."))
    return sorted(files)


def main(package_root: str, out_folder: str, jobs: int = 4, *, release: str) -> dict:
    """Pack every area the catalogue lists and the shared folder, and write the list; returns the list."""
    root, out = Path(package_root), Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(row["slug"] for row in json.loads((root / CATALOGUE_NAME).read_text())["areas"])
    names += [SHARED] if (root / SHARED).is_dir() else []
    work = [(str(root), str(out), name) for name in names]
    if int(jobs) > 1:
        with ProcessPoolExecutor(int(jobs)) as pool:
            archives = dict(pool.map(pack, work))
    else:
        archives = dict(map(pack, work))
    lock = {"format": 1, "release": release, "archives": dict(sorted(archives.items()))}
    (out / LOCK_NAME).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    shutil.copyfile(root / CATALOGUE_NAME, out / CATALOGUE_NAME)
    total = sum(entry["bytes"] for entry in archives.values())
    print(f"{len(archives)} archives, {total / 1e6:.0f} MB, largest {max(entry['bytes'] for entry in archives.values()) / 1e6:.0f} MB, "
          f"{sum(entry['files'] for entry in archives.values())} files")
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack the real area archives for a release.")
    parser.add_argument("package_root")
    parser.add_argument("out_folder")
    parser.add_argument("jobs", nargs="?", default=4, type=int)
    parser.add_argument("--release", required=True, help="tag of the release that will carry the archives")
    main(**vars(parser.parse_args()))
