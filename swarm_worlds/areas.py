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

"""The real areas: fetched beside the package, not shipped inside it.

The areas under maps/andorra are several gigabytes of meshes and pictures, too heavy for a repository every install
copies whole. So the package ships only their catalogue (areas.json) and a list (areas.lock.json) of one archive an
area, and one for the pieces the areas share, each with its size and SHA-256. The archives themselves are files of
the release named in the list. Fetching downloads the archives this install lacks, checks each against the list, and unpacks
it; an area already unpacked from the listed archive is left alone, so an update downloads only what changed.

    python -m swarm_worlds.areas            fetch what is missing
    python -m swarm_worlds.areas check      say what is missing, exit 1 if anything is

The install check reports folders whose archive mark or file count differs from the list.
"""

import argparse
import errno
import fcntl
import hashlib
import http.client
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import swarm_worlds

LOCK_NAME = "areas.lock.json"
CATALOGUE_NAME = "areas.json"
MARK_NAME = ".fetched"                 # inside an unpacked folder: the SHA-256 of the archive it came from
WORK_NAME = ".download"                # beside the areas: archives and folders in the making
FETCH_LOCK_NAME = ".fetch.lock"
SOURCE_ENV = "SWARM_WORLDS_AREAS_URL"  # another place to fetch from: a folder or a URL holding the archives
RELEASES = "https://github.com/swarm-subnet/swarm-worlds/releases/download"
COMMAND = "python -m swarm_worlds.areas"
CHUNK = 1 << 20
TRIES = 3
JOBS_MOST = 8                          # archives fetched at once unless told: the unpacking is what takes the time

PathLike = Union[str, Path]


def package_dir() -> Path:
    """Where the package keeps the catalogue and the list, and where the areas are unpacked unless told otherwise."""
    return Path(swarm_worlds.maps_dir()) / "andorra"


def read_lock(folder: Optional[PathLike] = None) -> Dict[str, dict]:
    """The archive entries keyed by the name of the folder they unpack to."""
    return _lock_data(folder)["archives"]


def _lock_data(folder: Optional[PathLike] = None) -> dict:
    """Read the archive list and its release from the given folder."""
    path = Path(folder if folder is not None else package_dir()) / LOCK_NAME
    return json.loads(path.read_text())


def source() -> str:
    """Use the source override when set or the archive release named in the list."""
    return os.environ.get(SOURCE_ENV) or f"{RELEASES}/{_lock_data()['release']}"


def stale(root: Optional[PathLike] = None, lock: Optional[Dict[str, dict]] = None) -> List[str]:
    """List folders whose archive mark or file count differs from the list."""
    root = Path(root if root is not None else package_dir())
    lock = read_lock() if lock is None else lock
    return [
        name for name, entry in sorted(lock.items())
        if _mark(root / name) != entry["sha256"] or sum(len(files) for _, _, files in os.walk(root / name)) != entry["files"] + 1
    ]


def problems(root: Optional[PathLike] = None, lock: Optional[Dict[str, dict]] = None) -> List[str]:
    """What this install lacks of the areas, one line, with the command that fetches it; empty when it has them all."""
    lock = read_lock() if lock is None else lock
    missing = stale(root, lock)
    if not missing:
        return []
    shown = ", ".join(missing[:4]) + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
    return [f"{len(missing)} of {len(lock)} area archives are not fetched ({shown}); run `{COMMAND}`"]


def fetch(root: Optional[PathLike] = None, source_: Optional[str] = None, lock: Optional[Dict[str, dict]] = None,
          jobs: Optional[int] = None, log: Callable[[str], None] = partial(print, flush=True)) -> List[str]:
    """Download, check and unpack every listed archive this install lacks; returns the folders it fetched. A root
    other than the package's own also gets the catalogue, so it can be read as a whole package."""
    package = package_dir()
    root = Path(root if root is not None else package)
    lock = read_lock() if lock is None else lock
    source_ = source() if source_ is None else source_
    # An install that has everything is often read-only for the user asking: it needs no lock file.
    listed = root.resolve() == package.resolve() or (root / CATALOGUE_NAME).is_file() or not (package / CATALOGUE_NAME).is_file()
    if listed and not stale(root, lock) and not (root / WORK_NAME).exists() and not any(root.glob(f"{WORK_NAME}-*")):
        log(f"areas: all {len(lock)} archives in place under {root}")
        return []
    root.mkdir(parents=True, exist_ok=True)
    # A shared writable root can have a lock file created by another user.
    with os.fdopen(os.open(root / FETCH_LOCK_NAME, os.O_RDONLY | os.O_CREAT, 0o666)) as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"areas: waiting for another fetch under {root}")
            fcntl.flock(guard, fcntl.LOCK_EX)
        return _fetch_locked(package, root, source_, lock, jobs, log)


def _fetch_locked(package: Path, root: Path, source_: str, lock: Dict[str, dict], jobs: Optional[int],
                  log: Callable[[str], None]) -> List[str]:
    """Clean abandoned scratch space and fetch the missing archives while holding the root lock."""
    for leftover in (root / WORK_NAME, *root.glob(f"{WORK_NAME}-*")):
        shutil.rmtree(leftover, ignore_errors=True)
    if root.resolve() != package.resolve() and (package / CATALOGUE_NAME).is_file():
        shutil.copyfile(package / CATALOGUE_NAME, root / CATALOGUE_NAME)
    wanted = stale(root, lock)
    if not wanted:
        log(f"areas: all {len(lock)} archives in place under {root}")
        return []
    total = sum(lock[name]["bytes"] for name in wanted)
    log(f"areas: fetching {len(wanted)} of {len(lock)} archives, {total / 1e6:.0f} MB, from {source_}")
    jobs = min(JOBS_MOST, os.cpu_count() or 1) if jobs is None else jobs
    with tempfile.TemporaryDirectory(dir=root, prefix=f"{WORK_NAME}-") as work:
        tasks = [(str(root), work, name, lock[name], source_) for name in wanted]
        # Process workers can unpack separate archives at the same time.
        if int(jobs) > 1 and len(tasks) > 1:
            _fetch_parallel(tasks, int(jobs), log)
        else:
            for task in tasks:
                try:
                    name = _install(task)
                except BaseException as error:
                    error.add_note(f"Archive: {task[3]['file']}")
                    raise
                log(f"areas: {name} ok")
    log(f"areas: {len(wanted)} fetched, all {len(lock)} in place under {root}")
    return wanted


def _fetch_parallel(tasks: List[tuple], jobs: int, log: Callable[[str], None]) -> None:
    """Log finished archives and cancel pending work on failure or interruption."""
    pool = ProcessPoolExecutor(jobs)
    try:
        pending = {pool.submit(_install, task): task[3]["file"] for task in tasks}
        for future in as_completed(pending):
            try:
                name = future.result()
            except BaseException as error:
                error.add_note(f"Archive: {pending[future]}")
                raise
            log(f"areas: {name} ok")
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
    else:
        pool.shutdown()


def _mark(folder: Path) -> Optional[str]:
    """The SHA-256 of the archive a folder was unpacked from, None for a folder that is not there or carries none."""
    try:
        return (folder / MARK_NAME).read_text().strip()
    except OSError:
        return None


def _open(source_: str, name: str):
    """A readable stream of one archive, from a folder on this machine or from a URL."""
    if "://" not in source_:
        return open(Path(source_) / name, "rb")
    request = urllib.request.Request(f"{source_.rstrip('/')}/{name}", headers={"User-Agent": f"swarm-worlds/{swarm_worlds.__version__}"})
    return urllib.request.urlopen(request, timeout=120)


def _download(source_: str, entry: dict, target: Path) -> None:
    """Write one archive to a file and hold it to its listed size and SHA-256; a wrong one is deleted and refused."""
    for attempt in range(TRIES):
        digest, size = hashlib.sha256(), 0
        try:
            with _open(source_, entry["file"]) as stream, open(target, "wb") as out:
                while True:
                    block = stream.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    size += len(block)
                    # A source that sends more than the listed archive holds is not read to its end.
                    if size > entry["bytes"]:
                        break
        except (OSError, http.client.HTTPException) as failed:
            if attempt == TRIES - 1:
                target.unlink(missing_ok=True)
                raise OSError(f"{entry['file']} could not be fetched from {source_}: {failed}") from failed
        else:
            sha256 = digest.hexdigest()
            if size == entry["bytes"] and sha256 == entry["sha256"]:
                return
            if attempt == TRIES - 1:
                target.unlink(missing_ok=True)
                raise ValueError(f"{entry['file']} from {source_} is not the listed archive: {size} bytes with SHA-256 {sha256}, "
                                 f"listed {entry['bytes']} bytes with {entry['sha256']}")
        time.sleep(2.0 * attempt)


def _unpack(archive: Path, name: str, into: Path) -> Path:
    """Unpack an archive that holds one folder of plain files, named as listed; anything else in it is refused."""
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not parts or parts[0] != name or ".." in parts or member.name.startswith("/") or not (member.isfile() or member.isdir()):
                raise ValueError(f"{archive.name} holds {member.name!r}, which is not a plain file under {name}/")
        # Every member was checked above; the library's own data filter is asked for too where it exists.
        tar.extractall(into, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    folder = into / name
    if not folder.is_dir():
        raise ValueError(f"{archive.name} did not unpack to a folder named {name}")
    return folder


def _install(task: tuple) -> str:
    """Fetch one archive and put its folder in place whole: the folder it replaces stands until the new one is
    ready. The task is (root, work folder, name, the archive's entry, source); returns the name."""
    root, work, name, entry, source_ = Path(task[0]), Path(task[1]), task[2], task[3], task[4]
    archive, fresh, old = work / entry["file"], work / f"{name}.new", work / f"{name}.old"
    for leftover in (fresh, old):
        shutil.rmtree(leftover, ignore_errors=True)
    _download(source_, entry, archive)
    folder = _unpack(archive, name, fresh)
    (folder / MARK_NAME).write_text(entry["sha256"] + "\n")
    archive.unlink()
    target = root / name
    if target.exists():
        try:
            os.replace(target, old)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.move(target, old)
    os.replace(folder, target)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(fresh, ignore_errors=True)
    return name


def main(argv: Optional[List[str]] = None) -> int:
    """Fetch the areas, or with `check` only say what is missing; the exit code is 1 when something still is."""
    parser = argparse.ArgumentParser(prog=COMMAND, description="Fetch the real areas this version of swarm-worlds names.")
    parser.add_argument("action", nargs="?", default="fetch", choices=("fetch", "check"))
    parser.add_argument("--root", help="unpack here instead of inside the installed package")
    parser.add_argument("--source", help=f"a folder or URL holding the archives (default: {SOURCE_ENV}, else the release named in the list)")
    parser.add_argument("--jobs", type=int, help=f"archives fetched at once (default: one a core, at most {JOBS_MOST})")
    args = parser.parse_args(argv)
    if args.action == "fetch":
        fetch(args.root, args.source, jobs=args.jobs)
    found = problems(args.root)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
