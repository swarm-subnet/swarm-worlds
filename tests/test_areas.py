"""Checks that the real areas are fetched whole or not at all: every archive held to its listed size and SHA-256,
only what changed downloaded again, and the shipped list in step with the shipped catalogue."""

import errno
import functools
import hashlib
import http.server
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from swarm_worlds import areas

_SPEC = importlib.util.spec_from_file_location("pack_areas", Path(__file__).resolve().parents[1] / "tools" / "pack_areas.py")
pack_areas = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pack_areas)

NAMES = ("01_meadow", "02_wood", "shared")
RELEASE = "v9.8.7"


def _built(root: Path) -> Path:
    """A small built package: two areas and the shared folder, a few files each, one of them deep in a folder."""
    root.mkdir()
    (root / "areas.json").write_text(json.dumps({"version": 1, "areas": [{"id": 1, "slug": "01_meadow"}, {"id": 2, "slug": "02_wood"}]}))
    for name in NAMES:
        (root / name / "terrain").mkdir(parents=True)
        (root / name / "manifest.json").write_text(json.dumps({"name": name}))
        (root / name / "terrain" / "tile.obj").write_text(f"o {name}\n" + "v 0 0 0\n" * 200)
    return root


def _packed(tmp_path: Path, built: Path, name: str = "archives") -> tuple:
    """A built package's archives in a folder, and the list of them."""
    out = tmp_path / name
    return out, pack_areas.main(str(built), str(out), 1, release=RELEASE)["archives"]


def _files(root: Path) -> dict:
    """Return payload files by relative path, excluding the archive marks and fetch lock."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*"))
            if path.is_file() and path.name not in (areas.MARK_NAME, areas.FETCH_LOCK_NAME)}


def test_a_fetch_unpacks_every_listed_archive_and_a_second_one_fetches_nothing(tmp_path):
    """From the archives of a built package a fetch gives back every file byte for byte, marks each folder with its
    archive's SHA-256, leaves no archive or half-made folder behind, and has nothing to do when asked again."""
    built = _built(tmp_path / "built")
    out, lock = _packed(tmp_path, built)
    root = tmp_path / "installed"
    said = []
    assert areas.stale(root, lock) == sorted(NAMES)
    assert sorted(areas.fetch(root, str(out), lock, jobs=2, log=said.append)) == sorted(NAMES)
    assert {name: content for name, content in _files(root).items() if name != "areas.json"} == \
        {name: content for name, content in _files(built).items() if name != "areas.json"}
    assert all((root / name / areas.MARK_NAME).read_text().strip() == lock[name]["sha256"] for name in NAMES)
    assert not (root / areas.WORK_NAME).exists() and areas.stale(root, lock) == [] and areas.problems(root, lock) == []
    assert areas.fetch(root, str(out), lock, log=said.append) == [] and "all 3 archives in place" in said[-1]


def test_an_update_downloads_only_the_area_that_changed_and_replaces_its_folder_whole(tmp_path):
    """After one area is rebuilt with a file changed and a file gone, the new list names another archive for it
    alone: a fetch downloads that one, and its folder holds exactly the new files, nothing of the old."""
    built = _built(tmp_path / "built")
    out, lock = _packed(tmp_path, built)
    root = tmp_path / "installed"
    areas.fetch(root, str(out), lock, log=lambda line: None)
    (built / "02_wood" / "terrain" / "tile.obj").unlink()
    (built / "02_wood" / "manifest.json").write_text(json.dumps({"name": "02_wood", "trees": 7}))
    newer_out, newer = _packed(tmp_path, built, "archives_2")
    assert {name for name in NAMES if newer[name]["sha256"] != lock[name]["sha256"]} == {"02_wood"}
    assert areas.stale(root, newer) == ["02_wood"]
    assert areas.fetch(root, str(newer_out), newer, log=lambda line: None) == ["02_wood"]
    assert sorted(path.name for path in (root / "02_wood").rglob("*") if path.is_file()) == sorted([areas.MARK_NAME, "manifest.json"])
    assert json.loads((root / "02_wood" / "manifest.json").read_text())["trees"] == 7


def test_packing_the_same_files_again_gives_the_same_archives(tmp_path):
    """An archive carries no time, owner or order of the machine that packed it, so its SHA-256 names its files."""
    built = _built(tmp_path / "built")
    _out, first = _packed(tmp_path, built, "a")
    (built / "01_meadow" / "manifest.json").touch()
    _out, second = _packed(tmp_path, built, "b")
    assert first == second


def test_an_archive_that_is_not_the_listed_one_is_refused_and_the_old_folder_stands(tmp_path):
    """A download whose bytes differ from the list raises, names both fingerprints, and leaves the area that was
    there untouched and no file of its own behind."""
    built = _built(tmp_path / "built")
    out, lock = _packed(tmp_path, built)
    root = tmp_path / "installed"
    areas.fetch(root, str(out), lock, log=lambda line: None)
    before = _files(root)
    wrong = dict(lock, **{"01_meadow": dict(lock["01_meadow"], sha256="0" * 64)})
    with pytest.raises(ValueError, match="is not the listed archive") as refused:
        areas.fetch(root, str(out), wrong, log=lambda line: None)
    assert lock["01_meadow"]["sha256"] in str(refused.value) and "0" * 64 in str(refused.value)
    assert _files(root) == before and not (root / areas.WORK_NAME).exists()
    assert areas.problems(root, wrong) == [f"1 of 3 area archives are not fetched (01_meadow); run `{areas.COMMAND}`"]


def test_an_archive_that_reaches_outside_its_folder_is_refused(tmp_path):
    """An archive with a member outside its own folder, even one whose size and SHA-256 match the list, is never
    unpacked."""
    (tmp_path / "loose.txt").write_text("x")
    archive = tmp_path / "01_meadow.tar.xz"
    for member in ("../loose.txt", "02_wood/loose.txt"):
        with tarfile.open(archive, "w:xz") as tar:
            tar.add(tmp_path / "loose.txt", arcname=member)
        with pytest.raises(ValueError, match="not a plain file under 01_meadow/"):
            areas._unpack(archive, "01_meadow", tmp_path / "into")
    assert not (tmp_path / "into").exists() or not any((tmp_path / "into").rglob("loose.txt"))


def test_archives_are_fetched_over_http_and_a_missing_one_says_where_it_looked(tmp_path, monkeypatch):
    """The same fetch works from a URL, as from a release's files; an archive the place does not hold fails with
    the file and the place named, after its tries."""
    monkeypatch.setattr(areas.time, "sleep", lambda seconds: None)
    built = _built(tmp_path / "built")
    out, lock = _packed(tmp_path, built)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    handler.log_message = lambda *args: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        root = tmp_path / "installed"
        assert sorted(areas.fetch(root, url, lock, jobs=1, log=lambda line: None)) == sorted(NAMES)
        assert areas.stale(root, lock) == []
        (out / lock["shared"]["file"]).unlink()
        (root / "shared" / areas.MARK_NAME).unlink()
        with pytest.raises(OSError, match="shared.tar.xz could not be fetched from http://127.0.0.1"):
            areas.fetch(root, url, lock, jobs=1, log=lambda line: None)
    finally:
        server.shutdown()


def test_the_command_fetches_into_a_root_and_its_exit_code_says_whether_all_is_there(tmp_path, monkeypatch, capsys):
    """`check` exits 1 and names what is missing; a fetch into a root of its own also brings the catalogue and
    exits 0."""
    built = _built(tmp_path / "built")
    out, lock = _packed(tmp_path, built)
    package = tmp_path / "package"
    package.mkdir()
    (package / areas.LOCK_NAME).write_text((out / areas.LOCK_NAME).read_text())
    (package / areas.CATALOGUE_NAME).write_text((out / areas.CATALOGUE_NAME).read_text())
    monkeypatch.setattr(areas, "package_dir", lambda: package)
    root = tmp_path / "elsewhere"
    assert areas.main(["check", "--root", str(root)]) == 1
    assert "3 of 3 area archives are not fetched" in capsys.readouterr().out
    assert areas.main(["--root", str(root), "--source", str(out), "--jobs", "1"]) == 0
    assert json.loads((root / areas.CATALOGUE_NAME).read_text())["areas"][1]["slug"] == "02_wood"
    assert areas.main(["check", "--root", str(root)]) == 0


def test_the_default_source_follows_the_list_release(tmp_path, monkeypatch):
    """The list holds the archive release even after a package version bump."""
    (tmp_path / areas.LOCK_NAME).write_text(json.dumps({"release": RELEASE, "archives": {}}))
    monkeypatch.setattr(areas, "package_dir", lambda: tmp_path)
    monkeypatch.setenv(areas.SOURCE_ENV, "")
    assert areas.source() == f"{areas.RELEASES}/{RELEASE}"
    monkeypatch.setenv(areas.SOURCE_ENV, "/somewhere/else")
    assert areas.source() == "/somewhere/else"


def test_the_packer_writes_the_archive_release(tmp_path):
    """The packer records the supplied release name in its list."""
    built = _built(tmp_path / "built")
    out = tmp_path / "out"
    subprocess.run([sys.executable, _SPEC.origin, str(built), str(out), "1", "--release", RELEASE], check=True)
    assert json.loads((out / areas.LOCK_NAME).read_text())["release"] == RELEASE


def test_the_shipped_list_names_an_archive_for_every_area_of_the_shipped_catalogue():
    """The list in the package has one archive for each area the catalogue lists and one for the shared folder,
    each with a size, a file named after its folder and a whole SHA-256."""
    folder = areas.package_dir()
    lock = areas.read_lock()
    catalogue = json.loads((folder / areas.CATALOGUE_NAME).read_text())["areas"]
    assert len(catalogue) == 48 and {row["id"] for row in catalogue} == set(range(1, 49))
    assert set(lock) == {row["slug"] for row in catalogue} | {"shared"} and len(lock) == 49
    for row in catalogue:
        assert row["files"] > 0
        assert row["files"] + lock["shared"]["files"] <= lock[row["slug"]]["files"]
        assert 0 < row["size_bytes"] < row["size_with_shared_bytes"]
    for name, entry in lock.items():
        assert entry["file"] == f"{name}.tar.xz" and entry["bytes"] > 0 and entry["files"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    assert max(entry["bytes"] for entry in lock.values()) < 2 * 1024 ** 3
    assert len({entry["sha256"] for entry in lock.values()}) == len(lock)


@pytest.mark.parametrize("change", ["mark_only", "deleted_file", "stray_file", "missing_mark"])
def test_a_folder_with_incomplete_install_evidence_is_fetched_again(tmp_path, change):
    """A matching mark and the listed number of files are both needed."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    name = NAMES[0]
    lock = {name: lock[name]}
    root = tmp_path / "installed"
    areas.fetch(root, str(out), lock, log=lambda line: None)
    folder = root / name
    if change in ("mark_only", "deleted_file"):
        (folder / "manifest.json").unlink()
    if change == "mark_only":
        (folder / "terrain" / "tile.obj").unlink()
    if change == "stray_file":
        (folder / "stray.txt").touch()
    if change == "missing_mark":
        (folder / areas.MARK_NAME).unlink()
    assert areas.stale(root, lock) == [name]
    assert areas.fetch(root, str(out), lock, log=lambda line: None) == [name]
    assert areas.problems(root, lock) == []


def test_two_fetches_of_the_same_root_take_turns(tmp_path):
    """A second fetch waits for the first and finds that every folder is ready."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    root = tmp_path / "installed"
    first_started, second_waiting, release_first = (threading.Event() for _ in range(3))
    second_log = []

    def first_log(line):
        """Pause the first fetch before it downloads an archive."""
        if line.startswith("areas: fetching"):
            first_started.set()
            assert release_first.wait(5)

    def log_second(line):
        """Record when the second fetch has tried to acquire the lock."""
        second_log.append(line)
        if "waiting" in line:
            second_waiting.set()

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(areas.fetch, root, str(out), lock, 1, first_log)
        try:
            assert first_started.wait(5)
            second = pool.submit(areas.fetch, root, str(out), lock, 1, log_second)
            assert second_waiting.wait(2)
            assert not second.done()
        finally:
            release_first.set()
        assert first.result() == sorted(NAMES)
        assert second.result() == []
    assert sum("waiting" in line for line in second_log) == 1
    assert areas.problems(root, lock) == []
    assert not list(root.glob(f"{areas.WORK_NAME}*"))


class _ResponseHandler(http.server.BaseHTTPRequestHandler):
    """Serve a sequence of raw HTTP responses."""

    def handle(self):
        """Send the next response and close the connection."""
        self.rfile.readline()
        while self.rfile.readline() not in (b"\r\n", b""):
            pass
        self.wfile.write(self.server.responses.pop(0))


@pytest.mark.parametrize("broken", ["length", "chunk", "error_page"])
def test_a_bad_http_body_is_retried(tmp_path, monkeypatch, broken):
    """A cut body can recover, and repeated status 200 error pages identify the mismatch."""
    payload = b"listed archive"
    entry = {"file": "area.tar.xz", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    target = tmp_path / entry["file"]
    header = b"HTTP/1.1 200 OK\r\n"
    length = f"Content-Length: {len(payload)}\r\n\r\n".encode()
    bad_responses = {
        "length": header + length + payload[:3],
        "chunk": header + b"Transfer-Encoding: chunked\r\n\r\ne\r\n" + payload[:3],
        "error_page": header + length + b"x" * len(payload),
    }
    good = header + length + payload
    waits = []
    monkeypatch.setattr(areas.time, "sleep", waits.append)
    with http.server.HTTPServer(("127.0.0.1", 0), _ResponseHandler) as server:
        server.responses = [bad_responses[broken], good] if broken != "error_page" else [bad_responses[broken]] * areas.TRIES
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            if broken == "error_page":
                with pytest.raises(ValueError, match="area.tar.xz .* is not the listed archive"):
                    areas._download(url, entry, target)
                assert not target.exists()
                assert len(waits) == areas.TRIES - 1
            else:
                areas._download(url, entry, target)
                assert target.read_bytes() == payload
                assert len(waits) == 1
            assert not server.responses
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(("name", "kind"), [
    ("area/link", tarfile.SYMTYPE), ("area/link", tarfile.LNKTYPE),
    ("area/device", tarfile.CHRTYPE), ("/area/file", tarfile.REGTYPE),
])
def test_hostile_members_are_rejected_before_extraction(tmp_path, name, kind):
    """Links, devices and absolute names never reach extraction."""
    archive = tmp_path / "area.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.addfile(tarfile.TarInfo("area/file"))
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "area/file"
        tar.addfile(member)
    into = tmp_path / "into"
    with pytest.raises(ValueError, match="not a plain file under area/"):
        areas._unpack(archive, "area", into)
    assert not into.exists()


@pytest.mark.parametrize("empty", [True, False])
def test_an_archive_without_an_area_folder_is_refused(tmp_path, empty):
    """An empty archive or top-level file names the missing folder."""
    archive = tmp_path / "area.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        if not empty:
            tar.addfile(tarfile.TarInfo("area"))
    with pytest.raises(ValueError, match="area.tar.xz .* folder"):
        areas._unpack(archive, "area", tmp_path / "into")


def test_a_fetch_recovers_if_interrupted_between_the_renames(tmp_path, monkeypatch):
    """A retry replaces the area after an interruption leaves the old folder in scratch space."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    name = NAMES[0]
    root = tmp_path / "installed"
    work = root / f"{areas.WORK_NAME}-abandoned"
    target = root / name
    target.mkdir(parents=True)
    (target / "old.txt").touch()
    work.mkdir()
    replace = areas.os.replace

    def interrupt_new(source, destination):
        """Interrupt the final rename after the previous install was moved aside."""
        if destination == target:
            raise KeyboardInterrupt
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(areas.os, "replace", interrupt_new)
        with pytest.raises(KeyboardInterrupt):
            areas._install((str(root), str(work), name, lock[name], str(out)))
    assert not target.exists()
    assert (work / f"{name}.old" / "old.txt").is_file()
    assert areas.fetch(root, str(out), {name: lock[name]}, log=lambda line: None) == [name]
    assert not work.exists()
    assert areas.problems(root, {name: lock[name]}) == []


def test_a_complete_install_is_answered_without_writing_into_it(tmp_path):
    """Asked to fetch when every area is in place, the tool takes no lock and leaves the folder as it found it, so an
    install the user cannot write to still answers."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    root = tmp_path / "installed"
    areas.fetch(root, str(out), lock, log=lambda line: None)
    (root / areas.FETCH_LOCK_NAME).unlink()
    before = sorted(path.name for path in root.iterdir())
    assert areas.fetch(root, str(out), lock, log=lambda line: None) == []
    assert sorted(path.name for path in root.iterdir()) == before


def test_a_complete_root_of_its_own_still_gets_the_catalogue(tmp_path):
    """A root other than the package's that holds every area but no catalogue is given one on the next fetch."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    root = tmp_path / "installed"
    areas.fetch(root, str(out), lock, log=lambda line: None)
    (root / areas.CATALOGUE_NAME).unlink()
    assert areas.fetch(root, str(out), lock, log=lambda line: None) == []
    assert (root / areas.CATALOGUE_NAME).is_file()


def test_a_source_that_sends_more_than_the_listed_archive_is_cut_off(tmp_path, monkeypatch):
    """A body longer than the listed size is read no further than one block past it, and refused."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    name = sorted(lock)[0]
    read = []

    class Endless:
        """A stream that never ends."""

        def __enter__(self):
            """Open the stream."""
            return self

        def __exit__(self, *exc):
            """Close the stream."""
            return False

        def read(self, size):
            """Give one more block of zeros."""
            read.append(size)
            return b"\0" * size

    monkeypatch.setattr(areas, "_open", lambda source_, file: Endless())
    monkeypatch.setattr(areas.time, "sleep", lambda seconds: None)
    with pytest.raises(ValueError, match="not the listed archive"):
        areas._download(str(out), lock[name], tmp_path / "big.tar.xz")
    assert sum(read) <= areas.TRIES * (lock[name]["bytes"] + 2 * areas.CHUNK)
    assert not (tmp_path / "big.tar.xz").exists()


def test_a_fetch_removes_legacy_work_even_when_all_archives_are_present(tmp_path):
    """Abandoned scratch files are cleaned on an otherwise idle fetch."""
    root = tmp_path / "installed"
    work = root / areas.WORK_NAME
    work.mkdir(parents=True)
    (work / "area.tar.xz").touch()
    assert areas.fetch(root, str(tmp_path), {}, log=lambda line: None) == []
    assert not work.exists()


def test_a_folder_on_a_lower_layer_can_be_replaced(tmp_path, monkeypatch):
    """Moving a lower-layer directory can fall back to copying it when rename returns EXDEV."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    root = tmp_path / "installed"
    name = NAMES[0]
    target = root / name
    target.mkdir(parents=True)
    (target / "old.txt").touch()
    replace = areas.os.replace

    def replace_upper(source, destination):
        """Emulate a directory rename across layers."""
        if source == target:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        replace(source, destination)

    monkeypatch.setattr(areas.os, "replace", replace_upper)
    monkeypatch.setattr(areas.os, "rename", replace_upper)
    assert areas.fetch(root, str(out), {name: lock[name]}, log=lambda line: None) == [name]
    assert not (target / "old.txt").exists()
    assert areas.problems(root, {name: lock[name]}) == []


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_a_failure_cancels_queued_archives(tmp_path, monkeypatch, failure):
    """A later archive logs first, then a failure cancels work that has yet to start."""
    out, lock = _packed(tmp_path, _built(tmp_path / "built"))
    failing = Future()
    queued = []
    progress = []

    class Pool:
        """An executor with one failed archive and queued work."""

        def __init__(self, jobs):
            """Accept the requested number of workers."""
            assert jobs == 2

        def submit(self, install, task):
            """Complete the second archive while the first is running."""
            if task[2] == NAMES[0]:
                return failing
            future = Future()
            if task[2] == NAMES[1]:
                future.set_result(task[2])
            else:
                queued.append(future)
            return future

        def shutdown(self, *, cancel_futures=False):
            """Cancel the queued work when requested."""
            if cancel_futures:
                for future in queued:
                    future.cancel()

    def log(line):
        """Fail the first archive once the second's progress has been reported."""
        progress.append(line)
        if line == f"areas: {NAMES[1]} ok":
            failing.set_exception(failure("worker stopped"))

    monkeypatch.setattr(areas, "ProcessPoolExecutor", Pool)
    with pytest.raises(failure) as stopped:
        areas.fetch(tmp_path / "installed", str(out), lock, jobs=2, log=log)
    assert progress[-1] == f"areas: {NAMES[1]} ok"
    assert queued and all(future.cancelled() for future in queued)
    assert any(lock[NAMES[0]]["file"] in note for note in stopped.value.__notes__)


def test_the_packer_ignores_hidden_files_and_folders(tmp_path):
    """Install marks and other hidden files or folders are omitted."""
    built = _built(tmp_path / "built")
    out = tmp_path / "out"
    out.mkdir()
    folder = built / NAMES[0]
    (folder / ".fetched").touch()
    (folder / ".DS_Store").touch()
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "file").touch()
    _, entry = pack_areas.pack((built, out, NAMES[0]))
    assert entry["files"] == 2


def test_the_packer_rejects_directory_link_cycles(tmp_path):
    """A link back to an ancestor is refused before packing begins."""
    built = _built(tmp_path / "built")
    out = tmp_path / "out"
    out.mkdir()
    folder = built / NAMES[0]
    (folder / "terrain" / "cycle").symlink_to(folder, target_is_directory=True)
    with pytest.raises(ValueError, match="link cycle"):
        pack_areas.pack((built, out, NAMES[0]))
