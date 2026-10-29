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

"""The solar park's intruders: every piece ships with its weights in the same vertex order, is a closed surface on
its own origin, and the rig holds six builds of one body. Standard library only, like the rest of these tests."""

import ast
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest

import swarm_worlds

INTRUDERS = Path(swarm_worlds.maps_dir()) / "custom" / "solar" / "intruders"
MANIFEST = json.loads((INTRUDERS / "intruders.json").read_text(encoding="utf-8"))
PIECES = [piece for garment in MANIFEST["garments"].values() for piece in garment["pieces"]]


def _shapes(path: Path) -> dict:
    """Array shapes inside an .npz, read from each .npy header."""
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as handle:
                head = handle.read(10)
                size = int.from_bytes(head[8:10], "little") if head[6] == 1 else None
                if size is None:
                    head += handle.read(2)
                    size = int.from_bytes(head[8:12], "little")
                header = ast.literal_eval(handle.read(size).decode("latin1").strip())
            shapes[name[:-4]] = header["shape"]
    return shapes


def _obj(path: Path) -> tuple:
    """Vertices and triangles of an OBJ."""
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append(tuple(float(x) for x in line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(tuple(int(token.split("/")[0]) - 1 for token in line.split()[1:]))
    return verts, faces


def test_manifest_names_shipped_files():
    """The manifest, rig, body, licence and every piece's OBJ, material and weights are shipped."""
    for name in ("rig.npz", "body.obj", "body.mtl", "LICENSE_SOMA.txt"):
        assert (INTRUDERS / name).is_file(), name
    assert len(MANIFEST["garments"]) == 24
    for piece in PIECES:
        for suffix in (".obj", ".mtl", ".npz"):
            assert (INTRUDERS / (piece["file"] + suffix)).is_file(), piece["file"] + suffix


def test_rig_holds_six_builds_of_one_body():
    """Six builds share the body's vertices, faces and 77-joint skeleton, 1.6 to 1.9 m tall."""
    shapes = _shapes(INTRUDERS / "rig.npz")
    verts, faces = _obj(INTRUDERS / "body.obj")
    assert shapes["build_verts"] == (6, len(verts), 3)
    assert shapes["faces"] == (len(faces), 3)
    assert shapes["build_bind"] == (6, 77, 4, 4) and shapes["joint_names"] == (77,) and shapes["soma30"] == (30,)
    assert shapes["weights"] == shapes["weight_joints"] == (len(verts), 8)
    assert all(1.6 < height < 1.9 for height in MANIFEST["builds"].values()) and len(MANIFEST["builds"]) == 6


@pytest.mark.parametrize("piece", PIECES, ids=[piece["file"] for piece in PIECES])
def test_piece_is_closed_on_its_own_origin_with_its_weights(piece):
    """Triangles only, every edge shared by two of them, base on z = 0 and centred, one weight row per vertex."""
    verts, faces = _obj(INTRUDERS / (piece["file"] + ".obj"))
    assert all(len(face) == 3 for face in faces)
    edges = Counter(tuple(sorted((face[k], face[(k + 1) % 3]))) for face in faces for k in range(3))
    assert set(edges.values()) == {2}
    for axis in (0, 1):
        lo, hi = min(v[axis] for v in verts), max(v[axis] for v in verts)
        assert abs(lo + hi) < 1e-3
    assert abs(min(v[2] for v in verts)) < 1e-4
    shapes = _shapes(INTRUDERS / (piece["file"] + ".npz"))
    for name in ("weights", "weight_joints"):
        assert shapes[name] == (len(verts), 8)
    for name in ("bary", "local"):
        assert shapes[name] == (len(verts), 3)
    assert shapes["body_face"] == shapes["rigid_joint"] == shapes["outer"] == (len(verts),)


@pytest.mark.parametrize("piece", PIECES, ids=[piece["file"] for piece in PIECES])
def test_piece_colours_and_temperature(piece):
    """Each piece offers colours in 0-1 RGB, and a temperature for the thermal camera or none for passive."""
    assert piece["colours"] and all(len(c) == 3 and all(0.0 <= x <= 1.0 for x in c) for c in piece["colours"])
    assert piece["temperature_c"] is None or 20.0 <= piece["temperature_c"] <= 35.0
