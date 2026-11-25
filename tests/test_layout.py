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

"""Checks that every asset root, manifest entry, robot description, material library and sky the package ships resolves."""

import ast
import json
import math
import os
import re
import zipfile
from pathlib import Path

import pytest

import swarm_worlds

MAPS = Path(swarm_worlds.maps_dir())
TEXTURES = Path(swarm_worlds.textures_dir())
SKIES = Path(swarm_worlds.skies_dir())

# GitHub refuses files above 100 MB and warns above 50 MB.
MAX_FILE_BYTES = 50 * 1024 * 1024

# The folders the simulator addresses directly below each root.
EXPECTED_MAP_DIRS = {
    "custom/buildings",
    "custom/mountains",
    "custom/office",
    "custom/people/lost_person_characters",
    "custom/people/open_mannequin_raw/split",
    "custom/solar",
    "custom/solar/intruders",
    "custom/testsite",
    "custom/testsite/terrain",
    "custom/testsite/plants",
    "custom/testsite/goats",
    "custom/testsite/dogs",
    "custom/testsite/motions",
    "custom/testsite/people",
    "custom/testsite/people/pieces",
    "custom/testsite/people/pictures",
    "custom/warehouse_shell",
    "forest/quaternius_ultimate_nature",
    "forest/textures",
    "kenney/holiday",
    "kenney/kenney_car-kit",
    "kenney/kenney_commercial",
    "kenney/kenney_conveyor-kit",
    "kenney/kenney_furniture-kit",
    "kenney/kenney_roads",
    "kenney/kenney_suburban",
    "other_sources/factory_fence_new",
    "other_sources/loading_kit",
    "other_sources/overhead_crane",
    "other_sources/vehicles",
}


def _all_files(root: Path):
    """Every file under a root, recursively."""
    return [p for p in root.rglob("*") if p.is_file()]


def test_roots_are_absolute_directories():
    """Each asset root resolves to an absolute directory."""
    for root in (MAPS, TEXTURES, SKIES):
        assert root.is_absolute()
        assert root.is_dir()


@pytest.mark.parametrize("rel", sorted(EXPECTED_MAP_DIRS))
def test_expected_map_directories_exist(rel):
    """Every folder the simulator addresses below the maps root exists."""
    assert (MAPS / rel).is_dir()


def test_textures_present():
    """The three standalone textures are shipped."""
    for name in ("tao.png", "Swarm.png", "Swarm_2.png"):
        assert (TEXTURES / name).is_file()


def test_sky_manifest_points_at_real_equirectangular_files():
    """Every sky in the manifest is a shipped 2048 x 1024 8-bit RGB PNG with a sun position and a source."""
    manifest = json.loads((SKIES / "skies.json").read_text(encoding="utf-8"))
    skies = manifest["skies"]
    assert len(skies) >= 8
    assert {p.name for p in SKIES.glob("*.png")} == {sky["file"] for sky in skies}
    for sky in skies:
        path = SKIES / sky["file"]
        assert path.is_file(), sky["file"]
        assert 0.0 <= sky["sun_azimuth_deg"] < 360.0
        assert 0.0 < sky["sun_elevation_deg"] < 90.0
        assert sky["source"].startswith("https://polyhaven.com/a/")
        with open(path, "rb") as handle:
            head = handle.read(29)
        width, height = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        bit_depth, colour_type = head[24], head[25]
        assert (width, height) == (2048, 1024), sky["file"]
        assert (bit_depth, colour_type) == (8, 2), sky["file"]


def test_solar_manifest_points_at_real_files():
    """Every solar item names files that ship, every placement names an item, and the forest's table and meshes ship."""
    solar = MAPS / "custom" / "solar"
    manifest = json.loads((solar / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    for name, item in items.items():
        for key in ("obj", "texture"):
            assert (solar / item["folder"] / item[key]).is_file(), f"{name}: {item[key]}"
    assert {place["item"] for place in manifest["placements"]} <= set(items)
    forest = manifest["forest"]
    for name in [forest["table"]] + forest["meshes"]:
        assert (solar / forest["folder"] / name).is_file(), name


def _npy_shape(archive, member):
    """Shape of one array in an npz, read from its header so the check needs no numpy."""
    with archive.open(member) as handle:
        head = handle.read(10)
        size = int.from_bytes(head[8:10], "little")
        return ast.literal_eval(handle.read(size).decode("latin1"))["shape"]


def test_solar_dogs_point_at_real_files():
    """Every dog body names files that ship, its pose table fits its mesh and covers every clip, and no dog is a threat."""
    dogs = MAPS / "custom" / "solar" / "dogs"
    table = json.loads((dogs / "dogs.json").read_text(encoding="utf-8"))
    assert (table["class"], table["threat"], table["decoy"]) == ("dog", False, True)
    assert table["bodies"]
    for name, body in table["bodies"].items():
        for file in [body["obj"], body["poses"], body["heat_map"], *body["coats"].values()]:
            assert (dogs / file).is_file(), f"{name}: {file}"
        lines = (dogs / body["obj"]).read_text(encoding="utf-8").splitlines()
        vertices = sum(line.startswith("v ") for line in lines)
        triangles = sum(line.startswith("f ") for line in lines)
        with zipfile.ZipFile(dogs / body["poses"]) as archive:
            frames, welded, axes = _npy_shape(archive, "poses.npy")
            (corners,) = _npy_shape(archive, "corners.npy")
        assert (welded, axes, corners) == (vertices, 3, 3 * triangles), name
        for clip, spec in body["clips"].items():
            assert spec["start"] + spec["frames"] <= frames, f"{name}: {clip}"
            assert spec.get("then", clip) in body["clips"], f"{name}: {clip}"


# The Test Field's animals, each a folder with its table and the class it must carry.
TESTSITE_ANIMALS = (("goats", "goats.json", "goat"), ("dogs", "dogs.json", "dog"), ("birds", "birds.json", "bird"),
                    ("boars", "boars.json", "boar"), ("rats", "rats.json", "rat"))


def test_testsite_animals_point_at_real_files():
    """Every animal of the Test Field names files that ship, its skin and pose table fit each other and cover every
    clip, every clip that follows another exists, and no animal is a threat."""
    for folder, table_name, kind in TESTSITE_ANIMALS:
        home = MAPS / "custom" / "testsite" / folder
        table = json.loads((home / table_name).read_text(encoding="utf-8"))
        assert (table["class"], table["threat"], table["decoy"]) == (kind, False, True)
        assert (home / "SOURCE.txt").is_file() and table["bodies"]
        for name, body in table["bodies"].items():
            for file in [body["obj"], body["poses"], body["heat_map"], *body["coats"].values()]:
                assert (home / file).is_file(), f"{name}: {file}"
            with zipfile.ZipFile(home / body["poses"]) as archive:
                frames, points, axes = _npy_shape(archive, "poses.npy")
                (drawn,) = _npy_shape(archive, "source.npy")
                assert _npy_shape(archive, "uv.npy") == (drawn, 2), name
                triangles, corners = _npy_shape(archive, "faces.npy")
            assert (axes, corners) == (3, 3) and drawn >= points, name
            assert (points, drawn, triangles) == (body["vertices"], body["render_vertices"], body["triangles"]), name
            lines = (home / body["obj"]).read_text(encoding="utf-8").splitlines()
            assert sum(line.startswith("v ") for line in lines) == drawn, name
            assert sum(line.startswith("f ") for line in lines) == triangles, name
            for clip, spec in body["clips"].items():
                assert spec["start"] + spec["frames"] <= frames, f"{name}: {clip}"
                assert spec.get("then", clip) in body["clips"], f"{name}: {clip}"
        for clip, followers in table["joins"].items():
            assert all(name in body["clips"] for body in table["bodies"].values() for name in [clip, *followers]), clip


def test_testsite_people_point_at_real_files():
    """The Test Field's people: every skin, piece and picture named ships, the skin and every piece hold their points
    on every build with bone weights and a drawn surface that fit each other, every tint comes from a palette that
    exists, women and men are both there, and the package carries its source note and the body's licence."""
    home = MAPS / "custom" / "testsite" / "people"
    table = json.loads((home / "people.json").read_text(encoding="utf-8"))
    assert (home / "SOURCE.txt").is_file() and (home / "LICENSE_SOMA.txt").is_file()
    with zipfile.ZipFile(home / "rig.npz") as archive:
        builds, joints, rows, cols = _npy_shape(archive, "build_bind.npy")
        points, per_point = _npy_shape(archive, "weights.npy")
        assert _npy_shape(archive, "build_rest.npy") == (builds, joints, 3) and (rows, cols) == (4, 4)
        assert _npy_shape(archive, "weight_joints.npy") == (points, per_point)
    assert builds == len(table["builds"]) and {spec["sex"] for spec in table["builds"].values()} == {"f", "m"}
    assert all(1.4 < spec["height_m"] < 2.0 for spec in table["builds"].values())

    def drawn_surface(archive, name):
        """Check one drawn surface: every drawn vertex has a picture place, and the triangles have three corners."""
        (drawn,) = _npy_shape(archive, "source.npy")
        assert _npy_shape(archive, "uv.npy") == (drawn, 2), name
        assert _npy_shape(archive, "faces.npy")[1] == 3, name

    with zipfile.ZipFile(home / "body.npz") as archive:
        assert _npy_shape(archive, "points.npy") == (builds, points, 3)
        drawn_surface(archive, "body")
    assert table["skins"] and {skin["sex"] for skin in table["skins"].values()} == {"f", "m"}
    for name, skin in table["skins"].items():
        assert (home / skin["file"]).is_file(), name
        assert skin["tone"] in table["eyes_of_tone"] and f"hair_{skin['hair']}" in table["palettes"], name
    for name, piece in table["pieces"].items():
        assert (home / piece["file"]).is_file() and piece["pictures"], name
        for picture in piece["pictures"].values():
            assert (home / picture["file"]).is_file(), name
            # A hair picture is tinted from the palette of its wearer's hair, dark or light.
            palettes = {key if piece["slot"] != "hair" else key.rsplit("_", 1)[0] for key in table["palettes"]}
            assert picture.get("palette") is None or picture["palette"] in palettes | set(table["palettes"]), name
        with zipfile.ZipFile(home / piece["file"]) as archive:
            piece_builds, piece_points, axes = _npy_shape(archive, "points.npy")
            assert (piece_builds, axes) == (builds, 3), name
            assert _npy_shape(archive, "weights.npy")[0] == _npy_shape(archive, "weight_joints.npy")[0] == piece_points, name
            drawn_surface(archive, name)
    assert {"eyes", "hair", "top", "bottom", "shoes"} <= {piece["slot"] for piece in table["pieces"].values()}


def test_solar_public_road_is_drivable_with_places_to_turn_round():
    """Both arms end on the same yard samples, no bend is tighter than a pickup turns, every pose is a unit rotation,
    and every turn starts on a sample of each arm it takes in and ends on one of each arm it leaves by: the yard's on
    both, a corner's from one arm onto the other, the rest back on their own."""
    road = json.loads((MAPS / "custom" / "solar" / "movers" / "public_road.json").read_text(encoding="utf-8"))
    lines, shared = road["lines"], road["yard_samples"]
    assert lines["north"][-shared:] == lines["east"][-shared:]
    tightest = 1.0 / max(abs(row[2]) for line in lines.values() for row in line)
    assert tightest > road["wheelbase_m"] * 1.5
    for row in lines["north"] + lines["east"]:
        assert len(row) == 19
        for pose in (row[6:10], row[14:18]):
            assert math.isclose(math.hypot(*pose), 1.0, abs_tol=1e-4)
    yards = [turn for turn in road["turns"] if len(turn["into"]) == 2]
    assert len(yards) == 1 and max(yards[0]["from_end"], yards[0]["to_end"]) < shared
    assert sorted((t["into"], t["out"]) for t in road["turns"] if t["into"] != t["out"]) == [(["east"], ["north"]), (["north"], ["east"])]
    for turn in road["turns"]:
        assert {row[2] for row in turn["rows"]} == {1.0, -1.0}
        for arm in turn["into"]:
            assert turn["rows"][0][:2] == lines[arm][len(lines[arm]) - turn["from_end"]][:2]
        for arm in turn["out"]:
            assert math.dist(turn["rows"][-1][:2], lines[arm][len(lines[arm]) - turn["to_end"]][:2]) < 1e-3
    for line in lines.values():
        steps = [math.dist(a[:2], b[:2]) for a, b in zip(line, line[1:])]
        assert max(steps) <= road["spacing_m"] * 1.5


def test_testsite_manifest_points_at_real_files():
    """Every test site item names files that ship, every placement names an item, and the forest ships."""
    testsite = MAPS / "custom" / "testsite"
    manifest = json.loads((testsite / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    for name, item in items.items():
        for key in ("obj", "texture"):
            assert (testsite / item["folder"] / item[key]).is_file(), f"{name}: {item[key]}"
    assert {place["item"] for place in manifest["placements"]} <= set(items)
    forest = manifest["forest"]
    for name in [forest["table"]] + forest["meshes"]:
        assert (testsite / forest["folder"] / name).is_file(), name


def test_lost_person_manifest_points_at_real_files():
    """Every lost-person character in the manifest has its model file, a scale and maps."""
    people = MAPS / "custom" / "people" / "lost_person_characters"
    manifest = json.loads((people / "manifest.json").read_text(encoding="utf-8"))
    characters = manifest["characters"]
    assert len(characters) == manifest["character_count"]
    for character in characters:
        assert (people / character["model"]).is_file(), character["key"]
        assert character["scale"] > 0
        assert character["maps"]


def test_the_package_ships_no_urdf():
    """Every URDF the simulator spawns lives in swarm-description: no robots root here, and no URDF in any root."""
    assert not hasattr(swarm_worlds, "robots_dir")
    assert not (Path(swarm_worlds.__file__).parent / "robots").exists()
    assert [p for root in (MAPS, TEXTURES, SKIES) for p in root.rglob("*.urdf")] == []


def test_no_file_exceeds_github_limit():
    """No shipped file is above the size GitHub refuses."""
    too_big = [p for root in (MAPS, TEXTURES, SKIES) for p in _all_files(root) if p.stat().st_size > MAX_FILE_BYTES]
    assert too_big == []


# Two source meshes reference a material library that was never shipped with
# them; the loaders fall back to their own colours. Anything new here is a
# mistake in a pull request.
KNOWN_MISSING_MTL = {
    "custom/buildings/SnowRoofs/lantern_roof.obj -> light.mtl",
    "custom/mountains/mountain_peak.obj -> Meshy_AI_A_compact_ski_mountai_0124194554_texture.mtl",
}


def test_obj_material_libraries_resolve():
    """Every material library an OBJ names is shipped, except the two known missing ones."""
    missing = set()
    for obj in MAPS.rglob("*.obj"):
        for line in obj.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("mtllib "):
                rel = line.split(None, 1)[1].strip()
                if not (obj.parent / rel).is_file():
                    missing.add(f"{obj.relative_to(MAPS).as_posix()} -> {rel}")
    assert missing == KNOWN_MISSING_MTL


def test_version_matches_tag_format():
    """The package version is a plain three-number tag."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", swarm_worlds.__version__)


def test_package_data_is_installed_not_only_checked_out():
    """The asset roots come from the installed package, not from a source checkout."""
    # A wheel or git install must ship the files, not just an editable checkout.
    assert os.path.isfile(MAPS / "README.md")
    assert os.path.isfile(MAPS / "custom" / "LICENSE_CUSTOM_MAPS.md")
