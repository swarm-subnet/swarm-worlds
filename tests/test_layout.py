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

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import swarm_worlds

MAPS = Path(swarm_worlds.maps_dir())
ROBOTS = Path(swarm_worlds.robots_dir())
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
    "other_sources/swarm_drone",
    "other_sources/vehicles",
}


def _all_files(root: Path):
    """Every file under a root, recursively."""
    return [p for p in root.rglob("*") if p.is_file()]


def test_roots_are_absolute_directories():
    """Each asset root resolves to an absolute directory."""
    for root in (MAPS, ROBOTS, TEXTURES, SKIES):
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


# The interceptor drone reuses the stock Crazyflie visual mesh, which the swarm
# repository stages next to the URDF from the gym package at runtime.
MESHES_FROM_GYM = {"cf2.dae"}


@pytest.mark.parametrize("urdf", sorted(p.name for p in ROBOTS.glob("*.urdf")))
def test_urdf_parses_and_meshes_resolve(urdf):
    """Each robot description parses and every mesh it names is shipped, but for the one staged from the gym."""
    path = ROBOTS / urdf
    root = ET.parse(path).getroot()
    assert root.tag == "robot"
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("./") and filename[2:] not in MESHES_FROM_GYM:
            assert (ROBOTS / filename[2:]).is_file(), filename


def test_robots_contain_the_two_swarm_drones():
    """The two drone descriptions and the tello mesh folder are shipped."""
    assert (ROBOTS / "tello.urdf").is_file()
    assert (ROBOTS / "interceptor_drone.urdf").is_file()
    assert (ROBOTS / "tello").is_dir()


def test_no_file_exceeds_github_limit():
    """No shipped file is above the size GitHub refuses."""
    too_big = [p for root in (MAPS, ROBOTS, TEXTURES, SKIES) for p in _all_files(root) if p.stat().st_size > MAX_FILE_BYTES]
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
