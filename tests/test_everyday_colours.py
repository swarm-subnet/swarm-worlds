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

"""The everyday clothing colours: every piece of an everyday garment has them, each colour group is clearly apart from
the others of its kind of part, and they live beside the thieves' own colours without touching them. Standard
library only, like the rest of these tests."""

import itertools
import json
import math
from pathlib import Path

import swarm_worlds

INTRUDERS = Path(swarm_worlds.maps_dir()) / "custom" / "solar" / "intruders"
EVERYDAY = json.loads((INTRUDERS / "everyday.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((INTRUDERS / "intruders.json").read_text(encoding="utf-8"))
GARMENTS = ("tshirt", "hoodie", "hoodie_up", "work_jacket", "puffer", "work_trousers", "jeans", "cargo", "work_boots",
            "trainers", "hair_short", "cap", "beanie")
# The parts a person is told apart by from above, each the main pieces of the garments that make it.
PARTS = {
    "top": ("tshirt", "hoodie", "hoodie_up", "work_jacket", "puffer"),
    "trousers": ("work_trousers", "jeans", "cargo"),
    "hair": ("hair_short",),
    "cap": ("cap",),
    "beanie": ("beanie",),
}
APART = 20.0                            # least CIE76 distance between two groups of one part


def _lab(rgb: list) -> tuple:
    """CIE Lab of an sRGB colour from 0 to 1, D65 white."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    xyz = [sum(m * c for m, c in zip(row, linear)) / white for row, white in
           zip(((0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722), (0.0193, 0.1192, 0.9505)), (0.95047, 1.0, 1.08883))]
    f = [v ** (1.0 / 3.0) if v > (6.0 / 29.0) ** 3 else v / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0 for v in xyz]
    return 116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])


def test_every_piece_of_an_everyday_garment_has_its_colours():
    """Each piece of the everyday garments, and nothing else, has at least one colour group of three values in 0 to 1."""
    pieces = {piece["file"] for name in GARMENTS for piece in MANIFEST["garments"][name]["pieces"]}
    assert set(EVERYDAY["colours"]) == pieces
    for groups in EVERYDAY["colours"].values():
        assert groups
        for rgb in groups.values():
            assert len(rgb) == 3 and all(0.0 <= c <= 1.0 for c in rgb)


def test_groups_of_one_part_are_clearly_apart():
    """Two colours of a part in different groups are at least APART in CIE Lab, across all the part's garments, and a
    group keeps one colour across them."""
    for part, garments in PARTS.items():
        colours = {}
        for name in garments:
            for group, rgb in EVERYDAY["colours"][MANIFEST["garments"][name]["pieces"][0]["file"]].items():
                assert colours.setdefault(group, rgb) == rgb, (part, group)
        for (a, rgb_a), (b, rgb_b) in itertools.combinations(colours.items(), 2):
            assert math.dist(_lab(rgb_a), _lab(rgb_b)) >= APART, (part, a, b)


def test_the_everyday_people_have_more_looks_than_ten():
    """At least a dozen top groups and five trouser groups, so ten people in one seed can all look different."""
    tops = {g for name in PARTS["top"] for g in EVERYDAY["colours"][f"pieces/{name}_main"]}
    trousers = {g for name in PARTS["trousers"] for g in EVERYDAY["colours"][f"pieces/{name}_main"]}
    assert len(tops) >= 12 and len(trousers) >= 5
