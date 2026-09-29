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

"""The intruders' motion library: every block is valid rotations on the intruder rig's 30 joints, every chain the
manifest allows meets without a jump, loops close on themselves, and the mirror rule reflects a pose exactly.
Standard library only, like the rest of these tests."""

import ast
import json
import math
import zipfile
from array import array
from pathlib import Path

import pytest

import swarm_worlds

SOLAR = Path(swarm_worlds.maps_dir()) / "custom" / "solar"
MOTIONS = SOLAR / "motions"
MANIFEST = json.loads((MOTIONS / "motions.json").read_text(encoding="utf-8"))
BLOCKS = MANIFEST["blocks"]
HUBS = ("stand", "kneel", "crouch", "prone")
SEAM_FLOOR_DEG = 4.0  # below this a joint's turn between two frames cannot be seen
SEAM_SPREAD = 1.5     # a seam may turn a joint half again as far as its largest turn between two frames; a pop is 3x or more


def _arrays(path: Path, names: tuple) -> dict:
    """The named arrays of an .npz as (flat values, shape): float, integer and unicode arrays."""
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in names:
            raw = archive.read(name + ".npy")
            size = int.from_bytes(raw[8:10], "little") if raw[6] == 1 else int.from_bytes(raw[8:12], "little")
            start = (10 if raw[6] == 1 else 12) + size
            header = ast.literal_eval(raw[start - size:start].decode("latin1").strip())
            kind = header["descr"]
            if kind[1] == "U":
                width = int(kind[2:])
                body = raw[start:].decode("utf-32-le")
                values = [body[i:i + width].rstrip("\0") for i in range(0, len(body), width)]
            else:
                values = array({"<f4": "f", "<f8": "d", "<i2": "h", "<i4": "i", "<i8": "q"}[kind])
                values.frombytes(raw[start:])
            out[name] = (values, header["shape"])
    return out


DATA = _arrays(MOTIONS / "motions.npz", ("rotations", "root"))
ROT, ROT_SHAPE = DATA["rotations"]
ROOT, ROOT_SHAPE = DATA["root"]
RIG = _arrays(SOLAR / "intruders" / "rig.npz", ("joint_names", "parents", "soma30", "build_names", "build_rest"))


def _quat(frame: int, joint: int) -> tuple:
    """Local xyzw rotation of one joint on one library frame."""
    k = (frame * 30 + joint) * 4
    return tuple(ROT[k:k + 4])


def _angle_deg(a: tuple, b: tuple) -> float:
    """Angle between two unit quaternions in degrees."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, dot)))


def _gaps(frame_a: int, frame_b: int) -> list:
    """Each joint's rotation between two library frames in degrees, the root (whose heading is placement) excluded."""
    return [_angle_deg(_quat(frame_a, j), _quat(frame_b, j)) for j in range(1, 30)]


def _normal_turn(*blocks: dict) -> list:
    """Per joint, its largest turn between neighbouring frames within the given blocks."""
    steps = [_gaps(f, f + 1) for b in blocks for f in range(b["first"], b["first"] + b["frames"] - 1)]
    return [max(s[j] for s in steps) for j in range(29)]


def _seamless(frame_a: int, frame_b: int, *blocks: dict) -> bool:
    """Whether going from frame_a to frame_b turns every joint no further than the blocks' own motion allows."""
    allowed = _normal_turn(*blocks)
    return all(g <= max(SEAM_FLOOR_DEG, SEAM_SPREAD * a) for g, a in zip(_gaps(frame_a, frame_b), allowed))


def _last(block: dict) -> int:
    """Library index of a block's last frame."""
    return block["first"] + block["frames"] - 1


def test_arrays_match_the_manifest():
    """One rotation per joint and one root per frame, blocks back to back, on the rig's 30 joints at 30 fps."""
    total = sum(b["frames"] for b in BLOCKS.values())
    assert ROT_SHAPE == (total, 30, 4) and ROOT_SHAPE == (total, 3)
    assert sorted((b["first"], b["frames"]) for b in BLOCKS.values())[0][0] == 0
    ends = sorted((b["first"], b["first"] + b["frames"]) for b in BLOCKS.values())
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:])) and ends[-1][1] == total
    assert MANIFEST["joints"] == RIG["soma30"][0] and MANIFEST["fps"] == 30


def test_every_rotation_is_a_unit_quaternion():
    """No NaN and no drifted length anywhere in the library."""
    for k in range(0, len(ROT), 4):
        norm = ROT[k] ** 2 + ROT[k + 1] ** 2 + ROT[k + 2] ** 2 + ROT[k + 3] ** 2
        assert abs(norm - 1.0) < 1e-4, k // 4


def test_every_block_starts_at_the_origin():
    """Each block's root starts at x = z = 0, so the director places it where the previous one ended."""
    for name, block in BLOCKS.items():
        k = block["first"] * 3
        assert abs(ROOT[k]) < 1e-3 and abs(ROOT[k + 2]) < 1e-3, name


def test_every_link_points_at_something_real():
    """start and next name a base pose or a loop block, and every base pose is both reached and left."""
    loops = {name for name, b in BLOCKS.items() if b["loop"]}
    reached, left = set(), set()
    for name, block in BLOCKS.items():
        for key, link in (("start", block["start"]), ("next", block["next"])):
            if link is None:
                assert key == "next" and not block["loop"], name
                continue
            ((kind, target),) = link.items()
            assert (kind == "hub" and target in HUBS) or (kind == "loop" and target in loops), (name, link)
            if kind == "hub":
                (left if key == "start" else reached).add(target)
    assert left == set(HUBS) and reached <= left


def test_work_and_walks_can_react():
    """Every take of the work and walking loops can freeze and can run: the reactions the story needs."""
    for loop, block in BLOCKS.items():
        if not block["loop"] or block["kind"] not in ("walk", "walk_wary", "cut_fence", "cut_cable", "pull_cable"):
            continue
        exits = {b["kind"] for b in BLOCKS.values() if b["start"] == {"loop": loop} and not b["loop"]}
        assert any(k.endswith("freeze") for k in exits) and any(k.endswith("to_run") for k in exits), loop


def test_loops_close_on_themselves():
    """A loop's last frame flows into its first like any two neighbouring frames of that loop."""
    for name, block in BLOCKS.items():
        if block["loop"]:
            assert _seamless(_last(block), block["first"], block), name


def test_chains_meet_without_a_jump():
    """A block leading into a loop ends one frame before that loop's first pose; every block leaving a base pose
    starts on the same pose, and every block arriving there ends on it."""
    for name, block in BLOCKS.items():
        link = block["next"]
        if link and "loop" in link and not block["loop"]:
            loop = BLOCKS[link["loop"]]
            assert _seamless(_last(block), loop["first"], block, loop), name
    for hub in HUBS:
        starts = [b["first"] for b in BLOCKS.values() if b["start"] == {"hub": hub}]
        assert all(max(_gaps(starts[0], s)) < SEAM_FLOOR_DEG for s in starts), hub
    for name, block in BLOCKS.items():
        link = block["next"]
        if link and "hub" in link:
            follower = next(b for b in BLOCKS.values() if b["start"] == link)
            assert _seamless(_last(block), follower["first"], block, follower), name


def _rest_offsets() -> list:
    """Each of the 30 joints' offset from its parent on the standard build, with the parent's 30-joint index."""
    names77 = RIG["joint_names"][0]
    parents77 = [int(p) for p in RIG["parents"][0]]
    builds = RIG["build_names"][0]
    rest, _ = RIG["build_rest"]
    b = builds.index("standard")
    names30 = RIG["soma30"][0]
    out = []
    for name in names30:
        j = names77.index(name)
        p = parents77[j]
        while p >= 0 and names77[p] not in names30:
            p = parents77[p]
        pos = [rest[(b * 77 + j) * 3 + a] for a in range(3)]
        base = [rest[(b * 77 + p) * 3 + a] for a in range(3)] if p >= 0 else [0.0, 0.0, 0.0]
        out.append((names30.index(names77[p]) if p >= 0 else -1, [x - y for x, y in zip(pos, base)]))
    return out


def _qmul(a: tuple, b: tuple) -> tuple:
    """Product of two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _rotate(q: tuple, v: list) -> list:
    """A vector turned by an xyzw quaternion."""
    x, y, z, _ = _qmul(_qmul(q, (v[0], v[1], v[2], 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return [x, y, z]


def _fk(quats: list, offsets: list) -> list:
    """Joint positions of one pose on the 30-joint chain, root at the origin; skipped finger joints add no turn."""
    world_q, world_p = [], []
    for j, (parent, offset) in enumerate(offsets):
        if parent < 0:
            world_q.append(quats[j])
            world_p.append([0.0, 0.0, 0.0])
        else:
            world_q.append(_qmul(world_q[parent], quats[j]))
            world_p.append([a + b for a, b in zip(world_p[parent], _rotate(world_q[parent], offset))])
    return world_p


def test_mirror_rule_reflects_the_pose():
    """Applying the manifest's mirror rule gives the pose reflected across x, within a centimetre, on every block."""
    offsets = _rest_offsets()
    pairs = MANIFEST["mirror"]["pairs"]
    swap = list(range(30))
    for a, b in pairs:
        swap[a], swap[b] = b, a
    for name, block in BLOCKS.items():
        frame = block["first"] + block["frames"] // 2
        quats = [_quat(frame, j) for j in range(30)]
        mirrored = [(q[0], -q[1], -q[2], q[3]) for q in quats]
        mirrored = [mirrored[swap[j]] for j in range(30)]
        original = _fk(quats, offsets)
        reflected = _fk(mirrored, offsets)
        for j in range(30):
            expected = original[swap[j]]
            got = reflected[j]
            assert math.dist((-expected[0], expected[1], expected[2]), got) < 0.01, (name, MANIFEST["joints"][j])


@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_block_passed_its_checks(name):
    """Every shipped block passed the generation gates: text match, seams, foot slide and jitter."""
    assert BLOCKS[name]["checks_failed"] == []
