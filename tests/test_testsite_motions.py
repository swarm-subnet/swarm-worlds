"""The everyday motions in `testsite` for Swarm Test Field v2: valid rotations on the people's 30 joints, every chain the manifest allows
meets without a jump, loops close on themselves, what the simulator cannot do without is there, and nothing done on one
spot leaves the body anywhere else. Standard library only, like the rest of these tests."""

import json
import math
from pathlib import Path

from test_motions import SEAM_FLOOR_DEG, SEAM_SPREAD, _angle_deg, _arrays

import swarm_worlds

TESTSITE = Path(swarm_worlds.maps_dir()) / "custom" / "testsite"
MOTIONS = TESTSITE / "motions"
MANIFEST = json.loads((MOTIONS / "motions.json").read_text(encoding="utf-8"))
BLOCKS = MANIFEST["blocks"]
HUBS = ("stand", "crouch", "kneel", "sit")
DATA = _arrays(MOTIONS / "motions.npz", ("rotations", "root"))
ROT, ROT_SHAPE = DATA["rotations"]
ROOT, ROOT_SHAPE = DATA["root"]
ON_SPOT_M = 0.01       # everything done on one spot ends this near where it began
ON_SPOT_RAD = 0.01     # and turned no further than this


def _quat(frame: int, joint: int) -> tuple:
    """Local xyzw rotation of one joint on one library frame."""
    k = (frame * 30 + joint) * 4
    return tuple(ROT[k:k + 4])


def _gaps(frame_a: int, frame_b: int) -> list:
    """Each joint's rotation between two library frames in degrees, the root (whose heading is placement) excluded."""
    return [_angle_deg(_quat(frame_a, j), _quat(frame_b, j)) for j in range(1, 30)]


def _seamless(frame_a: int, frame_b: int, *blocks: dict) -> bool:
    """Whether going from frame_a to frame_b turns every joint no further than the blocks' own motion allows."""
    steps = [_gaps(f, f + 1) for b in blocks for f in range(b["first"], b["first"] + b["frames"] - 1)]
    allowed = [max(s[j] for s in steps) for j in range(29)]
    return all(g <= max(SEAM_FLOOR_DEG, SEAM_SPREAD * a) for g, a in zip(_gaps(frame_a, frame_b), allowed))


def _last(block: dict) -> int:
    """Library index of a block's last frame."""
    return block["first"] + block["frames"] - 1


def _after(a: dict, b: dict) -> dict:
    """Two travels one after the other: where the second ends, in the first one's frame."""
    c, s = math.cos(a["turn"]), math.sin(a["turn"])
    return {"x": a["x"] + c * b["x"] + s * b["z"], "z": a["z"] - s * b["x"] + c * b["z"], "turn": a["turn"] + b["turn"]}


def _stays(travel: dict) -> bool:
    """Whether a travel leaves the body on its spot, facing as it did."""
    return math.hypot(travel["x"], travel["z"]) <= ON_SPOT_M and abs(travel["turn"]) <= ON_SPOT_RAD


def test_arrays_match_the_manifest_and_the_peoples_skeleton():
    """One rotation per joint and one root per frame, blocks back to back, on the people's 30 joints at 30 fps, with
    the note that says where the motions come from."""
    total = sum(b["frames"] for b in BLOCKS.values())
    assert ROT_SHAPE == (total, 30, 4) and ROOT_SHAPE == (total, 3)
    ends = sorted((b["first"], b["first"] + b["frames"]) for b in BLOCKS.values())
    assert ends[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ends, ends[1:])) and ends[-1][1] == total
    rig = _arrays(TESTSITE / "people" / "rig.npz", ("soma30",))
    assert MANIFEST["joints"] == rig["soma30"][0] and MANIFEST["fps"] == 30
    assert (MOTIONS / "SOURCE.txt").is_file()


def test_every_rotation_is_a_unit_quaternion_and_every_block_starts_at_the_origin():
    """No NaN and no drifted length anywhere, and each block's root starts at x = z = 0."""
    for k in range(0, len(ROT), 4):
        assert abs(ROT[k] ** 2 + ROT[k + 1] ** 2 + ROT[k + 2] ** 2 + ROT[k + 3] ** 2 - 1.0) < 1e-4, k // 4
    for name, block in BLOCKS.items():
        k = block["first"] * 3
        assert abs(ROOT[k]) < 1e-3 and abs(ROOT[k + 2]) < 1e-3, name


def test_every_link_points_at_something_real():
    """start and next name a base pose or a loop block, and every base pose is both reached and left."""
    loops = {name for name, b in BLOCKS.items() if b["loop"]}
    reached, left = set(), set()
    for name, block in BLOCKS.items():
        for key, link in (("start", block["start"]), ("next", block["next"])):
            assert link is not None, name
            ((kind, target),) = link.items()
            assert (kind == "hub" and target in HUBS) or (kind == "loop" and target in loops), (name, link)
            if kind == "hub":
                (left if key == "start" else reached).add(target)
    assert left == set(HUBS) and reached == set(HUBS)


def test_loops_close_and_chains_meet_without_a_jump():
    """A loop's last frame flows into its first; a block leading into a loop ends one frame before the loop's first
    pose; every block leaving a base pose starts on the same pose, and every block arriving there ends on it."""
    for name, block in BLOCKS.items():
        if block["loop"]:
            assert _seamless(_last(block), block["first"], block), name
        link = block["next"]
        if "loop" in link and not block["loop"]:
            loop = BLOCKS[link["loop"]]
            assert _seamless(_last(block), loop["first"], block, loop), name
        if "hub" in link:
            follower = next(b for b in BLOCKS.values() if b["start"] == link)
            assert _seamless(_last(block), follower["first"], block, follower), name
    for hub in HUBS:
        starts = [b["first"] for b in BLOCKS.values() if b["start"] == {"hub": hub}]
        assert all(max(_gaps(starts[0], s)) < SEAM_FLOOR_DEG for s in starts), hub


def test_the_library_holds_what_people_cannot_do_without():
    """A plain walk that can be played whole (setting off from standing, the loop, stopping), something to do
    standing, a small move from standing back to standing, and a loop to work in for every low pose."""
    kinds = {}
    for name, block in BLOCKS.items():
        kinds.setdefault(block["kind"], []).append(name)
    walks = [name for name in kinds.get("walk", []) if BLOCKS[name]["loop"]]
    assert walks and any(BLOCKS[name]["start"] == {"hub": "stand"} for name in kinds.get("stand_to_walk", []))
    for loop in walks:
        assert any(BLOCKS[name]["start"] == {"loop": loop} and BLOCKS[name]["next"] == {"hub": "stand"}
                   for name in kinds.get("walk_to_stand", [])), loop
    assert any(BLOCKS[name]["loop"] for name in kinds.get("idle", []))
    assert any(BLOCKS[name]["start"] == BLOCKS[name]["next"] == {"hub": "stand"} for name in BLOCKS)
    for hub in ("crouch", "kneel", "sit"):
        assert any(BLOCKS[name]["start"] == {"hub": hub} and "loop" in BLOCKS[name]["next"] for name in BLOCKS), hub


def test_nothing_done_on_one_spot_leaves_the_body_anywhere_else():
    """A block from standing back to standing ends where and facing as it began; so does every thing done standing,
    over its way in from standing and its way back, the loop between them being played on the spot."""
    small = [name for name, b in BLOCKS.items() if b["start"] == b["next"] == {"hub": "stand"}]
    assert small and all(_stays(BLOCKS[name]["travel"]) for name in small)
    bouts = 0
    for way_in, block in BLOCKS.items():
        if block["start"] != {"hub": "stand"} or "loop" not in block["next"] or block["loop"]:
            continue
        loop = block["next"]["loop"]
        cycle = BLOCKS[loop]["travel"]
        if math.hypot(cycle["x"], cycle["z"]) > 0.2 * BLOCKS[loop]["frames"] / MANIFEST["fps"]:
            continue    # a way of walking: it is meant to carry the body on
        for way_out, back in BLOCKS.items():
            if back["start"] == {"loop": loop} and back["next"] == {"hub": "stand"}:
                assert _stays(_after(block["travel"], back["travel"])), (way_in, way_out)
                bouts += 1
    assert bouts >= 8
