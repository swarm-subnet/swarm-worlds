"""Bake a generate.py run into the files the simulator reads: maps/custom/solar/motions/motions.npz and motions.json.

    python bake.py run1 ../../swarm_worlds/maps/custom/solar/motions
"""

import argparse
import json
import os

import catalogue as cat
import numpy as np
from scipy.spatial.transform import Rotation


def soma30_slots(rig_path: str) -> tuple:
    """The 30 joint names and their indices among the rig's 77 joints."""
    rig = np.load(rig_path)
    names77 = [str(n) for n in rig["joint_names"]]
    names30 = [str(n) for n in rig["soma30"]]
    return names30, [names77.index(n) for n in names30]


def mirror_pairs(names: list) -> list:
    """Index pairs of left and right joints."""
    pairs = []
    for i, name in enumerate(names):
        if name.startswith("Left"):
            pairs.append([i, names.index("Right" + name[4:])])
    return pairs


def main() -> None:
    """Write the baked library."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("dest")
    parser.add_argument("--rig", default=os.path.join(os.path.dirname(__file__), "..", "..", "swarm_worlds", "maps",
                                                      "custom", "solar", "intruders", "rig.npz"))
    args = parser.parse_args()
    with open(os.path.join(args.run, "library.json"), encoding="utf-8") as handle:
        run = json.load(handle)
    names30, slots = soma30_slots(args.rig)
    rotations, roots, blocks = [], [], {}
    skipped = {block_id for block_id, entry in run["blocks"].items() if entry.get("failed")}
    # A block that starts on or leads into a left-out loop goes too, so no link points at nothing.
    while True:
        gone = {loop for loop in skipped if run["blocks"][loop]["loop"]}
        more = {block_id for block_id, entry in run["blocks"].items() if block_id not in skipped
                and any(link and link.get("loop") in gone for link in (entry["start"], entry["next"]))}
        if not more:
            break
        skipped |= more
    first = 0
    for block_id in sorted(run["blocks"]):
        entry = run["blocks"][block_id]
        if block_id in skipped:
            continue
        with np.load(os.path.join(args.run, "blocks", block_id + ".npz")) as data:
            local, root = data["local"][:, slots], data["root"]
        rotations.append(Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(len(local), 30, 4))
        roots.append(root)
        measures = entry.get("measures", {})
        blocks[block_id] = {
            "kind": entry["kind"], "stage": entry["stage"], "loop": entry["loop"], "first": first,
            "frames": len(local), "start": entry["start"], "next": entry["next"], "travel": entry.get("travel"),
            "prompt": entry["prompt"], "checks_failed": entry.get("failed", []),
            "measures": {k: round(float(v), 4) for k, v in measures.items() if v is not None},
        }
        first += len(local)
    os.makedirs(args.dest, exist_ok=True)
    np.savez_compressed(os.path.join(args.dest, "motions.npz"),
                        rotations=np.concatenate(rotations).astype(np.float32),
                        root=np.concatenate(roots).astype(np.float32))
    manifest = {
        "format": 1,
        "fps": cat.FPS,
        "source": f"NVIDIA {run['model']}, {run['steps']} denoising steps, generated offline",
        "frame": "Kimodo SOMA frame: y up, metres; every block starts with its root at x = z = 0 facing +z",
        "rotations": "rotations[first:first + frames] are local xyzw rotations of the 30 joints below, in that order",
        "joints": names30,
        "mirror": {"pairs": mirror_pairs(names30), "rotation": "(x, y, z, w) -> (x, -y, -z, w), then swap each pair",
                   "root": "x -> -x", "travel": "x -> -x, turn -> -turn"},
        "chaining": ("a block may follow another when its start equals the other's next; a loop repeats itself, "
                     "each cycle placed at travel (x, z, turn) from the previous one; start {'hub': h} follows any "
                     "block whose next is {'hub': h}; next null means hold the last frame"),
        "travel": ("where the block that follows starts, relative to this block's first frame, in the standard "
                   "build's metres: scale x and z by the build's root scale, as the root already is when posed"),
        "pin_frames": run["pin_frames"],
        "gates": run["gates"],
        "sprint_gates": run["sprint_gates"],
        "blocks": blocks,
    }
    with open(os.path.join(args.dest, "motions.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(f"baked {len(blocks)} blocks, {first} frames; left out after failing a check or depending on one that did: "
          f"{', '.join(sorted(skipped)) or 'none'}")


if __name__ == "__main__":
    main()
