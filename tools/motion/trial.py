"""Try rewrites of one loop's prompt and rank them by how many takes pass the loop's gates.

Each candidate is drawn from the loop's own base pose, scored by TMR against the whole catalogue plus every candidate
and, for a work loop, by how fast its hands keep working, then printed best first, so a prompt is chosen by
measurement rather than by taste.

    python trial.py --out run1 --loop cut_cable "A person kneeling ..." "A person on one knee ..."
"""

import argparse
import os

import catalogue as cat
import generate as gen
import numpy as np


def main() -> None:
    """Draw and score every candidate prompt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--loop", required=True)
    parser.add_argument("--takes", type=int, default=cat.LOOP_CANDIDATES)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--stub-text", action="store_true")
    parser.add_argument("--hub", help="start from this base pose instead of the loop's own")
    parser.add_argument("prompts", nargs="+")
    args = parser.parse_args()
    stage, _, start, speed, keep = cat.LOOPS[args.loop]
    start = args.hub or start
    frames = cat.LOOP_FRAMES_BY_NAME.get(args.loop, cat.LOOP_FRAMES)
    for n, prompt in enumerate(args.prompts):
        cat.LOOPS[f"{args.loop}_trial{n}"] = (stage, prompt, start, speed, keep)
    g = gen.Generator(args)
    hubs = g.hubs()
    requests = []
    for n, prompt in enumerate(args.prompts):
        pins = [gen.pin(hubs[start]["local"], hubs[start]["root"], 0)] if start else []
        if speed:
            pins.append(gen.straight_path(frames, speed))
        requests += [{"id": f"trial.{args.loop}.{start}.{n}.{k}", "prompt": prompt, "frames": frames, "pins": pins,
                      "meta": {"kind": args.loop, "trial": n}} for k in range(args.takes)]
    got = g.generate(requests)
    working = args.loop in cat.WORKING
    results = []
    for n, prompt in enumerate(args.prompts):
        takes = [got[f"trial.{args.loop}.{start}.{n}.{k}"] for k in range(args.takes)]
        ranks = [c["meta"]["tmr_rank"] for c in takes]
        # A work take counts only if its hands keep working for a stretch as long as the shortest cycle.
        stretch = [np.diff(g.lively(c["joints"], g.hand_speed, gen.GATES["hands_cm_s"]))[0] / cat.FPS
                   if working else 0.0 for c in takes]
        passing = sum(r <= gen.GATES["tmr_rank"] and (s >= 1.5 or not working) for r, s in zip(ranks, stretch))
        results.append((passing, float(np.mean([c["meta"]["tmr"] for c in takes])), sorted(ranks),
                        [round(float(s), 1) for s in stretch], prompt))
    for passing, score, ranks, stretch, prompt in sorted(results, reverse=True):
        print(f"{passing}/{args.takes} pass  mean {score:.3f}  ranks {ranks}  working s {stretch}  {prompt}", flush=True)
    g.pool.shutdown()


if __name__ == "__main__":
    main()
