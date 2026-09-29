"""Try rewrites of one loop's prompt and rank them by how well Kimodo's takes match their own text.

Each candidate is drawn from the loop's own base pose, scored by TMR against the whole catalogue plus every candidate,
and printed best first, so a prompt is chosen by measurement rather than by taste.

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
    parser.add_argument("prompts", nargs="+")
    args = parser.parse_args()
    stage, _, start, speed, keep = cat.LOOPS[args.loop]
    for n, prompt in enumerate(args.prompts):
        cat.LOOPS[f"{args.loop}_trial{n}"] = (stage, prompt, start, speed, keep)
    g = gen.Generator(args)
    hubs = g.hubs()
    requests = []
    for n, prompt in enumerate(args.prompts):
        pins = [gen.pin(hubs[start]["local"], hubs[start]["root"], 0)] if start else []
        if speed:
            pins.append(gen.straight_path(cat.LOOP_FRAMES, speed))
        requests += [{"id": f"trial.{args.loop}.{n}.{k}", "prompt": prompt, "frames": cat.LOOP_FRAMES, "pins": pins,
                      "meta": {"kind": args.loop, "trial": n}} for k in range(args.takes)]
    got = g.generate(requests)
    results = []
    for n, prompt in enumerate(args.prompts):
        ranks = [got[f"trial.{args.loop}.{n}.{k}"]["meta"]["tmr_rank"] for k in range(args.takes)]
        scores = [got[f"trial.{args.loop}.{n}.{k}"]["meta"]["tmr"] for k in range(args.takes)]
        passing = sum(r <= gen.GATES["tmr_rank"] for r in ranks)
        results.append((passing, float(np.mean(scores)), sorted(ranks), prompt))
    for passing, score, ranks, prompt in sorted(results, reverse=True):
        print(f"{passing}/{args.takes} pass  mean {score:.3f}  ranks {ranks}  {prompt}")
    g.pool.shutdown()


if __name__ == "__main__":
    main()
