"""Generate the Solar Patrol motion library with NVIDIA Kimodo: base poses, then loops, then every join and reaction.

Runs where Kimodo is installed (a rented GPU for the real library). Every candidate is kept on disk with its measures,
the chosen ones become canonical blocks in <out>/blocks, and <out>/library.json records how they chain. A stage whose
output exists is not generated again, so an interrupted run resumes where it stopped.

    python generate.py --out run1
"""

import argparse
import importlib
import json
import multiprocessing
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import catalogue as cat
import motionlib as ml
import numpy as np
import torch
from kimodo.constraints import load_constraints_lst
from kimodo.model.load_model import load_model
from kimodo.postprocess import post_process_motion
from kimodo.skeleton.registry import build_skeleton
from scipy.spatial.transform import Rotation

LOADING = importlib.import_module("kimodo.model.load_model")
MODEL = "kimodo-soma-rp-v1.1"
TMR = "tmr-soma-rp"
PIN = 3  # frames pinned at each join, so both the pose and its speed carry across
GATES = {
    "seam_m": 0.03,        # pinned frames within 3 cm of the pose they join
    "loop_m": 0.06,        # cycle mismatch before closing
    "slide_cm_s": 6.0,     # feet on the ground drift less than 6 cm/s
    "jitter_m_s3": 400.0,  # no pops
    "tmr_rank": 5,         # its own prompt is among the 5 best matching prompts of the catalogue
    "legs_cm": 0.5,        # a walk or run lifts its feet (at least 0.5 cm of spread); a gliding statue does not
}
# A sprint is jerkier than a walk by nature: its own steady loop measures about 1,800 m/s^3 and 6.5 cm/s, so sprint
# moves are held to twice that instead of to the walking limits.
SPRINT_GATES = {"slide_cm_s": 13.0, "jitter_m_s3": 3600.0}

_SKELETON = None  # each clean-up worker's own 30-joint SOMA skeleton


def _worker_init() -> None:
    """A clean-up worker: one CPU thread and its own skeleton."""
    global _SKELETON
    torch.set_num_threads(1)
    _SKELETON = build_skeleton(30)


def clean_up(local77: np.ndarray, root: np.ndarray, contacts6: np.ndarray, pins: list) -> dict:
    """Kimodo's own foot-skate and constraint clean-up of one sample, returned on the 77-joint skeleton.

    The same steps Kimodo runs inside generation, moved to a worker so every CPU core cleans while the GPU draws.
    """
    contacts4 = torch.as_tensor(contacts6[:, [0, 1, 3, 4]])
    local30 = _SKELETON.from_SOMASkeleton77(torch.as_tensor(local77))
    fixed = post_process_motion(local30[None], torch.as_tensor(root)[None], contacts4[None], _SKELETON,
                                [load_constraints_lst(pins, _SKELETON, device="cpu")])
    out = _SKELETON.output_to_SOMASkeleton77({"local_rot_mats": fixed["local_rot_mats"][0],
                                              "root_positions": fixed["root_positions"][0],
                                              "foot_contacts": contacts4})
    return {"local": out["local_rot_mats"].numpy().astype(np.float32),
            "root": out["root_positions"].numpy().astype(np.float32),
            "joints": out["posed_joints"].numpy().astype(np.float32),
            "contacts": out["foot_contacts"].numpy() > 0.5}


class CachedText(torch.nn.Module):
    """Kimodo's text encoder with every prompt encoded once."""

    def __init__(self, inner):
        """Wrap the loaded encoder."""
        super().__init__()
        self.inner = inner
        self.cache = {}

    def _one(self, text: str) -> torch.Tensor:
        """The (1, 4096) embedding of one prompt."""
        if text not in self.cache:
            self.cache[text] = self.inner(text)[0]
        return self.cache[text]

    def forward(self, text):
        """Same contract as the wrapped encoder: (embedding, length) or a batch of both."""
        if isinstance(text, str):
            return self._one(text), 1
        return torch.stack([self._one(t) for t in text]), [1] * len(text)

    def get_device(self):
        """The wrapped encoder's device."""
        return self.inner.get_device()


class RandomText:
    """Stand-in text encoder for checking the pipeline without the 8B text model: a fixed random vector per prompt."""

    def __call__(self, text):
        """Kimodo's text encoder contract with seeded random embeddings."""
        texts = [text] if isinstance(text, str) else text
        emb = torch.stack([torch.from_numpy(np.random.default_rng(zlib.crc32(t.encode())).standard_normal((1, 4096)))
                           .float() for t in texts])
        return (emb[0], 1) if isinstance(text, str) else (emb, [1] * len(texts))

    def to(self, device):
        """No weights to move."""
        return self

    def eval(self):
        """No weights to switch."""
        return self

    def get_device(self):
        """Always on the CPU."""
        return "cpu"


def axis_angle(local: np.ndarray) -> list:
    """(K, J, 3, 3) rotations as nested axis-angle lists for a constraint."""
    return Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(local.shape[0], -1, 3).tolist()


def pin(local: np.ndarray, root: np.ndarray, first: int) -> dict:
    """A full-body constraint holding frames first.. to the given poses."""
    return {"type": "fullbody", "frame_indices": list(range(first, first + len(local))),
            "local_joints_rot": axis_angle(local), "root_positions": root.tolist()}


def straight_path(frames: int, speed: float) -> dict:
    """A root path straight along +z at a constant speed, facing forward the whole way."""
    z = np.arange(frames) / cat.FPS * speed
    return {"type": "root2d", "frame_indices": list(range(frames)),
            "smooth_root_2d": np.stack([np.zeros(frames), z], 1).tolist(),
            "global_root_heading": [[1.0, 0.0]] * frames}


class Generator:
    """Kimodo, TMR and the bookkeeping for one library run."""

    def __init__(self, args):
        """Load the models and open the run folder."""
        self.args, self.out = args, args.out
        self.pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_worker_init)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = load_model(MODEL, device=device, text_encoder=RandomText() if args.stub_text else None)
        self.model.text_encoder = CachedText(self.model.text_encoder)
        # load_model's text_encoder argument would replace TMR's own text head, so TMR loads with no text model and
        # then shares Kimodo's.
        select = LOADING._select_text_encoder_conf
        LOADING._select_text_encoder_conf = lambda *args, **kwargs: None
        try:
            self.tmr = load_model(TMR, device=device, default_family="TMR")
        finally:
            LOADING._select_text_encoder_conf = select
        self.tmr.raw_text_encoder = self.model.text_encoder
        self.sk77 = self.model.skeleton.somaskel77
        names = list(self.sk77.bone_order_names)
        self.hips = (names.index("RightLeg"), names.index("LeftLeg"))
        self.feet = [(names.index("LeftFoot"), 0), (names.index("LeftToeBase"), 1),
                     (names.index("RightFoot"), 3), (names.index("RightToeBase"), 4)]
        self.body = [names.index(n) for n, _ in self.model.skeleton.bone_order_names_with_parents]
        for sub in ("raw", "blocks"):
            os.makedirs(os.path.join(self.out, sub), exist_ok=True)
        self.log = open(os.path.join(self.out, "log.jsonl"), "a", encoding="utf-8")
        texts = list(cat.HUBS.values()) + [v[1] for v in cat.LOOPS.values()] + [v[1] for v in cat.MOVES.values()]
        self.all_texts = texts
        self.text_emb = self.tmr.encode_raw_text(texts, unit_vector=True).cpu().numpy()

    def fk(self, local: np.ndarray, root: np.ndarray) -> np.ndarray:
        """World joint positions (T, 77, 3) of 77-joint local rotations and a root path."""
        _, joints, _ = self.sk77.fk(torch.as_tensor(local, dtype=torch.float32, device=self.device),
                                    torch.as_tensor(root, dtype=torch.float32, device=self.device))
        return joints.cpu().numpy()

    def heading(self, joints: np.ndarray) -> np.ndarray:
        """Per-frame heading from the hip line, as Kimodo defines it."""
        d = joints[:, self.hips[0]] - joints[:, self.hips[1]]
        return np.arctan2(d[:, 2], -d[:, 0])

    def generate(self, requests: list) -> dict:
        """Run every request not already on disk, in batches; returns id -> candidate.

        The GPU draws batch k + 1 while the CPU pool cleans the feet of batch k.
        """
        # A take is stored under its id and what it was pinned to, so a changed pin is drawn again, never reused.
        for r in requests:
            r["key"] = f"{r['id']}.{zlib.crc32(json.dumps(r['pins']).encode()):08x}"
        todo = [r for r in requests if not os.path.exists(self._raw(r["key"]))]
        todo.sort(key=lambda r: r["frames"])
        pending = None
        for start in range(0, len(todo), self.args.batch):
            drawn = self._draw(todo[start:start + self.args.batch])
            if pending:
                self._finish(*pending)
            pending = drawn
        if pending:
            self._finish(*pending)
        got = {r["id"]: self.load(r["key"]) for r in requests}
        self._rescore(got.values())
        return got

    def _rescore(self, candidates) -> None:
        """Rank every candidate's text match against the whole current catalogue, so takes drawn before a prompt was
        added are held to the same list as the ones drawn after."""
        for c in candidates:
            emb = self.tmr.encode_motion(torch.as_tensor(c["joints"][None], device=self.device),
                                         original_skeleton=self.sk77, unit_vector=True).cpu().numpy().reshape(-1)
            scores = self.text_emb @ emb
            own = self.all_texts.index(c["meta"]["prompt"])
            c["meta"]["tmr"] = float(scores[own] / 2 + 0.5)
            c["meta"]["tmr_rank"] = int((scores > scores[own]).sum()) + 1

    def _raw(self, cid: str) -> str:
        """Path of one candidate's file."""
        return os.path.join(self.out, "raw", cid + ".npz")

    def load(self, cid: str) -> dict:
        """One stored candidate."""
        with np.load(self._raw(cid), allow_pickle=False) as data:
            c = {k: data[k] for k in data.files}
        c["meta"] = json.loads(str(c["meta"]))
        return c

    def _draw(self, batch: list) -> tuple:
        """Diffuse one batch on the GPU and hand each sample's foot clean-up to the CPU pool."""
        torch.manual_seed(zlib.crc32(batch[0]["id"].encode()))
        prompts = [r["prompt"] for r in batch]
        frames = [r["frames"] for r in batch]
        constraints = [load_constraints_lst(r["pins"], self.model.skeleton, device=self.device) for r in batch]
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.model(prompts, frames, self.args.steps, constraint_lst=constraints, post_processing=False,
                         return_numpy=True, progress_bar=lambda x: x)
        if self.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - t0
        futures = [self.pool.submit(clean_up, out["local_rot_mats"][k, :n], out["root_positions"][k, :n],
                                    out["foot_contacts"][k, :n], r["pins"])
                   for k, (r, n) in enumerate(zip(batch, frames))]
        return batch, futures, seconds, t0

    def _finish(self, batch: list, futures: list, seconds: float, t0: float) -> None:
        """Collect the cleaned samples, score them with TMR and store them."""
        cleaned = [f.result() for f in futures]
        wall = time.time() - t0
        # TMR's canonicalisation breaks on batches of more than one motion, so each is encoded alone.
        motion_emb = np.concatenate([
            self.tmr.encode_motion(torch.as_tensor(c["joints"][None], device=self.device),
                                   original_skeleton=self.sk77, unit_vector=True).cpu().numpy().reshape(1, -1)
            for c in cleaned])
        scores = motion_emb @ self.text_emb.T
        for k, (r, c) in enumerate(zip(batch, cleaned)):
            own = self.all_texts.index(r["prompt"])
            rank = int((scores[k] > scores[k, own]).sum()) + 1
            meta = dict(r["meta"], id=r["id"], prompt=r["prompt"], frames=r["frames"],
                        tmr=float(scores[k, own] / 2 + 0.5), tmr_rank=rank, gpu_s=seconds / len(batch))
            np.savez(self._raw(r["key"]), **c, meta=json.dumps(meta))
        self.log.write(json.dumps({"t": time.time(), "batch": len(batch), "frames": max(r["frames"] for r in batch),
                                   "gpu_s": seconds, "wall_s": wall, "first": batch[0]["id"]}) + "\n")
        self.log.flush()

    def measure(self, c: dict, lo: int = 0, hi: int = None) -> dict:
        """Foot slide and jitter over frames [lo, hi) of a candidate."""
        j = c["joints"][lo:hi]
        return {"slide_cm_s": ml.foot_slide(j, c["contacts"][lo:hi], self.feet, cat.FPS),
                "jitter_m_s3": ml.jitter(j, cat.FPS)}

    def canonical(self, local, root, joints) -> tuple:
        """A clip moved so its first frame stands at the origin facing +z."""
        yaw = self.heading(joints[:1])[0]
        return ml.reframe(local, root, joints, root[0, [0, 2]], yaw)

    def seam(self, joints: np.ndarray, target_local: np.ndarray, target_root: np.ndarray) -> float:
        """Largest mean joint distance in metres between pinned frames and the poses they were pinned to."""
        target = self.fk(target_local, target_root)
        return float(np.linalg.norm(joints - target, axis=-1).mean(-1).max())

    def save_block(self, block_id: str, local, root, joints, contacts, info: dict) -> dict:
        """Store one canonical block and return its entry for the library."""
        local, root, joints = self.canonical(local, root, joints)
        np.savez(os.path.join(self.out, "blocks", block_id + ".npz"), local=local.astype(np.float32),
                 root=root.astype(np.float32), joints=joints.astype(np.float32), contacts=contacts)
        return dict(info, id=block_id, frames=len(local))

    # ----------------------------------------------------------------------------------------------- stages

    def hubs(self) -> dict:
        """The four base poses: from each prompt's takes, the stillest frame of the best take, facing +z."""
        requests = [{"id": f"hub.{name}.{k}", "prompt": prompt, "frames": cat.HUB_FRAMES, "pins": [],
                     "meta": {"kind": name}}
                    for name, prompt in cat.HUBS.items() for k in range(cat.HUB_CANDIDATES)]
        got = self.generate(requests)
        hubs = {}
        for name in cat.HUBS:
            best = None
            for k in range(cat.HUB_CANDIDATES):
                c = got[f"hub.{name}.{k}"]
                speed = np.linalg.norm(np.diff(c["joints"], axis=0), axis=-1).mean(-1) * cat.FPS
                f = int(np.argmin(speed[len(speed) // 3:])) + len(speed) // 3
                score = c["meta"]["tmr"] - 0.05 * speed[f]
                if best is None or score > best[0]:
                    best = (score, c, f)
            _, c, f = best
            local, root, _ = self.canonical(c["local"][f:f + 1], c["root"][f:f + 1], c["joints"][f:f + 1])
            hubs[name] = {"local": np.repeat(local, PIN, 0), "root": np.repeat(root, PIN, 0),
                          "from": c["meta"]["id"], "frame": f}
        return hubs

    def loops(self, hubs: dict) -> dict:
        """Every loop: candidates, the best cycle of each, then the most distinct passing takes."""
        requests = []
        for name, (stage, prompt, start, speed, keep) in cat.LOOPS.items():
            frames = cat.LOOP_FRAMES_BY_NAME.get(name, cat.LOOP_FRAMES)
            pins = [pin(hubs[start]["local"], hubs[start]["root"], 0)] if start else []
            if speed:
                pins.append(straight_path(frames, speed))
            for k in range(cat.LOOP_CANDIDATES):
                requests.append({"id": f"loop.{name}.{k}", "prompt": prompt, "frames": frames, "pins": pins,
                                 "meta": {"kind": name}})
        got = self.generate(requests)
        loops = {}
        for name, (stage, prompt, start, speed, keep) in cat.LOOPS.items():
            span = (0.6, 3.0, 1.0) if speed else ((1.5, 5.0, 1.5) if name in ("cut_fence", "cut_cable", "pull_cable")
                                                  else (2.0, 6.0, 1.5))
            takes = []
            for k in range(cat.LOOP_CANDIDATES):
                c = got[f"loop.{name}.{k}"]
                heading = self.heading(c["joints"])
                feat = ml.canonical_features(c["joints"][:, self.body], c["root"], heading)
                # A walk or run is searched only where its feet really step: a take that stops stepping glides.
                lo, hi = self.stepping(c["joints"]) if name in cat.STEPPING else (0, len(feat))
                i, j, cost = ml.find_loop(feat[lo:hi], cat.FPS, span[0], span[1], max(0.0, span[2] - lo / cat.FPS))
                if i is None:
                    i, j, cost = lo, min(hi, lo + int(span[0] * cat.FPS) + 1), float("inf")
                else:
                    i, j = i + lo, j + lo
                m = self.measure(c, i, j)
                m.update(loop_m=cost, tmr=c["meta"]["tmr"], tmr_rank=c["meta"]["tmr_rank"])
                if name in cat.STEPPING:
                    m["legs_cm"] = self.leg_lift(c["joints"][i:j])
                if start:
                    m["seam_m"] = self.seam(c["joints"][:PIN], hubs[start]["local"], hubs[start]["root"])
                takes.append({"c": c, "i": i, "j": j, "m": m, "pass": self._passes(m, name),
                              "shape": ml.resampled(feat[i:j], c["root"][i:j])})
            loops[name] = self._pick(takes, keep)
        return loops

    def leg_lift(self, joints: np.ndarray) -> float:
        """How much the lower toe moves up and down, in cm: near zero means the feet do not step."""
        toes = [self.feet[1][0], self.feet[3][0]]
        height = joints[:, toes, 1] - joints[:, :, 1].min(1)[:, None]
        return float(height.std(0).min() * 100)

    def stepping(self, joints: np.ndarray) -> tuple:
        """The longest stretch [lo, hi) where every second of the take still lifts both feet."""
        second = cat.FPS
        alive = np.array([self.leg_lift(joints[max(0, t - second // 2):t + second // 2]) > GATES["legs_cm"]
                          for t in range(len(joints))])
        best, run = (0, 0), None
        for t, ok in enumerate(np.append(alive, False)):
            if ok and run is None:
                run = t
            elif not ok and run is not None:
                best = max(best, (run, t), key=lambda r: r[1] - r[0])
                run = None
        return best

    def _passes(self, m: dict, kind: str = "") -> list:
        """Names of the gates a candidate fails; sprint moves are held to the sprint limits."""
        gates = dict(GATES, **SPRINT_GATES) if kind == "run" or kind.endswith("_run") else GATES
        failed = [k for k in ("seam_m", "loop_m", "slide_cm_s", "jitter_m_s3") if k in m and m[k] > gates[k]]
        if m["tmr_rank"] > GATES["tmr_rank"]:
            failed.append("tmr_rank")
        if m.get("legs_cm", GATES["legs_cm"]) < GATES["legs_cm"]:
            failed.append("legs_cm")
        return failed

    def _pick(self, takes: list, keep: int) -> list:
        """The best passing take, then the ones most unlike those already kept; failing takes only fill a gap."""
        order = sorted(takes, key=lambda t: (len(t["pass"]) > 0, -t["m"]["tmr"] + t["m"]["slide_cm_s"] / 100))
        kept = [order[0]]
        rest = order[1:]
        while len(kept) < keep and rest:
            passing = [t for t in rest if not t["pass"]] or rest
            far = max(passing, key=lambda t: min(ml.clip_distance(t["shape"], k["shape"]) for k in kept))
            kept.append(far)
            rest = [t for t in rest if t is not far]
        for t in kept:
            t["m"]["nearest_m"] = min([ml.clip_distance(t["shape"], k["shape"]) for k in kept if k is not t],
                                      default=None)
        return kept

    def loop_blocks(self, hubs: dict, loops: dict) -> dict:
        """Close every kept cycle into a block; loops that start on a base pose also get their way in."""
        library = {}
        for name, takes in loops.items():
            stage, _, start, _, _ = cat.LOOPS[name]
            for n, t in enumerate(takes):
                c, i, j = t["c"], t["i"], t["j"]
                block = f"{name}.{n}"
                local, root = ml.close_loop(c["local"], c["root"], i, j)
                joints = self.fk(local, root)
                entry = self.save_block(block, local, root, joints, c["contacts"][i:j],
                                        {"kind": name, "stage": stage, "loop": True, "start": {"loop": block},
                                         "next": {"loop": block}, "source": c["meta"]["id"], "cycle": [i, j],
                                         "measures": t["m"], "failed": t["pass"], "prompt": c["meta"]["prompt"]})
                # Closing the cycle bends the root's turn away, so each cycle carries straight on.
                entry["travel"] = dict(self._travel(local, root, joints, c["local"][j:j + 1], c["root"][j:j + 1]),
                                       turn=0.0)
                library[block] = entry
                if start:
                    library[block + ".in"] = self.save_block(
                        block + ".in", c["local"][:i], c["root"][:i], c["joints"][:i], c["contacts"][:i],
                        {"kind": name + "_in", "stage": stage, "loop": False, "start": {"hub": start},
                         "next": {"loop": block}, "source": c["meta"]["id"], "prompt": c["meta"]["prompt"]})
                    library[block + ".in"]["travel"] = self._travel(
                        c["local"][:i], c["root"][:i], c["joints"][:i], c["local"][i:i + 1], c["root"][i:i + 1])
        return library

    def _travel(self, local, root, joints, next_local, next_root) -> dict:
        """Where the next block's first frame sits relative to this block's first frame: (x, z) and turn."""
        first = self.heading(joints[:1])[0]
        nxt = self.fk(next_local, next_root)
        d = ml.rot_y(-first) @ (next_root[0] - root[0])
        return {"x": float(d[0]), "z": float(d[2]), "turn": float(ml.wrap(self.heading(nxt)[0] - first))}

    def port(self, block: str) -> tuple:
        """The first PIN frames of a closed loop, canonical: what exits start on and entries end on."""
        with np.load(os.path.join(self.out, "blocks", block + ".npz")) as b:
            return b["local"][:PIN], b["root"][:PIN]

    def moves(self, hubs: dict, library: dict) -> dict:
        """Every join and reaction: pass one pins the start, pass two also pins the end where pass one arrived."""
        kept = {name: sorted(b for b in library if b.startswith(name + ".") and library[b]["loop"])
                for name in cat.LOOPS}
        jobs = []
        for name, (stage, prompt, start, end, frames) in cat.MOVES.items():
            starts = [("hub", start)] if isinstance(start, str) else [("loop", b) for b in kept[start[1]]]
            copies = 2 if isinstance(start, str) and isinstance(end, str) else 1
            for s, (skind, sname) in enumerate(starts * copies):
                if isinstance(end, tuple):
                    targets = kept[end[1]]
                    tgt = ("loop", targets[s % len(targets)])
                else:
                    tgt = ("hub", end) if end else None
                jobs.append({"name": name, "stage": stage, "prompt": prompt, "frames": frames,
                             "start": (skind, sname), "end": tgt, "slot": s})
        first = []
        for job in jobs:
            s_local, s_root = self._pose(hubs, job["start"])
            job["start_pose"] = (s_local, s_root)
            for k in range(cat.MOVE_CANDIDATES_BY_NAME.get(job["name"], cat.MOVE_CANDIDATES)):
                first.append({"id": f"move.{job['name']}.{job['slot']}.{k}.a", "prompt": job["prompt"],
                              "frames": job["frames"], "pins": [pin(s_local, s_root, 0)],
                              "meta": {"kind": job["name"], "pass": 1}})
        got = self.generate(first)
        second = []
        for job in jobs:
            if job["end"] is None:
                continue
            e_local, e_root = self._pose(hubs, job["end"])
            for k in range(cat.MOVE_CANDIDATES_BY_NAME.get(job["name"], cat.MOVE_CANDIDATES)):
                c = got[f"move.{job['name']}.{job['slot']}.{k}.a"]
                n = job["frames"] - PIN
                yaw = self.heading(c["joints"][n:n + 1])[0]
                p_local, p_root = ml.place(e_local, e_root, c["root"][n, [0, 2]], yaw)
                second.append({"id": f"move.{job['name']}.{job['slot']}.{k}.b", "prompt": job["prompt"],
                               "frames": job["frames"],
                               "pins": [pin(*job["start_pose"], 0), pin(p_local, p_root, n)],
                               "meta": {"kind": job["name"], "pass": 2, "end_pose": [p_local.tolist(),
                                                                                     p_root.tolist()]}})
        got.update(self.generate(second))
        for job in jobs:
            takes = []
            for k in range(cat.MOVE_CANDIDATES_BY_NAME.get(job["name"], cat.MOVE_CANDIDATES)):
                c = got[f"move.{job['name']}.{job['slot']}.{k}.{'b' if job['end'] else 'a'}"]
                n = job["frames"] - PIN if job["end"] else job["frames"]
                m = self.measure(c, 0, n)
                m.update(tmr=c["meta"]["tmr"], tmr_rank=c["meta"]["tmr_rank"],
                         seam_m=self.seam(c["joints"][:PIN], *job["start_pose"]))
                if job["end"]:
                    e_local, e_root = (np.asarray(x, dtype=np.float32) for x in c["meta"]["end_pose"])
                    m["seam_m"] = max(m["seam_m"], self.seam(c["joints"][n:], e_local, e_root))
                takes.append({"c": c, "m": m, "pass": self._passes(m, job["name"]), "n": n})
            best = sorted(takes, key=lambda t: (len(t["pass"]) > 0, -t["m"]["tmr"] + t["m"]["slide_cm_s"] / 100))[0]
            c, n = best["c"], best["n"]
            block = f"{job['name']}.{job['slot']}"
            info = {"kind": job["name"], "stage": job["stage"], "loop": False,
                    "start": {job["start"][0]: job["start"][1]},
                    "next": {job["end"][0]: job["end"][1]} if job["end"] else None, "source": c["meta"]["id"],
                    "measures": best["m"], "failed": best["pass"], "prompt": job["prompt"]}
            library[block] = self.save_block(block, c["local"][:n], c["root"][:n], c["joints"][:n],
                                             c["contacts"][:n], info)
            if job["end"]:
                library[block]["travel"] = self._travel(c["local"][:n], c["root"][:n], c["joints"][:n],
                                                        c["local"][n:n + 1], c["root"][n:n + 1])
        return library

    def _pose(self, hubs: dict, where: tuple) -> tuple:
        """The pinned frames of a base pose or of a loop's first frames."""
        return (hubs[where[1]]["local"], hubs[where[1]]["root"]) if where[0] == "hub" else self.port(where[1])


def main() -> None:
    """Run every stage and write the library description."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--stub-text", action="store_true", help="random text vectors, for checking the pipeline")
    parser.add_argument("--only", nargs="*", help="limit loops and moves to these names, for a smoke run")
    args = parser.parse_args()
    if args.only:
        cat.LOOPS = {k: v for k, v in cat.LOOPS.items() if k in args.only}
        cat.MOVES = {k: v for k, v in cat.MOVES.items() if k in args.only}
    t0 = time.time()
    gen = Generator(args)
    hubs = gen.hubs()
    np.savez(os.path.join(args.out, "hubs.npz"), **{f"{k}_local": v["local"] for k, v in hubs.items()},
             **{f"{k}_root": v["root"] for k, v in hubs.items()})
    library = gen.loop_blocks(hubs, gen.loops(hubs))
    library = gen.moves(hubs, library)
    with open(os.path.join(args.out, "library.json"), "w", encoding="utf-8") as handle:
        json.dump({"model": MODEL, "steps": args.steps, "pin_frames": PIN, "gates": GATES, "sprint_gates": SPRINT_GATES, "fps": cat.FPS,
                   "hubs": {k: {"from": v["from"], "frame": v["frame"]} for k, v in hubs.items()},
                   "blocks": library, "wall_s": time.time() - t0}, handle, indent=1, default=float)
    print(f"done: {len(library)} blocks in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
