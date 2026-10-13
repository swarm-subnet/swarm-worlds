"""Numpy helpers shared by the generator and the baker: headings, canonical frames, loops and quality measures.

Kimodo's frame: y up, metres, a body facing +z has heading 0, and heading a is a turn of the body by R_y(a).
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def rot_y(angle: float) -> np.ndarray:
    """Rotation matrix about +y by angle radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def heading_of(cos_sin: np.ndarray) -> np.ndarray:
    """Heading angles from Kimodo's per-frame (cos, sin) pairs."""
    return np.arctan2(cos_sin[..., 1], cos_sin[..., 0])


def wrap(angle):
    """Angle wrapped to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def reframe(local: np.ndarray, root: np.ndarray, joints: np.ndarray, origin: np.ndarray, yaw: float) -> tuple:
    """Move a clip so that origin (x, z) becomes (0, 0) and turn it by -yaw about +y.

    local: (T, J, 3, 3) local rotations, joint 0 the root; root: (T, 3); joints: (T, J, 3) world positions.
    """
    r = rot_y(-yaw)
    shift = np.array([origin[0], 0.0, origin[1]])
    local = local.copy()
    local[:, 0] = np.einsum("ij,tjk->tik", r, local[:, 0])
    root = (root - shift) @ r.T
    joints = (joints - shift) @ r.T
    return local, root, joints


def place(local: np.ndarray, root: np.ndarray, xz, yaw: float) -> tuple:
    """Put a canonical chunk (starting at the origin, heading 0) at xz with heading yaw."""
    r = rot_y(yaw)
    local = local.copy()
    local[:, 0] = np.einsum("ij,tjk->tik", r, local[:, 0])
    root = root @ r.T + np.array([xz[0], 0.0, xz[1]])
    return local, root


def canonical_features(joints: np.ndarray, root: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Per-frame joint positions relative to the root's ground point, turned so every frame faces +z."""
    rel = joints - np.stack([root[:, 0], np.zeros(len(root)), root[:, 2]], 1)[:, None]
    c, s = np.cos(-heading), np.sin(-heading)
    x = rel[..., 0] * c[:, None] + rel[..., 2] * s[:, None]
    z = -rel[..., 0] * s[:, None] + rel[..., 2] * c[:, None]
    return np.stack([x, rel[..., 1], z], -1)


def find_loop(feat: np.ndarray, fps: int, shortest_s: float, longest_s: float, skip_s: float) -> tuple:
    """Best cycle [i, j): the pair of frames whose pose and joint velocities match best.

    Returns (i, j, cost) with cost the root-mean-square joint mismatch in metres plus a tenth of a second of velocity
    mismatch.
    """
    t = len(feat)
    flat = feat.reshape(t, -1)
    vel = np.gradient(flat, axis=0) * fps
    n_joints = feat.shape[1]
    pos_d = np.sqrt(np.maximum(((flat[:, None] - flat[None]) ** 2).sum(-1) / n_joints, 0.0))
    vel_d = np.sqrt(np.maximum(((vel[:, None] - vel[None]) ** 2).sum(-1) / n_joints, 0.0))
    cost = pos_d + 0.1 * vel_d
    lo, hi, skip = int(shortest_s * fps), int(longest_s * fps), int(skip_s * fps)
    best = (None, None, np.inf)
    for i in range(skip, t - lo):
        js = np.arange(i + lo, min(i + hi, t - 1) + 1)
        if len(js) == 0:
            continue
        k = int(np.argmin(cost[i, js]))
        if cost[i, js[k]] < best[2]:
            best = (i, int(js[k]), float(cost[i, js[k]]))
    return best


def close_loop(local: np.ndarray, root: np.ndarray, i: int, j: int) -> tuple:
    """Frames [i, j) bent so that frame j would equal frame i exactly: the mismatch of every joint rotation and of the
    root height is spread evenly over the cycle. Returns (local, root) of the cycle."""
    seg, seg_root = local[i:j].copy(), root[i:j].copy()
    n = j - i
    t = np.arange(n) / n
    for joint in range(local.shape[1]):
        residual = Rotation.from_matrix(local[j, joint] @ local[i, joint].T)
        undo = Slerp([0.0, 1.0], Rotation.concatenate([Rotation.identity(), residual.inv()]))(t)
        seg[:, joint] = (undo * Rotation.from_matrix(seg[:, joint])).as_matrix()
    seg_root[:, 1] -= t * (root[j, 1] - root[i, 1])
    return seg, seg_root


def foot_slide(joints: np.ndarray, contacts: np.ndarray, feet: list, fps: int) -> float:
    """Mean ground-plane speed in cm/s of the heel and toe joints on frames Kimodo marks them in contact."""
    speeds = []
    for joint, channel in feet:
        v = np.linalg.norm(np.diff(joints[:, joint, [0, 2]], axis=0), axis=1) * fps
        on = contacts[1:, channel] & contacts[:-1, channel]
        speeds.append(v[on])
    speeds = np.concatenate(speeds)
    return float(speeds.mean() * 100) if len(speeds) else 0.0


def jitter(joints: np.ndarray, fps: int) -> float:
    """95th percentile of joint jerk in m/s^3, the measure of shaking and pops."""
    jerk = np.diff(joints, 3, axis=0) * fps ** 3
    return float(np.percentile(np.linalg.norm(jerk, axis=-1), 95))


def resampled(feat: np.ndarray, root: np.ndarray, samples: int = 60) -> np.ndarray:
    """A clip as a fixed-length sequence: body pose plus the ground path, for comparing clips of any length."""
    idx = np.linspace(0, len(feat) - 1, samples)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(feat) - 1)
    w = (idx - lo)[:, None, None]
    body = feat[lo] * (1 - w) + feat[hi] * w
    path = root[lo] * (1 - w[:, 0]) + root[hi] * w[:, 0]
    path = path - path[:1]
    path[:, 1] = 0.0
    return body + path[:, None]


def clip_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean joint distance in metres between two resampled clips."""
    return float(np.linalg.norm(a - b, axis=-1).mean())
