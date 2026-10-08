"""Bake the public road the farmers' pickup drives into maps/custom/solar/movers/public_road.json.

    python bake.py ../../swarm_worlds/maps/custom/solar

The road is the surveyed centre line the map's ground was painted along: a north arm down the east side of the park
and an east arm, meeting at a fork south-east of it. Both arms join the stem that runs west from the fork into the
yard outside the south tip of the fence, the one place a pickup can turn round, which it does in three moves.

Every sample is where the pickup's rear axle stands, with the body's pose resting on the terrain under its four
wheels, facing along the line and facing back, so the simulator only replays poses and never queries the ground.
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

# The public road's centre line, surveyed from aerial imagery and registered to the park's rows (map metres, east and
# north), from the north end, down the park's east side to the fork, and out along the east arm.
SURVEY = [[105.495, 142.274], [98.739, 131.192], [92.412, 119.202], [88.106, 107.168], [86.093, 97.329],
          [85.228, 88.589], [82.044, 76.53], [75.932, 64.087], [68.937, 52.336], [61.941, 40.586], [55.39, 28.601],
          [51.182, 21.057], [39.356, -11.787], [39.742, -23.302], [39.009, -37.025], [37.177, -49.194],
          [36.231, -61.366], [32.398, -75.074], [31.452, -87.247], [27.18, -100.289], [22.005, -105.951],
          [15.585, -113.987], [9.622, -121.578], [5.469, -128.741], [8.591, -131.489], [18.11, -128.004],
          [32.589, -124.584], [47.033, -123.869], [55.198, -120.367], [63.493, -106.942], [73.615, -91.737],
          [80.913, -85.516], [93.19, -78.008], [104.525, -73.644], [115.361, -72.883], [127.503, -75.748],
          [139.165, -80.862], [151.289, -85.079], [164.79, -87.51], [176.968, -87.669]]
# The stem from the fork west into the yard, traced on the painted ground, ending where the yard is widest.
STEM = [[4.0, -129.4], [0.0, -128.8], [-4.0, -127.6], [-8.0, -125.6], [-12.0, -123.2], [-16.0, -120.0],
        [-19.2, -116.5], [-22.2, -112.5], [-22.6, -108.5], [-22.3, -104.5]]

STEM_BLUR = 8.0            # the traced stem has few points, so its corners are eased harder than the survey's
SPACING_M = 0.5            # distance between samples along every line
DENSE_M = 0.05             # step of the dense curves samples are cut from
ROAD_RADIUS_M = 8.0        # tightest bend a join between an arm and the stem may have
TURN_RADIUS_M = 5.5        # rear axle radius at full lock, a pickup's turning circle of about 12.5 m
FENCE_CLEAR_M = 1.0        # the truck's outline keeps this far outside the fence
TREE_CLEAR_M = 0.3         # and this far from any tree crown while it turns in the yard


def _ring(posts: np.ndarray) -> np.ndarray:
    """The fence posts in the order the fence joins them, each to its nearest unvisited neighbour."""
    order, left = [0], set(range(1, len(posts)))
    while left:
        last = posts[order[-1]]
        nearest = min(left, key=lambda k: (float(np.hypot(*(posts[k] - last))), k))
        order.append(nearest)
        left.remove(nearest)
    return posts[order]


def _inside(ring: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which points lie inside the closed ring, by counting the sides a ray from each crosses."""
    a, b = ring, np.roll(ring, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return crosses.sum(axis=1) % 2 == 1


def _ring_gap(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Each point's distance to the nearest side of the closed ring."""
    side = np.roll(ring, -1, axis=0) - ring
    rel = points[:, None, :] - ring[None, :, :]
    t = np.clip((rel * side).sum(axis=2) / (side * side).sum(axis=1), 0.0, 1.0)
    return np.min(np.linalg.norm(rel - t[..., None] * side, axis=2), axis=1)


class Terrain:
    """The ground the simulator collides with, as the regular height grids of the map's core and near tiles."""

    def __init__(self, solar: str, manifest: dict):
        """Read every core and near tile's mesh into a height grid keyed by its placement."""
        self.tiles = []
        for place in manifest["placements"]:
            item = manifest["items"][place["item"]]
            if item["group"] != "terrain" or item.get("collision") != "mesh":
                continue
            if place["quaternion"] != [0.0, 0.0, 0.0, 1.0] or place["scale"] != [1.0, 1.0, 1.0]:
                raise ValueError(f"terrain tile {place['item']} is turned or scaled")
            with open(os.path.join(solar, item["folder"], item["obj"]), encoding="utf-8") as handle:
                verts = np.array([line.split()[1:4] for line in handle if line.startswith("v ")], dtype=float)
            xs, ys = np.unique(verts[:, 0]), np.unique(verts[:, 1])
            grid = np.full((len(ys), len(xs)), np.nan)
            grid[np.searchsorted(ys, verts[:, 1]), np.searchsorted(xs, verts[:, 0])] = verts[:, 2]
            ox, oy, oz = place["position"]
            self.tiles.append((xs + ox, ys + oy, grid + oz))

    def height(self, points: np.ndarray) -> np.ndarray:
        """Ground height under each point, on the triangle that holds it; each cell is split from its low corner to
        its high corner, as the tiles are."""
        out = np.full(len(points), np.nan)
        for xs, ys, grid in self.tiles:
            hit = (points[:, 0] >= xs[0]) & (points[:, 0] <= xs[-1]) & (points[:, 1] >= ys[0]) & (points[:, 1] <= ys[-1])
            hit &= np.isnan(out)
            if not hit.any():
                continue
            px, py = points[hit, 0], points[hit, 1]
            i = np.clip(np.searchsorted(xs, px) - 1, 0, len(xs) - 2)
            j = np.clip(np.searchsorted(ys, py) - 1, 0, len(ys) - 2)
            u = (px - xs[i]) / (xs[i + 1] - xs[i])
            v = (py - ys[j]) / (ys[j + 1] - ys[j])
            z00, z10, z01, z11 = grid[j, i], grid[j, i + 1], grid[j + 1, i], grid[j + 1, i + 1]
            lower = u >= v
            out[hit] = np.where(lower, z00 + u * (z10 - z00) + v * (z11 - z10), z00 + v * (z01 - z00) + u * (z11 - z01))
        if np.isnan(out).any():
            raise ValueError("a road point lies off the map's collision terrain")
        return out


class Pickup:
    """The pickup's wheels and outline, read from its placements, and how its body rests on the ground."""

    def __init__(self, manifest: dict):
        """Take the axles, the track and the wheel radius from the wheels, the outline from the body's bounds."""
        wheels = [p for p in manifest["placements"] if p["item"].startswith("pickup_wheel")]
        if len(wheels) != 4:
            raise ValueError("the pickup needs four wheel placements")
        local = np.array([w["local_position"] for w in wheels])
        self.front = float(np.mean(local[local[:, 0] > 0, 0]))
        self.rear = float(np.mean(local[local[:, 0] < 0, 0]))
        self.track = float(np.mean(np.abs(local[:, 1])))
        self.axle_z = float(np.mean(local[:, 2]))
        self.radius = float(np.mean([manifest["items"][w["item"]]["bounds_max"][2] for w in wheels]))
        body = manifest["items"]["pickup_body"]
        self.low, self.high = np.array(body["bounds_min"][:2]), np.array(body["bounds_max"][:2])

    @property
    def wheelbase(self) -> float:
        """Distance between the axles."""
        return self.front - self.rear

    def outline(self, rear: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """The body's four corners seen from above, for each rear axle point and heading, shape (n, 4, 2)."""
        c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
        xs = np.array([self.low[0], self.high[0], self.high[0], self.low[0]]) - self.rear
        ys = np.array([self.low[1], self.low[1], self.high[1], self.high[1]])
        return np.stack([rear[:, :1] + c * xs - s * ys, rear[:, 1:] + s * xs + c * ys], axis=2)

    def poses(self, terrain: Terrain, rear: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """The body pose (x, y, z, qx, qy, qz, qw) standing on the ground and the suspension's twist, for each rear
        axle point and heading.

        The body takes the plane fitted through the ground under its four wheels; its origin sits on the axles' line,
        a wheel radius above that plane. Four points never lie on one plane on rough ground, and what the plane misses
        is the same height at every wheel, up at the front left and rear right and down at the other two: the twist,
        which the suspension takes by moving each wheel that much along the body's vertical.
        """
        c, s = np.cos(heading), np.sin(heading)
        corners = [(self.front, self.track), (self.front, -self.track), (self.rear, self.track), (self.rear, -self.track)]
        contact = np.stack([np.stack([rear[:, 0] + c * (lx - self.rear) - s * ly,
                                      rear[:, 1] + s * (lx - self.rear) + c * ly], axis=1) for lx, ly in corners], axis=1)
        z = terrain.height(contact.reshape(-1, 2)).reshape(-1, 4)
        # The plane z = a + b x + c y through the four contacts, least squares in the body's own axes.
        lx = np.array([self.front, self.front, self.rear, self.rear])
        ly = np.array([self.track, -self.track, self.track, -self.track])
        design = np.stack([np.ones(4), lx, ly], axis=1)
        a, b, cc = (np.linalg.pinv(design) @ z.T)
        forward = np.stack([c, s, b], axis=1)
        side = np.stack([-s, c, cc], axis=1)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        up = np.cross(forward, side)
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        side = np.cross(up, forward)
        origin = np.stack([rear[:, 0] - c * self.rear, rear[:, 1] - s * self.rear, a], axis=1)
        origin += up * (self.radius - self.axle_z)
        twist = (z[:, 0] - z[:, 1] - z[:, 2] + z[:, 3]) / 4.0
        return np.concatenate([origin, _quaternions(np.stack([forward, side, up], axis=2)), twist[:, None]], axis=1)


def _quaternions(rot: np.ndarray) -> np.ndarray:
    """Unit quaternions (x, y, z, w) of rotation matrices whose columns are the body's axes in the world."""
    out = np.empty((len(rot), 4))
    for k, m in enumerate(rot):
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 2.0 * math.sqrt(trace + 1.0)
            q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            j, k2 = (i + 1) % 3, (i + 2) % 3
            s = 2.0 * math.sqrt(1.0 + m[i, i] - m[j, j] - m[k2, k2])
            q = [0.0, 0.0, 0.0, (m[k2, j] - m[j, k2]) / s]
            q[i] = 0.25 * s
            q[j] = (m[j, i] + m[i, j]) / s
            q[k2] = (m[k2, i] + m[i, k2]) / s
        out[k] = q if q[3] >= 0 else [-v for v in q]
    return out


def _painted(points: list, blur: float = 1.8) -> np.ndarray:
    """A centre line through the points as the map's road was painted along its survey: a shape-keeping spline every
    0.4 m, eased by a blur of that many samples."""
    raw = np.asarray(points)
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    t = np.r_[np.arange(0.0, along[-1], 0.4), along[-1]]
    line = PchipInterpolator(along, raw, axis=0)(t)
    line[1:-1] = gaussian_filter1d(line, sigma=blur, axis=0, mode="nearest")[1:-1]
    return line


def _resample(line: np.ndarray, step: float) -> np.ndarray:
    """The polyline cut into points step metres apart along it, counted from its last point, keeping both ends."""
    line = line[::-1]
    along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    t = np.r_[np.arange(0.0, along[-1], step), along[-1]]
    if along[-1] - t[-2] < step * 0.5:
        t = np.r_[t[:-2], along[-1]]
    return np.stack([np.interp(t, along, line[:, 0]), np.interp(t, along, line[:, 1])], axis=1)[::-1]


def _heading(line: np.ndarray) -> np.ndarray:
    """Direction of travel at every point of a polyline, radians."""
    d = np.gradient(line, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _curvature(line: np.ndarray) -> np.ndarray:
    """Signed curvature at every point of an evenly spaced polyline, positive turning left."""
    heading = np.unwrap(_heading(line))
    step = np.linalg.norm(np.diff(line, axis=0), axis=1).mean()
    return np.gradient(heading) / step


def _bezier(p0, t0, p3, t3, k0: float, k3: float) -> np.ndarray:
    """A cubic join leaving p0 along t0 and arriving at p3 along t3, densely sampled."""
    p1, p2 = p0 + k0 * t0, p3 - k3 * t3
    u = np.linspace(0.0, 1.0, 400)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3


def _join(arm: np.ndarray, stem: np.ndarray) -> np.ndarray:
    """The arm, a smooth join and the stem as one dense line: of every join from a point on the arm before the fork
    to a point on the stem, the one that keeps nearest the painted road while no bend is tighter than ROAD_RADIUS_M."""
    arm_along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(arm, axis=0), axis=1))]
    stem_along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(stem, axis=0), axis=1))]
    painted = np.concatenate([arm[-100:], stem[:600]])
    # The join leaves the arm before its last bend too tight to drive, the fork's own corner included.
    tight = np.flatnonzero(np.abs(_curvature(arm)) > 1.0 / ROAD_RADIUS_M)
    least = arm_along[-1] - arm_along[tight[0] - 1] if len(tight) else 2.0
    best = None
    for back in np.arange(least, least + 28.0, 1.0):
        i = int(np.searchsorted(arm_along, arm_along[-1] - back))
        p0, t0 = arm[i], arm[min(i + 1, len(arm) - 1)] - arm[max(i - 1, 0)]
        t0 = t0 / np.linalg.norm(t0)
        for ahead in np.arange(2.0, 20.0, 1.0):
            j = int(np.searchsorted(stem_along, ahead))
            p3, t3 = stem[j], stem[min(j + 1, len(stem) - 1)] - stem[max(j - 1, 0)]
            t3 = t3 / np.linalg.norm(t3)
            chord = float(np.linalg.norm(p3 - p0))
            for k in (0.3, 0.4, 0.5):
                curve = _bezier(p0, t0, p3, t3, k * chord, k * chord)
                bend = float(np.max(np.abs(_curvature(_resample(curve, DENSE_M * 4)))))
                if bend > 1.0 / ROAD_RADIUS_M:
                    continue
                off = float(np.max(np.min(np.linalg.norm(curve[:, None] - painted[None], axis=2), axis=1)))
                if best is None or off < best[0]:
                    best = (off, np.concatenate([arm[:i], curve, stem[j + 1:]]))
    if best is None:
        raise ValueError("no join between the arm and the stem keeps to the road")
    return best[1]


def _wrap(angle: float) -> float:
    """An angle brought into [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _ends(x: float, y: float, h: float, side: float, radius: float, lengths) -> tuple:
    """Where the rear axle stands and heads after arcs of the given signed lengths, steering to side first and
    alternating."""
    steer = side / radius
    for length in lengths:
        h2 = h + length * steer
        x, y, h = x + (math.sin(h2) - math.sin(h)) / steer, y - (math.cos(h2) - math.cos(h)) / steer, h2
        steer = -steer
    return x, y, h


def _turn(tail: np.ndarray, blocked) -> tuple:
    """Three moves that turn the pickup round in the yard: forward on full lock, back on the other lock, forward
    again, ending on a sample of the yard's line facing out.

    Returns the tail samples the turn starts and ends on and the dense rear axle path as rows of (x, y, heading,
    gear, curvature). Of all turns whose outline stays clear, the shortest.
    """
    heading = np.unwrap(_heading(tail))
    r, top = TURN_RADIUS_M, math.pi * TURN_RADIUS_M
    best = None
    for start in range(len(tail) // 3, len(tail), 2):
        x0, y0, h0 = float(tail[start, 0]), float(tail[start, 1]), float(heading[start])
        for side in (1.0, -1.0):
            for first in np.arange(1.0, 0.75 * top, 1.0):

                def across(v: np.ndarray) -> list:
                    """How far the end lies off the yard's line and how far it heads off facing back along it."""
                    x, y, h = _ends(x0, y0, h0, side, r, (first, -v[0], v[1]))
                    k = int(np.argmin(np.hypot(tail[:, 0] - x, tail[:, 1] - y)))
                    off = (x - tail[k, 0]) * -math.sin(heading[k]) + (y - tail[k, 1]) * math.cos(heading[k])
                    return [off, _wrap(h - heading[k] - math.pi) * r]

                rough = least_squares(across, [top / 3, top / 3], bounds=([0.1, 0.1], [top, top]))
                if max(map(abs, rough.fun)) > 0.05:
                    continue
                x, y, _ = _ends(x0, y0, h0, side, r, (first, -rough.x[0], rough.x[1]))
                end = int(np.argmin(np.hypot(tail[:, 0] - x, tail[:, 1] - y)))

                def onto(v: np.ndarray) -> list:
                    """How far the end lies from the chosen sample and how far it heads off facing back there."""
                    x, y, h = _ends(x0, y0, h0, side, r, (v[0], -v[1], v[2]))
                    return [x - tail[end, 0], y - tail[end, 1], _wrap(h - heading[end] - math.pi) * r]

                exact = least_squares(onto, [first, *rough.x], bounds=([0.1] * 3, [top] * 3), xtol=1e-12, ftol=1e-12)
                if max(map(abs, exact.fun)) > 1e-4:
                    continue
                length = float(sum(exact.x))
                if best is not None and length >= best[0]:
                    continue
                path = _arcs((x0, y0), h0, side, r, (exact.x[0], -exact.x[1], exact.x[2]))
                if not blocked(path[:, :2], path[:, 2]):
                    best = (length, start, end, path)
    if best is None:
        raise ValueError("the pickup cannot turn round in the yard")
    return best[1], best[2], best[3]


def _arcs(origin, heading: float, side: float, radius: float, lengths: tuple) -> np.ndarray:
    """Rear axle path along arcs of the given signed lengths (negative backs up), steering to side first and
    alternating; rows of (x, y, heading, gear, curvature)."""
    rows = []
    x, y, h = float(origin[0]), float(origin[1]), float(heading)
    steer = side
    for length in lengths:
        gear = 1.0 if length > 0 else -1.0
        n = max(int(math.ceil(abs(length) / DENSE_M)), 1)
        ds = length / n
        for _ in range(n):
            rows.append((x, y, h, gear, steer / radius))
            h_next = h + ds * steer / radius
            x += radius / steer * (math.sin(h_next) - math.sin(h))
            y -= radius / steer * (math.cos(h_next) - math.cos(h))
            h = h_next
        steer = -steer
    rows.append((x, y, h, rows[-1][3], rows[-1][4]))
    return np.array(rows)


def main() -> None:
    """Bake the road, check it, and write it next to the pickup's other tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("solar", help="the solar map folder, maps/custom/solar")
    args = parser.parse_args()
    with open(os.path.join(args.solar, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    terrain, pickup = Terrain(args.solar, manifest), Pickup(manifest)
    ring = _ring(np.array([p["position"][:2] for p in manifest["placements"] if p["item"] == "fence_post"], dtype=float))
    forest = manifest["forest"]
    with np.load(os.path.join(args.solar, forest["folder"], forest["table"])) as table:
        trees = np.concatenate([table["position"][:, :2], np.max(table["scale"][:, :2], axis=1, keepdims=True)], axis=1)

    def blocked(rear: np.ndarray, heading: np.ndarray, margin: float = TREE_CLEAR_M) -> bool:
        """Whether the pickup's outline anywhere along the path comes near the fence, or within margin of a crown."""
        outline = pickup.outline(rear, heading)
        points = np.concatenate([outline, (outline + np.roll(outline, -1, axis=1)) / 2], axis=1).reshape(-1, 2)
        if _inside(ring, points).any() or _ring_gap(points, ring).min() < FENCE_CLEAR_M:
            return True
        near = trees[np.min(np.linalg.norm(trees[:, None, :2] - rear[None, ::20], axis=2), axis=1) < 12.0]
        for corners in outline:
            centre = corners.mean(axis=0)
            axis_x = (corners[1] - corners[0]) / np.linalg.norm(corners[1] - corners[0])
            axis_y = np.array([-axis_x[1], axis_x[0]])
            rel = near[:, :2] - centre
            half = np.array([np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[0])]) / 2
            gap = np.maximum(np.abs(np.stack([rel @ axis_x, rel @ axis_y], axis=1)) - half, 0.0)
            if (np.linalg.norm(gap, axis=1) < near[:, 2] + margin).any():
                return True
        return False

    painted = _painted(SURVEY)
    fork = int(np.argmax(np.abs(_curvature(painted))))
    stem = _resample(_painted(STEM, STEM_BLUR), DENSE_M)
    # Each line starts a truck's length in from the road's end, so a truck leaving the map never stands in the forest.
    reach = int(math.ceil((pickup.high[0] - pickup.rear) / SPACING_M))
    lines = {"north": _resample(_join(painted[:fork + 1], stem), SPACING_M)[reach:],
             "east": _resample(_join(painted[fork:][::-1], stem), SPACING_M)[reach:]}
    # Both lines end on the same stretch of the stem; the yard is the part of it they share sample for sample.
    shared = 0
    while shared < min(map(len, lines.values())) and np.allclose(lines["north"][-1 - shared], lines["east"][-1 - shared], atol=1e-6):
        shared += 1
    tail = lines["north"][-shared:]
    start, end, path = _turn(tail, blocked)
    turn = _resample_turn(path)
    out = {"about": "the public road outside the solar park's fence, baked by tools/road/bake.py: rear axle samples "
                    f"{SPACING_M} m apart; a line row is x, y, curvature, then the pickup's body pose and suspension "
                    "twist facing along the line, then both facing back; a turn row is x, y, gear, curvature, pose, twist",
           "spacing_m": SPACING_M, "wheelbase_m": round(pickup.wheelbase, 4), "wheel_radius_m": round(pickup.radius, 4),
           "yard_samples": shared, "turn_from": shared - start, "turn_to": shared - end, "lines": {}}
    for name, line in lines.items():
        heading = _heading(line)
        ahead, back = pickup.poses(terrain, line, heading), pickup.poses(terrain, line, heading + math.pi)
        out["lines"][name] = np.round(np.concatenate([line, _curvature(line)[:, None], ahead, back], axis=1), 5).tolist()
    poses = pickup.poses(terrain, turn[:, :2], turn[:, 2])
    out["turn"] = np.round(np.concatenate([turn[:, :2], turn[:, 3:5], poses], axis=1), 5).tolist()
    # The first yard samples face a little differently on each arm, as a heading reads its neighbours; the yard is
    # only the samples the two lines agree on, written once so both hold them identically, and the turn lies inside it.
    north, east = out["lines"]["north"], out["lines"]["east"]
    same = 0
    while same < shared and np.allclose(north[-1 - same], east[-1 - same], rtol=0.0, atol=1e-4):
        same += 1
    if max(out["turn_from"], out["turn_to"]) >= same:
        raise ValueError("the turn reaches samples the two arms do not share")
    east[-same:] = north[-same:]
    out["yard_samples"] = same
    for name, line in lines.items():
        heading = _heading(line)
        if blocked(line, heading, 0.0) or blocked(line, heading + math.pi, 0.0):
            raise ValueError(f"the pickup on the {name} line touches a tree crown or comes near the fence")
    with open(os.path.join(args.solar, "movers", "public_road.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, separators=(",", ":"))
    print({name: round(len(line) * SPACING_M, 1) for name, line in lines.items()}, "yard", shared * SPACING_M,
          "turn", round(len(turn) * SPACING_M, 1), "m")


def _resample_turn(path: np.ndarray) -> np.ndarray:
    """The turn's dense rows cut every SPACING_M of travel within each move; each move keeps both its ends, and the
    point where the gear changes is kept once, as the last row of the move it ends."""
    pieces = np.split(path, np.flatnonzero(np.diff(path[:, 3]) != 0) + 1)
    rows = []
    for number, piece in enumerate(pieces):
        if number + 1 < len(pieces):
            piece = np.concatenate([piece, np.c_[pieces[number + 1][:1, :3], piece[-1:, 3:]]])
        along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1))]
        t = np.r_[np.arange(0.0, along[-1], SPACING_M), along[-1]]
        cut = np.stack([np.interp(t, along, piece[:, k]) for k in range(3)] + [np.full(len(t), piece[-1, 3]),
                                                                              np.full(len(t), piece[-1, 4])], axis=1)
        rows.append(cut if not rows else cut[1:])
    return np.concatenate(rows)


if __name__ == "__main__":
    main()
