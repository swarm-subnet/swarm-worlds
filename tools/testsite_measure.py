"""Measure exported maps with the Sentinel colour and thermal camera functions on four cores."""
# One-shot build tool, not installed with this package: it needs numpy, Pillow, shapely, pybullet (swarm-bullet3) and the swarm package from the simulator repository.
import argparse
import functools
import hashlib
import json
import math
import operator
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
from PIL import Image
from shapely.geometry import Polygon
from swarm.challenge_families.solar_patrol import camera, park
from swarm.core.daylight import daylight_render_kwargs, seeded_sun
from swarm.core.maps.solar.builder import build_solar_map, solar_fence
from swarm.core.maps.testsite.builder import build_testsite_map
from swarm.core.sky_pack import load_sky_pack, pick_sky


def main():
    """Build one map, warm each camera, then measure thirty distinct camera poses per view."""
    arg = argparse.ArgumentParser()
    arg.add_argument('kind', choices=['testsite', 'solar'])
    arg.add_argument('assets')
    arg.add_argument('out')
    args = arg.parse_args()
    args.assets = str(Path(args.assets).resolve())
    os.environ['SWARM_RENDER_THREADS'] = '4'
    cli = p.connect(p.DIRECT)
    start = time.perf_counter()
    world = (build_testsite_map if args.kind == 'testsite' else build_solar_map)(seed=0, cli=cli, asset_dir=args.assets)
    build = time.perf_counter() - start
    if args.kind == 'testsite':
        ring = np.array(json.load(open(Path(args.assets) / 'manifest.json'))['site']['field_outline'])
    else:
        ring = solar_fence(args.assets)
    centre = np.array(Polygon(ring).centroid.coords[0])
    ground = p.rayTest([*centre, 1600], [*centre, -1000], physicsClientId=cli)[0][3][2]
    sun = seeded_sun(0)
    photo = pick_sky(0, sun, load_sky_pack())
    texture = p.loadTexture(photo[0].path, physicsClientId=cli) if photo else None
    def daylight(unused):
        """Use the environment's daylight settings and sky photograph."""
        kw = daylight_render_kwargs(sun)
        if photo:
            kw.update(skyTextureId=texture, skyYaw=photo[1])
        return kw
    env = SimpleNamespace(CLIENT=cli, _sun=sun, _render_flags=p.ER_SWARM_RAYCAST, _sky_flags=p.ER_SWARM_SKY_SUN,
                          _light_direction=list(sun.direction), _sky_kwargs=lambda: {}, _daylight_kwargs=daylight,
                          _daylight_flags=functools.reduce(operator.or_, [getattr(p, n) for n in
                          ['ER_SWARM_DAYLIGHT', 'ER_SWARM_SHADOW_MAP', 'ER_SWARM_MOVER_SHADOW', 'ER_EDGE_ANTIALIAS',
                           'ER_ALPHA_CUTOUT', 'ER_TEXTURE_FILTER', 'ER_SPECULAR_GLINT', 'ER_SWARM_LINEAR_LIGHT']]))
    ep = SimpleNamespace(airframe=None, seed=0, camera={'captures': 0}, park=park.heat(sun, 0, 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'engine_sha256': hashlib.file_digest(open(p.__file__, 'rb'), 'sha256').hexdigest(), 'build_s': build, 'size_MB': sum(f.stat().st_size for f in Path(args.assets).rglob('*') if f.is_file()) / 1e6,
              'disk_MiB': sum(f.stat().st_blocks * 512 for f in [Path(args.assets), *Path(args.assets).rglob('*')]) / 1048576,
              'trees': world['trees'], 'bodies': world['body_count'], 'eye': [*centre, ground + 20], 'frames': {}}
    frames = 0
    for tilt in [90, 45, 15]:
        angle = math.radians(tilt)
        for feed in ['colour', 'thermal']:
            times = []
            for i in range(32):
                # Millimetre movements invalidate frame reuse, as a flying drone does.
                eye = (float(centre[0]) + (i % 2) * .002, float(centre[1]), ground + 20)
                width, height, diag = (640, 480, 82) if feed == 'colour' else (640, 512, 45)
                shot = camera.View(feed, eye, (0., math.cos(angle), -math.sin(angle)), (0., math.sin(angle), math.cos(angle)),
                                   width, height, camera.vertical_fov_deg(diag, width, height), True, i)
                ep.camera['captures'] = i
                start = time.perf_counter()
                frame, _ = camera.colour_frame(env, shot) if feed == 'colour' else camera._thermal_frame(env, ep, shot)
                elapsed = (time.perf_counter() - start) * 1000
                frames += 1
                if frames == 60:
                    result["peak_after_60_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if i >= 2:
                    times.append(elapsed)
            result['frames'][f'{feed}_{tilt}'] = float(np.median(times))
            if feed == 'colour':
                Image.fromarray(np.uint8(np.clip(frame * 255, 0, 255))).save(str(out) + f'_{tilt}.png')
            print(feed, tilt, result['frames'][f'{feed}_{tilt}'], flush=True)
    result['peak_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json.dump(result, open(str(out) + '.json', 'w'), indent=1)
    print(json.dumps(result), flush=True)
    p.disconnect(cli)


if __name__ == '__main__':
    main()
