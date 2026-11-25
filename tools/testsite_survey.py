"""Derive site metadata from exported geometry and prove its field outline against collision rays."""
# One-shot build tool, not installed with this package: it needs numpy, scipy, shapely and pybullet (swarm-bullet3).
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.optimize import minimize
from shapely import polygons, union_all
from shapely.geometry import Point, Polygon


def vertices(path):
    """Read the OBJ vertex positions without loading renderer state."""
    return np.array([list(map(float, line.split()[1:4])) for line in path.open() if line.startswith('v ')])


def main():
    """Write crown and house metadata and a measured field alignment report."""
    root, research, output = map(Path, sys.argv[1:])
    manifest = json.load(open(root / 'manifest.json'))
    field = Polygon(manifest['site']['field_outline'])
    table = dict(np.load(root / 'plants/forest.npz'))
    bounds = []
    for mesh in manifest['forest']['meshes']:
        v = vertices(root / 'plants' / mesh)
        bounds.append((v.min(0), v.max(0)))
    crowns, tops = [], []
    p.connect(p.DIRECT)
    item = manifest['items']['field_soil']
    path = root / item['folder'] / item['obj']
    shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(path.resolve()), flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    body = p.createMultiBody(0, shape)
    for i, mesh in enumerate(table['mesh']):
        if table['tier'][i] not in (0, 1, 2):
            continue
        low, high = [v * table['scale'][i] for v in bounds[mesh]]
        yaw = table['yaw'][i]
        c, s = np.cos(yaw), np.sin(yaw)
        turn = np.array([[c, -s], [s, c]])
        xy = table['position'][i, :2] + turn @ ((low[:2] + high[:2]) / 2)
        # The circumscribed radius encloses every rotated crown corner, including asymmetric masters.
        radius = float(np.linalg.norm((high[:2] - low[:2]) / 2))
        if field.distance(Point(xy)) > radius + 10:
            continue
        crowns.append([round(float(xy[0]), 4), round(float(xy[1]), 4), round(radius, 4)])
        q = np.array(field.exterior.interpolate(field.exterior.project(Point(xy))).coords[0]) if not field.covers(Point(xy)) else xy
        hit = p.rayTest([*q, 1200], [*q, 800])[0]
        if hit[0] != body:
            raise RuntimeError('No field under crown projection')
        tops.append(float(table['position'][i, 2] + high[2] - hit[3][2]))
    manifest['site']['passable'] = crowns
    manifest['site']['crown_top_m'] = round(max(tops), 4)
    footprints = json.load(open(research / 'footprints.json'))['footprints']
    nearest = sorted(footprints, key=lambda f: field.distance(Polygon(f['outline'])))[:16]
    buildings = []
    for house in nearest:
        rect = Polygon(house['outline']).minimum_rotated_rectangle
        co = np.array(rect.exterior.coords)[:4]
        delta = np.roll(co, -1, axis=0) - co
        lengths = np.linalg.norm(delta, axis=1)
        long = delta[np.argmax(lengths)]
        buildings.append([round(rect.centroid.x, 4), round(rect.centroid.y, 4), round(float(lengths.max()), 4),
                          round(float(lengths.min()), 4), round(float(np.degrees(np.arctan2(long[0], long[1])) % 180), 4)])
    manifest['site']['buildings'] = buildings
    v = vertices(path)
    faces = []
    count = 0
    for line in path.open():
        if line.startswith('v '):
            count += 1
        elif line.startswith('f '):
            row = [int(t.split('/')[0]) for t in line.split()[1:]]
            faces.append([index - 1 if index > 0 else count + index for index in row])
    faces = np.array(faces)
    surface = union_all(polygons(v[faces, :2]))
    inset = field.buffer(-.5)
    boundary = np.array([inset.exterior.interpolate(t).coords[0] for t in np.arange(0, inset.length, .1)])
    xx, yy = np.meshgrid(np.arange(*[field.bounds[j] for j in (0, 2)], .5), np.arange(*[field.bounds[j] for j in (1, 3)], .5))
    interior = np.array([q for q in np.c_[xx.ravel(), yy.ravel()] if inset.covers(Point(q))])
    points = np.vstack([boundary, interior])
    hits = p.rayTestBatch([[*q, 1200] for q in points], [[*q, 800] for q in points])
    misses = sum(h[0] != body for h in hits)
    sample = np.array([field.exterior.interpolate(t).coords[0] for t in np.arange(0, field.length, .1)])
    distances = np.array([surface.boundary.distance(Point(q)) for q in sample])
    def objective(shift):
        """Measure boundary error for a proposed constant survey translation."""
        return np.mean([surface.boundary.distance(Point(q + shift)) ** 2 for q in sample[::5]])
    fit = minimize(objective, [0., 0.], method='Nelder-Mead', options={'xatol': .001, 'initial_simplex': [[0., 0.], [1., 0.], [0., 1.]]})
    report = {'rays': len(hits), 'misses': misses, 'inset_boundary_rays': len(boundary), 'inset_outside_mesh_area_m2': float(inset.difference(surface).area),
              'edge_distance_median_m': float(np.median(distances)), 'edge_distance_max_m': float(distances.max()),
              'best_translation_east_north_m': fit.x.tolist(), 'translation_applied_m': [0, 0],
              'crown_count': len(crowns), 'crown_top_m': manifest['site']['crown_top_m'], 'building_count': len(buildings)}
    if misses or inset.difference(surface).area > 1e-8:
        raise RuntimeError(json.dumps(report))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
