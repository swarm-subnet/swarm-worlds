"""Verify lossless geometry compaction and complete preservation of the protected map area."""
# One-shot build tool, not installed with this package: it needs numpy and shapely.
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box


def geometry_hash(path):
    """Hash the ordered OBJ face attributes after resolving all vertex indices."""
    values = {'v': [None], 'vt': [None], 'vn': [None]}
    digest = hashlib.sha256()
    for line in path.open():
        words = line.split()
        if not words:
            continue
        if words[0] in values:
            values[words[0]].append(np.asarray([float(v) for v in words[1:]], dtype='<f8').tobytes())
        elif words[0] == 'f':
            for corner in words[1:]:
                for key, index in zip(values, corner.split('/')):
                    if index:
                        digest.update(values[key][int(index)])
    return digest.hexdigest()


def main():
    """Compare each retained mesh and all nearby forest rows with their original exports."""
    src, dest, output = map(Path, sys.argv[1:])
    original = json.load(open(src / 'manifest.json'))
    final = json.load(open(dest / 'manifest.json'))
    field = Polygon(original['site']['field_outline'])
    meshes = 0
    for path in dest.rglob('*.obj'):
        assert geometry_hash(path) == geometry_hash(src / path.relative_to(dest)), path
        meshes += 1
    protected_items = 0
    for name, item in original['items'].items():
        if not name.startswith('ground_horizon_') and field.distance(box(*item['bounds_min'][:2], *item['bounds_max'][:2])) <= 150:
            assert final['items'][name] == item
            protected_items += 1
            if item.get('texture'):
                rel = Path(item['folder']) / item['texture']
                assert (src / rel).read_bytes() == (dest / rel).read_bytes(), rel
    source_table = dict(np.load(src / 'plants/forest.npz'))
    final_table = dict(np.load(dest / 'plants/forest.npz'))
    max_reach = 0.0
    for mesh, name in enumerate(original['forest']['meshes']):
        v = np.array([list(map(float, line.split()[1:3])) for line in (src / 'plants' / name).open() if line.startswith('v ')])
        scales = source_table['scale'][source_table['mesh'] == mesh, :2]
        if len(scales):
            max_reach = max(max_reach, float(np.linalg.norm(np.max(np.abs(v), axis=0) * scales.max(0))))
    assert max_reach < 50
    kept = {tuple(row) for row in final_table['position']}
    protected_trees = 0
    for xy in source_table['position']:
        if field.distance(Point(xy[:2])) <= 200:
            assert tuple(xy) in kept
            protected_trees += 1
    report = {'maximum_crown_reach_m': max_reach, 'identical_face_attribute_meshes': meshes, 'protected_items': protected_items, 'protected_forest_rows': protected_trees,
              'removed_houses': sum(v['group'] == 'buildings' for v in original['items'].values()) -
              sum(v['group'] == 'buildings' for v in final['items'].values()),
              'retained_houses': sum(v['group'] == 'buildings' for v in final['items'].values()),
              'retained_terrain': sum(v['group'] == 'terrain' for v in final['items'].values())}
    output.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
