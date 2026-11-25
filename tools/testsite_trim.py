"""Create measured export variants while preserving all geometry and placements near the field."""
# One-shot build tool, not installed with this package: it needs numpy, Pillow and shapely.
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon, box


def compact_obj(path):
    """Deduplicate identical OBJ attributes and remove redundant numeric spelling without moving vertices."""
    maps = {'v': {}, 'vt': {}, 'vn': {}}
    indices = {k: [0] for k in maps}
    lines = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind in maps:
            values = tuple(float(v) for v in parts[1:])
            if values not in maps[kind]:
                maps[kind][values] = len(maps[kind]) + 1
                lines.append(kind + ' ' + ' '.join(format(v, '.12g') for v in values))
            indices[kind].append(maps[kind][values])
        elif kind == 'f':
            corners = []
            for token in parts[1:]:
                fields = token.split('/')
                corners.append('/'.join(str(indices[k][int(v)]) if v else '' for k, v in zip(['v', 'vt', 'vn'], fields)))
            lines.append('f ' + ' '.join(corners))
        elif not kind.startswith('#'):
            lines.append(line)
    path.write_text('\n'.join(lines) + '\n')


def main():
    """Make a density-only variant or the complete cut from an untouched source directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('dest')
    parser.add_argument('mode', choices=['density30', 'density10', 'geometry', 'textures', 'cut'])
    args = parser.parse_args()
    src, dest = Path(args.source), Path(args.dest)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    manifest = json.load(open(dest / 'manifest.json'))
    field = Polygon(manifest['site']['field_outline'])
    table = dict(np.load(dest / 'plants/forest.npz'))
    protected = np.array([field.distance(Point(x, y)) <= 200 for x, y in table['position'][:, :2]])
    limit = .3 if args.mode == 'density30' else .1
    keep = protected | (table['tier'] == 0) | (table['tier'] == 3) | ((table['tier'] != 2) & (table['rank'] < limit))
    np.savez_compressed(dest / 'plants/forest.npz', **{k: v[keep] for k, v in table.items()})
    if args.mode in ('geometry', 'textures', 'cut'):
        removed = []
        for name, item in manifest['items'].items():
            lo, hi = item['bounds_min'], item['bounds_max']
            distance = field.distance(box(*lo[:2], *hi[:2]))
            if (item['group'] == 'buildings' and distance > 500) or (item['group'] == 'terrain' and (distance > 2000 or name.startswith('ground_horizon_'))):
                removed.append(name)
                for ext in ['obj', 'mtl', 'png']:
                    (dest / item['folder'] / (name + '.' + ext)).unlink(missing_ok=True)
            elif args.mode != 'geometry' and distance > 150 and item.get('texture'):
                path = dest / item['folder'] / item['texture']
                im = Image.open(path)
                im.thumbnail((256, 256), Image.Resampling.LANCZOS)
                im.save(path)
        manifest['items'] = {k: v for k, v in manifest['items'].items() if k not in removed}
        manifest['placements'] = [v for v in manifest['placements'] if v['item'] not in removed]
        if args.mode == 'cut':
            for path in dest.rglob('*.obj'):
                compact_obj(path)
        for path in (dest / 'manifest.d').glob('*.json'):
            part = json.load(open(path))
            part['items'] = [i for i in part.get('items', []) if i['name'] not in removed]
            part['placements'] = [i for i in part.get('placements', []) if i['item'] not in removed]
            path.write_text(json.dumps(part, indent=1) + '\n')
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(args.mode, 'trees', int(keep.sum()), flush=True)


if __name__ == '__main__':
    main()
