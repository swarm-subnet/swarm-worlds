"""Write equivalent OBJ faces with nearby relative indices to reduce uncompressed download size."""
import sys
from pathlib import Path


def stream_obj(path):
    """Emit each attribute just before its first face and use the shortest valid OBJ index spelling."""
    arrays = {'v': [None], 'vt': [None], 'vn': [None]}
    emitted = {key: {} for key in arrays}
    output = []
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        kind = words[0]
        if kind in arrays:
            arrays[kind].append(' '.join(words[1:]))
        elif kind == 'f':
            corners = []
            for corner in words[1:]:
                fields = []
                for key, value in zip(arrays, corner.split('/')):
                    if not value:
                        fields.append(None)
                        continue
                    old = int(value)
                    if old < 0:
                        old += len(arrays[key])
                    if old not in emitted[key]:
                        emitted[key][old] = len(emitted[key]) + 1
                        output.append(key + ' ' + arrays[key][old])
                    fields.append((key, emitted[key][old]))
                corners.append(fields)
            tokens = []
            for fields in corners:
                indices = []
                for field in fields:
                    if field is None:
                        indices.append('')
                    else:
                        key, index = field
                        positive = str(index)
                        negative = str(index - len(emitted[key]) - 1)
                        indices.append(negative if len(negative) < len(positive) else positive)
                tokens.append('/'.join(indices))
            output.append('f ' + ' '.join(tokens))
        elif not kind.startswith('#'):
            output.append(line)
    path.write_text('\n'.join(output) + '\n')


def main():
    """Rewrite every OBJ in an already trimmed export in place."""
    root = Path(sys.argv[1])
    before = sum(path.stat().st_size for path in root.rglob('*.obj'))
    for path in root.rglob('*.obj'):
        stream_obj(path)
    after = sum(path.stat().st_size for path in root.rglob('*.obj'))
    print({'before_bytes': before, 'after_bytes': after}, flush=True)


if __name__ == '__main__':
    main()
