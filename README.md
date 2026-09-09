# swarm-worlds

The maps, robot descriptions and textures that the [Swarm](https://github.com/swarm-subnet/swarm) simulator loads. This repository holds files; the code that turns them into worlds lives in the swarm repository.

## Layout

```
swarm_worlds/
  maps/
    custom/          project-made assets: office twin, mountains, buildings, warehouse shell, lost-person characters
    forest/          Quaternius nature packs and the forest ground texture
    kenney/          Kenney kits: suburban, commercial, roads, car, conveyor, furniture, holiday
    other_sources/   third-party assets kept under their original licenses
  robots/            tello.urdf with its mesh folder, interceptor_drone.urdf
  textures/          tao.png (landing pad), Swarm.png and Swarm_2.png (wall posters)
```

Every folder keeps the license and source notice of the pack it came from. `maps/README.md` and `maps/custom/LICENSE_CUSTOM_MAPS.md` describe the origin of each group.

## Using it

```python
import swarm_worlds

swarm_worlds.maps_dir()      # .../swarm_worlds/maps
swarm_worlds.robots_dir()    # .../swarm_worlds/robots
swarm_worlds.textures_dir()  # .../swarm_worlds/textures
```

Each function returns an absolute path. Assets are addressed by their relative name below that root, for example `kenney/kenney_suburban/building-type-a.obj`.

The swarm repository pins a tagged release in its `requirements.txt`:

```
swarm-worlds @ git+https://github.com/swarm-subnet/swarm-worlds.git@v1.0.0
```

pip, uv and pixi all install that line. The simulator writes small caches next to some assets at runtime, so the package must be installed somewhere the running user can write, which is the case for a virtualenv or a pixi environment owned by that user.

## Changing an asset

1. Open a pull request here. CI installs the built wheel and checks that every root, manifest entry, URDF and material library resolves.
2. Bump `__version__` in `swarm_worlds/__init__.py` and tag the merge commit with the same number (`v1.1.0`).
3. Bump the tag in the swarm repository's `requirements.txt`, refresh its `pixi.lock`, and bump the swarm version. A map change can change flight results, so it follows the same release rules as a scoring change.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest pre-commit
pytest
pre-commit run --all-files
```
