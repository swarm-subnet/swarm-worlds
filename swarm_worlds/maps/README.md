# Map Asset Sources

Assets in this directory are grouped by origin. A directory name is an
organizational category, not a replacement for an asset's license notice.

- `custom/`: project-created or project-modified assets. See the nearest
  license, manifest, or source notice before reuse.
- `other_sources/`: third-party assets retained under their original licenses.
- `kenney/`: assets sourced from Kenney; preserve the included source and
  license information.
- `forest/`: forest asset packs; preserve each pack's included source and
  license information.

## Solar park

`custom/solar/` is a 1:1 replica of a real photovoltaic park, read by the solar map builder in the
swarm repository. The seed decides how much of the forest and the grass stands and when the movers
move; everything else stays where the real park has it.

- `manifest.json`: every piece, its placements, and the forest entry.
- `park/`, `terrain/`, `grass/`, `movers/`: the pieces, one folder per group.
- `plants/`: the forest table and the tree meshes it names.
- `intruders/`: the thieves, dressed and posed by the Solar Patrol family's `intruders.py`.

### Intruders

`custom/solar/intruders/` holds one body in six male builds and the 24 garments and tools made on it.

- `rig.npz`: the 77-joint skeleton, the six builds (vertices, bind and rest transforms) and the body's skin
  weights, in the SOMA frame (y up, metres).
- `body.obj`: the standard build's body, z up.
- `pieces/<garment>_<colour>.obj`: one closed mesh per colour of a garment or tool, on its own origin; the
  manifest gives the offset that places it on the standard build. The `.npz` beside it carries, in the same
  vertex order, the bone weights, the skin point each vertex follows onto another build, and the bone a
  rigid piece rides.
- `intruders.json`: the builds, and per garment its slot, the body parts it hides, and per piece the colours
  a seed chooses from and its temperature for the thermal camera.
- `LICENSE_SOMA.txt`: the body, skeleton and skin weights derive from NVIDIA Kimodo and SOMA-X under the
  Apache License 2.0, reproduced there. The garments and tools were made for Swarm.

## Lost-person characters

`custom/people/lost_person_characters/` contains character targets intended
for Search and Rescue simulations. The collection includes both
AI-generated/modified assets and open-source assets.

See the collection's `manifest.json` for each character's scale, origin, and
license status. Its open-source characters are sourced from Quaternius under
CC0; see the collection's `SOURCE.txt`.
