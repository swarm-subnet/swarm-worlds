# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""The maps, robot descriptions and textures the Swarm simulator loads.

The package carries files, not code. Each function below returns the absolute
path of one asset root; everything under a root is addressed by the relative
names the simulator has always used.
"""

from importlib.resources import files

__version__ = "1.0.0"

_ROOT = files(__name__)


def maps_dir() -> str:
    """Root of the map asset tree: custom, forest, kenney and other_sources."""
    return str(_ROOT / "maps")


def robots_dir() -> str:
    """Root of the robot descriptions: the URDFs and the mesh folders they reference."""
    return str(_ROOT / "robots")


def textures_dir() -> str:
    """Root of the standalone textures the map builders apply to generated geometry."""
    return str(_ROOT / "textures")
