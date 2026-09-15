# Sky photographs

Real skies for the daylight model of the simulator. Each file is an equirectangular PNG, 2048 x 1024,
its top row the zenith and column 0 the +x heading, holding sRGB-encoded linear radiance. The renderer
scales every sky to one mean brightness, turns it so its sun stands where the seed's sun does, and
treats texels clipped to white as the sun and its glare.

`skies.json` lists each file with the heading and height of its sun, read from the brightest spot of
the original, and the page it came from.

## Source and license

All sixteen come from the Poly Haven pure-sky set (https://polyhaven.com/hdris/skies), released under
CC0: free to use and redistribute for any purpose, no attribution required.

## How the files were made

From each 2k Radiance HDR: the upper hemisphere was scaled so its brightest one percent clips to white,
values were clipped to one, encoded with the sRGB curve to 8 bits and saved as PNG. The sun position is
the circular mean of the brightest 0.01 percent of the upper half.
