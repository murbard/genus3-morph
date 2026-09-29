# genus3-morph

A WebGL2 viewer that morphs Mizhaev's genus-3 polyhedron into a smooth triple torus.

The polyhedron comes from [*Integer Realization of an Equivelar Octahedron of Genus 3*](https://arxiv.org/abs/2609.17700)
(Ruslan Mizhaev, arXiv:2609.17700), which pins it down in exact integer coordinates. It has eight planar simple
nonagonal faces, 24 vertices and 36 edges, with three faces at every vertex — an equivelar map of type {9,3} — and
C₄ symmetry. Counting gives the genus: 24 − 36 + 8 = −4, and χ = 2 − 2g, so g = 3.

Every pair of faces touches: 20 of the 28 pairs share one edge and the remaining 8 share two. The tetrahedron and the
Szilassi polyhedron are the classical examples of pairwise-adjacent faces, and they are the only two where each pair
shares *exactly* one edge; eight faces cannot do that, since three-valent vertices would force the genus
(f−3)(f−4)/12 to be a whole number and at f = 8 it is 20/12. Hence the eight doubled pairs.

The point of the animation is that counting is not seeing. The faces are big thin nonagons that weave past one
another, and the flat object does not look like a triple torus from any angle. The morph relaxes it under surface
tension at constant volume until the three tunnels are obvious. Every keyframe, and every linear interpolation
between keyframes, is certified free of self-intersection, so the surface never passes through itself on the way.
Each triangle keeps the colour of the flat face it started on.

## Running it

`morph.bin` is fetched over HTTP, and browsers block `fetch` for `file://` URLs, so serve the folder rather than
opening the file directly:

```
python3 -m http.server 8000
```

then open <http://127.0.0.1:8000/>. Needs WebGL2. Nothing downloads until you press **Load animation** (4 MB).

| File | Role |
|---|---|
| `index.html` | The viewer: per-triangle face colours, transparency, triangle-mesh toggle, timeline. |
| `morph_loader.js` | Parses `morph.bin` and replays the remeshing; tested bit-exact against the packager. |
| `morph.bin` | Animation data — keyframes plus the edge collapses and splits needed to replay each remesh round. |
| `polyhedron.html` | The flat polyhedron on its own, no animation, with CPU-sorted transparency. |

## Controls

Drag to rotate, scroll to zoom. The opacity slider sees through the surface. The `F1`–`F8` checkboxes show and hide
individual faces — switch off all but two and you can see exactly where those two meet, for any pair you pick, which
is easiest part way into the morph once the razor-thin nonagons have fattened into ribbons.

Query parameters, handy for screenshots:

| Parameter | Meaning |
|---|---|
| `u` | Position in the animation, 0 to 1. |
| `opacity` | 0.1 to 1. |
| `dist` | Camera distance. |
| `rx`, `ry` | Rotate about x and y, in radians. |
| `mesh=1` | Show the triangle wireframe. |
| `autoload` | Skip the load button. |
| `method` | `peel` (depth peeling, the default), `sort` (CPU back-to-front triangle sort), or `wboit` (weighted blended OIT). |

## Known issue

On some ANGLE-on-NVIDIA configurations the depth-peeling path renders concentric moiré bands at opacity below 100%,
while the same browser and driver are fine with `?method=sort`. Because the banding survives comparing exact float
depth copies and rejecting the same triangle by id, plain depth precision does not explain it. Not diagnosed.

## What is not here

The simulation and packaging pipeline that produces `morph.bin` is not included; this repository is the viewer and
the data it replays.

## See also

A second, combinatorially inequivalent eight-faced polyhedron with the same property appeared shortly after
Mizhaev's: [arXiv:2609.32998](https://arxiv.org/abs/2609.32998).

## Licence

MIT, see [LICENSE](LICENSE). The polyhedron's coordinates are from Mizhaev's paper, cited above.
