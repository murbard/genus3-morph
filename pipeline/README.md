# pipeline

The code that produces `../morph.bin` from the polyhedron's integer coordinates.

This is research code. It is a set of stages driven by hand, one at a time, not a turnkey build — there is no single
command that regenerates the animation, and the intermediate `.npz` and `.pkl` files are not in the repository.
`../morph.bin` is the output of a run of `package4.py`; its bytes are exactly what the viewer replays.

Needs Python 3 with `numpy`, `scipy`, `numba` and `networkx`, plus `matplotlib` for the two inspection scripts
(`render.py`, `section.py`).

## The input

`poly.json` is the polyhedron itself: 24 integer vertices in the range ±300, and the 8 nonagonal faces, each with its
triangulation, plane normal and offset, and boundary walk. The coordinates are Mizhaev's, from
[arXiv:2609.17700](https://arxiv.org/abs/2609.17700).

## The shape of the run

**Stage 1 — relax the flat polyhedron.** `mesh.py` builds a fine conforming triangulation of the 8 planar faces at a
given edge length and writes `mesh0.npz`. `opt.py` then runs IPC-style gradient descent on it: surface tension, a
volume constraint, mesh quality, an optional pull toward a target, and a log-barrier against contact so triangles
cannot pass through one another. `coll.py` is what makes that affordable — exact triangle-triangle distance behind a
numba spatial-hash broad phase.

**Stage 2 — pull toward the triple torus.** `target.py` builds the smooth target: a constant-radius tube, via
soft-min distance, around a symmetric skeleton. `phaseB.py` and `flow.py` pull the surface onto it, remeshing every
few iterations, because the triangulation degrades badly over that much motion.

**Remeshing, certifiably.** `remesh.py` is the reason the animation can be trusted. An edge collapse is realised as a
certified motion of both endpoints to their midpoint, so a remesh is itself a verified piece of the animation rather
than a discontinuity spliced in. `verify.py` certifies that linear interpolation between consecutive keyframes never
brings two triangles into contact — which is the guarantee the viewer relies on when it interpolates.

**Keeping the face colouring honest.** Each triangle carries the label of the flat face it came from, and those
regions have to stay well behaved as the mesh moves and is rebuilt. `regions.py` checks each region is still a disk,
`relabel.py` shortens the seams by discrete curve shortening without changing the topology, `seams.py` smooths the
seam geometry by moving vertices and never labels, and `bridges.py` and `necks.py` find the thin isthmuses and
narrowest necks where a region is about to pinch.

**Packaging.** `package4.py phaseA.npz flow.pkl out.bin [dmax]` selects keyframes, quantizes, re-certifies exactly
what will be displayed, and writes the binary the viewer loads. It also asserts that replaying each remesh round
from the recorded collapses and splits reproduces the next epoch's triangulation exactly — the same replay
`../morph_loader.js` performs in the browser. `package.py` and `package2.py` are earlier packagers that wrote a
`morph_data.js`; `package4.py` supersedes them.

## Support and inspection

`sdf.py`, `vox.py` (voxelize by ray parity, plus skeleton tools), `mt.py` (Kuhn/marching tetrahedra on the cube),
`skelgraph.py`, `section.py`, `render.py` (software painter's-order render to PNG), `rough2.py` and `wrinkle.py`
(normal-deviation measures for spotting high-frequency junk in a frame).
