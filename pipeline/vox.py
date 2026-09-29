"""Voxelize a closed triangle mesh (ray parity along z) + skeleton tools."""
import numpy as np
from numba import njit

@njit(cache=True)
def voxelize(X, F, gx, gy, gz):
    nx, ny, nz = len(gx), len(gy), len(gz)
    hits = [[0.0 for _ in range(0)] for _ in range(nx * ny)]
    for t in range(len(F)):
        a = X[F[t, 0]]; b = X[F[t, 1]]; c = X[F[t, 2]]
        x0 = min(a[0], b[0], c[0]); x1 = max(a[0], b[0], c[0]); y0 = min(a[1], b[1], c[1]); y1 = max(a[1], b[1], c[1])
        i0 = max(0, int(np.searchsorted(gx, x0))); i1 = min(nx, int(np.searchsorted(gx, x1, 'right')))
        j0 = max(0, int(np.searchsorted(gy, y0))); j1 = min(ny, int(np.searchsorted(gy, y1, 'right')))
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < 1e-14: continue
        for i in range(i0, i1):
            for j in range(j0, j1):
                px = gx[i] + 1e-7; py = gy[j] + 1.3e-7
                u = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / det
                v = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / det
                if u >= 0 and v >= 0 and u + v <= 1:
                    hits[i * ny + j].append(a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]))
    out = np.zeros((nx, ny, nz), np.bool_)
    for i in range(nx):
        for j in range(ny):
            h = hits[i * ny + j]
            if len(h) < 2: continue
            zs = np.sort(np.array(h))
            for k in range(0, len(zs) - 1, 2):
                for l in range(nz):
                    if gz[l] >= zs[k] and gz[l] < zs[k + 1]: out[i, j, l] = True
    return out
