import json, numpy as np, sys
d = json.load(open(__file__.rsplit('/',1)[0] + '/poly.json'))
TRI = []; LAB = []
for i, f in enumerate(d['faces']):
    t = np.array(f["tris"], float).reshape(-1, 3, 3)
    if i in (2, 3, 4, 5): t = t[:, ::-1]   # paper's orientation (+,+,-,-,-,-,+,+)
    TRI.append(t); LAB += [i] * len(t)
TRI = np.concatenate(TRI); LAB = np.array(LAB)

def tri_dist2(p, a, b, c):
    # squared distance from points p (N,3) to triangle abc (Ericson's closest point)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ap @ ab, ap @ ac
    bp = p - b; d3, d4 = bp @ ab, bp @ ac
    cp = p - c; d5, d6 = cp @ ab, cp @ ac
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    with np.errstate(divide='ignore', invalid='ignore'):
        v = vb / den; w = vc / den
        q = a + v[:, None] * ab + w[:, None] * ac
        # edges
        vab = d1 / (d1 - d3); qab = a + vab[:, None] * ab
        wac = d2 / (d2 - d6); qac = a + wac[:, None] * ac
        wbc = (d4 - d3) / ((d4 - d3) + (d5 - d6)); qbc = b + wbc[:, None] * (c - b)
    m = lambda cond, val: (cond, val)
    cases = [
        (d1 <= 0) & (d2 <= 0), a,
        (d3 >= 0) & (d4 <= d3), b,
        (d6 >= 0) & (d5 <= d6), c,
        (vc <= 0) & (d1 >= 0) & (d3 <= 0), qab,
        (vb <= 0) & (d2 >= 0) & (d6 <= 0), qac,
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), qbc,
    ]
    out = q.copy(); done = np.zeros(len(p), bool)
    for k in range(0, len(cases), 2):
        cond = cases[k] & ~done
        val = cases[k + 1]
        out[cond] = val if val.ndim == 1 else val[cond]
        done |= cond
    return ((p - out) ** 2).sum(1)

def solid_angle(p, a, b, c):
    A, B, C = a - p, b - p, c - p
    la, lb, lc = [np.linalg.norm(X, axis=1) for X in (A, B, C)]
    num = np.einsum('ij,ij->i', A, np.cross(B, C))
    den = la * lb * lc + np.einsum('ij,ij->i', A, B) * lc + np.einsum('ij,ij->i', A, C) * lb + np.einsum('ij,ij->i', B, C) * la
    return 2 * np.arctan2(num, den)

def sdf(p, chunk=200000):
    """Signed distance (negative inside) and label of the nearest face."""
    D = np.empty(len(p)); L = np.empty(len(p), int); W = np.empty(len(p))
    for s in range(0, len(p), chunk):
        q = p[s:s + chunk]
        best = np.full(len(q), np.inf); lab = np.zeros(len(q), int); wn = np.zeros(len(q))
        for t, l in zip(TRI, LAB):
            d2 = tri_dist2(q, *t)
            m = d2 < best; best[m] = d2[m]; lab[m] = l
            wn += solid_angle(q, *t)
        wn /= 4 * np.pi
        D[s:s + chunk] = np.sqrt(best); L[s:s + chunk] = lab; W[s:s + chunk] = wn
    return D, L, W

if __name__ == '__main__':
    N = int(sys.argv[1]); lo, hi = -330, 330
    g = np.linspace(lo, hi, N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    p = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    D, L, W = sdf(p)
    print('winding values', np.unique(np.round(W, 3))[:10])
    sign = np.where(np.abs(W) > 0.5, -1, 1)
    np.savez_compressed(__file__.rsplit('/',1)[0] + f'/sdf{N}.npz', D=(sign * D).reshape(N, N, N), L=L.reshape(N, N, N), g=g, W=W.reshape(N,N,N))
    inside = sign < 0
    print('inside fraction', inside.mean(), 'max inscribed dist', D[inside].max())
