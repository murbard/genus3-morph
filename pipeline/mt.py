import numpy as np, itertools
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Kuhn triangulation of the cube: 6 tets sharing the 000-111 diagonal (consistent across cells)
TETS = []
for perm in itertools.permutations(range(3)):
    c = [0, 0, 0]; path = [tuple(c)]
    for ax in perm: c[ax] += 1; path.append(tuple(c))
    TETS.append(path)
TETS = np.array(TETS)  # (6,4,3)

def march(F, g):
    """Marching tetrahedra on grid F (inside where F<0). Returns verts, faces (oriented outward)."""
    N = F.shape[0]; n = N - 1
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    base = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    # only cells with a sign change
    corners = np.array(list(itertools.product((0, 1), repeat=3)))
    cv = np.stack([F[tuple((base + c).T)] for c in corners], 1)
    keep = (cv < 0).any(1) & (cv >= 0).any(1)
    base = base[keep]
    tris = []
    P = []  # accumulate (keyA, keyB) edge endpoints for each triangle vertex
    flat = lambda q: (q[:, 0] * N + q[:, 1]) * N + q[:, 2]
    for tet in TETS:
        idx = np.stack([flat(base + tet[m]) for m in range(4)], 1)       # (M,4) grid indices
        val = F.ravel()[idx]
        inside = val < 0
        cnt = inside.sum(1)
        for c in (1, 2, 3):
            sel = np.where(cnt == c)[0]
            if not len(sel): continue
            I, V = idx[sel], inside[sel]
            order = np.argsort(~V, axis=1, kind='stable')                # inside verts first
            S = np.take_along_axis(I, order, 1)
            if c == 1:
                a = S[:, 0]; tri = [(a, S[:, 1]), (a, S[:, 2]), (a, S[:, 3])]; tris.append((tri, S, 1))
            elif c == 3:
                d = S[:, 3]; tri = [(S[:, 0], d), (S[:, 1], d), (S[:, 2], d)]; tris.append((tri, S, 3))
            else:
                a, b, cc, d = S.T
                tris.append(([(a, cc), (a, d), (b, d)], S, 2))
                tris.append(([(a, cc), (b, d), (b, cc)], S, 2))
    # collect edges -> unique vertices
    ea, eb, owner = [], [], []
    for tri, S, c in tris:
        for (x, y) in tri: ea.append(x); eb.append(y)
    ea = np.concatenate(ea); eb = np.concatenate(eb)
    ntri = len(ea) // 3
    # reshape so that each triangle has 3 consecutive entries
    blocks_a = []; blocks_b = []
    for tri, S, c in tris:
        blocks_a.append(np.stack([t[0] for t in tri], 1)); blocks_b.append(np.stack([t[1] for t in tri], 1))
    A = np.concatenate(blocks_a); B = np.concatenate(blocks_b)                 # (T,3) inside/outside endpoints
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    key = lo.astype(np.int64) * (N ** 3) + hi
    uk, inv = np.unique(key.ravel(), return_inverse=True)
    faces = inv.reshape(-1, 3)
    ka, kb = uk // (N ** 3), uk % (N ** 3)
    fa, fb = F.ravel()[ka], F.ravel()[kb]
    t = fa / (fa - fb)
    coord = lambda k: np.stack([g[k // (N * N)], g[(k // N) % N], g[k % N]], 1)
    verts = coord(ka) + t[:, None] * (coord(kb) - coord(ka))
    # orient outward: normal should point from inside endpoint to outside endpoint
    nrm = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    out = coord(B[:, 0].ravel()) - coord(A[:, 0].ravel())
    flip = (nrm * out).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return verts, faces, (ka, kb, t)

def topology(verts, faces):
    E = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    V = len(verts); chi = V - len(Eu) + len(faces)
    G = coo_matrix((np.ones(len(Eu)), (Eu[:, 0], Eu[:, 1])), shape=(V, V))
    ncomp, lab = connected_components(G, directed=False)
    return dict(chi=int(chi), components=int(ncomp), manifold=bool((cnt == 2).all()), comp_sizes=sorted(np.bincount(lab).tolist(), reverse=True)[:6])
