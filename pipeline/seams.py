"""Smooth seams by moving the mesh vertices that lie on seams (never the labels).
Each seam is a chain of mesh edges between junctions (vertices where >= 3 faces meet). All seam vertices, junctions
included, are relaxed together: a chain vertex moves toward the midpoint of its two chain neighbours, a junction toward
the mean of its neighbours along each incident chain (so the seams meet at even angles instead of kinking). Moves are
restricted to the tangent plane and to at most `clamp` x the local edge length. Colours stay per triangle, so the
displayed face topology is exactly the verified one."""
import numpy as np


def chains(F, L):
    F = F.astype(np.int64); nt = len(F)
    dir_ = {}
    for t in range(nt):
        for j in range(3):
            dir_[(F[t, j], F[t, (j + 1) % 3])] = t
    seam_nb = {}; vlabs = {}
    for t in range(nt):
        for v in F[t]: vlabs.setdefault(int(v), set()).add(int(L[t]))
    for (a, b), t in dir_.items():
        if a > b: continue
        t2 = dir_.get((b, a))
        if t2 is None or L[t] == L[t2]: continue
        seam_nb.setdefault(int(a), []).append(int(b)); seam_nb.setdefault(int(b), []).append(int(a))
    is_j = lambda v: len(vlabs[v]) >= 3 or len(seam_nb.get(v, [])) != 2
    used = set(); out = []
    def walk(v0, v1):
        c = [v0, v1]; used.add((min(v0, v1), max(v0, v1)))
        while not is_j(c[-1]) and c[-1] != c[0]:
            v, p = c[-1], c[-2]; nx = [u for u in seam_nb[v] if u != p]
            if not nx: break
            k = (min(v, nx[0]), max(v, nx[0]))
            if k in used: break
            used.add(k); c.append(nx[0])
        out.append(c)
    for v in seam_nb:
        if is_j(v):
            for u in seam_nb[v]:
                if (min(v, u), max(v, u)) not in used: walk(v, u)
    for v in seam_nb:
        for u in seam_nb[v]:
            if (min(v, u), max(v, u)) not in used: walk(v, u)
    return out


def vertex_normals(X, F):
    fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); N = np.zeros_like(X)
    for k in range(3): np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def _seam_graph(C):
    """Neighbour lists along seams for every seam vertex (junctions get one neighbour per incident chain end)."""
    nb = {}
    for c in C:
        loop = c[0] == c[-1]
        for i, v in enumerate(c):
            if loop and i == len(c) - 1: continue
            lst = nb.setdefault(v, [])
            if i > 0: lst.append(c[i - 1])
            elif loop: lst.append(c[-2])
            if i < len(c) - 1: lst.append(c[i + 1])
    return {v: sorted(set(u for u in l if u != v)) for v, l in nb.items()}


def _relax_around(X, X0, F, N, seam_verts, rings=3, iters=30, clamp=1.5):
    """Make room: tangentially relax the ordinary vertices within `rings` of the seams (seam vertices held fixed), so
    the triangles next to a straightened seam stay well shaped instead of being squashed."""
    import scipy.sparse as sp
    n = len(X)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float).tocsr()
    near = np.zeros(n, bool); near[seam_verts] = True; front = near.copy()
    for _ in range(rings):
        front = (A @ front.astype(float)) > 0; near |= front
    free = near.copy(); free[seam_verts] = False
    idx = np.where(free)[0]
    if not len(idx): return X
    deg = np.asarray(A.sum(1)).ravel()
    el = np.array([np.linalg.norm(X0[F[:, 0]] - X0[F[:, 1]], axis=1).mean()])[0]
    lim = clamp * el
    for _ in range(iters):
        mean = (A @ X) / deg[:, None]
        D = (X[idx] + 0.5 * (mean[idx] - X[idx])) - X0[idx]
        D -= (D * N[idx]).sum(1, keepdims=True) * N[idx]
        dl = np.linalg.norm(D, axis=1); f = np.minimum(1, lim / np.maximum(dl, 1e-12))
        X[idx] = X0[idx] + D * f[:, None]
    return X


def tooth_tips(F, L):
    """Tip vertex of every triangle with two edges on seams (it would be flattened by straightening, so it stays)."""
    F = np.asarray(F, np.int64); L = np.asarray(L)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); T = np.tile(np.arange(len(F)), 3)
    K = np.sort(E, 1); o = np.lexsort((K[:, 1], K[:, 0])); K = K[o]; T = T[o]
    t1, t2 = T[0::2], T[1::2]; Ke = K[0::2]; bd = L[t1] != L[t2]
    seam_edges = {}
    for (a, b), x, y in zip(Ke[bd], t1[bd], t2[bd]):
        seam_edges.setdefault(int(x), []).append((int(a), int(b))); seam_edges.setdefault(int(y), []).append((int(a), int(b)))
    tips = set()
    for t, es in seam_edges.items():
        if len(es) == 2:
            common = set(es[0]) & set(es[1])
            if len(common) == 1: tips.add(common.pop())
    return tips


def smooth_seams(X, F, C, iters=60, clamp=1.5, move_junctions=True, relax=True, fixed=()):
    X = X.copy(); N = vertex_normals(X, F); X0 = X.copy()
    G = _seam_graph(C)
    if not G: return X
    verts = np.array(sorted(G)); pos = {v: i for i, v in enumerate(verts)}
    # local edge length for the clamp: shortest seam edge at the vertex
    lim = np.array([clamp * min(np.linalg.norm(X0[u] - X0[v]) for u in G[v]) for v in verts])
    junction = np.array([len(G[v]) != 2 for v in verts])
    nbr = [np.array([pos[u] for u in G[v]]) for v in verts]
    P = X0[verts].copy(); Nv = N[verts]; B = X0[verts]
    fixed = set(fixed); fix_mask = np.array([v in fixed for v in verts])
    for _ in range(iters):
        target = np.array([P[n].mean(0) for n in nbr])
        Q = P + 0.5 * (target - P)
        if not move_junctions: Q[junction] = P[junction]
        if len(fixed): Q[fix_mask] = P[fix_mask]
        D = Q - B; D -= (D * Nv).sum(1, keepdims=True) * Nv                  # tangential only
        dl = np.linalg.norm(D, axis=1); f = np.minimum(1, lim / np.maximum(dl, 1e-12))
        P = B + D * f[:, None]
    X[verts] = P
    if relax: X = _relax_around(X, X0, F.astype(np.int64), N, verts)
    return X
