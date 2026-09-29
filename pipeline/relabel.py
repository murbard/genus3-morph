"""Topology-preserving relabelling of face regions (discrete curve shortening of the seams).

A boundary triangle t of region A may switch to a neighbouring region B if
  * both regions stay topological disks: the part of t's boundary (3 vertices + 3 edges, in cyclic order) touched by
    the rest of A must be a single contiguous proper arc, and likewise for B (the 'simple triangle' test);
  * it lowers  E = total seam length + lam * sum_r (area_r - phi_r * total_area)^2.
Only the colouring changes; the geometry (and its certificate) is untouched."""
import numpy as np


class Topo:
    def __init__(self, F):
        self.F = F = F.astype(np.int64); nt = len(F)
        # edge neighbours: nb[t, i] = triangle across edge (F[t,i], F[t,(i+1)%3])
        key = {}
        self.nb = -np.ones((nt, 3), np.int64)
        for t in range(nt):
            for i in range(3):
                a, b = F[t, i], F[t, (i + 1) % 3]; k = (min(a, b), max(a, b))
                if k in key:
                    s, j = key[k]; self.nb[t, i] = s; self.nb[s, j] = t
                else: key[k] = (t, i)
        self.vt = [[] for _ in range(F.max() + 1)]
        for t in range(nt):
            for v in F[t]: self.vt[v].append(t)


def ring(topo, t):
    """Triangles sharing a vertex with t, in cyclic order around t (fans around v0, v1, v2 in turn)."""
    F, nb = topo.F, topo.nb; out = []
    for i in range(3):
        v = F[t, i]
        prev, cur, end = t, nb[t, (i - 1) % 3], nb[t, i]      # from across edge (v_{i-1}, v_i) to across (v_i, v_{i+1})
        guard = 0
        while cur >= 0 and guard < 64:
            if not out or out[-1] != cur: out.append(cur)
            if cur == end: break
            # next triangle around v: across the edge of cur that contains v and is not shared with prev
            nxt = -1
            for k in range(3):
                a, b = F[cur, k], F[cur, (k + 1) % 3]
                if v in (a, b) and nb[cur, k] != prev: nxt = nb[cur, k]; break
            prev, cur = cur, nxt; guard += 1
    if len(out) > 1 and out[0] == out[-1]: out.pop()
    return out


def _one_run(flags):
    n = len(flags); k = sum(flags)
    if k == 0 or k == n: return k > 0 and False
    return sum(1 for i in range(n) if flags[i] and not flags[i - 1]) == 1


def simple(topo, L, t, A, B):
    """t may switch from face A to face B iff, in the cyclic ring of triangles around t, the A-triangles form one
    contiguous run and the B-triangles form one contiguous run (both faces stay disks, no pinch at t's corners)."""
    R = ring(topo, t)
    return _one_run([L[s] == A for s in R]) and _one_run([L[s] == B for s in R])


def sweep(topo, L, X, phi, lam, rng):
    """One pass over the boundary triangles in random order. Returns number of flips."""
    F = topo.F
    area = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    Ar = np.bincount(L, area, 8); At = Ar.sum(); T = phi * At
    elen = np.stack([np.linalg.norm(X[F[:, (i + 1) % 3]] - X[F[:, i]], axis=1) for i in range(3)], 1)
    nbl = L[topo.nb]
    cand = np.where((nbl != L[:, None]).any(1))[0]
    rng.shuffle(cand); flips = 0
    for t in cand:
        A = L[t]; nl = L[topo.nb[t]]
        best, bestB = 0.0, -1
        for B in set(nl.tolist()) - {A}:
            dP = float((elen[t] * ((nl != B).astype(float) - (nl != A).astype(float))).sum())
            a = area[t]
            dA = lam * (((Ar[A] - a - T[A])**2 - (Ar[A] - T[A])**2) + ((Ar[B] + a - T[B])**2 - (Ar[B] - T[B])**2))
            d = dP + dA
            if d < best - 1e-9 and simple(topo, L, t, A, B): best, bestB = d, B
        if bestB >= 0:
            Ar[A] -= area[t]; Ar[bestB] += area[t]; L[t] = bestB; flips += 1
    return flips


def transfer(Xold, Fold, Lold, Xnew, Fnew):
    """Carry labels across a remesh (identical geometry): each new triangle takes the label of the old triangle
    nearest to its centroid."""
    from scipy.spatial import cKDTree
    co = Xold[Fold].mean(1); cn = Xnew[Fnew].mean(1)
    _, idx = cKDTree(co).query(cn)
    return Lold[idx].copy()


def _region_necks(F, L, r, min_side=3):
    import networkx as nx
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); T = np.tile(np.arange(len(F)), 3)
    K = np.sort(E, 1); o = np.lexsort((K[:, 1], K[:, 0])); T = T[o]; t1, t2 = T[0::2], T[1::2]
    m = (L[t1] == r) & (L[t2] == r)
    G = nx.Graph(); G.add_nodes_from(np.where(L == r)[0].tolist()); G.add_edges_from(zip(t1[m].tolist(), t2[m].tolist()))
    out = set()
    for a in nx.articulation_points(G):
        H = G.copy(); H.remove_node(a); sizes = sorted(len(c) for c in nx.connected_components(H))
        if len(sizes) >= 2 and sizes[-2] >= min_side: out.add(int(a))
    return out


def fatten_necks(F, L, max_passes=30):
    """Widen one-triangle necks by recolouring a neighbouring triangle to the neck's face. A flip is kept only if both
    faces stay disks (simple-triangle test) and the two faces' neck count strictly decreases. Colouring only."""
    L = L.copy(); topo = Topo(F)
    cnt = {r: _region_necks(F, L, r) for r in range(8)}; n0 = sum(len(v) for v in cnt.values())
    for _ in range(max_passes):
        improved = False
        for r in range(8):
            for t in sorted(cnt[r]):
                if L[t] != r or t not in cnt[r]: continue
                for s in topo.nb[t]:
                    if s < 0 or L[s] == r: continue
                    q = L[s]
                    if not simple(topo, L, s, q, r): continue
                    L[s] = r; nr = _region_necks(F, L, r); nq = _region_necks(F, L, q)
                    if len(nr) + len(nq) < len(cnt[r]) + len(cnt[q]): cnt[r], cnt[q] = nr, nq; improved = True; break
                    L[s] = q
        if not improved: break
    return L, n0, sum(len(v) for v in cnt.values())


def erode_tails(F, L, max_loss=0.05):
    """Remove dangling one-triangle-wide tails: repeatedly recolour a 'tip' triangle (touching its own face along
    only one edge) to the face around it, if both faces stay disks. Never cuts a neck (a strip's middle triangles are
    not tips, and cutting would fail the simple test). Each face loses at most max_loss of its triangles."""
    L = L.copy(); topo = Topo(F); start = np.bincount(L, minlength=8); lost = np.zeros(8, int); flips = 0
    changed = True
    while changed:
        changed = False
        nbl = L[topo.nb]
        same = (nbl == L[:, None]).sum(1)
        for t in np.where(same == 1)[0]:
            r = L[t]
            if (L[topo.nb[t]] == r).sum() != 1: continue          # re-check live: still a tip?
            if sum(L[u] == r for u in ring(topo, t)) > 3: continue  # a tooth on a jagged seam, not a thin tail
            others = [q for q in L[topo.nb[t]] if q != r]
            if not others or lost[r] + 1 > max_loss * start[r]: continue
            q = max(set(others), key=others.count)
            if simple(topo, L, t, r, q):
                L[t] = q; lost[r] += 1; flips += 1; changed = True
    return L, flips
