"""Seam length and narrowest neck of each face region (bottleneck of the widest path between its two deepest points)."""
import numpy as np, pickle, sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
def necks(X, F, L):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); T = np.tile(np.arange(len(F)), 3)
    K = np.sort(E, 1); o = np.lexsort((K[:, 1], K[:, 0])); K = K[o]; T = T[o]
    t1, t2 = T[0::2], T[1::2]; Ke = K[0::2]; bd = L[t1] != L[t2]
    el = np.linalg.norm(X[Ke[:, 0]] - X[Ke[:, 1]], axis=1); perim = el[bd].sum()
    res = {}
    for r in range(8):
        inr = (L[t1] == r) | (L[t2] == r)
        Vr = np.unique(F[L == r]); bverts = np.unique(Ke[bd & inr].ravel())
        interior_edges = Ke[inr]
        n = len(X); G = coo_matrix((el[inr], (interior_edges[:, 0], interior_edges[:, 1])), shape=(n, n))
        d = dijkstra(G, directed=False, indices=bverts, min_only=True)
        d = np.where(np.isin(np.arange(n), Vr), d, -1)
        # widest path (maximin on d) between the two deepest vertices in different directions: approximate with
        # a threshold sweep: the largest t such that {d >= t} still connects the deepest vertex to the region's centre mass
        order = np.argsort(-d); deep = order[0]
        # maximin via union-find over vertices sorted by decreasing d
        parent = {}; comp_best = {}
        def find(a):
            while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
            return a
        adj = {}
        for a, b in interior_edges: adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        worst = np.inf; big = 0.25 * d.max()
        # neck = smallest d at which two 'lobes' each deeper than `big` merge
        for v in order:
            if d[v] < 0: break
            parent[v] = v; comp_best[v] = d[v]
            for u in adj.get(v, []):
                if u in parent:
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        if comp_best[ru] > big and comp_best[rv] > big: worst = min(worst, d[v])
                        parent[ru] = rv; comp_best[rv] = max(comp_best[rv], comp_best[ru])
        res[r] = round(float(2 * worst), 1) if np.isfinite(worst) else None
    return perim, res
if __name__ == '__main__':
    ep = pickle.load(open(sys.argv[1], 'rb'))['epochs']; e = ep[-1]
    p, r = necks(e['frames'][-1].astype(float), e['F'].astype(np.int64), e['L'])
    print('seam length %.0f  narrowest neck width per face (units):' % p, r)
