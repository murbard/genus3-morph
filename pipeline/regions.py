"""Topology of each face region (set of triangles with one label): must stay a disk (connected, chi = 1, no vertex pinches)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def region_components(F, L):
    """Edge-connected components of each face region (vectorised)."""
    out = []
    for lab in range(8):
        T = F[L == lab]
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        idx = np.tile(np.arange(len(T)), 3); key = E[:, 0] * 10**7 + E[:, 1]
        o = np.argsort(key); k = key[o]; ii = idx[o]; same = k[1:] == k[:-1]
        G = coo_matrix((np.ones(same.sum()), (ii[:-1][same], ii[1:][same])), shape=(len(T),) * 2)
        out.append((connected_components(G, directed=False)[0],))
    return out


def region_report(F, L):
    out = {}
    for lab in range(8):
        T = F[L == lab]
        if len(T) == 0: out[lab] = 'EMPTY'; continue
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        Eu, cnt = np.unique(E, axis=0, return_counts=True)
        Vu = np.unique(T)
        chi = len(Vu) - len(Eu) + len(T)
        # triangle connectivity through shared edges
        idx = np.tile(np.arange(len(T)), 3); key = E[:, 0] * 10**7 + E[:, 1]
        o = np.argsort(key); k = key[o]; ii = idx[o]; same = k[1:] == k[:-1]
        G = coo_matrix((np.ones(same.sum()), (ii[:-1][same], ii[1:][same])), shape=(len(T),) * 2)
        ncomp = connected_components(G, directed=False)[0]
        # vertex pinches: a vertex whose incident region-triangles form more than one fan
        pinch = 0
        for v in Vu:
            tv = np.where((T == v).any(1))[0]
            if len(tv) < 2: continue
            sub = T[tv]; Es = np.sort(np.concatenate([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]]), 1)
            Es = Es[(Es == v).any(1)]; ks = Es[:, 0] * 10**7 + Es[:, 1]; ids = np.tile(np.arange(len(tv)), 3)[((np.concatenate([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]]) == v).any(1))]
            oo = np.argsort(ks); s2 = ks[oo][1:] == ks[oo][:-1]
            g2 = coo_matrix((np.ones(s2.sum()), (ids[oo][:-1][s2], ids[oo][1:][s2])), shape=(len(tv),) * 2)
            if connected_components(g2, directed=False)[0] > 1: pinch += 1
        out[lab] = (int(ncomp), int(chi), pinch)
    return out
if __name__ == '__main__':
    import pickle, sys
    A = np.load('phaseA.npz'); print('phaseA', region_report(A['F'].astype(np.int64), A['L']))
    for path in sys.argv[1:]:
        ep = pickle.load(open(path, 'rb'))['epochs']
        for k in list(range(0, len(ep), max(1, len(ep) // 6))) + [len(ep) - 1]:
            print(path, 'epoch', k, region_report(ep[k]['F'].astype(np.int64), ep[k]['L']))
