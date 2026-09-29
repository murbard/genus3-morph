"""Thin isthmuses: triangles (and vertices) whose removal splits a face region into two parts of >= 15 triangles."""
import numpy as np, pickle, sys, networkx as nx
def isthmus(F, L, X):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); T = np.tile(np.arange(len(F)), 3)
    K = np.sort(E, 1); o = np.lexsort((K[:, 1], K[:, 0])); K = K[o]; T = T[o]; t1, t2 = T[0::2], T[1::2]
    out = {}
    for r in range(8):
        m = (L[t1] == r) & (L[t2] == r)
        G = nx.Graph(); G.add_nodes_from(np.where(L == r)[0]); G.add_edges_from(zip(t1[m], t2[m]))
        thin = []
        for a in nx.articulation_points(G):
            H = G.copy(); H.remove_node(a); sizes = sorted(len(c) for c in nx.connected_components(H))
            if len(sizes) >= 2 and sizes[-2] >= 15: thin.append(a)
        # vertex-level isthmus: a vertex whose region fan touches the boundary on two separate sides
        out[r] = (len(thin), X[F[thin]].mean(1).round(0).tolist()[:3])
    return out
if __name__ == '__main__':
    for path in sys.argv[1:]:
        e = pickle.load(open(path, 'rb'))['epochs'][-1]
        print(path, isthmus(e['F'].astype(np.int64), e['L'], e['frames'][-1].astype(float)))
