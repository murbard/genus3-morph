"""Wrinkliness: angle between each triangle's normal and the mean normal of its 2-ring patch (high-frequency only)."""
import numpy as np, pickle, sys, scipy.sparse as sp
def wrink(X, F):
    fn = np.cross(X[F[:,1]]-X[F[:,0]], X[F[:,2]]-X[F[:,0]])
    nt = len(F); nv = X.shape[0]
    VT = sp.coo_matrix((np.ones(3*nt), (F.ravel(), np.repeat(np.arange(nt), 3))), shape=(nv, nt)).tocsr()
    TT = (VT.T @ VT); TT.data[:] = 1                       # triangles sharing a vertex
    TT2 = TT @ TT; TT2.data[:] = 1                          # 2-ring
    m = TT2 @ fn; m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    n = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    ok = np.linalg.norm(fn, axis=1) > 1e-6
    return np.degrees(np.arccos(np.clip((n * m).sum(1), -1, 1)))[ok]
for path in sys.argv[1:]:
    ep = pickle.load(open(path, 'rb'))['epochs']; out = []
    for e in ep:
        F = e['F'].astype(np.int64)
        out.append(max(np.percentile(wrink(X.astype(float), F), 99) for X in e['frames'][::max(1, len(e['frames'])//3)]))
    print(path, '| wrinkle p99 by epoch:', ' '.join('%.0f' % x for x in out))
