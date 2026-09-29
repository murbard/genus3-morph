"""Fine conforming triangulation of the polyhedron's 8 planar faces."""
import json, numpy as np, triangle, sys
d = json.load(open(__file__.rsplit('/', 1)[0] + '/poly.json'))
V0 = np.array(d['V'], float); WALKS = [f['walk'] for f in d['faces']]
SIGN = [1, 1, -1, -1, -1, -1, 1, 1]   # paper's orientation; +1 walks are outward-CCW after the check below

def build(ell):
    verts = [v for v in V0]; edge_pts = {}
    def edge_chain(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_pts:
            p, q = V0[key[0] - 1], V0[key[1] - 1]
            n = max(1, int(np.ceil(np.linalg.norm(q - p) / ell)))
            ids = []
            for k in range(1, n):
                verts.append(p + (q - p) * k / n); ids.append(len(verts) - 1)
            edge_pts[key] = [key[0] - 1] + ids + [key[1] - 1]
        c = edge_pts[key]
        return c if key[0] == a else c[::-1]
    faces, labels = [], []
    for fi, walk in enumerate(WALKS):
        loop = []
        for i in range(9):
            loop += edge_chain(walk[i], walk[(i + 1) % 9])[:-1]
        P3 = np.array([verts[i] for i in loop])
        n = np.array(d['faces'][fi]['n']); u = np.cross(n, [0.3, 0.5, 0.8]); u /= np.linalg.norm(u); w = np.cross(n, u)
        o = P3[0]; P2 = np.c_[(P3 - o) @ u, (P3 - o) @ w]
        segs = np.c_[np.arange(len(loop)), (np.arange(len(loop)) + 1) % len(loop)]
        area = ell * ell * np.sqrt(3) / 4
        T = triangle.triangulate({'vertices': P2, 'segments': segs}, f'pYq28a{area:.3f}')
        tv, tt = T['vertices'], T['triangles']
        idmap = list(loop)
        for k in range(len(loop), len(tv)):
            verts.append(o + tv[k, 0] * u + tv[k, 1] * w); idmap.append(len(verts) - 1)
        assert np.allclose(tv[:len(loop)], P2)
        tri = np.array(idmap)[tt]
        # orient consistently with the face walk direction times the paper's sign
        e1 = tv[tt[:, 1]] - tv[tt[:, 0]]; e2 = tv[tt[:, 2]] - tv[tt[:, 0]]; a2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        walk_ccw = np.sum(P2[:, 0] * np.roll(P2[:, 1], -1) - np.roll(P2[:, 0], -1) * P2[:, 1]) > 0
        want = (1 if walk_ccw else -1) * SIGN[fi]
        tri[np.sign(a2) != want] = tri[np.sign(a2) != want][:, ::-1]
        faces.append(tri); labels += [fi] * len(tri)
    X = np.array(verts); F = np.concatenate(faces); L = np.array(labels)
    # make outward: signed volume should be positive
    vol = np.einsum('ij,ij->i', X[F[:, 0]], np.cross(X[F[:, 1]], X[F[:, 2]])).sum() / 6
    if vol < 0: F = F[:, ::-1]
    return X, F, L

if __name__ == '__main__':
    ell = float(sys.argv[1])
    X, F, L = build(ell)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    Eu, c = np.unique(E, axis=0, return_counts=True)
    # directed-edge check for consistent orientation
    De = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    dup = len(De) - len(np.unique(De, axis=0))
    vol = np.einsum('ij,ij->i', X[F[:, 0]], np.cross(X[F[:, 1]], X[F[:, 2]])).sum() / 6
    print('V', len(X), 'F', len(F), 'chi', len(X) - len(Eu) + len(F), 'manifold', (c == 2).all(), 'dup directed edges', dup, 'volume', round(vol))
    np.savez(__file__.rsplit('/', 1)[0] + '/mesh0.npz', X=X, F=F, L=L)
