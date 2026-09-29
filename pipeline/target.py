"""Smooth T-symmetric triple-torus target: constant-radius tube (soft-min distance) around the smoothed skeleton."""
import numpy as np, os
from numba import njit, prange
import skelgraph
HERE = os.path.dirname(os.path.abspath(__file__))
T = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], float); Ti = T.T

def resample(P, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)

def smooth(P, it=200, lam=0.5):
    P = P.copy()
    for _ in range(it): P[1:-1] += lam * (0.5 * (P[:-2] + P[2:]) - P[1:-1])
    return P

def build(n=40):
    z = np.load(os.path.join(HERE, 'voxA.npz')); sk = np.load(os.path.join(HERE, 'skel.npy'))
    cent, edges = skelgraph.graph(sk, z['g'], z['sd'])
    E = {tuple(sorted(int(t) for t in e['touch'])): e['pts'] for e in edges if len(e['touch']) == 2}
    C = {i + 1: c for i, c in enumerate(cent)}
    def path(a, b):   # oriented polyline from junction a to b (single branch)
        p = E[tuple(sorted((a, b)))]
        return p if np.linalg.norm(p[0] - C[a]) < np.linalg.norm(p[-1] - C[a]) else p[::-1]
    J1, J5, J7, J2 = C[1], C[5], C[7], C[2]
    v = np.mean([J1, Ti @ J5, Ti @ Ti @ J7, Ti @ Ti @ Ti @ J2], axis=0)
    V = [v, T @ v, T @ T @ v, T @ T @ T @ v]
    # slanted edges J1->J5, J5->J7, J7->J2, J2->J1 mapped back to the J1->J5 slot and averaged
    sl = [path(1, 5), path(5, 7), path(7, 2), path(2, 1)]
    rep = np.mean([resample(np.vstack([C[a], p, C[b]]), n) @ np.linalg.matrix_power(Ti, k).T
                   for k, (p, (a, b)) in enumerate(zip(sl, [(1, 5), (5, 7), (7, 2), (2, 1)]))], axis=0)
    # arch J1 -> J4 -> J7 (top) and J5 -> J6 -> J3 -> J2 (bottom, mapped back by T^-1)
    top = np.vstack([C[1], path(1, 4), C[4], path(4, 7), C[7]])
    bot = np.vstack([C[5], path(5, 6), C[6], path(6, 3), C[3], path(3, 2), C[2]])
    arch = np.mean([resample(top, 2 * n), resample(bot, 2 * n) @ Ti.T], axis=0)
    arch = 0.5 * (arch + (arch @ (T @ T).T)[::-1])        # symmetric under T^2 (swaps J1 and J7)
    rep = smooth(rep); arch = smooth(arch)
    rep[0], rep[-1] = V[0], V[1]; arch[0], arch[-1] = V[0], V[2]
    rep = smooth(rep, 50); arch = smooth(arch, 50)
    curves = [rep @ np.linalg.matrix_power(T, k).T for k in range(4)] + [arch, arch @ T.T]
    return V, curves

def segments(curves):
    return np.ascontiguousarray(np.concatenate([np.stack([c[:-1], c[1:]], 1) for c in curves]))

@njit(parallel=True, cache=True)
def softmin_dist(X, segs, k):
    """Smooth distance to a union of segments: -k log sum exp(-d_i/k), with gradient."""
    f = np.empty(len(X)); G = np.zeros_like(X)
    for i in prange(len(X)):
        m = 1e30; ds = np.empty(len(segs)); gs = np.empty((len(segs), 3))
        for s in range(len(segs)):
            a = segs[s, 0]; b = segs[s, 1]; ab = b - a; ap = X[i] - a
            t = min(max((ab[0]*ap[0] + ab[1]*ap[1] + ab[2]*ap[2]) / (ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]), 0.0), 1.0)
            r0 = ap[0] - t*ab[0]; r1 = ap[1] - t*ab[1]; r2 = ap[2] - t*ab[2]
            d = np.sqrt(r0*r0 + r1*r1 + r2*r2) + 1e-12
            ds[s] = d; gs[s, 0] = r0/d; gs[s, 1] = r1/d; gs[s, 2] = r2/d
            m = min(m, d)
        acc = 0.0; g0 = 0.0; g1 = 0.0; g2 = 0.0
        for s in range(len(segs)):
            w = np.exp(-(ds[s] - m) / k); acc += w; g0 += w*gs[s, 0]; g1 += w*gs[s, 1]; g2 += w*gs[s, 2]
        f[i] = m - k * np.log(acc); G[i, 0] = g0/acc; G[i, 1] = g1/acc; G[i, 2] = g2/acc
    return f, G

if __name__ == '__main__':
    import mt
    V, curves = build(); segs = segments(curves)
    print('junctions', np.round(V, 1).tolist())
    print('spine length %.0f' % sum(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in curves))
    g = np.linspace(-330, 330, 150); Xg = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    f, _ = softmin_dist(Xg, segs, 6.0); f = f.reshape(150, 150, 150); h = g[1] - g[0]
    VP = 4455360
    for R in [26, 28, 30, 32, 34, 36]:
        vol = (f < R).sum() * h**3
        v, fa, _ = mt.march(f - R, g); t = mt.topology(v, fa)
        print('R', R, 'volume/V_P %.3f' % (vol / VP), 'chi', t['chi'], 'components', t['components'])
    np.save(os.path.join(HERE, 'target_segs.npy'), segs)
