"""Certify that linear interpolation between consecutive keyframes never makes two triangles touch.
Non-adjacent pairs: distance(t) >= d(0) - t * max_{i,j} |D_Ai - D_Bj|  (exact bound for linear motion), with subdivision.
Adjacent pairs (sharing a vertex/edge): checked at every endpoint and subdivision point."""
import numpy as np, sys, coll, opt

def certify_segment(X0, X1, F, vt, depth=0, max_depth=12):
    D = X1 - X0
    reach = np.linalg.norm(D, axis=1).max() + 1e-6
    P = coll.unique_pairs(coll.candidate_pairs(X0, F, reach))
    a = opt.ccd_alpha(X0, F, P, D, 1.0) if len(P) else 1.0
    if a >= 1.0:
        return True, 1
    if depth >= max_depth:
        return False, 1
    Xm = 0.5 * (X0 + X1)
    if coll.vertex_adjacent_violations(Xm, F, *vt) != 0: return False, 1
    ok1, n1 = certify_segment(X0, Xm, F, vt, depth + 1, max_depth)
    if not ok1: return False, n1
    ok2, n2 = certify_segment(Xm, X1, F, vt, depth + 1, max_depth)
    return ok2, n1 + n2

def certify(frames, F):
    F = F.astype(np.int64); vt = coll.vertex_tri_csr(F, frames.shape[1])
    report = []
    for k in range(len(frames)):
        X = frames[k].astype(float)
        if coll.vertex_adjacent_violations(X, F, *vt) != 0: return False, f'fold at keyframe {k}'
        P = coll.unique_pairs(coll.candidate_pairs(X, F, 0.05))
        if len(P) and coll.pair_dists(X, F, P).min() <= 0: return False, f'contact at keyframe {k}'
        if k + 1 < len(frames):
            ok, n = certify_segment(X, frames[k + 1].astype(float), F, vt)
            report.append(n)
            if not ok: return False, f'could not certify interval {k}->{k+1}'
    return True, f'{len(frames)} keyframes, {len(report)} intervals certified, subdivisions used: max {max(report) if report else 0}, total {sum(report)}'

if __name__ == '__main__':
    d = np.load(sys.argv[1]); fr = d['frames']
    if len(sys.argv) > 2: fr = fr[[int(i) for i in sys.argv[2].split(',')]] if ',' in sys.argv[2] else fr[::int(sys.argv[2])]
    print(certify(fr, d['F']))


def certify_collapse_interval(X0, X1, F, Xnext, Fnext):
    """Certify an epoch's final interval in which some edges collapse (both endpoints meet at the midpoint).
    Pairs that become adjacent at the merge are checked at sample times; all other pairs with the exact
    continuous bound; adjacency is checked along the way and, after the merge, on the next epoch's mesh."""
    F = F.astype(np.int64); D = X1 - X0
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    col = E[(np.linalg.norm(X1[E[:, 0]] - X1[E[:, 1]], axis=1) < 1e-9) & (np.linalg.norm(X0[E[:, 0]] - X0[E[:, 1]], axis=1) > 1e-9)]
    mate = -np.ones(len(X0), np.int64); mate[col[:, 0]] = col[:, 1]; mate[col[:, 1]] = col[:, 0]
    P = coll.unique_pairs(coll.candidate_pairs(X0, F, np.linalg.norm(D, axis=1).max() + 1e-6))
    A = F[P[:, 0]]; B = F[P[:, 1]]
    merging = np.zeros(len(P), bool)
    for i in range(3):
        for j in range(3): merging |= (mate[A[:, i]] == B[:, j]) & (mate[A[:, i]] >= 0)
    if not coll.certify_pairs(X0, D, F, P[~merging], 14).all(): return False, 'moving pair not certified'
    for s in np.linspace(0.02, 0.98, 25):
        if merging.any() and coll.pair_dists(X0 + s * D, F, P[merging]).min() <= 0: return False, 'merging pair touches early'
        vt = coll.vertex_tri_csr(F, len(X0))
        moved = np.where(np.abs(D).sum(1) > 0)[0]
        verts = np.unique(F[np.isin(F, moved).any(1)].ravel())
        if coll.adjacent_violations_at(X0 + s * D, F, vt[0], vt[1], verts) != 0: return False, 'fold during collapse'
    vtn = coll.vertex_tri_csr(Fnext.astype(np.int64), len(Xnext))
    if coll.vertex_adjacent_violations(Xnext, Fnext.astype(np.int64), *vtn) != 0: return False, 'fold after merge'
    return True, f'{len(col)} collapses certified'
