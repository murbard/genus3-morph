"""Exact triangle-triangle distance + spatial-hash broad phase (numba)."""
import numpy as np
from numba import njit, prange

@njit(cache=True, inline='always')
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit(cache=True)
def pt_tri_d2(p, a, b, c):
    ab = b - a; ac = c - a; ap = p - a
    d1 = dot(ab, ap); d2 = dot(ac, ap)
    if d1 <= 0 and d2 <= 0: q = a
    else:
        bp = p - b; d3 = dot(ab, bp); d4 = dot(ac, bp)
        if d3 >= 0 and d4 <= d3: q = b
        else:
            vc = d1*d4 - d3*d2
            if vc <= 0 and d1 >= 0 and d3 <= 0: q = a + ab * (d1 / (d1 - d3))
            else:
                cp = p - c; d5 = dot(ab, cp); d6 = dot(ac, cp)
                if d6 >= 0 and d5 <= d6: q = c
                else:
                    vb = d5*d2 - d1*d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0: q = a + ac * (d2 / (d2 - d6))
                    else:
                        va = d3*d6 - d5*d4
                        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                            q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
                        else:
                            tot = va + vb + vc
                            if tot == 0.0:   # degenerate triangle: distance to its edges
                                return min(seg_seg_d2(p, p, a, b), seg_seg_d2(p, p, b, c), seg_seg_d2(p, p, c, a))
                            den = 1.0 / tot; q = a + ab * (vb * den) + ac * (vc * den)
    r = p - q
    return dot(r, r)

@njit(cache=True)
def seg_seg_d2(p1, q1, p2, q2):
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = dot(d1, d1); e = dot(d2, d2); f = dot(d2, r)
    if a <= 1e-30 and e <= 1e-30: return dot(r, r)
    if a <= 1e-30: s = 0.0; t = min(max(f / e, 0.0), 1.0)
    else:
        c = dot(d1, r)
        if e <= 1e-30: t = 0.0; s = min(max(-c / a, 0.0), 1.0)
        else:
            b = dot(d1, d2); den = a*e - b*b
            s = min(max((b*f - c*e) / den, 0.0), 1.0) if den > 1e-30 else 0.0
            t = (b*s + f) / e
            if t < 0: t = 0.0; s = min(max(-c / a, 0.0), 1.0)
            elif t > 1: t = 1.0; s = min(max((b - c) / a, 0.0), 1.0)
    w = p1 + d1*s - (p2 + d2*t)
    return dot(w, w)

@njit(cache=True)
def seg_tri_hit(p, q, a, b, c):
    # does segment pq cross triangle abc (Moller-Trumbore, closed)
    e1 = b - a; e2 = c - a; d = q - p
    h = np.cross(d, e2); det = dot(e1, h)
    scale = np.sqrt(dot(d, d) * dot(e1, e1) * dot(e2, e2))
    if abs(det) <= 1e-9 * scale: return False   # (near-)parallel: handled by the distance terms
    inv = 1.0 / det; s = p - a; u = dot(s, h) * inv
    if u < 0 or u > 1: return False
    qq = np.cross(s, e1); v = dot(d, qq) * inv
    if v < 0 or u + v > 1: return False
    t = dot(e2, qq) * inv
    return t >= 0 and t <= 1

@njit(cache=True)
def tri_tri_dist(A0, A1, A2, B0, B1, B2):
    """Exact distance between two triangles (0 if they intersect)."""
    if (seg_tri_hit(A0, A1, B0, B1, B2) or seg_tri_hit(A1, A2, B0, B1, B2) or seg_tri_hit(A2, A0, B0, B1, B2) or
        seg_tri_hit(B0, B1, A0, A1, A2) or seg_tri_hit(B1, B2, A0, A1, A2) or seg_tri_hit(B2, B0, A0, A1, A2)):
        return 0.0
    m = pt_tri_d2(A0, B0, B1, B2)
    m = min(m, pt_tri_d2(A1, B0, B1, B2)); m = min(m, pt_tri_d2(A2, B0, B1, B2))
    m = min(m, pt_tri_d2(B0, A0, A1, A2)); m = min(m, pt_tri_d2(B1, A0, A1, A2)); m = min(m, pt_tri_d2(B2, A0, A1, A2))
    As = (A0, A1, A2); Bs = (B0, B1, B2)
    for i in range(3):
        for j in range(3):
            m = min(m, seg_seg_d2(As[i], As[(i+1) % 3], Bs[j], Bs[(j+1) % 3]))
    return np.sqrt(m)

@njit(cache=True)
def candidate_pairs(X, F, reach):
    """Pairs of vertex-disjoint triangles whose AABBs (grown by reach) overlap. Spatial hash on a uniform grid."""
    nt = F.shape[0]
    lo = np.empty((nt, 3)); hi = np.empty((nt, 3))
    for t in range(nt):
        for k in range(3):
            a = X[F[t, 0], k]; b = X[F[t, 1], k]; c = X[F[t, 2], k]
            lo[t, k] = min(a, b, c) - reach; hi[t, k] = max(a, b, c) + reach
    cell = 0.0
    for t in range(nt):
        for k in range(3): cell += hi[t, k] - lo[t, k]
    cell = cell / (3 * nt) * 1.5
    gmin = np.empty(3)
    for k in range(3): gmin[k] = lo[:, k].min()
    # hash each triangle into all cells its box covers
    keys = []; tids = []
    for t in range(nt):
        i0 = int((lo[t, 0] - gmin[0]) / cell); i1 = int((hi[t, 0] - gmin[0]) / cell)
        j0 = int((lo[t, 1] - gmin[1]) / cell); j1 = int((hi[t, 1] - gmin[1]) / cell)
        k0 = int((lo[t, 2] - gmin[2]) / cell); k1 = int((hi[t, 2] - gmin[2]) / cell)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    keys.append((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)); tids.append(t)
    keys = np.array(keys); tids = np.array(tids)
    order = np.argsort(keys, kind='mergesort'); keys = keys[order]; tids = tids[order]
    out_a = []; out_b = []
    s = 0; n = len(keys)
    while s < n:
        e = s
        while e < n and keys[e] == keys[s]: e += 1
        for p in range(s, e):
            for q in range(p + 1, e):
                a = tids[p]; b = tids[q]
                if a == b: continue
                if a > b: a, b = b, a
                share = False
                for u in range(3):
                    for v in range(3):
                        if F[a, u] == F[b, v]: share = True
                if share: continue
                ok = True
                for k in range(3):
                    if lo[a, k] > hi[b, k] or lo[b, k] > hi[a, k]: ok = False
                if ok: out_a.append(a); out_b.append(b)
        s = e
    P = np.empty((len(out_a), 2), np.int64)
    for i in range(len(out_a)): P[i, 0] = out_a[i]; P[i, 1] = out_b[i]
    return P

def unique_pairs(P):
    if len(P) == 0: return P
    k = P[:, 0].astype(np.int64) * 10_000_000 + P[:, 1]
    _, idx = np.unique(k, return_index=True)
    return P[idx]

@njit(parallel=True, cache=True)
def pair_dists(X, F, P):
    out = np.empty(len(P))
    for i in prange(len(P)):
        a = F[P[i, 0]]; b = F[P[i, 1]]
        out[i] = tri_tri_dist(X[a[0]], X[a[1]], X[a[2]], X[b[0]], X[b[1]], X[b[2]])
    return out

@njit(cache=True)
def _adj_bad(X, a, b, v):
    shared = 0
    for u in range(3):
        for w in range(3):
            if a[u] == b[w]: shared += 1
    if shared == 1:
        # edges opposite the shared vertex must not cross the other triangle
        a0 = -1; a1 = -1; b0 = -1; b1 = -1
        for k in range(3):
            if a[k] != v:
                if a0 < 0: a0 = a[k]
                else: a1 = a[k]
            if b[k] != v:
                if b0 < 0: b0 = b[k]
                else: b1 = b[k]
        return seg_tri_hit(X[a0], X[a1], X[b[0]], X[b[1]], X[b[2]]) or seg_tri_hit(X[b0], X[b1], X[a[0]], X[a[1]], X[a[2]])
    if shared == 2:
        na = np.cross(X[a[1]] - X[a[0]], X[a[2]] - X[a[0]]); nb = np.cross(X[b[1]] - X[b[0]], X[b[2]] - X[b[0]])
        c = dot(na, nb) / np.sqrt(dot(na, na) * dot(nb, nb) + 1e-300)
        return c < -0.99999   # folded to within ~0.25 degrees of flat
    return False

@njit(parallel=True, cache=True)
def vertex_adjacent_violations(X, F, VT_ptr, VT_idx):
    """Triangle pairs sharing a vertex: sharing one vertex, the opposite edges must not cross the other triangle;
    sharing an edge, they must not fold flat. Returns the number of violations."""
    nv = len(VT_ptr) - 1; bad = np.zeros(nv, np.int64)
    for v in prange(nv):
        cnt = 0
        for ii in range(VT_ptr[v], VT_ptr[v + 1]):
            for jj in range(ii + 1, VT_ptr[v + 1]):
                if _adj_bad(X, F[VT_idx[ii]], F[VT_idx[jj]], v): cnt += 1
        bad[v] = cnt
    return bad.sum()

def vertex_tri_csr(F, nv):
    cnt = np.bincount(F.ravel(), minlength=nv); ptr = np.r_[0, np.cumsum(cnt)]
    order = np.argsort(F.ravel(), kind='stable'); return ptr, (order // 3).astype(np.int64)

@njit(cache=True)
def adjacent_violations_at(X, F, VT_ptr, VT_idx, verts):
    cnt = 0
    for v in verts:
        for ii in range(VT_ptr[v], VT_ptr[v + 1]):
            for jj in range(ii + 1, VT_ptr[v + 1]):
                if _adj_bad(X, F[VT_idx[ii]], F[VT_idx[jj]], v): cnt += 1
    return cnt

@njit(parallel=True, cache=True)
def certify_pairs(X0, D, F, P, max_depth):
    """For each vertex-disjoint triangle pair, certify that linear motion X0 + t*D (t in [0,1]) never brings them into
    contact: adaptive bisection in t with the bound  dist(t) >= dist(t0) - (t - t0) * max_ij |D_Ai - D_Bj|."""
    ok = np.ones(len(P), np.bool_)
    for k in prange(len(P)):
        A = F[P[k, 0]]; B = F[P[k, 1]]
        m = 0.0
        for i in range(3):
            for j in range(3):
                r = D[A[i]] - D[B[j]]; m = max(m, np.sqrt(dot(r, r)))
        # explicit stack of intervals
        st0 = np.empty(2 * max_depth + 2); st1 = np.empty(2 * max_depth + 2); dep = np.empty(2 * max_depth + 2, np.int64)
        n = 1; st0[0] = 0.0; st1[0] = 1.0; dep[0] = 0
        while n > 0:
            n -= 1; t0 = st0[n]; t1 = st1[n]; dd = dep[n]
            a0 = X0[A[0]] + t0 * D[A[0]]; a1 = X0[A[1]] + t0 * D[A[1]]; a2 = X0[A[2]] + t0 * D[A[2]]
            b0 = X0[B[0]] + t0 * D[B[0]]; b1 = X0[B[1]] + t0 * D[B[1]]; b2 = X0[B[2]] + t0 * D[B[2]]
            d = tri_tri_dist(a0, a1, a2, b0, b1, b2)
            if d <= 0.0: ok[k] = False; break
            if d > (t1 - t0) * m * 1.001: continue
            if dd >= max_depth: ok[k] = False; break
            tm = 0.5 * (t0 + t1)
            st0[n] = t0; st1[n] = tm; dep[n] = dd + 1; n += 1
            st0[n] = tm; st1[n] = t1; dep[n] = dd + 1; n += 1
    return ok
