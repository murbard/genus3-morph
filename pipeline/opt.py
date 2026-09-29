"""IPC-style gradient descent: tension + volume + mesh quality + (optional) target, with a log-barrier against contact
and a certified step bound (no contact anywhere along each linear step)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, time, sys, os
from numba import njit, prange
import coll

@njit(cache=True)
def pt_tri_closest(p, a, b, c):
    """Closest point on triangle abc to p, as barycentric weights (wa, wb, wc)."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = coll.dot(ab, ap); d2 = coll.dot(ac, ap)
    if d1 <= 0 and d2 <= 0: return 1.0, 0.0, 0.0
    bp = p - b; d3 = coll.dot(ab, bp); d4 = coll.dot(ac, bp)
    if d3 >= 0 and d4 <= d3: return 0.0, 1.0, 0.0
    vc = d1*d4 - d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3); return 1 - v, v, 0.0
    cp = p - c; d5 = coll.dot(ab, cp); d6 = coll.dot(ac, cp)
    if d6 >= 0 and d5 <= d6: return 0.0, 0.0, 1.0
    vb = d5*d2 - d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6); return 1 - w, 0.0, w
    va = d3*d6 - d5*d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return 0.0, 1 - w, w
    tot = va + vb + vc
    if tot == 0.0: return 1.0, 0.0, 0.0      # degenerate triangle (only transiently, at a collapse)
    den = 1.0 / tot; v = vb * den; w = vc * den
    return 1 - v - w, v, w

@njit(cache=True)
def seg_seg_st(p1, q1, p2, q2):
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = coll.dot(d1, d1); e = coll.dot(d2, d2); f = coll.dot(d2, r)
    c = coll.dot(d1, r); b = coll.dot(d1, d2); den = a*e - b*b
    s = min(max((b*f - c*e) / den, 0.0), 1.0) if den > 1e-30 else 0.0
    t = (b*s + f) / e
    if t < 0: t = 0.0; s = min(max(-c / a, 0.0), 1.0)
    elif t > 1: t = 1.0; s = min(max((b - c) / a, 0.0), 1.0)
    return s, t

@njit(cache=True)
def tri_tri_closest(X, A, B, wa, wb):
    """Distance between vertex-disjoint triangles A, B (index triples) and barycentric weights of closest points."""
    best = 1e300
    for side in range(2):
        P = A if side == 0 else B; Q = B if side == 0 else A
        for i in range(3):
            u, v, w = pt_tri_closest(X[P[i]], X[Q[0]], X[Q[1]], X[Q[2]])
            q = u * X[Q[0]] + v * X[Q[1]] + w * X[Q[2]]
            r = X[P[i]] - q; d2 = coll.dot(r, r)
            if d2 < best:
                best = d2
                ww = np.zeros(3); ww[i] = 1.0
                if side == 0:
                    wa[:] = ww; wb[0] = u; wb[1] = v; wb[2] = w
                else:
                    wb[:] = ww; wa[0] = u; wa[1] = v; wa[2] = w
    for i in range(3):
        for j in range(3):
            i2 = (i + 1) % 3; j2 = (j + 1) % 3
            s, t = seg_seg_st(X[A[i]], X[A[i2]], X[B[j]], X[B[j2]])
            r = X[A[i]] + s * (X[A[i2]] - X[A[i]]) - X[B[j]] - t * (X[B[j2]] - X[B[j]])
            d2 = coll.dot(r, r)
            if d2 < best:
                best = d2; wa[:] = 0; wb[:] = 0
                wa[i] = 1 - s; wa[i2] = s; wb[j] = 1 - t; wb[j2] = t
    return np.sqrt(best)

@njit(parallel=True, cache=True)
def _pair_dist(X, F, P):
    out = np.empty(len(P))
    for k in prange(len(P)):
        A = F[P[k, 0]]; B = F[P[k, 1]]
        out[k] = coll.tri_tri_dist(X[A[0]], X[A[1]], X[A[2]], X[B[0]], X[B[1]], X[B[2]])
    return out

@njit(cache=True)
def _barrier_grad(X, F, P, idx, dhat, G):
    wa = np.zeros(3); wb = np.zeros(3)
    for k in idx:
        A = F[P[k, 0]]; B = F[P[k, 1]]
        d = tri_tri_closest(X, A, B, wa, wb)
        if d >= dhat or d <= 0: continue
        db = -2 * (d - dhat) * np.log(d / dhat) - (d - dhat)**2 / d
        pa = wa[0]*X[A[0]] + wa[1]*X[A[1]] + wa[2]*X[A[2]]
        pb = wb[0]*X[B[0]] + wb[1]*X[B[1]] + wb[2]*X[B[2]]
        u = (pa - pb) / d
        for i in range(3):
            G[A[i]] += db * wa[i] * u; G[B[i]] -= db * wb[i] * u

def barrier(X, F, P, dhat, want_grad):
    G = np.zeros_like(X)
    if len(P) == 0: return 0.0, G, np.inf
    d = _pair_dist(X, F, P); dmin = d.min()
    if dmin <= 0: return np.inf, G, 0.0
    act = np.where(d < dhat)[0]; da = d[act]
    E = float((-(da - dhat)**2 * np.log(da / dhat)).sum())
    if want_grad and len(act): _barrier_grad(X, F, P, act, dhat, G)
    return E, G, dmin

@njit(parallel=True, cache=True)
def ccd_alpha(X, F, P, D, safety):
    """Largest alpha in [0,1] such that moving X + alpha*D linearly keeps every candidate pair apart
    (distance can shrink by at most the max relative vertex displacement)."""
    out = np.ones(len(P)); wa = np.zeros(3); wb = np.zeros(3)
    for k in prange(len(P)):
        A = F[P[k, 0]]; B = F[P[k, 1]]
        d = coll.tri_tri_dist(X[A[0]], X[A[1]], X[A[2]], X[B[0]], X[B[1]], X[B[2]])
        m = 0.0
        for i in range(3):
            for j in range(3):
                r = D[A[i]] - D[B[j]]; m = max(m, np.sqrt(coll.dot(r, r)))
        if m > 0: out[k] = min(1.0, safety * d / m)
    return out.min() if len(out) else 1.0

_edge_cache = {}
def edge_tris(F):
    """Interior edges with their two triangles (cached per triangulation)."""
    key = (id(F), len(F))
    if key not in _edge_cache:
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); T = np.tile(np.arange(len(F)), 3)
        K = np.sort(E, 1); o = np.lexsort((K[:, 1], K[:, 0])); K = K[o]; T = T[o]
        _edge_cache.clear(); _edge_cache[key] = (T[0::2], T[1::2])
    return _edge_cache[key]

def fold_barrier(X, F, c0, want_grad):
    """Barrier on dihedral folding: for adjacent triangles with unit normals n1, n2 and c = n1.n2 < c0,
    b(c) = -(c - c0)^2 log((1 + c) / (1 + c0)), which blows up as the pair folds flat (c -> -1)."""
    t1, t2 = edge_tris(F)
    x0, x1, x2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    e1 = x1 - x0; e2 = x2 - x0; N = np.cross(e1, e2); nN = np.linalg.norm(N, axis=1); n = N / nN[:, None]
    c = (n[t1] * n[t2]).sum(1)
    act = c < c0
    if not act.any(): return 0.0, (np.zeros_like(X) if want_grad else None)
    ca = c[act]
    if (ca <= -1 + 1e-12).any(): return np.inf, None
    E = float((-(ca - c0)**2 * np.log((1 + ca) / (1 + c0))).sum())
    if not want_grad: return E, None
    db = -2 * (ca - c0) * np.log((1 + ca) / (1 + c0)) - (ca - c0)**2 / (1 + ca)
    G = np.zeros_like(X)
    for ta, tb in ((t1[act], t2[act]), (t2[act], t1[act])):
        gN = db[:, None] * (n[tb] - ca[:, None] * n[ta]) / nN[ta][:, None]      # d b / d N_a
        g1 = np.cross(e2[ta], gN); g2 = np.cross(gN, e1[ta])
        np.add.at(G, F[ta, 1], g1); np.add.at(G, F[ta, 2], g2); np.add.at(G, F[ta, 0], -(g1 + g2))
    return E, G

_lap_cache = {}
def umbrella(F, n):
    """M = D^-1 A - I (uniform Laplacian), cached per triangulation."""
    key = (id(F), len(F), n)
    if key not in _lap_cache:
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); A = (A + A.T).tocsr()
        M = (sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A - sp.eye(n)).tocsr()
        _lap_cache.clear(); _lap_cache[key] = (M, M.T.tocsr())
    return _lap_cache[key]

def tri_geom(X, F):
    x0, x1, x2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    fn = np.cross(x1 - x0, x2 - x0); A2 = np.linalg.norm(fn, axis=1)
    return x0, x1, x2, fn, A2

def smooth_energy(X, F, w, want_grad=True):
    """tension*area + k_vol*(V-V0)^2/V0 + quality + target. Returns E, grad."""
    x0, x1, x2, fn, A2 = tri_geom(X, F)
    if (A2 <= 1e-12).any(): return np.inf, None
    nhat = fn / A2[:, None]
    area = A2.sum() / 2
    V = np.einsum('ij,ij->i', x0, np.cross(x1, x2)).sum() / 6
    S = ((x1 - x0)**2).sum(1) + ((x2 - x1)**2).sum(1) + ((x0 - x2)**2).sum(1)
    Q = S / (2 * np.sqrt(3) * A2)                      # =1 for equilateral, -> inf when degenerate
    E = w['tension'] * area + w['kvol'] * (V - w['V0'])**2 / w['V0'] + w['quality'] * (Q - 1).sum()
    ku = w.get('uniform', 0.0); Abar = A2.mean() / 2
    if ku: E += ku * ((A2 / 2 / Abar - 1)**2).sum() * Abar          # equalize triangle areas (tangential redistribution)
    ka = w.get('kalign', 0.0); Ga = None
    if ka and w.get('target_grad') is not None:
        # untangling: sum_t (A_t - N_t . g_t / 2), zero iff every triangle lies flat on the target facing outward
        if w.get('align_sigma') and w.get('target') is not None:
            fc, g = w['target']((x0 + x1 + x2) / 3)
            wt = np.exp(-(fc / w['align_sigma'])**2)      # only where the surface is already near the target
        else:
            g = w['target_grad']((x0 + x1 + x2) / 3); wt = np.ones(len(F))
        E += ka * (wt * (A2 / 2 - 0.5 * (fn * g).sum(1))).sum()
        if want_grad:
            e1 = x1 - x0; e2 = x2 - x0
            gA0 = 0.5 * np.cross(nhat, x2 - x1); gA1 = 0.5 * np.cross(nhat, x0 - x2); gA2 = 0.5 * np.cross(nhat, x1 - x0)
            g1 = gA1 - 0.5 * np.cross(e2, g); g2 = gA2 - 0.5 * np.cross(g, e1); g0 = gA0 + 0.5 * (np.cross(e2, g) + np.cross(g, e1))
            Ga = np.zeros_like(X)
            for k, gk in enumerate((g0, g1, g2)): np.add.at(Ga, F[:, k], ka * wt[:, None] * gk)
    kp = w.get('kperim', 0.0); Gp = None
    if kp and w.get('L') is not None and len(w['L']) == len(F):
        Lab = w['L']; t1, t2 = edge_tris(F)
        bd = Lab[t1] != Lab[t2]
        # the shared edge's vertices
        sh = np.array([[u for u in F[a] if u in F[b]] for a, b in zip(t1[bd], t2[bd])])
        ev = X[sh[:, 0]] - X[sh[:, 1]]; el = np.linalg.norm(ev, axis=1)
        E += kp * el.sum()
        # face-area balance: each region keeps its initial share of the total area
        Ar = np.bincount(Lab, A2 / 2, 8); At = Ar.sum(); phi = w['phi']
        ka_r = w.get('karea_reg', 1.0); dev = Ar - phi * At
        E += ka_r * (dev * dev).sum() / At
        if want_grad:
            Gp = np.zeros_like(X); u = ev / el[:, None]
            np.add.at(Gp, sh[:, 0], kp * u); np.add.at(Gp, sh[:, 1], -kp * u)
            cA = 2 * ka_r * (dev[Lab] - (dev * phi).sum()) / At            # dE/dA_t
            dA = (0.5 * np.cross(nhat, x2 - x1), 0.5 * np.cross(nhat, x0 - x2), 0.5 * np.cross(nhat, x1 - x0))
            for k in range(3): np.add.at(Gp, F[:, k], cA[:, None] * dA[k])
            # tangential only: seams slide along the surface instead of denting it
            vnrm = np.zeros_like(X)
            for k in range(3): np.add.at(vnrm, F[:, k], fn)
            vnrm /= np.maximum(np.linalg.norm(vnrm, axis=1, keepdims=True), 1e-12)
            Gp -= (Gp * vnrm).sum(1, keepdims=True) * vnrm
    kl = w.get('klap', 0.0); Gl = None
    if kl:   # mesh smoothness: squared umbrella Laplacian (penalizes zig-zags / pleats at the mesh scale)
        M, MT = umbrella(F, len(X)); MX = M @ X
        E += kl * (MX * MX).sum()
        if want_grad:
            Gl = 2 * kl * (MT @ MX)
            if w.get('klap_normal_only'):   # leave tangential sliding (seams) free; only straighten the surface
                vnl = np.zeros_like(X)
                for k in range(3): np.add.at(vnl, F[:, k], fn)
                vnl /= np.maximum(np.linalg.norm(vnl, axis=1, keepdims=True), 1e-12)
                Gl = (Gl * vnl).sum(1, keepdims=True) * vnl
    kf = w.get('kfold', 0.0); Gf = None
    if kf:
        Ef, Gf = fold_barrier(X, F, w.get('fold_c0', -0.3), want_grad)
        if not np.isfinite(Ef): return np.inf, None
        E += kf * Ef
    tgt = None
    if w.get('target') is not None and w.get('ktarget', 0) > 0:
        f, gf = w['target'](X)                          # signed distance to target surface and its gradient
        va = np.bincount(F.ravel(), np.repeat(A2 / 6, 3), len(X))
        if w.get('split'):
            # outside the target: pull along the smooth ambient direction -grad f (flaps slide back, both layers alike)
            fo = np.maximum(f, 0.0); E += w['ktarget'] * (va * fo * fo).sum(); tgt = (va, fo, gf, f)
        else:
            E += w['ktarget'] * (va * f * f).sum(); tgt = (va, f, gf, f)
    if not want_grad: return E, None
    G = np.zeros_like(X)
    # area gradient: dA/dx_i = 0.5 * nhat x (x_k - x_j) for (i,j,k) cyclic
    dA0 = 0.5 * np.cross(nhat, x2 - x1); dA1 = 0.5 * np.cross(nhat, x0 - x2); dA2 = 0.5 * np.cross(nhat, x1 - x0)
    dV0 = np.cross(x1, x2) / 6; dV1 = np.cross(x2, x0) / 6; dV2 = np.cross(x0, x1) / 6
    cv = 2 * w['kvol'] * (V - w['V0']) / w['V0']
    A = A2 / 2; c1 = 1 / (4 * np.sqrt(3) * A); c2 = S / (4 * np.sqrt(3) * A * A)
    dQ0 = (2 * (2 * x0 - x1 - x2)) * c1[:, None] - dA0 * c2[:, None]
    dQ1 = (2 * (2 * x1 - x2 - x0)) * c1[:, None] - dA1 * c2[:, None]
    dQ2 = (2 * (2 * x2 - x0 - x1)) * c1[:, None] - dA2 * c2[:, None]
    for k, (dA, dV, dQ) in enumerate(((dA0, dV0, dQ0), (dA1, dV1, dQ1), (dA2, dV2, dQ2))):
        g = w['tension'] * dA + cv * dV + w['quality'] * dQ
        if ku: g = g + (ku * 2 * (A2 / 2 / Abar - 1))[:, None] * dA
        np.add.at(G, F[:, k], g)
    if tgt is not None:
        va, f, gf, _ = tgt
        G += w['ktarget'] * (2 * va * f)[:, None] * gf
    if Gf is not None: G += kf * Gf
    if Gl is not None: G += Gl
    if Ga is not None: G += Ga
    if Gp is not None: G += Gp
    # region term E = kregion * integral of f over the solid: its gradient is f(x_v) * (area-weighted normal).
    # Returned separately: it drives the direction but is not part of the line-search energy.
    w['_Gregion'] = None
    if w.get('kregion', 0) > 0:
        f = tgt[3] if tgt is not None else w['target'](X)[0]
        if w.get('split'): f = np.minimum(f, 0.0)     # inside the target: inflate along the surface normal
        Nv = np.zeros_like(X)
        for k in range(3): np.add.at(Nv, F[:, k], fn / 6)
        w['_Gregion'] = w['kregion'] * f[:, None] * Nv
    return E, G

class Optimizer:
    def __init__(self, X, F, w, dhat=1.0, kbar=1.0, sigma=20.0, step_max=0.5, V0=None, vol_rate=0.2):
        self.V0 = V0; self.vol_rate = vol_rate
        self.F = F.astype(np.int64); self.w = w; self.dhat = dhat; self.kbar = kbar; self.step_max = step_max
        n = len(X)
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        Adj = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); Adj = (Adj + Adj.T).tocsr()
        K = sp.diags(np.asarray(Adj.sum(1)).ravel()) - Adj
        self.pre = spla.splu((sp.eye(n) + sigma * K).tocsc())    # Sobolev (H1) preconditioner
        self.vt_ptr, self.vt_idx = coll.vertex_tri_csr(self.F, n)

    def energy(self, X, P, grad=True):
        Es, Gs = smooth_energy(X, self.F, self.w, grad)
        if not np.isfinite(Es): return np.inf, None, 0
        Eb, Gb, dmin = barrier(X, self.F, P, self.dhat, grad)
        E = Es + self.kbar * Eb
        return E, (Gs + self.kbar * Gb) if grad else None, dmin

    def step(self, X):
        reach = self.step_max + self.dhat
        P = coll.unique_pairs(coll.candidate_pairs(X, self.F, reach))
        E0, G, dmin = self.energy(X, P)
        Gr = self.w.get('_Gregion')
        D = -self.pre.solve(G if Gr is None else G + Gr)
        if self.w.get('seam_pull'): D = self._seam_moves(X, D)
        # hard volume constraint: project the (preconditioned) direction so volume moves toward V0
        if self.V0 is None: return self._line_search(X, P, E0, G, D, dmin)
        x0, x1, x2, _, _ = tri_geom(X, self.F)
        V = np.einsum('ij,ij->i', x0, np.cross(x1, x2)).sum() / 6
        gV = np.zeros_like(X)
        for k, g in enumerate((np.cross(x1, x2), np.cross(x2, x0), np.cross(x0, x1))): np.add.at(gV, self.F[:, k], g / 6)
        Dv = self.pre.solve(gV)
        want = self.vol_rate * (self.V0 - V) / max(self.step_max / max(np.linalg.norm(D, axis=1).max(), 1e-12), 1e-12)
        lam = ((gV * D).sum() - want) / (gV * Dv).sum()
        D = D - lam * Dv
        return self._line_search(X, P, E0, G, D, dmin)

    def _seam_moves(self, X, D):
        """Seam vertices: lock sideways drift from the other terms, and pull each one (tangentially) toward the midpoint
        of its two seam neighbours - curve shortening of the seams themselves. Junctions stay put."""
        import seams
        key = (id(self.F), len(self.F))
        if getattr(self, '_chains_key', None) != key:
            self._chains = [np.array(c) for c in seams.chains(self.F, self.w['L'])]; self._chains_key = key
        N = seams.vertex_normals(X, self.F)
        alpha0 = self.step_max / max(np.linalg.norm(D, axis=1).max(), 1e-12)
        # only where the surface has already reached the target (seams of retracting plates must move freely)
        fv = self.w['target'](X)[0]; near = np.exp(-(fv / self.w.get('seam_sigma', 5.0))**2)
        D = D.copy()
        for c in self._chains:
            if len(c) < 3: continue
            mid = c[1:-1]
            t = X[c[2:]] - X[c[:-2]]; t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
            b = np.cross(N[mid], t)                                   # in the surface, across the seam
            wn = near[mid][:, None]
            D[mid] -= wn * (D[mid] * b).sum(1, keepdims=True) * b      # other terms may not push seams sideways
            m = 0.5 * (X[c[:-2]] + X[c[2:]]) - X[mid]
            m -= (m * N[mid]).sum(1, keepdims=True) * N[mid]           # tangential only
            D[mid] += wn * (self.w['seam_pull'] / alpha0) * m
        return D

    def _line_search(self, X, P, E0, G, D, dmin):
        mx = np.linalg.norm(D, axis=1).max()
        alpha = min(1.0, self.step_max / mx)
        a_ccd = ccd_alpha(X, self.F, P, D * alpha, 0.8)
        alpha *= a_ccd; halvings = 0
        slope = min((G * D).sum(), 0.0)
        for _ in range(30):
            Xt = X + alpha * D
            Et, _, _ = self.energy(Xt, P, grad=False)
            decrease_ok = self.w.get('kregion', 0) > 0 or Et <= E0 + 1e-4 * alpha * slope   # safety never depends on this
            if np.isfinite(Et) and decrease_ok and \
               coll.vertex_adjacent_violations(Xt, self.F, self.vt_ptr, self.vt_idx) == 0:
                return Xt, dict(E=Et, dmin=dmin, alpha=alpha, move=alpha * mx, pairs=len(P), a_ccd=a_ccd, halvings=halvings)
            alpha *= 0.5; halvings += 1
        return X, dict(E=E0, dmin=dmin, alpha=0.0, move=0.0, pairs=len(P), a_ccd=a_ccd, halvings=halvings)

if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    m = np.load(os.path.join(HERE, 'mesh0.npz')); X = m['X'].astype(float); F = m['F']
    args = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = args.pop('out', 'opt_run.npz'); args_every = args.pop('every', 20); nit = args.pop('iters', 300); start = args.pop('start', None)
    frames = [X.astype(np.float32)]
    if start:
        frames = list(np.load(os.path.join(HERE, start))['frames']); X = frames[-1].astype(float)
    x0, x1, x2, _, _ = tri_geom(m['X'].astype(float), F.astype(np.int64))
    V_P = np.einsum('ij,ij->i', x0, np.cross(x1, x2)).sum() / 6
    w = dict(tension=1.0, kvol=0.0, V0=V_P, quality=args.pop('quality', 5.0))
    opt = Optimizer(X, F, w, V0=V_P * args.pop('vol_factor', 1.0), **args)
    t0 = time.time(); acc = 0
    for it in range(nit):
        X, info = opt.step(X)
        acc += info['move']
        if acc >= 1.0: frames.append(X.astype(np.float32)); acc = 0
        if it % args_every == 0 or info['alpha'] == 0:
            x0, x1, x2, fn, A2 = tri_geom(X, opt.F)
            V = np.einsum('ij,ij->i', x0, np.cross(x1, x2)).sum() / 6
            print(it, 'ccd %.3g halv %d E %.4g dmin %.3f alpha %.3g move %.3f area %.0f vol/V_P %.3f pairs %d frames %d %.0fs' %
                  (info['a_ccd'], info['halvings'], info['E'], info['dmin'], info['alpha'], info['move'], A2.sum() / 2, V / V_P, info['pairs'], len(frames), time.time() - t0), flush=True)
        if it % 100 == 99:
            np.savez(os.path.join(HERE, out), frames=np.array(frames + [X.astype(np.float32)]), F=F, L=m['L'])
    frames.append(X.astype(np.float32))
    np.savez(os.path.join(HERE, out), frames=np.array(frames), F=F, L=m['L'])
