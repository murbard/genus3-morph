"""Certifiable remeshing: edge collapses (realised as a certified motion of both endpoints to the midpoint, after which
the two degenerate triangles are dropped) and edge splits (exact: the new vertex lies on the old edge)."""
import numpy as np
import coll, verify


def _edges(F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.sort(E, 1)


def _quality(X, F):
    x0, x1, x2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    A2 = np.linalg.norm(np.cross(x1 - x0, x2 - x0), axis=1)
    S = ((x1 - x0)**2).sum(1) + ((x2 - x1)**2).sum(1) + ((x0 - x2)**2).sum(1)
    return S / (2 * np.sqrt(3) * np.maximum(A2, 1e-12)), A2


def plan_collapses(X, F, ell, max_frac=0.05, L=None):
    nv = len(X)
    E = np.unique(_edges(F), axis=0)
    elen = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    Q, _ = _quality(X, F)
    cand = set(map(tuple, E[elen < 0.5 * ell]))
    for t in np.where(Q > 3.0)[0]:                      # badly shaped triangle: its shortest edge
        tri = F[t]; pairs = [(tri[i], tri[(i + 1) % 3]) for i in range(3)]
        a, b = min(pairs, key=lambda p: np.linalg.norm(X[p[0]] - X[p[1]]))
        cand.add((min(a, b), max(a, b)))
    # sharp creases (pleats): the crease edge itself and the shortest edge of each creased triangle, highest priority
    Ts = np.tile(np.arange(len(F)), 3); K = _edges(F); o = np.lexsort((K[:, 1], K[:, 0])); K = K[o]; Ts = Ts[o]
    fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); nn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    cc = (nn[Ts[0::2]] * nn[Ts[1::2]]).sum(1); crease = set()
    for i in np.where(cc < 0.2)[0]:
        crease.add(tuple(K[2 * i]))
        for t in (Ts[2 * i], Ts[2 * i + 1]):
            tri = F[t]; pairs = [(tri[j], tri[(j + 1) % 3]) for j in range(3)]
            a, b = min(pairs, key=lambda p: np.linalg.norm(X[p[0]] - X[p[1]])); crease.add((min(a, b), max(a, b)))
    # seam teeth: a triangle with two edges on seams can never lie flat along a straight seam; collapse its tip into
    # the nearer base corner (the seam then runs straight along the old base)
    if TEETH and L is not None:
        Lt = np.asarray(L); nb_lab = {}
        for i in range(len(K) // 2):
            a_, b_ = K[2 * i]; t1_, t2_ = Ts[2 * i], Ts[2 * i + 1]
            if Lt[t1_] != Lt[t2_]:
                for t_ in (t1_, t2_): nb_lab.setdefault(int(t_), []).append((int(a_), int(b_)))
        for t_, es in nb_lab.items():
            if len(es) != 2: continue
            (p0, p1), (q0, q1) = es
            apex = ({p0, p1} & {q0, q1})
            if len(apex) != 1: continue
            b_ = apex.pop(); a1 = p0 if p1 == b_ else p1; a2 = q0 if q1 == b_ else q1
            a_ = a1 if np.linalg.norm(X[a1] - X[b_]) <= np.linalg.norm(X[a2] - X[b_]) else a2
            crease.add((min(a_, b_), max(a_, b_)))
    cand = sorted(crease, key=lambda e: np.linalg.norm(X[e[0]] - X[e[1]])) + \
           sorted(cand - crease, key=lambda e: np.linalg.norm(X[e[0]] - X[e[1]]))
    vn = np.zeros_like(X)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    # adjacency
    nbr = [set() for _ in range(nv)]
    for a, b in E: nbr[a].add(b); nbr[b].add(a)
    vtri = [[] for _ in range(nv)]
    for t, tri in enumerate(F):
        for v in tri: vtri[v].append(t)
    locked = np.zeros(nv, bool); chosen = []
    if SEAM_AWARE and L is not None:
        vl = [set() for _ in range(nv)]
        for t, tri in enumerate(F):
            for v in tri: vl[v].add(int(L[t]))
        kind = np.array([len(x) for x in vl])                        # 1 interior, 2 seam, >=3 junction
    for a, b in cand:
        if SEAM_AWARE and L is not None:
            # keep seams on seams: collapse only inside a face, or along a seam between two seam vertices
            if kind[a] >= 3 or kind[b] >= 3: continue
            if kind[a] != kind[b]: continue
            if kind[a] == 2:
                sh = [t for t in vtri[a] if b in F[t]]
                if len(sh) != 2 or L[sh[0]] == L[sh[1]]: continue
        if locked[a] or locked[b]: continue
        shared = [t for t in vtri[a] if b in F[t]]
        if len(shared) != 2: continue
        opp = {int(v) for t in shared for v in F[t] if v != a and v != b}
        if nbr[a] & nbr[b] != opp: continue            # link condition (keeps a 2-manifold)
        if any(len(nbr[o]) <= 3 for o in opp): continue
        if len(nbr[a]) + len(nbr[b]) - 4 > 10: continue   # avoid creating very high valence
        m = 0.5 * (X[a] + X[b]); ok = True
        is_crease = (a, b) in crease
        ring = set(vtri[a]) | set(vtri[b])
        ref = sum(vn[v] for t in ring for v in F[t])       # smoothed normal of the neighbourhood
        for t in ring:
            if t in shared: continue
            tri = F[t]; P = X[tri].copy()
            n0 = ref if is_crease else np.cross(P[1] - P[0], P[2] - P[0])
            P[(tri == a) | (tri == b)] = m
            n1 = np.cross(P[1] - P[0], P[2] - P[0])
            if np.dot(n0, n1) <= 0.3 * np.linalg.norm(n0) * np.linalg.norm(n1): ok = False; break
            e = [np.linalg.norm(P[i] - P[(i + 1) % 3]) for i in range(3)]
            if sum(x * x for x in e) / (2 * np.sqrt(3) * max(np.linalg.norm(n1), 1e-12)) > 6: ok = False; break
        if not ok: continue
        chosen.append((a, b))
        locked[a] = locked[b] = True
        for v in nbr[a] | nbr[b]: locked[v] = True
        if len(chosen) >= max_frac * nv: break
    return chosen


def apply_collapses_remap(X, F, chosen):
    remap = np.arange(len(X))
    for a, b in chosen: remap[b] = a
    F2 = remap[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])
    return F2[keep].ravel()


def apply_collapses(X, F, L, chosen):
    """Returns (X_moved, same F) for the certified motion, and the new (X, F, L) after the merge."""
    Xm = X.copy()
    for a, b in chosen:
        m = 0.5 * (X[a] + X[b]); Xm[a] = m; Xm[b] = m
    remap = np.arange(len(X))
    for a, b in chosen: remap[b] = a
    F2 = remap[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])
    F2 = F2[keep]; L2 = L[keep]
    used = np.zeros(len(X), bool); used[F2.ravel()] = True
    newid = -np.ones(len(X), np.int64); newid[used] = np.arange(used.sum())
    return Xm, Xm[used], newid[F2], L2


def splits(X, F, L, ell):
    E = _edges(F)
    elen = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1).reshape(3, -1).T      # per triangle, per edge
    long = elen > 1.6 * ell
    if not long.any(): return X, F, L
    Es = np.unique(E.reshape(3, -1, 2).transpose(1, 0, 2)[long].reshape(-1, 2), axis=0)
    # at most one split edge per triangle: greedy over longest edges
    order = np.argsort(-np.linalg.norm(X[Es[:, 0]] - X[Es[:, 1]], axis=1))
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(Es)}
    tri_of_edge = {}
    for t, tri in enumerate(F):
        for i in range(3):
            a, b = sorted((int(tri[i]), int(tri[(i + 1) % 3])))
            if (a, b) in key: tri_of_edge.setdefault((a, b), []).append(t)
    busy = np.zeros(len(F), bool); pick = []
    for i in order:
        e = (int(Es[i, 0]), int(Es[i, 1])); ts = tri_of_edge.get(e, [])
        if len(ts) != 2 or busy[ts].any(): continue
        busy[ts] = True; pick.append(e)
    Xn = list(X); Fn = [tuple(t) for t in F]; Ln = list(L)
    for a, b in pick:
        m = len(Xn); Xn.append(0.5 * (X[a] + X[b]))
        for t in tri_of_edge[(a, b)]:
            tri = Fn[t]; i = [k for k in range(3) if tri[k] in (a, b) and tri[(k + 1) % 3] in (a, b)][0]
            p, q, r = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
            Fn[t] = (p, m, r); Fn.append((m, q, r)); Ln.append(Ln[t])
    return np.array(Xn), np.array(Fn, np.int64), np.array(Ln)


def _certify_local(X0, X1, F, P, vt, verts, depth=0, max_depth=10, final=True):
    """Certify linear motion X0->X1. At the collapse endpoint (final) the two edge triangles are degenerate, so the
    adjacency test there is done on the post-collapse mesh by the caller instead."""
    import opt
    D = X1 - X0
    if not final and coll.adjacent_violations_at(X1, F, vt[0], vt[1], verts) != 0: return False
    a = opt.ccd_alpha(X0, F, P, D, 0.999) if len(P) else 1.0
    if a >= 1.0: return True
    if depth >= max_depth: return False
    Xm = 0.5 * (X0 + X1)
    return (_certify_local(X0, Xm, F, P, vt, verts, depth + 1, max_depth, False) and
            _certify_local(Xm, X1, F, P, vt, verts, depth + 1, max_depth, final))


def _post_ok(Xc, Fc, merged):
    vt = coll.vertex_tri_csr(Fc, len(Xc))
    return coll.adjacent_violations_at(Xc, Fc, vt[0], vt[1], np.asarray(merged, np.int64)) == 0


def fans_contiguous(F, L, vt, v):
    """At vertex v, the incident triangles of each face label must form one contiguous fan (no pinch)."""
    ts = vt[1][vt[0][v]:vt[0][v + 1]]
    if len(ts) < 2: return True
    labs = L[ts]
    if len(set(labs.tolist())) == 1: return True
    # adjacency of incident triangles through edges containing v
    other = {}
    for i, t in enumerate(ts):
        for u in F[t]:
            if u != v: other.setdefault(int(u), []).append(i)
    parent = list(range(len(ts)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for u, lst in other.items():
        if len(lst) == 2 and labs[lst[0]] == labs[lst[1]]: parent[find(lst[0])] = find(lst[1])
    for lab in set(labs.tolist()):
        if len({find(i) for i in range(len(ts)) if labs[i] == lab}) > 1: return False
    return True


NECK_RULE = False
TEETH = False
SEAM_AWARE = False


def neck_triangles(F, L, min_side=3):
    """Triangles whose removal splits their face region into two parts of >= min_side triangles (one-triangle necks)."""
    import networkx as nx
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); T = np.tile(np.arange(len(F)), 3)
    K = np.sort(E, 1); o = np.lexsort((K[:, 1], K[:, 0])); T = T[o]; t1, t2 = T[0::2], T[1::2]
    out = set()
    for r in np.unique(L):
        m = (L[t1] == r) & (L[t2] == r)
        G = nx.Graph(); G.add_nodes_from(np.where(L == r)[0].tolist()); G.add_edges_from(zip(t1[m].tolist(), t2[m].tolist()))
        for a in nx.articulation_points(G):
            H = G.copy(); H.remove_node(a); sizes = sorted(len(c) for c in nx.connected_components(H))
            if len(sizes) >= 2 and sizes[-2] >= min_side: out.add(int(a))
    return out


def certified_collapses(X, F, L, chosen, rounds=4):
    """Certify the combined collapse motion; drop collapses involved in any failing check and retry."""
    vt = coll.vertex_tri_csr(F, len(X))
    for _ in range(rounds):
        if not chosen: break
        Xm, Xc, Fc, Lc = apply_collapses(X, F, L, chosen)
        D = Xm - X
        P = coll.unique_pairs(coll.candidate_pairs(X, F, np.linalg.norm(D, axis=1).max() + 1e-6))
        mate = -np.ones(len(X), np.int64); cid = -np.ones(len(X), np.int64)
        for k, (a, b) in enumerate(chosen): mate[a] = b; mate[b] = a; cid[a] = cid[b] = k
        A = F[P[:, 0]]; B = F[P[:, 1]]
        merging = np.zeros(len(P), bool)
        for i in range(3):
            for j in range(3): merging |= (mate[A[:, i]] == B[:, j]) & (mate[A[:, i]] >= 0)
        moving = (np.abs(D[A]).sum((1, 2)) + np.abs(D[B]).sum((1, 2))) > 0
        Pm = P[moving & ~merging]
        good = coll.certify_pairs(X, D, F, Pm, 14)
        bad = set()
        for T in (Pm[~good, 0], Pm[~good, 1]):
            for v in F[T].ravel():
                if cid[v] >= 0: bad.add(int(cid[v]))
        Pg = P[merging]
        if len(Pg):
            for s_ in np.linspace(0.05, 0.95, 12):
                dd = coll.pair_dists(X + s_ * D, F, Pg)
                for T in (Pg[dd <= 0, 0], Pg[dd <= 0, 1]):
                    for v in F[T].ravel():
                        if cid[v] >= 0: bad.add(int(cid[v]))
        moved = np.where(np.abs(D).sum(1) > 0)[0]
        verts = np.unique(F[np.isin(F, moved).any(1)].ravel())
        for s_ in np.linspace(0.1, 0.9, 9):
            if coll.adjacent_violations_at(X + s_ * D, F, vt[0], vt[1], verts) != 0:
                # locate offending collapses
                for k, (a, b) in enumerate(chosen):
                    vv = np.unique(F[(F == a).any(1) | (F == b).any(1)].ravel())
                    if coll.adjacent_violations_at(X + s_ * D, F, vt[0], vt[1], vv) != 0: bad.add(k)
        # post-collapse adjacency at merged vertices
        used = np.zeros(len(X), bool); used[Fc.ravel() * 0 + apply_collapses_remap(X, F, chosen)] = True
        newid = np.cumsum(used) - 1
        vtc = coll.vertex_tri_csr(Fc, len(Xc))
        for k, (a, b) in enumerate(chosen):
            if coll.adjacent_violations_at(Xc, Fc, vtc[0], vtc[1], np.array([newid[a]], np.int64)) != 0: bad.add(k); continue
            # face regions must keep their topology: no pinch at the merged vertex or the two opposite vertices
            opp = {int(u) for t in np.where((F == a).any(1) & (F == b).any(1))[0] for u in F[t] if u != a and u != b}
            for v in [a] + list(opp):
                if not fans_contiguous(Fc, Lc, vtc, newid[v]): bad.add(k); break
        # face regions must not get thinner than one triangle: no new neck triangles
        if not bad and NECK_RULE:
            before = len(neck_triangles(F, L)); necks = neck_triangles(Fc, Lc)
            if len(necks) > before:
                Vn = set(Fc[list(necks)].ravel().tolist())
                for k, (a, b) in enumerate(chosen):
                    if newid[a] in Vn or any(newid[u] in Vn for u in np.unique(F[(F == a).any(1) | (F == b).any(1)].ravel())): bad.add(k)
                if not bad: bad = set(range(len(chosen)))
        if not bad: return chosen, Xm, (Xc, Fc, Lc)
        chosen = [c for k, c in enumerate(chosen) if k not in bad]
    return [], X, (X, F, L)


def remesh(X, F, L, ell):
    """Returns (motion_end, new_state) where motion_end is X after collapse motion (old F), or None if nothing to do,
    and new_state = (X, F, L) on the new connectivity with the same geometry."""
    F = F.astype(np.int64)
    chosen = plan_collapses(X, F, ell, L=L)
    chosen, Xm, (Xc, Fc, Lc) = certified_collapses(X, F, L, chosen) if chosen else ([], X, (X, F, L))
    Xs, Fs, Ls = splits(Xc, Fc, Lc, ell)
    return Xm, (Xs, Fs, Ls), len(chosen), len(Fs) - len(Fc)
