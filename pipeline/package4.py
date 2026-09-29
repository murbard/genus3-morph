"""Compact package for the web: morph.bin (binary, internally gzip'd blobs) + certification of exactly what is shown.

Stored: epoch 0's triangles; for every remesh round only its collapse pairs and split edges (the viewer replays them);
face labels per epoch; positions only for frames that cannot be derived. Derived frames:
  * collapse end of an epoch = previous keyframe with each collapsed pair at its midpoint,
  * first frame of the next epoch = that geometry renumbered, plus split vertices at their edge midpoints.
Seams are smoothed by moving seam vertices (tangentially, clamped) on the stored frames; colours stay per triangle.
Every displayed interval (including each collapse) is certified; seam smoothing is backed off where it fails."""
import numpy as np, sys, json, gzip, pickle, time, struct
from scipy.spatial import cKDTree
import coll, verify, relabel, seams
from package import select

SCALE = 0.05
def qi(x): return np.floor(x / SCALE + 0.5).astype(np.int64)      # same rounding as the viewer (floor(x + 1/2))


def collapse_apply(F, pairs, nv):
    """Exactly what the viewer does: b -> a for each (a < b) pair, drop degenerate triangles, compact ids."""
    remap = np.arange(nv)
    for a, b in pairs: remap[b] = a
    F2 = remap[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])
    F2 = F2[keep]
    used = np.zeros(nv, bool); used[F2.ravel()] = True
    newid = -np.ones(nv, np.int64); newid[used] = np.arange(used.sum())
    return newid[F2], keep, newid, remap


def split_apply(F, L, picks, nv):
    """Exactly what the viewer does: for each picked edge (a, b) in order, append a vertex and split its two triangles
    (in increasing triangle order) into (p, m, r) in place and (m, q, r) appended."""
    Fl = [list(t) for t in F]; Ll = list(L); n = nv
    edge_tris = {}
    for t, tri in enumerate(F):
        for i in range(3):
            key = (min(tri[i], tri[(i + 1) % 3]), max(tri[i], tri[(i + 1) % 3])); edge_tris.setdefault(key, []).append(t)
    for a, b in picks:
        m = n; n += 1
        for t in sorted(edge_tris[(min(a, b), max(a, b))]):
            tri = Fl[t]; i = [k for k in range(3) if tri[k] in (a, b) and tri[(k + 1) % 3] in (a, b)][0]
            p, q, r = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
            Fl[t] = [p, m, r]; Fl.append([m, q, r]); Ll.append(Ll[t])
    return np.array(Fl, np.int64), np.array(Ll), n


def recover_round(Fa, Xa_last, Fb, Xb_first, collapse_end):
    """Collapse pairs and split edges that turn epoch a's last state into epoch b's first state."""
    nv = len(Xa_last)
    pairs = []
    if collapse_end:
        E = np.unique(np.sort(np.concatenate([Fa[:, [0, 1]], Fa[:, [1, 2]], Fa[:, [2, 0]]]), 1), axis=0)
        z = np.linalg.norm(Xa_last[E[:, 0]] - Xa_last[E[:, 1]], axis=1) < 1e-9
        pairs = [tuple(map(int, e)) for e in E[z]]
    Fc, keep, newid, _ = collapse_apply(Fa, pairs, nv)
    ncol = int((newid >= 0).sum())
    Xc = np.zeros((ncol, 3)); Xc[newid[newid >= 0]] = Xa_last[newid >= 0]
    # split vertices: ids >= ncol in epoch b, in order; endpoints = the two collapsed-mesh neighbours whose midpoint it is
    picks = []
    for m in range(ncol, len(Xb_first)):
        nb = np.unique(Fb[(Fb == m).any(1)].ravel()); nb = nb[(nb != m) & (nb < ncol)]
        best = None
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                d = np.linalg.norm(0.5 * (Xc[nb[i]] + Xc[nb[j]]) - Xb_first[m])
                if best is None or d < best[0]: best = (d, int(nb[i]), int(nb[j]))
        picks.append((best[1], best[2]))
    return pairs, picks, ncol


def derive_next_first(X_collapsed_old, pairs, picks, Fa, nv_old):
    _, _, newid, _ = collapse_apply(Fa, pairs, nv_old)
    ncol = int((newid >= 0).sum()); X = np.zeros((ncol + len(picks), 3))
    X[newid[newid >= 0]] = X_collapsed_old[newid >= 0]
    for k, (a, b) in enumerate(picks): X[ncol + k] = 0.5 * (X[a] + X[b])
    return X


def gz(arr): return gzip.compress(np.ascontiguousarray(arr).tobytes(), 9)


def planes(q):
    """int32 -> zigzag -> two byte planes (low, high) of uint16; values must fit in 16 bits after zigzag."""
    z = np.where(q >= 0, 2 * q, -2 * q - 1).astype(np.int64); assert z.max() < 65536
    u = z.astype(np.uint16).view(np.uint8).reshape(-1, 2)
    return np.ascontiguousarray(u[:, 0]).tobytes() + np.ascontiguousarray(u[:, 1]).tobytes()


if __name__ == '__main__':
    phaseA, flow, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dmax = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    t0 = time.time()
    A = np.load(phaseA); ep = pickle.load(open(flow, 'rb'))['epochs']
    epochs = [dict(F=A['F'].astype(np.int64), L=A['L'], frames=np.concatenate([A['frames'], ep[0]['frames'][1:]]).astype(np.float64), n_stage1=len(A['frames']))]
    epochs += [dict(F=e['F'].astype(np.int64), L=e['L'], frames=e['frames'].astype(np.float64)) for e in ep[1:]]
    K = len(epochs)
    # ---- rounds, keyframes, display labels ----
    for k, e in enumerate(epochs):
        fr, F = e['frames'], e['F']
        El = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        e['ce'] = ce = k + 1 < K and len(fr) > 1 and bool((np.linalg.norm(fr[-1][El[:, 0]] - fr[-1][El[:, 1]], axis=1) < 1e-9).any())
        keep = select(fr[:-1] if ce else fr, F, dmax) if len(fr) > 1 else [0]
        keep = sorted(set(keep) | {len(fr) - 2 if ce else len(fr) - 1} | ({len(fr) - 1} if ce else set()))
        if k == 0: keep = sorted(set(keep) | {e['n_stage1'] - 1}); e['stage1_idx'] = keep.index(e['n_stage1'] - 1)
        e['keep'] = keep
        e['Lshow'] = e['L']   # labels exactly as simulated (per-epoch trimming caused sudden recolouring at remesh switches)
        e['chains'] = seams.chains(F, e['Lshow']); e['tips'] = seams.tooth_tips(F, e['Lshow'])
    for k in range(K - 1):
        a, b = epochs[k], epochs[k + 1]
        pairs, picks, ncol = recover_round(a['F'], a['frames'][-1], b['F'], b['frames'][0], a['ce'])
        Fc, keep, _, _ = collapse_apply(a['F'], pairs, len(a['frames'][0]))
        Fs, _, n = split_apply(Fc, np.zeros(len(Fc)), picks, ncol)
        assert np.array_equal(Fs, b['F']) and n == len(b['frames'][0]), f'round {k}: replay does not reproduce the triangulation'
        a['pairs'], a['picks'] = pairs, picks
    print('rounds recovered and replay-verified (%.0fs)' % (time.time() - t0), flush=True)

    # ---- displayed frames: stored (quantized, seam-smoothed) + derived; certify; back off seam smoothing on failures ----
    def build(k, wts, first):
        e = epochs[k]; F = e['F']; keep = e['keep']; ce = e['ce']
        disp = [first] if first is not None else []
        stored_idx = [i for i in range(len(keep)) if not (i == 0 and first is not None) and not (ce and i == len(keep) - 1)]
        for i in stored_idx:
            raw = e['frames'][keep[i]]
            sm = seams.smooth_seams(raw, F, e['chains'], fixed=e['tips']) if wts[i] > 0 else raw
            disp.append(qi(raw + wts[i] * (sm - raw)) * SCALE)
        if ce:
            Xl = disp[-1].copy()
            for a_, b_ in e['pairs']: Xl[a_] = Xl[b_] = 0.5 * (disp[-1][a_] + disp[-1][b_])
            disp.append(Xl)
        return np.array(disp), stored_idx

    out_eps = []; first = None; g = 0; times = [0.0]; total = 0; stats = dict(backed_off=0)
    for k, e in enumerate(epochs):
        F = e['F']; n = len(e['keep']); wts = np.ones(n); cache = {}
        if k == 0: wts = np.minimum(1.0, np.arange(n) / 10.0)   # the start is the exact polyhedron: ramp seam smoothing in
        for attempt in range(60):
            disp, stored_idx = build(k, wts, first)
            nxt_first = None
            if e['ce'] or k + 1 < K:
                nxt_first = derive_next_first(disp[-1], e.get('pairs', []), e.get('picks', []), F, len(disp[-1])) if k + 1 < K else None
            bad = None
            last_is_collapse = e['ce']
            for i in range(len(disp) - 1):
                key = (i, disp[i].tobytes().__hash__(), disp[i + 1].tobytes().__hash__())
                if key in cache: continue
                if last_is_collapse and i == len(disp) - 2:
                    ok, _ = verify.certify_collapse_interval(disp[i], disp[i + 1], F, nxt_first, epochs[k + 1]['F'])
                else:
                    ok, _ = verify.certify(disp[i:i + 2], F)
                if not ok: bad = i; break
                cache[key] = True
            if bad is None: break
            # back off the seam smoothing of the stored frames adjacent to the failing interval
            changed = False
            for j in (bad, bad + 1):
                if j < n and j in stored_idx and wts[j] > 0: wts[j] = 0.0 if wts[j] < 0.1 else wts[j] * 0.5; changed = True
            if not changed: print('epoch', k, 'interval', bad, 'cannot be certified'); sys.exit(1)
            stats['backed_off'] += 1
        e['disp'] = disp
        Q = qi(disp)
        stored = [i for i in range(len(disp)) if not (i == 0 and first is not None) and not (e['ce'] and i == len(disp) - 1)]
        # deltas: each stored frame relative to the previous displayed frame (derived first frame included)
        prev = None; blobs = []
        for i in stored:
            ref = qi(disp[i - 1]) if i > 0 else np.zeros_like(Q[i])
            blobs.append(Q[i] - ref if i > 0 else Q[i])
        body = planes(np.concatenate([b_.ravel() for b_ in blobs])) if blobs else b''
        out_eps.append(dict(start=g, nv=len(disp[0]), nframes=len(disp), stored=stored, frames=body,
                            labels=e['Lshow'].astype(np.uint8).tobytes(),
                            pairs=np.array(e.get('pairs', []), np.uint32).reshape(-1, 2), picks=np.array(e.get('picks', []), np.uint32).reshape(-1, 2)))
        for i in range(len(disp) - 1): times.append(times[-1] + max(float(np.linalg.norm(disp[i + 1] - disp[i], axis=1).max()), 0.3))
        total += len(disp) - 1
        if k == 0: stage1_end = e['stage1_idx']
        g += len(disp) - 1
        first = derive_next_first(disp[-1], e.get('pairs', []), e.get('picks', []), F, len(disp[-1])) if k + 1 < K else None
        if k % 10 == 0: print('epoch %d/%d certified (%.0fs, backed off %d)' % (k, K, time.time() - t0, stats['backed_off']), flush=True)

    # ---- write morph.bin: [u32 header length][header JSON][gzip blobs...] ----
    blobs = []; hdr_eps = []
    def add(b):
        blobs.append(gzip.compress(b, 9)); return len(blobs) - 1
    for k, o in enumerate(out_eps):
        h = dict(start=o['start'], nv=o['nv'], nframes=o['nframes'], stored=o['stored'], frames=add(o['frames']), labels=add(o['labels']))
        if k == 0: h['faces'] = add(epochs[0]['F'].astype(np.uint32).tobytes())
        h['pairs'] = add(o['pairs'].tobytes()); h['picks'] = add(o['picks'].tobytes())
        hdr_eps.append(h)
    offs = np.cumsum([0] + [len(b) for b in blobs]).tolist()
    header = dict(version=2, scale=SCALE, total=g + 1, times=[round(t, 3) for t in times], offsets=offs,
                  stages=[dict(name='Relaxing: surface tension at constant volume', **{'from': 0, 'to': stage1_end}),
                          dict(name='Pulling onto a smooth triple torus (with remeshing)', **{'from': stage1_end, 'to': g})],
                  epochs=hdr_eps)
    hj = json.dumps(header).encode()
    with open(out, 'wb') as f:
        f.write(struct.pack('<I', len(hj))); f.write(hj)
        for b in blobs: f.write(b)
    size = 4 + len(hj) + offs[-1]
    print('wrote %s: %.2f MB, %d epochs, %d keyframes, %d certified intervals, seam smoothing backed off %d times (%.0fs)' % (
        out, size / 1e6, K, g + 1, total, stats['backed_off'], time.time() - t0), flush=True)
    # reference for the JS replay test: a few epochs' reconstructed faces and displayed frames
    np.savez_compressed(out + '.check.npz', **{f'F{k}': epochs[k]['F'] for k in (0, 1, 5, K - 1)},
                        **{f'X{k}': epochs[k]['disp'] for k in (0, 1, 5, K - 1)})
