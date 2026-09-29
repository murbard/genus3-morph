"""Stage 2 with remeshing: pull toward the smooth triple torus, remeshing every few iterations.
Output: a list of epochs, each with its own triangulation (F, L) and frames; consecutive epochs meet at identical
geometry (collapse motions are certified inside the old epoch; splits don't move anything)."""
import numpy as np, sys, os, time, pickle
import opt, target, remesh, coll, regions, relabel
HERE = os.path.dirname(os.path.abspath(__file__))
args0 = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
remesh.SEAM_AWARE = bool(args0.get('seam_aware', False))
remesh.TEETH = bool(args0.get('teeth', False))
args = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
start = args.get('start', 'phaseA.npz')
if start.endswith('.pkl'):
    last = pickle.load(open(os.path.join(HERE, start), 'rb'))['epochs'][-1]
    F = last['F'].astype(np.int64); L = last['L']; X = last['frames'][-1].astype(float)
else:
    src = np.load(os.path.join(HERE, start)); F = src['F'].astype(np.int64); L = src['L']; X = src['frames'][-1].astype(float)
V, curves = target.build(); segs0 = target.segments(curves)
S_end = np.array(args.get('scale', [2.4, 2.4, 1.5])); S_start = np.array(args.get('scale0', [1.3, 1.3, 1.3]))
R_end = args.get('R', 24.0); R_start = args.get('R0', 18.0); ramp = args.get('ramp', 300); nit = args.get('iters', 2000)
every = args.get('remesh_every', 40); ell = args.get('ell', 7.0)
state = dict(S=S_start, R=R_start)
def tgt(Y):
    f, G = target.softmin_dist(Y, np.ascontiguousarray(segs0 * state['S']), 6.0)
    return f - state['R'], G
PHI = np.bincount(L, np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1), 8)
PHI = PHI / PHI.sum()
def make_opt(X, F, L):
    w = dict(uniform=args.get('uniform', 2.0), tension=args.get('tension', 0.15), kvol=0.0, V0=1.0, quality=args.get('quality', 5.0),
             target=tgt, ktarget=args.get('ktarget', 0.02), kregion=args.get('kregion', 0.02), split=args.get('split', True),
             kfold=args.get('kfold', 5.0), fold_c0=args.get('fold_c0', -0.3), klap=args.get('klap', 0.0), kalign=args.get('kalign', 0.0), align_sigma=args.get('align_sigma', None), L=L, phi=PHI, kperim=args.get('kperim', 0.0), karea_reg=args.get('karea_reg', 1.0), klap_normal_only=args.get('klap_normal_only', False), seam_pull=args.get('seam_pull', 0.0),
             target_grad=lambda Y: target.softmin_dist(Y, np.ascontiguousarray(segs0 * state['S']), 6.0)[1])
    return opt.Optimizer(X, F, w, V0=None, step_max=args.get('step_max', 0.5), sigma=args.get('sigma', 20.0))
if args.get('fatten'):
    L, n0, n1 = relabel.fatten_necks(F.astype(np.int64), L); print('initial necks widened: %d -> %d' % (n0, n1), flush=True)
epochs = [dict(F=F, L=L, frames=[X.astype(np.float32)])]
o = make_opt(X, F, L); acc = 0; t0 = time.time()
out = os.path.join(HERE, args.get('out', 'flow.pkl'))
def save():
    pickle.dump(dict(epochs=[dict(F=e['F'], L=e['L'], frames=np.array(e['frames'] + ([X.astype(np.float32)] if e is epochs[-1] else [])))
                             for e in epochs]), open(out, 'wb'))
for it in range(nit):
    s = min(1.0, it / ramp); s = s * s * (3 - 2 * s)
    state['S'] = S_start + s * (S_end - S_start); state['R'] = R_start + s * (R_end - R_start)
    X, info = o.step(X)
    acc += info['move']
    if acc >= 1.0: epochs[-1]['frames'].append(X.astype(np.float32)); acc = 0
    if it % every == every - 1:
        Xm, (X2, F2, L2), nc, ns = remesh.remesh(X, F, L, ell)
        if (nc or ns) and coll.vertex_adjacent_violations(X2, F2, *coll.vertex_tri_csr(F2, len(X2))) != 0:
            print('remesh round rejected (fold in new mesh)', flush=True); nc = ns = 0
        if (nc or ns) and not all(r[0] == 1 for r in regions.region_components(F2, L2)):
            print('remesh round rejected (face region split)', flush=True); nc = ns = 0
        if nc or ns:
            epochs[-1]['frames'].append(X.astype(np.float32))
            if nc: epochs[-1]['frames'].append(Xm.astype(np.float32))      # certified collapse motion, old triangulation
            X, F, L = X2, F2, L2
            if args.get('fatten'):
                L, n0, n1 = relabel.fatten_necks(F, L)
                if n0: print('necks widened: %d -> %d' % (n0, n1), flush=True)
            epochs.append(dict(F=F, L=L, frames=[X.astype(np.float32)]))
            o = make_opt(X, F, L); acc = 0
    if it % 50 == 0 or info['alpha'] == 0:
        f, _ = tgt(X)
        print(it, 'ccd %.3g dmin %.3f move %.3f |f| mean %.2f max %.1f V %d epochs %d frames %d %.0fs' % (
            info['a_ccd'], info['dmin'], info['move'], np.abs(f).mean(), np.abs(f).max(), len(X), len(epochs),
            sum(len(e['frames']) for e in epochs), time.time() - t0), flush=True)
    if it % 200 == 199: save()
save()
