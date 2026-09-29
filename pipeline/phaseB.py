"""Stage 2: pull toward a smooth triple torus (tube of radius R around a growing, symmetric spine)."""
import numpy as np, sys, os, time
import opt, target
HERE = os.path.dirname(os.path.abspath(__file__))
args = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
src = np.load(os.path.join(HERE, args.get('start', 'phaseA.npz')))
F = src['F']; X = src['frames'][-1].astype(float)
V, curves = target.build(); segs0 = target.segments(curves)
S_end = np.array(args.get('scale', [2.4, 2.4, 1.5])); S_start = np.array(args.get('scale0', [1.3, 1.3, 1.3]))
R_end = args.get('R', 24.0); R_start = args.get('R0', 18.0); ramp = args.get('ramp', 300); nit = args.get('iters', 800)
state = dict(S=S_start, R=R_start)
def tgt(Y):
    f, G = target.softmin_dist(Y, np.ascontiguousarray(segs0 * state['S']), 6.0)
    return f - state['R'], G
w = dict(uniform=args.get('uniform', 1.0), tension=args.get('tension', 0.05), kvol=0.0, V0=1.0, quality=args.get('quality', 5.0), target=tgt, ktarget=args.get('ktarget', 0.01), kregion=args.get('kregion', 0.0), split=args.get('split', False))
o = opt.Optimizer(X, F, w, V0=None, step_max=args.get('step_max', 0.5))
frames = [X.astype(np.float32)]; acc = 0; t0 = time.time()
for it in range(nit):
    s = min(1.0, it / ramp); s = s * s * (3 - 2 * s)
    state['S'] = S_start + s * (S_end - S_start); state['R'] = R_start + s * (R_end - R_start)
    X, info = o.step(X)
    acc += info['move']
    if acc >= 1.0: frames.append(X.astype(np.float32)); acc = 0
    if it % 50 == 0 or info['alpha'] == 0:
        f, _ = tgt(X)
        print(it, 'ccd %.3g halv %d dmin %.3f move %.3f |f| mean %.2f max %.1f frames %d %.0fs' % (info['a_ccd'], info['halvings'], info['dmin'], info['move'], np.abs(f).mean(), np.abs(f).max(), len(frames), time.time() - t0), flush=True)
    if it % 100 == 99: np.savez(os.path.join(HERE, args.get('out', 'phaseB.npz')), frames=np.array(frames + [X.astype(np.float32)]), F=F, L=src['L'])
frames.append(X.astype(np.float32))
np.savez(os.path.join(HERE, args.get('out', 'phaseB.npz')), frames=np.array(frames), F=F, L=src['L'])
