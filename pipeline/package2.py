"""Package stage 1 (phaseA.npz) + stage 2 epochs (flow*.pkl) into morph_data.js.
Each epoch: certified keyframes (greedy), quantized, and the *quantized* frames re-certified."""
import numpy as np, sys, json, gzip, base64, pickle, time
import coll, verify, relabel
ERODE = True
from package import select

def z(arr): return base64.b64encode(gzip.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()

if __name__ == '__main__':
    phaseA, flow, out = sys.argv[1], sys.argv[2], sys.argv[3]
    dmax = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    A = np.load(phaseA); ep = []
    for path in flow.split(','):
        more = pickle.load(open(path, 'rb'))['epochs']
        if ep:   # a continuation run starts from the previous run's last state (same triangulation): merge
            ep[-1]['frames'] = np.concatenate([ep[-1]['frames'], more[0]['frames'][1:]]); more = more[1:]
        ep += more
    epochs = [dict(F=A['F'], L=A['L'], frames=np.concatenate([A['frames'], ep[0]['frames'][1:]]), n_stage1=len(A['frames']))]
    epochs += [dict(F=e['F'], L=e['L'], frames=e['frames']) for e in ep[1:]]
    scale = 0.02; out_eps = []; g = 0; stage1_end = None; total_cert = 0; t0 = time.time()
    times = [0.0]
    for k, e in enumerate(epochs):
        fr = e['frames'].astype(np.float64); F = e['F'].astype(np.int64)
        El = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        collapse_end = len(fr) > 1 and bool((np.linalg.norm(fr[-1][El[:, 0]] - fr[-1][El[:, 1]], axis=1) < 1e-9).any())
        keep = select(fr[:-1] if collapse_end else fr, F, dmax) if len(fr) > 1 else [0]
        if collapse_end: keep = sorted(set(keep) | {len(fr) - 2, len(fr) - 1})
        if k == 0:   # make sure the end of stage 1 is a keyframe (for the stage label)
            keep = sorted(set(keep) | {e['n_stage1'] - 1})
            stage1_end = g + keep.index(e['n_stage1'] - 1)
        Q = np.round(fr[keep] / scale).astype(np.int32); assert np.abs(Q).max() < 32767
        Xq = (Q * scale).astype(np.float64)
        body = Xq[:-1] if collapse_end else Xq
        ok, msg = verify.certify(body, F) if len(body) > 1 else (True, 'single frame')
        if ok and collapse_end:
            nxt = epochs[k + 1]; Xn = (np.round(nxt['frames'][0].astype(np.float64) / scale) * scale)
            ok, msg2 = verify.certify_collapse_interval(Xq[-2], Xq[-1], F, Xn, nxt['F']); msg += '; ' + msg2
        if not ok: print('epoch', k, 'FAILED', msg); sys.exit(1)
        total_cert += len(keep) - 1
        for i in range(len(keep) - 1):   # playback time proportional to motion
            times.append(times[-1] + max(float(np.linalg.norm(Xq[i + 1] - Xq[i], axis=1).max()), 0.3))
        delta = np.diff(np.concatenate([np.zeros_like(Q[:1]), Q]), axis=0).astype(np.int16)
        Lshow = e['L']
        if k > 0 and ERODE:   # display only: remove thin dangling tails (both faces stay disks)
            Lshow, _ = relabel.erode_tails(F, e['L'])
        out_eps.append(dict(start=g, nv=int(fr.shape[1]), nframes=len(keep), faces=z(F.astype(np.uint32)),
                            labels=z(Lshow.astype(np.uint8)), frames=z(delta)))
        g += len(keep) - 1
        print('epoch %d: %d frames -> %d keyframes, certified (%.0fs)' % (k, len(fr), len(keep), time.time() - t0), flush=True)
    stages = [dict(name='Relaxing: surface tension at constant volume', **{'from': 0, 'to': stage1_end}),
              dict(name='Pulling onto a smooth triple torus (with remeshing)', **{'from': stage1_end, 'to': g})]
    data = dict(scale=scale, total=g + 1, stages=stages, epochs=out_eps, times=[round(t, 3) for t in times])
    s = 'const MORPH_DATA = ' + json.dumps(data) + ';\n'
    open(out, 'w').write(s)
    print('wrote %s: %.1f MB, %d epochs, %d keyframes, %d certified intervals' % (out, len(s) / 1e6, len(out_eps), g + 1, total_cert))
