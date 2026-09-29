"""Select certified keyframes, quantize, re-certify exactly what is displayed, write morph_data.js."""
import numpy as np, sys, json, gzip, base64, time
import coll, verify

def select(frames, F, dmax):
    F = F.astype(np.int64); vt = coll.vertex_tri_csr(F, frames.shape[1])
    keep = [0]; i = 0; n = len(frames)
    while i < n - 1:
        disp = np.array([np.linalg.norm(frames[j] - frames[i], axis=1).max() for j in range(i + 1, n)])
        j = i + 1 + max(0, int(np.searchsorted(disp > dmax, True)) - 1)
        while j > i + 1:
            ok, _ = verify.certify_segment(frames[i].astype(float), frames[j].astype(float), F, vt, max_depth=8)
            if ok: break
            j -= 1
        keep.append(j); i = j
    return keep

if __name__ == '__main__':
    parts = sys.argv[1].split(','); out = sys.argv[2]; dmax = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    names = ['Relaxing: surface tension at constant volume', 'Pulling onto a smooth triple torus']
    allf = []; bounds = []; F = L = None
    for k, p in enumerate(parts):
        d = np.load(p); fr = d['frames'].astype(np.float64); F, L = d['F'], d['L']
        if allf: fr = fr[1:]
        bounds.append((sum(len(a) for a in allf), sum(len(a) for a in allf) + len(fr) - 1)); allf.append(fr)
    frames = np.concatenate(allf); print('total frames', len(frames))
    t0 = time.time(); keep = select(frames, F, dmax); print('keyframes', len(keep), '%.0fs' % (time.time() - t0))
    scale = 0.02
    Q = np.round(frames[keep] / scale).astype(np.int32)
    assert np.abs(Q).max() < 32767
    deq = (Q * scale).astype(np.float64)
    ok, msg = verify.certify(deq, F); print('certify displayed (quantized) frames:', ok, msg)
    if not ok: sys.exit(1)
    delta = np.diff(np.concatenate([np.zeros_like(Q[:1]), Q]), axis=0).astype(np.int16)
    stages = []
    for k, (a, b) in enumerate(bounds):
        ka = int(np.searchsorted(keep, a)); kb = int(np.searchsorted(keep, b, 'right')) - 1
        stages.append(dict(name=names[k] if k < len(names) else f'stage {k+1}', **{'from': ka, 'to': kb}))
    z = lambda arr: base64.b64encode(gzip.compress(arr.tobytes(), 9)).decode()
    data = dict(nv=int(frames.shape[1]), nframes=len(keep), scale=scale, stages=stages,
                faces=z(F.astype(np.uint32)), labels=z(L.astype(np.uint8)), frames=z(delta))
    open(out, 'w').write('const MORPH_DATA = ' + json.dumps(data) + ';\n')
    print('wrote', out, '%.1f MB' % (len(json.dumps(data)) / 1e6), 'stages', stages)
