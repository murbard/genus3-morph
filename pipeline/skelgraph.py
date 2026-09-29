import numpy as np
from scipy import ndimage
def graph(sk, g, sd):
    K = np.ones((3, 3, 3), int); K[1, 1, 1] = 0
    deg = ndimage.convolve(sk.astype(int), K, mode='constant') * sk
    J = (deg >= 3) & sk
    jl, nj = ndimage.label(J, structure=np.ones((3, 3, 3)))
    B = sk & ~J
    bl, nb = ndimage.label(B, structure=np.ones((3, 3, 3)))
    # for each branch: which junction clusters touch it, and its ordered voxel path
    Jd = ndimage.grey_dilation(jl, size=(3, 3, 3))
    edges = []
    for b in range(1, nb + 1):
        vox = np.argwhere(bl == b)
        touch = sorted(set(Jd[tuple(vox.T)]) - {0})
        # order voxels along the branch by walking from an end
        S = set(map(tuple, vox)); nbr = lambda p: [q for q in ((p[0]+i, p[1]+j, p[2]+k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)) if q != p and q in S]
        startc = [p for p in S if len(nbr(p)) <= 1] or [next(iter(S))]
        path = [startc[0]]; seen = {startc[0]}
        while True:
            nx = [q for q in nbr(path[-1]) if q not in seen]
            if not nx: break
            path.append(nx[0]); seen.add(nx[0])
        pts = g[np.array(path)]
        edges.append(dict(touch=touch, pts=pts, n=len(vox)))
    cent = np.array([np.interp(c, np.arange(len(g)), g) for c in ndimage.center_of_mass(J, jl, range(1, nj + 1))])
    return cent, edges
if __name__ == '__main__':
    z = np.load('voxA.npz'); sk = np.load('skel.npy')
    cent, edges = graph(sk, z['g'], z['sd'])
    for i, c in enumerate(cent): print('J%d' % (i + 1), c.round(1))
    for e in edges: print('branch touching', e['touch'], 'voxels', e['n'], 'from', e['pts'][0].round(0), 'to', e['pts'][-1].round(0))
