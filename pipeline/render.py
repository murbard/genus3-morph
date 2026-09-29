"""Quick software render (painter's order, Lambert shading) of mesh frames to PNG for inspection."""
import numpy as np, sys, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
C = np.array([[230,25,75],[60,180,75],[255,225,25],[67,99,216],[245,130,49],[145,30,180],[66,212,244],[240,50,230]]) / 255
def rot(el, az):
    el, az = np.radians(el), np.radians(az)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return Rx @ Rz
def draw(ax, X, F, L, el, az, title=''):
    Y = X @ rot(el, az).T
    tri = Y[F]; z = tri[:, :, 2].mean(1); order = np.argsort(z)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, 0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(fn @ light)
    col = C[L] * shade[:, None]
    pc = PolyCollection(tri[order][:, :, [0, 1]], facecolors=col[order], edgecolors='none', antialiased=False)
    ax.add_collection(pc); ax.set_xlim(-330, 330); ax.set_ylim(-330, 330); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=9)
if __name__ == '__main__':
    d = np.load(sys.argv[1]); frames = d['frames']; F = d['F']; L = d['L']
    idx = [int(i) for i in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, len(frames) // 2, len(frames) - 1]
    views = [(0, 0), (-90, 0), (-60, 35)]    # top (looking down z), side (z up), oblique
    fig, axs = plt.subplots(len(idx), len(views), figsize=(5 * len(views), 5 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        for c, (el, az) in enumerate(views): draw(axs[r, c], frames[i].astype(float), F, L, el, az, f'frame {i}/{len(frames)-1}  el={el} az={az}')
    plt.tight_layout(); plt.savefig(sys.argv[3] if len(sys.argv) > 3 else 'render.png', dpi=60)
