"""Compare neighbour-normal angles of the current mesh with the same triangulation snapped onto the target."""
import numpy as np, pickle, sys, target, opt
e=pickle.load(open(sys.argv[1],'rb'))['epochs'][-1]; X=e['frames'][-1].astype(float); F=e['F'].astype(np.int64)
V,curves=target.build(); segs=np.ascontiguousarray(target.segments(curves)*np.array([2.4,2.4,1.5]))
def angles(Y):
    fn=np.cross(Y[F[:,1]]-Y[F[:,0]],Y[F[:,2]]-Y[F[:,0]]); nn=fn/np.linalg.norm(fn,axis=1,keepdims=True)
    t1,t2=opt.edge_tris(F); return np.degrees(np.arccos(np.clip((nn[t1]*nn[t2]).sum(1),-1,1)))
Y=X.copy()
for _ in range(6):
    f,g=target.softmin_dist(Y,segs,6.0); Y=Y-(f-24)[:,None]*g
fx,_=target.softmin_dist(X,segs,6.0)
a=angles(X); b=angles(Y)
print('mesh V %d, median edge %.1f, |f| mean %.2f p99 %.2f' % (len(X), np.median(np.linalg.norm(X[F[:,0]]-X[F[:,1]],axis=1)), np.abs(fx-24).mean(), np.percentile(np.abs(fx-24),99)))
print('current  : angle p50 %.1f p90 %.1f p99 %.1f  #>60deg %d' % (*np.percentile(a,[50,90,99]), (a>60).sum()))
print('snapped  : angle p50 %.1f p90 %.1f p99 %.1f  #>60deg %d   (same triangulation, vertices exactly on the target)' % (*np.percentile(b,[50,90,99]), (b>60).sum()))
