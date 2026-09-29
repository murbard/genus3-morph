import numpy as np, target, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=np.load(sys.argv[1]); fr=d['frames']; F=d['F'].astype(np.int64); L=d['L']; X=fr[int(sys.argv[3]) if len(sys.argv)>3 else -1].astype(float)
C=np.array([[230,25,75],[60,180,75],[255,225,25],[67,99,216],[245,130,49],[145,30,180],[66,212,244],[240,50,230]])/255
V,curves=target.build(); segs=target.segments(curves)*np.array([2.4,2.4,1.5])
f,_=target.softmin_dist(X,np.ascontiguousarray(segs),6.0); f-=24
bad=np.abs(f)>10; print('vertices |f|>10:', bad.sum(), 'at', X[bad][::max(1,bad.sum()//6)].round(0).tolist())
def section(ax, axis, val):
    s=X[:,axis]-val
    a=s[F]; cross=~((a>0).all(1)|(a<0).all(1))
    for t in np.where(cross)[0]:
        pts=[]; aa=a[t]
        for i,j in ((0,1),(1,2),(2,0)):
            if (aa[i]>0)!=(aa[j]>0):
                u=aa[i]/(aa[i]-aa[j]); pts.append(X[F[t,i]]+u*(X[F[t,j]]-X[F[t,i]]))
        if len(pts)==2:
            o=[k for k in range(3) if k!=axis]; p=np.array(pts)[:,o]; ax.plot(p[:,0],p[:,1],'-',color=C[L[t]],lw=1)
    g=np.linspace(-220,220,300); A,B=np.meshgrid(g,g,indexing='ij'); P=np.zeros((300*300,3)); o=[k for k in range(3) if k!=axis]
    P[:,o[0]]=A.ravel(); P[:,o[1]]=B.ravel(); P[:,axis]=val
    ft,_=target.softmin_dist(P,np.ascontiguousarray(segs),6.0); ax.contour(A,B,(ft-24).reshape(300,300),[0],colors='k',linewidths=0.5)
    ax.set_aspect('equal'); ax.set_title(f'section {"xyz"[axis]}={val}')
vals=[float(v) for v in sys.argv[4].split(',')] if len(sys.argv)>4 else [0,0,0]
fig,axs=plt.subplots(1,3,figsize=(21,7))
for ax,axis,v in zip(axs,[2,0,1],vals): section(ax,axis,v)
plt.tight_layout(); plt.savefig(sys.argv[2],dpi=60)
