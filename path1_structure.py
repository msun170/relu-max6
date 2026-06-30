# Path 1: structure of the weight-3 approximant of max_7. Classify the LS coefficients by block type
# (#vertices, complexity = #generators, zonotope vs join) to see whether the correction concentrates on a clean
# family -- which would characterize the contraction operator T and point toward an analytic F_w / closure proof.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; t0 = time.time()
W3 = [c for c in itertools.product(range(4), repeat=n) if sum(c) == 3]; W3set = set(W3); v3 = {p:i for i,p in enumerate(W3)}
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
g3 = set()
for p in W3:
    for q in W3:
        if p==q: continue
        dd=tuple(q[k]-p[k] for k in range(n)); gg=reduce(gcd,[abs(x) for x in dd]); dd=tuple(x//gg for x in dd)
        if dd>tuple(-x for x in dd): g3.add(dd)
g3=sorted(g3)
rootset = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a!=b}
def ssz(p,G):
    verts=set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G,s):
            q=p
            for gg in S: q=shift(q,gg)
            if q not in W3set: return None
            verts.add(q)
    return frozenset(verts)
zon=set()
for p in W3:
    for gg in g3:
        v=ssz(p,[gg])
        if v and len(v)>=2: zon.add(v)
for p in W3:
    for i in range(len(g3)):
        if shift(p,g3[i]) not in W3set: continue
        for j in range(i+1,len(g3)):
            v=ssz(p,[g3[i],g3[j]])
            if v and len(v)>=3: zon.add(v)
for zz in [zz for zz in list(zon) if len(zz)>=4]:
    for gg in g3:
        verts=set(zz); ok=True
        for u in zz:
            w=shift(u,gg)
            if w not in W3set: ok=False; break
            verts.add(w)
        if ok and len(verts)>len(zz): zon.add(frozenset(verts))
zon=list(zon); zorb=core.orbits_of(zon,n); zreps=[zorb[k][0] for k in zorb]; zonset=set(zon)
bl=set(zon)
for a in zreps:
    for b in zon:
        u=a|b
        if len(u)<=16: bl.add(u)
orb=core.orbits_of(list(bl),n); okeys=list(orb); N=len(okeys)
print(f"weight-3: {N} orbits  [{time.time()-t0:.0f}s]", flush=True)

# metadata per orbit (from its representative): #verts, is it a single zonotope (in zonset) or a join
def meta(rep):
    nv = len(rep); isz = rep in zonset
    return nv, ("zono" if isz else "join")

m = 22000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
P3 = (X @ np.array(W3, dtype=np.int64).T).astype(np.float32)
C = np.empty((m, N), dtype=np.float32); reps=[]
for j,k in enumerate(okeys):
    reps.append(orb[k][0])
    acc=np.zeros(m,dtype=np.float32)
    for vs in orb[k]:
        acc += P3[:,[v3[p] for p in vs]].max(axis=1)
    C[:,j]=acc
A=np.column_stack([C, X.astype(np.float32)])
# CGLS coefficients
Ag=cp.asarray(A); bg=cp.asarray(bmax)
cn=cp.linalg.norm(Ag,axis=0); cn=cp.where(cn>0,cn,cp.float32(1.0)); Ag/=cn
x=cp.zeros(Ag.shape[1],dtype=cp.float32); r=bg.copy(); s=Ag.T@r; p=s.copy(); gamma=float(s@s)
for it in range(12000):
    q=Ag@p; al=gamma/float(q@q); x+=al*p; r-=al*q
    s=Ag.T@r; gn=float(s@s); be=gn/gamma; p=s+be*p; gamma=gn
coef=cp.asnumpy(x/cn)[:N]   # weight-3 orbit coefficients (drop linear)
relres=float(cp.linalg.norm(r))/float(cp.linalg.norm(bg))
print(f"f(3)={relres:.5f}  [{time.time()-t0:.0f}s]\n", flush=True)

ac=np.abs(coef)
print(f"coefficient stats: max|c|={ac.max():.4f}, nonzero(>1e-3)={int((ac>1e-3).sum())}/{N}")
# energy by #vertices and type
from collections import defaultdict
byv=defaultdict(float); byt=defaultdict(float)
for j in range(N):
    nv,tp=meta(reps[j]); byv[nv]+=coef[j]**2; byt[tp]+=coef[j]**2
tot=sum(coef**2)
print("\ncoeff energy by block type:")
for tp in ("zono","join"): print(f"  {tp}: {byt[tp]/tot*100:.1f}%")
print("coeff energy by #vertices:")
for nv in sorted(byv): print(f"  {nv:>2} verts: {byv[nv]/tot*100:5.1f}%")
# top orbits
top=np.argsort(-ac)[:12]
print("\ntop-12 orbits by |coef|:")
for j in top:
    nv,tp=meta(reps[j]); print(f"  c={coef[j]:+.4f}  {tp}  {nv} verts")
print(f"\n[{time.time()-t0:.0f}s]  clean concentration on few types => structured correction => analytic F_w plausible")
