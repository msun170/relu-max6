# TEST 1: is the weight-3 OUT certificate r_3 killed by weight-4 blocks? Compute normalized pairing
# |<r_3, h_Q>|/(||r_3|| ||h_Q||) over sampled weight-4 blocks (root + non-root). Large => weight-4 sees r_3
# (not a limiting obstruction -> Box B). Small (annihilates) => candidate invariant (Box C).
# PRECISION CHECK built in: pairing with WEIGHT-3 blocks must be ~0 (r_3 _|_ weight-3); that ~0 level is the
# float32 noise floor -- only weight-4 pairings ABOVE it are real signal.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
m = 20000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.float32)); lin = cp.asarray(X.astype(np.float32))

def lat(Wt): L = [c for c in itertools.product(range(Wt+1), repeat=n) if sum(c)==Wt]; return L, set(L), {p:i for i,p in enumerate(L)}
def dgen(L):
    g=set()
    for p in L:
        for q in L:
            if p==q: continue
            dd=tuple(q[k]-p[k] for k in range(n)); gg=reduce(gcd,[abs(x) for x in dd]); dd=tuple(x//gg for x in dd)
            if dd>tuple(-x for x in dd): g.add(dd)
    return sorted(g)
def shift(p,g): return tuple(p[k]+g[k] for k in range(n))

# build complete weight-3 -> r_3
L3, L3s, v3 = lat(3); g3 = dgen(L3)
def ssz(p,Gs,Ls):
    verts=set()
    for s in range(len(Gs)+1):
        for S in itertools.combinations(Gs,s):
            q=p
            for g in S: q=shift(q,g)
            if q not in Ls: return None
            verts.add(q)
    return frozenset(verts)
zon=set()
for p in L3:
    for g in g3:
        v=ssz(p,[g],L3s)
        if v and len(v)>=2: zon.add(v)
for p in L3:
    for i in range(len(g3)):
        if shift(p,g3[i]) not in L3s: continue
        for j in range(i+1,len(g3)):
            v=ssz(p,[g3[i],g3[j]],L3s)
            if v and len(v)>=3: zon.add(v)
for zz in [zz for zz in list(zon) if len(zz)>=4]:
    for g in g3:
        verts=set(zz); ok=True
        for u in zz:
            w=shift(u,g)
            if w not in L3s: ok=False; break
            verts.add(w)
        if ok and len(verts)>len(zz): zon.add(frozenset(verts))
zon=list(zon); zorb=core.orbits_of(zon,n); zreps=[zorb[k][0] for k in zorb]
bl=set(zon)
for a in zreps:
    for b in zon:
        u=a|b
        if len(u)<=16: bl.add(u)
orb3=core.orbits_of(list(bl),n); ok3=list(orb3)
P3=(X@np.array(L3,dtype=np.int64).T).astype(np.float32)
C3=np.empty((m,len(ok3)),dtype=np.float32)
for j,k in enumerate(ok3):
    acc=np.zeros(m,np.float32)
    for vs in orb3[k]: acc+=P3[:,[v3[p] for p in vs]].max(axis=1)
    C3[:,j]=acc
A=cp.concatenate([cp.asarray(C3),lin],axis=1); Q,_=cp.linalg.qr(A); del A
r3=bmax-Q@(Q.T@bmax); del Q; cp.get_default_memory_pool().free_all_blocks()
nr3=float(cp.linalg.norm(r3))
print(f"weight-3: {len(ok3)} orbits, ||r3||/||b||={nr3/float(cp.linalg.norm(bmax)):.6f}  [{time.time()-t0:.0f}s]", flush=True)

def sample_pairings(Wt, K, nonroot_only=False):
    L, Ls, vi = lat(Wt); gens = dgen(L); ng=len(gens)
    roots={tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a!=b}
    gpool=[g for g in gens if (not nonroot_only) or (g not in roots and tuple(-x for x in g) not in roots)] or gens
    Pm=cp.asarray((X@np.array(L,dtype=np.int64).T).astype(np.float32))
    rng=np.random.default_rng(7)
    def zo():
        p=L[rng.integers(len(L))]; verts={p}; added=0; tries=0
        while added<4 and len(verts)<12 and tries<60:
            tries+=1; g=gpool[rng.integers(len(gpool))]; nv=set(); ok=True
            for u in verts:
                w=shift(u,g)
                if w not in Ls: ok=False; break
                nv.add(w)
            if ok and nv-verts: verts|=nv; added+=1
        return [vi[v] for v in verts]
    cols=cp.empty((m,K),dtype=cp.float32)
    for j in range(K):
        a=zo(); b=zo(); cols[:,j]=cp.maximum(Pm[:,a].max(axis=1),Pm[:,b].max(axis=1))
    cn=cp.linalg.norm(cols,axis=0); pair=cp.abs(r3@cols)/(nr3*cn+1e-9)
    return float(cp.max(pair)), float(cp.mean(pair)), float(cp.percentile(pair,99))

for tag, Wt, nro in (("weight-3 (CONTROL, should ~0)",3,False), ("weight-4 root",4,False), ("weight-4 NON-root",4,True), ("weight-5",5,False)):
    mx, mn, p99 = sample_pairings(Wt, 6000, nro)
    print(f"  {tag:>30}: max|cos(r3,h_Q)|={mx:.4f}  p99={p99:.4f}  mean={mn:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print(f"\n[{time.time()-t0:.0f}s] if weight-4 pairings >> weight-3 control -> weight-4 KILLS r3 (Box B). if ~control -> r3 survives (Box C).", flush=True)
