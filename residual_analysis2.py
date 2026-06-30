# Residual structure analysis (toward a CONTRACTION / closure proof) -- fast version using the cached weight-3
# columns (oddeven_w3_m16000.npz: CX at X, ODD => CXm = CX - 2*ODD, LIN = X). Weight-2 built fast on the 28-pt
# lattice. Extract R_w = max7 - F_w (w=2,3), then: (1) contraction rho=||R3||/||R2||, (2) even/odd energy split,
# (3) concentration of |R_w| by tie-stratum depth (active-set size at the max). Constant R3/R2 across strata = self-similarity.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"

def proj_residual(Acols, b):
    Ag = cp.asarray(Acols, dtype=cp.float32); bg = cp.asarray(b, dtype=cp.float32)
    cn = cp.linalg.norm(Ag, axis=0); cn = cp.where(cn > 0, cn, cp.float32(1.0)); Ag2 = Ag / cn
    x = cp.zeros(Ag2.shape[1], dtype=cp.float32); r = bg.copy()
    s = Ag2.T @ r; p = s.copy(); gamma = float(s @ s); bn = float(cp.linalg.norm(bg))
    for it in range(8000):
        q = Ag2 @ p; al = gamma/float(q @ q); x += al*p; r -= al*q
        s = Ag2.T @ r; gnew = float(s @ s); be = gnew/gamma; p = s + be*p; gamma = gnew
        if it % 500 == 0 and float(cp.linalg.norm(s))/(float(cp.linalg.norm(Ag2))*float(cp.linalg.norm(r))+1e-30) < 1e-7: break
    coef = cp.asnumpy(x/cn)
    return float(cp.linalg.norm(r))/bn, coef

# ---- weight-3 built FAST (84-point lattice), m > rank(~18866) so NON-vacuous ----
m = 22000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmaxX = X.max(axis=1).astype(np.float32); bmaxXm = (-X).max(axis=1).astype(np.float32)
W3 = [c for c in itertools.product(range(4), repeat=n) if sum(c) == 3]; W3set = set(W3); v3 = {p:i for i,p in enumerate(W3)}
def shift3(p, g): return tuple(p[k]+g[k] for k in range(n))
g3 = set()
for p in W3:
    for q in W3:
        if p==q: continue
        dd=tuple(q[k]-p[k] for k in range(n)); gg=reduce(gcd,[abs(x) for x in dd]); dd=tuple(x//gg for x in dd)
        if dd>tuple(-x for x in dd): g3.add(dd)
g3=sorted(g3)
def ssz3(p,G):
    verts=set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G,s):
            q=p
            for gg in S: q=shift3(q,gg)
            if q not in W3set: return None
            verts.add(q)
    return frozenset(verts)
z3=set()
for p in W3:
    for gg in g3:
        v=ssz3(p,[gg])
        if v and len(v)>=2: z3.add(v)
for p in W3:
    for i in range(len(g3)):
        if shift3(p,g3[i]) not in W3set: continue
        for j in range(i+1,len(g3)):
            v=ssz3(p,[g3[i],g3[j]])
            if v and len(v)>=3: z3.add(v)
for zz in [zz for zz in list(z3) if len(zz)>=4]:
    for gg in g3:
        verts=set(zz); ok=True
        for u in zz:
            w=shift3(u,gg)
            if w not in W3set: ok=False; break
            verts.add(w)
        if ok and len(verts)>len(zz): z3.add(frozenset(verts))
z3=list(z3); z3orb=core.orbits_of(z3,n); z3reps=[z3orb[k][0] for k in z3orb]
bl3=set(z3)
for a in z3reps:
    for b in z3:
        u=a|b
        if len(u)<=16: bl3.add(u)
orb3=core.orbits_of(list(bl3),n); ok3=list(orb3); N3=len(ok3)
P3 = (X @ np.array(W3,dtype=np.int64).T).astype(np.float32)     # m x 84
CX3 = np.empty((m,N3),dtype=np.float32); CXm3 = np.empty((m,N3),dtype=np.float32)
for j,k in enumerate(ok3):
    acc=np.zeros(m,dtype=np.float32); accm=np.zeros(m,dtype=np.float32)
    for vs in orb3[k]:
        idx=[v3[p] for p in vs]
        acc += P3[:,idx].max(axis=1); accm += -P3[:,idx].min(axis=1)
    CX3[:,j]=acc; CXm3[:,j]=accm
print(f"weight-3 built fast: {N3} orbits, m={m} (>rank~18866 so non-vacuous)  [{time.time()-t0:.0f}s]", flush=True)

# ---- weight-2 built fast on the 28-point lattice ----
W2 = [c for c in itertools.product(range(3), repeat=n) if sum(c) == 2]; W2set = set(W2); v2 = {p:i for i,p in enumerate(W2)}
def shift(p,g): return tuple(p[k]+g[k] for k in range(n))
g2 = sorted({tuple((q[k]-p[k]) for k in range(n)) for p in W2 for q in W2 if p!=q
             for d in [reduce(gcd,[abs(q[k]-p[k]) for k in range(n)])] for _ in [0]
             if (lambda dd: dd>tuple(-x for x in dd))(tuple((q[k]-p[k])//d for k in range(n)))})
# simpler: regenerate g2 cleanly
g2 = set()
for p in W2:
    for q in W2:
        if p==q: continue
        dd=tuple(q[k]-p[k] for k in range(n)); gg=reduce(gcd,[abs(x) for x in dd]); dd=tuple(x//gg for x in dd)
        if dd>tuple(-x for x in dd): g2.add(dd)
g2=sorted(g2)
def ssz(p,G):
    verts=set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G,s):
            q=p
            for g in S: q=shift(q,g)
            if q not in W2set: return None
            verts.add(q)
    return frozenset(verts)
zon=set()
for p in W2:
    for g in g2:
        v=ssz(p,[g])
        if v and len(v)>=2: zon.add(v)
for p in W2:
    for i in range(len(g2)):
        if shift(p,g2[i]) not in W2set: continue
        for j in range(i+1,len(g2)):
            v=ssz(p,[g2[i],g2[j]])
            if v and len(v)>=3: zon.add(v)
zon=list(zon); zorb=core.orbits_of(zon,n); zreps=[zorb[k][0] for k in zorb]
bl=set(zon)
for a in zreps:
    for b in zon:
        u=a|b
        if len(u)<=12: bl.add(u)
orb2=core.orbits_of(list(bl),n); ok2=list(orb2); N2=len(ok2)
P2 = (X @ np.array(W2,dtype=np.int64).T).astype(np.float32)     # m x 28
P2m = -P2                                                        # X.W at -X = -(X.W); max over subset = -min
CX2 = np.empty((m,N2),dtype=np.float32); CXm2 = np.empty((m,N2),dtype=np.float32)
for j,k in enumerate(ok2):
    cols_idx_acc = np.zeros(m,dtype=np.float32); accm = np.zeros(m,dtype=np.float32)
    for vs in orb2[k]:
        idx=[v2[p] for p in vs]
        cols_idx_acc += P2[:,idx].max(axis=1); accm += -P2[:,idx].min(axis=1)
    CX2[:,j]=cols_idx_acc; CXm2[:,j]=accm
print(f"weight-2: {N2} orbits built fast  [{time.time()-t0:.0f}s]", flush=True)

# ---- project ----
lin = X.astype(np.float32)
rel2, c2 = proj_residual(np.column_stack([CX2,lin]), bmaxX)
rel3, c3 = proj_residual(np.column_stack([CX3,lin]), bmaxX)
F2X = np.column_stack([CX2,lin]) @ c2; F2Xm = np.column_stack([CXm2,(-X).astype(np.float32)]) @ c2
F3X = np.column_stack([CX3,lin]) @ c3; F3Xm = np.column_stack([CXm3,(-X).astype(np.float32)]) @ c3
R2 = bmaxX - F2X; R2m = bmaxXm - F2Xm
R3 = bmaxX - F3X; R3m = bmaxXm - F3Xm
print(f"floors: f(2)={rel2:.5f}  f(3)={rel3:.5f}  [{time.time()-t0:.0f}s]", flush=True)

n2=np.linalg.norm(R2); n3=np.linalg.norm(R3)
print(f"\n(1) CONTRACTION rho_23 = ||R3||/||R2|| = {n3/n2:.4f}")
for tag,RX,RXm in (("2",R2,R2m),("3",R3,R3m)):
    ev=(RX+RXm)/2; od=(RX-RXm)/2; eE=float(np.dot(ev,ev)); oE=float(np.dot(od,od)); tot=eE+oE
    print(f"(2) weight-{tag}: even {eE/tot*100:.1f}%  odd {oE/tot*100:.1f}%")
mx=X.max(axis=1)
active=(np.abs(X-mx[:,None])<=0).sum(axis=1)   # exact ties at the max (integer coords)
print("\n(3) CONCENTRATION by exact max-multiplicity k (how many coords equal the max):")
print(f"  {'k':>3} {'#pts':>7} {'mean|R2|':>10} {'mean|R3|':>10} {'R3/R2':>7}")
for k in range(1,n+1):
    msk=active==k
    if msk.sum()<5: continue
    a2=np.abs(R2[msk]).mean(); a3=np.abs(R3[msk]).mean()
    print(f"  {k:>3} {int(msk.sum()):>7} {a2:>10.4f} {a3:>10.4f} {a3/max(a2,1e-9):>7.3f}")
# also by gap structure: distance of 2nd-highest to highest
Xs=np.sort(X,axis=1)[:,::-1]; gap=Xs[:,0]-Xs[:,1]
print("\n(3b) by top gap (max - 2nd):")
print(f"  {'gap':>4} {'#pts':>7} {'mean|R2|':>10} {'mean|R3|':>10} {'R3/R2':>7}")
for gg in range(0,8):
    msk=gap==gg
    if msk.sum()<20: continue
    a2=np.abs(R2[msk]).mean(); a3=np.abs(R3[msk]).mean()
    print(f"  {gg:>4} {int(msk.sum()):>7} {a2:>10.4f} {a3:>10.4f} {a3/max(a2,1e-9):>7.3f}")
print(f"\n[{time.time()-t0:.0f}s]  constant R3/R2 across strata => self-similar residual => contraction plausible")
