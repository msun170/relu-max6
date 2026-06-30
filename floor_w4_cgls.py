# Clean saturated weight-4 floor: generate a large UNBIASED random set of weight-4 join columns ONCE, then a
# single CGLS projection (no incremental-SVD slowdown that stalled the span-saturation version). If #cols > rank
# and m > rank, this is the true floor f(4); rho_34 = f(4)/f(3) then tests the geometric contraction.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
n = 7; W = 4
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W4 = wpoints(W); W4set = set(W4); vidx = {p: i for i, p in enumerate(W4)}
Wn = np.array(W4, dtype=np.int64)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = list(gens)
t0 = time.time(); rng = np.random.default_rng(1)
m = 30000
X = rng.integers(-14, 15, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
P = cp.asarray((X @ Wn.T).astype(np.float32))      # m x 210
print(f"{len(W4)} pts, {len(gens)} gens, m={m}  [{time.time()-t0:.0f}s]", flush=True)

def random_zono_verts():
    p = W4[rng.integers(len(W4))]; verts = {p}; added = 0
    for j in rng.permutation(len(gens)):
        g = gens[j]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok:
            verts |= nv; added += 1
            if added >= 5 or len(verts) >= 16: break
    return [vidx[v] for v in verts]

K = 15000                                           # random zonotopes -> their columns
Zc = cp.empty((m, K), dtype=cp.float32)
for b in range(K):
    Zc[:, b] = P[:, random_zono_verts()].max(axis=1)
    if b % 3000 == 0: print(f"  zono {b}/{K}  [{time.time()-t0:.0f}s]", flush=True)
# join columns = elementwise max of random zonotope pairs; assemble [zonos | joins | linear]
perm = cp.asarray(rng.permutation(K)); J1 = cp.maximum(Zc, Zc[:, perm])
perm2 = cp.asarray(rng.permutation(K)); J2 = cp.maximum(Zc, Zc[:, perm2])
Xg = cp.asarray(X.astype(np.float32))
A = cp.concatenate([Zc, J1, J2, Xg], axis=1)        # ~45007 columns
print(f"A {A.shape} (cols > rank, m={m} > rank)  [{time.time()-t0:.0f}s]", flush=True)

cn = cp.linalg.norm(A, axis=0); cn = cp.where(cn > 0, cn, cp.float32(1.0)); A /= cn
bg = cp.asarray(bmax); bn = float(cp.linalg.norm(bg)); An = float(cp.linalg.norm(A))
x = cp.zeros(A.shape[1], dtype=cp.float32); r = bg.copy()
s = A.T @ r; p = s.copy(); gamma = float(s @ s)
for it in range(1, 6001):
    q = A @ p; al = gamma/float(q @ q); x += al*p; r -= al*q
    s = A.T @ r; gnew = float(s @ s); be = gnew/gamma; p = s + be*p; gamma = gnew
    if it % 400 == 0:
        rr = float(cp.linalg.norm(r))/bn; o = float(cp.linalg.norm(s))/(An*float(cp.linalg.norm(r))+1e-30)
        print(f"  it {it}: f(4)~={rr:.5f} ortho={o:.1e}  [{time.time()-t0:.0f}s]", flush=True)
        if o < 1e-6: break
rr = float(cp.linalg.norm(r))/bn
print(f"\nf(4) = {rr:.5f}   f(3)=0.00159 f(2)=0.0320 => rho_34 = f(4)/f(3) = {rr/0.00159:.3f}, rho_23=0.050", flush=True)
print(f"(geometric contraction if rho_34 ~ rho_23; closure if rho_w bounded < 1)", flush=True)
