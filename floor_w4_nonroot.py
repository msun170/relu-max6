# Does NON-ROOT weight-4 structure push f(4) below the root-only 0.00090? Reuses cached complete-weight-3 (C3)
# and root-weight-4 (C4) columns, then ADDS a large sample of non-root weight-4 join columns. If f(4) drops a lot
# => contraction continues (closure). If it stays ~0.0009 => plateau (f_inf>0 => separation finitely provable).
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
m = 26000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
brand = np.random.default_rng(5).standard_normal(m).astype(np.float32)
C3 = np.load(f"{SCR}/Crel_w3full_m{m}.npy"); C4 = np.load(f"{SCR}/Crel_w4root_m{m}.npy")
print(f"loaded C3 {C3.shape}, C4root {C4.shape}  [{time.time()-t0:.0f}s]", flush=True)

# weight-4 lattice + NON-root generators
W4 = [c for c in itertools.product(range(5), repeat=n) if sum(c) == 4]; W4set = set(W4); vidx = {p:i for i,p in enumerate(W4)}
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
allg = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); gg = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//gg for x in dd)
        if dd > tuple(-x for x in dd): allg.add(dd)
rootset = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a!=b}
nonroot = [g for g in allg if g not in rootset and tuple(-x for x in g) not in rootset]
print(f"{len(W4)} pts, {len(nonroot)} non-root gens  [{time.time()-t0:.0f}s]", flush=True)
P = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(np.float32))
rng = np.random.default_rng(7)
def rz():  # random NON-root weight-4 zonotope (at least one non-root generator)
    p = W4[rng.integers(len(W4))]; verts = {p}; added = 0
    for j in rng.permutation(len(nonroot)):
        g = nonroot[j]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok:
            verts |= nv; added += 1
            if added >= 4 or len(verts) >= 14: break
    return [vidx[v] for v in verts] if len(verts) >= 2 else None

K = 6000
Zc = cp.empty((m, K), dtype=cp.float32); cnt = 0
while cnt < K:
    idx = rz()
    if idx is None: continue
    Zc[:, cnt] = P[:, idx].max(axis=1); cnt += 1
    if cnt % 4000 == 0: print(f"  nonroot zono {cnt}/{K}  [{time.time()-t0:.0f}s]", flush=True)
perm = cp.asarray(rng.permutation(K)); J = cp.maximum(Zc, Zc[:, perm])
NRcpu = np.concatenate([cp.asnumpy(Zc), cp.asnumpy(J)], axis=1)
del Zc, J, P; cp.get_default_memory_pool().free_all_blocks()
A_cpu = np.column_stack([C3, C4, NRcpu, X.astype(np.float32)])   # assemble on CPU
del NRcpu
print(f"non-root weight-4 added; combined A {A_cpu.shape} (CPU)  [{time.time()-t0:.0f}s]", flush=True)
A = cp.asarray(A_cpu); del A_cpu
cn = cp.linalg.norm(A, axis=0); cn = cp.where(cn>0, cn, cp.float32(1.0)); A /= cn   # in place
An = float(cp.linalg.norm(A))
print(f"on GPU, normalized  [{time.time()-t0:.0f}s]", flush=True)

def cgls(bnp, tag):
    bg = cp.asarray(bnp); bn = float(cp.linalg.norm(bg))
    x = cp.zeros(A.shape[1], dtype=cp.float32); r = bg.copy(); s = A.T@r; p = s.copy(); gamma = float(s@s)
    for it in range(1, 20001):
        q = A@p; al = gamma/float(q@q); x += al*p; r -= al*q
        s = A.T@r; gn = float(s@s); be = gn/gamma; p = s+be*p; gamma = gn
        if it % 2000 == 0:
            o = float(cp.linalg.norm(s))/(An*float(cp.linalg.norm(r))+1e-30)
            print(f"    {tag} it{it}: f={float(cp.linalg.norm(r))/bn:.5f} ortho={o:.1e}  [{time.time()-t0:.0f}s]", flush=True)
            if o < 1e-6: break
    return float(cp.linalg.norm(r))/bn

fr = cgls(brand, "ctrl"); print(f"  CONTROL = {fr:.4f} (must be >>0)", flush=True)
f4 = cgls(bmax, "max7")
print(f"\nf(4) with NON-ROOT = {f4:.5f}   [root-only was 0.00090, f(3)=0.00159]")
print(f"  rho_34 = {f4/0.00159:.3f}.  much-below-0.0009 => contraction continues; ~0.0009 => plateau => separation")
