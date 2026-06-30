# EXACT floor via spectral projection (no slow CGLS). floor(b) = ||b - P b|| where P projects onto colspan(A).
# range(A) = range(A A^T); eigendecompose G=AA^T, project b onto eigenvectors with eigenvalue > tol.
# Report eigenvalue spectrum near the rank cutoff, floor at several tol, and a random-target CONTROL (must stay
# high). This gives the TRUE floor in one shot -- the plateau CGLS was crawling toward.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa: register cusolver DLLs before importing cupy
import cupy as cp
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
m = 26000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
brand = np.random.default_rng(5).standard_normal(m).astype(np.float32)
C3 = np.load(f"{SCR}/Crel_w3full_m{m}.npy"); C4 = np.load(f"{SCR}/Crel_w4root_m{m}.npy")

# non-root weight-4 sample (seed 7, same as before)
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
P = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(np.float32))
rng = np.random.default_rng(7)
def rz():
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
K = 6000; Zc = cp.empty((m, K), dtype=cp.float32); cnt = 0
while cnt < K:
    idx = rz()
    if idx is None: continue
    Zc[:, cnt] = P[:, idx].max(axis=1); cnt += 1
perm = cp.asarray(rng.permutation(K)); Jn = cp.maximum(Zc, Zc[:, perm])
NRcpu = np.concatenate([cp.asnumpy(Zc), cp.asnumpy(Jn)], axis=1); del Zc, Jn, P; cp.get_default_memory_pool().free_all_blocks()
print(f"families ready  [{time.time()-t0:.0f}s]", flush=True)

def spectral_floor(Anp, tag):
    A = cp.asarray(Anp); cn = cp.linalg.norm(A, axis=0); cn = cp.where(cn>0, cn, cp.float32(1.0)); A /= cn
    G = A @ A.T; del A; cp.get_default_memory_pool().free_all_blocks()
    w, V = cp.linalg.eigh(G); del G                          # ascending eigenvalues
    wmax = float(w[-1])
    bM = cp.asarray(bmax); bR = cp.asarray(brand)
    proj = V.T @ bM; projR = V.T @ bR                        # coords in eigenbasis
    nb = float(cp.linalg.norm(bM)); nr = float(cp.linalg.norm(bR))
    print(f"  {tag}: eigsum top, wmax={wmax:.3e}", flush=True)
    # spectrum: how many eigenvalues above each threshold
    for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        keep = w > tol*wmax
        rank = int(keep.sum().item())
        res2 = nb**2 - float((proj[keep]**2).sum()); resR2 = nr**2 - float((projR[keep]**2).sum())
        fM = (max(res2,0.0)**0.5)/nb; fR = (max(resR2,0.0)**0.5)/nr
        print(f"    tol={tol:.0e}*wmax: rank={rank:5d}  f(max7)={fM:.6f}  f(control)={fR:.4f}", flush=True)
    del V, w; cp.get_default_memory_pool().free_all_blocks()

lin = X.astype(np.float32)
spectral_floor(np.column_stack([C3, lin]), "f(3) weight-3 complete")
spectral_floor(np.column_stack([C3, C4, lin]), "f(4) +root")
spectral_floor(np.column_stack([C3, C4, NRcpu, lin]), "f(4) +root+nonroot")
print(f"\n[{time.time()-t0:.0f}s] EXACT floors. control must be >>0 (non-vacuous). f(max7) stable across tol = trustworthy.")
