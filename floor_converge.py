# Run CGLS to TRUE convergence (plateau) for f(3) and f(4), so we read real floors not under-converged residuals.
# Stop when the residual stops decreasing (relative drop over a 2000-iter window < 2e-3 for 3 consecutive windows).
import sys, itertools, time
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
C3 = np.load(f"{SCR}/Crel_w3full_m{m}.npy"); C4 = np.load(f"{SCR}/Crel_w4root_m{m}.npy")
print(f"loaded C3 {C3.shape}, C4root {C4.shape}  [{time.time()-t0:.0f}s]", flush=True)

# rebuild the non-root weight-4 sample (seed 7, same as floor_w4_nonroot)
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
perm = cp.asarray(rng.permutation(K)); J = cp.maximum(Zc, Zc[:, perm])
NRcpu = np.concatenate([cp.asnumpy(Zc), cp.asnumpy(J)], axis=1); del Zc, J, P; cp.get_default_memory_pool().free_all_blocks()
print(f"non-root sample built  [{time.time()-t0:.0f}s]", flush=True)

def converge(Anp, tag):
    A = cp.asarray(Anp); cn = cp.linalg.norm(A, axis=0); cn = cp.where(cn>0, cn, cp.float32(1.0)); A /= cn
    bg = cp.asarray(bmax); bn = float(cp.linalg.norm(bg))
    x = cp.zeros(A.shape[1], dtype=cp.float32); r = bg.copy(); s = A.T@r; p = s.copy(); gamma = float(s@s)
    last = 1.0; flat = 0; it = 0
    while it < 200000:
        for _ in range(2000):
            q = A@p; al = gamma/float(q@q); x += al*p; r -= al*q
            s = A.T@r; gn = float(s@s); be = gn/gamma; p = s+be*p; gamma = gn
        it += 2000
        f = float(cp.linalg.norm(r))/bn
        drop = (last - f)/last
        print(f"    {tag} it{it}: f={f:.6f}  drop/2k={drop*100:.2f}%  [{time.time()-t0:.0f}s]", flush=True)
        if drop < 2e-3: flat += 1
        else: flat = 0
        if flat >= 3: break
        last = f
    del A, bg, x, r, s, p; cp.get_default_memory_pool().free_all_blocks()
    return f

lin = X.astype(np.float32)
f3 = converge(np.column_stack([C3, lin]), "f3      ")
f4r = converge(np.column_stack([C3, C4, lin]), "f4root  ")
f4nr = converge(np.column_stack([C3, C4, NRcpu, lin]), "f4nonrt ")
print(f"\nCONVERGED floors: f(2)=0.0320  f(3)={f3:.6f}  f(4)root={f4r:.6f}  f(4)nonroot={f4nr:.6f}")
print(f"  rho_23={f3/0.0320:.3f}  rho_34(root)={f4r/f3:.3f}  rho_34(nonroot)={f4nr/f3:.3f}")
print(f"  if f(4) plateaus >0 well above 0 and rho_34>>rho_23 (rising toward 1): PLATEAU => separation provable.")
print(f"  if rho_34 ~ rho_23 or floor still ~halving: geometric-ish => closure.")
