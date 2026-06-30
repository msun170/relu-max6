# Exact floor via SVD of A directly (condition number kappa, NOT kappa^2 like eigh(AA^T)) -- the numerically
# stable way. The block matrices are EXTREMELY ill-conditioned (singular values span many orders), which is the
# real reason the floor is delicate. Report floor(max7) and floor(control) as a function of #singular-directions
# kept (the floor-vs-rank curve); the true floor is the value as rank -> full. cusolver now available (gpu_init).
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
m = 22000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
brand = np.random.default_rng(5).standard_normal(m).astype(np.float32)
C3 = np.load(f"{SCR}/Crel_w3full_m26000.npy")[:m]; C4 = np.load(f"{SCR}/Crel_w4root_m26000.npy")[:m]
print(f"loaded C3 {C3.shape}, C4 {C4.shape}  [{time.time()-t0:.0f}s]", flush=True)

def floor_curve(Anp, tag):
    A = cp.asarray(Anp, dtype=cp.float32)
    cn = cp.linalg.norm(A, axis=0); cn = cp.where(cn>0, cn, cp.float32(1.0)); A /= cn
    # economic SVD: A = U s Vt, U is m x min(m,N)
    U, s, _ = cp.linalg.svd(A, full_matrices=False); del A, _
    bM = cp.asarray(bmax); bR = cp.asarray(brand)
    pM = U.T @ bM; pR = U.T @ bR; nbM = float(cp.linalg.norm(bM)); nbR = float(cp.linalg.norm(bR))
    smax = float(s[0])
    print(f"  {tag}: smax={smax:.3e}, smin={float(s[-1]):.3e}, #s={s.size}", flush=True)
    print(f"    {'rank':>6} {'s_cut/smax':>11} {'f(max7)':>10} {'f(control)':>11}", flush=True)
    cumM = float(cp.sum(pM**2)); cumR = float(cp.sum(pR**2))  # full projection (all directions)
    # floor using top-k singular directions for several k
    p2M = cp.asnumpy(pM**2); p2R = cp.asnumpy(pR**2); sv = cp.asnumpy(s)
    for k in (50, 200, 685, 2000, 5000, 10000, 15000, s.size):
        if k > s.size: k = s.size
        keepM = p2M[:k].sum(); keepR = p2R[:k].sum()
        fM = (max(nbM**2 - keepM, 0.0)**0.5)/nbM; fR = (max(nbR**2 - keepR, 0.0)**0.5)/nbR
        print(f"    {k:>6} {sv[k-1]/smax:>11.2e} {fM:>10.5f} {fR:>11.4f}", flush=True)
    del U, s, bM, bR, pM, pR; cp.get_default_memory_pool().free_all_blocks()

lin = X.astype(np.float32)
floor_curve(np.column_stack([C3, lin]), "f(3) complete weight-3")
floor_curve(np.column_stack([C3, C4, lin]), "f(4) weight-3 + root-weight-4")
print(f"\n[{time.time()-t0:.0f}s] true floor = f at rank->full (last row). control should stay high until near-full rank.", flush=True)
