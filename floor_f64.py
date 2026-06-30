# Exact f(3) and f(4)-root floors in FLOAT64 via eigh(AA^T) (cusolver). float64 resolves the ill-conditioned
# small directions that float32 cannot. Report floor vs #directions kept; true floor = value at full rank.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
m = 20000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float64)
brand = np.random.default_rng(5).standard_normal(m).astype(np.float64)
C3 = np.load(f"{SCR}/Crel_w3full_m26000.npy")[:m].astype(np.float64)
C4 = np.load(f"{SCR}/Crel_w4root_m26000.npy")[:m].astype(np.float64)
print(f"loaded C3 {C3.shape}, C4 {C4.shape}  [{time.time()-t0:.0f}s]", flush=True)

def floor_curve(Anp, tag):
    A = cp.asarray(Anp, dtype=cp.float64)
    cn = cp.linalg.norm(A, axis=0); cn = cp.where(cn>0, cn, cp.float64(1.0)); A /= cn
    G = A @ A.T; del A; cp.get_default_memory_pool().free_all_blocks()
    w, V = cp.linalg.eigh(G); del G; cp.get_default_memory_pool().free_all_blocks()  # ascending
    w = w[::-1]; V = V[:, ::-1]                       # descending
    wmax = float(w[0])
    bM = cp.asarray(bmax); bR = cp.asarray(brand)
    pM = V.T @ bM; pR = V.T @ bR; nbM = float(cp.linalg.norm(bM)); nbR = float(cp.linalg.norm(bR))
    p2M = cp.asnumpy(pM**2); p2R = cp.asnumpy(pR**2); wv = cp.asnumpy(w)
    print(f"  {tag}: wmax={wmax:.3e}, wmin={float(w[-1]):.3e}", flush=True)
    print(f"    {'rank':>6} {'w_cut/wmax':>11} {'f(max7)':>10} {'f(control)':>11}", flush=True)
    for k in (685, 2000, 5000, 10000, 15000, 18000, 19000, m):
        if k > m: k = m
        wc = wv[k-1]/wmax if k-1 < len(wv) else 0.0
        fM = (max(nbM**2 - p2M[:k].sum(), 0.0)**0.5)/nbM; fR = (max(nbR**2 - p2R[:k].sum(), 0.0)**0.5)/nbR
        print(f"    {k:>6} {wc:>11.2e} {fM:>10.6f} {fR:>11.4f}", flush=True)
    del V, w, bM, bR, pM, pR; cp.get_default_memory_pool().free_all_blocks()

lin = X.astype(np.float64)
floor_curve(np.column_stack([C3, lin]), "f(3) complete weight-3")
floor_curve(np.column_stack([C3, C4, lin]), "f(4) weight-3 + root-weight-4")
print(f"\n[{time.time()-t0:.0f}s] true floor = f(max7) where control still high but rank near full.", flush=True)
