# EXPERIMENT 1: controlled QR sweep over n=7,8,9,10 and W=4,5,6. For each (n,W,K): sample K random two-way joins,
# one-shot cusolver QR, project max_n and several SYMMETRIC controls + a random control. Metric = ratio
# floor(max_n)/floor(control mean) (robust even as the near-full-dim span makes raw floors slide to 0).
# HYPOTHESIS: n=7,8,9 ratio drops strongly (special => IN-leaning), n=10 behaves like control (negative case).
# Symmetric controls: random CPWL functions of the SORTED coordinates (symmetric like max_n, but generic).
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
t0 = time.time()

def sym_controls(Xcp, seedbase, ncon=3):
    # random CPWL functions of sorted coords: relu net on sort(x). symmetric, generic.
    Xs = cp.sort(Xcp, axis=1)[:, ::-1].astype(cp.float32)   # descending sorted coords (m x n)
    n = Xs.shape[1]; outs = []
    for c in range(ncon):
        r = cp.random.RandomState(seedbase + c)
        H = 40; W1 = r.standard_normal((n, H)).astype(cp.float32); b1 = r.standard_normal((1, H)).astype(cp.float32)
        w2 = r.standard_normal((H, 1)).astype(cp.float32)
        o = cp.maximum(Xs @ W1 + b1, 0) @ w2
        outs.append(o[:, 0])
    return outs

def build(n, W, m, rng):
    L = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; Ls = set(L); vidx = {p:i for i,p in enumerate(L)}
    def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
    gens = set()
    for p in L:
        for q in L:
            if p == q: continue
            dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
            if dd > tuple(-x for x in dd): gens.add(dd)
    gens = list(gens); ng = len(gens)
    X = rng.integers(-15, 16, size=(m, n)).astype(np.int64)
    P = cp.asarray((X @ np.array(L, dtype=np.int64).T).astype(np.float32))
    def rzi():
        p = L[rng.integers(len(L))]; verts = {p}; added = 0; tries = 0
        while added < 4 and len(verts) < 12 and tries < 60:
            tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
            for u in verts:
                w = shift(u, g)
                if w not in Ls: ok = False; break
                nv.add(w)
            if ok and nv - verts: verts |= nv; added += 1
        return [vidx[v] for v in verts]
    return L, P, X, rzi

def run(n, W, m, KS):
    rng = np.random.default_rng(0)
    L, P, X, rzi = build(n, W, m, rng)
    Xcp = cp.asarray(X)
    bmax = cp.asarray(X.max(axis=1).astype(np.float32))
    ctrls = sym_controls(Xcp, 1000) + [cp.asarray(rng.standard_normal(m).astype(np.float32))]   # 3 sym + 1 gaussian
    nbM = float(cp.linalg.norm(bmax)); nbC = [float(cp.linalg.norm(c)) for c in ctrls]
    Kmax = max(KS); cols = cp.empty((m, Kmax), dtype=cp.float32)
    for j in range(Kmax):
        a = rzi(); b = rzi(); cols[:, j] = cp.maximum(P[:, a].max(axis=1), P[:, b].max(axis=1))
    lin = cp.asarray(X.astype(np.float32))
    for K in KS:
        A = cp.concatenate([cols[:, :K], lin], axis=1)
        Q, _ = cp.linalg.qr(A); del A
        fM = float(cp.linalg.norm(bmax - Q@(Q.T@bmax)))/nbM
        fCs = [float(cp.linalg.norm(c - Q@(Q.T@c)))/nbC[i] for i, c in enumerate(ctrls)]
        fCmean = sum(fCs)/len(fCs)
        print(f"  n={n} W={W} K={K:>6}: f(max{n})={fM:.5f}  f(ctrl_mean)={fCmean:.5f}  RATIO={fM/max(fCmean,1e-9):.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del Q; cp.get_default_memory_pool().free_all_blocks()
    del cols, P; cp.get_default_memory_pool().free_all_blocks()

m = 22000; KS = [5000, 10000, 18000]
for n in (7, 8, 9, 10):
    for W in (4, 5, 6):
        run(n, W, m, KS)
print(f"\n[{time.time()-t0:.0f}s] LOOK: do n=7,8,9 ratios drop strongly at some W while n=10 stays high (like control)?", flush=True)
