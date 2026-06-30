# Cheap stable QR floor pushed harder, for max_7 AND max_8, max_9 (n<=9 conjecture probe).
# For each n: random weight-4 join columns, one-shot cusolver QR, floor(max_n) and floor(control) vs K.
# If floor(max_n) -> 0 and ratio floor(max_n)/floor(control) -> 0, max_n is in closure(weight-4 span) => IN-leaning.
# Compare the n: do 7,8,9 all show the same closure pattern (supports n<=9 conjecture)?
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
W = 4; t0 = time.time()

def run(n, m, KS):
    W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; W4set = set(W4); vidx = {p:i for i,p in enumerate(W4)}
    Wn = np.array(W4, dtype=np.int64)
    def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
    gens = set()
    for p in W4:
        for q in W4:
            if p == q: continue
            dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
            if dd > tuple(-x for x in dd): gens.add(dd)
    gens = list(gens); ng = len(gens)
    rng = np.random.default_rng(0)
    X = rng.integers(-15, 16, size=(m, n)).astype(np.int64)
    bmax = cp.asarray(X.max(axis=1).astype(np.float32)); brand = cp.asarray(rng.standard_normal(m).astype(np.float32))
    P = cp.asarray((X @ Wn.T).astype(np.float32)); nbM = float(cp.linalg.norm(bmax)); nbR = float(cp.linalg.norm(brand))
    def rzi():
        p = W4[rng.integers(len(W4))]; verts = {p}; added = 0; tries = 0
        while added < 4 and len(verts) < 12 and tries < 60:
            tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
            for u in verts:
                w = shift(u, g)
                if w not in W4set: ok = False; break
                nv.add(w)
            if ok and nv - verts: verts |= nv; added += 1
        return [vidx[v] for v in verts]
    Kmax = max(KS); cols = cp.empty((m, Kmax), dtype=cp.float32)
    for j in range(Kmax):
        a = rzi(); b = rzi(); cols[:, j] = cp.maximum(P[:, a].max(axis=1), P[:, b].max(axis=1))
    lin = cp.asarray(X.astype(np.float32))
    print(f"=== max_{n} (m={m}, {len(W4)} pts) ===  [{time.time()-t0:.0f}s]", flush=True)
    print(f"  {'K':>6} {'f(maxn)':>10} {'f(ctrl)':>9} {'ratio':>8}", flush=True)
    for K in KS:
        A = cp.concatenate([cols[:, :K], lin], axis=1)
        Q, _ = cp.linalg.qr(A); del A
        fM = float(cp.linalg.norm(bmax - Q @ (Q.T @ bmax)))/nbM
        fR = float(cp.linalg.norm(brand - Q @ (Q.T @ brand)))/nbR
        print(f"  {K:>6} {fM:>10.6f} {fR:>9.4f} {fM/max(fR,1e-9):>8.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del Q; cp.get_default_memory_pool().free_all_blocks()
    del cols, P, bmax, brand, lin; cp.get_default_memory_pool().free_all_blocks()

run(7, 22000, [6000, 12000, 18000])
run(8, 22000, [6000, 12000, 18000])
run(9, 22000, [6000, 12000, 18000])
print(f"\n[{time.time()-t0:.0f}s] if f(maxn) << f(ctrl) and ratio shrinks for all n=7,8,9 -> all closure-leaning (n<=9 IN consistent).", flush=True)
