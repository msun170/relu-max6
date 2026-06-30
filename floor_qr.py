# CHEAP STABLE weight-4 floor: generate K random weight-4 join columns, ONE-SHOT cusolver QR (stable, no GS drift),
# floor = ||b - Q Q^T b||. Report for a few K (floor-vs-#columns trend) with a random-target CONTROL. Keep K < m so
# the columns are full-rank and Q spans exactly their colspan (non-vacuous; control stays high).
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = 7; W = 4; t0 = time.time()
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
m = 24000
X = rng.integers(-15, 16, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.float32)); brand = cp.asarray(rng.standard_normal(m).astype(np.float32))
P = cp.asarray((X @ Wn.T).astype(np.float32))
nbM = float(cp.linalg.norm(bmax)); nbR = float(cp.linalg.norm(brand))
print(f"{len(W4)} pts, {ng} gens, m={m}  [{time.time()-t0:.0f}s]", flush=True)
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

KS = [8000, 14000, 20000]
Kmax = max(KS)
cols = cp.empty((m, Kmax), dtype=cp.float32)
for j in range(Kmax):
    a = rzi(); b = rzi(); cols[:, j] = cp.maximum(P[:, a].max(axis=1), P[:, b].max(axis=1))
    if j % 5000 == 0: print(f"  gen {j}/{Kmax}  [{time.time()-t0:.0f}s]", flush=True)
lin = cp.asarray(X.astype(np.float32))
print(f"  {'K':>6} {'f(max7)':>10} {'f(control)':>11}", flush=True)
for K in KS:
    A = cp.concatenate([cols[:, :K], lin], axis=1)
    Q, _ = cp.linalg.qr(A); del A                       # one-shot stable QR (cusolver)
    fM = float(cp.linalg.norm(bmax - Q @ (Q.T @ bmax)))/nbM
    fR = float(cp.linalg.norm(brand - Q @ (Q.T @ brand)))/nbR
    print(f"  {K:>6} {fM:>10.6f} {fR:>11.4f}  [{time.time()-t0:.0f}s]", flush=True)
    del Q; cp.get_default_memory_pool().free_all_blocks()
print(f"\n[{time.time()-t0:.0f}s] floor decreasing with K toward the true weight-4 floor; control high = non-vacuous.", flush=True)
