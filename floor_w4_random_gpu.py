# TRUE weight-4 floor with NO greedy plateau and NO full enumeration.
# Key: the floor only depends on the COLUMN SPACE of all weight-4 joins (a subspace of R^m, dim <= m).
# Greedy colgen plateaus because it picks biased columns; instead we UNBIASEDLY sample random weight-4 joins
# and grow an orthonormal basis on GPU until the rank SATURATES (the full span is captured). Then
#   f(4) = ||bmax - Q Q^T bmax|| / ||bmax||  is the true floor.
# A join column = elementwise max of its two zonotopes' columns; a zonotope column = max over its vertex set
# of P[:, verts], where P = X . W4^T (m x 210) is precomputed once. So everything is fast indexing + max on GPU.
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
t0 = time.time()
rng = np.random.default_rng(0)
m = 26000
X = rng.integers(-14, 15, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
P = cp.asarray((X @ Wn.T).astype(np.float32))      # m x 210, on GPU
bg = cp.asarray(bmax)
print(f"{len(W4)} lattice pts, {len(gens)} gens, m={m}  [{time.time()-t0:.0f}s]", flush=True)

def random_zono_verts():
    # random valid weight-4 zonotope: random base, add random generators while all vertices stay in W4 (dim<=4 -> few gens)
    p = W4[rng.integers(len(W4))]
    verts = {p}
    gi = rng.permutation(len(gens))
    added = 0
    for j in gi:
        g = gens[j]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok:
            verts |= nv; added += 1
            if added >= 5 or len(verts) >= 16: break
    return [vidx[v] for v in verts]

def zono_cols(batch):
    # build `batch` random zonotope columns (m x batch) on GPU
    cols = cp.empty((m, batch), dtype=cp.float32)
    for b in range(batch):
        idx = random_zono_verts()
        cols[:, b] = P[:, idx].max(axis=1)
    return cols

# orthonormal basis Q (m x k); start with linear functions
Q = cp.linalg.qr(cp.asarray(X.astype(np.float32)))[0]
bn = float(cp.linalg.norm(bg))
TOL = 1e-4
batch = 400
stall = 0; rounds = 0
while stall < 6 and Q.shape[1] < m - 100:
    rounds += 1
    Z = zono_cols(batch)
    # random join columns = elementwise max of random zonotope pairs (plus the singles themselves)
    perm = cp.asarray(rng.permutation(batch))
    J = cp.maximum(Z, Z[:, perm])
    C = cp.concatenate([Z, J], axis=1)             # singles + joins this round
    # project off Q, keep rank-increasing directions (modified Gram-Schmidt, batched)
    R = C - Q @ (Q.T @ C)
    R = R - Q @ (Q.T @ R)                           # reorthogonalize once for stability
    added = 0
    # SVD of residual block to extract new orthonormal directions
    U, s, _ = cp.linalg.svd(R, full_matrices=False)
    newdirs = U[:, s > TOL * float(cp.sqrt(cp.asarray(m, dtype=cp.float32)))]
    if newdirs.shape[1] > 0:
        Q = cp.concatenate([Q, newdirs], axis=1); added = newdirs.shape[1]
    relres = float(cp.linalg.norm(bg - Q @ (Q.T @ bg))) / bn
    k = Q.shape[1]
    stall = stall + 1 if added == 0 else 0
    if rounds % 5 == 0 or added == 0:
        print(f"  round {rounds}: rank={k} (+{added})  f(4)~={relres:.5f}  stall={stall}  [{time.time()-t0:.0f}s]", flush=True)
    if k > m - 100:
        print("  WARNING: rank approaching m -> increase m (vacuity risk)", flush=True); break

relres = float(cp.linalg.norm(bg - Q @ (Q.T @ bg))) / bn
print(f"\nf(4) = {relres:.5f}   (rank {Q.shape[1]}, m={m}, saturated after stall={stall})  [{time.time()-t0:.0f}s]", flush=True)
print(f"compare: f(2)=0.0308  f(3)=0.0022   (if f(4)<<f(3) the collapse continues; if ~0 max_7 may be IN weight-4)", flush=True)
