# DECISIVE weight-4 test: span the COMPLETE weight-4 join space by random raw joins (no canonicalization, no
# cusolver -- mod-p forward elimination is matmul-only), saturate the rank, then test max_7 membership with a
# random-target CONTROL. If rank saturates < m (control OUT = non-vacuous) and max_7 is IN -> max_7 is 2-layer at
# weight-4 (DECISIVE, verify exactly after). If max_7 OUT while the rank has saturated (adding joins stops growing
# rank) -> max_7 OUT of COMPLETE weight-4 (so needs weight>=5 or 3 layers). Saturation = the random sample
# captured the full join span.
import sys, itertools, time, gc
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
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
gens = list(gens)
rng = np.random.default_rng(0)
m = 28000
X = rng.integers(-15, 16, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64)
brand = rng.integers(-100, 101, size=m).astype(np.int64)
Pmat = (X @ Wn.T)                                   # m x 210, integer
print(f"{len(W4)} pts, {len(gens)} gens, m={m}  [{time.time()-t0:.0f}s]", flush=True)

ng = len(gens)
def rand_zono_idx():
    p = W4[rng.integers(len(W4))]; verts = {p}; added = 0; tries = 0
    while added < 4 and len(verts) < 12 and tries < 60:
        tries += 1
        g = gens[rng.integers(ng)]; nv = set(); ok = True   # sample one generator at a time (fast)
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok and nv - verts:
            verts |= nv; added += 1
    return [vidx[v] for v in verts]

K = 31000                                           # random raw join columns (> expected rank for saturation)
cols = np.empty((m, K), dtype=np.int64)
for k in range(K):
    a = rand_zono_idx(); b = rand_zono_idx()
    ia = Pmat[:, a].max(axis=1); ib = Pmat[:, b].max(axis=1)
    cols[:, k] = np.maximum(ia, ib)                 # join column = max of two zonotope columns
    if k % 6000 == 0: print(f"  gen {k}/{K}  [{time.time()-t0:.0f}s]", flush=True)
raw = [cols, X.astype(np.int64), bmax.reshape(-1,1), brand.reshape(-1,1)]
RAW = np.concatenate(raw, axis=1); del cols, raw; gc.collect()
L = K + n                                            # design columns (joins + linear); targets are last 2
print(f"matrix {RAW.shape} (L={L} design cols)  [{time.time()-t0:.0f}s]", flush=True)

def membership(P):
    A = cp.asarray(RAW).astype(cp.int64) % P; rows, ncols = A.shape; r = 0
    rank_at = []
    for c in range(L):
        sub = A[r:, c]; nz = cp.nonzero(sub)[0]
        if nz.size == 0:
            if c % 3000 == 0: rank_at.append((c, r))
            continue
        piv = r + int(nz[0].item())
        if piv != r:
            tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
        A[r] = (A[r] * pow(int(A[r, c].item()), P-2, P)) % P
        if r+1 < rows:
            f = A[r+1:, c]; A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= P
        r += 1
        if c % 3000 == 0: print(f"  [p={P}] col {c}: rank {r}  [{time.time()-t0:.0f}s]", flush=True)
        if r == rows: break
    rank = r
    mi = bool(cp.all(A[rank:, L] % P == 0).item()); ri = bool(cp.all(A[rank:, L+1] % P == 0).item())
    del A; cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    return rank, mi, ri

for P in (2147483647,):
    rank, mi, ri = membership(P)
    sat = rank < L - 2000          # rank saturated well before using all columns => full join span captured
    print(f"[p={P}] rank={rank}/{m} (of {L} cols; saturated~={sat}) | rand {'IN(VACUOUS)' if ri else 'OUT(ok)'} | "
          f"max7 {'IN => 2 LAYERS at weight-4 !!!' if mi else 'OUT of COMPLETE weight-4 (if saturated)'} | "
          f"valid={rank<m and not ri}  [{time.time()-t0:.0f}s]", flush=True)
