# IN probe: is max_7 a signed sum of ORBIT-SUMMED structured weight-4 joins? Generate diverse weight-4 join
# block reps (joins of small zonotopes in varied relative positions), orbit-sum each over S_7 ON THE FLY (no
# global canonicalization), then exact mod-p membership. IN => max_7 is 2-layer at weight-4 (verify exactly).
# OUT => inconclusive (this is a subfamily of complete weight-4).
import sys, itertools, time, gc
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; W = 4; t0 = time.time()
perms = list(itertools.permutations(range(n)))   # S_7, 5040
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W4 = wpoints(W); W4set = set(W4)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = sorted(gens)
def ssz(p, G):
    verts = set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G, s):
            q = p
            for gg in S: q = shift(q, gg)
            if q not in W4set: return None
            verts.add(q)
    return frozenset(verts)
# small zonotopes (<=2 gens), take S_7 orbit reps (canonicalize just these -- few)
zonos = set()
for p in W4:
    for g in gens:
        v = ssz(p, [g])
        if v and len(v) >= 2: zonos.add(v)
for p in W4:
    for i in range(len(gens)):
        if shift(p, gens[i]) not in W4set: continue
        for j in range(i+1, len(gens)):
            v = ssz(p, [gens[i], gens[j]])
            if v and len(v) >= 3: zonos.add(v)
zonos = list(zonos); zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
print(f"{len(zreps)} small-zonotope orbit reps  [{time.time()-t0:.0f}s]", flush=True)

def orbit_of(block):
    seen = set()
    for g in perms:
        seen.add(frozenset(tuple(v[g[k]] for k in range(n)) for v in block))
    return list(seen)

# diverse weight-4 join block reps: rep_i | (permuted rep_j), keep those that are valid weight-4 blocks (verts in W4)
rng = np.random.default_rng(11)
blockreps = []
seenkeys = set()
zlist = zonos
TARGET = 2500
for zi in zreps:
    for _ in range(300):
        zj = zlist[rng.integers(len(zlist))]
        u = zi | zj
        if len(u) > 12 or len(u) < 4: continue
        key = frozenset(u)                      # cheap identity dedup (orbit-dups harmless: redundant columns)
        if key in seenkeys: continue
        seenkeys.add(key); blockreps.append(u)
        if len(blockreps) >= TARGET: break
    if len(blockreps) >= TARGET: break
print(f"{len(blockreps)} distinct weight-4 join block reps  [{time.time()-t0:.0f}s]", flush=True)

N = len(blockreps)
m = N + 1200
X = np.random.default_rng(303).integers(-18, 19, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64); brand = np.random.default_rng(13).integers(-90,91,size=m).astype(np.int64)
cols = []
for bi, block in enumerate(blockreps):
    cols.append(core.orbit_column(orbit_of(block), X).astype(np.int64))
    if bi % 800 == 0: print(f"  col {bi}/{N}  [{time.time()-t0:.0f}s]", flush=True)
raw = cols + [X[:, i].astype(np.int64) for i in range(n)]; L = len(raw); raw += [bmax, brand]
RAW = np.column_stack(raw).astype(np.int64); del raw, cols; gc.collect()
print(f"matrix {RAW.shape} (L={L})  [{time.time()-t0:.0f}s]", flush=True)

def membership(P):
    A = cp.asarray(RAW) % P; rows, ncols = A.shape; r = 0
    for c in range(L):
        sub = A[r:, c]; nz = cp.nonzero(sub)[0]
        if nz.size == 0: continue
        piv = r + int(nz[0].item())
        if piv != r:
            tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
        A[r] = (A[r] * pow(int(A[r, c].item()), P-2, P)) % P
        if r+1 < rows:
            f = A[r+1:, c]; A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= P
        r += 1
        if r == rows: break
    rank = r
    mi = bool(cp.all(A[rank:, L] % P == 0).item()); ri = bool(cp.all(A[rank:, L+1] % P == 0).item())
    del A; cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    return rank, mi, ri

for P in (2147483647, 2147483629):
    rank, mi, ri = membership(P)
    print(f"[p={P}] rank={rank}/{m} | rand {'IN(VAC)' if ri else 'OUT(ok)'} | "
          f"max7 {'IN => 2 LAYERS weight-4!!' if mi else 'OUT of this subfamily'} | valid={rank<m and not ri}  [{time.time()-t0:.0f}s]", flush=True)
