# GPU (CuPy) definitive test: is max7 in COMPLETE weight-3? Builds the matrix on CPU, runs the exact mod-p
# forward elimination on the GPU. m=20000 > #columns(19226) >= rank, so the test is GUARANTEED non-vacuous
# (no rank guessing). Tuned for a 12 GB card (int64 matrix ~3 GB + one outer-product temp ~3 GB ~= 6 GB peak).
# Run AFTER the CUDA driver is healthy (nvidia-smi must show a CUDA version, not "N/A"). Two primes confirm.
import sys, itertools, time, gc
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; W = 3
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W3 = wpoints(W); W3set = set(W3)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W3:
    for q in W3:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = sorted(gens)
t0 = time.time()
def ssz(p, G):
    verts = set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G, s):
            q = p
            for g in S: q = shift(q, g)
            if q not in W3set: return None
            verts.add(q)
    return frozenset(verts)
zonos = set()
for p in W3:
    for g in gens:
        v = ssz(p, [g])
        if v and len(v) >= 2: zonos.add(v)
for p in W3:
    for i in range(len(gens)):
        if shift(p, gens[i]) not in W3set: continue
        for j in range(i+1, len(gens)):
            v = ssz(p, [gens[i], gens[j]])
            if v and len(v) >= 3: zonos.add(v)
base3 = set(zonos)
for z in [z for z in zonos if len(z) >= 4]:
    for g in gens:
        verts = set(z); ok = True
        for u in z:
            w = shift(u, g)
            if w not in W3set: ok = False; break
            verts.add(w)
        if ok and len(verts) > len(z): base3.add(frozenset(verts))
zonos = list(base3)
zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
blocks = set(zonos)
for a in zreps:
    for b_ in zonos:
        u = a | b_
        if len(u) <= 16: blocks.add(u)
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"N={N} weight-3 orbits  [{time.time()-t0:.0f}s]", flush=True)

m = 20000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64)
brand = np.random.default_rng(7).integers(-80, 81, size=m).astype(np.int64)

# build the RAW integer matrix ONCE (design columns first, then the two targets), reuse mod each prime
raw_cols = [core.orbit_column(orb[k], X).astype(np.int64) for k in okeys] + [X[:, i].astype(np.int64) for i in range(n)]
L = len(raw_cols)                      # number of design columns (pivot only on these)
raw_cols += [bmax, brand]
RAW = np.column_stack(raw_cols).astype(np.int64); del raw_cols; gc.collect()
print(f"raw matrix {RAW.shape} built (L={L} design cols)  [{time.time()-t0:.0f}s]", flush=True)

def gpu_membership(P):
    A = cp.asarray(RAW) % P            # reduce mod P on GPU; targets are the last 2 columns
    rows, ncols = A.shape; r = 0
    for c in range(L):                 # pivot only on design columns
        sub = A[r:, c]
        nz = cp.nonzero(sub)[0]
        if nz.size == 0: continue
        piv = r + int(nz[0].item())
        if piv != r:
            tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
        pivval = int(A[r, c].item())
        A[r] = (A[r] * pow(pivval, P-2, P)) % P
        if r + 1 < rows:
            f = A[r+1:, c]
            A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= P
        r += 1
        if r == rows: break
        if r % 2000 == 0: print(f"  [p={P}] rank {r}  [{time.time()-t0:.0f}s]", flush=True)
    rank = r
    max_in = bool(cp.all(A[rank:, L] % P == 0).item())
    rand_in = bool(cp.all(A[rank:, L+1] % P == 0).item())
    del A; cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    return rank, max_in, rand_in

for P in (2147483647, 2147483629):
    rank, max_in, rand_in = gpu_membership(P)
    valid = rank < m and not rand_in
    print(f"\n[p={P}] rank={rank} (m={m}) | random {'IN(VACUOUS)' if rand_in else 'OUT(valid)'} | "
          f"max7 {'IN => 2 HIDDEN LAYERS' if max_in else 'OUT => needs 3 layers'} | VALID={valid}  [{time.time()-t0:.0f}s]", flush=True)
print("done", flush=True)
