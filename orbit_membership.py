# EXP 3 (decisive for a small ansatz): the low-complexity weight-4 join family is SMALL (~hundreds of orbits) ->
# enumerable. Build all distinct low-complexity orbit-summed blocks, EXACT mod-p membership: is max_7 in their
# span? IN => max_7 has a small clean weight-4 construction (extract+verify). OUT => no small construction in this
# complexity class. Random control must be OUT (non-vacuous); m > N.
import sys, itertools, time, gc
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = 7; W = 4; t0 = time.time()
W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; W4set = set(W4); vidx = {p:i for i,p in enumerate(W4)}
NP = len(W4)
perms = list(itertools.permutations(range(n))); G = len(perms)
pa = np.empty((G, NP), dtype=np.int32)
for gi, g in enumerate(perms):
    for i, p in enumerate(W4): pa[gi, i] = vidx[tuple(p[g[k]] for k in range(n))]
pa = cp.asarray(pa)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = sorted(gens)
# enumerate small zonotopes (<=2 gens) completely
def ssz(p, Gset):
    verts = set()
    for s in range(len(Gset)+1):
        for S in itertools.combinations(Gset, s):
            q = p
            for g in S: q = shift(q, g)
            if q not in W4set: return None
            verts.add(q)
    return frozenset(verts)
zon = set()
for p in W4:
    for g in gens:
        v = ssz(p, [g])
        if v and len(v) >= 2: zon.add(v)
for p in W4:
    for i in range(len(gens)):
        if shift(p, gens[i]) not in W4set: continue
        for j in range(i+1, len(gens)):
            v = ssz(p, [gens[i], gens[j]])
            if v and len(v) >= 3: zon.add(v)
zon = list(zon)
print(f"{len(zon)} small zonotopes  [{time.time()-t0:.0f}s]", flush=True)
import core
zorb = core.orbits_of(zon, n); zreps = [zorb[k][0] for k in zorb]
print(f"{len(zreps)} zonotope orbits  [{time.time()-t0:.0f}s]", flush=True)
# joins of zono-rep with any zono, cap verts; orbit-reduce
blocks = set(frozenset(z) for z in zon)
for a in zreps:
    for b in zon:
        u = a | b
        if len(u) <= 10: blocks.add(frozenset(u))
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"N={N} low-complexity weight-4 join orbits  [{time.time()-t0:.0f}s]", flush=True)

m = N + 1500
X = np.random.default_rng(404).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64); brand = np.random.default_rng(9).integers(-90, 91, size=m).astype(np.int64)
raw = [core.orbit_column(orb[k], X).astype(np.int64) for k in okeys] + [X[:, i].astype(np.int64) for i in range(n)]
L = len(raw); raw += [bmax, brand]; RAW = np.column_stack(raw).astype(np.int64); del raw; gc.collect()
print(f"matrix {RAW.shape} (L={L})  [{time.time()-t0:.0f}s]", flush=True)
def membership(Pm):
    A = cp.asarray(RAW) % Pm; rows, ncols = A.shape; r = 0
    for c in range(L):
        sub = A[r:, c]; nz = cp.nonzero(sub)[0]
        if nz.size == 0: continue
        piv = r + int(nz[0].item())
        if piv != r:
            tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
        A[r] = (A[r] * pow(int(A[r, c].item()), Pm-2, Pm)) % Pm
        if r+1 < rows:
            f = A[r+1:, c]; A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= Pm
        r += 1
        if r == rows: break
    rank = r
    mi = bool(cp.all(A[rank:, L] % Pm == 0).item()); ri = bool(cp.all(A[rank:, L+1] % Pm == 0).item())
    del A; cp.get_default_memory_pool().free_all_blocks(); return rank, mi, ri
for Pm in (2147483647, 2147483629):
    rank, mi, ri = membership(Pm)
    print(f"[p={Pm}] rank={rank}/{m} | rand {'IN(VAC)' if ri else 'OUT(ok)'} | "
          f"max7 {'IN => SMALL weight-4 construction EXISTS!!' if mi else 'OUT of low-complexity weight-4'} | valid={rank<m and not ri}  [{time.time()-t0:.0f}s]", flush=True)
