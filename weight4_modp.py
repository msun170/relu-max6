# Is max_7 IN a STRUCTURED weight-4 family (small/low-complexity joins)?  Exact mod-p membership test (matmul-only
# elimination, no cusolver). Finds IN if it exists at low weight-4 complexity (cannot certify OUT -- family is a
# subfamily of complete weight-4). Structured = zonotopes with <=2 generators (segments/parallelograms/hexagons),
# at the weight-4 lattice, plus their joins. This is the analog of the low-complexity blocks that built max_6.
import sys, itertools, time, gc
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; W = 4; t0 = time.time()
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
# zonotopes with <=2 generators
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
zonos = list(zonos)
print(f"{len(gens)} gens, {len(zonos)} small zonotopes  [{time.time()-t0:.0f}s]", flush=True)
zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
print(f"{len(zorb)} zonotope orbits  [{time.time()-t0:.0f}s]", flush=True)
blocks = set(zonos)
for a in zreps:
    for b_ in zonos:
        u = a | b_
        if len(u) <= 12: blocks.add(u)     # tight cap to keep N feasible
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"N={N} structured weight-4 join orbits  [{time.time()-t0:.0f}s]", flush=True)
if N > 28000:
    print(f"N too large ({N}) for the mod-p matrix; would need tighter cap. Stopping.", flush=True); sys.exit()

m = N + 1500
X = np.random.default_rng(202).integers(-18, 19, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64)
brand = np.random.default_rng(9).integers(-90, 91, size=m).astype(np.int64)
raw = [core.orbit_column(orb[k], X).astype(np.int64) for k in okeys] + [X[:, i].astype(np.int64) for i in range(n)]
L = len(raw); raw += [bmax, brand]
RAW = np.column_stack(raw).astype(np.int64); del raw; gc.collect()
print(f"matrix {RAW.shape} (L={L} design cols)  [{time.time()-t0:.0f}s]", flush=True)

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
        if r % 3000 == 0: print(f"  [p={P}] rank {r}  [{time.time()-t0:.0f}s]", flush=True)
    rank = r
    max_in = bool(cp.all(A[rank:, L] % P == 0).item())
    rand_in = bool(cp.all(A[rank:, L+1] % P == 0).item())
    del A; cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    return rank, max_in, rand_in

for P in (2147483647, 2147483629):
    rank, max_in, rand_in = membership(P)
    valid = rank < m and not rand_in
    print(f"[p={P}] rank={rank}/{m} | random {'IN(VACUOUS)' if rand_in else 'OUT(ok)'} | "
          f"max7 {'IN => 2 LAYERS (weight-4)!!' if max_in else 'OUT of this structured family'} | valid={valid}  [{time.time()-t0:.0f}s]", flush=True)
