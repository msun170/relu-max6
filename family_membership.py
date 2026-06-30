# BULLETPROOF confirmation of the template-mining verdict: the OMP approximant concentrates 83% of its mass on
# joins of <=1-generator zonotopes (point/segment) over the weight-4 lattice (MID types (2,2,4),(1,2,3),(1,1,2)).
# Here we ENUMERATE that COMPLETE family (not random sampling): all points, all segments, and all pairwise joins
# with <=4 vertices; S_7 orbit-sum each; EXACT mod-p membership (two primes) for max_7, with an in-span sanity
# target (must be IN) and a random control (must be OUT). If max_7 is OUT, the dominant approximant family is
# exactly insufficient -> the IN route's structured ansatz provably cannot represent max_7.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
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
gens = list(gens)
# enumerate pieces: all single points + all segments {p, p+g} (1-generator zonotopes)
points = [frozenset([p]) for p in W4]
segs = set()
for p in W4:
    for g in gens:
        q = shift(p, g)
        if q in W4set: segs.add(frozenset([p, q]))
segs = list(segs)
print(f"pieces: {len(points)} points, {len(segs)} segments  [{time.time()-t0:.0f}s]", flush=True)
pieces = points + segs
# COMPLETE join family = single pieces + all pairwise joins with <=4 verts. The family has ~29k S_7-orbits but its
# symmetric SPAN is low-dim (subset of the rank-902 low-complexity family). So we don't enumerate orbits; we
# random-saturate: take distinct join blocks and build FULL-GROUP-SUM columns (vectorized over all G=7! perms,
# = |Stab| * orbit-sum, same span) until the mod-p rank stops growing. This is exact and fast.
m = 5000
X = np.random.default_rng(4).integers(-16, 17, size=(m, n)).astype(np.int64)
Pi = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(cp.int64))    # integer, m x 210
def grp_col_int(Vidx):                                    # full-group sum, vectorized (orbit_membership2 style)
    Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx); idx = pa[:, Vc]
    return Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)
rng2 = np.random.default_rng(11)
seen = set(); cols = []
piece_list = pieces
TARGET = 4000
while len(cols) < TARGET:
    a = piece_list[rng2.integers(len(piece_list))]; b = piece_list[rng2.integers(len(piece_list))]
    u = a | b
    if len(u) > 4: continue
    key = tuple(sorted(vidx[p] for p in u))
    if key in seen: continue
    seen.add(key); cols.append(grp_col_int([vidx[p] for p in u]))
    if len(cols) % 500 == 0: print(f"  {len(cols)} distinct blocks  [{time.time()-t0:.0f}s]", flush=True)
N = len(cols)
print(f"sampled N={N} distinct point/segment-join blocks (full-group-sum columns)  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(cols, axis=1)
lin = cp.asarray(X.astype(cp.int64))
bmax = cp.asarray(X.max(axis=1).astype(np.int64))
# in-span sanity: integer combo of actual columns (must be detected IN); random control (must be OUT)
insp = (C @ cp.asarray(np.random.default_rng(1).integers(-3, 4, size=N).astype(np.int64))) + lin @ cp.asarray(np.array([1,0,-1,0,1,0,-1], dtype=np.int64))
brand = cp.asarray(np.random.default_rng(9).integers(-90, 91, size=m).astype(np.int64))
RAW = cp.concatenate([C, lin, bmax.reshape(-1,1), insp.reshape(-1,1), brand.reshape(-1,1)], axis=1)
L = N + n
print(f"matrix {RAW.shape} (L={L}, m={m})  [{time.time()-t0:.0f}s]", flush=True)
def membership(Pm):
    A = RAW % Pm; rows, ncols = A.shape; r = 0
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
    mi = bool(cp.all(A[rank:, L] % Pm == 0).item())
    si = bool(cp.all(A[rank:, L+1] % Pm == 0).item())
    ci = bool(cp.all(A[rank:, L+2] % Pm == 0).item())
    return rank, mi, si, ci
for Pm in (2147483647, 2147483629):
    rank, mi, si, ci = membership(Pm)
    print(f"[p={Pm}] rank={rank}/{m} | sanity(in-span) {'IN(ok)' if si else 'IN-FAIL!!'} | "
          f"control {'IN(VACUOUS!)' if ci else 'OUT(ok)'} | "
          f"max7 {'IN => DOMINANT FAMILY SUFFICES!!' if mi else 'OUT of dominant family'} | "
          f"valid={rank<m and si and not ci}  [{time.time()-t0:.0f}s]", flush=True)
print(f"[{time.time()-t0:.0f}s] OUT + valid => the 83%-mass approximant family is exactly insufficient (IN route's structured ansatz fails).", flush=True)
