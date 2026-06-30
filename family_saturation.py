# SATURATION + VACUITY check for the point/segment-join OUT result. The N=4000 run gave rank=3299 (>> the ~902 I
# expected), so confirm: (1) does the rank PLATEAU as we add blocks (is 3299 the complete family span, or undersampled
# -> more blocks could bring max_7 IN)? (2) a proper SYMMETRIC control (random PL of sorted coords) must be OUT too
# (guards against the family being all of symmetric-PL on X). Build 8000 blocks at m=8000; report rank at 4000 and
# 8000 columns; exact mod-p membership for max_7 + symmetric control + in-span sanity.
import sys, itertools, time
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
gens = list(gens)
points = [frozenset([p]) for p in W4]
segs = set()
for p in W4:
    for g in gens:
        q = shift(p, g)
        if q in W4set: segs.add(frozenset([p, q]))
pieces = points + list(segs)
print(f"pieces: {len(points)} pts, {len(segs)} segs  [{time.time()-t0:.0f}s]", flush=True)
m = 8000
X = np.random.default_rng(4).integers(-16, 17, size=(m, n)).astype(np.int64)
Pi = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(cp.int64))
def grp_col_int(Vidx):
    Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx); idx = pa[:, Vc]
    return Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)
rng2 = np.random.default_rng(11)
seen = set(); cols = []; TARGET = 8000
while len(cols) < TARGET:
    a = pieces[rng2.integers(len(pieces))]; b = pieces[rng2.integers(len(pieces))]
    u = a | b
    if len(u) > 4: continue
    key = tuple(sorted(vidx[p] for p in u))
    if key in seen: continue
    seen.add(key); cols.append(grp_col_int([vidx[p] for p in u]))
    if len(cols) % 1000 == 0: print(f"  {len(cols)} blocks  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(cols, axis=1)
lin = cp.asarray(X.astype(cp.int64))
bmax = cp.asarray(X.max(axis=1).astype(np.int64))
# proper SYMMETRIC control: random PL of sorted coords  sum_k a_k * sort(x)_k  (integer)
Xs = np.sort(X, axis=1)            # ascending
acoef = np.random.default_rng(5).integers(-9, 10, size=n)
bsym = cp.asarray((Xs * acoef).sum(axis=1).astype(np.int64))
insp = (C @ cp.asarray(np.random.default_rng(1).integers(-3, 4, size=TARGET).astype(np.int64)))
P1 = 2147483647
def rank_of(M, ncols, Pm):
    A = (M[:, :ncols] % Pm).copy(); rows = A.shape[0]; r = 0
    for c in range(ncols):
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
    return r
Cl = cp.concatenate([C, lin], axis=1)
r4 = rank_of(Cl, 4000 + n, P1)
r8 = rank_of(Cl, TARGET + n, P1)
print(f"rank @4000 cols = {r4} | rank @8000 cols = {r8}  (plateau => saturated; climbing => undersampled)  [{time.time()-t0:.0f}s]", flush=True)
# full membership at p1
RAW = cp.concatenate([C, lin, bmax.reshape(-1,1), insp.reshape(-1,1), bsym.reshape(-1,1)], axis=1)
L = TARGET + n
A = RAW % P1; rows = A.shape[0]; r = 0
for c in range(L):
    sub = A[r:, c]; nz = cp.nonzero(sub)[0]
    if nz.size == 0: continue
    piv = r + int(nz[0].item())
    if piv != r:
        tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
    A[r] = (A[r] * pow(int(A[r, c].item()), P1-2, P1)) % P1
    if r+1 < rows:
        f = A[r+1:, c]; A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= P1
    r += 1
    if r == rows: break
rank = r
mi = bool(cp.all(A[rank:, L] % P1 == 0).item())
si = bool(cp.all(A[rank:, L+1] % P1 == 0).item())
sy = bool(cp.all(A[rank:, L+2] % P1 == 0).item())
print(f"[p={P1}] rank={rank}/{m} | sanity {'IN(ok)' if si else 'FAIL'} | sym-control {'IN(VACUOUS!)' if sy else 'OUT(ok)'} | "
      f"max7 {'IN!!' if mi else 'OUT'} | valid={rank<m and si and not sy}  [{time.time()-t0:.0f}s]", flush=True)
print(f"[{time.time()-t0:.0f}s] saturated(rank plateau) + sym-control OUT + max7 OUT => COMPLETE dominant family is exactly insufficient.", flush=True)
