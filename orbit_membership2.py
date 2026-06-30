# EXP 3 (fast): random-saturate the small low-complexity weight-4 join family (it's small, ~hundreds), build
# integer ORBIT-SUMMED columns on GPU, EXACT mod-p membership for max_7 + random control. IN => small clean
# weight-4 construction exists. OUT => none in this complexity class.
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
gens = list(gens); ng = len(gens)
rng = np.random.default_rng(4)
m = 4000
X = rng.integers(-16, 17, size=(m, n)).astype(np.int64)
Pi = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(cp.int64))   # integer, m x 210
def zono():
    p = W4[rng.integers(NP)]; verts = {p}; added = 0; tries = 0
    while added < 3 and len(verts) < 8 and tries < 40:
        tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok and nv - verts: verts |= nv; added += 1
    return verts
def orbit_col_int(Vidx):
    Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx); idx = pa[:, Vc]
    return Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)   # integer
# random-saturate distinct blocks
seen = set(); cols = []; fails = 0
while fails < 8000 and len(cols) < 2500:
    v1 = zono(); v2 = zono(); u = v1 | v2; idx = tuple(sorted(vidx[p] for p in u)[:10])
    if idx in seen: fails += 1; continue
    seen.add(idx); fails = 0; cols.append(orbit_col_int(idx))
    if len(cols) % 200 == 0: print(f"  {len(cols)} distinct blocks  [{time.time()-t0:.0f}s]", flush=True)
N = len(cols)
print(f"saturated at N={N} distinct low-complexity orbit blocks  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(cols, axis=1)                          # m x N integer
lin = cp.asarray(X.astype(cp.int64))
bmax = cp.asarray(X.max(axis=1).astype(np.int64)); brand = cp.asarray(np.random.default_rng(9).integers(-90,91,size=m).astype(np.int64))
RAW = cp.concatenate([C, lin, bmax.reshape(-1,1), brand.reshape(-1,1)], axis=1)
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
    mi = bool(cp.all(A[rank:, L] % Pm == 0).item()); ri = bool(cp.all(A[rank:, L+1] % Pm == 0).item())
    return rank, mi, ri
for Pm in (2147483647, 2147483629):
    rank, mi, ri = membership(Pm)
    print(f"[p={Pm}] rank={rank}/{m} | rand {'IN(VAC)' if ri else 'OUT(ok)'} | "
          f"max7 {'IN => SMALL weight-4 CONSTRUCTION EXISTS!!' if mi else 'OUT of low-complexity weight-4'} | valid={rank<m and not ri}  [{time.time()-t0:.0f}s]", flush=True)
