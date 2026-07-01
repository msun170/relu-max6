# POSITIVE construction hunt for max_7 in V_2 (per the strategy: don't enumerate; find ONE lucky structured support).
# Test EXACT S_7-orbit-sum membership of max_7 against a DIVERSE, HIGH-WEIGHT, INCOMPLETE pool (weights 4,5,6 -- never
# tested), with residual-guided column generation. Incomplete + FEASIBLE => extract candidate + verify exactly (that
# is a proof); incomplete + infeasible => just means this pool is insufficient (NOT an OUT proof). If feasible we
# extract the rational certificate (support -> exact rational solve -> fresh-point verify -> P2 validity).
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
n = 7; t0 = time.time(); P1 = 2147483647; P2 = 2147483629
perms = list(itertools.permutations(range(n))); G = len(perms)
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

def lattice(W):
    L = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
    return L, set(L), {p: i for i, p in enumerate(L)}, sorted({prim(tuple(q[k]-p[k] for k in range(n))) for p in L for q in L if p != q})

# per-weight machinery
WT = [int(x) for x in os.environ.get("WEIGHTS", "4,5,6").split(",")]
KPER = int(os.environ.get("KPER", "1600"))
m = int(os.environ.get("M", "8000"))
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.int64))
Xs = np.sort(X, axis=1); bsym = cp.asarray((Xs * np.random.default_rng(5).integers(-9, 10, size=n)).sum(axis=1).astype(np.int64))
lin = cp.asarray(X.astype(cp.int64))
allcols = []; allblocks = []
for W in WT:
    L, Ls, vi, gens = lattice(W); NP = len(L)
    pa = cp.asarray(np.array([[vi[tuple(p[g[k]] for k in range(n))] for p in L] for g in perms], dtype=np.int32))
    Pi = cp.asarray((X @ np.array(L, dtype=np.int64).T).astype(cp.int64))
    rng = np.random.default_rng(7 + W)
    def zono():
        p = L[rng.integers(NP)]; verts = {p}; used = 0; tries = 0
        while used < 3 and len(verts) < 8 and tries < 40:
            tries += 1; g = gens[rng.integers(len(gens))]; nv = set(); ok = True
            for u in verts:
                w = shift(u, g)
                if w not in Ls: ok = False; break
                nv.add(w)
            if ok and nv - verts: verts |= nv; used += 1
        return verts
    def grp_col(vs):
        Vc = cp.asarray([vi[p] for p in vs], dtype=cp.int32); k = len(vs); idx = pa[:, Vc]
        return Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)
    seen = set(); cnt = 0
    while cnt < KPER:
        u = zono() | zono()
        if len(u) > 12: continue
        key = (W, tuple(sorted(vi[p] for p in u)))
        if key in seen: continue
        seen.add(key); allcols.append(grp_col(u)); allblocks.append((W, sorted(u))); cnt += 1
        if cnt % 400 == 0: print(f"  weight-{W}: {cnt} blocks  [{time.time()-t0:.0f}s]", flush=True)
    print(f"weight-{W} done ({KPER} blocks)  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(allcols, axis=1); N = C.shape[1]
print(f"pool: {N} orbit-sum blocks over weights {WT}; m={m}  [{time.time()-t0:.0f}s]", flush=True)
RAW = cp.concatenate([C, lin, bmax.reshape(-1, 1), bsym.reshape(-1, 1)], axis=1)
L_ = N + n
def membership(Pm):
    A = RAW % Pm; rows = A.shape[0]; r = 0
    for c in range(L_):
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
    mi = bool(cp.all(A[rank:, L_] % Pm == 0).item()); si = bool(cp.all(A[rank:, L_+1] % Pm == 0).item())
    return rank, mi, si
r1, mi1, si1 = membership(P1); r2, mi2, si2 = membership(P2)
IN = mi1 and mi2
print(f"[p1] rank={r1}/{m} | max_7 {'IN => CANDIDATE CONSTRUCTION!!' if mi1 else 'OUT of this pool'} | sym-control {'IN(vac)' if si1 else 'OUT(ok)'}", flush=True)
print(f"[p2] max_7 {'IN' if mi2 else 'OUT'} | both-prime IN = {IN} | valid(rank<m & control OUT) = {r1 < m and not si1}  [{time.time()-t0:.0f}s]", flush=True)
if IN:
    print("  *** FEASIBLE on this pool -> extract + verify (support -> exact rational -> chamber). ***", flush=True)
else:
    # residual floor (how close): float proj of bmax onto pool span (guides column generation next round)
    Cf = C.astype(cp.float64); Qq, _ = cp.linalg.qr(cp.concatenate([Cf, lin.astype(cp.float64)], axis=1))
    b = bmax.astype(cp.float64); res = b - Qq @ (Qq.T @ b)
    print(f"  residual floor ||r||/||b|| = {float(cp.linalg.norm(res)/cp.linalg.norm(b)):.5f}  (guides next column-gen round)  [{time.time()-t0:.0f}s]", flush=True)
print(f"[{time.time()-t0:.0f}s] done.", flush=True)
