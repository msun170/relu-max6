# ROUTE B: DISCIPLINED construction search for max_7 in V_2 (weights 3,4,5 pool).
#
# Method (no residual-chasing): (1) build a rich multi-weight P2 pool; (2) float least-squares approximant of
# max_7 over the pool -> residual floor + per-block importance; (3) PROPOSE a finite support = top-K important
# orbit blocks; (4) EXACT mod-p membership (two primes) of max_7 in span(support + linear) for several K; (5) if
# IN -> extract rational + chamber-verify (a real construction); if OUT for all K -> report floors honestly and
# stop (this pool/support insufficient; NOT an OUT proof). Disciplined = approximation only proposes support;
# the decision is always the exact test.
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

WT = [int(x) for x in os.environ.get("WEIGHTS", "3,4,5").split(",")]
KPER = int(os.environ.get("KPER", "1200"))
m = int(os.environ.get("M", "9000"))
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
    print(f"weight-{W} done ({KPER} blocks)  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(allcols, axis=1).astype(cp.float64); Npool = C.shape[1]
print(f"pool: {Npool} orbit blocks over weights {WT}; m={m}  [{time.time()-t0:.0f}s]", flush=True)

# (2) float least-squares over [pool | linear] -> residual floor + importance
D = cp.concatenate([C, lin.astype(cp.float64)], axis=1)
b = bmax.astype(cp.float64)
sol, *_ = cp.linalg.lstsq(D, b, rcond=None)
res = b - D @ sol
floor = float(cp.linalg.norm(res)/cp.linalg.norm(b))
colnorm = cp.linalg.norm(C, axis=0)
importance = cp.abs(sol[:Npool]) * colnorm           # per-block importance
order = cp.argsort(-importance).get()
print(f"least-squares floor over full pool = {floor:.5f}  [{time.time()-t0:.0f}s]", flush=True)

# (4) EXACT mod-p membership on top-K supports
Cint = cp.stack(allcols, axis=1)                      # integer columns
def membership(cols_idx, Pm):
    sub = Cint[:, cp.asarray(cols_idx)]
    A = cp.concatenate([sub, lin, bmax.reshape(-1, 1), bsym.reshape(-1, 1)], axis=1) % Pm
    rows, ncols = A.shape; Ldes = len(cols_idx) + n; r = 0
    for c in range(Ldes):
        s = A[r:, c]; nz = cp.nonzero(s)[0]
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
    mi = bool(cp.all(A[rank:, Ldes] % Pm == 0).item()); si = bool(cp.all(A[rank:, Ldes+1] % Pm == 0).item())
    return rank, mi, si
Ks = [int(x) for x in os.environ.get("KS", "40,80,150,250,400").split(",")]
found = None
for K in Ks:
    if K > Npool: continue
    idx = order[:K].tolist()
    r1, mi1, si1 = membership(idx, P1); r2, mi2, si2 = membership(idx, P2)
    IN = mi1 and mi2; valid = (r1 < m) and (not si1)
    print(f"  top-{K:>3} support: rank={r1}/{m} | max_7 {'IN (both primes)!!' if IN else 'OUT'} | control {'IN(vac)' if si1 else 'OUT(ok)'} | valid={valid}  [{time.time()-t0:.0f}s]", flush=True)
    if IN and valid: found = (K, idx); break
if found:
    print(f"\n*** max_7 IN on top-{found[0]} support -> extract rational + chamber verify (a real construction). ***", flush=True)
else:
    print(f"\nOUT on all proposed supports {Ks}. Floor {floor:.5f}. This pool/support insufficient (NOT an OUT proof).", flush=True)
    print(f"  next disciplined step if pursued: richer pool (more zono geometry / higher weight) or softmax-anneal support.", flush=True)
print(f"[{time.time()-t0:.0f}s] done.", flush=True)
