# ROUTE (a): TARGETED SPARSE search for a max_7 construction, motivated by route A's dyadic 4:2:1 template
# (max_5, max_6 have EXACT 3+3 = 6-block minimal decompositions). Method: true ORTHOGONAL MATCHING PURSUIT to
# select a sparse ordered support from a weight-{3,4,5} pool (OMP favors sparse solutions far better than a
# single-shot importance ranking), then the DECISION is an EXACT mod-p membership test (two primes) on the
# first-K selected blocks for increasing K. IN at small K => a sparse construction candidate -> extract + verify.
# Residual only ORDERS candidates; it never decides (no residual-chasing). If residual plateaus above 0 AND every
# exact test is OUT, no sparse solution lives in this pool.
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

# weight-4-heavy pool of small (6-8 vertex) joins-of-two-zonotopes (matching route A's block sizes)
WT = {3: int(os.environ.get("K3", "800")), 4: int(os.environ.get("K4", "2400")), 5: int(os.environ.get("K5", "800"))}
m = int(os.environ.get("M", "4000"))
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.int64))
Xs = np.sort(X, axis=1); bsym = cp.asarray((Xs * np.random.default_rng(5).integers(-9, 10, size=n)).sum(axis=1).astype(np.int64))
lin = cp.asarray(X.astype(cp.int64))
allcols = []; allblocks = []
for W, KPER in WT.items():
    L, Ls, vi, gens = lattice(W); NP = len(L)
    pa = cp.asarray(np.array([[vi[tuple(p[g[k]] for k in range(n))] for p in L] for g in perms], dtype=np.int32))
    Pi = cp.asarray((X @ np.array(L, dtype=np.int64).T).astype(cp.int64))
    rng = np.random.default_rng(7 + W)
    def zono(maxgen):
        p = L[rng.integers(NP)]; verts = {p}; used = 0; tries = 0
        while used < maxgen and len(verts) < 6 and tries < 40:
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
    seen = set(); cnt = 0; guard = 0
    while cnt < KPER and guard < KPER*50:
        guard += 1
        u = zono(rng.integers(1, 3)) | zono(rng.integers(1, 3))
        if not (4 <= len(u) <= 8): continue          # small blocks like route A (6-8v)
        key = (W, tuple(sorted(vi[p] for p in u)))
        if key in seen: continue
        seen.add(key); allcols.append(grp_col(u)); allblocks.append((W, sorted(u))); cnt += 1
    print(f"weight-{W} done ({cnt} blocks)  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(allcols, axis=1).astype(cp.float64); Npool = C.shape[1]
print(f"pool: {Npool} small orbit blocks over weights {list(WT)}; m={m}  [{time.time()-t0:.0f}s]", flush=True)

# ---- OMP: always keep the n linear columns in the model; select blocks by residual correlation ----
Lin = lin.astype(cp.float64); b = bmax.astype(cp.float64)
colnorm = cp.linalg.norm(C, axis=0); colnorm = cp.where(colnorm == 0, 1.0, colnorm)
def refit(cols):                                     # least-squares fit over [Lin | C[:,cols]], return residual
    D = Lin if not cols else cp.concatenate([Lin, C[:, cp.asarray(cols)]], axis=1)
    sol, *_ = cp.linalg.lstsq(D, b, rcond=None)
    return b - D @ sol
resid = refit([]); sel = []
CAP = int(os.environ.get("CAP", "60")); TOL = 1e-9
rel0 = float(cp.linalg.norm(b))
print(f"OMP start: linear-only residual = {float(cp.linalg.norm(resid))/rel0:.5f}  [{time.time()-t0:.0f}s]", flush=True)
curve = []
while len(sel) < CAP:
    corr = cp.abs(C.T @ resid) / colnorm
    if sel: corr[cp.asarray(sel)] = -1
    j = int(cp.argmax(corr).item()); sel.append(j)
    resid = refit(sel); rn = float(cp.linalg.norm(resid))/rel0; curve.append(rn)
    if len(sel) <= 12 or len(sel) % 5 == 0:
        print(f"  OMP |S|={len(sel):>2}: rel-residual = {rn:.6f}  [{time.time()-t0:.0f}s]", flush=True)
    if rn < TOL: print(f"  residual ~0 at |S|={len(sel)} -> candidate!", flush=True); break

# ---- EXACT mod-p membership decision on first-K OMP blocks ----
Cint = cp.stack(allcols, axis=1)
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
print(f"\nEXACT membership on first-K OMP blocks (two primes):  [{time.time()-t0:.0f}s]", flush=True)
found = None
for K in [6, 8, 10, 12, 16, 20, 30, 40, 50, 60]:
    if K > len(sel): break
    idx = sel[:K]
    r1, mi1, si1 = membership(idx, P1); _, mi2, _ = membership(idx, P2)
    IN = mi1 and mi2; valid = (r1 < m) and (not si1)
    print(f"  K={K:>2}: rank={r1} | max_7 {'IN (both primes)!!' if IN else 'OUT'} | control {'IN(vac)' if si1 else 'OUT(ok)'} | valid={valid}", flush=True)
    if IN and valid: found = (K, idx); break
if found:
    print(f"\n*** SPARSE construction candidate: max_7 IN on {found[0]} OMP-selected blocks -> extract rational + chamber verify. ***", flush=True)
else:
    tail = curve[-1] if curve else 1.0
    print(f"\nOUT at every K up to {min(60,len(sel))}. Final OMP rel-residual = {tail:.6f}.", flush=True)
    print(f"  Read: {'residual plateaued above 0 -> max_7 not even in the full pool span (no sparse soln here)' if tail>1e-4 else 'residual near 0 but no EXACT sparse hit -> approximation only'}. Not an OUT proof (incomplete pool).", flush=True)
print(f"[{time.time()-t0:.0f}s] done.", flush=True)
