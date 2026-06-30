# DYADIC CONSTRUCTION SEARCH for max_7 (principled, motivated by BBHSY + the dyadic-necessity refinement).
# AHM mod-2 forbids odd-denominator/integer max_7 in 2 layers, so any 2-layer max_7 is EVEN-DENOMINATOR / dyadic.
# The dyadic regime = a 2-refined lattice; the weight-4 lattice is that refinement of weight-2. BBHSY's MAX_5
# construction uses blocks = max(small zonotope, small zonotope) where each zonotope is a Minkowski sum of a few
# SHORT segments (e.g. max(2x5, x1+x2) and max(x1,x3)+max(x1,x4)). So we search the COMPLETE family of joins of two
# zonotopes with <= 2 SHORT generators each on the weight-4 lattice (this reaches 5-8 vertex parallelogram-joins
# that the <=4-vertex test misses) -- orbit-summed, EXACT mod-p membership (two primes) + symmetric control.
# IN => a dyadic construction (extract via extract_certificate.py + chamber verifier). OUT => this BBHSY-complexity
# dyadic family at weight-4 is insufficient; the construction needs higher weight (weight-8) or more generators.
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
n = 7; W = 4; t0 = time.time(); P1 = 2147483647; P2 = 2147483629
W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; W4set = set(W4); vidx = {p: i for i, p in enumerate(W4)}
NP = len(W4)
perms = list(itertools.permutations(range(n))); G = len(perms)
pa = cp.asarray(np.array([[vidx[tuple(p[g[k]] for k in range(n))] for p in W4] for g in perms], dtype=np.int32))
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
# SHORT primitive generators (L1 <= 4): roots e_i-e_j (L1 2), and 2e_i-e_j-e_k, e_i+e_j-e_k-e_l (L1 4) -- the
# directions appearing in BBHSY-style blocks.
L1CAP = int(os.environ.get("L1CAP", "4"))
gens = sorted({prim(tuple(q[k]-p[k] for k in range(n))) for p in W4 for q in W4 if p != q
               if sum(abs(q[k]-p[k]) for k in range(n)) <= L1CAP})
print(f"weight-4 lattice {NP} pts; {len(gens)} short generators (L1<={L1CAP})  [{time.time()-t0:.0f}s]", flush=True)
# enumerate zonotopes with <=2 short generators
zon = set()
for p in W4: zon.add(frozenset([p]))
for p in W4:
    for g in gens:
        w = shift(p, g)
        if w in W4set: zon.add(frozenset([p, w]))
for p in W4:
    for i in range(len(gens)):
        a = shift(p, gens[i])
        if a not in W4set: continue
        for j in range(i+1, len(gens)):
            b = shift(p, gens[j]); c = shift(a, gens[j])
            if b in W4set and c in W4set: zon.add(frozenset([p, a, b, c]))
zon = list(zon)
print(f"{len(zon)} <=2-gen zonotopes  [{time.time()-t0:.0f}s]", flush=True)
# joins of two zonotopes, <= CAP vertices (reps x all, to stay complete up to S_7)
CAP = int(os.environ.get("VCAP", "8"))
zorb = core.orbits_of(zon, n); zreps = [zorb[k][0] for k in zorb]
blocks = set(zon)
for a in zreps:
    for b in zon:
        u = a | b
        if len(u) <= CAP: blocks.add(u)
orb = core.orbits_of(list(blocks), n); reps = list(orb)
Nrep = len(reps)
print(f"COMPLETE BBHSY-complexity dyadic family: {len(blocks)} blocks -> {Nrep} S_7 orbits (CAP={CAP})  [{time.time()-t0:.0f}s]", flush=True)
m = int(os.environ.get("M", "9000"))
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
Pi = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(cp.int64))
bmax = cp.asarray(X.max(axis=1).astype(np.int64))
Xs = np.sort(X, axis=1); bsym = cp.asarray((Xs * np.random.default_rng(5).integers(-9, 10, size=n)).sum(axis=1).astype(np.int64))
lin = cp.asarray(X.astype(cp.int64))
def orbit_col(members):
    acc = cp.zeros(m, dtype=cp.int64)
    for vs in members:
        Vc = cp.asarray([vidx[p] for p in vs], dtype=cp.int32); k = len(vs); idx = pa[:, Vc]
        acc += Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1) // 1  # full-group sum
    return acc
# faster: full-group-sum per orbit rep (|Stab|*orbit-sum, same span)
def grp_col(rep):
    Vc = cp.asarray([vidx[p] for p in rep], dtype=cp.int32); k = len(rep); idx = pa[:, Vc]
    return Pi[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)
C = cp.stack([grp_col(orb[k][0]) for k in reps], axis=1)
RAW = cp.concatenate([C, lin, bmax.reshape(-1, 1), bsym.reshape(-1, 1)], axis=1)
L = Nrep + n
print(f"matrix {RAW.shape} (m={m}, N={Nrep})  [{time.time()-t0:.0f}s]", flush=True)
res = {}
for Pm in (P1, P2):
    A = RAW % Pm; rows = A.shape[0]; r = 0
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
    mi = bool(cp.all(A[rank:, L] % Pm == 0).item()); si = bool(cp.all(A[rank:, L+1] % Pm == 0).item())
    res[Pm] = (rank, mi, si)
(r1, mi1, si1) = res[P1]; (r2, mi2, si2) = res[P2]
verdict = "IN => DYADIC CONSTRUCTION EXISTS!! (extract + verify)" if (mi1 and mi2) else "OUT of this complete dyadic family"
valid = (r1 < m) and (not si1)
print(f"[result] rank={r1}/{m} | max_7 {verdict} | sym-control {'IN(VAC!)' if si1 else 'OUT(ok)'} | valid={valid}  [{time.time()-t0:.0f}s]", flush=True)
print(f"[{time.time()-t0:.0f}s] done. OUT => push to weight-8 dyadic or >2 generators; IN => we have the construction.", flush=True)
