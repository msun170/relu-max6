# MIN-SUPPORT / MIN-DENOMINATOR weight-2 dual certificate (bounded experiment, decision rule C).
#
# The certificate set is {lambda over points : lambda.h_Q=0 all blocks, lambda.x_i=0 all linear,
# lambda.max=1}. lambda lives in null(M^T) with M=[block cols | linear cols]; its SUPPORT is a set of
# points whose block/linear rows are linearly dependent (that's the annihilation) while max stays
# independent. GENERIC lower bound: min-support = rank(M)+1 (girth of the row matroid). We (1) compute
# rank(M) exactly (mod p) => the generic min-support, then (2) GREEDY-REMOVE points from a working
# certificate to find the smallest ACTUAL irreducible witnessing set (this is where STRUCTURED / near-wall
# points could beat rank+1), (3) reconstruct the exact rational certificate on that minimal set, report
# its integer-normalized coefficients / max magnitude / denominators, and (4) analyze geometry: how many
# distinct braid chambers the support touches, and wall-proximity. Decision: small+structured => possible
# invariant; large+diffuse => stop the certificate route, pivot to A/B.
import sys, itertools, time, os
from math import gcd
from functools import reduce
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

n = int(os.environ.get("N", "7")); Wt = int(os.environ.get("WT", "2")); t0 = time.time()
P1 = 2147483647

def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)

def build_complete(n, Wt, cap):
    L = [c for c in itertools.product(range(Wt+1), repeat=n) if sum(c) == Wt]; Ls = set(L)
    vi = {p: i for i, p in enumerate(L)}
    gens = sorted({prim(tuple(q[k]-p[k] for k in range(n))) for p in L for q in L if p != q})
    def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
    def ssz(p, Gs):
        verts = set()
        for s in range(len(Gs)+1):
            for S in itertools.combinations(Gs, s):
                q = p
                for g in S: q = shift(q, g)
                if q not in Ls: return None
                verts.add(q)
        return frozenset(verts)
    zon = set()
    for p in L:
        for g in gens:
            v = ssz(p, [g])
            if v and len(v) >= 2: zon.add(v)
    for p in L:
        for i in range(len(gens)):
            if shift(p, gens[i]) not in Ls: continue
            for j in range(i+1, len(gens)):
                v = ssz(p, [gens[i], gens[j]])
                if v and len(v) >= 3: zon.add(v)
    for zz in [zz for zz in list(zon) if len(zz) >= 4]:
        for g in gens:
            verts = set(zz); ok = True
            for u in zz:
                wv = shift(u, g)
                if wv not in Ls: ok = False; break
                verts.add(wv)
            if ok and len(verts) > len(zz): zon.add(frozenset(verts))
    zon = list(zon); zorb = core.orbits_of(zon, n); zreps = [zorb[k][0] for k in zorb]
    bl = set(zon)
    for a in zreps:
        for b in zon:
            u = a | b
            if len(u) <= cap: bl.add(u)
    return core.orbits_of(list(bl), n)

orb = build_complete(n, Wt, int(os.environ.get("CAP", "12"))); okeys = list(orb); Nb = len(okeys)
print(f"n={n} weight-{Wt}: {Nb} complete orbit blocks  [{time.time()-t0:.0f}s]", flush=True)

m = int(os.environ.get("M", "400"))
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64)
Ccols = [core.orbit_column(orb[k], X).astype(np.int64) for k in okeys]
Lcols = [X[:, i].astype(np.int64) for i in range(n)]
M = np.column_stack(Ccols + Lcols).astype(np.int64)     # m x (Nb+n) design (blocks + linear)
ncol = M.shape[1]
print(f"M = {M.shape} (blocks+linear); m={m} points  [{time.time()-t0:.0f}s]", flush=True)

def rank_modp(A, P=P1):
    A = (A.astype(np.int64) % P).copy(); rows, cols = A.shape; r = 0
    for c in range(cols):
        col = A[r:, c]; nz = np.nonzero(col)[0]
        if nz.size == 0: continue
        piv = r + int(nz[0])
        if piv != r: A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), P-2, P); A[r] = (A[r] * inv) % P
        if r+1 < rows:
            f = A[r+1:, c][:, None]; A[r+1:] = (A[r+1:] - f * A[r]) % P
        r += 1
        if r == rows: break
    return r

def out_witnessed(rowsidx):    # True iff max is NOT in colspan of M restricted to these rows
    sub = M[rowsidx]; base = rank_modp(sub)
    aug = rank_modp(np.column_stack([sub, bmax[rowsidx]]))
    return aug > base, base

rkM = rank_modp(M)
print(f"rank(M) = {rkM} of {ncol} columns  =>  GENERIC min-support = rank(M)+1 = {rkM+1}", flush=True)
allidx = np.arange(m)
ow, _ = out_witnessed(allidx)
print(f"OUT witnessed on all {m} points: {ow}  [{time.time()-t0:.0f}s]", flush=True)
if not ow:
    print("  max IN on this point set (unexpected) -- abort.", flush=True); raise SystemExit

# GREEDY REMOVAL: shrink to an irreducible witnessing set. Try several random orders; keep the smallest.
best = None
for seed in range(int(os.environ.get("PASSES", "5"))):
    R = list(range(m)); order = list(range(m))
    np.random.default_rng(1000+seed).shuffle(order)
    for r in order:
        if r not in R: continue
        cand = [x for x in R if x != r]
        ok, _ = out_witnessed(np.array(cand))
        if ok: R = cand           # r not essential -> drop it
    ok, base = out_witnessed(np.array(R))
    print(f"  pass {seed}: irreducible support = {len(R)} points (OUT={ok}, rank on them={base})  [{time.time()-t0:.0f}s]", flush=True)
    if best is None or len(R) < len(best): best = R
R = best; s = len(R)
print(f"\nMIN irreducible support found: {s} points  (generic bound rank+1 = {rkM+1})", flush=True)

# exact rational certificate on the minimal support
sub = M[R]
Brows = [[Fr(int(sub[i, j])) for i in range(s)] for j in range(ncol)] + [[Fr(int(bmax[R[i]])) for i in range(s)]]
rs = [Fr(0)] * ncol + [Fr(1)]
lam = core.exact_solve(Brows, rs)
if lam is None:
    print("  (exact solve found no certificate on the minimal set -- numerical edge; report support only)", flush=True)
else:
    lam = list(lam)
    dens = [lam[i].denominator for i in range(s) if lam[i] != 0]
    lc = reduce(lambda a, b: a*b//gcd(a, b), dens, 1)          # lcm of denominators
    ints = [int(lam[i] * lc) for i in range(s)]
    g = reduce(gcd, [abs(v) for v in ints if v != 0]) or 1
    ints = [v // g for v in ints]
    nz = [i for i in range(s) if ints[i] != 0]
    print(f"  exact certificate on min support: {len(nz)} nonzero coeffs; integer-normalized max|coeff| = {max(abs(v) for v in ints)}; #digits = {len(str(max(abs(v) for v in ints)))}", flush=True)
    # geometry: how many distinct braid chambers (sort-orders) does the support touch? wall proximity?
    chambers = set(); mingap = 99
    for i in R:
        xv = X[i]; sd = tuple(np.argsort(-xv))               # descending sort order = braid chamber id
        chambers.add(sd)
        sv = np.sort(xv); mingap = min(mingap, int(np.min(np.diff(sv))))   # 0 => on a wall (a tie)
    import math as _math
    print(f"  geometry: support touches {len(chambers)} distinct braid chambers (of {min(m, _math.factorial(n))} possible); min coord-gap among support pts = {mingap} (0 = on a braid wall)", flush=True)
    print(f"  INTERPRETABILITY: {'SMALL / possibly structured' if s <= 2*(rkM+1)//3 or max(abs(v) for v in ints) < 1000 else 'LARGE & diffuse (basis artifact, not a named invariant)'}", flush=True)

print(f"\n[{time.time()-t0:.0f}s] DECISION: min-support ~= rank(M)+1 = {rkM+1} => certificate is {'compact' if s < 20 else 'inherently large/diffuse'}; "
      f"{'inspect geometry' if s < 20 else 'stop scaling certificates to weight-3; pivot to route A (decomposition polyhedron) or B (construction).'}", flush=True)
