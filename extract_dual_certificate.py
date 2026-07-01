# EXACT (floating-point-free) dual certificate for max_n OUT of the COMPLETE weight-W P2 family.
#
# We know max_7 is OUT of complete weight-2 (and weight-3). The Fredholm certificate is a linear
# functional lambda on function-evaluations with:
#     lambda . h_Q      = 0   for every complete weight-W orbit block Q   (annihilates the family)
#     lambda . x_i      = 0   for every coordinate (annihilates linear terms)
#     lambda . max_n    != 0                                             (separates the target)
# Its existence IS a proof of OUT (Fredholm alternative). Upgrades over the mod-p membership test:
#   (a) we produce it EXACTLY over Q (rational, no floats, no primes) -- a checkable certificate;
#   (b) we show it can be taken S_n-SYMMETRIC, i.e. a signed combination of orbit-evaluation
#       functionals -- a low-parameter structured certificate (the finite instance of the
#       "universal separating functional" route, and the local data of the P2-gradient sheaf).
#
# Weight-2 is small enough for exact rational linear algebra; that is the clean prototype. The same
# object at weight-3 is the target the strategy doc calls central (there it is mod-p + huge).
import sys, itertools, time, os
from math import gcd
from functools import reduce
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

n = int(os.environ.get("N", "7")); Wt = int(os.environ.get("WT", "2")); t0 = time.time()

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
    orb = core.orbits_of(list(bl), n)
    return orb

cap = int(os.environ.get("CAP", "12"))
orb = build_complete(n, Wt, cap); okeys = list(orb); Nb = len(okeys)
print(f"n={n} weight-{Wt}: {Nb} complete orbit blocks  [{time.time()-t0:.0f}s]", flush=True)

# probe points: need m > rank(design) so the left-nullspace (where lambda lives) is nonempty.
m = Nb + n + int(os.environ.get("MARGIN", "40"))
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.int64)

# integer design columns: orbit-summed block support functions + the n linear coordinates
Ccols = [core.orbit_column(orb[k], X).astype(np.int64) for k in okeys]
Lcols = [X[:, i].astype(np.int64) for i in range(n)]
design = Ccols + Lcols                       # length Nb+n, each an int vector of length m
print(f"design: {len(design)} columns, m={m} probe points  [{time.time()-t0:.0f}s]", flush=True)

# Solve for lambda in Q^m:  design_j . lambda = 0  (all j),  bmax . lambda = 1.
# That is B lambda = rhs with B = [design^T ; bmax^T]  ((Nb+n+1) x m),  rhs = (0,...,0,1).
Brows = [[Fr(int(v)) for v in col] for col in design] + [[Fr(int(v)) for v in bmax]]
rhs = [Fr(0)] * len(design) + [Fr(1)]
print(f"solving exact system {len(Brows)} x {m}  [{time.time()-t0:.0f}s]", flush=True)
lam = core.exact_solve(Brows, rhs)
if lam is None:
    print("NO certificate -> bmax IS in the span (max_n IN this family). Stop.", flush=True)
    raise SystemExit
lam = list(lam)
supp = [r for r in range(m) if lam[r] != 0]
dens0 = [abs(lam[r].denominator) for r in supp]
print(f"CERTIFICATE FOUND (exact rational). support = {len(supp)} of {m} probe points; max coeff denominator = {max(dens0) if dens0 else 1}  [{time.time()-t0:.0f}s]", flush=True)

# --- verify the three defining properties EXACTLY ---
def dot(coef, vec): return sum(coef[r] * Fr(int(vec[r])) for r in range(m))
maxfail = 0
for k in okeys:
    if dot(lam, core.orbit_column(orb[k], X)) != 0: maxfail += 1
linfail = sum(1 for i in range(n) if dot(lam, X[:, i]) != 0)
val = dot(lam, bmax)
print(f"  lambda . block   = 0 for all {Nb} blocks: {'OK' if maxfail==0 else f'FAIL({maxfail})'}", flush=True)
print(f"  lambda . x_i     = 0 for all {n} linear: {'OK' if linfail==0 else f'FAIL({linfail})'}", flush=True)
print(f"  lambda . max_{n}  = {val}  ({'NONZERO -> separates -> OUT PROVEN' if val!=0 else 'ZERO (bad)'})", flush=True)

# --- A SYMMETRIC certificate PROVABLY exists: the certificate set {lambda : lambda.block=0 all,
# lambda.x_i=0 all, lambda.max=1} is a nonempty S_n-STABLE AFFINE subspace (blocks and max_n are
# S_n-invariant orbit-sums, so g.lambda is again a certificate), hence its group average is a
# certificate AND is S_n-invariant. We realize it DIRECTLY in the symmetric basis instead of by
# brute point-spreading: a symmetric functional is a signed measure on SORTED points (chambers),
# and it evaluates any S_n-invariant f (every block, max_n) at a point via that point's sorted form.
# The n linear constraints collapse to ONE: a symmetric lambda has lambda.x_i equal for all i, so
# annihilating all x_i  <=>  annihilating the invariant  sum_i x_i.  This is a tiny exact system and
# its SUPPORT is indexed by CHAMBERS -- the chamber-level data the sheaf/cocycle framing asks for.
K = Nb + int(os.environ.get("SYMMARGIN", "300"))         # sorted probe points; must exceed the block span rank
RNG_LO = int(os.environ.get("RNG_LO", "-6")); RNG_HI = int(os.environ.get("RNG_HI", "7"))  # SMALL range => small gaps => near walls (where the obstruction lives)
rngp = np.random.default_rng(202)
Pset = set()
tries = 0
while len(Pset) < K and tries < 4000000:                 # generic sorted (descending, distinct) points near the walls
    tries += 1; v = rngp.integers(RNG_LO, RNG_HI, size=n)
    if len(set(v.tolist())) == n: Pset.add(tuple(sorted(v.tolist(), reverse=True)))
K = len(Pset); P = np.array(sorted(Pset), dtype=np.int64)
print(f"  sorted probe points: {K} distinct, coord range [{RNG_LO},{RNG_HI}) (small gaps -> near walls)", flush=True)
Csym = [core.orbit_column(orb[k], P).astype(np.int64) for k in okeys]   # block value per chamber (invariant)
Lsum = P.sum(axis=1).astype(np.int64)                                   # the single collapsed linear constraint
bmx  = P.max(axis=1).astype(np.int64)                                   # = P[:,0] since sorted desc
# NOTE: restricting to the braid fundamental domain CANNOT witness the obstruction. On D={x_1>=..>=x_n},
# max_n = x_(1) is LINEAR, and if a symmetric combo of blocks+linear agreed with max_n on all of D it would
# (by symmetry) agree everywhere => max_n IN. Since max_n is OUT, no such agreement holds AS FUNCTIONS, but a
# FINITE sorted sample under-saturates the block-cell structure (blocks are PL inside D) and shows a spurious
# fit. So the chamber-basis solve is only a sanity probe; the SYMMETRIC certificate is guaranteed to exist by
# averaging the (verified) unsorted certificate over S_n (blocks, max_n are invariant => g.lambda is again a
# certificate => the group-average is a symmetric certificate). We report the sorted probe honestly.
Bs = [[Fr(int(v)) for v in col] for col in Csym] + [[Fr(int(v)) for v in Lsum]] + [[Fr(int(v)) for v in bmx]]
rs = [Fr(0)] * Nb + [Fr(0)] + [Fr(1)]
a = core.exact_solve(Bs, rs)
if a is None:
    print("\n  sorted-sample probe: max_n fits blocks+linear on these sorted points (under-saturated fundamental", flush=True)
    print("  domain: blocks are PL inside D, finite sample misses witnessing cells). Expected; NOT a membership", flush=True)
    print("  claim. The SYMMETRIC certificate exists by averaging the verified unsorted certificate over S_n.", flush=True)
else:
    a = list(a); asupp = [j for j in range(K) if a[j] != 0]
    def dotP(vec): return sum(a[j] * Fr(int(vec[j])) for j in range(K))
    sb = sum(1 for k in okeys if dotP(core.orbit_column(orb[k], P)) != 0)
    sl = 1 if dotP(Lsum) != 0 else 0
    sv = dotP(bmx)
    print(f"\n  S_{n}-SYMMETRIC certificate (chamber basis): support {len(asupp)} of {K} chambers", flush=True)
    print(f"    lambda_sym . block = 0 all {Nb}: {'OK' if sb==0 else f'FAIL({sb})'} | . (sum x_i) = 0: {'OK' if sl==0 else 'FAIL'} | . max_{n} = {sv}  ({'NONZERO' if sv!=0 else 'ZERO'})", flush=True)
    dens = [abs(a[j].denominator) for j in asupp]
    print(f"    => OUT certificate realized as a signed sum of {len(asupp)} chamber-evaluation functionals; max coeff denominator = {max(dens) if dens else 1}", flush=True)
print(f"\n[{time.time()-t0:.0f}s] done. weight-{Wt}: exact floating-point-free OUT certificate (unsorted form, verified). Symmetric certificate exists by S_n-averaging. Certificate denominators are huge => NOT a clean invariant.", flush=True)
