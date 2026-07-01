# HIGHER-ORDER OBSTRUCTION LADDER for max_7 at COMPLETE weight-3. We know: codim-1 (wall-jump) constraints are
# FEASIBLE (Type-II>0, braid target reachable) but the FULL identity is INFEASIBLE (max_7 exactly OUT). This finds
# the FIRST codimension at which infeasibility appears, by testing membership of max_7's data against the weight-3
# block span under finite-difference constraint families of increasing order:
#   L1 (codim-1): single centered 2nd differences  f(x+d)+f(x-d)-2f(x)        (single wall / gradient jump)
#   L2 (codim-2): mixed 2nd differences   f(x+d1+d2)-f(x+d1-d2)-f(x-d1+d2)+f(x-d1-d2)   (two-wall interaction)
#   L3 (codim-3): triple mixed differences (8-corner alternating sum)
#   Lfull: raw evaluations  f(x)            (= exact membership; must be INFEASIBLE since max_7 is OUT)
# A_k c = b_k tested exactly (mod p, two primes). First infeasible level = order of the obstruction.
# Soundness: a level INFEASIBLE on the COMPLETE family is a real obstruction (incomplete would falsely infeasible
# too early -- hence the complete weight-3 family).
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
P1 = 2147483647; P2 = 2147483629
def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
def build_complete(n, Wt, cap):
    L = [c for c in itertools.product(range(Wt+1), repeat=n) if sum(c) == Wt]; Ls = set(L); vi = {p:i for i,p in enumerate(L)}
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
    orb = core.orbits_of(list(bl), n); reps = list(orb)
    return L, vi, [orb[k][0] for k in reps]

n = int(os.environ.get("N", "7")); Wt = int(os.environ.get("WT", "3")); t0 = time.time()
L, vi, reps = build_complete(n, Wt, 16); NP = len(L); Nb = len(reps)
print(f"n={n} weight-{Wt}: {NP} pts, {Nb} orbit blocks  [{time.time()-t0:.0f}s]", flush=True)
perms = list(itertools.permutations(range(n))); G = len(perms)
pa = cp.asarray(np.array([[vi[tuple(p[g[k]] for k in range(n))] for p in L] for g in perms], dtype=np.int32))
rng = np.random.default_rng(7)
# build probe points + difference-row recipes (indices into the point list + signed weights)
pts = []; idx = {}
def addpt(x):
    x = tuple(int(t) for t in x)
    if x not in idx: idx[x] = len(pts); pts.append(x)
    return idx[x]
def rdir(): return rng.integers(-2, 3, size=n)
LEVELS = os.environ.get("LEVELS", "1,2,3,full").split(",")
LEVELS = [int(x) if x != "full" else "full" for x in LEVELS]
rows = {1: [], 2: [], 3: [], 'full': []}   # each row = list of (point_index, weight)
K = int(os.environ.get("K", "260"))
for _ in range(K):
    x0 = rng.integers(-8, 9, size=n); d1, d2, d3 = rdir(), rdir(), rdir()
    if 1 in LEVELS: rows[1].append([(addpt(x0+d1), 1), (addpt(x0-d1), 1), (addpt(x0), -2)])
    if 2 in LEVELS: rows[2].append([(addpt(x0+d1+d2), 1), (addpt(x0+d1-d2), -1), (addpt(x0-d1+d2), -1), (addpt(x0-d1-d2), 1)])
    if 3 in LEVELS:
        r3 = []
        for s1 in (1,-1):
            for s2 in (1,-1):
                for s3 in (1,-1):
                    r3.append((addpt(x0+s1*d1+s2*d2+s3*d3), s1*s2*s3))
        rows[3].append(r3)
for _ in range(int(os.environ.get("KFULL", "1600")) if 'full' in LEVELS else 0):
    rows['full'].append([(addpt(rng.integers(-12, 13, size=n)), 1)])
Y = np.array(pts, dtype=np.int64); PY = cp.asarray((Y @ np.array(L, dtype=np.int64).T).astype(cp.int64))
print(f"{len(pts)} probe points; building {Nb} columns  [{time.time()-t0:.0f}s]", flush=True)
def col(vs):
    Vc = cp.asarray([vi[p] for p in vs], dtype=cp.int32); k = len(vs); ix = pa[:, Vc]
    return PY[:, ix.reshape(-1)].reshape(len(Y), G, k).max(axis=2).sum(axis=1)
C = cp.stack([col(vs) for vs in reps], axis=1)              # (#pts) x Nb  (integer)
bmaxY = cp.asarray(Y.max(axis=1).astype(cp.int64))
print(f"columns built  [{time.time()-t0:.0f}s]", flush=True)
def assemble(recipe):   # rows recipe -> (rows x Nb) block matrix and target vector
    A = cp.zeros((len(recipe), Nb), dtype=cp.int64); b = cp.zeros(len(recipe), dtype=cp.int64)
    for r, terms in enumerate(recipe):
        for (pi, wgt) in terms:
            A[r] += wgt * C[pi]; b[r] += wgt * bmaxY[pi]
    return A, b
def feasible(A, b):
    def rkaug(M):
        M = M.copy(); rows_, ncols = M.shape; r = 0
        for c in range(ncols):
            sub = M[r:, c]; nz = cp.nonzero(sub)[0]
            if nz.size == 0: continue
            piv = r + int(nz[0].item())
            if piv != r:
                tmp = M[r].copy(); M[r] = M[piv]; M[piv] = tmp
            M[r] = (M[r] * pow(int(M[r, c].item()), P-2, P)) % P
            if r+1 < rows_:
                f = M[r+1:, c]; M[r+1:] -= cp.outer(f, M[r]); M[r+1:] %= P
            r += 1
            if r == rows_: break
        return r
    res = []
    for P in (P1, P2):
        rA = rkaug((A % P)); rAb = rkaug((cp.concatenate([A, b.reshape(-1,1)], axis=1) % P))
        res.append((rA, rAb, rA == rAb))
    return res[0][0], all(x[2] for x in res)   # rank(A) at P1, feasible(both primes)
print(f"\n  level                 rank(A)  #rows  saturated?  feasible?  (saturated+INFEASIBLE = real obstruction)", flush=True)
cum = []
for lvl in LEVELS:
    cum = cum + rows[lvl]
    A, b = assemble(cum); rk, feas = feasible(A, b); nrows = len(cum)
    sat = rk < nrows   # saturated iff the family span is smaller than the row space (=> non-vacuous)
    name = {1:'codim-1 (walls)', 2:'+codim-2 (ridges)', 3:'+codim-3', 'full':'+full values'}[lvl]
    tag = ('FEASIBLE' if feas else 'INFEASIBLE') + ('' if sat else '  (VACUOUS: rank>=#rows, ignore)')
    print(f"  {name:<20} {rk:>7}  {nrows:>5}  {str(sat):>9}   {tag}  [{time.time()-t0:.0f}s]", flush=True)
print(f"\n[{time.time()-t0:.0f}s] first INFEASIBLE level = the codimension/order of the obstruction killing max_7 at weight-3.", flush=True)
