# DECISIVE: is max_7 in the COMPLETE weight-3 (2-hidden-layer) span? max_5,6 are 2-layer; max_7 only tested
# at weight-2 (OUT) and RESTRICTED weight-3. If max_7 IN weight-3 => max_7 IS 2-layer (resolves it, like
# max_6). If OUT => strong evidence for the 2-vs-3 separation at n=7.
# Completeness: weight-3 zonotopes have dim <=3 (token lemma), so we enumerate base + <=3 generators drawn
# from the FULL set of primitive lattice-point differences (NOT just roots -- captures non-root edges), with
# all subset sums in the 3*Delta lattice. Joins via the zonotope-orbit-rep trick (complete up to symmetry).
# Span test uses the small-rank trick: rank is a few hundred, so a few thousand sample rows suffice.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

n = 7; W = 3; P = 2147483647
def wpoints(W):
    return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W3 = wpoints(W); W3set = set(W3)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

# full primitive generator directions = primitive differences of weight-3 points (both orientations dedup)
gens = set()
for p in W3:
    for q in W3:
        if p == q: continue
        d = tuple(q[k]-p[k] for k in range(n))
        g = reduce(gcd, [abs(x) for x in d])
        d = tuple(x//g for x in d)
        if d > tuple(-x for x in d): gens.add(d)
gens = sorted(gens)
t0 = time.time()
print(f"weight-3: {len(W3)} points, {len(gens)} primitive generator directions", flush=True)

# enumerate zonotopes: base + <=3 generators, all subset sums in lattice
zonos = set()
def subset_sums_ok(p, G):
    verts = set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G, s):
            q = p
            for g in S: q = shift(q, g)
            if q not in W3set: return None
            verts.add(q)
    return frozenset(verts)
for p in W3:
    # 1 generator
    for g in gens:
        v = subset_sums_ok(p, [g])
        if v and len(v) >= 2: zonos.add(v)
for p in W3:
    for i in range(len(gens)):
        gi = gens[i]
        if shift(p, gi) not in W3set: continue
        for j in range(i+1, len(gens)):
            v = subset_sums_ok(p, [gi, gens[j]])
            if v and len(v) >= 3: zonos.add(v)
zonos = list(zonos)
print(f"  zonotopes (<=2 gen): {len(zonos)}   [{time.time()-t0:.0f}s]", flush=True)
# add 3-generator zonotopes from existing 2-gen by extending (dim<=3)
two = [z for z in zonos if len(z) >= 4]
base3 = set(zonos)
for z in two:
    p0 = min(z)
    for g in gens:
        v = None
        verts = set(z); ok = True
        for u in z:
            w = shift(u, g)
            if w not in W3set: ok = False; break
            verts.add(w)
        if ok and len(verts) > len(z): base3.add(frozenset(verts))
zonos = list(base3)
print(f"  zonotopes (<=3 gen, complete): {len(zonos)}   [{time.time()-t0:.0f}s]", flush=True)

# zonotope orbit reps, then joins rep x all-zonotopes (complete up to S_n), plus zonotopes & points
zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
print(f"  zonotope orbits: {len(zreps)}   [{time.time()-t0:.0f}s]", flush=True)
blocks = set(zonos)
for a in zreps:
    for b in zonos:
        u = a | b
        if len(u) <= 16: blocks.add(u)
print(f"  blocks before canon: {len(blocks)}   [{time.time()-t0:.0f}s]", flush=True)
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"  JOIN-ORBITS (complete weight-3 family): {N}   [{time.time()-t0:.0f}s]", flush=True)

# small-rank span test (rank is a few hundred, so ~1500 sample rows safely exceed it; verified by rank<m)
m = 1500
X = np.random.default_rng(11).integers(-9, 10, size=(m, n)).astype(np.int64)
cols = [core.orbit_column(orb[k], X).astype(np.int64) for k in okeys]
A = np.column_stack(cols + [X[:, i].astype(np.int64) for i in range(n)])
b = X.max(axis=1).astype(np.int64)
def modp_rank(M, extra=None):
    Aa = (M % P).astype(np.int64)
    if extra is not None: Aa = np.column_stack([Aa, (extra % P).astype(np.int64)])
    Aa = Aa.copy(); rows, ncols = Aa.shape; r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, rows) if Aa[i, c] % P != 0), -1)
        if piv < 0: continue
        Aa[[r, piv]] = Aa[[piv, r]]; Aa[r] = (Aa[r]*pow(int(Aa[r, c]), P-2, P)) % P
        for i in range(rows):
            if i != r and Aa[i, c] % P != 0: Aa[i] = (Aa[i]-Aa[i, c]*Aa[r]) % P
        r += 1
        if r == rows: break
    return r
ra = modp_rank(A); rab = modp_rank(A, b)
safe = "OK (rank<m)" if ra < m else "WARNING: rank==m, need more rows!"
print(f"\nrank(A)={ra}, rank(A|b)={rab}, m={m} [{safe}]  =>  max7 "
      f"{'IN weight-3 (2-LAYER! resolves max7)' if ra==rab else 'OUT of complete weight-3'}  [{time.time()-t0:.0f}s]", flush=True)
