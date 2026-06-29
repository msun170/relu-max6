# Does the EXISTING weight-2 dual certificate lambda also annihilate WEIGHT-3 blocks?
# lambda (223 rational coeffs on 223 points X) satisfies: lambda . (w2 orbit-sum)(X) = 0 for all weight-2
# orbits, lambda . linear = 0, lambda . max7 != 0. If ALSO lambda . (w3 orbit-sum)(X) = 0 for EVERY weight-3
# orbit, then by symmetrization max7 is not in the weight-3 span => max7 OUT of weight-3, PROVEN by the same
# certificate (and evidence lambda is scale-invariant -> the proof route). Test it exactly.
import sys, itertools, json, time
from math import gcd
from fractions import Fraction as F
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7
d = json.load(open("C:/Users/nuswe/relu-max6/results/max7_certificate.json"))
X = [tuple(p) for p in d["X"]]; lam = [F(s) for s in d["lambda"]]
m = len(X)
def col(members):   # orbit-sum support fn at the 223 points (exact ints)
    out = [0]*m
    for vs in members:
        V = list(vs)
        for i, x in enumerate(X):
            out[i] += max(sum(v[k]*x[k] for k in range(n)) for v in V)
    return out
def dot(c): return sum(lam[i]*c[i] for i in range(m))

# sanity: lambda . max7 != 0, and lambda . linear == 0
bmax = [max(x) for x in X]
print("lambda . max7 =", dot(bmax), "(want != 0)", flush=True)
for i in range(n):
    li = [x[i] for x in X]
    assert dot(li) == 0, f"lambda not orthogonal to linear coord {i}!"
print("lambda orthogonal to all linear coords: OK", flush=True)

# weight-3 family (full generator set, dim<=3), test lambda on every orbit-sum
W = 3
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W3 = wpoints(W); W3set = set(W3)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W3:
    for q in W3:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = sorted(gens)
def ssz(p, G):
    verts = set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G, s):
            q = p
            for g in S: q = shift(q, g)
            if q not in W3set: return None
            verts.add(q)
    return frozenset(verts)
t0 = time.time()
zonos = set()
for p in W3:
    for g in gens:
        v = ssz(p, [g])
        if v and len(v) >= 2: zonos.add(v)
for p in W3:
    for i in range(len(gens)):
        if shift(p, gens[i]) not in W3set: continue
        for j in range(i+1, len(gens)):
            v = ssz(p, [gens[i], gens[j]])
            if v and len(v) >= 3: zonos.add(v)
base3 = set(zonos)
for z in [z for z in zonos if len(z) >= 4]:
    for g in gens:
        verts = set(z); ok = True
        for u in z:
            w = shift(u, g)
            if w not in W3set: ok = False; break
            verts.add(w)
        if ok and len(verts) > len(z): base3.add(frozenset(verts))
zonos = list(base3)
zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
blocks = set(zonos)
for a in zreps:
    for b_ in zonos:
        u = a | b_
        if len(u) <= 16: blocks.add(u)
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"weight-3 orbits: {N}  [{time.time()-t0:.0f}s]", flush=True)

nonzero = 0; checked = 0; first_bad = None
for k in okeys:
    val = dot(col(orb[k]))
    checked += 1
    if val != 0:
        nonzero += 1
        if first_bad is None: first_bad = (k, val)
    if checked % 2000 == 0:
        print(f"  checked {checked}/{N}, nonzero so far {nonzero}  [{time.time()-t0:.0f}s]", flush=True)
print(f"\nweight-3 orbits annihilated by lambda: {N-nonzero}/{N}; NONZERO: {nonzero}", flush=True)
if nonzero == 0:
    print("=> SAME certificate kills ALL weight-3 blocks => max7 OUT of weight-3, PROVEN. lambda is scale-extending!", flush=True)
else:
    print(f"=> lambda is weight-2-specific (e.g. orbit {first_bad[0]} gives {first_bad[1]}). Does NOT prove weight-3.", flush=True)
print(f"done [{time.time()-t0:.0f}s]", flush=True)
