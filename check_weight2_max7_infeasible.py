"""Verify the exact certificate that max7 has no weight-2 two-hidden-layer representation.

Regenerates the complete weight-2 building-block family from scratch, then checks the saved dual vector
lambda against it: lambda . column = 0 for every block-orbit column and every linear column, and
lambda . target != 0. All arithmetic is exact (Fraction). Pass means max7 is provably outside the span.
"""
import json, time
from fractions import Fraction as F
import numpy as np
import core

n = 7; t0 = time.time()
cert = json.load(open("results/max7_certificate.json"))
X = np.array(cert["X"], dtype=np.int64)
lam = [F(s) for s in cert["lambda"]]
m = X.shape[0]

# regenerate the complete weight-2 family (same construction as the lower bound)
W2 = core.weight2_points(n); W2set = set(W2)
gens = sorted({d for p in W2 for q in W2 if p != q
               for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
zonos = set()
def dfs(verts, gi):
    if len(verts) >= 2: zonos.add(frozenset(verts))
    for j in range(gi+1, len(gens)):
        g = gens[j]; new = set(); ok = True
        for v in verts:
            w = tuple(v[k]+g[k] for k in range(n))
            if w not in W2set: ok = False; break
            new.add(w)
        if ok: dfs(verts | new, j)
for p in W2: dfs({p}, -1)
P1 = [frozenset([p]) for p in W2] + list(zonos)
blocks = set(frozenset(a) for a in P1)
for i in range(len(P1)):
    for j in range(i, len(P1)):
        u = P1[i] | P1[j]
        if len(u) <= 2*n + 2: blocks.add(u)
orb = core.orbits_of(list(blocks), n); okeys = list(orb)
print(f"regenerated family: {len(blocks)} blocks, {len(okeys)} orbits [{time.time()-t0:.0f}s]", flush=True)

# lambda must annihilate every block-orbit column and every linear column
bad = 0
for k in okeys:
    col = core.orbit_column(orb[k], X)
    if sum(lam[i]*int(col[i]) for i in range(m)) != 0: bad += 1
for i in range(n):
    if sum(lam[r]*int(X[r, i]) for r in range(m)) != 0: bad += 1
target = X.max(axis=1)
dot_target = sum(lam[i]*int(target[i]) for i in range(m))

print(f"orbit+linear columns annihilated by lambda: {'ALL' if bad == 0 else f'{bad} FAILED'}", flush=True)
print(f"lambda . max7  = {dot_target}  (must be nonzero)", flush=True)
ok = (bad == 0 and dot_target != 0)
print("RESULT:", "PASS -- max7 is not in the complete weight-2 two-layer span (exact)." if ok
      else "FAIL", flush=True)
