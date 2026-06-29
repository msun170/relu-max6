"""Build an exact dual certificate that max7 is not in the complete weight-2 two-layer family.

We want lambda (rational, one entry per sample point) with lambda . column_j = 0 for every weight-2
block-orbit column and every linear column, and lambda . target != 0. Its existence proves the target
(max7 evaluated at the sample points) is outside the span, so max7 has no weight-2 representation.

We find lambda by solving [A^T ; b^T] lambda = [0,...,0,1] exactly over Q. The result is saved to
results/max7_certificate.json and is checked by check_weight2_max7_infeasible.py.
"""
import json, time
from fractions import Fraction as F
import numpy as np
import core

n = 7; t0 = time.time()
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
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"zonotopes={len(zonos)} blocks={len(blocks)} orbits={N} [{time.time()-t0:.0f}s]", flush=True)

m = N + n + 80
X = np.random.default_rng(5).integers(-10, 11, size=(m, n)).astype(np.int64)
cols = [core.orbit_column(orb[k], X) for k in okeys] + [X[:, i] for i in range(n)]
b = X.max(axis=1)

# solve [A^T ; b^T] lambda = [0,...,0,1] over Q
rows = [[F(int(c[i])) for i in range(m)] for c in cols] + [[F(int(v)) for v in b]]
rhs = [F(0)]*len(cols) + [F(1)]
lam = core.exact_solve(rows, rhs)
assert lam is not None, "system consistent -> max7 WOULD be in span (unexpected)"

# sanity-verify before saving
assert all(sum(lam[i]*int(c[i]) for i in range(m)) == 0 for c in cols)
assert sum(lam[i]*int(v) for i, v in enumerate(b)) != 0
out = {"n": n, "X": X.tolist(),
       "lambda": [f"{x.numerator}/{x.denominator}" for x in lam],
       "num_orbits": N}
json.dump(out, open("results/max7_certificate.json", "w"))
print(f"saved results/max7_certificate.json (lambda has {sum(1 for x in lam if x)} nonzeros) [{time.time()-t0:.0f}s]", flush=True)
