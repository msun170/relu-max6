"""Conditional lower bound: does max_n have a two-hidden-layer representation on the weight-2 lattice?

The two-hidden-layer functions whose building blocks have weight-2 vertices form a FINITE family: a
weight-2 zonotope has vertices among finitely many lattice points, and the correct building blocks are the
TWO-WAY joins of such zonotopes (a join of three zonotopes is a max of three 1-layer functions, which is
three hidden layers, not two; only two-way joins are valid). Degenerate point zonotopes are included, so
pyramids appear. We enumerate the family completely, using ALL weight-2 point-difference generators (braid
and non-braid), and solve over Q. Inconsistent means max_n has no weight-2 two-layer representation.

Usage: python lower_bound.py n      (n=6 is consistent; n=7, n=8 are inconsistent)
"""
import sys, time
from fractions import Fraction as F
import numpy as np
import core

n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
t0 = time.time()
W2 = core.weight2_points(n); W2set = set(W2)

# full undirected generators: every difference of two weight-2 points (braid and non-braid)
gens = sorted({d for p in W2 for q in W2 if p != q
               for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})

# all weight-2 zonotopes by pruned DFS (grow by Minkowski-adding a segment, require vertices stay on lattice)
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

P1 = [frozenset([p]) for p in W2] + list(zonos)      # points (degenerate zonotopes) + zonotopes
blocks = set(frozenset(a) for a in P1)
for i in range(len(P1)):
    for j in range(i, len(P1)):
        u = P1[i] | P1[j]
        if len(u) <= 2*n + 2: blocks.add(u)            # two-way joins only
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"n={n}: zonotopes={len(zonos)} two-way blocks={len(blocks)} orbits={N} [{time.time()-t0:.0f}s]", flush=True)

# exact solve over Q: is max_n in the span of the orbit support functions (+ linear)?
m = N + n + 80
X = np.random.default_rng(5).integers(-10, 11, size=(m, n)).astype(np.int64)
cols = [core.orbit_column(orb[k], X) for k in okeys]
A = [[F(int(v)) for v in row] for row in np.column_stack(cols + [X[:, i] for i in range(n)])]
b = [F(int(v)) for v in X.max(axis=1)]
w = core.exact_solve(A, b)
if w is None:
    print(f"n={n}: INCONSISTENT -> max{n} has NO weight-2 two-hidden-layer representation. [{time.time()-t0:.0f}s]")
else:
    print(f"n={n}: consistent -> max{n} HAS a weight-2 two-hidden-layer representation. [{time.time()-t0:.0f}s]")
