# Print the explicit minimal 2-layer representations of max_5, max_6: which orbits (block shapes) and coefficients.
# Look for a generalizable pattern (vertex counts, complexity, coefficient denominators) toward max_7.
import sys, itertools
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

def family(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q
                   for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zonos = set()
    def dfs(verts, gi):
        if len(verts) >= 2: zonos.add(frozenset(verts))
        for j in range(gi+1, len(gens)):
            g = gens[j]; new = set(); ok = True
            for v in verts:
                wv = tuple(v[k]+g[k] for k in range(n))
                if wv not in W2set: ok = False; break
                new.add(wv)
            if ok: dfs(verts | new, j)
    for p in W2: dfs({p}, -1)
    P1 = [frozenset([p]) for p in W2] + list(zonos)
    blocks = set(P1)
    for i in range(len(P1)):
        for j in range(i, len(P1)): blocks.add(P1[i] | P1[j])
    return core.orbits_of(list(blocks), n)

def hQ(members, x):
    return sum(max(sum(v[k]*x[k] for k in range(len(x))) for v in vs) for vs in members)

def solve(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 25; rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7,8,size=n)) for _ in range(m)]
    A = [[F(hQ(orb[k],x)) for k in okeys] + [F(x[i]) for i in range(n)] for x in Xs]
    b = [F(max(x)) for x in Xs]
    w = core.exact_solve(A, b)
    if w is None: print(f"max_{n}: OUT"); return
    print(f"\n=== max_{n}: {sum(1 for c in w[:N] if c!=0)} nonzero orbits ===")
    for k in range(N):
        if w[k] == 0: continue
        rep = orb[okeys[k]][0]  # representative vertex set
        # describe: #verts, the actual points (sorted), orbit size
        verts = sorted(tuple(p) for p in rep)
        print(f"  coeff {str(w[k]):>8}  orbitsize={len(orb[okeys[k]]):>3}  nverts={len(rep)}  verts={verts}")
    print(f"  linear part: {[str(c) for c in w[N:]]}")

for n in (5, 6):
    solve(n)
