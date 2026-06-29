"""Minimize the max6 construction: among all exact signed combinations of weight-2 P2 orbit support
functions equal to max6, find one with the fewest orbit terms and a small denominator.

L1 minimization (LP) gives a sparse support; we solve exactly on it, verify on fresh points, then
greedily drop terms while an exact solution still exists. Every candidate is verified exactly.
"""
import time
import numpy as np
from fractions import Fraction as F
from math import gcd
from scipy.optimize import linprog
import core

n = 6; t0 = time.time()
terms = core.building_blocks(n, R=3, cap=12, join=2)
orb = core.orbits_of(terms, n); okeys = list(orb); N = len(okeys)
m = N + n + 300; X = np.random.default_rng(1).integers(-14, 15, size=(m, n)).astype(np.int64)
OC = np.column_stack([core.orbit_column(orb[k], X) for k in okeys]).astype(np.float64)
A = np.column_stack([OC] + [X[:, i].astype(np.float64) for i in range(n)])
b = X.max(axis=1).astype(np.float64)
print(f"orbits={N} [{time.time()-t0:.0f}s]", flush=True)

# exact solve restricted to a chosen orbit support (linear term always allowed), with exact verification
def exact_on_support(supp):
    M = []
    for r in range(m):
        row = [F(int(core.orbit_column(orb[okeys[k]], X[r:r+1])[0])) for k in supp]
        row += [F(int(X[r, i])) for i in range(n)]
        M.append(row)
    w = core.exact_solve(M, [F(int(v)) for v in X.max(axis=1)])
    if w is None: return None
    cs = w[:len(supp)]; Ls = w[len(supp):len(supp)+n]
    Y = np.random.default_rng(7).integers(-30, 31, size=(3000, n)).astype(np.int64)
    out = np.zeros(len(Y), dtype=object)
    for j, k in enumerate(supp):
        if cs[j] != 0:
            col = core.orbit_column(orb[okeys[k]], Y)
            for r in range(len(Y)): out[r] += cs[j]*int(col[r])
    for i in range(n):
        if Ls[i] != 0:
            for r in range(len(Y)): out[r] += Ls[i]*int(Y[r, i])
    if any(out[r] != int(Y[r].max()) for r in range(len(Y))): return None
    D = 1
    for c in list(cs)+list(Ls):
        if c != 0: D = D*c.denominator//gcd(D, c.denominator)
    return cs, Ls, D

# L1 minimization over orbit coefficients to get a sparse starting support
cobj = [1.0]*N + [0.0]*n + [1.0]*N + [0.0]*n
res = linprog(c=cobj, A_eq=np.column_stack([A, -A]), b_eq=b, bounds=[(0, None)]*(2*(N+n)), method='highs')
sol = res.x[:N+n] - res.x[N+n:]
supp = [k for k in range(N) if abs(sol[k]) > 1e-7]
print(f"L1 support: {len(supp)} orbits [{time.time()-t0:.0f}s]", flush=True)

got = exact_on_support(supp)
if got is None:
    supp = list(range(N)); got = exact_on_support(supp)
cs, Ls, D = got
cur = [supp[j] for j in range(len(supp)) if cs[j] != 0]
print(f"exact support: {len(cur)} orbits, D={D} [{time.time()-t0:.0f}s]", flush=True)

# greedy term removal
improved = True
while improved:
    improved = False
    for k in list(cur):
        trial = [x for x in cur if x != k]
        r = exact_on_support(trial)
        if r is not None:
            cs, Ls, D = r; cur = [trial[j] for j in range(len(trial)) if cs[j] != 0]
            improved = True; print(f"  dropped -> {len(cur)} orbits, D={D}", flush=True); break

cs, Ls, D = exact_on_support(cur)
print(f"MINIMAL: {len(cur)} orbit terms, D={D} [{time.time()-t0:.0f}s]", flush=True)
with open("results/construction_min.txt", "w") as fh:
    fh.write(f"max6 minimal construction: {len(cur)} P2 orbit terms, D={D}, L={[str(x) for x in Ls]}\n")
    for j, k in enumerate(cur):
        fh.write(f"coeff {cs[j]} (orbit size {len(orb[okeys[k]])}): {sorted(orb[okeys[k]][0])}\n")
print("wrote results/construction_min.txt", flush=True)
