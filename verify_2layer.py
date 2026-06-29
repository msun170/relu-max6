# LINCHPIN CHECK. Recent literature (arXiv:2502.09324) states max_5 needs 3 hidden layers. Our span test
# says max_5, max_6 are IN the weight-2 (signed sum of joins-of-two-zonotopes) span = 2 hidden layers. These
# cannot both be true unless the models differ. Settle it by EXPLICIT CONSTRUCTION + numerical check:
# solve exactly for max_n = sum_t c_t h_{Q_t} + linear over the weight-2 family, then evaluate BOTH sides at
# fresh random points. If they match exactly, max_n IS a signed sum of joins of two zonotopes (= 2 hidden
# ReLU layers), as a verifiable fact. Also report whether the representation uses NEGATIVE coefficients
# (a genuine difference) vs only nonnegative (a pure Minkowski / convex case).
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
        for j in range(i, len(P1)):
            blocks.add(P1[i] | P1[j])
    return core.orbits_of(list(blocks), n)

def hQ(members, x):
    # support function of an orbit (sum over members) at a single point x, exact integer
    tot = 0
    for vs in members:
        tot += max(sum(v[k]*x[k] for k in range(len(x))) for v in vs)
    return tot

def solve_and_verify(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    # build exact rational system at integer sample points: columns = orbit support fns + n linear coords
    m = N + n + 25
    rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    A = []
    for x in Xs:
        row = [F(hQ(orb[k], x)) for k in okeys] + [F(x[i]) for i in range(n)]
        A.append(row)
    b = [F(max(x)) for x in Xs]
    w = core.exact_solve(A, b)
    if w is None:
        print(f"n={n}: NO exact solution in weight-2 (max_{n} OUT) -- consistent with needing 3 layers.")
        return
    # verify at FRESH random points (including non-integer rationals) that sum_t w_t hQ + linear == max
    rng2 = np.random.default_rng(99)
    maxerr = F(0); used_neg = False
    coeffs = w[:N]; lincoeff = w[N:]
    for k in range(N):
        if coeffs[k] < 0: used_neg = True
    for _ in range(400):
        x = tuple(F(int(t), 6) for t in rng2.integers(-30, 31, size=n))   # rationals k/6
        lhs = F(max(x))
        rhs = sum(lincoeff[i]*x[i] for i in range(n))
        for k in range(N):
            if coeffs[k] == 0: continue
            s = 0
            for vs in orb[okeys[k]]:
                s += max(sum(F(v[j])*x[j] for j in range(n)) for v in vs)
            rhs += coeffs[k]*s
        maxerr = max(maxerr, abs(lhs - rhs))
    nz = sum(1 for c in coeffs if c != 0)
    print(f"n={n}: exact solution found. nonzero orbits={nz}/{N}. uses negative coeffs (genuine difference)? "
          f"{used_neg}.  max |max_n - representation| over 400 fresh rational points = {maxerr}  "
          f"({'EXACT MATCH => max_%d IS 2 hidden ReLU layers' % n if maxerr==0 else 'MISMATCH!!'})", flush=True)

for n in (4, 5, 6):
    solve_and_verify(n)
