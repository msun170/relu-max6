# EXACT scale-invariant k=1 certificate. T(v)=sum_j a_j (v.x_j)_+ ; on the sign-chamber where v.x_j>0 for
# j in S, T(v)=(sum_{j in S} a_j x_j).v, so T==0 on ALL of R^n  <=>  sum_{j in S} a_j x_j = 0 for every
# realizable positive-set S. Enumerate the S by sampling integer v, impose those exact integer conditions,
# pin mu(max_n)=1, and solve over Q. A solution is an EXACT, lattice-free (scale-invariant) certificate that
# max_n is not 1-hidden-layer.
import sys, itertools
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

for n in (3,):
    base = set()
    for p in itertools.product(range(-2, 3), repeat=n):
        if 1 <= sum(abs(c) for c in p) <= 3: base.add(p)
    pts = sorted(base); J = len(pts)
    Xp = [np.array(p, dtype=np.int64) for p in pts]

    # enumerate realizable positive-sets S via sampling integer directions v
    rng = np.random.default_rng(0)
    patterns = set()
    for _ in range(40000):
        v = rng.integers(-6, 7, size=n)
        S = tuple(int(np.dot(v, Xp[j]) > 0) for j in range(J))
        patterns.add(S)
    # build exact conditions: for each pattern, sum_{j in S} a_j x_j = 0  (n rows each)
    rows = []; rhs = []
    for S in patterns:
        for k in range(n):
            rows.append([F(int(S[j]) * int(pts[j][k])) for j in range(J)]); rhs.append(F(0))
    # pin mu(max_n) = 1
    rows.append([F(max(pts[j])) for j in range(J)]); rhs.append(F(1))
    a = core.exact_solve(rows, rhs)
    if a is None:
        print(f"n={n}: J={J}, patterns={len(patterns)} -> NO certificate (max_n in 1-layer span at this support?)")
        continue
    # exact verification: every pattern condition holds, and mu(max_n)=1
    ok = True
    for S in patterns:
        for k in range(n):
            if sum(a[j]*F(int(S[j])*int(pts[j][k])) for j in range(J)) != 0: ok = False
    mu_max = sum(a[j]*F(max(pts[j])) for j in range(J))
    supp = sum(1 for c in a if c != 0)
    print(f"n={n}: J={J}, chambers={len(patterns)}, support={supp}, all-conditions-exact={ok}, mu(max_n)={mu_max}"
          f"  => {'EXACT scale-invariant certificate: max_'+str(n)+' is NOT 1 hidden layer' if ok and mu_max!=0 else 'FAILED'}")
