"""Search: is max_n a signed sum of weight-2 P2 support functions (i.e. a 2-hidden-layer network)?

Builds the S_n-orbit basis of building blocks, then tries to solve exactly for orbit coefficients and
a linear term reproducing max_n. A float least-squares residual first decides consistency cheaply; if
consistent we solve exactly and verify on fresh integer points.

Usage:  python search.py n [Rk] [Cm] [Jj]
  Rk = max generators per zonotope (default 3), Cm = vertex cap (default 12), Jj = max join arity (2).

Results so far: max5, max6 -> YES (constructions found and verified). max7, max8 -> inconsistent even
in much richer classes (strong evidence the weight-2 2-layer frontier ends at 6). A negative here is
class-restricted, not a formal lower bound.
"""
import sys, time
import numpy as np
from fractions import Fraction as F
from math import gcd
import core

def run(n, R=3, cap=12, join=2):
    t0 = time.time()
    terms = core.building_blocks(n, R, cap, join)
    orb = core.orbits_of(terms, n); okeys = list(orb); N = len(okeys)
    print(f"n={n} R={R} cap={cap} join={join}: blocks={len(terms)} orbits={N} [{time.time()-t0:.0f}s]", flush=True)

    # support-function matrix on integer sample points (more rows than columns)
    m = N + n + 200; X = np.random.default_rng(0).integers(-12, 13, size=(m, n)).astype(np.int64)
    cols = [core.orbit_column(orb[k], X) for k in okeys]
    Amat = np.column_stack(cols + [X[:, i] for i in range(n)]).astype(np.float64)
    bvec = X.max(axis=1).astype(np.float64)
    sol, _, rank, _ = np.linalg.lstsq(Amat, bvec, rcond=None)
    resid = float(np.linalg.norm(Amat @ sol - bvec))
    print(f"float lstsq: rank={rank}/{Amat.shape[1]}, residual={resid:.3e} [{time.time()-t0:.0f}s]", flush=True)
    if resid > 1e-6:
        print(f"INCONSISTENT: max{n} is not in the span of these {N} P2 orbit support functions "
              f"(class-restricted negative, not a lower bound).", flush=True)
        return None

    # exact solve and exact verification on fresh points
    A = [[F(int(v)) for v in row] for row in np.column_stack(cols + [X[:, i] for i in range(n)])]
    w = core.exact_solve(A, [F(int(v)) for v in X.max(axis=1)])
    cs = w[:N]; Ls = w[N:N+n]
    Y = np.random.default_rng(999).integers(-40, 41, size=(4000, n)).astype(np.int64)
    out = np.zeros(len(Y), dtype=object)
    for i, k in enumerate(okeys):
        if cs[i] != 0:
            col = core.orbit_column(orb[k], Y)
            for r in range(len(Y)): out[r] += cs[i]*int(col[r])
    for i in range(n):
        if Ls[i] != 0:
            for r in range(len(Y)): out[r] += Ls[i]*int(Y[r, i])
    bad = sum(1 for r in range(len(Y)) if out[r] != int(Y[r].max()))
    D = 1
    for c in list(cs)+list(Ls):
        if c != 0: D = D*c.denominator//gcd(D, c.denominator)
    nz = sum(1 for c in cs if c != 0)
    print(f"VERIFY max{n}: {len(Y)} fresh points, mismatches={bad}; {nz} orbit terms, D={D} [{time.time()-t0:.0f}s]", flush=True)
    if bad == 0:
        print(f"max{n} IS computable in 2 hidden layers.", flush=True)
        return okeys, orb, cs, Ls, D
    return None

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    R, cap, join = 3, 12, 2
    for a in sys.argv[2:]:
        if a.startswith("R"): R = int(a[1:])
        elif a.startswith("C"): cap = int(a[1:])
        elif a.startswith("J"): join = int(a[1:])
    run(n, R, cap, join)
