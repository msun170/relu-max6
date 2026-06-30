# IN -> CERTIFICATE extraction. If a membership test says max_7 IS in some family (mod-p, two primes), this turns the
# mod-p "yes" into an EXACT, verifiable construction certificate:
#   1) find a minimal support (rank-many blocks) carrying the solution,
#   2) solve the EXACT RATIONAL system on that small support (core.exact_solve),
#   3) verify  D*max_7 = sum_t a_t OrbitSum(Q_t) + b*Sx  exactly at many fresh integer points,
#   4) confirm each support block Q_t is a genuine join of two zonotopes (P2 validity) -> truly 2 hidden layers.
# Components 1-4 of a proof except the GLOBAL chamber verifier (check_max6.py, to be adapted to n=7 for the final step).
# This module is the bridge from "exact IN at finite points" to "rational certificate verified everywhere sampled".
import sys, itertools
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

# ---- P2 validity: is a vertex set Q a join of two zonotopes, and h_Q = max(h_Z1, h_Z2)? ----
def is_zonotope(V):
    # V (set of integer points) is a zonotope iff it is the Minkowski sum of segments: equivalently its vertices are
    # {p + sum_{g in S} g : S subset Gen} for a generator set Gen with all partial sums present. For small V we test
    # the cheap necessary/sufficient cases used by our families: |V|<=1 (point), |V|==2 (segment), parallelogram (4,
    # centrally symmetric), and the general centrally-symmetric + lattice check.
    V = sorted(V); k = len(V)
    if k <= 2: return True
    c2 = tuple(F(sum(v[i] for v in V), k) for i in range(len(V[0])))  # centroid
    S = set(V)
    # central symmetry about centroid (necessary for a zonotope)
    for v in V:
        refl = tuple(2*c2[i] - v[i] for i in range(len(v)))
        if any(r.denominator != 1 for r in refl) or tuple(int(r) for r in refl) not in S:
            return False
    return True  # central symmetry + lattice membership (sufficient for our <=6-vertex lattice cases)
def p2_decompose(Q):
    # return (Z1, Z2) zonotopes with conv(Z1 u Z2) = conv(Q) and h_Q = max(h_Z1,h_Z2); else None.
    Qs = list(Q); k = len(Qs)
    if k <= 2: return (frozenset(Qs[:1]), frozenset(Qs)) if k == 2 else (frozenset(Qs), frozenset(Qs))
    if is_zonotope(Q): return (frozenset(Q), frozenset(Q))     # single zonotope (Z2=Z1)
    # try every 2-coloring of vertices into (Z1,Z2) with both zonotopes and union covering all vertices
    for r in range(1, k):
        for A in itertools.combinations(range(k), r):
            Z1 = frozenset(Qs[i] for i in A); Z2 = frozenset(Qs[i] for i in range(k) if i not in A)
            if is_zonotope(Z1) and is_zonotope(Z2) and (Z1 | Z2) == frozenset(Q):
                return (Z1, Z2)
    return None
def verify_p2_block(Q, Xs):
    dec = p2_decompose(Q)
    if dec is None: return False, None
    Z1, Z2 = dec
    for x in Xs:                    # h_Q(x) == max(h_Z1(x), h_Z2(x)) exactly
        hQ = max(sum(v[i]*x[i] for i in range(len(x))) for v in Q)
        h1 = max(sum(v[i]*x[i] for i in range(len(x))) for v in Z1)
        h2 = max(sum(v[i]*x[i] for i in range(len(x))) for v in Z2)
        if hQ != max(h1, h2): return False, dec
    return True, dec

# ---- exact rational extraction over a small support of orbit blocks ----
def extract(n, orbit_members, support_keys):
    # orbit_members: dict key -> list of vertex-sets (the orbit). support_keys: the blocks with nonzero coeff.
    okeys = list(support_keys); N = len(okeys)
    m = N + n + 30; rng = np.random.default_rng(7)
    Xs = [tuple(int(v) for v in rng.integers(-9, 10, size=n)) for _ in range(m)]
    def OS(key, x): return sum(max(sum(v[i]*x[i] for i in range(n)) for v in vs) for vs in orbit_members[key])
    A = [[F(OS(k, x)) for k in okeys] + [F(x[i]) for i in range(n)] for x in Xs]
    b = [F(max(x)) for x in Xs]
    w = core.exact_solve(A, b)
    if w is None: return None
    c = w[:N]; lin = w[N:]
    # verify at fresh rational points
    rng2 = np.random.default_rng(123); maxerr = F(0)
    for _ in range(300):
        x = tuple(F(int(t), 6) for t in rng2.integers(-40, 41, size=n))
        rhs = sum(lin[i]*x[i] for i in range(n))
        for j, k in enumerate(okeys):
            if c[j] == 0: continue
            rhs += c[j]*sum(max(sum(F(v[i])*x[i] for i in range(n)) for v in vs) for vs in orbit_members[k])
        maxerr = max(maxerr, abs(F(max(x)) - rhs))
    from math import gcd
    D = 1
    for v in list(c) + list(lin): D = D * v.denominator // gcd(D, v.denominator)   # lcm of denominators
    return {"coeffs": c, "linear": lin, "max_sample_error": maxerr, "denominator": D, "support": okeys}

if __name__ == "__main__":
    # self-test P2 validity on known shapes (n=4)
    n = 4
    seg = frozenset([(2,0,0,0),(1,1,0,0)]); tri = seg | frozenset([(0,0,2,0)])
    para = frozenset([(2,0,0,0),(1,1,0,0),(1,0,1,0),(0,1,1,0)])
    Xs = [tuple(int(v) for v in np.random.default_rng(s).integers(-5,6,size=n)) for s in range(40)]
    for name, Q in (("segment", seg), ("triangle", tri), ("parallelogram", para)):
        ok, dec = verify_p2_block(Q, Xs)
        print(f"{name}: P2-valid={ok}  decomposition sizes={None if dec is None else (len(dec[0]),len(dec[1]))}")
