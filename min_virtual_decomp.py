# ROUTE A: minimal virtual Minkowski decomposition for max_n (n in {5,6}, both IN weight-2).
#
# Every 2-layer rep is a SIGNED identity  Delta + N = M  (forced negativity, n>=5), with
#   M = sum_{c_i>0} c_i Q_i   (positive part),   N = sum_{c_i<0} (-c_i) Q_i   (negative part),
# so h_Delta + h_N = h_M. The DECOMPOSITION POLYHEDRON (Brandenburg-Grillo-Hertrich) is the set of all such
# (a,b)>=0; its minimal-mass vertex = the sparsest / least-total-mass virtual decomposition. We compute it:
#   minimize sum_i |c_i|  s.t.  sum_i c_i h_{Q_i} = h_Delta + linear   (L1 LP; linear term free),
# then EXACT-solve on the LP support to get a rational minimal decomposition, verify at fresh points, and
# EXHIBIT M and N (which orbit blocks, coefficients). Goal: read the structure of the extreme decomposition
# and look for a max_5 -> max_6 pattern that could suggest a max_7 support (feeds route B).
import sys, itertools, os
from fractions import Fraction as F
from math import gcd
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
from verify_2layer import family, hQ

def min_decomp(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 40
    rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    # design (float) for the LP: [orbit cols | linear cols]
    A = np.array([[hQ(orb[k], x) for k in okeys] + [x[i] for i in range(n)] for x in Xs], dtype=np.float64)
    b = np.array([max(x) for x in Xs], dtype=np.float64)
    # variables: c+ (N), c- (N), lin+ (n), lin- (n); minimize sum(c+ + c-) only (linear term is free/unpenalized)
    Aeq = np.hstack([A[:, :N], -A[:, :N], A[:, N:], -A[:, N:]])
    cost = np.concatenate([np.ones(N), np.ones(N), np.zeros(n), np.zeros(n)])
    res = linprog(cost, A_eq=Aeq, b_eq=b, bounds=(0, None), method="highs")
    if not res.success:
        print(f"n={n}: LP infeasible (max_{n} OUT of weight-2?) -- unexpected.", flush=True); return
    x = res.x; c = x[:N] - x[N:2*N]
    supp = [k for k in range(N) if abs(c[k]) > 1e-6]
    print(f"n={n}: LP min-L1 total mass = {res.fun:.4f}; support = {len(supp)} orbit blocks (of {N})", flush=True)
    # EXACT rational solve on the LP support (+ full linear), verify, then exhibit M / N
    cols = supp
    Ae = [[F(hQ(orb[okeys[k]], x)) for k in cols] + [F(x[i]) for i in range(n)] for x in Xs]
    be = [F(max(x)) for x in Xs]
    w = core.exact_solve(Ae, be)
    if w is None:
        print(f"  exact solve on LP support failed (support too small) -- widening not attempted here.", flush=True); return
    coeffs = {cols[j]: w[j] for j in range(len(cols)) if w[j] != 0}; lin = w[len(cols):]
    # verify at fresh rational points
    rng2 = np.random.default_rng(99); maxerr = F(0)
    for _ in range(300):
        xx = tuple(F(int(t), 6) for t in rng2.integers(-30, 31, size=n))
        lhs = F(max(xx)); rhs = sum(lin[i]*xx[i] for i in range(n))
        for k, ck in coeffs.items():
            s = 0
            for vs in orb[okeys[k]]:
                s += max(sum(F(v[j])*xx[j] for j in range(n)) for v in vs)
            rhs += ck*s
        maxerr = max(maxerr, abs(lhs-rhs))
    pos = {k: v for k, v in coeffs.items() if v > 0}; neg = {k: -v for k, v in coeffs.items() if v < 0}
    def dens(d): return [v.denominator for v in d.values()]
    alld = dens(pos)+dens(neg) or [1]
    L = 1
    for dd in alld: L = L*dd//gcd(L, dd)
    print(f"  EXACT minimal decomposition: {len(pos)} positive blocks (M), {len(neg)} negative blocks (N); "
          f"common denominator = {L}; verify err over 300 fresh pts = {maxerr}  ({'EXACT' if maxerr==0 else 'MISMATCH'})", flush=True)
    def show(name, d):
        print(f"    {name}: " + "; ".join(
            f"[{','.join(str(int(v*L)) for v in [c])}/{L}]x(orbit#{k}, {len(orb[okeys[k]])} members, rep {min(len(vs) for vs in [orb[okeys[k]][0]])}v)"
            for k, c in sorted(d.items())), flush=True)
    # concise: coefficient*L over L, and the vertex-count signature of each block's rep
    def sig(d):
        out = []
        for k, c in sorted(d.items(), key=lambda kv: -kv[1]):
            rep = orb[okeys[k]][0]
            out.append(f"{int(c*L)}/{L} x |rep|={len(rep)}v,orbit={len(orb[okeys[k]])}")
        return out
    print(f"    M (positive part, Delta+N=M): " + " | ".join(sig(pos)), flush=True)
    print(f"    N (negative part):            " + " | ".join(sig(neg)), flush=True)
    if os.environ.get("GEOM"):    # dump actual block geometry (rep vertex sets on the 2*Delta lattice)
        def dumpgeom(name, d):
            print(f"    -- {name} geometry --", flush=True)
            for k, c in sorted(d.items(), key=lambda kv: -kv[1]):
                rep = sorted(orb[okeys[k]][0])
                print(f"      coeff {int(c*L)}/{L}: rep verts = {rep}", flush=True)
        dumpgeom("M", pos); dumpgeom("N", neg)
    return {"n": n, "pos": len(pos), "neg": len(neg), "denom": L, "mass": res.fun}

print("ROUTE A: minimal virtual Minkowski decomposition (decomposition-polyhedron vertex)\n", flush=True)
rows = []
for n in (4, 5, 6):
    r = min_decomp(n)
    if r: rows.append(r)
    print("", flush=True)
print("SUMMARY (pattern to extend toward max_7):", flush=True)
for r in rows:
    print(f"  max_{r['n']}: M={r['pos']} pos blocks, N={r['neg']} neg blocks, common denom={r['denom']}, min L1 mass={r['mass']:.4f}", flush=True)
