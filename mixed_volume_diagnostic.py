# ROUTE (b) probe #2: MIXED-VOLUME / ALEKSANDROV-FENCHEL diagnostic (the last flagged max_7 tool).
#
# HONEST SCOPE. The full test on the orbit-summed identity Delta+N=M is INFEASIBLE (M,N are Minkowski sums of
# hundreds of orbit members -> astronomically many vertices, no hull volume) and first-order TOOTHLESS (the
# identity holds exactly as polytopes, so all LINEAR mixed-volume relations V(.,Delta^{d-1}) are automatically
# satisfied). Teeth, if any, are QUADRATIC (AF) and need N's geometry (the infeasible part). So we run the
# feasible, genuine proxy: the mixed-volume SIGNATURE of each decomposition block Q vs the simplex Delta, and the
# AF deficit  V(Q,Delta,C..)^2 - V(Q,Q,C..) V(Delta,Delta,C..) >= 0.  Question: are the blocks AF-EXTREMAL
# (zonotope-like rigidity) or do they carry a signature Delta cannot match? A clean extremality/rigidity would be
# a lead; generic slack = the tool has no teeth here (report honestly).
import sys, itertools, os
from math import factorial
from fractions import Fraction as F
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
from verify_2layer import family, hQ

def hyper_basis(n):     # orthonormal basis of {sum x_i = 0} in R^n  (columns), shape n x (n-1)
    A = np.eye(n) - np.ones((n, n))/n
    U, S, _ = np.linalg.svd(A)
    return U[:, :n-1]

def vol(points):        # d-dim volume of the convex hull of points (R^d); 0 if degenerate
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < P.shape[1] + 1: return 0.0
    try: return float(ConvexHull(P).volume)
    except Exception: return 0.0

def minksum(A, B):      # Minkowski sum of two vertex arrays, hull-reduced
    S = (A[:, None, :] + B[None, :, :]).reshape(-1, A.shape[1])
    try: return S[ConvexHull(S).vertices]
    except Exception: return S

def mixed_volume(bodies):   # V(K_1,...,K_d), bodies = list of d vertex-arrays in R^d (polarization)
    d = len(bodies); total = 0.0
    for r in range(1, d+1):
        for S in itertools.combinations(range(d), r):
            M = bodies[S[0]]
            for i in S[1:]: M = minksum(M, bodies[i])
            total += ((-1)**(d-r)) * vol(M)
    return total / factorial(d)

def min_decomp_blocks(n):   # reuse route A: minimal virtual decomposition -> (pos blocks, neg blocks) as vertex sets
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 40; rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    A = np.array([[hQ(orb[k], x) for k in okeys] + [x[i] for i in range(n)] for x in Xs], dtype=np.float64)
    b = np.array([max(x) for x in Xs], dtype=np.float64)
    Aeq = np.hstack([A[:, :N], -A[:, :N], A[:, N:], -A[:, N:]])
    cost = np.concatenate([np.ones(N), np.ones(N), np.zeros(n), np.zeros(n)])
    res = linprog(cost, A_eq=Aeq, b_eq=b, bounds=(0, None), method="highs")
    c = res.x[:N] - res.x[N:2*N]; supp = [k for k in range(N) if abs(c[k]) > 1e-6]
    pos = [(okeys[k], c[k]) for k in supp if c[k] > 0]; neg = [(okeys[k], -c[k]) for k in supp if c[k] < 0]
    return orb, pos, neg

for n in [int(x) for x in os.environ.get("NS", "5,6").split(",")]:
    d = n - 1; B = hyper_basis(n)
    print(f"\n===== n={n} (ambient dim d={d}) =====", flush=True)
    orb, pos, neg = min_decomp_blocks(n)
    # simplex Delta = conv(e_i), projected
    Delta = (np.eye(n) @ B)
    volD = vol(Delta)
    # reference zonotope (a generic weight-2 segment sum) and a generic simplex-free body for calibration
    print(f"  vol(Delta) = {volD:.5f}", flush=True)
    def block_pts(key):     # ONE orbit representative's vertex set, projected
        rep = orb[key][0]
        return np.array([[float(v[i]) for i in range(n)] for v in rep]) @ B
    def af_report(tag, blocks):
        print(f"  -- {tag} blocks: mixed-volume signature vs Delta + AF deficit --", flush=True)
        for key, coeff in sorted(blocks, key=lambda kv: -kv[1]):
            Q = block_pts(key)
            # signature V(Q^k, Delta^{d-k}) for k=0..d
            sig = [mixed_volume([Q]*k + [Delta]*(d-k)) for k in range(d+1)]
            # AF: V(Q,Delta,Delta^{d-2})^2 >= V(Q,Q,Delta^{d-2}) V(Delta,Delta,Delta^{d-2})
            vQD = mixed_volume([Q, Delta] + [Delta]*(d-2))
            vQQ = mixed_volume([Q, Q] + [Delta]*(d-2))
            vDD = mixed_volume([Delta, Delta] + [Delta]*(d-2))
            deficit = vQD*vQD - vQQ*vDD
            rel = deficit/(abs(vQD*vQD)+1e-12)
            sigs = ", ".join(f"{s:.4f}" for s in sig)
            print(f"    coeff {coeff:+.4f} |rep|={len(orb[key][0])}v: V(Q^k,D^(d-k))=[{sigs}]  AF deficit={deficit:.4e} (rel {rel:+.3e}) {'~EXTREMAL' if abs(rel)<1e-6 else ''}", flush=True)
    af_report("M (positive)", pos)
    af_report("N (negative)", neg)
    # calibration: AF deficit for Delta vs a generic centrally-symmetric zonotope (should be strictly positive)
    rng = np.random.default_rng(0)
    seg = [np.array([[0]*n, list(np.eye(n, dtype=int)[i])]) for i in range(n)]
    Z = np.zeros((1, n))
    for i in range(min(n, 4)):
        Z = (Z[:, None, :] + seg[i][None, :, :]).reshape(-1, n)
    Zp = Z @ B
    vZD = mixed_volume([Zp, Delta] + [Delta]*(d-2)); vZZ = mixed_volume([Zp, Zp] + [Delta]*(d-2))
    vDD = mixed_volume([Delta, Delta] + [Delta]*(d-2))
    calib = vZD*vZD - vZZ*vDD
    print(f"  CALIBRATION (generic zonotope vs Delta): AF deficit = {calib:.4e} (should be >= 0)  [{'ok' if calib>-1e-9 else 'NEG?!'}]", flush=True)

print("\nAssessment: look for blocks with AF deficit ~ 0 (extremal/rigid). Generic slack everywhere => the mixed-", flush=True)
print("volume tool has no teeth on this problem (as anticipated); a systematic ~0 deficit family would be a lead.", flush=True)
