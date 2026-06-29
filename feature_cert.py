# Test the self-similar reduction: a measure mu annihilates ALL 2-layer functions  <=>  Sum_i mu_i (b_i . a)_+ = 0
# for all a, where b_i is the V1-FEATURE vector of support point p_i (its values under a basis of the 1-layer
# functions V1). I.e. the lifted measure Sum mu_i delta_{b_i} is odd-balanced in feature space. By chambers:
# for every realizable positive-set S of the features, Sum_{i in S} mu_i b_i = 0. A k=2 certificate is such a
# mu with mu(max_n) != 0. For n where max_n IS 2-layer (n<=6), NO certificate should exist (consistency check).
import sys, itertools
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
P = 2147483647

def modp_pivots(M):  # column indices forming a basis of the column space of integer matrix M (mod P)
    A = (M % P).astype(np.int64).copy(); rows, cols = A.shape; r = 0; piv = []
    for c in range(cols):
        nz = [i for i in range(r, rows) if A[i, c] % P != 0]
        if not nz: continue
        i0 = nz[0]; A[[r, i0]] = A[[i0, r]]
        A[r] = (A[r] * pow(int(A[r, c]), P-2, P)) % P
        for i in range(rows):
            if i != r and A[i, c] % P != 0: A[i] = (A[i] - A[i, c]*A[r]) % P
        piv.append(c); r += 1
        if r == rows: break
    return piv

def test(n, rad=2, nv=4000, nchamber=60000, seed=0):
    pts = [p for p in itertools.product(range(-rad, rad+1), repeat=n) if 1 <= sum(abs(c) for c in p) <= 2]
    m = len(pts); Xp = np.array(pts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    # V1 value vectors w_v = ((v.p_i)_+)_i for many integer v
    Vs = rng.integers(-5, 6, size=(nv, n))
    Wcols = np.maximum(Xp @ Vs.T, 0).astype(np.int64)     # m x nv, columns are value vectors
    piv = modp_pivots(Wcols)
    B = Wcols[:, piv].astype(np.int64)                    # m x d basis of the V1-value space
    d = B.shape[1]
    feats = [B[i, :] for i in range(m)]                   # feature vectors b_i in R^d
    # enumerate realizable positive-sets S of the features
    patterns = set()
    for _ in range(nchamber):
        a = rng.integers(-7, 8, size=d)
        S = tuple(int(np.dot(feats[i], a) > 0) for i in range(m))
        patterns.add(S)
    # conditions: for each S, sum_{i in S} mu_i b_i = 0  (d rows); pin mu(max_n)=1
    rows = []; rhs = []
    for S in patterns:
        for k in range(d):
            rows.append([F(int(S[i]) * int(B[i, k])) for i in range(m)]); rhs.append(F(0))
    rows.append([F(max(pts[i])) for i in range(m)]); rhs.append(F(1))
    mu = core.exact_solve(rows, rhs)
    print(f"n={n}: support m={m}, dim V1={d}, chambers={len(patterns)} -> "
          f"{'NO certificate (consistent: max_%d is 2-layer)' % n if mu is None else 'CERTIFICATE FOUND (max_%d NOT 2-layer?!)' % n}", flush=True)
    return mu

for n in (3, 4):
    test(n)
