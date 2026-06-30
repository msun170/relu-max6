# Degree-2 intersection numbers on the permutohedral variety X(B_n), via mixed volumes.
#
# Divisors / generalized permutohedra <-> polytopes in the braid hyperplane (dim d = n-1).
# Top intersection of d divisors = mixed volume V(P_1,...,P_d).  We compute the symmetric
# bilinear "Hodge-Riemann degree-1" form  B(D,D') = V(D, D', A^{d-2})  with A the permutohedron
# (the canonical ample class).  Alexandrov-Fenchel / Hodge-Riemann => B has signature (1, N-1)
# on nef classes (one positive eigenvalue).  We:
#   (1) build the mixed-volume calculator and validate vol/MV,
#   (2) assemble the Gram matrix of B over {simplex (=max_n=U_{1,n} matroid polytope)} u {join blocks},
#   (3) check the Lorentzian signature numerically (sanity: AHK holds for our objects),
#   (4) test whether D_{max_n}'s degree-2 signature is distinguishable from the join cone.
import sys, itertools, numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

def project_basis(n):
    # orthonormal basis of the hyperplane sum=0 in R^n  ->  R^{n-1}
    B = np.linalg.svd(np.eye(n) - np.ones((n, n)) / n)[0][:, :n - 1]
    return B  # columns span sum=0

def verts_of(block, n, Bproj):
    P = np.array(sorted(block), float)        # integer points, on some hyperplane sum=const
    P = P - P.mean(axis=0, keepdims=True)      # center (kills the constant component)
    return P @ Bproj                           # -> R^{n-1}

def minkowski(Vs):
    # Minkowski sum of vertex arrays -> vertex set of the sum (convex hull of pairwise sums)
    acc = Vs[0]
    for W in Vs[1:]:
        S = (acc[:, None, :] + W[None, :, :]).reshape(-1, acc.shape[1])
        if S.shape[0] > 8:
            try:
                S = S[ConvexHull(S).vertices]
            except Exception:
                pass
        acc = S
    return acc

def vol(V):
    if V.shape[0] <= V.shape[1]:
        return 0.0
    try:
        return ConvexHull(V).volume
    except Exception:
        return 0.0

def perm_vertices(n, Bproj):
    pts = np.array(list(itertools.permutations(range(n))), float)
    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts @ Bproj

def Bform(D, A, d, npts=None):
    # B(D,D) = V(D,D,A^{d-2}); fit vol(s*D + A) as a poly of degree d in s, take coeff of s^2.
    # vol(sD+A) = sum_{j} C(d,j) V(D[j],A[d-j]) s^j  =>  [s^2] = C(d,2) V(D,D,A^{d-2}).
    npts = npts or (d + 2)
    svals = np.linspace(0.0, 2.0, npts)
    y = np.array([vol(minkowski([s * D, A]) if s > 0 else A) for s in svals])
    coef = np.polyfit(svals, y, d)            # highest power first
    c2 = coef[d - 2]                          # coefficient of s^2
    from math import comb
    return c2 / comb(d, 2)

def Bbilinear(D1, D2, A, d):
    # polarize: V(D1,D2,A^{d-2}) = (B(D1+D2)-B(D1)-B(D2))/2
    s = minkowski([D1, D2])
    return (Bform(s, A, d) - Bform(D1, A, d) - Bform(D2, A, d)) / 2.0

def af_defect(D, A, d):
    # Alexandrov-Fenchel / Hodge index defect: B(A,D)^2 - B(A,A) B(D,D) >= 0 for all nef D,
    # =0 iff D proportional to A.  A scale-invariant version normalizes by B(A,A)B(D,D).
    bAA = Bform(A, A, d); bDD = Bform(D, D, d); bAD = Bbilinear(A, D, A, d)
    raw = bAD * bAD - bAA * bDD
    return raw / (bAA * bDD) if bAA * bDD > 0 else float("nan")

if __name__ == "__main__":
    import time
    for n in (4, 5, 6):
        d = n - 1
        Bp = project_basis(n)
        A = perm_vertices(n, Bp)
        # simplex = max_n divisor = matroid polytope of U_{1,n}; use weight-2 vertices conv{2 e_i}
        simplex = verts_of(frozenset(tuple(2 if k == i else 0 for k in range(n)) for i in range(n)), n, Bp)
        t0 = time.time()
        print(f"\n=== n={n} (d={d}) ===")
        print(f"  vol(permutohedron A) = {vol(A):.4f}   (exact = n^(n-2)*sqrt(n) = {n**(n-2)*np.sqrt(n):.4f})")
        nb = 6 if n <= 5 else 3
        blocks = core.building_blocks(n, R=2, cap=2 * n, join=2)
        sample = blocks[:nb]
        Ds = [simplex] + [verts_of(b, n, Bp) for b in sample]
        N = len(Ds)
        G = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                G[i, j] = G[j, i] = (Bform(Ds[i], A, d) if i == j else Bbilinear(Ds[i], Ds[j], A, d))
        ev = np.linalg.eigvalsh(G)
        pos = int((ev > 1e-6).sum()); neg = int((ev < -1e-6).sum())
        print(f"  HR Gram over {{max_n}} u {nb} join blocks: signature (+{pos}, -{neg}, 0:{N-pos-neg})  Lorentzian={pos==1}")
        # AF defect: scale-invariant Hodge-index gap, max_n vs the join blocks
        dmax = af_defect(simplex, A, d)
        djoin = [af_defect(Ds[j], A, d) for j in range(1, N)]
        print(f"  AF/Hodge-index defect  max_n = {dmax:.4f}   joins = {[round(x,4) for x in djoin]}")
        print(f"  ({time.time()-t0:.1f}s)")
