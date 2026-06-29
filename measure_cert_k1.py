# Proof-of-concept for the SCALE-INVARIANT certificate route: a finite signed measure mu = sum a_j delta_{x_j}
# that annihilates EVERY zonotope support function (any generators, any scale) yet mu(max_n) != 0, proving
# max_n is not 1-hidden-layer WITHOUT any lattice -- i.e. scale-invariantly.
# h_Z(x) = sum_i (g_i . x)_+, generators g_i arbitrary, so mu kills all zonotopes  <=>  T(v):=sum_j a_j (v.x_j)_+ == 0  for all v.
# (v.x)_+ = (v.x + |v.x|)/2, so T==0  <=>  sum_j a_j x_j = 0  AND  S(v):=sum_j a_j |v.x_j| == 0 for all v.
# Solve the homogeneous system for a over a symmetric point set; among solutions seek mu(max_n) != 0.
import numpy as np, itertools
n = 3
# symmetric candidate support set: 0, +-e_i, +-(e_i+e_j) type, weight-2 simplex pts and negatives
pts = set()
base = [(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1),(1,-1,0),(0,1,-1),(1,0,-1),
        (1,1,1),(2,1,0),(2,0,1),(1,2,0),(0,2,1),(1,0,2),(0,1,2),(3,0,0),(0,3,0),(0,0,3)]
for p in base:
    for s in itertools.permutations(range(n)):
        q = tuple(p[s[i]] for i in range(n)); pts.add(q); pts.add(tuple(-x for x in q))
pts = sorted(pts); X = np.array(pts, dtype=float); J = len(pts)
print(f"support points: {J}")

rng = np.random.default_rng(0)
V = rng.standard_normal((1500, n))                 # test directions for S(v)=0
rows = [np.abs(V @ X[j]) for j in range(J)]        # |v.x_j| columns -> rows of M^T
M = np.array(rows).T                               # (1500 x J): row r = (|v_r . x_j|)_j
lin = X.T                                          # (n x J): sum_j a_j x_j = 0
Aeq = np.vstack([M, lin])                          # all homogeneous conditions on a
# null space via SVD
U, s, Vt = np.linalg.svd(Aeq, full_matrices=True)
tol = 1e-8 * s[0]
null = Vt[np.sum(s > tol):]                        # rows = null basis
print(f"conditions={Aeq.shape[0]}, null-space dim={null.shape[0]}")

maxvals = np.array([max(p) for p in pts])          # max_n at each support point
# does any null vector a have sum_j a_j max_n(x_j) != 0 ?
proj = null @ maxvals                              # component of max_n functional on each null basis vec
if null.shape[0] == 0:
    print("null space trivial -> need richer/larger point set")
else:
    a = null.T @ proj                              # the null vector maximizing |mu(max_n)|
    if np.linalg.norm(a) < 1e-12:
        print("max_n functional is ORTHOGONAL to null space -> no certificate from this support set")
    else:
        a = a / np.linalg.norm(a)
        # verify: residuals of zonotope-annihilation, and the max_n value
        res_lin = np.linalg.norm(X.T @ a)
        res_S = np.max(np.abs(M @ a))
        mu_max = float(maxvals @ a)
        print(f"candidate certificate: ||sum a_j x_j||={res_lin:.2e}, max_v|S(v)|={res_S:.2e}, mu(max_n)={mu_max:.4f}")
        if res_lin < 1e-6 and res_S < 1e-6 and abs(mu_max) > 1e-3:
            print("=> SCALE-INVARIANT k=1 certificate FOUND: kills all zonotopes, mu(max_n)!=0. Validates the measure route.")
        else:
            print("=> not clean; refine point set / directions.")
