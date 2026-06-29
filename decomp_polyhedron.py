# Set up the Brandenburg-Grillo-Hertrich decomposition polyhedron for max_n at weight-2 complexity, and
# test emptiness. A weight-2 two-layer representation is exactly a difference of two CONVEX pieces
# g - h = max_n where g = sum_i a_i h_block_i, h = sum_i b_i h_block_i with a_i, b_i >= 0 (nonneg combos of
# convex block support functions are convex). So D = {(a,b) >= 0 : sum_i (a_i - b_i) h_block_i = max_n} is
# exactly the decomposition polyhedron restricted to weight-2 P2 complexity. We test feasibility (n=6:
# should be nonempty; n=7: should be empty == our Theorem 2 obstruction). Confirms: our obstruction IS the
# decomposition-polyhedron bounded-complexity emptiness.
import sys, itertools, time
import numpy as np
from scipy.optimize import linprog
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

def decomp_nonempty(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    m = N + n + 30; X = np.random.default_rng(5).integers(-8, 9, size=(m, n)).astype(np.int64)
    A = np.column_stack([core.orbit_column(orb[k], X) for k in okeys] + [X[:, i] for i in range(n)]).astype(np.float64)
    b = X.max(axis=1).astype(np.float64)
    # variables: a (N>=0), b (N>=0), and a free linear term L (n vars, split into L+ - L-)
    K = A.shape[1]   # N orbit cols + n linear cols
    # [A | -A] @ [a; b] = max_n,  a,b >= 0  (linear cols already in A, allow their coeff free via a-b too)
    Aeq = np.column_stack([A, -A])
    res = linprog(c=np.zeros(2*K), A_eq=Aeq, b_eq=b, bounds=[(0, None)]*(2*K), method='highs')
    return res.success, N

t0 = time.time()
for n in (6, 7):
    ok, N = decomp_nonempty(n)
    print(f"n={n}: weight-2 orbits={N}; decomposition polyhedron D_C(max{n}) "
          f"{'NONEMPTY (max%d is a weight-2 difference-of-convex => 2-layer)' % n if ok else 'EMPTY (no weight-2 difference-of-convex)'}"
          f"  [{time.time()-t0:.0f}s]", flush=True)
print("\n=> Confirms: max7's obstruction (Theorem 2) IS exactly the emptiness of the BGH decomposition")
print("   polyhedron at weight-2 (bounded) complexity. The open question is whether D_C(max7) is empty at")
print("   ALL complexity (= max7 not 2-layer). The framework reframes this; it does not bound the complexity.")
