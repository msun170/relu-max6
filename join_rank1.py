# Clean base theorem check: max_n is a SINGLE depth-2 block (join rank 1) iff n <= 4.
# Reason (rigorous): max_n = c*h_Q + linear means Delta_n = c*Q + translate, so Delta_n is a dilated join of
# two zonotopes. A zonotope that is a simplex is only a point or a segment, so conv(zono u zono) has <= 4
# vertices; Delta_n (n vertices) is a single join iff n <= 4. Here we confirm computationally: search the
# lattice family for a single block Q with max_n in span{h_Q, linear}.  Expect: found for n=4, none for n>=5.
import sys, itertools
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
P = 2147483647

def rank(cols, b=None):
    A = np.column_stack(cols).astype(np.int64) % P
    if b is not None: A = np.column_stack([A, b.astype(np.int64) % P])
    A = A.copy(); rows, ncols = A.shape; r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, rows) if A[i, c] % P != 0), -1)
        if piv < 0: continue
        A[[r, piv]] = A[[piv, r]]; A[r] = (A[r]*pow(int(A[r, c]), P-2, P)) % P
        for i in range(rows):
            if i != r and A[i, c] % P != 0: A[i] = (A[i]-A[i, c]*A[r]) % P
        r += 1
        if r == rows: break
    return r

def lattice_blocks(n):
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
    return blocks

for n in (3, 4, 5, 6, 7):
    rng = np.random.default_rng(1)
    X = rng.integers(-9, 10, size=(n+35, n)).astype(np.int64)
    lin = [X[:, i].astype(np.int64) for i in range(n)]
    b = X.max(axis=1).astype(np.int64)
    found = None
    for Q in lattice_blocks(n):
        col = core.orbit_column([Q], X).astype(np.int64)
        base = lin + [col]
        if rank(base, b) == rank(base):
            found = Q; break
    print(f"n={n}: max_n a single block? {'YES  ' + str(sorted(found)) if found else 'NO'}", flush=True)
