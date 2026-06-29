# Positive controls for the real-weight test: prove the pipeline is NOT trivially always-OUT.
#  (1) n=6 roots-only must be IN (max6 IS 2-layer)  -- detects IN correctly.
#  (2) n=7 with the actual weight-2 generators that DO contain a smaller max (max5 embedded) sanity.
#  (3) flip test: a deliberately rich family for n=5 must be IN.
import sys, itertools
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
P = 2147483647

def modp_rank(M, extra=None):
    A = (M % P).astype(np.int64).copy()
    if extra is not None: A = np.column_stack([A, (extra % P).astype(np.int64)])
    rows, cols = A.shape; r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c] % P != 0), -1)
        if piv < 0: continue
        A[[r, piv]] = A[[piv, r]]; A[r] = (A[r]*pow(int(A[r, c]), P-2, P)) % P
        for i in range(rows):
            if i != r and A[i, c] % P != 0: A[i] = (A[i]-A[i, c]*A[r]) % P
        r += 1
        if r == rows: break
    return r

def shift(p, g, n): return tuple(p[k]+g[k] for k in range(n))

def root_family(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q
                   for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zonos = set()
    def dfs(verts, gi):
        if len(verts) >= 2: zonos.add(frozenset(verts))
        for j in range(gi+1, len(gens)):
            g = gens[j]; new = set(); ok = True
            for v in verts:
                wv = shift(v, g, n)
                if wv not in W2set: ok = False; break
                new.add(wv)
            if ok: dfs(verts | new, j)
    for p in W2: dfs({p}, -1)
    RZ = [frozenset([p]) for p in W2] + list(zonos)
    blocks = set(RZ)
    for i in range(len(RZ)):
        for j in range(i, len(RZ)):
            blocks.add(RZ[i] | RZ[j])
    return core.orbits_of(list(blocks), n)

def verdict(n, label):
    orbits = root_family(n); okeys = list(orbits); N = len(okeys)
    m = N + n + 40
    X = np.random.default_rng(7).integers(-9, 10, size=(m, n)).astype(np.int64)
    cols = [core.orbit_column(orbits[k], X) for k in okeys]
    A = np.column_stack(cols + [X[:, i] for i in range(n)]).astype(np.int64)
    b = X.max(axis=1).astype(np.int64)
    inspan = modp_rank(A, b) == modp_rank(A)
    print(f"  [{label}] n={n} orbits={N}  max{n} {'IN' if inspan else 'OUT'}", flush=True)
    return inspan

print("positive controls (same pipeline as realweight_test):", flush=True)
verdict(5, "roots-only n=5 (expect IN)")
verdict(6, "roots-only n=6 (expect IN)")
verdict(7, "roots-only n=7 (expect OUT)")
print("=> if n=5,6 IN and n=7 OUT, the pipeline detects IN correctly; the real-weight OUTs are meaningful.", flush=True)
