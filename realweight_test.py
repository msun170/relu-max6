# THE never-done test: real-weight (non-root) generators. The whole lower-bound program restricts to ROOT
# generators (weight-2/3 lattice). The lattice normal-form conjecture says non-root (real) generators cannot
# help represent max_n in 2 layers. We have NEVER tested it. Method: take the canonical root weight-2 family
# (max7 OUT), then AUGMENT it with blocks carrying NON-root edge directions (complexity-2 lattice edges and
# genuinely off-lattice edges): the segments [p, p+d] and their joins with the root zonotopes. Ask whether
# max7 enters the span.
#   if IN  : lattice normal form is FALSE; max7 IS 2-layer with real weights (separation pushed past 7).
#   if OUT across many seeds: first real-weight evidence that max7 needs 3 layers and lattice is WLOG.
import sys, itertools, time
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core

n = 7
W2 = core.weight2_points(n); W2set = set(W2)
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

def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

# root weight-2 zonotopes (the canonical family), staying in the 2-Delta lattice
def root_zonotopes():
    gens = sorted({d for p in W2 for q in W2 if p != q
                   for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zonos = set()
    def dfs(verts, gi):
        if len(verts) >= 2: zonos.add(frozenset(verts))
        for j in range(gi+1, len(gens)):
            g = gens[j]; new = set(); ok = True
            for v in verts:
                wv = shift(v, g)
                if wv not in W2set: ok = False; break
                new.add(wv)
            if ok: dfs(verts | new, j)
    for p in W2: dfs({p}, -1)
    return [frozenset([p]) for p in W2] + list(zonos)

RZ = root_zonotopes()

def base_family():
    blocks = set(RZ)
    for i in range(len(RZ)):
        for j in range(i, len(RZ)):
            blocks.add(RZ[i] | RZ[j])
    return blocks

def augmented_family(seeds, cap=14):
    blocks = base_family()
    # non-root segments anchored at each weight-2 point, plus the S_n-orbit (canon handles symmetry)
    segs = []
    for d in seeds:
        for p in W2:
            segs.append(frozenset([tuple(p), shift(p, d)]))
    blocks.update(segs)
    # joins: each non-root segment with every root zonotope (and with each other), capped
    pool = RZ + segs
    for s in segs:
        for b in pool:
            u = s | b
            if len(u) <= cap: blocks.add(u)
    return core.orbits_of(list(blocks), n)

def verdict(orbits, label, t0):
    okeys = list(orbits); N = len(okeys)
    m = N + n + 40
    X = np.random.default_rng(7).integers(-9, 10, size=(m, n)).astype(np.int64)
    cols = [core.orbit_column(orbits[k], X) for k in okeys]
    A = np.column_stack(cols + [X[:, i] for i in range(n)]).astype(np.int64)
    b = X.max(axis=1).astype(np.int64)
    inspan = modp_rank(A, b) == modp_rank(A)
    print(f"  [{label}] orbits={N}  max7 {'IN (representable!)' if inspan else 'OUT'}  [{time.time()-t0:.0f}s]", flush=True)
    return inspan

t0 = time.time()
print("real-weight generator test for max7 (n=7)", flush=True)
verdict(core.orbits_of(list(base_family()), n), "roots only (control)", t0)
verdict(augmented_family([(1,1,-1,-1,0,0,0)]), "+ edge (1,1,-1,-1,0,0,0)  [cx-2 lattice]", t0)
verdict(augmented_family([(2,-1,-1,0,0,0,0)]), "+ edge (2,-1,-1,0,0,0,0)  [cx-2 lattice]", t0)
verdict(augmented_family([(2,-1,0,0,0,0,0)]),  "+ edge (2,-1,0,0,0,0,0)   [off-lattice]", t0)
verdict(augmented_family([(3,-1,-1,-1,0,0,0)]),"+ edge (3,-1,-1,-1,0,0,0)  [cx-3 lattice]", t0)
verdict(augmented_family([(1,1,-1,-1,0,0,0),(2,-1,-1,0,0,0,0),(2,-1,0,0,0,0,0)]), "+ all three together", t0)
print(f"done [{time.time()-t0:.0f}s]", flush=True)
