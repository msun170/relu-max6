# Fast exact (mod p) min-block-width (join rank) for max_n in weight-2. Vectorized k=2 test: after reducing
# against span{lin, q1}, max_n is representable with q2 iff its residual rmax is PARALLEL to q2's residual.
import sys, itertools, time
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
P = 2147483647

def echelon_reduce(basis_cols, targets):
    # row-reduce the augmented system: return residuals of each target column after projecting out basis.
    # basis_cols: list of m-vectors; targets: m x T matrix. Work mod P with a pivot list.
    m = targets.shape[0]
    B = np.array(basis_cols, dtype=np.int64).T % P if basis_cols else np.zeros((m, 0), np.int64)
    T = targets.copy() % P
    pivrows = []
    for c in range(B.shape[1]):
        col = B[:, c].copy()
        for (pr, pc, prow) in pivrows:  # reduce col against existing pivots
            f = col[pr]
            if f: col = (col - f*prow) % P
        nz = np.nonzero(col % P)[0]
        if len(nz) == 0: continue
        pr = int(nz[0]); inv = pow(int(col[pr]), P-2, P); prow = (col*inv) % P
        pivrows.append((pr, c, prow))
    # reduce targets against the pivot rows
    for (pr, pc, prow) in pivrows:
        f = T[pr, :].copy()
        T = (T - np.outer(prow, f)) % P
    return T  # residuals (m x T); zero column => target in span(basis)

def blocks_of(n):
    W2 = core.weight2_points(n); W2set = set(W2)
    gens = sorted({d for p in W2 for q in W2 if p != q
                   for d in [tuple(p[k]-q[k] for k in range(n))] if d > tuple(-x for x in d)})
    zon = set()
    def dfs(v, gi):
        zon.add(frozenset(v))
        for j in range(gi+1, len(gens)):
            g = gens[j]; nw = set(); ok = True
            for u in v:
                w = tuple(u[k]+g[k] for k in range(n))
                if w not in W2set: ok = False; break
                nw.add(w)
            if ok: dfs(v | nw, j)
    for p in W2: dfs({p}, -1)
    P1 = list(zon); bl = set()
    for i in range(len(P1)):
        for j in range(i, len(P1)): bl.add(P1[i] | P1[j])
    return list(bl)

def parallel_mask(rmax, R):
    # columns j with rmax in span{R[:,j]}, i.e. R[:,j] = beta*rmax with beta != 0 (R[:,j]=0 does NOT count)
    nz = np.nonzero(rmax % P)[0]
    if len(nz) == 0: return np.ones(R.shape[1], bool)  # rmax=0 => already representable
    i0 = int(nz[0]); a_inv = pow(int(rmax[i0]), P-2, P)
    alpha = (R[i0, :] * a_inv) % P                      # beta candidate per column
    pred = (np.outer(rmax, alpha)) % P
    return np.all((R - pred) % P == 0, axis=0) & (alpha % P != 0)

def join_rank(n, tlimit=120):
    t0 = time.time()
    bl = blocks_of(n); Nb = len(bl)
    orb = core.orbits_of(bl, n); reps = [orb[k][0] for k in orb]
    m = 80
    X = np.random.default_rng(0).integers(-7, 8, size=(m, n)).astype(np.int64)
    lin = [X[:, i].astype(np.int64) for i in range(n)]
    bvec = (X.max(axis=1).astype(np.int64) % P)
    Bcols = np.column_stack([core.orbit_column([Q], X).astype(np.int64) for Q in bl]) % P
    repcols = [core.orbit_column([Q], X).astype(np.int64) % P for Q in reps]
    # k=1: bvec in span{lin, q} for some block q
    res1 = echelon_reduce(lin, np.column_stack([bvec]))
    if np.all(res1 % P == 0): return 0, Nb, time.time()-t0   # linear (shouldn't happen)
    # reduce bvec and all blocks against lin
    redlin = echelon_reduce(lin, np.column_stack([bvec, Bcols]))
    rb = redlin[:, 0]; Bres = redlin[:, 1:]
    if np.any(parallel_mask(rb, Bres)): return 1, Nb, time.time()-t0
    # k=2: for each rep q1, reduce against {lin,q1}, test parallel
    for rc in repcols:
        red = echelon_reduce(lin + [rc], np.column_stack([bvec, Bcols]))
        rmax = red[:, 0]; R = red[:, 1:]
        if np.any(parallel_mask(rmax, R)): return 2, Nb, time.time()-t0
        if time.time()-t0 > tlimit: return ">2(timeout)", Nb, time.time()-t0
    return ">2", Nb, time.time()-t0

for n in (3, 4, 5, 6):
    k, nb, t = join_rank(n)
    print(f"n={n}: join rank (min blocks, weight-2) = {k}   (#blocks={nb})  [{t:.0f}s]", flush=True)
