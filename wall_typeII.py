# TYPE-II EXTRACTION (user step 3): on the COMPLETE weight-2 family, recover the construction for max_5, max_6 and
# show it IS the unique non-braid-clean braid-producing wall circuit (the Type-II generator, dim 1). This gives a
# wall-level explanation of WHY max_6 is 2-layer: the unique element of ker(NB) whose braid jumps are nonzero is
# exactly the max_6 construction. For max_7, Type-II = 0 -> no such circuit exists.
import sys, itertools, time
from fractions import Fraction as F
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
from math import gcd
from functools import reduce
t0 = time.time(); P1 = 2147483647

def family(n):  # complete weight-2 P2 orbits (= verify_2layer.family)
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
    P1l = [frozenset([p]) for p in W2] + list(zonos)
    blocks = set(P1l)
    for i in range(len(P1l)):
        for j in range(i, len(P1l)): blocks.add(P1l[i] | P1l[j])
    return core.orbits_of(list(blocks), n)

def prim(d):
    g = reduce(gcd, [abs(x) for x in d]) or 1; d = tuple(x//g for x in d)
    return d if d >= tuple(-x for x in d) else tuple(-x for x in d)
def rank_modp(A, Pm=P1):
    A = (A % Pm).copy(); rows, ncols = A.shape; r = 0
    for c in range(ncols):
        sub = A[r:, c]; nz = cp.nonzero(sub)[0]
        if nz.size == 0: continue
        piv = r + int(nz[0].item())
        if piv != r:
            tmp = A[r].copy(); A[r] = A[piv]; A[piv] = tmp
        A[r] = (A[r] * pow(int(A[r, c].item()), Pm-2, Pm)) % Pm
        if r+1 < rows:
            f = A[r+1:, c]; A[r+1:] -= cp.outer(f, A[r]); A[r+1:] %= Pm
        r += 1
        if r == rows: break
    return r

def hQ(members, x):
    return sum(max(sum(v[k]*x[k] for k in range(len(x))) for v in vs) for vs in members)

def analyze(n):
    orb = family(n); okeys = list(orb); N = len(okeys)
    # exact construction: solve max_n = sum c_k OS_k + linear (rational)
    m = N + n + 25; rng = np.random.default_rng(3)
    Xs = [tuple(int(v) for v in rng.integers(-7, 8, size=n)) for _ in range(m)]
    A = [[F(hQ(orb[k], x)) for k in okeys] + [F(x[i]) for i in range(n)] for x in Xs]
    b = [F(max(x)) for x in Xs]
    w = core.exact_solve(A, b)
    feasible_str = "NONE (OUT)" if w is None else "exact construction found"
    print(f"\n=== n={n}: complete weight-2 family N={N} | {feasible_str}  [{time.time()-t0:.0f}s]", flush=True)
    # Type-II dim via wall matrices (rank is column-scaling invariant -> fast full-group-sum columns OK for the DIM)
    W2 = core.weight2_points(n); vi = {p: i for i, p in enumerate(W2)}; NPt = len(W2)
    perms = list(itertools.permutations(range(n))); Gn = len(perms)
    pa = cp.asarray(np.array([[vi[tuple(p[g[k]] for k in range(n))] for p in W2] for g in perms], dtype=np.int32))
    braid_t = tuple(sorted(prim(tuple((1 if k==0 else -1 if k==1 else 0) for k in range(n)))))
    rng2 = np.random.default_rng(50 + n); probes = []
    for _ in range(300):
        i, j = rng2.choice(n, size=2, replace=False)
        x = rng2.integers(-12, 0, size=n).astype(np.int64); tt = int(rng2.integers(3, 12)); x[i]=tt; x[j]=tt
        d = np.zeros(n, dtype=np.int64); d[i]=1; d[j]=-1; probes.append((x, d, True))
    nbd = list({prim(tuple(list(vs)[a][q]-list(vs)[b][q] for q in range(n)))
                for k in okeys for vs in orb[k] for a in range(len(vs)) for b in range(a+1, len(vs))
                if tuple(sorted(prim(tuple(list(vs)[a][q]-list(vs)[b][q] for q in range(n))))) != braid_t})
    rng2.shuffle(nbd)
    for _ in range(450):
        d = np.array(nbd[rng2.integers(len(nbd))], dtype=np.int64)
        x = rng2.integers(-6, 7, size=n).astype(np.int64); top = int(rng2.integers(0, n))
        x[top] = int(x.max()) + int(np.abs(d).sum()) + int(rng2.integers(3, 8)); probes.append((x, d, False))
    Yp = []; trip = []
    for (x, d, fl) in probes:
        i0 = len(Yp); Yp += [x+d, x-d, x]; trip.append((i0, i0+1, i0+2, fl))
    Y = np.array(Yp, dtype=np.int64); PY = cp.asarray((Y @ np.array(W2, dtype=np.int64).T).astype(cp.int64))
    reps = [next(iter(orb[k])) for k in okeys]
    def col(vs):
        Vc = cp.asarray([vi[p] for p in vs], dtype=cp.int32); k = len(vs); idx = pa[:, Vc]
        return PY[:, idx.reshape(-1)].reshape(len(Y), Gn, k).max(axis=2).sum(axis=1)
    cols = cp.stack([col(r) for r in reps], axis=1)
    Jr=[]; NBr=[]
    for (ip, im, i0, fl) in trip:
        row = cols[ip]+cols[im]-2*cols[i0]
        (Jr if fl else NBr).append(row)
    Jm = cp.stack(Jr); NBm = cp.stack(NBr)
    typeII = rank_modp(cp.concatenate([NBm, Jm], axis=0)) - rank_modp(NBm)
    print(f"   Type-II dim = {typeII}  ({'UNIQUE non-braid-clean braid circuit' if typeII==1 else 'no such circuit' if typeII==0 else str(typeII)+'-dim'})", flush=True)
    if w is not None:
        c = w[:N]; lin = w[N:]
        nz = [(okeys[k], c[k]) for k in range(N) if c[k] != 0]
        used_neg = any(v < 0 for _, v in nz)
        print(f"   construction: {len(nz)} nonzero orbit blocks, signed(neg coeffs)={used_neg}, linear part={[str(t) for t in lin]}", flush=True)
        # describe the support by (vertices, value-profile) so we can name the building blocks
        from collections import Counter
        shape = Counter()
        for key, coef in nz:
            members = orb[key]; rep = next(iter(members)); shape[(len(rep),)] += 1
        print(f"   support by #vertices of the join: {dict(shape)}", flush=True)
        # print the few distinct orbit types with coefficients (the explicit max_n wall circuit)
        for key, coef in sorted(nz, key=lambda kc: -abs(kc[1]))[:8]:
            rep = sorted(next(iter(orb[key])))
            print(f"      coeff {str(coef):>8}  x  orbit-sum of block {rep}", flush=True)
    print(f"   => {'the construction IS the unique Type-II wall circuit' if (w is not None and typeII==1) else 'no non-braid-clean braid circuit exists (Type-II=0) => OUT at weight-2'}", flush=True)

import os
for n in [int(x) for x in os.environ.get("TN","5,6,7").split(",")]:
    analyze(n)
print(f"\n[{time.time()-t0:.0f}s] done.", flush=True)
