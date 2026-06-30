# DEFINITIVE positive/negative control for the wall-circuit method, using the EXACT same complete weight-2 family
# as verify_2layer.py (which proved: max_4,5,6 IN weight-2, max_7 OUT). If the wall method is sound it MUST report:
#   n=5,6 weight-2 : J+NB FEASIBLE (construction exists -> its coeffs satisfy J c=jt, NB c=0)
#   n=7   weight-2 : J+NB INFEASIBLE (no weight-2 construction)
# If n=6 comes back INFEASIBLE here, the wall machinery is BUGGED (false negative) and the earlier wall-circuit
# OUT results are invalid. Uses orbit MEMBERS directly (verify_2layer's hQ), no full-group-sum, to match exactly.
import sys, itertools, time, os
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
t0 = time.time(); P1 = 2147483647

def family(n):  # EXACT copy of verify_2layer.family(n): complete weight-2 P2 orbits
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
        for j in range(i, len(P1)): blocks.add(P1[i] | P1[j])
    return core.orbits_of(list(blocks), n)

from math import gcd
from functools import reduce
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
def feasible(M, t):
    return rank_modp(M) == rank_modp(cp.concatenate([M, t.reshape(-1, 1)], axis=1))

def run(n, K_J=400, K_NB=600):
    orb = family(n); okeys = list(orb); N = len(okeys)
    braid1 = tuple((1 if k == 0 else -1 if k == 1 else 0) for k in range(n))
    braid_t = tuple(sorted(prim(braid1)))
    rng = np.random.default_rng(50 + n)
    probes = []
    for _ in range(K_J):
        i, j = rng.choice(n, size=2, replace=False)
        x = rng.integers(-12, 0, size=n).astype(np.int64); tt = int(rng.integers(3, 12)); x[i] = tt; x[j] = tt
        d = np.zeros(n, dtype=np.int64); d[i] = 1; d[j] = -1
        probes.append((x, d, True))
    # non-braid directions = block edge diffs that are not braid
    nbdirs = set()
    for k in okeys:
        for vs in orb[k]:
            V = list(vs)
            for a in range(len(V)):
                for b in range(a+1, len(V)):
                    d = prim(tuple(V[a][q]-V[b][q] for q in range(n)))
                    if tuple(sorted(d)) != braid_t: nbdirs.add(d)
    nbdirs = list(nbdirs); rng.shuffle(nbdirs)
    for _ in range(K_NB):
        d = np.array(nbdirs[rng.integers(len(nbdirs))], dtype=np.int64)
        x = rng.integers(-6, 7, size=n).astype(np.int64)
        top = int(rng.integers(0, n)); x[top] = int(x.max()) + int(np.abs(d).sum()) + int(rng.integers(3, 8))
        probes.append((x, d, False))
    # evaluate orbit-sum (sum over members of max over vertices) at probe points; second differences
    def osum_at(x):
        out = np.empty(N, dtype=np.int64)
        for ci, k in enumerate(okeys):
            tot = 0
            for vs in orb[k]:
                tot += max(int(sum(v[q]*x[q] for q in range(n))) for v in vs)
            out[ci] = tot
        return out
    Jrows = []; jt = []; NBrows = []
    for (x, d, fl) in probes:
        a = osum_at(x + d); b = osum_at(x - d); c = osum_at(x)
        row = a + b - 2*c
        tgt = int(max(x+d) + max(x-d) - 2*max(x))
        if fl: Jrows.append(row); jt.append(tgt)
        else:
            assert tgt == 0, f"NB probe nonzero max_n jump {tgt}"; NBrows.append(row)
    Jm = cp.asarray(np.stack(Jrows)); NBm = cp.asarray(np.stack(NBrows)); jt = cp.asarray(np.array(jt, dtype=np.int64))
    M = cp.concatenate([NBm, Jm], axis=0); tt = cp.concatenate([cp.zeros(NBm.shape[0], dtype=cp.int64), jt])
    rNB = rank_modp(NBm); rM = rank_modp(M)
    fJ = feasible(Jm, jt); fJNB = feasible(M, tt)
    c0 = cp.asarray(np.random.default_rng(99).integers(-4, 5, size=N).astype(np.int64))
    fctrl = feasible(M, M @ c0)
    frand = feasible(M, cp.asarray(np.random.default_rng(7).integers(-50, 51, size=M.shape[0]).astype(np.int64)))
    expect = "FEASIBLE" if n <= 6 else "INFEASIBLE"
    got = "FEASIBLE" if fJNB else "INFEASIBLE"
    print(f"n={n}: complete wt-2 family N={N} | J feasible={fJ} | J+NB {got} (EXPECT {expect}) | "
          f"Type-II dim={rM-rNB} | controls in-fam={fctrl} rand={frand} | {'OK' if got==expect else '*** METHOD BUG ***'}  [{time.time()-t0:.0f}s]", flush=True)

for n in (int(x) for x in os.environ.get("PC_NS", "5,6,7").split(",")):
    run(n)
