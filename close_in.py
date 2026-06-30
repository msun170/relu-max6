# CLOSE IN on max_7 weight-4: work in the SYMMETRIC (S_7-invariant) subspace, which is far lower-dimensional than
# the full weight-4 join span, so it should SATURATE at a feasible rank. Build ORBIT-SUMMED weight-4 columns on GPU
# (precompute the S_7 action on the 210 weight-4 lattice points; orbit-sum = sum over g of max over the block's
# permuted vertices). QR floor (cusolver) of max_7 + control vs #orbit-blocks. If the floor SATURATES (rank stops
# growing) with f(max7) ~ 0 and control > 0 -> max_7 IN the symmetric weight-4 span = 2-layer (DECISIVE). If floor
# saturates positive -> OUT.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = 7; W = 4; t0 = time.time()
W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; W4set = set(W4); vidx = {p:i for i,p in enumerate(W4)}
NP = len(W4)
# S_7 action on the 210 lattice points: pa[g] = index array mapping point i -> permuted point's index
perms = list(itertools.permutations(range(n)))
pa = np.empty((len(perms), NP), dtype=np.int32)
for gi, g in enumerate(perms):
    for i, p in enumerate(W4):
        pa[gi, i] = vidx[tuple(p[g[k]] for k in range(n))]
pa = cp.asarray(pa)
print(f"{NP} pts, |S_7|={len(perms)}, perm action ready  [{time.time()-t0:.0f}s]", flush=True)

def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); gg = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//gg for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = list(gens); ng = len(gens)
rng = np.random.default_rng(1)
m = 16000
X = rng.integers(-15, 16, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.float32)); brand = cp.asarray(rng.standard_normal(m).astype(np.float32))
P = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(np.float32))    # m x 210
nbM = float(cp.linalg.norm(bmax)); nbR = float(cp.linalg.norm(brand))

def rand_block():
    # structured weight-4 join: two small zonotopes, union of vertex index sets
    def zono():
        p = W4[rng.integers(NP)]; verts = {p}; added = 0; tries = 0
        while added < 3 and len(verts) < 8 and tries < 40:
            tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
            for u in verts:
                w = shift(u, g)
                if w not in W4set: ok = False; break
                nv.add(w)
            if ok and nv - verts: verts |= nv; added += 1
        return verts
    u = zono() | zono()
    idx = [vidx[p] for p in u]
    return idx[:10]                                              # cap |V|<=10 for the one-shot gather

G = len(perms)
def orbit_col(Vidx):
    # sum over g of max over v in V of P[:, pa[g, v]]  -- ONE gather (no python loop)
    Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx)
    idx = pa[:, Vc]                                              # (G x k)
    return P[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)

# stream orbit columns, grow orthonormal basis via block QR, track floor
Q = cp.linalg.qr(cp.asarray(X.astype(np.float32)))[0]
print(f"  {'#blk':>6} {'rank':>6} {'f(max7)':>10} {'f(ctrl)':>9}  [{time.time()-t0:.0f}s]", flush=True)
batch = []; nblk = 0
for it in range(1, 14001):
    batch.append(orbit_col(rand_block())); nblk += 1
    if len(batch) == 300:
        C = cp.stack(batch, axis=1); batch = []
        R = C - Q @ (Q.T @ C); R = R - Q @ (Q.T @ R)
        Qr, Rr = cp.linalg.qr(R); keep = cp.abs(cp.diag(Rr)) > 1e-2
        if int(keep.sum()) > 0: Q = cp.concatenate([Q, Qr[:, keep]], axis=1)
        if nblk % 600 == 0:
            fM = float(cp.linalg.norm(bmax - Q@(Q.T@bmax)))/nbM; fR = float(cp.linalg.norm(brand - Q@(Q.T@brand)))/nbR
            print(f"  {nblk:>6} {Q.shape[1]-n:>6} {fM:>10.6f} {fR:>9.4f}  [{time.time()-t0:.0f}s]", flush=True)
        if Q.shape[1] > 0.7*m: print("  rank near m -> stop", flush=True); break
fM = float(cp.linalg.norm(bmax - Q@(Q.T@bmax)))/nbM; fR = float(cp.linalg.norm(brand - Q@(Q.T@brand)))/nbR
print(f"\nFINAL: blocks={nblk} rank={Q.shape[1]-n}  f(max7)={fM:.6f}  f(control)={fR:.4f}", flush=True)
print(f"  rank stabilizing + f(max7)->0 + control>0 => max_7 IN symmetric weight-4 (2-layer). [{time.time()-t0:.0f}s]", flush=True)
