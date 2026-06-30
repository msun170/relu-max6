# CHEAP, WELL-CONDITIONED weight-4 floor: stream random weight-4 join columns, maintain an ORTHONORMAL basis Q
# (block modified Gram-Schmidt + cusolver QR on the small residual block). Orthonormal => no ill-conditioning.
# Report floor(max7) and floor(control) vs rank captured. The control staying high certifies non-vacuity; the
# floor-vs-rank curve shows where max7's floor heads as the weight-4 span fills.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
n = 7; W = 4; t0 = time.time()
W4 = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; W4set = set(W4); vidx = {p:i for i,p in enumerate(W4)}
Wn = np.array(W4, dtype=np.int64)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = list(gens); ng = len(gens)
rng = np.random.default_rng(0)
m = 24000
X = rng.integers(-15, 16, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.float32))
brand = cp.asarray(rng.standard_normal(m).astype(np.float32))
P = cp.asarray((X @ Wn.T).astype(np.float32))      # m x 210
print(f"{len(W4)} pts, {ng} gens, m={m}  [{time.time()-t0:.0f}s]", flush=True)

def rand_zono_idx():
    p = W4[rng.integers(len(W4))]; verts = {p}; added = 0; tries = 0
    while added < 4 and len(verts) < 12 and tries < 60:
        tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok and nv - verts: verts |= nv; added += 1
    return [vidx[v] for v in verts]

def gen_block(B):
    C = cp.empty((m, B), dtype=cp.float32)
    for j in range(B):
        a = rand_zono_idx(); b = rand_zono_idx()
        C[:, j] = cp.maximum(P[:, a].max(axis=1), P[:, b].max(axis=1))
    return C

# orthonormal basis Q (m x k), start with linear functions
Q, _ = cp.linalg.qr(cp.asarray(X.astype(np.float32)))
bn = float(cp.linalg.norm(bmax)); rn = float(cp.linalg.norm(brand))
TOL = 1e-3; Bsz = 800
nb = float(cp.linalg.norm(bmax)); rnb = float(cp.linalg.norm(brand))
print(f"  {'rank':>6} {'f(max7)':>10} {'f(control)':>11}  [{time.time()-t0:.0f}s]", flush=True)
for rnd in range(1, 81):
    C = gen_block(Bsz)
    R = C - Q @ (Q.T @ C); R = R - Q @ (Q.T @ R)           # project off Q, reorthogonalize
    Qr, Rr = cp.linalg.qr(R)                                # orthonormalize residual block (cusolver)
    keep = cp.abs(cp.diag(Rr)) > TOL                        # rank-revealing: keep significant new directions
    newd = Qr[:, keep]
    if newd.shape[1] > 0: Q = cp.concatenate([Q, newd], axis=1)
    if rnd % 5 == 0 or rnd <= 2:
        rM = bmax - Q @ (Q.T @ bmax); rR = brand - Q @ (Q.T @ brand)
        fM = float(cp.linalg.norm(rM))/nb; fR = float(cp.linalg.norm(rR))/rnb
        print(f"  {Q.shape[1]-n:>6} {fM:>10.6f} {fR:>11.4f}  [{time.time()-t0:.0f}s]", flush=True)
    if Q.shape[1] > 0.85*m:
        print(f"  rank approaching m -> stop (vacuity).  [{time.time()-t0:.0f}s]", flush=True); break
rM = bmax - Q @ (Q.T @ bmax); rR = brand - Q @ (Q.T @ brand)
print(f"\nFINAL rank={Q.shape[1]-n}: f(max7)={float(cp.linalg.norm(rM))/nb:.6f}  f(control)={float(cp.linalg.norm(rR))/rnb:.4f}  [{time.time()-t0:.0f}s]", flush=True)
