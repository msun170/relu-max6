# EXP 2+3: is the best weight-4 approximant of max_7 SPARSE? Build a pool of ORBIT-SUMMED structured weight-4
# blocks (with metadata: #verts, type, root-ness), run ORTHOGONAL MATCHING PURSUIT to greedily fit max_7, and
# report residual vs #blocks selected + the dominant blocks' types/coefficients. Sparse + patterned => candidate
# ansatz (-> exact rational solve next). Diffuse => no clean construction.
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
perms = list(itertools.permutations(range(n))); G = len(perms)
pa = np.empty((G, NP), dtype=np.int32)
for gi, g in enumerate(perms):
    for i, p in enumerate(W4): pa[gi, i] = vidx[tuple(p[g[k]] for k in range(n))]
pa = cp.asarray(pa)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
roots = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a!=b}
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = list(gens); ng = len(gens)
rng = np.random.default_rng(2)
m = 9000
X = rng.integers(-14, 15, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.float32))
P = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(np.float32))
print(f"{NP} pts, perm action ready, m={m}  [{time.time()-t0:.0f}s]", flush=True)

def zono():  # returns (vertex set, generators used, allroot?)
    p = W4[rng.integers(NP)]; verts = {p}; used = []; added = 0; tries = 0
    while added < 3 and len(verts) < 8 and tries < 40:
        tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok and nv - verts: verts |= nv; used.append(g); added += 1
    allroot = all((g in roots or tuple(-x for x in g) in roots) for g in used)
    return verts, used, allroot

def orbit_col(Vidx):
    Vc = cp.asarray(Vidx, dtype=cp.int32); k = len(Vidx)
    idx = pa[:, Vc]
    return P[:, idx.reshape(-1)].reshape(m, G, k).max(axis=2).sum(axis=1)

# build pool of distinct orbit-summed blocks + metadata
POOL = 2200; cols = []; meta = []; seen = set()
while len(cols) < POOL:
    (v1, u1, r1) = zono(); (v2, u2, r2) = zono(); u = v1 | v2
    idx = sorted(vidx[p] for p in u)
    if len(idx) > 10: idx = idx[:10]
    key = tuple(idx)
    if key in seen: continue
    seen.add(key)
    cols.append(orbit_col(idx)); meta.append((len(u), len(u1)+len(u2), r1 and r2))   # (#verts, #gens, allroot)
    if len(cols) % 500 == 0: print(f"  pool {len(cols)}/{POOL}  [{time.time()-t0:.0f}s]", flush=True)
C = cp.stack(cols, axis=1)                              # m x POOL
Cn = C / cp.linalg.norm(C, axis=0, keepdims=True)
lin = cp.asarray(X.astype(np.float32)); Ql, _ = cp.linalg.qr(lin)
b = bmax - Ql @ (Ql.T @ bmax); b = b / cp.linalg.norm(bmax)   # residual after linear, normalized
print(f"pool ready {C.shape}. OMP:  [{time.time()-t0:.0f}s]", flush=True)

# OMP
sel = []; Qs = Ql.copy(); r = bmax - Ql @ (Ql.T @ bmax)
nb = float(cp.linalg.norm(bmax))
print(f"  {'#sel':>5} {'rel_resid':>10}  newly-added block (verts,gens,allroot)", flush=True)
for step in range(1, 121):
    corr = cp.abs(Cn.T @ r)
    for s in sel: corr[s] = -1
    j = int(cp.argmax(corr))
    sel.append(j)
    c = C[:, j:j+1]; c = c - Qs @ (Qs.T @ c); nc = float(cp.linalg.norm(c))
    if nc < 1e-6: continue
    Qs = cp.concatenate([Qs, c/nc], axis=1)
    r = bmax - Qs @ (Qs.T @ bmax); rel = float(cp.linalg.norm(r))/nb
    if step <= 30 or step % 10 == 0:
        print(f"  {step:>5} {rel:>10.5f}  {meta[j]}", flush=True)
    if rel < 1e-4: print("  -> EXACT-ish fit!", flush=True); break
print(f"\n[{time.time()-t0:.0f}s] sparse if rel_resid drops fast with few blocks; inspect whether dominant blocks share type.", flush=True)
