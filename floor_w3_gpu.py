# Exact-magnitude residual FLOOR of max_7 against the COMPLETE weight-3 join family, on GPU.
# We already know (gpu_w3.py, exact mod-p) that max_7 is OUT of complete weight-3, so the floor is > 0.
# Here we want its MAGNITUDE f(3) = ||bmax - proj||/||bmax||, to compare against f(2)=0.0308 and judge whether
# the floor is collapsing toward 0 (density => singular cert) or holding positive (separation provable, finite cert).
# Guard against the old LSMR non-convergence trap: report the orthogonality residual ||A^T r|| / (||A|| ||r||);
# the floor estimate is only trustworthy if that is ~0 (r genuinely orthogonal to the column space).
import sys, itertools, time, gc
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import lsmr
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; W = 3
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W3 = wpoints(W); W3set = set(W3)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W3:
    for q in W3:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = sorted(gens)
t0 = time.time()
def ssz(p, G):
    verts = set()
    for s in range(len(G)+1):
        for S in itertools.combinations(G, s):
            q = p
            for g in S: q = shift(q, g)
            if q not in W3set: return None
            verts.add(q)
    return frozenset(verts)
zonos = set()
for p in W3:
    for g in gens:
        v = ssz(p, [g])
        if v and len(v) >= 2: zonos.add(v)
for p in W3:
    for i in range(len(gens)):
        if shift(p, gens[i]) not in W3set: continue
        for j in range(i+1, len(gens)):
            v = ssz(p, [gens[i], gens[j]])
            if v and len(v) >= 3: zonos.add(v)
base3 = set(zonos)
for z in [z for z in zonos if len(z) >= 4]:
    for g in gens:
        verts = set(z); ok = True
        for u in z:
            w = shift(u, g)
            if w not in W3set: ok = False; break
            verts.add(w)
        if ok and len(verts) > len(z): base3.add(frozenset(verts))
zonos = list(base3)
zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
blocks = set(zonos)
for a in zreps:
    for b_ in zonos:
        u = a | b_
        if len(u) <= 16: blocks.add(u)
orb = core.orbits_of(list(blocks), n); okeys = list(orb); N = len(okeys)
print(f"N={N} weight-3 orbits  [{time.time()-t0:.0f}s]", flush=True)

m = 20000
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
import os
cache = f"{SCR}/A_w3_m{m}.npy"; bcache = f"{SCR}/b_w3_m{m}.npy"
if os.path.exists(cache):
    A = np.load(cache); bmax = np.load(bcache)
    print(f"loaded cached A {A.shape}  [{time.time()-t0:.0f}s]", flush=True)
else:
    X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
    bmax = X.max(axis=1).astype(np.float64)
    ncol = N + n
    A = np.empty((m, ncol), dtype=np.float32)        # preallocate float32 (~1.5 GB)
    for j, k in enumerate(okeys):
        A[:, j] = core.orbit_column(orb[k], X).astype(np.float32)
        if j % 4000 == 0: print(f"  built col {j}/{N}  [{time.time()-t0:.0f}s]", flush=True)
    for i in range(n): A[:, N + i] = X[:, i].astype(np.float32)
    np.save(cache, A); np.save(bcache, bmax)
    print(f"A {A.shape} built+cached  [{time.time()-t0:.0f}s]", flush=True)

Ag = cp.asarray(A); bg = cp.asarray(bmax.astype(np.float32)); del A; gc.collect()
colnorm = cp.linalg.norm(Ag, axis=0); colnorm = cp.where(colnorm > 0, colnorm, cp.float32(1.0))
Ag /= colnorm                                        # column scaling (does not change column SPACE / the floor)
Anorm = float(cp.linalg.norm(Ag)); bnorm = float(cp.linalg.norm(bg))
print(f"CGLS start (||A||={Anorm:.1f}, ||b||={bnorm:.1f})  [{time.time()-t0:.0f}s]", flush=True)

# manual CGLS: min ||Ag x - bg||, printing relres & orthogonality each block
x = cp.zeros(Ag.shape[1], dtype=cp.float32)
r = bg.copy()
s = Ag.T @ r; p = s.copy(); gamma = float(s @ s)
best = 1.0
for it in range(1, 4001):
    q = Ag @ p
    alpha = gamma / float(q @ q)
    x += alpha * p; r -= alpha * q
    s = Ag.T @ r; gnew = float(s @ s)
    beta = gnew / gamma; p = s + beta * p; gamma = gnew
    if it % 100 == 0 or it == 1:
        relres = float(cp.linalg.norm(r)) / bnorm
        ortho = float(cp.linalg.norm(s)) / (Anorm * float(cp.linalg.norm(r)) + 1e-30)
        best = min(best, relres)
        print(f"  it {it}: f(3)~={relres:.5f}  ortho={ortho:.2e}  [{time.time()-t0:.0f}s]", flush=True)
        if ortho < 1e-6: break
relres = float(cp.linalg.norm(r)) / bnorm
ortho = float(cp.linalg.norm(Ag.T @ r)) / (Anorm * float(cp.linalg.norm(r)) + 1e-30)
print(f"\nf(3) = {relres:.5f}   orthogonality = {ortho:.2e}  (trustworthy if << 1e-3)", flush=True)
print(f"compare: f(2)=0.03080 exact  [{time.time()-t0:.0f}s]", flush=True)
