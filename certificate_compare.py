# Compare the OUT certificates (residual directions) for max_7 across families: weight-2, weight-3, low-complexity
# weight-4. r_w = max_7 - proj(max_7 onto span(weight-w family)+linear), the dual functional that annihilates the
# whole family but not max_7. Compare: norms (floors), pairwise cosines (do they converge?), concentration by
# tie-stratum (shared support / same gradient defect?), parity. A STABLE limiting r_w = a lower-bound invariant.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import gpu_init  # noqa
import cupy as cp
import core
n = 7; t0 = time.time()
m = 20000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
Xm = -X
bmax = cp.asarray(X.max(axis=1).astype(np.float32)); bmaxm = cp.asarray(Xm.max(axis=1).astype(np.float32))
nb = float(cp.linalg.norm(bmax))
lin = cp.asarray(X.astype(np.float32))

def fast_lattice(Wt):
    L = [c for c in itertools.product(range(Wt+1), repeat=n) if sum(c) == Wt]; Ls = set(L); vi = {p:i for i,p in enumerate(L)}
    return L, Ls, vi
def diffgens(L):
    g = set()
    for p in L:
        for q in L:
            if p == q: continue
            dd = tuple(q[k]-p[k] for k in range(n)); gg = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//gg for x in dd)
            if dd > tuple(-x for x in dd): g.add(dd)
    return sorted(g)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

def build_complete(Wt, cap):
    L, Ls, vi = fast_lattice(Wt); gens = diffgens(L)
    def ssz(p, Gs):
        verts = set()
        for s in range(len(Gs)+1):
            for S in itertools.combinations(Gs, s):
                q = p
                for g in S: q = shift(q, g)
                if q not in Ls: return None
                verts.add(q)
        return frozenset(verts)
    zon = set()
    for p in L:
        for g in gens:
            v = ssz(p, [g])
            if v and len(v) >= 2: zon.add(v)
    for p in L:
        for i in range(len(gens)):
            if shift(p, gens[i]) not in Ls: continue
            for j in range(i+1, len(gens)):
                v = ssz(p, [gens[i], gens[j]])
                if v and len(v) >= 3: zon.add(v)
    if Wt >= 3:
        for zz in [zz for zz in list(zon) if len(zz) >= 4]:
            for g in gens:
                verts = set(zz); ok = True
                for u in zz:
                    w = shift(u, g)
                    if w not in Ls: ok = False; break
                    verts.add(w)
                if ok and len(verts) > len(zz): zon.add(frozenset(verts))
    zon = list(zon); zorb = core.orbits_of(zon, n); zreps = [zorb[k][0] for k in zorb]
    bl = set(zon)
    for a in zreps:
        for b in zon:
            u = a | b
            if len(u) <= cap: bl.add(u)
    orb = core.orbits_of(list(bl), n); ok = list(orb); N = len(ok)
    Pm = (X @ np.array(L, dtype=np.int64).T).astype(np.float32); vmap = vi
    C = np.empty((m, N), dtype=np.float32); Cm = np.empty((m, N), dtype=np.float32)
    Pmm = (Xm @ np.array(L, dtype=np.int64).T).astype(np.float32)
    for j, k in enumerate(ok):
        acc = np.zeros(m, np.float32); accm = np.zeros(m, np.float32)
        for vs in orb[k]:
            ix = [vmap[p] for p in vs]; acc += Pm[:, ix].max(axis=1); accm += Pmm[:, ix].max(axis=1)
        C[:, j] = acc; Cm[:, j] = accm
    return C, Cm, N

def residual(Cnp, Cm_np):
    A = cp.concatenate([cp.asarray(Cnp), lin], axis=1)
    Q, _ = cp.linalg.qr(A); del A
    r = bmax - Q @ (Q.T @ bmax)
    Am = cp.concatenate([cp.asarray(Cm_np), cp.asarray(Xm.astype(np.float32))], axis=1)
    Qm, _ = cp.linalg.qr(Am); rm = bmaxm - Qm @ (Qm.T @ bmaxm)   # residual at -X (for parity comparison structure)
    del Q, Qm; cp.get_default_memory_pool().free_all_blocks()
    return r, rm

res = {}
for Wt, cap in ((2, 12), (3, 16)):
    C, Cm, N = build_complete(Wt, cap)
    r, rm = residual(C, Cm)
    res[Wt] = (r, rm); del C, Cm
    print(f"weight-{Wt}: N={N} orbits, floor=||r||/||b||={float(cp.linalg.norm(r))/nb:.5f}  [{time.time()-t0:.0f}s]", flush=True)

# compare directions and support
mx = X.max(axis=1); active = (X == mx[:, None]).sum(axis=1)   # max-multiplicity (#coords == max)
gap = np.sort(X, axis=1)[:, ::-1]; topgap = gap[:, 0] - gap[:, 1]
r2 = res[2][0]; r3 = res[3][0]
c23 = float((r2 @ r3) / (cp.linalg.norm(r2) * cp.linalg.norm(r3)))
print(f"\ncosine(r2, r3) = {c23:.4f}   (=floor ratio if nested; direction overlap)", flush=True)
for tag, r in (("r2", r2), ("r3", r3)):
    ra = cp.asnumpy(cp.abs(r))
    print(f"\n{tag} concentration: mean|{tag}| by max-multiplicity k (#coords==max):", flush=True)
    for k in range(1, 5):
        msk = active == k
        if msk.sum() < 10: continue
        print(f"    k={k}: #pts={int(msk.sum()):>6}  mean|{tag}|={ra[msk].mean():.4f}  (frac of total energy {((ra[msk]**2).sum()/(ra**2).sum())*100:5.1f}%)", flush=True)
print(f"\n[{time.time()-t0:.0f}s] do r2,r3 concentrate on the SAME strata (stable obstruction) -> candidate invariant?", flush=True)
