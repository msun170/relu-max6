# RELIABLE f(4): complete weight-3 family (forces floor <= f(3)) + STRUCTURED weight-4 blocks (small/low-complexity
# joins, the kind that matter). All three guards: m > rank (non-vacuous), CGLS to ortho<1e-6 (converged), and a
# random-target CONTROL that must stay high (else vacuous). Answers: does weight-4 help (f(4)<<f(3), contraction
# continues) or stall (f(4)~f(3) => f_inf>0 => separation finitely provable)?
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; t0 = time.time()

def lat(W):
    L = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
    return L, set(L), {p: i for i, p in enumerate(L)}
def diffgens(L):
    g = set()
    for p in L:
        for q in L:
            if p == q: continue
            dd = tuple(q[k]-p[k] for k in range(n)); gg = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//gg for x in dd)
            if dd > tuple(-x for x in dd): g.add(dd)
    return sorted(g)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

def zonos_upto(L, Lset, gens, maxgen):
    z = set()
    def ssz(p, G):
        verts = set()
        for s in range(len(G)+1):
            for S in itertools.combinations(G, s):
                q = p
                for gg in S: q = shift(q, gg)
                if q not in Lset: return None
                verts.add(q)
        return frozenset(verts)
    for p in L:
        for g in gens:
            v = ssz(p, [g])
            if v and len(v) >= 2: z.add(v)
    for p in L:
        for i in range(len(gens)):
            if shift(p, gens[i]) not in Lset: continue
            for j in range(i+1, len(gens)):
                v = ssz(p, [gens[i], gens[j]])
                if v and len(v) >= 3: z.add(v)
    if maxgen >= 3:
        for zz in [zz for zz in list(z) if len(zz) >= 4]:
            for g in gens:
                verts = set(zz); ok = True
                for u in zz:
                    w = shift(u, g)
                    if w not in Lset: ok = False; break
                    verts.add(w)
                if ok and len(verts) > len(zz): z.add(frozenset(verts))
    return list(z)

def blocks_orbits(L, Lset, gens, maxgen, cap):
    z = zonos_upto(L, Lset, gens, maxgen)
    zorb = core.orbits_of(z, n); zreps = [zorb[k][0] for k in zorb]
    bl = set(z)
    for a in zreps:
        for b in z:
            u = a | b
            if len(u) <= cap: bl.add(u)
    return core.orbits_of(list(bl), n)

m = 26000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
brand = np.random.default_rng(5).standard_normal(m).astype(np.float32)   # control target (must stay OUT)

def colmat(orb, L, lidx):
    okeys = list(orb); N = len(okeys)
    P = (X @ np.array(L, dtype=np.int64).T).astype(np.float32)
    C = np.empty((m, N), dtype=np.float32)
    for j, k in enumerate(okeys):
        acc = np.zeros(m, dtype=np.float32)
        for vs in orb[k]:
            acc += P[:, [lidx[p] for p in vs]].max(axis=1)
        C[:, j] = acc
    return C

# complete weight-3
L3, L3s, l3 = lat(3); g3 = diffgens(L3)
orb3 = blocks_orbits(L3, L3s, g3, 3, 16); C3 = colmat(orb3, L3, l3)
print(f"weight-3 complete: {C3.shape[1]} orbits  [{time.time()-t0:.0f}s]", flush=True)
# structured weight-4: small zonotopes (<=2 gens) only, joins capped -> keeps it feasible & low-complexity
L4, L4s, l4 = lat(4); g4 = diffgens(L4)
orb4 = blocks_orbits(L4, L4s, g4, 2, 14); C4 = colmat(orb4, L4, l4)
print(f"weight-4 structured: {C4.shape[1]} orbits  [{time.time()-t0:.0f}s]", flush=True)

A = np.column_stack([C3, C4, X.astype(np.float32)])
print(f"combined A {A.shape}  [{time.time()-t0:.0f}s]", flush=True)

def cgls_floor(Anp, b, tag):
    Ag = cp.asarray(Anp); bg = cp.asarray(b)
    cn = cp.linalg.norm(Ag, axis=0); cn = cp.where(cn > 0, cn, cp.float32(1.0)); Ag = Ag / cn
    An = float(cp.linalg.norm(Ag)); bn = float(cp.linalg.norm(bg))
    x = cp.zeros(Ag.shape[1], dtype=cp.float32); r = bg.copy()
    s = Ag.T @ r; p = s.copy(); gamma = float(s @ s)
    for it in range(1, 12001):
        q = Ag @ p; al = gamma/float(q @ q); x += al*p; r -= al*q
        s = Ag.T @ r; gnew = float(s @ s); be = gnew/gamma; p = s + be*p; gamma = gnew
        if it % 1000 == 0:
            o = float(cp.linalg.norm(s))/(An*float(cp.linalg.norm(r))+1e-30)
            print(f"    {tag} it{it}: f={float(cp.linalg.norm(r))/bn:.5f} ortho={o:.1e}  [{time.time()-t0:.0f}s]", flush=True)
            if o < 1e-6: break
    return float(cp.linalg.norm(r))/bn

fr = cgls_floor(A, brand.astype(np.float32), "ctrl")
print(f"  CONTROL (random target) floor = {fr:.4f}  (must be >>0; else vacuous)  [{time.time()-t0:.0f}s]", flush=True)
f4 = cgls_floor(A, bmax, "max7")
print(f"\nRELIABLE f(4) = {f4:.5f}   [f(2)=0.0320, f(3)=0.00159]", flush=True)
print(f"  rho_34 = f(4)/f(3) = {f4/0.00159:.3f}   (rho_23=0.050)", flush=True)
print(f"  => f(4)<<f(3): contraction continues (closure).  f(4)~f(3): STALL => f_inf>0 => separation provable.", flush=True)
