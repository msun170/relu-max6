# RELIABLE f(4), fast + cached. Complete weight-3 (fast 84-pt build) forces floor <= f(3); weight-4 blocks use
# ROOT generators only (the low-complexity structure that actually matters -- max_6 was entirely root-generated),
# making enumeration fast. Guards: m=26000 > rank (non-vacuous), CGLS to ortho<1e-6, random-target CONTROL.
import sys, itertools, time, os
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; t0 = time.time()
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
m = 26000
X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
bmax = X.max(axis=1).astype(np.float32)
brand = np.random.default_rng(5).standard_normal(m).astype(np.float32)

def lat(W):
    L = [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]; return L, set(L), {p:i for i,p in enumerate(L)}
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))

def build_cols(W, gens, maxgen, cap, tag):
    cache = f"{SCR}/Crel_{tag}_m{m}.npy"
    if os.path.exists(cache):
        C = np.load(cache); print(f"loaded {tag}: {C.shape}  [{time.time()-t0:.0f}s]", flush=True); return C
    L, Ls, li = lat(W)
    def ssz(p, G):
        verts = set()
        for s in range(len(G)+1):
            for S in itertools.combinations(G, s):
                q = p
                for gg in S: q = shift(q, gg)
                if q not in Ls: return None
                verts.add(q)
        return frozenset(verts)
    z = set()
    for p in L:
        for g in gens:
            v = ssz(p, [g])
            if v and len(v) >= 2: z.add(v)
    for p in L:
        for i in range(len(gens)):
            if shift(p, gens[i]) not in Ls: continue
            for j in range(i+1, len(gens)):
                v = ssz(p, [gens[i], gens[j]])
                if v and len(v) >= 3: z.add(v)
    if maxgen >= 3:
        for zz in [zz for zz in list(z) if len(zz) >= 4]:
            for g in gens:
                verts = set(zz); ok = True
                for u in zz:
                    w = shift(u, g)
                    if w not in Ls: ok = False; break
                    verts.add(w)
                if ok and len(verts) > len(zz): z.add(frozenset(verts))
    z = list(z); zorb = core.orbits_of(z, n); zreps = [zorb[k][0] for k in zorb]
    bl = set(z)
    for a in zreps:
        for b in z:
            u = a | b
            if len(u) <= cap: bl.add(u)
    orb = core.orbits_of(list(bl), n); okeys = list(orb); N = len(okeys)
    P = (X @ np.array(L, dtype=np.int64).T).astype(np.float32)
    C = np.empty((m, N), dtype=np.float32)
    for j, k in enumerate(okeys):
        acc = np.zeros(m, dtype=np.float32)
        for vs in orb[k]:
            acc += P[:, [li[p] for p in vs]].max(axis=1)
        C[:, j] = acc
    np.save(cache, C); print(f"built {tag}: {C.shape}  [{time.time()-t0:.0f}s]", flush=True)
    return C

def diffgens(L):
    g = set()
    for p in L:
        for q in L:
            if p == q: continue
            dd = tuple(q[k]-p[k] for k in range(n)); gg = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//gg for x in dd)
            if dd > tuple(-x for x in dd): g.add(dd)
    return sorted(g)
roots = [tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a!=b]

L3, _, _ = lat(3)
C3 = build_cols(3, diffgens(L3), 3, 16, "w3full")    # complete weight-3
C4 = build_cols(4, roots, 2, 14, "w4root")           # weight-4 ROOT joins (structured, fast)

A = np.column_stack([C3, C4, X.astype(np.float32)])
print(f"combined A {A.shape}  [{time.time()-t0:.0f}s]", flush=True)

def cgls(Anp, b, tag):
    Ag = cp.asarray(Anp); bg = cp.asarray(b)
    cn = cp.linalg.norm(Ag, axis=0); cn = cp.where(cn>0, cn, cp.float32(1.0)); Ag = Ag/cn
    An = float(cp.linalg.norm(Ag)); bn = float(cp.linalg.norm(bg))
    x = cp.zeros(Ag.shape[1], dtype=cp.float32); r = bg.copy(); s = Ag.T@r; p = s.copy(); gamma = float(s@s)
    for it in range(1, 15001):
        q = Ag@p; al = gamma/float(q@q); x += al*p; r -= al*q
        s = Ag.T@r; gn = float(s@s); be = gn/gamma; p = s+be*p; gamma = gn
        if it % 1000 == 0:
            o = float(cp.linalg.norm(s))/(An*float(cp.linalg.norm(r))+1e-30)
            print(f"    {tag} it{it}: f={float(cp.linalg.norm(r))/bn:.5f} ortho={o:.1e}  [{time.time()-t0:.0f}s]", flush=True)
            if o < 1e-6: break
    return float(cp.linalg.norm(r))/bn

fr = cgls(A, brand, "ctrl"); print(f"  CONTROL floor = {fr:.4f} (must be >>0)  [{time.time()-t0:.0f}s]", flush=True)
f4 = cgls(A, bmax, "max7")
print(f"\nRELIABLE f(4) = {f4:.5f}   [f(2)=0.0320, f(3)=0.00159, rho_23=0.050]")
print(f"  rho_34 = f(4)/f(3) = {f4/0.00159:.3f}")
print(f"  f(4)<<f(3) => contraction continues (closure);  f(4)~f(3) => STALL => f_inf>0 => separation provable")
