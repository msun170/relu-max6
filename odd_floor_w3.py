# Does the ODD-part floor hold while the full floor collapses?  Scale-invariant-route test.
# max_7 = even_part + odd_part;  odd_part = (max+min)/2 (central asymmetry).  Zonotopes are centrally symmetric
# so a join's support function = (linear odd) + (even); its NONLINEAR odd content is structurally limited.
# Test: floor of odd_part(max_7) against {odd-parts of complete weight-3 joins} + linear pass-through.
# If FULL f(3)~=0.0022 but ODD f(3) stays large, the obstruction lives in the (scale-invariant) odd part.
import sys, itertools, time, gc, os
from math import gcd
from functools import reduce
import numpy as np
import cupy as cp
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
gens = sorted(gens); t0 = time.time()
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

m = 16000
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
cache = f"{SCR}/oddeven_w3_m{m}.npz"
if os.path.exists(cache):
    z = np.load(cache); ODD = z["ODD"]; EVN = z["EVN"]; LIN = z["LIN"]
    odd_t = z["odd_t"]; even_t = z["even_t"]; full_t = z["full_t"]; CX = z["CX"]
    print(f"loaded cache  [{time.time()-t0:.0f}s]", flush=True)
else:
    X = np.random.default_rng(101).integers(-16, 17, size=(m, n)).astype(np.int64)
    # FAST build: all block vertices live in the 84-point weight-3 lattice. Compute X.W3 once (m x 84),
    # then h_Q(X)=max over the block's vertex columns, h_Q(-X) = -min over the same columns.
    Wn = np.array(W3, dtype=np.int64)                 # (84 x n)
    vidx = {p: i for i, p in enumerate(W3)}
    P = (X @ Wn.T).astype(np.int32)                   # (m x 84)
    CX = np.empty((m, N), dtype=np.float32); CXm = np.empty((m, N), dtype=np.float32)
    for j, k in enumerate(okeys):
        cx = np.zeros(m, dtype=np.int64); cxm = np.zeros(m, dtype=np.int64)
        for vs in orb[k]:
            cols = P[:, [vidx[p] for p in vs]]
            cx += cols.max(axis=1); cxm += -cols.min(axis=1)
        CX[:, j] = cx; CXm[:, j] = cxm
        if j % 5000 == 0: print(f"  col {j}/{N}  [{time.time()-t0:.0f}s]", flush=True)
    ODD = (CX - CXm) / 2.0; EVN = (CX + CXm) / 2.0
    LIN = X.astype(np.float32)
    mx = X.max(axis=1).astype(np.float32); mn = X.min(axis=1).astype(np.float32)
    odd_t = (mx + mn) / 2.0; even_t = (mx - mn) / 2.0; full_t = mx
    np.savez(cache, ODD=ODD, EVN=EVN, LIN=LIN, odd_t=odd_t, even_t=even_t, full_t=full_t, CX=CX)
    print(f"built+cached  [{time.time()-t0:.0f}s]", flush=True)

def cgls_floor(Anp, bnp, tag, iters=4000):
    Ag = cp.asarray(Anp); bg = cp.asarray(bnp)
    cn = cp.linalg.norm(Ag, axis=0); cn = cp.where(cn > 0, cn, cp.float32(1.0)); Ag /= cn
    An = float(cp.linalg.norm(Ag)); bn = float(cp.linalg.norm(bg))
    x = cp.zeros(Ag.shape[1], dtype=cp.float32); r = bg.copy()
    s = Ag.T @ r; p = s.copy(); gamma = float(s @ s)
    for it in range(1, iters+1):
        q = Ag @ p; alpha = gamma/float(q @ q); x += alpha*p; r -= alpha*q
        s = Ag.T @ r; gnew = float(s @ s); beta = gnew/gamma; p = s + beta*p; gamma = gnew
        if it % 500 == 0:
            rr = float(cp.linalg.norm(r))/bn; o = float(cp.linalg.norm(s))/(An*float(cp.linalg.norm(r))+1e-30)
            print(f"    {tag} it{it}: floor~={rr:.5f} ortho={o:.1e}  [{time.time()-t0:.0f}s]", flush=True)
            if o < 1e-6: break
    rr = float(cp.linalg.norm(r))/bn; o = float(cp.linalg.norm(Ag.T@r))/(An*float(cp.linalg.norm(r))+1e-30)
    del Ag, bg, x, r, s, p; cp.get_default_memory_pool().free_all_blocks()
    print(f"  {tag} floor = {rr:.5f}  (ortho {o:.1e})  [{time.time()-t0:.0f}s]", flush=True)
    return rr

print(f"computing ODD/EVEN floors (weight-3 complete family, m={m}):", flush=True)
fo = cgls_floor(np.column_stack([ODD, LIN]), odd_t, "ODD ")
fe = cgls_floor(np.column_stack([EVN]), even_t, "EVEN")
print(f"\nweight-3:  ODD={fo:.5f}   EVEN={fe:.5f}   (FULL was 0.0022 from full m=20000 solve)", flush=True)
print(f"weight-2 was: FULL=0.0400  ODD=0.1189  EVEN=0.0247", flush=True)
print(f"=> if ODD holds (~0.1) while FULL collapsed to 0.002, the obstruction is the scale-invariant odd part.", flush=True)
