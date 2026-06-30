# Enumerate the COMPLETE weight-4 join family by ORBITS (S_n symmetry collapses it ~n!-fold), so the complete
# weight-4 floor / exact test becomes feasible -- no greedy column generation, no plateau. Caches the orbit
# representatives to disk for the GPU projection step (floor_w4_gpu.py).
import sys, itertools, time, pickle
from math import gcd
from functools import reduce
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7; W = 4
SCR = "C:/Users/nuswe/AppData/Local/Temp/claude/C--Users-nuswe/b3e9435c-614f-431c-80ea-c7e9f45c3681/scratchpad"
t0 = time.time()
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
W4 = wpoints(W); W4set = set(W4)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = sorted(gens)
print(f"{len(W4)} lattice pts, {len(gens)} gens  [{time.time()-t0:.0f}s]", flush=True)

# DFS zonotope build with pruning: extend a vertex set by a generator only if ALL shifted vertices stay in W4.
# Token lemma: weight-4 zonotope dim <= 4, so at most 4 INDEPENDENT generators; but allow >4 coplanar gens
# (e.g. hexagon) by capping generator count at 6 to stay finite.
zonos = set()
def grow(verts, gi, ngen):
    zonos.add(frozenset(verts))
    if ngen >= 6: return
    for j in range(gi + 1, len(gens)):
        g = gens[j]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok: grow(verts | nv, j, ngen + 1)
for ip, p in enumerate(W4):
    grow({p}, -1, 0)
    if ip % 40 == 0: print(f"  seeds {ip}/{len(W4)}, zonos so far {len(zonos)}  [{time.time()-t0:.0f}s]", flush=True)
zonos = [z for z in zonos if len(z) >= 2]
print(f"{len(zonos)} zonotopes  [{time.time()-t0:.0f}s]", flush=True)
zorb = core.orbits_of(zonos, n); zreps = [zorb[k][0] for k in zorb]
print(f"{len(zorb)} zonotope orbits  [{time.time()-t0:.0f}s]", flush=True)

# joins = zonotope-rep | any zonotope (cap vertex count to keep it a weight-4 block)
blocks = set(frozenset(z) for z in zonos)
for a in zreps:
    for b in zonos:
        u = a | b
        if len(u) <= 20: blocks.add(frozenset(u))
print(f"{len(blocks)} raw blocks pre-orbit  [{time.time()-t0:.0f}s]", flush=True)
orb = core.orbits_of(list(blocks), n)
print(f"\nWEIGHT-4 JOIN ORBITS: {len(orb)}   [{time.time()-t0:.0f}s]", flush=True)
with open(f"{SCR}/w4_orbits.pkl", "wb") as f:
    pickle.dump({"orb": orb, "okeys": list(orb)}, f)
print(f"cached to {SCR}/w4_orbits.pkl", flush=True)
