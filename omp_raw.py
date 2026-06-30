# EXP 2 (fast): is max_7's best weight-4 approximant SPARSE? OMP over RAW weight-4 join blocks (fast to generate,
# no orbit-summing). Track residual vs #blocks. Then canonicalize only the SELECTED blocks to see their orbit
# types (verts, gens, root-ness) -- do a few types dominate (=> ansatz) or is it diffuse?
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
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
roots = {tuple((1 if k==a else -1 if k==b else 0) for k in range(n)) for a in range(n) for b in range(n) if a!=b}
gens = set()
for p in W4:
    for q in W4:
        if p == q: continue
        dd = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in dd]); dd = tuple(x//g for x in dd)
        if dd > tuple(-x for x in dd): gens.add(dd)
gens = list(gens); ng = len(gens)
rng = np.random.default_rng(3)
m = 16000
X = rng.integers(-14, 15, size=(m, n)).astype(np.int64)
bmax = cp.asarray(X.max(axis=1).astype(np.float32)); nb = float(cp.linalg.norm(bmax))
P = cp.asarray((X @ np.array(W4, dtype=np.int64).T).astype(np.float32))
print(f"{NP} pts, m={m}  [{time.time()-t0:.0f}s]", flush=True)

def zono():
    p = W4[rng.integers(NP)]; verts = {p}; used = []; added = 0; tries = 0
    while added < 3 and len(verts) < 8 and tries < 40:
        tries += 1; g = gens[rng.integers(ng)]; nv = set(); ok = True
        for u in verts:
            w = shift(u, g)
            if w not in W4set: ok = False; break
            nv.add(w)
        if ok and nv - verts: verts |= nv; used.append(g); added += 1
    return verts, used

POOL = 24000; PAD = 12
cols = cp.empty((m, POOL), dtype=cp.float32); blocks = []
padidx = np.empty((POOL, PAD), dtype=np.int32)
for j in range(POOL):
    v1, u1 = zono(); v2, u2 = zono(); u = v1 | v2; idx = [vidx[p] for p in u][:PAD]
    padidx[j, :len(idx)] = idx; padidx[j, len(idx):] = idx[0]    # pad by repeating (no effect on max)
    blocks.append((frozenset(u), u1 + u2))
# batched gather+max to build columns fast
pg = cp.asarray(padidx)
for s in range(0, POOL, 2000):
    e = min(s+2000, POOL)
    cols[:, s:e] = P[:, pg[s:e].reshape(-1)].reshape(m, e-s, PAD).max(axis=2)
print(f"pool {POOL} raw blocks built (batched)  [{time.time()-t0:.0f}s]", flush=True)
Cn = cols / cp.linalg.norm(cols, axis=0, keepdims=True)
lin = cp.asarray(X.astype(np.float32)); Qs = cp.linalg.qr(lin)[0]
r = bmax - Qs @ (Qs.T @ bmax)
print(f"  {'#sel':>5} {'rel_resid':>10}", flush=True)
sel = []
for step in range(1, 251):
    corr = cp.abs(Cn.T @ r)
    if sel: corr[cp.asarray(sel)] = -1
    j = int(cp.argmax(corr)); sel.append(j)
    c = cols[:, j:j+1]; c = c - Qs @ (Qs.T @ c); ncn = float(cp.linalg.norm(c))
    if ncn < 1e-7: continue
    Qs = cp.concatenate([Qs, c/ncn], axis=1); r = bmax - Qs @ (Qs.T @ bmax); rel = float(cp.linalg.norm(r))/nb
    if step <= 20 or step % 20 == 0:
        print(f"  {step:>5} {rel:>10.5f}  [{time.time()-t0:.0f}s]", flush=True)
    if rel < 1e-4: print("  -> near-exact fit!", flush=True); break

# canonicalize selected blocks to orbit types
perms = list(itertools.permutations(range(n)))
def canon(fs):
    best = None
    for g in perms:
        s = frozenset(tuple(p[g[k]] for k in range(n)) for p in fs)
        t = tuple(sorted(s))
        if best is None or t < best: best = t
    return best
from collections import Counter
typ = Counter()
for s in sel:
    fs, used = blocks[s]
    allroot = all((g in roots or tuple(-x for x in g) in roots) for g in used)
    typ[(len(fs), len(used), allroot)] += 1
print(f"\nselected {len(sel)} blocks. orbit-type histogram (verts, #gens, allroot) -> count:", flush=True)
for k, v in sorted(typ.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {k} -> {v}", flush=True)
cset = set(canon(blocks[s][0]) for s in sel)
print(f"distinct orbit types among selected: {len(cset)} / {len(sel)} blocks", flush=True)
print(f"[{time.time()-t0:.0f}s] sparse+few-types => ansatz; many distinct types => diffuse.", flush=True)
