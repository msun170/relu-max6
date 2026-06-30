# THE DECISIVE EXPERIMENT (per user's plan): template mining on the OMP-selected weight-4 blocks.
# Not more floors. Question: are the OMP-selected blocks SAMPLES FROM A SIMPLE RULE? i.e. do the ~hundreds
# of selected blocks collapse into a few combinatorial FAMILIES that carry ~all the coefficient mass?
# For each selected block (join U = Z1 u Z2) compute a structural signature:
#   v1,v2,vj  = |Z1|,|Z2|,|U|        g1,g2 = #generators of Z1,Z2
#   nroot,nnon = #root vs #non-root generators (root = e_a - e_b)
#   stab       = |Stab_{S_7}(U)|     part = coordinate-profile partition of U (S_7 symmetry type)
# Then: refit exact LS coefficients over the selected RAW columns, cluster by signature, and report
#   per cluster: #blocks, |coefficient| mass, fraction of mass, mean signed coeff.
# DECISION: do <=10 clusters hold >=80% of the |coefficient| mass?  yes -> pursue IN; no -> stop IN, OUT theory.
import sys, itertools, time
from math import gcd
from functools import reduce
from collections import Counter, defaultdict
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
    padidx[j, :len(idx)] = idx; padidx[j, len(idx):] = idx[0]
    blocks.append((frozenset(u), frozenset(v1), frozenset(v2), u1, u2))
pg = cp.asarray(padidx)
for s in range(0, POOL, 2000):
    e = min(s+2000, POOL); cols[:, s:e] = P[:, pg[s:e].reshape(-1)].reshape(m, e-s, PAD).max(axis=2)
print(f"pool {POOL} raw blocks built  [{time.time()-t0:.0f}s]", flush=True)
Cn = cols / cp.linalg.norm(cols, axis=0, keepdims=True)
lin = cp.asarray(X.astype(np.float32)); Qs = cp.linalg.qr(lin)[0]
r = bmax - Qs @ (Qs.T @ bmax)
NSEL = 500
sel = []
for step in range(1, NSEL+1):
    corr = cp.abs(Cn.T @ r)
    if sel: corr[cp.asarray(sel)] = -1
    j = int(cp.argmax(corr));
    c = cols[:, j:j+1]; c = c - Qs @ (Qs.T @ c); ncn = float(cp.linalg.norm(c))
    if ncn < 1e-7: continue
    sel.append(j)
    Qs = cp.concatenate([Qs, c/ncn], axis=1); r = bmax - Qs @ (Qs.T @ bmax); rel = float(cp.linalg.norm(r))/nb
    if step <= 10 or step % 50 == 0:
        print(f"  sel={len(sel):>4} rel_resid={rel:.5f}  [{time.time()-t0:.0f}s]", flush=True)
    if rel < 1e-4: print("  -> near-exact fit!", flush=True); break
print(f"selected {len(sel)} blocks, final rel_resid={rel:.5f}  [{time.time()-t0:.0f}s]", flush=True)

# exact-ish LS refit for per-block coefficients over the RAW selected columns + linear
Asel = cp.concatenate([cols[:, cp.asarray(sel)], lin], axis=1)
w, _, _, _ = cp.linalg.lstsq(Asel, bmax, rcond=None)
wsel = cp.asnumpy(w[:len(sel)])
fit = float(cp.linalg.norm(Asel @ w - bmax) / nb)
print(f"LS refit over {len(sel)} raw cols + linear: rel_resid={fit:.5f}  [{time.time()-t0:.0f}s]", flush=True)

# structural signatures
def coord_partition(U):  # S_7 symmetry type: group coords by their value-multiset across vertices
    V = list(U); prof = [tuple(sorted(v[c] for v in V)) for c in range(n)]
    sizes = sorted(Counter(prof).values(), reverse=True); return tuple(sizes)
def stab_order(U):       # |Stab_{S_n}(U)|, restricted to profile-preserving perms (the only candidates)
    V = list(U); prof = [tuple(sorted(v[c] for v in V)) for c in range(n)]
    classes = defaultdict(list)
    for c in range(n): classes[prof[c]].append(c)
    order = sorted(classes); cnt = 0
    choices = [list(itertools.permutations(classes[p])) for p in order]
    base = [classes[p] for p in order]
    for combo in itertools.product(*choices):
        pm = list(range(n))
        for src, dst in zip(base, combo):
            for s, t in zip(src, dst): pm[s] = t
        if frozenset(tuple(v[pm[k]] for k in range(n)) for v in V) == U: cnt += 1
    return cnt

sig = []
for s in sel:
    U, Z1, Z2, u1, u2 = blocks[s]
    a, b = sorted([len(Z1), len(Z2)]); ga, gb = sorted([len(u1), len(u2)])
    allg = u1 + u2; nroot = sum((g in roots or tuple(-x for x in g) in roots) for g in allg); nnon = len(allg) - nroot
    sig.append((a, b, len(U), ga, gb, nroot, nnon, coord_partition(U), stab_order(U)))
print(f"signatures computed  [{time.time()-t0:.0f}s]", flush=True)

aw = np.abs(wsel); tot = aw.sum() + 1e-12
def report(keyfn, label):
    mass = defaultdict(float); cnt = Counter(); smass = defaultdict(float)
    for s_, k in zip(range(len(sel)), [keyfn(x) for x in sig]):
        mass[k] += aw[s_]; cnt[k] += 1; smass[k] += wsel[s_]
    rows = sorted(mass.items(), key=lambda kv: -kv[1])
    print(f"\n=== cluster by {label}: {len(rows)} clusters ===", flush=True)
    cum = 0.0
    for i, (k, mss) in enumerate(rows[:15]):
        cum += mss
        print(f"  {str(k):<34} blocks={cnt[k]:>4}  |w|mass={mss/tot*100:5.1f}%  cum={cum/tot*100:5.1f}%  meanSign={smass[k]/cnt[k]:+.3f}", flush=True)
    top10 = sum(mss for _, mss in rows[:10]) / tot
    print(f"  -> top-10 clusters hold {top10*100:.1f}% of |w| mass  ({len(rows)} clusters total)", flush=True)
    return top10

# coarse (verts,#gens,allroot) as before, MID full structural, and pure symmetry type
report(lambda x: (x[2], x[3]+x[4], x[6] == 0), "COARSE (vj, #gens, allroot)")
t10mid = report(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6]), "MID (v1,v2,vj,g1,g2,nroot,nnon)")
report(lambda x: (x[7],), "SYMMETRY TYPE (coord partition)")
report(lambda x: (x[2], x[7], x[8]), "FAMILY (vj, partition, stab)")
print(f"\n[{time.time()-t0:.0f}s] DECISION: if a few FAMILIES (MID) carry >=80% mass -> pursue IN construction;", flush=True)
print(f"            if mass stays spread over many families -> stop IN, switch to OUT theory.", flush=True)
