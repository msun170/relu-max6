# Column generation with INCREMENTAL orthonormal projection (fast): maintain an orthonormal basis Q of the
# working columns; residual r = bmax - Q(Q^T bmax) is the dual lambda; pricing finds a weight-W block maximizing
# |lambda.col| via the W(g)=relu(GX^T).lambda decomposition; orthogonalize+append. Watch whether relres -> 0
# (max7 IN weight-W => 2 layers) or plateaus (OUT). Keep basis size k << m so "relres->0" is not vacuous.
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7
W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
m = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
NIT = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
WP = wpoints(W); WPset = set(WP)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in WP:
    for q in WP:
        if p == q: continue
        d = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in d]); d = tuple(x//g for x in d)
        gens.add(d)
gens = [g for g in gens if g > tuple(-x for x in g)]
gens = gens + [tuple(-x for x in g) for g in gens]
gidx = {g: i for i, g in enumerate(gens)}
G = np.array(gens, dtype=np.float64)
rng = np.random.default_rng(0)
X = rng.integers(-10, 11, size=(m, n)).astype(np.float64)
reluGX = np.maximum(G @ X.T, 0.0)
bmax = X.max(axis=1).astype(np.float64)
print(f"weight-{W}: {len(WP)} pts, {len(gens)} gens, m={m}", flush=True)

Xp_cache = {}
def base_col(p):
    if p not in Xp_cache: Xp_cache[p] = X @ np.array(p, dtype=np.float64)
    return Xp_cache[p]
def valid_base(T):
    sums = [(0,)*n]
    for g in T: sums = sums + [tuple(s[k]+g[k] for k in range(n)) for s in sums]
    for p in WP:
        if all(tuple(p[k]+s[k] for k in range(n)) in WPset for s in sums): return p
    return None
def zono_col(p, T):
    c = base_col(p).copy()
    for g in T: c = c + reluGX[gidx[g]]
    return c

# orthonormal basis Q (m x k), start with linear functions orthonormalized
Q = np.linalg.qr(np.column_stack([X[:, i] for i in range(n)]))[0]
bn = np.linalg.norm(bmax)
t0 = time.time(); plateau = 0; last = 1.0
for it in range(NIT):
    proj = Q @ (Q.T @ bmax); r = bmax - proj; relres = np.linalg.norm(r)/bn
    k = Q.shape[1] - n
    if relres < 1e-9:
        print(f"iter {it}: relres {relres:.2e} -> max7 IN weight-{W} ({k} blocks, m={m}, k/m={ (k+n)/m:.2f})", flush=True); break
    lam = r
    Wg = reluGX @ lam
    order = np.argsort(-np.abs(Wg)); topg = [gens[int(i)] for i in order[:28]]
    cands = []
    for g in topg[:10]:
        p = valid_base((g,));
        if p is not None: cands.append((p, (g,)))
    for a in range(min(14, len(topg))):
        for b in range(a+1, min(14, len(topg))):
            p = valid_base((topg[a], topg[b]))
            if p is not None: cands.append((p, (topg[a], topg[b])))
    cols = [zono_col(p, T) for (p, T) in cands]
    best_val = 0.0; best = None
    for c in cols:
        v = abs(float(lam @ c))
        if v > best_val: best_val, best = v, c
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            jc = np.maximum(cols[i], cols[j]); v = abs(float(lam @ jc))
            if v > best_val: best_val, best = v, jc
    if best is None or best_val < 1e-7*np.linalg.norm(lam):
        print(f"iter {it}: no violator (best={best_val:.2e}) -> max7 OUT weight-{W} on priced blocks ({k} blocks)", flush=True); break
    # orthogonalize best against Q and append
    w = best - Q @ (Q.T @ best); nw = np.linalg.norm(w)
    if nw < 1e-8 * np.linalg.norm(best):   # already in span; skip (shouldn't happen since lam.best!=0)
        continue
    Q = np.column_stack([Q, w/nw])
    if k+n > 0.6*m:
        print(f"iter {it}: basis k+n={k+n} approaching m={m}; stopping to avoid vacuity. relres={relres:.4e}", flush=True); break
    if it % 100 == 0 or it < 3:
        print(f"iter {it}: relres={relres:.4e}, blocks={k}, best|lam.col|={best_val:.2e}  [{time.time()-t0:.0f}s]", flush=True)
print(f"done: blocks={Q.shape[1]-n}, relres={np.linalg.norm(bmax-Q@(Q.T@bmax))/bn:.4e}, m={m}  [{time.time()-t0:.0f}s]", flush=True)
