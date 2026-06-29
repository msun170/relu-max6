# Fast column generation for "is max7 in the weight-W span". Pricing via the zonotope decomposition:
#   col(zonotope base p, gens T) = X@p + sum_{g in T} relu(g.X);  lambda . col = sum_{g in T} W(g)  (lam _|_ linear)
# where W(g) = relu(G X^T) . lambda. So pricing single zonotopes = pick lattice-valid generator sets with large
# |sum W(g)|; joins = max of two zonotope columns (priced directly). Blocks generated on the fly (no 8 TB).
import sys, itertools, time
from math import gcd
from functools import reduce
import numpy as np
sys.path.insert(0, "C:/Users/nuswe/relu-max6")
import core
n = 7
W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
m = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
def wpoints(W): return [c for c in itertools.product(range(W+1), repeat=n) if sum(c) == W]
WP = wpoints(W); WPset = set(WP); WPa = np.array(WP, dtype=np.float64)
def shift(p, g): return tuple(p[k]+g[k] for k in range(n))
gens = set()
for p in WP:
    for q in WP:
        if p == q: continue
        d = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in d]); d = tuple(x//g for x in d)
        gens.add(d)
gens = [g for g in gens if g > tuple(-x for x in g)]
gens = gens + [tuple(-x for x in g) for g in gens]     # both orientations (zonotope generators)
G = np.array(gens, dtype=np.float64)
print(f"weight-{W}: {len(WP)} points, {len(gens)} oriented generators, m={m}", flush=True)

rng = np.random.default_rng(0)
X = rng.integers(-10, 11, size=(m, n)).astype(np.float64)
reluGX = np.maximum(G @ X.T, 0.0)            # (#gens x m)
bmax = X.max(axis=1).astype(np.float64)
Xp_cache = {}
def base_col(p):                              # X @ p
    if p not in Xp_cache: Xp_cache[p] = X @ np.array(p, dtype=np.float64)
    return Xp_cache[p]

def valid_base(T):                            # a base p with p + every subset-sum of T in lattice
    sums = [(0,)*n]
    for g in T:
        sums = sums + [tuple(s[k]+g[k] for k in range(n)) for s in sums]
    for p in WP:
        if all(tuple(p[k]+s[k] for k in range(n)) in WPset for s in sums): return p
    return None

def zono_col(p, T):
    c = base_col(p).copy()
    for g in T: c = c + reluGX[gens.index(g)]
    return c

t0 = time.time()
working = [X[:, i] for i in range(n)]          # linear functions
Wmat = np.column_stack(working)
hist = []
for it in range(2000):
    sol, *_ = np.linalg.lstsq(Wmat, bmax, rcond=None)
    r = bmax - Wmat @ sol; relres = np.linalg.norm(r) / np.linalg.norm(bmax)
    nb = Wmat.shape[1] - n
    hist.append(relres)
    if relres < 1e-9:
        print(f"iter {it}: relres {relres:.2e} -> max7 IN weight-{W} with {nb} blocks (m={m}); VACUOUS? blocks/m={nb/m:.2f}", flush=True)
        break
    lam = r
    Wg = reluGX @ lam                          # W(g) per oriented generator
    order = np.argsort(-np.abs(Wg))
    # candidate zonotopes: best segments + parallelograms among top generators
    cands = []                                  # (value_estimate, p, T)
    K = 24
    topg = [gens[int(i)] for i in order[:K]]
    # segments
    for g in topg[:8]:
        p = valid_base((g,))
        if p is not None: cands.append((abs(Wg[gens.index(g)]), p, (g,)))
    # parallelograms (pairs with valid base)
    for a in range(min(K, 12)):
        for b in range(a+1, min(K, 12)):
            g1, g2 = topg[a], topg[b]
            p = valid_base((g1, g2))
            if p is not None: cands.append((abs(Wg[order[a]] + Wg[order[b]]), p, (g1, g2)))
    cands.sort(key=lambda z: -z[0])
    cands = cands[:30]
    cols = [zono_col(p, T) for (_, p, T) in cands]
    best_val = 0.0; best_col = None
    for c in cols:
        v = abs(float(lam @ c))
        if v > best_val: best_val, best_col = v, c
    # joins of candidate pairs
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            jc = np.maximum(cols[i], cols[j]); v = abs(float(lam @ jc))
            if v > best_val: best_val, best_col = v, jc
    if best_col is None or best_val < 1e-6 * np.linalg.norm(lam):
        print(f"iter {it}: pricing found no violator (best={best_val:.2e}) -> max7 OUT (lambda certifies on these blocks)", flush=True)
        break
    working.append(best_col); Wmat = np.column_stack(working)
    if it % 50 == 0 or it < 3:
        print(f"iter {it}: relres={relres:.4e}, blocks={nb}, best|lam.col|={best_val:.3e}  [{time.time()-t0:.0f}s]", flush=True)
print(f"done: blocks={Wmat.shape[1]-n}, final relres={hist[-1]:.4e}, m={m}  [{time.time()-t0:.0f}s]", flush=True)
