# Reusable pricer for column generation, any n and weight w.
# Single-zonotope pricing is COMPLETE: given lambda _|_ linear, lambda.col(zono) = sum_{g in T} W(g),
# W(g) = sum_i lambda_i (g.x_i)_+. The base drops out, so the max over zonotopes is the max-weight set T of
# generators (|T| <= w, since weight-w zonotopes have dim <= w) that is lattice-realizable (some base p with
# p + every subset-sum of T in the lattice). Sorted branch-and-bound over the positive-W (and negative-W)
# generators gives the exact max |lambda.col| over ALL single zonotopes. Joins are searched over a pool of the
# best zonotopes (strong, not yet complete -- the join max-coupling is the remaining frontier).
import itertools
from functools import reduce
from math import gcd
import numpy as np

class Pricer:
    def __init__(self, n, w, m=4000, seed=0):
        self.n, self.w = n, w
        self.WP = [c for c in itertools.product(range(w+1), repeat=n) if sum(c) == w]
        self.WPset = set(self.WP)
        gens = set()
        for p in self.WP:
            for q in self.WP:
                if p == q: continue
                d = tuple(q[k]-p[k] for k in range(n)); g = reduce(gcd, [abs(x) for x in d]); d = tuple(x//g for x in d)
                gens.add(d)
        gens = [g for g in gens if g > tuple(-x for x in g)]
        self.gens = gens + [tuple(-x for x in g) for g in gens]
        self.G = np.array(self.gens, dtype=np.float64)
        rng = np.random.default_rng(seed)
        self.X = rng.integers(-10, 11, size=(m, n)).astype(np.float64)
        self.reluGX = np.maximum(self.G @ self.X.T, 0.0)   # (#gens x m)
        self.bmax = self.X.max(axis=1).astype(np.float64)
        self.m = m

    def realizable_base(self, T):
        # generators are sum-0, so vertices p+s (s = subset-sum) have sum w automatically; need p+s >= 0.
        # realizable iff sum_k max(0, -min_s s_k) <= w; then a base exists.
        n, w = self.n, self.w
        sums = [(0,)*n]
        for g in T: sums = sums + [tuple(s[k]+g[k] for k in range(n)) for s in sums]
        mn = [min(s[k] for s in sums) for k in range(n)]
        need = [max(0, -mn[k]) for k in range(n)]
        if sum(need) > w: return None
        p = list(need); rem = w - sum(need); p[0] += rem      # distribute slack to coord 0
        return tuple(p)

    def _best_positive(self, W, idxs):
        # max sum_{g in T} W[g] over lattice-realizable T subset of idxs (sorted desc by W), |T|<=w
        idxs = sorted(idxs, key=lambda i: -W[i])
        pref = np.cumsum([W[i] for i in idxs] + [0.0])   # prefix sums for the B&B bound
        best = [0.0, None]
        n, w = self.n, self.w
        def dfs(start, chosen, cur):
            if cur > best[0]:
                p = self.realizable_base([self.gens[i] for i in chosen])
                if p is not None: best[0], best[1] = cur, (tuple(chosen), p)
            if len(chosen) == w: return
            for pos in range(start, len(idxs)):
                # bound: cur + best possible from remaining (take next w-len positives)
                rem = w - len(chosen)
                bound = cur + (pref[pos+rem] - pref[pos] if pos+rem < len(pref) else pref[-1]-pref[pos])
                if bound <= best[0]: break
                dfs(pos+1, chosen+[idxs[pos]], cur + W[idxs[pos]])
        dfs(0, [], 0.0)
        return best

    def price_single(self, lam, topK=70):
        W = self.reluGX @ lam
        pos = sorted([i for i in range(len(self.gens)) if W[i] > 1e-12], key=lambda i: -W[i])[:topK]
        neg = sorted([i for i in range(len(self.gens)) if W[i] < -1e-12], key=lambda i: W[i])[:topK]
        bp = self._best_positive(W, pos)
        bn = self._best_positive(-W, neg)   # max of -W over negatives = -(min sum)
        # bp: largest positive lambda.col; bn[0]: largest |negative|
        cands = []
        if bp[1] is not None: cands.append((bp[0], bp[1]))
        if bn[1] is not None: cands.append((bn[0], bn[1]))
        cands.sort(key=lambda z: -z[0])
        return cands  # list of (|value|, (gen-index-tuple, base))

    def zono_col(self, gentup, base):
        c = self.X @ np.array(base, dtype=np.float64)
        for gi in gentup: c = c + self.reluGX[gi]
        return c

    def pool_zonos(self, lam, K=200):
        # a pool of high-|W| single zonotopes for join search: top generators + their realizable small sets
        W = self.reluGX @ lam
        order = np.argsort(-np.abs(W))[:max(K, 40)]
        pool = []
        for gi in order[:40]:
            p = self.realizable_base([self.gens[gi]])
            if p is not None: pool.append(((gi,), p))
        topg = list(order[:24])
        for a in range(len(topg)):
            for b in range(a+1, len(topg)):
                p = self.realizable_base([self.gens[topg[a]], self.gens[topg[b]]])
                if p is not None: pool.append(((topg[a], topg[b]), p))
        # also the complete-pricer winners
        for _, (gt, p) in self.price_single(lam):
            pool.append((gt, p))
        return pool[:K]

    def price_join(self, lam, pool):
        cols = [self.zono_col(gt, p) for (gt, p) in pool]
        best = 0.0
        for i in range(len(cols)):
            for j in range(i, len(cols)):
                jc = np.maximum(cols[i], cols[j])
                v = abs(float(lam @ jc))
                if v > best: best, bestcol = v, jc
        return (best, bestcol) if best > 0 else (0.0, None)
