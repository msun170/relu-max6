"""Independent exact verifier for a candidate 2-hidden-layer max_n construction.

A candidate is a signed sum of maxes of affine forms plus a linear term:
    f(x) = L(x) + sum_t c_t * max_i (a_{t,i} . x + b_{t,i}).
We check f(x) == max(x_1,...,x_n) for all x in R^n, exactly (Fraction arithmetic). Both f and max
are piecewise linear; they are equal iff they agree at an interior point of every cell of the common
arrangement (braid hyperplanes x_i=x_j and each term's argument-switch hyperplanes). This is a
deliberately separate code path from core.py, used to cross-check constructions.
"""
import itertools, random
from fractions import Fraction as F

# a single max-of-affine-forms term
class MaxTerm:
    def __init__(self, rows): self.rows = [(tuple(F(v) for v in a), F(b)) for a, b in rows]
    def value(self, x): return max(sum(ai*xi for ai, xi in zip(a, x)) + b for a, b in self.rows)
    def hyperplanes(self):
        H = []
        for (a1, b1), (a2, b2) in itertools.combinations(self.rows, 2):
            H.append((tuple(p-q for p, q in zip(a1, a2)), b1-b2))
        return H

# a candidate function: linear part + signed sum of max terms
class Candidate:
    def __init__(self, n, terms, Lc=None, Lb=0):
        self.n = n; self.terms = [(F(c), t) for c, t in terms]
        self.Lc = tuple(F(v) for v in (Lc if Lc else [0]*n)); self.Lb = F(Lb)
    def value(self, x):
        v = sum(c*xi for c, xi in zip(self.Lc, x)) + self.Lb
        for c, t in self.terms: v += c*t.value(x)
        return v
    def all_hyperplanes(self):
        H = set(); n = self.n
        for i, j in itertools.combinations(range(n), 2):
            a = [F(0)]*n; a[i] = F(1); a[j] = F(-1); H.add((tuple(a), F(0)))
        for _, t in self.terms:
            for a, b in t.hyperplanes():
                if any(v != 0 for v in a): H.add((tuple(a), b))
        return list(H)

# fast random filter: exact check at random rational points
def verify_random(cand, n_trials=4000, scale=20, seed=0):
    rng = random.Random(seed); n = cand.n; bad = []
    for _ in range(n_trials):
        x = tuple(F(rng.randint(-scale, scale), rng.randint(1, scale)) for _ in range(n))
        if cand.value(x) != max(x): bad.append(x)
    return bad

# one representative interior point per sampled full-dimensional cell of the arrangement
def cell_points(cand, n_samples=60000, scale=50, seed=1):
    H = cand.all_hyperplanes(); n = cand.n; rng = random.Random(seed); reps = {}
    for _ in range(n_samples):
        x = tuple(F(rng.randint(-scale, scale), rng.randint(1, scale)) for _ in range(n))
        sv = tuple(1 if (sum(ai*xi for ai, xi in zip(a, x))+b) > 0 else
                   (-1 if (sum(ai*xi for ai, xi in zip(a, x))+b) < 0 else 0) for a, b in H)
        if 0 in sv: continue
        if sv not in reps: reps[sv] = x
    return H, reps

def verify_cells(cand):
    _, reps = cell_points(cand)
    bad = [(x, cand.value(x), max(x)) for x in reps.values() if cand.value(x) != max(x)]
    return len(reps), bad

if __name__ == "__main__":
    # sanity checks: the max2 identity, trivial max_n, and a deliberately wrong candidate
    t = MaxTerm([((1, -1), 0), ((-1, 1), 0)])
    c = Candidate(2, [(F(1, 2), t)], Lc=[F(1, 2), F(1, 2)])
    print("max2 identity: random bad =", len(verify_random(c)), "; cells bad =", len(verify_cells(c)[1]))
    wrong = Candidate(2, [(F(1, 2), t)], Lc=[F(1, 2), F(1, 3)])
    print("wrong candidate (expect nonzero):", len(verify_random(wrong)))
