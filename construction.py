"""The max6 construction, loaded from results/construction_max6.txt and expanded to its P2 terms.

The file lists 6 coefficients, each attached to one S6-orbit of a weight-2 P2 building block (given by a
representative vertex set). Expanding each orbit under all 6! coordinate permutations gives the full
signed sum of support functions equal to max6. flat_terms() returns those individual terms with integer
coefficients (coef * D) for use by the prover and verifier.
"""
import itertools, ast, os
from fractions import Fraction as F
import numpy as np

n = 6
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "construction_max6.txt")

def load():
    coeffs = []; reps = []; D = None; L = None
    for line in open(PATH):
        s = line.strip()
        if s.startswith("max6") and "D=" in s:
            D = int(s.split("D=")[1].split(",")[0])
            L = [F(x) for x in ast.literal_eval(s.split("L=")[1])]
        elif s.startswith("coeff"):
            coeffs.append(F(s.split()[1]))
            reps.append(ast.literal_eval(s[s.index("["):]))
    return coeffs, reps, D, L

def orbit(rep):
    seen = set()
    for pm in itertools.permutations(range(n)):
        seen.add(frozenset(tuple(v[pm[k]] for k in range(n)) for v in rep))
    return seen

def flat_terms():
    coeffs, reps, D, L = load()
    flatcoef = []; flatV = []
    for c, rep in zip(coeffs, reps):
        ic = int(c*D)
        for vs in orbit(rep):
            flatcoef.append(ic); flatV.append(np.array(sorted(vs), dtype=np.int64))
    return np.array(flatcoef, dtype=np.int64), flatV, D, L

if __name__ == "__main__":
    fc, fV, D, L = flat_terms()
    print(f"max6 construction: {len(load()[0])} orbits -> {len(fV)} P2 terms, D={D}, L={[str(x) for x in L]}")
