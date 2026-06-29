"""Cross-check: run the actual max6 construction through the independent verifier in verify.py.

This is a separate code path from prove_max6.py (which proves the theorem) and from core.py (which
builds and solves). Agreement here is an independent confirmation that the construction equals max6.
"""
from fractions import Fraction as F
import construction
from verify import MaxTerm, Candidate, verify_random, verify_cells

flatcoef, flatV, D, L = construction.flat_terms()
terms = []
for c, V in zip(flatcoef, flatV):
    rows = [(tuple(int(x) for x in g), 0) for g in V]   # h_Q(x) = max_g <g,x>
    terms.append((F(int(c), D), MaxTerm(rows)))
cand = Candidate(6, terms, Lc=[F(x) for x in L])

bad = len(verify_random(cand, 5000))
ncells, badcells = verify_cells(cand)
print(f"independent verify: random bad={bad}; sampled cells {ncells} bad {len(badcells)}")
print("OK: construction == max6 (independent path agrees)" if bad == 0 and not badcells else "MISMATCH")
