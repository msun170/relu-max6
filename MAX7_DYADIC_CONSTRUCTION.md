# max_7 dyadic construction search (the favored IN direction)

WHY DYADIC (LITERATURE.md sec 7): AHM's mod-2 valuation forbids odd-denominator / integer max_7 in 2 layers
(ceil(log_2 7)=3). So ANY 2-layer max_7 is necessarily EVEN-DENOMINATOR / DYADIC (confirmed: BBHSY max_5 dyadic;
our max_6 denom 90,180). The dyadic regime = a 2-refined lattice; the weight-4 lattice IS the 2-refinement of
weight-2. So the construction (if it exists) lives on weight-4 (or finer weight-8) with dyadic coefficients, and
our weight-4 tests are exactly the dyadic probe.

## What we have decided (exact, complete families)
- weight-2 COMPLETE: OUT. weight-3 COMPLETE: OUT.  (= HHL/AHM integer-regime, expected.)
- weight-4 <=2-vertex COMPLETE (65 orbits): OUT. <=3-vertex COMPLETE (1180 orbits): OUT.
- weight-4 <=4-vertex COMPLETE (29024 orbits): RUNNING (the all-4-vertex dyadic family).
- weight-4 BBHSY-complexity (joins of two <=2-generator ROOT zonotopes, <=6 verts) COMPLETE: RUNNING (dyadic_search.py)
  -- this reaches 5-6 vertex parallelogram-joins that the <=4v test misses, with the exact block types BBHSY use.

## The block types we are searching (BBHSY-motivated)
BBHSY's MAX_5 formula is M = 1/2 ( P1+P2+P3+P4 + Q - R13-R14-R23-R24 ), each term a join max(Z1,Z2) where the Z_i are
Minkowski sums of a few SHORT segments, e.g. P1 = max( max(2x5, x1+x2),  max(x1,x3)+max(x1,x4) ). So the right block
family is: joins of two zonotopes each built from <=2 SHORT generators (roots e_i-e_j and the short directions
2e_i-e_j-e_k, e_i+e_j-e_k-e_l), on the weight-4 lattice. dyadic_search.py enumerates this COMPLETELY (root version
first; L1<=4 version is larger).

## Decision logic
- IN at any of these complete dyadic families => extract the dyadic rational coefficients (extract_certificate.py:
  minimal support -> exact rational solve -> verify at fresh points -> P2-validity), then the global chamber proof
  (adapt check_max6.py to n=7). That is the construction certificate => max_7 in V_2 (a real theorem; resolves the
  open case and extends BBHSY/our max_6).
- OUT of all feasible complete weight-4 dyadic families => the construction (if any) needs weight-8 (finer dyadic)
  or >2 generators per zonotope; both push past the feasible enumeration boundary, so the next move is the
  THEORY route: generalize BBHSY's simplex mixed-subdivision (Cayley trick) to n=7, or the decomposition-polyhedron
  vertex search (Brandenburg-Grillo-Hertrich arXiv:2410.04907).

## Boundary (feasibility)
weight-4 complete enumeration is feasible up to ~10^4-10^5 orbits (membership matrix m x N, m > rank). <=4v (29024)
is near the edge; >2-generator or weight-8 families are unenumerable (same wall as before). So the dyadic LATTICE
search effectively ends at weight-4; beyond it is the BBHSY/Cayley THEORY construction or the decomposition polyhedron.

## Status / results
- dyadic_search.py ROOT version (joins of two <=2-gen ROOT zonotopes, <=6 verts, 6081 orbits): max_7 OUT (rank 880,
  two primes, sym-control OUT, valid). CAVEAT: ROOT generators only (L1<=2). BBHSY's max_5 ALSO uses NON-root short
  generators (e.g. 2e_5-(e_1+e_2), L1=4 in max(2x5,x1+x2)), so this OUT does NOT cover the full BBHSY-complexity
  family. The L1<=4 (non-root) version has ~66549 zonotopes -> joins explode -> not completely enumerable as-is;
  needs a targeted non-root generator set.
- <=4v (all 4-vertex blocks, 29024 orbits): running.

## Honest convergence
Every FEASIBLE complete weight-4 dyadic lattice family tested is OUT: <=2v, <=3v, root <=2-gen-<=6v. The non-root
BBHSY family and <=4v remain (running / partially walled). The pattern strongly suggests max_7 is OUT of all feasible
weight-4 lattice families -> the construction, if it exists, needs weight-8 (finer dyadic, unenumerable) or the
THEORY route (generalize BBHSY's Cayley/mixed-subdivision of Delta_6, or the decomposition-polyhedron vertex search).
The brute lattice search is reaching its limit; the genuine next step is the Cayley subdivision theory (hard) -- OR
accept that all computational evidence (every feasible OUT) leans toward max_7 NOT being 2-layer, while noting we
CANNOT prove that (no real lower-bound tool; AHM mod-2 only kills odd-denominator).
