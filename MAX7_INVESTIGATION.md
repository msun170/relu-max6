# max_7 investigation: what we tried, and the most recent finding

The open question: is max_7(x) = max(x_1,...,x_7) exactly computable by a two-hidden-layer ReLU network over REAL
weights? Equivalently, is h_Delta (the simplex support function) a finite signed sum of support functions of joins of
two zonotopes, plus a linear term. Known: max_2..6 are 2-layer (max_6 is our result); max_7 is open. This document
records the approaches we tried on max_7 and the latest, sharpest finding.

## What is established about max_7 (all exact: two primes, S_7-orbit-sum membership, control-validated)
- max_7 is NOT in COMPLETE weight-2 V_2, and NOT in COMPLETE weight-3 V_2.
- max_7 is NOT in several COMPLETE LOW-COMPLEXITY weight-4 subfamilies: all <=2-vertex blocks (65 orbits), all
  <=3-vertex blocks (1180 orbits), all joins of two <=2-generator root zonotopes <=6 vertices (6081 orbits).
- The COMPLETE 4-vertex weight-4 family (29732 orbits) is too large for exact membership at the needed sampling;
  left UNDECIDED. We do NOT claim "OUT of weight-4."
- max_7 NOT in integer-2-layer is already a theorem (Haase-Hertrich-Loho, ceil(log2 7)=3); our lattice OUTs are
  instances of it. The REAL question is untouched by any of this.
- max_7 in V_3 trivially (max_7 = max(max_6, x_7), one extra max/ReLU layer over the 2-layer max_6).

## What we tried (and what each showed)
1. APPROXIMATION FLOORS (best L2 residual of max_7 against weight-w spans). UNINFORMATIVE: floors slide to 0 as the
   span becomes near-full-dimensional (a density artifact, not closeness to an exact rep). Retired.
2. RESIDUAL / VALUATION CERTIFICATES (the dual functional annihilating a family but not max_7). They COLLAPSE and
   ROTATE with weight (do not converge to a stable invariant) and fall below float resolution. No lower-bound
   invariant; precision-walled. Retired.
3. TEMPLATE MINING (IN direction). OMP best approximant concentrates by COMBINATORIAL TYPE (joins of segments,
   triangles -- 83% of mass) but is DIFFUSE at equivariant-orbit granularity (80 clusters, top-10 = 48%). No small
   or few-coefficient ansatz. The dominant types are a subset of families we proved exactly OUT. IN-via-clean-ansatz
   closed.
4. WALL-CIRCUITS (decompose representability into braid (J) vs non-braid (NB) wall jumps). On COMPLETE families the
   method is sound (validated: max_5,6 FEASIBLE, max_7 weight-2 INFEASIBLE). KEY LESSON: an earlier "infeasible at
   weight 2/3/4" table came from an INCOMPLETE block family and was a FALSE-NEGATIVE artifact; RETRACTED. On complete
   families the wall (codim-1) relaxation is too coarse to obstruct max_7 at weight-3 (see most-recent finding).
5. FORCED-NEGATIVITY THEOREM (proved). For n >= 5, any 2-layer rep of max_n needs a NEGATIVE coefficient: a
   nonnegative rep would make every block a Minkowski summand of (a translate of) the simplex, hence a homothet
   (simplex indecomposable), hence a join of two zonotopes only for n <= 4. So every max_7 rep is a SIGNED/VIRTUAL
   Minkowski identity Delta + N = M; the simplex appears only after CANCELLATION. This is why single-polytope
   (central-symmetry, volume) obstructions fail.
6. DYADIC CONSTRUCTION SEARCH. Since AHM's mod-2 valuation forbids odd-denominator/integer max_7 in 2 layers
   (ceil(log2 7)=3), any 2-layer max_7 is necessarily EVEN-DENOMINATOR (dyadic) -- consistent with BBHSY max_5 and
   our max_6. Searching the dyadic (weight-4) lattice: every feasible complete dyadic family tested came back OUT.
7. CANCELLATION STRUCTURE of the known max_5/max_6 identities. The non-braid bridge-wall cancellation is a single
   FULL-SUPPORT circuit (globally coupled, IRREDUCIBLE) -- it does NOT factor into bounded local valuation moves.
   So a "descent via bounded local moves" proof route is closed.
8. INFINITESIMAL RIGIDITY. Coefficient rigidity (the circuit) does NOT imply geometric rigidity: letting the block
   geometry move, the max_5/max_6 reps have essential deformation dimension 24 / 35 -- they sit on positive-
   dimensional continuous (semialgebraic) families, NOT isolated points. So "representations are discrete objects"
   is FALSE; the right object is a semialgebraic family / the decomposition polyhedron.

## THE MOST RECENT FINDING: the IN mechanism reappears at weight-3, but the obstruction is HIGHER-ORDER
We measured Type-II dim = rank([NB;J]) - rank(NB) on the COMPLETE weight-3 family: the dimension of non-braid-clean
combinations with NONZERO braid effect -- i.e. whether the structural INGREDIENT that builds max_5,6 (a full-support
NB-circuit that produces braid walls) even EXISTS for max_7.
   n=7 weight-2:  Type-II = 0     (mechanism ABSENT)
   n=7 weight-3:  Type-II = 470   (mechanism PRESENT), and the braid target is WALL-LEVEL reachable
   (control: n=6 weight-3 Type-II = 253, machinery validated)
So for n=7 the IN mechanism is ABSENT at weight-2 but PRESENT at weight-3. CRUCIALLY, max_7 is nonetheless EXACTLY OUT
of complete weight-3 (gpu_w3, proven). Reconciling:
   At weight-3 the WALL-LEVEL (codim-1) necessary conditions PASS -- the mechanism exists and the braid target is
   reachable -- YET exact function-membership FAILS. Therefore the obstruction is HIGHER-ORDER: finer than the
   codim-1 walls. The wall relaxation cannot see it; only exact membership does.
Consequences:
- The weight-3 OUT is NOT "the ingredient is missing" (it is present). So the IN route is not blocked by mechanism
  absence; a construction at weight >= 4 (where the mechanism is even richer) is not excluded.
- Wall-based lower-bound arguments (the entire wall-circuit line) are PROVABLY too coarse for max_7 at weight-3 --
  they report feasible. A genuine obstruction must use HIGHER-ORDER (full-function) structure.

## How DEEP is the obstruction? (obstruction ladder, higher_order_obstruction_w3.py)
We measured the FIRST codimension at which max_7's data leaves the complete weight-3 block span, by testing
membership of finite-difference data of increasing order (codim-1 = single 2nd-differences = walls; codim-2 = mixed
2nd-differences = ridges/two-wall), saturating each level. Controls: n=5 weight-2 all FEASIBLE (max_5 IN); n=7
weight-2 codim-1 saturated + INFEASIBLE (first-order obstruction, matches Type-II=0). RESULT for n=7 weight-3:
  codim-1:   rank 2199/2200 (marginal), FEASIBLE.
  codim-1+2: rank 3683/4400 (saturated, margin 717), FEASIBLE.
So max_7's codim-<=2 (ridge / two-wall) data IS matchable -> the obstruction is NOT second-order. Since full
membership is OUT, it lives in codim>=3 / higher structure: the obstruction is DEEP / GLOBAL. (Caveats: finite-diff
is a proxy for codim; the full and codim-3 levels need ~19000 rows to saturate (family near-full-dim) and were not
reached, so the exact order -- codim-3 vs genuinely global -- is not pinned.)

## Where this points (updated)
- IN (construction): not blocked by the mechanism; would live at weight >= 4 (dyadic), where complete exact
  enumeration is currently infeasible (29732 orbits at <=4 vertices already over the membership limit).
- OUT (lower bound): must be HIGHER-ORDER. The ladder REDIRECTS away from mixed volumes / Aleksandrov-Fenchel:
  those are essentially codim-2 (2-face) invariants, and codim-2 data is MATCHABLE here, so mixed-volume-type
  quantities alone cannot obstruct max_7. The obstruction is deeper -> a GLOBAL certificate is needed. The
  best-motivated object is the DECOMPOSITION POLYHEDRON (Brandenburg-Grillo-Hertrich, arXiv:2410.04907): the
  continuous space of difference-of-convex (virtual Minkowski) decompositions, where the OUT is a genuinely global
  (non-bounded-order) property. (A direct exact DUAL CERTIFICATE lambda from the weight-3 OUT, and an analysis of its
  order/support, is the concrete next probe.)
- The bounded-weight normal form (if provable for an explicit W(7)) would make max_7 DECIDABLE by exact membership;
  this is the max_n-specific case of Averkov-Hojny-Merkert Question 18 (real-weight => rational-weight at fixed
  depth), which is open and "almost holds."

Honest status: max_7 in V_2 is OPEN. We have ruled out every wall-level and low-complexity route, proved the rep must
be a signed dyadic virtual identity, and localized the weight-3 obstruction as higher-order. A resolution needs
either a dyadic construction at weight >= 4 or a higher-order (mixed-volume-type) obstruction -- no current tool
delivers either.

## Files
gpu_w3.py (weight-3 OUT), construction_search.py (complete weight-4 <=k-vertex), dyadic_search.py, wall_poscontrol.py
+ weight3_typeII.py (Type-II ladder), virtual_dissect.py + valuation_factorization_max6.py (cancellation = irreducible
circuit), infinitesimal_rigidity_max6.py (not rigid). Context: MAX7_CLASSIFICATION.md, OUT_THEORY.md, NEW_THEORY.md,
LITERATURE.md, RESEARCH_LOG.md (iters 1-50).
