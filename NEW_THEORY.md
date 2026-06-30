# New theory: a bounded-weight normal form for max_n (our own program)

The general normal-form question (real -> rational at fixed depth) is AHM Question 18, open. But max_n has maximal
symmetry, and our data exposes a SHARP, NARROWER phenomenon nobody has stated. We develop a max_n-SPECIFIC theory.

## The empirical foundation (exact)
- max_2,3,4,5,6 are ALL representable on the WEIGHT-2 lattice (BBHSY max_5 weight-2; our max_6 weight-2, denom 90,180).
- max_7 is exactly OUT of weight-2, weight-3, and weight-4 (<=2v, <=3v, root <=2-gen; <=4v running). [n=7,8 weight-2
  exact confirmation running -> w2_transition]
So there is a sharp transition: low max_n live at weight 2; max_7 does not.

## The conjecture (ours)
**BOUNDED-WEIGHT NORMAL FORM (for max_n).** If max_n in V_2 (real weights), then max_n is representable by an
S_n-symmetric signed sum of joins-of-two-zonotopes whose blocks lie on the weight-<= W(n) lattice, with W(n) an
EXPLICIT, slowly-growing bound.

Why plausible (unlike the general AHM Q18): (a) Lemma 1 (symmetrization) already restricts WLOG to S_n-orbit-summed
reps; (b) max_n's normal fan is the braid fan, the coarsest possible, so a representing combination cannot need
arbitrarily fine non-braid structure without it cancelling -- intuitively bounding the block complexity; (c) every
known construction (n<=6) is weight-2; BBHSY's general method grows the subdivision only like log_3, suggesting W(n)
grows slowly.

## THE REDUCTION (why this is the whole game)
If the bounded-weight normal form holds with an explicit W(n), then **max_n in V_2 is DECIDABLE**: run the exact
S_n-orbit-sum mod-p membership test over the complete weight-<=W(n) family.
  - IN  => extract the dyadic rational certificate (extract_certificate.py) + chamber proof => max_n in V_2.
  - OUT => max_n NOT in V_2 => the first real-weight depth lower bound.
We have already decided weight <= 4 (OUT for n=7). So a proof of W(7) <= 4 INSTANTLY gives max_7 not in V_2. Even a
larger explicit W(7) reduces the problem to a finite (if big) computation.

## CORRECTION (2026-07-01): the braid-spline module does NOT bound weight (tool retracted)
Initial proposal: bound W(n) via Castelnuovo-Mumford regularity of the Billera-Rose / Schenck braid-spline module.
ON EXAMINATION THIS DOES NOT WORK. The spline module is graded by POLYNOMIAL DEGREE on a FIXED fan; the ring action
(multiply by a linear form) raises polynomial degree but does NOT refine the fan. Our "weight" is FAN REFINEMENT
(lattice dilation), a DIFFERENT parameter. So spline-module regularity bounds the wrong grading and does not give
W(n). The framing conflated polynomial degree with weight. Tool retracted.

Also: the S_n-IRREP decomposition of the obstruction is necessarily TRIVIAL -- max_n is S_n-symmetric and Lemma 1
restricts to symmetric reps, so the obstruction lives entirely in the trivial irrep (= the orbit-sum membership we
already have). No extra structure there.

## THEOREM (new, clean): forced negativity via simplex indecomposability
Use the support-function <-> Minkowski dictionary. Split a rep by sign:
   h_Delta + sum_{c<0} |c_t| h_{Q_t} = sum_{c>0} c_t h_{Q_t} + linear   <=>   Delta + sum_neg |c_t| Q_t = sum_pos c_t Q_t.
NONNEGATIVE case: if all c_t>=0 then sum c_t Q_t = Delta (Minkowski), so each Q_t is a Minkowski SUMMAND of Delta.
A simplex (dim>=2) is Minkowski-INDECOMPOSABLE (Shephard/McMullen): its only summands are homothets lambda*Delta. And
Delta_{n-1} is a join of two zonotopes only for n<=4 (= join of two segments = tetrahedron); for n>=5 the simplex's
affinely-independent vertices cannot be covered by two centrally-symmetric (zonotope) vertex sets.
=> THEOREM: for n>=5, max_n requires NEGATIVE coefficients in any 2-layer rep (genuine signed/virtual combination).
Matches data exactly (n=4: 1 block, no neg; n=5,6: neg required).

CONSEQUENCE -- this LOCALIZES the whole difficulty and identifies the correct grading:
- NONNEGATIVE reps => weight BOUNDED (summands of Delta fit inside Delta, weight<=1). Trivial.
- SIGNED reps: Delta + sum_neg Q_t = sum_pos Q_t; both sides can be ARBITRARILY LARGE as long as the negative side
  cancels, so the Q_t have unbounded weight PURELY from cancellation. The entire obstruction is cancellation in a
  VIRTUAL (signed) Minkowski identity.
So the correct grading is by VIRTUAL-MINKOWSKI COMPLEXITY (not spline/polynomial degree, not plain weight): the right
tool is the DECOMPOSITION POLYHEDRON (Brandenburg-Grillo-Hertrich arXiv:2410.04907), whose vertices are MINIMAL
difference-of-convex (virtual) decompositions, bounded by Schrijver. That object IS graded by virtual complexity,
which tracks weight.

REMAINING GAP (stated honestly): the decomposition polyhedron bounds the minimal DC decomposition, but 2-layer-ness
needs the convex parts to be SUMS OF JOINS. Minimal-in-DC-complexity != minimal-among-join-realizable. Closing this
-- reduce a join-realizable signed rep to a minimal (bounded) one while STAYING join-realizable -- is the precise
remaining theorem (and the precise form of W(n) finiteness).

## What a CORRECT tool would need to grade by (superseded by the above)
The bound is on FAN REFINEMENT / lattice dilation (weight), so candidate frameworks are those that grade polytopes
by SIZE/dilation, not polynomial degree:
- McMullen's polytope algebra graded by DIMENSION/dilation (the weight ~ the dilation factor of the blocks).
- The SECONDARY / FIBER POLYTOPE of subdivisions of the simplex (Billera-Sturmfels): the structure of all polyhedral
  subdivisions of Delta, whose "fineness" is the weight; bounding the subdivision needed = a fiber-polytope statement.
- A direct 2-adic / denominator bound (AHM-style) translated to a dilation bound.
None of these is a ready off-the-shelf regularity result; the bounded-weight normal form genuinely lacks machinery.
Honest status: the CONJECTURE + REDUCTION (below) are sound and valuable; the PROOF TOOL is open.

## Two supporting sub-attacks (independent, also new)
- WALL-DESCENT as a module syzygy: our wall taxonomy (joins inject non-braid bridge walls) is, in module language,
  a statement about the relations (syzygies) among the generators. A finite generating set of syzygies (the
  "valuation circuits") that lower complexity would give the descent => bounded weight. The Tits-algebra action
  (Bastidas) is the tool to bound these syzygies.
- SUBMODULAR-CONE FACE membership: max_n's simplex is a submodular function; zonotope (graphical) ones are a
  sub-cone (faces). Bounded weight <-> the simplex's distance to the graphical sub-cone is realized by bounded
  generators. (Coxeter submodular functions, arXiv:1904.11029.)

## Status / next concrete steps
1. [running] Confirm the weight-2 transition exactly at n=7,8 (foundation).
2. Compute the S_n-irreducible decomposition of the weight-2 OBSTRUCTION at n=7 (the residual / dual certificate):
   which S_7-irreps carry the obstruction. This identifies the module generators that "first fail" at n=7 -- the
   seed of the generator-degree bound.
3. Set up the braid-spline module Hilbert function for small n (n=3,4,5) and locate max_n's degree; look for the
   pattern that predicts W(n).
If a clean generator-degree bound emerges, we have a real theorem (decidability of max_n in V_2) -- and applied to
our weight<=4 OUT data, potentially the first real-weight lower bound for max_7.
