# Proof attempts on the depth-2 max_n lower bound, and exactly why each dies

Target: max_n (n >= 7) is not computable by any depth-2 ReLU network. Equivalent to the weight-2 normal
form (a 2-layer max_n is weight-2 representable), equivalent (n=7) to the separation itself.

The recurring obstacle: weight-w building blocks form a STRICTLY GROWING family (rank(weight-3 span) >
rank(weight-2 span); confirmed numerically). So higher-complexity blocks are genuinely independent new
functions -- they are never redundant in general. The conjecture is therefore NOT "high complexity is
redundant"; it is "max_n is special: it stays outside every one of the growing spans." Any proof must use
something specific to max_n, not a generic reduction.

## Route 1 -- local first-order (Hessian on codim-1)
At one hidden layer this WINS: a homogeneous 1-layer function is sum_a D_a |a.x|; its Hessian is a sum of
mutually singular measures on hyperplanes a^perp; matching max_n (Hessian on braid hyperplanes) forces
every a to be a root. Hence 1-layer max_n iff n<=2. PROVEN.
Why it dies at 2 layers: P2 blocks are atomic. A complexity-2 bridge is internal to a block; bridges
cancel only collectively (sum_t c_t * bridge_length = 0 per non-braid hyperplane), so the first-order
Hessian is blind to them. Confirmed: max6 provably requires non-conforming (non-braid-bridge) blocks.

## Route 2 -- local higher-order (codim-2 monodromy, ..., codim-k)
A complexity-2 bridge links two braid hyperplanes (visible at codim-2); complexity-3 would link three.
Idea: match max_n's codim-2 tripod structure. Why it dies: atomicity RECURS at every codimension -- a
block's high-complexity face cancels collectively at each stratum independently. No single-codimension
(local) argument can work; this was proven-in-spirit, not just observed.

## Route 3 -- Cayley lift + summand closure (global, linear)
The Cayley trick lifts joins to Minkowski sums in R^{n+1}, so the classical summand-closure theorem
(summands of an F-deformation are F-deformations) now APPLIES. Why it dies: summand closure goes the WRONG
direction -- it keeps summands in their cone, it does not COARSEN a general (fine) block down to weight-2.
The lift linearizes the problem but does not deliver the reduction.

## Route 4 -- dimension / saturation
If the weight-w span saturated (weight-w = weight-(w+1)) the family would be finite -> decidable. Why it
dies: it does NOT saturate; the span strictly grows with w. So no finite family from saturation.

## Route 5 -- reduce a weight-3 block modulo max_n
If every weight-3 block h_R were in span(weight-2 blocks, max_n, linear), a rep using weight-3 blocks would
collapse to weight-2. Why it dies: weight-3 blocks are NOT in span(weight-2, max_n) -- already false at
n=6 (max6 is in the weight-2 span, so adding it changes nothing, yet weight-3 blocks are outside).

## Route 6 -- universal dual certificate
We have an exact lambda with lambda . (weight-2 orbit sums) = 0 and lambda . max7 = 1. A UNIVERSAL lambda
(annihilating all-complexity orbit sums) would prove the separation. Why it dies: "annihilate all blocks"
is infinitely many conditions (all zonotopes, all complexity); existence of such lambda IS the separation,
not a reduction to it. Our specific lambda does not extend (checked: it is nonzero on weight-3 orbit cols).

## Route 7 -- a priori width bound + existential theory of the reals (the current best framing)
For FIXED width W, depth-2 representability of max_n is a semialgebraic feasibility (ReLU relation is
degree 3), hence decidable over the reals by Tarski-Seidenberg -- NO rationality assumption needed (this
sidesteps the oriented-matroid / irrationality obstruction entirely). So:
   SEPARATION <== a computable a priori width bound W*(n) for depth-2 exact representation.
Why it is open: a depth-2 network's parameters depend on the weights with DEGREE 2 (products across
layers), so the realization set is semialgebraic, not a rational polyhedron; Cramer-style denominator/size
bounds fail. No a priori width bound for fixed-depth exact representation exists in the literature (2026).
This is the precise missing lemma. Bounded width DOES give bounded #generators-per-block (<= w1) but the
generators are real vectors of unbounded magnitude/direction, so it does not pin a finite lattice.

## The lead idea: cross-codimensional rigidity (turn atomicity into a weapon)
Atomicity defeats every LOCAL argument (Routes 1,2). But a block is ONE polytope, so its faces across all
codimensions are RIGIDLY linked. A representation must cancel non-max structure SIMULTANEOUSLY at every
codimension. Single-stratum: under-determined (cancellation free). All strata at once: potentially
over-determined (the chains lock). Conjecture: the joint multi-stratum cancellation admits only finitely
many block types -> bounds complexity -> bounds width. Tool: McMullen's polytope algebra (tracks the full
stratified face structure as one algebraic object) -- exactly the decomposition-polyhedra setting. This is
the only route that exploits atomicity rather than being defeated by it. Not yet executed.
