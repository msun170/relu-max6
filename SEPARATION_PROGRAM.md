# Research program: from the weight-2 lower bound to a depth separation

This is the speculative layer. It is kept separate from the two theorems (see `WRITEUP.md`), which stand
on their own.

## Goal

Prove that some `max_n` requires three hidden layers over the reals. No CPWL function is currently known to
need three hidden layers with unbounded width and real weights; this is a central open problem. `max7` is
the natural candidate.

## What is proven

- **Theorem 1.** `max6` is two-hidden-layer (exact, `check_max6.py`).
- **Theorem 2.** `max7` and `max8` have no two-hidden-layer representation inside the weight-2 model
  (exact, finite family, dual certificate, `check_weight2_max7_infeasible.py`).

Here, by the **weight-2 model** we mean the finite family of building blocks whose support-function
vertices lie among the weight-2 points `{2 e_i, e_i + e_j}`. This is a restriction on which blocks we
allow, not a claim that weight-2 vertices are equivalent to any fan condition: a function can share a
breakpoint fan with different slope values, so "weight-2 vertices" is strictly a restriction on the
building blocks, which is how we use it.

## The gap: one conjecture

**Conjecture (lattice normal form).** If `max_n` is two-hidden-layer, it has a representation inside the
weight-2 model.

Granting this, Theorem 2 says `max7` needs three hidden layers, i.e. the separation. We do not claim this
is close.

## A cleaner, equivalent form: edge complexity

Define the complexity of an edge as the fewest braid roots `e_a - e_b` summing to it. Then a polytope has
weight-k vertices (lattice points of `k*Delta`, up to translation) iff every edge has complexity at most k
(proof: a weight-k vertex difference needs at most k roots by transport; conversely root edges preserve
weight and the 1-skeleton is connected). So the weight-2 model is exactly "every block edge is a sum of at
most two roots," and the conjecture is:

**Conjecture (edge form).** A two-hidden-layer representation of `max_n` can be taken with every block edge
of complexity at most 2.

Structural lemma (verified n=3..9): a weight-k lattice zonotope has dimension at most k (a weight-2
zonotope is a point, segment, or parallelogram, <= 4 vertices). A weight-k point is "k tokens" on the
coordinates; a braid generator moves one token; with k tokens there are at most k independent move
directions, so dimension <= k. Hence the ladder is also: k hidden layers <-> dimension-<=-k building
blocks. Consequence: for n=7 the weight-2 two-way joins have <= 8 vertices, so the 136-orbit family is
genuinely complete and Theorem 2 is independent of any vertex cap.

This is translation-invariant and lives purely in the root lattice. It sits in a ladder:
- one hidden layer: edges of complexity at most 1 (single roots). PROVEN: a one-hidden-layer homogeneous
  function is `L + sum_a D_a |a.x|`; its distributional Hessian is a sum of mutually singular measures on
  the hyperplanes `a^perp`, so matching `max_n` (Hessian on the braid arrangement only) forces every `a`
  to be a root. Hence one-hidden-layer `max_n` iff `n <= 2`, and the weight-1 model is complete.
- two hidden layers: edges of complexity at most 2. Conjectured; confirmed on our `max6` construction
  (zonotope generators are single roots, every bridge is a sum of exactly two roots).

Why the one-layer proof does not extend: P2 blocks are atomic. A complexity-2 bridge is internal to a
block (you cannot delete it without destroying the block), and bridges cancel only collectively across
blocks (`sum_t c_t * (bridge length) = 0` on each non-braid hyperplane). So the first-order Hessian is
blind to them. The natural next tool is second-order / codimension-2: a complexity-2 bridge `r1 + r2` links
two braid hyperplanes, visible where their walls meet (codim 2); a complexity-3 bridge would link three at
once, a codim-2 pattern `max_n` does not have. Making that matching precise is the open crux.

## The separation reduces to a width bound, via the existential theory of the reals

For a FIXED width W, "is `max_n` exactly a depth-2 width-W ReLU network?" is a sentence in the first-order
theory of the reals: `exists weights, for all x, net(x) = max_n(x)`, and the depth-2 parameter space is
semialgebraic (the ReLU relation `y = max(0,x)` is `y>=0 and y>=x and y(y-x)=0`, degree <= 3). By
Tarski-Seidenberg this is DECIDABLE over the reals, with no rationality assumption -- so the realization /
irrationality difficulties never arise. Hence:

> SEPARATION <== a computable a priori width bound `W*(n)` for depth-2 exact representation of `max_n`.

Given `W*(n)`, "is `max_n` 2-layer?" equals "is it 2-layer with width <= `W*(n)`?", a decidable real
sentence; for `max_n` not 2-layer this proves it. A 2026 literature search found no such width bound (none
is known for exact fixed-depth representation; no CPWL function is known to need > 2 hidden layers), and
did not find this `exists-R` reduction written down. So the entire open problem is now this one lemma, on a
new attack surface (semialgebraic geometry) distinct from the lattice/tropical/extremal-graph approaches.
Anchor: training ReLU nets is exists-R-complete (Bertschinger et al., NeurIPS 2023), so the family lives at
exists-R.

## A new angle on the width bound: cross-codimensional (stratified) rigidity

Why single-codimension arguments fail: a block is atomic, so its non-braid bridge cancels collectively at
codim 1 and cannot be removed term by term. But a block is ONE polytope, so its contributions across ALL
codimensions (codim-1 edges, codim-2 two-faces, codim-3, ...) are RIGIDLY linked -- they are the faces of a
single object. A representation of `max_n` must cancel the non-`max` structure SIMULTANEOUSLY at every
codimension. That cross-codimensional consistency is strictly stronger than the codim-1 Hessian condition
that atomicity defeats, and it is rigid (the strata of a polytope are not independent). Conjecture: this
simultaneous multi-stratum cancellation is finitely rigid and bounds the block complexity, hence the width.
This is the genuinely new line: exploit atomicity (which blocked us locally) as rigidity across strata
(which over-determines the global structure).

Concrete evidence for the mechanism (gradient-jump localization of max7's weight-2 obstruction): split the
codim-1 (Hessian) matching into (J) "match max7's gradient jumps across braid walls -- the creases" and
(NB) "cancel the block jumps across non-braid weight-2 hyperplanes -- the bridges." Result: (J) alone is
satisfiable, (NB) alone is satisfiable, but (J)+(NB) together is INFEASIBLE. So the same coefficients
cannot match the creases and cancel the bridges at once -- each stratum is under-determined alone, the
joint system is over-determined. That is exactly the rigidity mechanism, made concrete.

Where this lives (real-weight tooling): not McMullen's polytope algebra (its computable shadow on a fixed
fan is the toric Chow ring, and its headline invariant, volume, fails for real weights). The right
framework is the DECOMPOSITION POLYHEDRA of Brandenburg, Grillo, Hertrich (arXiv:2410.04907): for a fixed
complex, the polyhedron of all difference-of-support-function (difference-of-convex) decompositions of a
CPWL function, valid for real weights, with a finite vertex-enumeration of minimal-complexity
decompositions.

CONFIRMED (computationally): our obstruction IS the bounded-complexity emptiness of this polyhedron. The
weight-2 decomposition polyhedron is D(max_n) = {(a,b) >= 0 : sum_i (a_i - b_i) h_block_i = max_n}, where
g = sum a_i block_i and h = sum b_i block_i are convex, so g - h = max_n is a difference of convex with
weight-2 P2 pieces. We computed (scipy LP): D(max6) is NONEMPTY, D(max7) is EMPTY. So Theorem 2 is exactly
"the BGH decomposition polyhedron of max7 is empty at weight-2 complexity," and the (J)+(NB)
over-determination is its "intersection of two cones is empty" structure, with an explicit separating
certificate. Real-weight generality is automatic (a rational vector is in the real span iff in the rational
span). The framework reframes the open question precisely -- is D(max7) empty at ALL complexity? -- but, like
every tool here, it does not bound the complexity for free; that bound is the lemma.

## What is settled along the way

- Reduction chain (clean, provable): any representation may be `S_n`-symmetrized; two-hidden-layer `max_n`
  is equivalent to a Minkowski equation `Delta + P = Q` with `P, Q in P_2`; and `Delta + P = Q` forces the
  normal fan of `Q` to refine the braid fan.
- Methodological correction: only two-way joins are valid two-layer building blocks. A three-way join is a
  max of three 1-layer functions, hence three hidden layers. Earlier exploratory searches that switched on
  three- and four-way joins were testing a larger, non-two-layer space; the lower bound uses two-way joins
  only.
- Evidence is stated exactly. The non-membership is an exact inconsistency over the rationals with a saved
  dual certificate, confirmed independently mod three large primes. We do not rely on least-squares
  residuals.

## Honest assessment

The braid-conforming route is false: `max6` itself requires non-conforming blocks. The remaining plausible
route is a weight-2 / lattice normal form. We prove the finite part of this program: `max7` is not
representable inside the complete weight-2 `P_2` family. The open problem is whether every two-hidden-layer
representation can be normalized into this family.

The relevant tools for the conjecture are the decomposition polyhedra of Brandenburg, Grillo, Hertrich
(arXiv:2410.04907) and deformation-cone theory. The difficulty is that the building blocks are joins and
the representation is a difference, where the classical Minkowski-summand results do not directly transfer.
