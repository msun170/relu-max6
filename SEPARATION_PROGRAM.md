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
