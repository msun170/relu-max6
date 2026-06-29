# Toward a normal-form lemma: turning the max7 search into a depth lower bound

## Why this is the whole game

Our engine decides, for a fixed finite family of building blocks, whether `maxn` is a signed sum of
their support functions. For `n <= 6` it finds one; for `n = 7, 8` it does not, even on rich families.
A *failed* search is only a lower bound if the family is provably exhaustive. The bridge from
"computational evidence" to "theorem" is therefore a single statement:

> NORMAL-FORM LEMMA (target). If `maxn` is computable by a ReLU network with two hidden layers, then it
> is a signed sum of support functions of joins of two zonotopes whose generators are braid directions
> `e_a - e_b` and whose vertices lie in a bounded lattice (conjecturally the weight-2 lattice of
> `2*Delta^{n-1}`).

COROLLARY. Two-hidden-layer representability of `maxn` becomes decidable by the finite linear-feasibility
computation we already run. In particular, our `n = 7` infeasibility would upgrade to: `max7` requires at
least three hidden layers. That is the first separation of two from three hidden layers over the reals,
the central open problem of the area.

## Definitions and the working characterization

A 2-hidden-layer ReLU network computes exactly the functions of the form `f = h_Q - h_P + (linear)` with
`P, Q in P2`, where `P0` = points, `P_{k+1} = { sum_i (A_i join B_i) : A_i, B_i in P_k }`, `join` =
convex hull of union, sum = Minkowski. `P1` = zonotopes. Using `h_{A+B} = h_A + h_B` and
`h_{conv(A u B)} = max(h_A, h_B)`, this is equivalent to: `f` plus a linear term is a signed linear
combination of building blocks `h_R`, `R = conv(Z1 u Z2)` a join of two zonotopes. We want to represent
`maxn = h_Delta`, `Delta = conv{e_1, ..., e_n}`.

Two standard facts we use freely: support functions are additive under Minkowski sum, and two polytopes
with equal support functions are equal.

## The reduction chain (Parts 1-3 are proved below; Part 4 is the crux)

### Part 1: symmetrization is free (proved)

CLAIM. If `maxn` has a 2-layer representation, it has an `Sn`-symmetric one (a signed sum of full
`Sn`-orbits of building blocks).

PROOF. The 2-layer functions are closed under (a) the `Sn` action on inputs, since `f` composed with a
coordinate permutation is computed by the same network with permuted first-layer weights, and (b) finite
signed sums, since concatenating the building-block lists of `f` and `g` (with coefficients) represents
`f + g`. Hence the average `(1/n!) sum_{sigma in Sn} f compose sigma` is again 2-layer. If `f = maxn`,
which is `Sn`-invariant, this average equals `maxn` and is `Sn`-symmetric. QED.

Consequence: we may search over `Sn`-orbits of building blocks, exactly as the engine does.

### Part 2: reduction to a Minkowski equation (proved)

CLAIM. `maxn` is 2-layer iff there exist `P, Q in P2` with `Delta + P = Q`.

PROOF. `maxn` 2-layer iff `h_Delta = h_Q - h_P + (linear)` for some `P, Q in P2`. A linear functional is
`h_{t}` for a point `t`, and `h_Delta + h_P + h_t = h_{Delta + P + t}`. So the identity reads
`h_{Delta + P + t} = h_Q`, i.e. `Delta + P + t = Q`. Translating absorbs `t`. QED.

### Part 3: everything refines the braid fan (proved)

The braid fan `B` is the complete fan in `R^n` whose maximal cones are the closures of
`{x : x_{pi(1)} > ... > x_{pi(n)}}`. Its support is partitioned by the braid hyperplanes `x_i = x_j`.

CLAIM. The normal fan of `Delta` is the braid fan, and for any `P`, the normal fan of `Q = Delta + P`
refines the braid fan.

PROOF. `h_Delta(x) = max_i x_i` is linear exactly on each braid chamber `{x_i >= x_j for all j}`, where
its gradient is `e_i`; so the normal fan of `Delta` is `B`. The normal fan of a Minkowski sum is the
common refinement of the normal fans of the summands, so `NF(Q) = NF(Delta) ^ NF(P) = B ^ NF(P)`, which
refines `B`. QED.

So `Q` is a `P2` polytope whose normal fan refines the braid fan, and likewise (subtracting) the building
blocks must collectively align with the braid fan. This is precisely the "conforming to the braid
arrangement" condition studied by Grillo, Hertrich, and Loho [GHL].

### Part 4 (the crux): bounded lattice. (Conformity ruled OUT below.)

4a (conformity WLOG) -- RULED OUT by engine probe `conforming_probe.py`. A block is braid-conforming iff
its argmax vertex is constant on each open braid chamber (equivalently all edges are braid directions).
Of the 78 weight-2 orbits for `n = 6`, only 23 are conforming, and `max6` is NOT in their span (residual
6.2). So `max6` provably REQUIRES non-conforming building blocks: the per-block nonlinearities inside a
braid chamber cancel in the sum. Consequences:
  - "WLOG braid-conforming" is FALSE, so the route "upgrade [GHL]'s conditional bound to unconditional via
    a conformity normal form" is dead. This also explains WHY our construction evades [GHL]: it is a
    genuinely non-conforming network, which is exactly the case their conditional bound does not cover.
  - The normal form is therefore NOT about conformity. It is about the LATTICE.

4b (bounded lattice) -- THE live crux. The construction we have uses only weight-2 vertices (joins of
zonotopes with braid generators `e_a - e_b`; bridges may be non-braid, which is fine, that is what makes
the blocks non-conforming). The target:

> LATTICE NORMAL FORM. If `maxn` is computable in two hidden layers, it has a representation with all
> building-block vertices in the weight-2 lattice `{2 e_i, e_i + e_j}` (up to translation/scaling).

This is consistent with the probe (it does NOT require conformity) and with the sanity checks: weight-1
is too small (`max6` fails there), weight-2 suffices for `max5, max6`, weight-3 does not rescue `max7`.
The "2" in weight-2 matching "2" hidden layers is the Bakaev et al. subdivision intuition; the right tool
for the bound is the lattice-polytope machinery of [HHL], which controls exactly these objects in the
integer case. There is no conformity shortcut; this is the open mathematics.

## Status of the route

The clean GHL upgrade is gone (Part 4a false). The live program is the lattice normal form (4b), proved
via [HHL]-style lattice-polytope arguments, with no conformity assumption. This is harder and less
packaged than the conformity route, but it is the honest target, and the engine now lets us test any
proposed lattice/complexity bound instantly (as it just falsified conformity).

## Literature synthesis (verified research, 2026-06-28) and the revised program

Key facts confirmed against primary sources:
- [GHL] define "B_d^0-conforming" = every neuron affine on each braid chamber (breakpoints only on
  x_i=x_j or x_i=0). Their Thm 5.2: B_d^0-conforming nets need 3 hidden layers for max5. Since [BBHSY]
  build max5 in 2 hidden layers, the optimal shallow max net is provably NON-conforming. [GHL]'s own
  conclusion says conformity is a real restriction and suggests using a DIFFERENT, finer fan. This
  matches our probe exactly; the conformity route is closed by the authors themselves.
- No published canonical normal form for P_2 exists. The closest machinery is [BGH] "Decomposition
  Polyhedra of Piecewise Linear Functions" (arXiv:2410.04907): for a FIXED polyhedral complex, the
  decompositions f = g - h compatible with it form a polyhedron. Right tool, but it does not show the
  complex can be taken to be ours WLOG.
- Classical deformation/type-cone fact: Minkowski summands of an F-coarsening polytope are themselves
  F-coarsening (generalized permutahedra are closed under summands). This is real and usable, but our
  blocks are JOINS and the representation is a DIFFERENCE (virtual polytope), where summand-closure does
  not transfer. That gap is exactly where the max5 non-conformity counterexample bites.
- [Safran] (arXiv:2601.01417, Jan 2026): unconditional WIDTH bound at depth >= 3 (Turan/extremal graph
  theory), NOT a depth separation; explicitly notes it is unknown whether max6 is computable at depth 3
  (= two hidden layers). So our max6 theorem resolves a problem listed open as recently as Jan 2026.

REVISED PROGRAM. Take the finer fan F = the weight-2 arrangement W (all 2e_i, e_i+e_j difference
hyperplanes). Our 2-layer max constructions ARE W-conforming (their breakpoints lie on W), unlike braid-
conforming. W-conforming is equivalent to "vertices in the weight-2 lattice." The separation then splits:

  (A) NORMAL FORM (open crux): if max_n is 2-layer, it has a W-conforming (weight-2) representation.
  (B) DECISION (finite, in reach): no W-conforming representation of max_n exists for n = 7.

(B) is decidable: W-conforming blocks have vertices in a finite lattice (28 points for n = 7), so there
are finitely many blocks; an uncapped exact enumeration turns our rank-1481 evidence into the theorem
"max7 has no weight-2 / W-conforming two-hidden-layer representation." (A) is the remaining mathematics;
the relevant tools are [BGH] decomposition polyhedra and deformation cones for the fan W, the route [GHL]
themselves point to. The full 2-vs-3 separation = (A) + (B), and (B) is the part we can finish.

## (B) IS NOW DONE (2026-06-28, exact theorem). Plus a methodological correction.

CORRECTION: only TWO-way joins are valid two-layer building blocks. A three-way join conv(Z1uZ2uZ3) has
support max(h1,h2,h3); since max of two 1-layer functions is 2 layers, max of three is 3 layers. So our
earlier frontier searches that switched on 3- and 4-way joins (J3, J4) were testing a LARGER, non-2-layer
space (the rank-1481 / rank-19040 numbers). The correct family is 2-way joins only.

THEOREM (B), exact: max7 (and max8) have NO weight-2 two-hidden-layer representation. Proof (lower_bound.py
/ rigorous_b.py + exact_b7.py): enumerate the COMPLETE finite weight-2 building-block family -- all weight-
2 zonotopes via a pruned DFS over the FULL generator set (all weight-2 point differences, braid AND non-
braid, closing the generator loophole), plus points (degenerate zonotopes, so pyramids appear), then all
2-way joins -- and solve over Q. Result: INCONSISTENT for n=7,8; CONSISTENT for n=6 (recovers max6, so the
method is validated). Confirmed both exactly over Q (core.exact_solve returns None) and mod three large
primes. So the only thing between us and the 2-vs-3 separation is conjecture (A), the lattice normal form.

Counts (n=7): full generators 252, weight-2 zonotopes 588 (vs 378 braid-only -> non-braid zonotopes are
genuinely included), 2-way blocks ~101k, orbits 136; max7 value vector not in the rank-52 column span.

## Computational tests guiding the proof (engine probes)

- P1 (sufficiency of braid generators): every construction the engine finds for `max5`, `max6` uses only
  braid generators, and braid generators alone suffice. Confirmed by construction.
- P2 (necessity of weight-2): CONFIRMED. On the weight-1 lattice the engine FAILS for `max6` (residual
  5.3); it succeeds on weight-2. The right lattice is bracketed at weight-2, matching "2" hidden layers.
- P3 (no help from heavier lattices): CONFIRMED. `max7` stays infeasible on weight-2 (rich, rank 1481),
  weight-3 (rank 255), AND weight-2 u weight-3 (rank 523). The threshold-6 conjecture passed its decisive
  test: heavier lattices do not rescue `max7`.
- P5 (conformity, NEW): RULED OUT the conformity normal form. 23/78 weight-2 orbits are conforming and
  `max6` is not in their span; `max6` requires non-conforming blocks. See Part 4a above.
- P4 (bounded complexity): record the maximum vertex count and generator count over building blocks used
  in minimal representations for `n <= 6`; a uniform bound supports the weaker "bounded-complexity"
  normal form, which also makes the search exhaustive.

## Honest assessment

Parts 1-3 are real and clean. Part 4 is the open mathematics and is hard; it is the same wall that keeps
the field's problem open. Our contribution is a concrete route through it: reduce to braid-conformity
(4a), connect to [GHL]'s existing conditional machinery, and use [HHL]'s lattice-polytope tools for the
lattice bound (4b). Even partial progress (4a for the symmetric, minimal case) plus the hardened
computational evidence makes a strong joint paper with Hertrich's group, whose techniques ([GHL], [HHL])
are exactly the ones this route needs.

References: [GHL] Grillo, Hertrich, Loho, arXiv:2502.09324; [HHL] Haase, Hertrich, Loho, arXiv:2302.12553;
[BBHSY] Bakaev, Brunck, Hertrich, Stade, Yehudayoff, arXiv:2505.14338.
