# The global-gluing obstruction for max_n (single-chamber invisibility)

A conceptual note behind the max_7 depth question. The one clean, unconditional theorem so far is the
single-chamber invisibility lemma; the rest is the honest structural picture it supports.

## 1. Single-chamber invisibility lemma

Let `D = {x : x_1 >= x_2 >= ... >= x_n}` be the (closed) braid chamber. On `D`,
```
max_n(x) = x_1,
```
which is a LINEAR function.

**Lemma.** Let `lambda` be a linear functional realized as a finite signed combination of point
evaluations, all of whose evaluation points lie in a single open braid chamber, and suppose `lambda`
annihilates every linear function. Then `lambda(max_n) = 0`.

*Proof.* On one open braid chamber, `max_n` agrees with a single coordinate `x_{sigma(1)}` (the largest),
a linear function. Since all evaluation points lie in that chamber, `lambda(max_n) = lambda(x_{sigma(1)})
= 0` because `lambda` kills all linear functions. QED.

**Corollary (certificates are cross-chamber).** Any separating certificate for a symmetric P2 orbit-span
plus linear terms, i.e. any `lambda` with
```
lambda(h_Q) = 0  for all blocks Q,     lambda(x_i) = 0  for all linear,     lambda(max_n) != 0,
```
must use evaluation data from at least two braid chambers.

*Proof.* Its restriction to any single chamber annihilates the blocks and the linear functions and, by
the Lemma, would give `lambda(max_n) = 0`; so no single-chamber piece can carry the `!= 0`. QED.

This is the first exact statement behind the "global gluing" language: local linear pieces of `max_n`
exist chamber by chamber, but the obstruction to a symmetric P2 representation is a compatibility
condition ACROSS chambers.

## 2. Symmetric orbit-span criterion (why the domain restriction is honest)

By the symmetrization lemma (Lemma 1 of the project), `max_n in V_2` iff it has an `S_n`-symmetric
representation (orbit-summed blocks + a symmetric, i.e. `c * sum_i x_i`, linear term). If such a
symmetric `F = sum_t c_t h_{Q_t} + c * sum_i x_i` agreed with `max_n` on all of `D`, then for any `x`,
sorting `x` into `D` by some `sigma` gives `F(x) = F(sigma x) = max_n(sigma x) = max_n(x)` (both sides
symmetric), so `F = max_n` everywhere and `max_n in V_2`. Contrapositive: if `max_n` is OUT, then no
symmetric `F` agrees with it as a FUNCTION on `D` -- but a finite sorted SAMPLE can under-saturate the
piecewise-linear block cells inside `D` and show a spurious fit. So `D`-restricted sampling is only a
sanity probe; a genuine certificate must be cross-chamber (Section 1).

## 3. Finite-family gluing obstruction

For a fixed finite `S_n`-stable P2 family `F`, `max_n in span(F) + linear` iff the exact membership system
is feasible, iff the P2-restricted decomposition polyhedron `D_F^{P2}(h_Delta)` is nonempty (Delta + N = M
as a virtual Minkowski identity with M, N nonneg combinations of blocks in `F`). Emptiness for a finite
family = a finite-family obstruction; the open REAL question is whether emptiness persists as `F` grows to
all real P2 blocks (AHM Question 18 in our specific form). Section 1 says any certificate of emptiness is
cross-chamber; it does NOT by itself say the emptiness persists.

## 4. Weight-2 exact certificate (what we have)

At complete weight-2 (121 orbit blocks), an EXACT floating-point-free dual certificate exists and is
verified: `lambda . h_Q = 0` (all 121), `lambda . x_i = 0` (all 7), `lambda . max_7 = 1`. So `max_7` is OUT
of complete weight-2, with a checkable rational certificate (not just a mod-p membership test).

## 5. Why the raw lambda is not yet a named invariant

The naive particular solution has a ~200-digit denominator and dense support. By the min-support analysis
(MIN_CERT_WEIGHT2.md), the minimal support of ANY such certificate is `rank(M)+1` for generic points, where
`rank(M)` is the functional dimension of the weight-2 P2 orbit-span plus linear; this is `Theta(#blocks)`,
i.e. inherently large. So the raw dual certificate is a basis artifact of a high-dimensional nullspace, not
a sparse local invariant -- consistent with, and explained by, the cross-chamber nature in Section 1.

## 6. Connection to P2-restricted decomposition polyhedra

The correct continuous object is the P2-restricted decomposition polyhedron (Brandenburg-Grillo-Hertrich
arXiv:2410.04907, restricted to join-of-two-zonotope-realizable decompositions). Membership = nonemptiness;
OUT = a face/vertex certificate of emptiness. Because the obstruction is cross-chamber (Section 1), the
promising certificates live there, not in single-chamber/local-linear tests. This is the recommended
theory route (route A).

## Careful scope (what NOT to claim)

Do NOT say "this is why every local tool fails." The precise statement is: single-chamber certificates
cannot separate `max_n`; the obstruction must compare how local linear pieces glue across chambers.
Adjacent-chamber wall tests and higher-order finite differences CAN still detect finite-family obstructions
(indeed the weight-2 failure is visible in some codim-1/gradient formulations). The lemma constrains
single-chamber certificates only.
