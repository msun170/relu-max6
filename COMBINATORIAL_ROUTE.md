# The combinatorial route to the depth-2 lower bound

The density finding (`PROOF_ROUTE.md`) rules out the measure/functional route for exact non-representability,
and the only unconditional `max` lower bounds in the literature are combinatorial (2601.01417: first-layer
hyperplanes -> ridge cliques -> Turan, for depth >= 3). This note sets up the combinatorial route for depth 2
and records the key enabling fact: the depth-2 width to be lower-bounded actually grows.

## Setup

A depth-2 representation is `max_n = sum_t c_t h_{Q_t}`, `Q_t` a join of two zonotopes. Two widths:
- **first-layer width** = number of distinct generator directions (segments) across all zonotopes;
- **second-layer width** = number of join units = number of blocks `W` = the **join rank** of `max_n`.

A Turan-style lower bound aims to show one of these must grow with `n`. For that to be possible the quantity
must actually grow -- otherwise there is nothing to prove.

## The second-layer width grows (the enabling fact)

`fast_min_block.py` computes the minimum number of blocks (join rank) for `max_n`.

- **Rigorous:** join rank `= 1` iff `n <= 4`. `Delta_3 = conv(point u segment)` and `Delta_4 = conv(segment
  u segment)` are single joins; `Delta_5` (4-simplex) has 5 vertices in general position, which conv of two
  centrally symmetric polytopes cannot produce (a centrally symmetric piece with >= 3 vertices has 4 coplanar
  points), so join rank `>= 2` for `n >= 5`.
- **Computed (weight-2 model):** join rank `= 1, 1, >=4, >=3` for `n = 3,4,5,6` (k<=2 ruled out exhaustively;
  a 2-million-triple randomized search found no 3-block representation of `max_5`).

So the depth-2 second-layer width is not bounded -- it grows past 1 at `n = 5`. There is a genuine width
lower bound to prove, of exactly the kind 2601.01417 proves at depth `>= 3`.

(Caveat: the `>= 3, >= 4` figures are in the weight-2 lattice model; the real-weight join rank is `<=` these.
But `>= 2` for `n >= 5` is unconditional, and `= 1` for `n <= 4` is exact and real.)

## What a Turan-style argument keys on, and why depth 2 resists

At each codimension-2 braid cone `C_{ijk}` (where `x_i = x_j = x_k` are the tied max), `max_n` restricts to
`max(x_i, x_j, x_k)` -- locally the support function of a **triangle**, an asymmetric 3-fold ridge. There are
`C(n,3)` such triangle corners. From the `max6` dissection (`STRUCTURE_MAX6.md`): the asymmetric (triangle)
content at a corner must be supplied by a **bridge** face of a join (a zonotope 2-face is a centrally
symmetric parallelogram = a 4-fold "X", never a 3-fold "Y"). So the bridges ARE the ridge structure a
clique/Turan argument counts, and the codim-2 corners are the cliques.

Two obstacles specific to depth 2:
1. **The Turan bound of 2601.01417 diverges at `k = 2`.** Its width bound `Omega(d^{1+1/(2^{k-2}-1)})` has
   exponent `-> infinity` as `k -> 2` (the `2^{k-2}-1` denominator vanishes), so the technique gives nothing
   at depth 2. Depth 2 is genuinely outside their method, not a tweak away.
2. **Cancellation.** The signed output sum lets ridges cancel, so a ridge of `max_n` need not lie on a
   first-layer hyperplane -- it can emerge from cancelling join creases. This breaks the direct
   ridge-to-hyperplane (clique) correspondence the argument relies on. This cancellation is exactly the
   `(NB)` condition / the `J+NB` over-determination we found: at depth 2 the bridges must self-annihilate at
   codimension 1 while the roots assemble the simplex, and these are individually satisfiable but jointly
   rigid.

## The target

Prove a growing depth-2 width lower bound for `max_n` by a combinatorial argument over the `C(n,3)` triangle
corners and their bridge (Y-ridge) requirements, that **accounts for cancellation** -- i.e., turns the
`J+NB` rigidity into a Turan/Kovari-Sos-Turan-type incidence bound. This is the open frontier (no one has
depth 2), but unlike the measure route it is not blocked by density, it matches the only technique that
proves `max` lower bounds, and it targets a quantity (width) we have now verified actually grows.
