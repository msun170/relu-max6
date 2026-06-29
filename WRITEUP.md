# Two hidden layers compute the maximum of six numbers

This note has three layers, kept separate on purpose: a clean theorem, a separate exact lower bound on a
finite family, and an open conjecture that would connect them.

## Background

A ReLU network with `k` hidden layers computes a continuous piecewise-linear function, and every such
function comes from some ReLU network. The standard test case is the maximum, `max_n(x) = max(x_1,...,x_n)`.
A binary tree of pairwise maxima computes `max_n` with `ceil(log2 n)` hidden layers, and it was conjectured
this is optimal (Hertrich, Basu, Di Summa, Skutella). Bakaev, Brunck, Hertrich, Stade, and Yehudayoff
(STOC 2026) disproved this for `n = 5`: `max5` needs only two hidden layers, not three. They left `max6`
open. (It is still listed as open as of Safran, Jan 2026.) This note resolves `max6`.

A useful fact: a two-hidden-layer ReLU network computes exactly the signed sums of support functions of
joins of two zonotopes (plus a linear term). The support function of a polytope `P` is
`h_P(x) = max over vertices g of <g,x>`, and `max_n = h_Delta` for the simplex `Delta = conv{e_1,...,e_n}`.
So computing `max6` in two layers means writing `h_Delta` as a signed sum of such building blocks.

By the **weight-2 model** we mean the finite family of building blocks whose support-function vertices lie
among the weight-2 lattice points `{2 e_i}` and `{e_i + e_j}` of the dilated simplex `2 Delta^{n-1}`. This
is the lattice underlying the `max5` construction of Bakaev et al.

## Theorem 1: max6 is a two-hidden-layer ReLU network

We give an explicit construction and prove it correct exactly (no floating point in the certificate).

**Construction.** `max6` equals a signed sum of **six S6-orbits** of weight-2 building blocks, with
denominator `360`:
```
max6 = sum over 6 orbits  c_o * (sum over the orbit of h_R),   c_o in {1/360,-1/360,1/360,1/90,-1/90,-1/180}
```
with zero linear part. The six orbit representatives are in `results/construction_max6.txt`. Each `R` is a
join of two weight-2 zonotopes, so each `h_R` is a two-hidden-layer function and the signed sum is two
hidden layers. Expanding the six orbits over `S6` gives 1530 individual building blocks.

**Verification (`check_max6.py`).** Both sides are piecewise linear, so equality reduces to checking
gradients on every cell of an arrangement. The gradient of the construction changes only across the
"weight-2 difference" hyperplanes; by `S6`-symmetry we restrict to the chamber `x1 >= ... >= x6`, and in
gap coordinates only **35 of the 120** difference hyperplanes are unforced. That arrangement has exactly
**2608 cells**. The proof has three exact steps:

1. enumerate all 2608 cells deterministically, each with an exact integer interior point;
2. check, in exact integer arithmetic, that the construction's gradient equals `e_1` on every cell;
3. prove the enumeration complete by exact adjacency closure: every single-flip neighbor of every cell is
   either another enumerated cell or provably empty via an exact rational Gordan certificate (13756
   in-set, 77524 empty, 0 uncertified).

Since the cells tile the chamber and the gradient is `e_1` everywhere, the construction equals `x1 = max6`
on the chamber, hence (by symmetry and continuity) on all of `R^6`. An independent verifier agrees.

## Theorem 2: max7 has no weight-2 two-layer representation

This is a separate, exact, finite-family result. It is **not** the full lower bound; it is restricted to
the weight-2 model.

The weight-2 model is finite: a weight-2 zonotope has vertices among finitely many lattice points, and the
building blocks are the two-way joins of such zonotopes. (Only two-way joins count: a join of three
zonotopes is `max(h1,h2,h3)`, which is three hidden layers, not two.) We enumerate the **complete** family
for `n = 7` (all weight-2 zonotopes via every weight-2 point-difference generator, braid and non-braid;
points included so pyramids appear; all two-way joins), giving 101087 blocks in 136 `S7`-orbits.

**Theorem 2.** `max7` is not a signed sum of these building blocks. Equivalently, no two-hidden-layer ReLU
network whose building blocks have weight-2 vertices computes `max7`.

This is exact. The linear system "is `max7` in the span" is inconsistent over the rationals. We provide a
**dual certificate** (`results/max7_certificate.json`, checked by `check_weight2_max7_infeasible.py`): a
rational vector `lambda` with `lambda . column = 0` for every block-orbit column and every linear column,
and `lambda . max7 = 1 != 0`. Its existence proves `max7` is outside the span. The same holds for `max8`.
For comparison, the method is consistent for `max6` (it recovers Theorem 1), which validates it.

## Discussion: the open conjecture

Theorems 1 and 2 suggest that six is the exact threshold: `max_n` is two-hidden-layer iff `n <= 6`. The
gap between Theorem 2 and a real two-versus-three hidden-layer separation is a single statement.

**Conjecture (lattice normal form).** If `max_n` is computable in two hidden layers, then it has a
representation inside the weight-2 model.

If this holds, Theorem 2 immediately gives that `max7` needs three hidden layers, which would be the first
separation of two from three hidden layers over the reals, a central open problem. We do **not** claim this
is close. It is probably the hardest part of the field-level problem.

One natural route is ruled out: a "braid-conforming" normal form is false, because `max6` itself requires
non-conforming blocks (the conditional lower bounds of Grillo, Hertrich, Loho only apply to conforming
networks). The plausible remaining route is the lattice normal form above; the relevant tools are the
decomposition polyhedra of Brandenburg, Grillo, Hertrich and deformation-cone theory. The difficulty is
that the building blocks are joins and the representation is a difference, where the clean classical
results do not directly apply.

## How to check the results

```
python check_max6.py                      # verifies Theorem 1 (exact; the adjacency closure step is slow)
python check_weight2_max7_infeasible.py   # verifies Theorem 2 via the exact dual certificate
```

## References

Arora, Basu, Mianjy, Mukherjee, ICLR 2018 (arXiv:1611.01491). Hertrich, Basu, Di Summa, Skutella, SIDMA
2023 (arXiv:2105.14835). Haase, Hertrich, Loho, ICLR 2023 (arXiv:2302.12553). Bakaev, Brunck, Hertrich,
Stade, Yehudayoff, STOC 2026 (arXiv:2505.14338). Brandenburg, Grillo, Hertrich, ICLR 2025
(arXiv:2410.04907). Grillo, Hertrich, Loho, NeurIPS 2025 (arXiv:2502.09324). Safran, 2026
(arXiv:2601.01417).
