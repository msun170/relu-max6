# The layer structure of the max6 construction (and what it says about generalizing)

Dissecting the explicit 6-block construction of `max6` (`analyze_max6.py`) shows a clean, uniform mechanism.

## What the 6 blocks are

Every block is a **join of two ROOT zonotopes** -- the two zonotopes have only root edges `e_a - e_b`, so the
*first ReLU layer uses only root directions*. The blocks differ only in size and in the complexity of their
**bridge** (the cross-edges joining the two zonotopes):

| block | coeff | orbit | type | bridge edges |
|------|-------|-------|------|--------------|
| 1 | +1/360 | 360 | pyramid (zonotope + apex), 5 vtx | low |
| 2 | -1/360 | 360 | join of two root segments (tetrahedron), 4 vtx | 1 root, 3 complexity-2 |
| 3 | +1/360 | 180 | pyramid, 5 vtx | low |
| 4 | +1/90 | 180 | join of two root parallelograms, 8 vtx | 6 root, 10 complexity-2 |
| 5 | -1/90 | 180 | join of two root parallelograms, 8 vtx | 8 root, 8 complexity-2 |
| 6 | -1/180 | 90 | join of two root parallelograms, 8 vtx | **16 complexity-2 (all)** |

The two zonotope "halves" are always root (complexity-1 edges); the only complexity-2 edges in the whole
construction are the **bridges**. Block 6 is the purest: its bridge is entirely complexity-2.

## The mechanism (this is the generalizable part)

`Delta_6 = conv{e_1,...,e_6}` has only root edges (complexity 1) and triangular 2-faces. So in the signed
identity `max6 = sum_t c_t h_{Q_t}`:

- the **root** (zonotope-half) content of the blocks must, after signed cancellation, assemble exactly the
  root edges of `Delta_6` -- this is the **(J)** matching of `PROOFS.md`/the J-NB split;
- the **complexity-2 bridges** must **cancel completely** in the signed sum (the simplex has no complexity-2
  edges) -- this is **(NB)**.

So the construction is a delicate signed arrangement of root-zonotope joins whose bridges *exactly cancel* while
their root parts build the simplex. The bridges are not incidental: they are the only source of the
asymmetric (triangle) 2-face content the simplex needs, and they must self-annihilate at codimension 1. This
is exactly the J + NB picture, now exhibited concretely on the one construction that *works*.

## What it says about generalizing to max_n

1. **The construction is entirely root-generated (weight-2).** Layer 1 uses only root directions; the bridges
   are complexity-2 differences of weight-2 points. There is no non-root generator anywhere. So the natural
   way to "generalize the construction" is to find an analogous root/weight-2 (or weight-3) assembly for
   `max7` -- and Theorems 2 and 2b prove **no such assembly exists** (`max7` is OUT of weight-2 and complete
   weight-3). The mechanism that builds `max6` provably does not reach `max7`.

2. **The obstruction is the joint J + NB feasibility.** `max6` is the largest `n` for which one can
   simultaneously (J) assemble the simplex's roots and (NB) cancel all bridges using weight-2 root-zonotope
   joins. The jump_test experiments showed that for `max7` these two are individually satisfiable but jointly
   over-determined. The explicit `max6` blocks are the witness that they *can* be jointly satisfied at `n=6`.

3. **Threshold conjecture, sharpened.** Combined with `max5,max6` IN and `max7` OUT of weight-2 and weight-3,
   the data says the 2-hidden-layer threshold is `n = 6`, independent of lattice refinement. Proving it stops
   at 6 for *all* weights is the scale-invariant obstruction of `SCALE_INVARIANT_ROUTE.md`: a measure killing
   every join-of-zonotopes (equivalently, killing every `|h_{Z_1} - h_{Z_2}|`) but not `max_n` for `n >= 7`.

## Natural next step

Compute the explicit `max5` construction the same way and compare: if the `max6` blocks are `max5`'s blocks
plus a bounded set of "correction" bridges, the recursion `max_n = max(max_{n-1}, x_n)` has a concrete block-
level form, and the point where the correction can no longer cancel is the threshold. That is the inductive
handle on generalizing.
