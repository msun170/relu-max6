# Where the proof of the separation stands, and which route can work

Synthesis after (a) exhausting the lattice computation at weight-3, (b) developing the measure-certificate /
self-similar hierarchy, and (c) a literature + empirical check on whether that route can succeed.

## The 2-vs-3 separation is still open (literature, June 2026)

- Bakaev-Brunck-Hertrich-Stade-Yehudayoff, "Subdividing the Simplex" (arXiv:2505.14338): two hidden layers
  suffice for `max5`; upper bound `ceil(log3(n-2)) + 1` layers for `max_n`. Disproves the old `ceil(log2(n+1))`
  conjecture.
- Grillo-Hertrich-Loho, braid arrangement (arXiv:2502.09324): bounded-rank-maxout depth results; states the
  general ReLU lower bound "is still 2".
- "A Depth Hierarchy for Computing the Maximum ... via Extremal Graph Theory" (arXiv:2601.01417, Jan 2026):
  proves the FIRST unconditional super-linear WIDTH lower bound for `max` at depths `3 <= k <= log2 log2 d`,
  via associating the maximum's non-differentiable ridges with cliques in a graph from the first hidden layer
  and applying Turan's theorem. It does NOT address depth 2.

So no one has the depth-2-vs-3 separation. Our results (`max7` OUT of weight-2 and complete weight-3,
exact) sit at the genuine frontier, and `max5,max6` IN is the established upper boundary.

## Why the measure-certificate route is probably the wrong tool

The scale-invariant route (`SCALE_INVARIANT_ROUTE.md`, `HIERARCHY.md`) seeks a measure `mu` annihilating all
2-layer functions with `mu(max_n) != 0`. Such a `mu` exists only if `max_n` is NOT in the closure of the
2-layer span. Two warnings:

1. **Universal approximation.** ReLU nets with biases are dense in `C(K)`; the route survives only because the
   `max_n` framework is homogeneous (no biases), where 1-layer functions are the (non-dense) range of the
   cosine transform. Whether the homogeneous 2-layer span is dense is exactly the open question -- and if it
   is dense, no separating measure exists.

2. **Empirical evidence it IS dense.** `max7` lies at normalized distance 0.0017 from the (52-dimensional)
   weight-2 span, versus 0.27 for a generic max-of-7-linears -- about 160x closer than a control. So although
   `max7` is exactly OUT of weight-2 and weight-3, it is *almost* representable, strongly suggesting
   `max7 in closure(V_2)`. If so, `max7` is approximable to any precision by 2-layer functions, no continuous
   functional separates it, and the measure-certificate route cannot prove exact non-representability.

This matches the k=1 vs k=2 contrast: for k=1 the certificate works because the simplex is genuinely FAR from
zonoid-differences (zonoids are closed, the simplex is not a zonoid). For k=2 the 2-layer span appears dense,
so the same idea stalls.

## The route that is actually proving things: combinatorial / first-layer

The only unconditional `max` lower bounds (2601.01417) come from a COMBINATORIAL analysis of the first hidden
layer: its hyperplanes induce a graph, the maximum's ridges force cliques, and Turan's theorem bounds the
width. This is not a functional/measure argument. It strongly suggests that the exact depth-2 lower bound,
if provable, will be combinatorial -- about the first-layer hyperplane arrangement and how the simplex's
codimension-2 ridge pattern (its triangular 2-faces) cannot be covered -- not a separating measure.

Our assets feed directly into this: the weight-2/weight-3 lower bounds, the `max6` dissection (root-zonotope
joins with complexity-2 bridges), and the J+NB rigidity are all combinatorial statements about edges and
ridges. The bridges ARE the codim-2 ridge structure the Turan-style argument keys on.

## Next step

Attempt the depth-2 lower bound for `max7` combinatorially, in the spirit of the ridge-clique-Turan technique:
formalize "a depth-2 width-W network's first layer induces a hyperplane graph; representing `max7` forces a
configuration of cliques / codim-2 incidences that a bounded first layer cannot realize." This is the open
frontier (2601.01417 reaches depth 3 but not 2), so it is genuinely hard -- but it is the route that matches
both the successful literature technique and our own combinatorial structure, and unlike the measure route it
is not blocked by density. The measure/hierarchy work remains a correct structural reframing and a clean
account of why depth-2 resists depth-1 tools, but it is unlikely to yield the exact separation by itself.
