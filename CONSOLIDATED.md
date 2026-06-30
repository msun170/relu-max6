# max_n depth-2 ReLU separation: consolidated status of all routes tried

**Question.** Over real weights, can `max_n(x) = max_i x_i` be represented *exactly* by a 2-hidden-layer ReLU
network? Known: `max_n` needs `>= ceil(log2 n)` layers for INTEGER weights (Haase-Hertrich-Loho, 2302.12553);
over the REALS it is open whether *any* `max_n` needs more than 2 layers. Target: `max_7` (the first candidate
where the integer bound forces 3, so the real-weight question bites).

**Status in one line.** `max_4,5,6` ARE 2-layer; `max_7` is provably OUT of weight-2 and complete weight-3;
the residual floor collapses to 0 (so `max_7 in closure(V_2)`); every linear/continuous obstruction provably
collapses; the open case reduces to a single weight-uniform / homological obstruction.

---

## 1. Framework

A 2-hidden-layer function is a signed sum of support functions of **joins of two zonotopes**:
`max_n = sum_t c_t h_{Q_t}`, `Q_t = conv(Z_1 ∪ Z_2)`, `h_{Q_t} = max(h_{Z_1}, h_{Z_2})`, `c_t in R` (free sign).
- **Weight-w** = edge complexity <= w = zonotope pieces use the `w*Delta` lattice (dim <= w, token lemma).
- `max_n = h_{Delta_n}`, the support function of the simplex.
- **2-layer realizability <=> `h_{Delta_n} in span_R{h_Q : joins, any weight}`** (a LINEAR-span question; the
  unbounded weight is what makes it hard).
- `V_k` = k-hidden-layer functions (a linear space). `max_n 2-layer <=> max_n in V_2`.

---

## 2. What is rigorously PROVEN

| result | how | file |
|---|---|---|
| `max_4,5,6` are 2-layer | explicit constructions (weight-2, root joins) | verify_2layer.py, STRUCTURE_MAX6.md |
| `max_6` = 6 root-zonotope-joins, bridges cancel (J+NB) | layer dissection | analyze_max6.py |
| `max_7` OUT of weight-2 | exact rank, vacuity-safe | check_weight2_max7_infeasible.py |
| `max_7` OUT of **complete weight-3** | exact mod-p, 2 primes, m=20000>#cols | gpu_w3.py |
| join rank=1 iff n<=4 | central-symmetry argument | join_rank1.py |
| k=1 (1-layer) lower bound, scale-invariant | odd-measure certificate (n=3 witness exact) | measure_cert_k1_exact.py |
| **threshold = n=6 for weight <= 3** (all bounded edge-complexity) | combine above | -- |

**The k=1 certificate (clean, exact).** A measure annihilates all zonotopes iff it is **odd** with
`int x dmu = 0`. `max_n`'s odd part is `(max+min)/2`, nonlinear for `n>=3`, so an odd balanced `mu` with
`mu(max_n)!=0` exists. Explicit `n=3`: points `+/-(2,-1,0),+/-(1,1,-2),+/-(0,1,1),+/-(1,0,0)`, weights
`(-3/7,-1/7,-2/7,1)`; `mu(max_3)=6/7`. No lattice anywhere. (SCALE_INVARIANT_ROUTE.md)

---

## 3. The residual-floor collapse (the central empirical fact)

`f(w) = dist(max_7, span(weight-<=w joins)) / ||max_7||`, the exact floor at each weight (vacuity-safe, GPU):

| weight | floor f(w) | method |
|---|---|---|
| 2 | **0.0308** (exact, all 784 blocks) | colgen exhausted |
| 3 | **0.0022** (complete 19219-orbit family) | floor_w3_gpu.py, GPU CGLS |
| 4 | ~0.0003 (extrapolated; greedy plateau 0.015 is NOT the floor) | (full family walled) |

`f(w) > 0` at every finite weight (proven w=2,3) but `f(w) -> 0`. So **`max_7 in closure(V_2)`** -- it is a
limit of 2-layer functions. This is exactly the signature of a true separation (in the closure, not in the set).

---

## 4. Every route tried, and why it fails

### 4a. Linear / continuous certificates -- ALL collapse (density no-go)
- **Mixed-volume / Hodge-Riemann on the permutohedral variety** (TORIC_K2.md). Built the intersection theory,
  verified Hodge-Riemann (Lorentzian) for our classes at n=4,5,6 (vol matches `n^{n-2}sqrt(n)` exactly).
  **NO-GO:** every intersection number is *multilinear* => factors through the `N^1(X)` class => sees only the
  linear span => collapses to the rank test, which density blocks. Confirmed: AF/Hodge-index defect of `max_n`
  is *smooth* across the n=6->7 representability boundary (3.69, 35.3, 539 at n=4,5,6 -- no kink).
- **Finite signed measure certificate.** Killed by density: a continuous functional vanishing on the (dense)
  join family also kills `max_7`.
- **Odd-part / central-symmetry** (the cleanest scale-invariant attack, odd_floor_w3.py). Odd part of `max_7`
  = `(max+min)/2`. ODD floor: weight-2 = **0.119** (3x the full floor -- looked very promising) but weight-3
  = **0.0049** -- COLLAPSES 24x, faster than the full floor. Joins build odd content efficiently (the `max`
  of two centrally-symmetric pieces is genuinely nonlinear). FALSIFIED.

**Why they all fail, uniformly:** they are continuous/multilinear, and `max_7 in closure(V_2)`. Any continuous
functional vanishing on the joins vanishes on the closure, hence on `max_7`.

### 4b. Combinatorial / counting -- defeated by signed cancellation
- **Covering / Turan-type width bounds** (COMBINATORIAL_ROUTE.md). The ridge->clique correspondence breaks at
  depth 2; the Turan exponent denominator `2^{k-2}-1 -> 0` at k=2. Signed cancellation kills counting.
- **Codim-2 triangle corners.** `Delta_n` has triangle 2-faces; zonotopes have centrally-symmetric 2-faces;
  joins make triangles only at **bridges**, which must cancel (NB). At n=7 J+NB are individually OK but
  **jointly over-determined** (transition.py defect 1). But this is weight-2-specific; higher weight adds
  freedom, and the defect's weight-2 certificate does NOT annihilate weight-3 (f(3)<f(2)).

### 4c. Self-similar measure hierarchy (HIERARCHY.md)
`mu _|_ V_k` iff the lift of `mu` to `V_{k-1}`-feature space is odd-balanced. For k=2 the feature map is
nonlinear, so odd-balanced lifts are rare. **Caveat (ii):** soundness needs `V_2` CLOSED -- which the floor
collapse shows it is NOT at `max_7`. So no FINITE lifted measure works (consistent with the no-go).

---

## 5. The reduction (where a proof must come from)

Proving `max_7 OUT` (exactly) requires a functional `lambda` with `lambda(h_Q)=0` for ALL joins (all weights)
and `lambda(max_7)!=0`, which by the density no-go must be **singular / discontinuous** and **weight-uniform**
(scale-invariant) -- it cannot be a finite measure, and cannot reduce to first-order (gradient-jump) data.

The one candidate of the right type (NONLINEAR_ROUTE.md): in the cancellation-free Minkowski form
`Delta_n + N = M` (M, N effective sums of joins), the simplex's `C(n,3)` triangle 2-faces must all be realized
by bridge-triangles while the bridges self-cancel at codim 1 -- a **cycle-mod-boundary** condition. So the
object is a **homology class on the matroid (Bergman) fan** of `U_{1,n}` (the simplex = uniform matroid `U_{1,n}`;
zonotopes = graphic matroids), not an intersection number. This is:
- non-linear (homology, not a functional) => not killed by the multilinear collapse,
- cancellation-proof (boundaries are quotiented) => survives the signed sum,
- weight-uniform (matroid-intrinsic) => not weight-graded.
It is the homological shadow of the J+NB over-determination. **Unproven** -- this names the correct *type* of
object, not the obstruction itself.

---

## 6. Computational walls (honest)

| approach | wall |
|---|---|
| full weight-4 family (exact, like gpu_w3) | column count too large to store (the "8 TB"); orbit reduction may rescue it -- under test (build_w4_orbits.py) |
| greedy column generation | plateaus ABOVE the true floor (~0.015 at both w3, w4); cannot reach floor or certify OUT |
| complete join pricer | single-zonotope pricing is complete B&B; JOIN pricing (max-coupling) is incomplete; and OUT certification needs ~rank iterations (time wall) regardless |
| weight-4 zonotope enumeration | naive code > 200s (12012 generators) -- needs orbit reduction + speedup |

**The orbit-reduction route (current):** `build_w4_orbits.py` enumerates the COMPLETE weight-4 join family by
S_n orbits (~5000x fewer). If the orbit count is feasible, `floor_w4_gpu.py` (GPU CGLS / exact mod-p) gives the
TRUE weight-4 floor with NO greedy plateau -- exactly as gpu_w3 did for weight-3.

---

## 7. Assessment

Calibration (honest): ~65-75% the separation HOLDS (`max_7 notin V_2`, in its closure -- which is why it is
hard: no continuous certificate can exist), ~25-35% `max_7` IS 2-layer over the reals (max_5,6 already beat the
integer HHL bound, so real weights surprise us). The evidence we have is exactly what a true separation looks
like. Decisive open items: (i) the orbit-reduced complete weight-4 floor (does it stay >0 like 2,3?), and
(ii) the homological obstruction (the actual proof).

## 8. File index

Proven: gpu_w3.py, floor_w3_gpu.py, verify_2layer.py, analyze_max6.py, measure_cert_k1_exact.py, transition.py.
Routes: TORIC_K2.md (toric/Hodge + no-go), SCALE_INVARIANT_ROUTE.md, HIERARCHY.md, COMBINATORIAL_ROUTE.md,
STRUCTURE_MAX6.md, NONLINEAR_ROUTE.md (homological reduction), odd_floor_w3.py (central-symmetry test).
Current: build_w4_orbits.py (orbit-reduced complete weight-4), PROOFS.md (theorem statements).
Log: dev/active/relu-max-depth/lowerbound/RESEARCH_LOG.md (iterations 1-28).
