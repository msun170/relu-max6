# Classifying max_7 in the depth / closure / complexity hierarchy

The right question is not "can a bigger weight-4/5 span fit max_7" (dead end: the span is near-full-dimensional, so
sampling/elimination cannot resolve exact membership, and a small floor only means the family is dense). The right
question is structural: **which box does max_7 belong to?**

## What is SETTLED (control-validated, exact where stated)

```
max_7 in V_3                               -- max_7 = max(max_6, x_7); max_6 is 2-layer (proven); one more max/ReLU step
max_7 NOT in weight-2 V_2                  -- exact (two primes); floor 0.0308
max_7 NOT in complete weight-3 V_2         -- exact (gpu_w3, two primes)
max_7 NOT in clean low-complexity wt-4 V_2 -- exact (orbit_membership2, two primes; the low-complexity symmetric span
                                              saturates at rank ~902 and max_7 is exactly outside)
max_7 ?in full V_2                         -- OPEN (full weight-4 near-full-dim, computationally walled)
max_7 ?in closure(V_2)                     -- floors get tiny but that is dense-span artifact, NOT proof
```

The naive recursion max_7 = max(max_6, x_7) gives 3 layers (ReLU of a 2-layer object); a 2-layer construction would
need a special polyhedral/valuation cover, not the recursion. (RESEARCH_LOG, STRUCTURE_MAX6.md.)

## The three boxes (the whole game)

### Box A: prove max_7 in V_2 (exact 2-layer construction exists)
Likely HIGH-complexity, weight >= 4, diffuse (no small clean ansatz: OMP is diffuse, max_7 exactly OUT of clean
low-complexity weight-4). Search path (NOT random floor minimization):
  best approximant -> dominant orbit types -> impose S_7 symmetry -> rational reconstruction -> exact solve on a
  chosen orbit family -> chamber verifier (like max_6).
Status: OMP says the approximant is diffuse (250 raw blocks, 205 distinct orbit types) -> small-construction route
looks BAD. A high-complexity construction is not ruled out but has no positive evidence.

### Box B: prove max_7 in closure(V_2) \ V_2 (approximable, not exactly representable)
Need an explicit convergent sequence F_w in V_2 with ||F_w - max_7|| -> 0 PLUS a singular (necessarily
discontinuous) annihilating functional. Finite floors are NOT enough (density/ill-conditioning/unconverged-solver
artifacts -- caught repeatedly in the log). Need an analytic recurrence or a density theorem.
Status: floors shrink with weight, but this is the dense-span artifact; no analytic sequence found.

### Box C: prove max_7 in V_3 \ V_2 (first real-weight depth separation)
Need a NORMAL-FORM theorem: every exact 2-layer rep of max_7 reduces to a bounded-complexity family. Then the exact
finite obstructions (weight-2, weight-3, low-complexity weight-4 OUT) become a real lower bound. This is the cleanest
path to "max_7 needs 3 layers", and the clean-construction break at n=7 is the strongest indirect support. But no
real-weight lower-bound technique reaches n=7 (all known LBs are base-b volume args bottoming out ~n=10).
Status: the most promising NEW idea is the OBSTRUCTION-CERTIFICATE comparison (below).

## The decisive experiment: obstruction-certificate comparison (certificate_compare.py)

The OUT certificate for a finite family F is the residual direction r_F = max_7 - proj(max_7 onto span(F)) -- it
annihilates all of F but not max_7. Compare r across weight-2, weight-3, low-complexity weight-4:
  - Do they CONVERGE (stabilize toward a limiting functional)?  -> that limit is the lower-bound INVARIANT (Box C).
  - Do they share SUPPORT / concentrate on the same chambers (same gradient defect)?
  - Do they survive when tested on random higher-weight blocks?
If the certificates converge to a stable object, that is the seed of a depth lower bound. If they change wildly /
fail on higher-weight blocks, the OUT route needs a different idea.

## RESULT of the certificate comparison (certificate_compare.py, 2026-07-01)

Computed r_2, r_3 (residual directions of max_7 against complete weight-2 and weight-3, cusolver QR, m=20000>rank):
  weight-2 floor ||r2||/||b|| = 0.0320;  weight-3 floor = 0.00018;  cosine(r2, r3) = 0.0055 (~orthogonal).
- The obstruction COLLAPSES ~175x (0.032 -> 0.0002) and ROTATES (r3 nearly orthogonal to r2). It does NOT stabilize
  toward a limiting functional. => NO stable lower-bound invariant (argues AGAINST Box C).
- No shared special support: r2, r3 both spread ~uniformly across tie-strata (energy fractions just track the point
  distribution, ~88% unique-max / ~11% two-way-tie). No concentrated "gradient defect".
- Caveat: r3 (~0.0002) is below float32 resolution, so its direction/fine-structure is partly roundoff (float64 QR
  OOMs on 16 GB). The COLLAPSE is robust; the fine direction is not.
VERDICT: the certificates COLLAPSE rather than converge -> points to BOX B (max_7 in closure(V_2), approximable but
likely not exactly representable). Does NOT yield a stable certificate for a clean lower bound (Box C). The
"interesting separation" (in closure, not in the set) is the most consistent classification.

## Strongest honest statement right now
max_7 is definitely in V_3; it is excluded from every CLEAN low-complexity 2-layer family tested (weight-2,
weight-3, low-complexity weight-4), exactly; whether it is in full V_2 is the open frontier. The clean-construction
pattern (max_4,5,6 -> small weight-2 constructions) BREAKS at n=7. To classify max_7, look for either an exact
construction PATTERN (Box A) or a STABLE limiting obstruction certificate (Box C) -- not raw closeness.
