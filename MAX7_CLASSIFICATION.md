# Classifying max_7 in the depth / closure / complexity hierarchy

## KEY REFRAMING (2026-07-01): closure is NOT the object -> the question is exact membership

The whole content is binary: **is max_7 exactly in V_2, or in V_3 \ V_2?** Closure/floors are no longer useful.

NUANCE on "closure is trivial" (homogeneous model, corrected 2026-07-01):
- Our V_2 is the BIASLESS support-function model: every block h_Q(x)=max_{g in Q}<g,x> is positively
  homogeneous degree 1, so signed sums stay homogeneous. The ambient space is NOT C(compact); it is the
  cone of positively-homogeneous-degree-1 CPWL functions (~ C(S^{n-1}) restricted to PL).
- Leshno-Pinkus universal approximation is about nets WITH BIASES approximating ARBITRARY continuous
  functions. It does NOT transfer for free to the biasless/homogeneous cone. So "max_7 in closure(V_2)
  is automatic" was too glib.
- What IS true: homogeneous CPWL = differences of support functions (DC decomposition), so the biasless
  2-layer model with ARBITRARY polytope blocks contains max_7 in its span. But our blocks are the
  RESTRICTED joins-of-two-zonotopes; density of THAT sub-cone on the slice is an unproven theorem.
- Either way the operational conclusion is identical: **approximation floors are dead. Exact membership
  is the whole problem.** Closure analysis is retired.



The right question is not "can a bigger weight-4/5 span fit max_7" (dead end: the span is near-full-dimensional, so
sampling/elimination cannot resolve exact membership, and a small floor only means the family is dense). The right
question is structural: **which box does max_7 belong to?**

## What is SETTLED (control-validated, exact where stated)

```
max_7 in V_3                               -- max_7 = max(max_6, x_7); max_6 is 2-layer (proven); one more max/ReLU step
max_7 NOT in weight-2 V_2 (COMPLETE)       -- exact (two primes); floor 0.0308; family fully enumerated (small)
max_7 NOT in complete weight-3 V_2         -- exact (gpu_w3, two primes); 19219 orbits, family fully enumerated
max_7 NOT in SAMPLED low-complexity wt-4   -- exact (two primes) but for SAMPLED blocks only, NOT the complete family:
   and SAMPLED point/segment wt-4 V_2          saturation check shows rank climbs 3298->5386 (m=8000), NOT plateaued,
                                              so these wt-4 lattice families are high-dim and UNDERSAMPLED. OUT is
                                              sound for the sampled blocks (OUT-at-points => OUT-as-function), but
                                              "complete weight-4 family" is NOT decided. (orbit_membership2's old
                                              "rank 902 saturated" was undersaturated -- corrected.)
max_7 ?in full V_2 (real or lattice)       -- OPEN (weight-4 families high-dim/undersampled; real model needs normal form)
max_7 ?in closure(V_2)                     -- floors get tiny but that is dense-span artifact, NOT proof
```

The naive recursion max_7 = max(max_6, x_7) gives 3 layers (ReLU of a 2-layer object); a 2-layer construction would
need a special polyhedral/valuation cover, not the recursion. (RESEARCH_LOG, STRUCTURE_MAX6.md.)

## TWO boxes (the whole game; was three, closure box retired)

### Box A: prove max_7 in V_2 (exact 2-layer construction exists)  -- the IN route
Likely HIGH-complexity, weight >= 4, diffuse (no small clean ansatz: OMP is diffuse, max_7 exactly OUT of clean
low-complexity weight-4). The ONLY useful IN route now is TEMPLATE MINING, not floor minimization:
  best approximant -> cluster selected blocks into a few combinatorial FAMILIES -> one symbolic coefficient per
  family (S_7-equivariant) -> exact equality test -> rational reconstruct ONLY after family compression -> verifier.
DECISION RULE (omp_family_cluster.py): do <=10 families carry >=80% of the coefficient mass? yes -> pursue IN;
no -> the IN route has no structure to exploit, switch to OUT theory.
Status: OMP approximant is diffuse (250 raw blocks, 205 distinct orbit types) -> small-construction route looks BAD;
family-clustering is the test of whether a LARGE structured ansatz exists.

### Box B: prove max_7 NOT in V_2 (=> in V_3 \ V_2; first real-weight depth separation)  -- the OUT route
Finite exact obstructions (weight-2, weight-3, low-complexity weight-4 OUT) are not enough alone. Need a NORMAL-FORM
theorem: every exact 2-layer rep of max_7 reduces to a bounded-complexity family; then the finite obstructions
become a real lower bound. Baby lemma to start: every exact 2-layer rep of max_7 can be SYMMETRIZED into an
S_7-invariant signed measure on P2 blocks (averaging over S_7; max_7 is S_7-invariant). Then generalize the
weight-2 phenomenon (J feasible, NB feasible, J+NB infeasible) to: non-braid-wall cancellation + braid-wall
matching forces bounded bridge complexity. This is the cleanest path to "max_7 needs 3 layers".
Status: certificate-comparison (below) showed residual certificates do NOT stabilize and are precision-walled, so
the lower bound will NOT come from residual functionals -- it needs the structural normal-form theorem.

(Closure box retired: approximation/closure is not discriminating; see KEY REFRAMING. The remaining question is
exact membership, full stop.)

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

## RESULT of Test 1 (r_3 stability, test1_r3_stability.py, 2026-07-01)

Pairing |cos(r_3, h_Q)| over sampled blocks (with the weight-3 control = float32 noise floor):
  weight-3 CONTROL (should be 0): max 0.0018, mean 0.0012  <- this is the float32 NOISE FLOOR
  weight-4 root: 0.0020 / 0.0012;  weight-4 non-root: 0.0019 / 0.0011;  weight-5: 0.0018 / 0.0011
All weight-4/5 pairings sit AT the control noise floor => r_3 (||r3||/||b||=0.000184) is BELOW float32 resolution;
its direction is noise-dominated. Cannot test whether weight-4 kills r_3. The CERTIFICATE ROUTE IS PRECISION-WALLED
(float64 QR OOMs on 16 GB). Robust takeaway: max_7 is approximated by complete weight-3 to within ~1e-4 (exactly OUT
by mod-p), a >150x collapse from weight-2. => BOX B (closure) reinforced; BOX C blocked two ways (certificates do
not stabilize AND are below resolution). A depth lower bound for max_7 will NOT come from residual certificates --
it needs the structural normal-form theorem (Box C, hard theory).

Test 2 (exact r_3): the natural certificate is the min-norm residual; exact rational over rank-18866 is infeasible,
and a mod-p certificate has no norm/direction. Feasible only on small restricted families (large floor, float32
already suffices) -- not the interesting tiny obstruction. Test 3 (nested rate): weight hierarchy not cleanly nested
(different lattices) + complete weight-4 walled -> a DEFINED convergent sequence past weight-3 is out of reach.

## Strongest honest statement right now
max_7 is definitely in V_3; it is excluded from every CLEAN low-complexity 2-layer family tested (weight-2,
weight-3, low-complexity weight-4), exactly; whether it is in full V_2 is the open frontier. The clean-construction
pattern (max_4,5,6 -> small weight-2 constructions) BREAKS at n=7. To classify max_7, look for either an exact
construction PATTERN (Box A) or a STABLE limiting obstruction certificate (Box C) -- not raw closeness.
