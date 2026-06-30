# IN-direction experiments for max_7 (2026-07-01): does max_7 have a 2-layer construction?

Goal: instead of trying to prove max_7 OUT (lower bound, which is walled), look for a weight-4 construction and
its structure. Ran the controlled-sweep / structure-extraction / sparse-reconstruction plan. **All three came back
negative**, and the most decisive one is a clean, control-validated, two-prime exact result.

## Background (established earlier)
- max_4,5,6 ARE 2-layer, with CLEAN SMALL constructions: 1, 5, 6 S_n-orbit blocks respectively, all at WEIGHT-2
  (small zonotope joins). max_6 is novel + rigorously verified (resolves BBHSY/STOC-2026's open n=6).
- max_7 is provably OUT of weight-2 and complete weight-3 (exact mod-p, two primes).
- The full weight-4 join span is near-full-dimensional, so sampling/floor tests cannot certify exact membership.
- cusolver fixed (gpu_init.py) + the "orthonormalize via QR, never form AA^T" insight make floors cheap+stable.

## EXP 1 -- controlled QR floor sweep, n=7,8,9,10 x W=4,5,6 (qr_sweep_full.py)
Metric = ratio floor(max_n)/floor(symmetric control), deconfounded by the control (cancels the family-size/sampling
density effect). Symmetric controls = random CPWL functions of the sorted coordinates.

Ratio at W=4, K=18000:  max_7 = 0.035,  max_8 = 0.058,  max_9 = 0.094,  max_10 = 0.100.
- All of max_7,8,9,10 are modestly special (2-5x vs symmetric control) and the ratio drops with K for all.
- The growth is SMOOTH and FLATTENS at n=9->10 (0.094 ~ 0.100) -- NO cliff at n=10.
VERDICT: the floor test is NOT DISCRIMINATING. The weight-4/5/6 joins are dense enough to approximate everything
(including the symmetric controls) well, so the floor cannot separate the 2-layer threshold. (The earlier ~80x
"specialness" was vs a Gaussian control; vs a symmetric control it shrinks to 2-5x.)

## EXP 2 -- sparsity of the approximant via OMP (omp_raw.py)
Orthogonal Matching Pursuit over raw weight-4 join blocks, fitting max_7.
- Residual: 0.325 (1 block) -> 0.114 (20) -> 0.061 (100) -> 0.043 (250). First ~20 do most, then SLOW/DIFFUSE.
- Selected 250 blocks span 205 DISTINCT orbit types. Dominant types are low-complexity ((4 verts,2 gens),
  (3 verts,1 gen), (2 verts,0 gens)) but spread over many distinct orbits.
VERDICT: DIFFUSE -- no small sparse ansatz. (Raw OMP is confounded for a symmetric target -> exp 3 settles it.)

## EXP 3 -- EXACT membership in the complete low-complexity weight-4 family (orbit_membership2.py)
Random-saturated the low-complexity symmetric weight-4 family (small zonotope joins: <=3 generators, joins <=10
vertices), orbit-summed integer columns, exact mod-p elimination, two primes, random-target control, m=4000 > rank.
- 2500 distinct orbit blocks span only RANK 902 => the low-complexity SYMMETRIC weight-4 span SATURATES ~902-dim
  (so 2500 samples essentially cover the complete low-complexity family).
- max_7 is EXACTLY OUT of this 902-dim span. Random control OUT (valid, non-vacuous). BOTH primes agree.
VERDICT: max_7 has NO clean small low-complexity weight-4 construction.

## Conclusion -- the clean-construction pattern BREAKS at n=7

| n      | clean small construction? |
|--------|---------------------------|
| 4,5,6  | YES (1/5/6 orbit blocks, weight-2) |
| 7      | NO -- exactly OUT of weight-2, weight-3, AND the complete low-complexity weight-4 family |

The floor "closeness" of max_7 to the weight-4 span was MISLEADING: it reflects the DENSE full weight-4 span
approximating everything (confirmed by the diffuse OMP), not a special structure. Restricted to the clean
low-complexity blocks that actually yield constructions (as for max_4,5,6), max_7 is exactly outside.

This corrects the earlier "max_7 very likely IN" reading (which over-weighted the floor) and SHIFTS the evidence
BACK toward max_7 being the first genuine 3-layer case (threshold n=6/7), against the n<=9 conjecture.

Two live readings remain:
1. max_7 is 3-LAYER (first real-weight depth separation for max_n). Supported by the clean break at n=7; but a
   PROOF needs a scale-invariant lower-bound technique that does not currently exist (all known LBs bottom out ~n=10).
2. max_7 is 2-layer only via a HIGH-COMPLEXITY weight-4 (or weight>=5) construction unlike anything for n<=6.
   Not ruled out -- the full weight-4 span is near-full-dimensional and computationally walled -- but it would be
   a surprise, and no clean construction exists.

## What this plan accomplished
Turned "plausibly IN (from the floor)" into a sharp, evidence-backed statement: **n=7 is exactly where the clean
2-layer structure stops.** That is the most pointed thing provable about max_7 short of a depth lower bound.

## Files
qr_sweep_full.py (exp1), omp_raw.py (exp2), orbit_omp.py / orbit_membership2.py (exp3), gpu_init.py (cusolver fix),
floor_qr.py / floor_qr_sweep.py (cheap stable floors). Log: dev/active/relu-max-depth/lowerbound/RESEARCH_LOG.md
iters 39-41. Master record: FINDINGS.md.
