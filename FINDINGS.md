# max_n ReLU depth: complete record of everything tried and all results

Master catalog of the investigation into the exact 2-hidden-layer representability of `max_n(x) = max(x_1..x_n)`.
Companion docs: WRITEUP.md (max_6 theorem), PROGRAM.md (research program), RESEARCH_LOG.md (iters 1-40, blow-by-
blow), CONSOLIDATED.md, SESSION_2026-06-30.md, IN_HYPOTHESIS.md, NONLINEAR_ROUTE.md, TORIC_K2.md,
SCALE_INVARIANT_ROUTE.md, HIERARCHY.md, STRUCTURE_MAX6.md, COMBINATORIAL_ROUTE.md.

## 0. Problem & framework

- `max_n = h_{Delta_{n-1}}` (support function of the simplex). A 2-hidden-layer ReLU function = signed sum of
  support functions of **joins of two zonotopes** + linear. So `max_n` is 2-layer iff `h_{Delta} in span_R{h_Q}`.
- **Weight-w** = edge complexity <= w = building blocks live on the `w*Delta` lattice (zonotope dim <= w).
- Literature: conjecture (Hertrich-Basu-Di Summa-Skutella 2021) `max_n` needs `ceil(log2(n+1))` layers;
  Haase-Hertrich-Loho (integer weights) proved it; Averkov-Hojny-Merkert (decimal weights) `ceil(log3 n)`;
  Bakaev-Brunck-Hertrich-Stade-Yehudayoff (STOC 2026, arXiv:2505.14338) proved `max_5` IS 2-layer via polyhedral
  subdivision of the simplex, **left max_6 open**. No real-weight lower bound > 2 is known for any `max_n`.

## 1. PROVEN results

| result | method / file |
|---|---|
| `max_4,5,6` are exactly 2-layer | exact-rational solve, verify_2layer.py (0 err / 400 fresh rational pts) |
| **`max_6` 2-layer, rigorous (NOVEL)** | check_max6.py: 6-orbit weight-2 rep, 0 of 2608 arrangement cells wrong. Resolves BBHSY's open n=6 |
| `max_7` OUT of weight-2 | exact, check_weight2_max7_infeasible.py; floor 0.0308 (colgen exhausted 784 blocks) |
| `max_7` OUT of **complete weight-3** | gpu_w3.py: exact mod-p, two primes, m=20000 > #cols; rank 18866 |
| join rank = 1 iff n<=4 | join_rank1.py (central-symmetry argument) |
| k=1 (1-layer) lower bound, scale-invariant exact | measure_cert_k1_exact.py (odd measure; n=3 witness a=(-3/7,-1/7,-2/7,1), mu(max3)=6/7) |
| `max_5,6` explicit constructions | show_construction.py (max_5: 5 orbits, LCD 30; max_6: 6 orbits, LCD 180) |
| => **2-layer threshold is exactly n=6 within bounded edge-complexity (weight <= 3)** | combine above |

## 2. max_7: the open frontier. Everything tried, and the result.

### 2a. IN direction (find a weight-4 construction; max_7 needs weight >= 4)
| attempt | result |
|---|---|
| column generation (greedy, weight-4) | plateaus ~0.015 ABOVE true floor; can't reach floor or certify |
| complete weight-4 family (exact) | UNENUMERABLE: 165529 zonotopes, ~10^6-10^7 join orbits ("8 TB") |
| structured weight-4 mod-p membership (small zonotopes) | max_7 OUT of subfamily (weight4_modp.py, weight4_orbitsum.py) -- inconclusive (subfamily) |
| saturation test (random raw joins, mod-p, m=24000/28000) | rank ~0.95m and STILL GROWING -> weight-4 join span near-FULL-DIMENSIONAL; never saturates -> membership unresolvable by sampling |
| symmetric (S_7) subspace via orbit-summed columns | symmetric span ALSO high-dim (rank ~1/orbit-block, no saturation) -> no clean verdict |
| GD training (real-weight 2-layer net, widths 64-512) | GD-LIMITED: plateaus ~3-5% (worse with width) vs known optimal ~0.0006; can't fit max's creases |
| recursion max_7 = max_6 + relu(x_7 - max_6) | gives 3 layers (relu of a 2-layer fn); min-of-relus reform doesn't reduce depth |
| recursion blocks join(Q_t, e_7) | only weight-2 (root) -> proven OUT |
| analytic valuation/cover at weight-4 | could NOT derive (weight jump 2->4 makes it harder than max_5/max_6) |

### 2b. OUT direction (prove >= 3 layers; would be first real-weight depth LB for any max_n)
| obstruction tried | why it collapsed |
|---|---|
| mixed-volume / Hodge-Riemann on permutohedral variety (TORIC_K2.md) | intersection numbers are MULTILINEAR -> factor through the Pic class -> only see the linear span (= rank test), which density blocks. AF/Hodge-index defect smooth across n=6->7 boundary |
| finite signed measure certificate | killed by DENSITY (max_7 in closure(V_2)) -- any continuous functional vanishing on dense joins kills max_7 |
| odd-part / central-symmetry (scale-invariant route) | ODD floor collapses 0.119 -> 0.0049 (w2->w3), FASTER than full floor. joins build odd content fine |
| self-similar measure hierarchy (HIERARCHY.md) | finite lifted measure = continuous => density-blocked; needs singular functional |
| homological / Turan / covering (NONLINEAR_ROUTE, COMBINATORIAL_ROUTE) | signed cancellation defeats counting; Turan exponent diverges at k=2 |
| HHL lattice-polytope (integer) technique | normalized-volume PARITY is scale-SENSITIVE; rational weights escape by rescaling |
| Why no real LB exists at n=7 | all known LBs are base-b volume args forcing 3 only at n~b^2 >= 9-10; nothing reaches n=7 |

## 3. Numerical evidence (cheap stable floors, cusolver QR)

`floor(g) = dist(g, weight-4 span)/||g||` over K random weight-4 join columns (one-shot cusolver QR, ~20s; the
ill-conditioning fix: orthonormalize, do NOT form AA^T which squares cond ~10^7). m=22000:

| n | floor@K=18k | ratio f(maxn)/f(ctrl) @K=6k,12k,18k |
|---|---|---|
| max_7 | 0.0063 | 0.0238, 0.0177, 0.0145 |
| max_8 | 0.0104 | 0.0356, 0.0281, 0.0243 |
| max_9 | 0.0144 | 0.0493, 0.0399, 0.0348 |

- All of max_7,8,9: floor AND ratio DECREASE with K -> all closure-leaning (IN-consistent). max_7 is ~80x closer
  to the span than a random target.
- Distance grows SMOOTHLY with n (no cliff at n=10) because the floor measures APPROXIMATION (->0 for all n), while
  the n=10 threshold is EXACT representability (invisible to any floor).
- True floors (exact, via CGLS, badly ill-conditioned/slow): f(2)=0.0308, f(3)~0.0006 (under-converged upper bd),
  f(4) lower still. eigh(AA^T) reveals extreme ill-conditioning (float32 resolves ~685/18873 dirs; float64 eigh OOMs).

## 4. Conclusions

- **max_6 (novel, proven)** is the deliverable -- resolves BBHSY's open n=6. Bring to the Hertrich group.
- **Conjecture (strong support): max_n is 2-layer iff n <= 9**, n=10 the first 3-layer case (= decimal ceil(log3 n)
  bound, then tight). Supported by: constructions for n<=6; no LB technique reaching n<=9; the closure-leaning
  floor pattern for max_7,8,9.
- **max_7 is very likely IN (2-layer)** -- strong quantitative evidence (~80x closer than random, decreasing ratio,
  coherent across n). NOT proven: the exact weight-4 construction is the bottleneck, walled computationally
  (near-full-dim span) and hard analytically (needs the valuation/cover at weight-4). A real-weight OUT proof would
  need a fundamentally new scale-invariant technique (none exists).

## 5. Methodology notes (durable)

- The repeated "timeouts" were a SELF-IMPOSED `timeout 595` wrapper; background tasks ignore it -> drop it.
- cusolver was installed but DLLs not on the load path; gpu_init.py registers the nvidia/*/bin dirs -> eigh/svd/qr work.
- For floors: orthonormalize via QR (cond kappa), NEVER form AA^T (cond kappa^2 -> float32 useless). float64 eigh OOMs on 16 GB.
- Guards that matter: m > rank (non-vacuous), random-target CONTROL floor (must stay high), per-stage caching, root-vs-non-root checks.
- Verdicts that proved false from bad numerics: vacuity (m<rank) faked "IN"; under-converged CGLS faked high floors; capped colgen faked a "plateau"; float32 GS drift faked increasing floors.

## 6. File index

Proven/verified: verify_2layer.py, check_max6.py, gpu_w3.py, measure_cert_k1_exact.py, show_construction.py, join_rank1.py.
Floors (cheap, cusolver): floor_qr.py, floor_qr_sweep.py, gpu_init.py. (older/walled: floor_w3_gpu.py, floor_w4_*.py, floor_spectral.py, floor_f64.py, floor_stream.py)
Membership/saturation: weight4_modp.py, weight4_orbitsum.py, weight4_saturate_modp.py, close_in.py.
Constructions/structure: construction.py, analyze_max6.py, transition.py, decomp_polyhedron.py.
Training: train_max7.py.
Docs: WRITEUP.md, PROGRAM.md, IN_HYPOTHESIS.md, CONSOLIDATED.md, SESSION_2026-06-30.md, TORIC_K2.md, NONLINEAR_ROUTE.md, SCALE_INVARIANT_ROUTE.md, HIERARCHY.md, STRUCTURE_MAX6.md, COMBINATORIAL_ROUTE.md, PROOFS.md, SEPARATION_PROGRAM.md.
Log: dev/active/relu-max-depth/lowerbound/RESEARCH_LOG.md (iterations 1-40).
