# The exact depth of max_n: max_6 in two hidden layers, and the max_7 frontier

A self-contained account of (i) a new exact result -- max_6 is computable by a two-hidden-layer ReLU network,
resolving the n=6 case left open by Bakaev-Brunck-Hertrich-Stade-Yehudayoff (STOC 2026) -- (ii) a structural theorem
(forced negativity), and (iii) a sharply mapped frontier for max_7, including an honest catalogue of proof mechanisms
that we tested and ruled out.

## 0. Model and notation
- maxn(x) = max(x_1,...,x_n) = h_{Delta_{n-1}}, the support function of the standard simplex.
- A two-hidden-layer ReLU network computes exactly a finite SIGNED sum of support functions of "P2 blocks", where a
  P2 block is a join of two zonotopes Q = conv(Z1 u Z2), with h_Q = max(h_{Z1}, h_{Z2}) (the maxout / rank-2 picture:
  depth-2 polytopes are zonotopes; Balakin-Cox-Loho-Sturmfels, arXiv:2509.21286). So
        max_n in V_2  <=>  h_Delta = sum_t c_t h_{Q_t} + (linear),  Q_t joins of two zonotopes, c_t real.
- "Weight w": the blocks live on the lattice of w*Delta (integer points summing to w). Weight = lattice dilation.
- V_2 is the depth-2 class over REAL weights. Lemma 1 (below) lets us take the representation S_n-symmetric (each
  c_t h_{Q_t} replaced by an orbit-sum), which we use throughout for exact linear algebra.

## 1. Known landscape (corrected literature)
- max_2 in 1 hidden layer; max_3, max_4 in 2 (binary tree).
- max_5 in 2 (Bakaev, Brunck, Hertrich, Stade, Yehudayoff, "Subdividing the Simplex", arXiv:2505.14338, STOC 2026),
  DISPROVING the Hertrich-Basu-DiSumma-Skutella conjecture (arXiv:2105.14835) that depth grows as ceil(log2(n+1)).
  Their construction is DYADIC (powers-of-two denominators); their general upper bound is ceil(log3(n-2))+1 (=3 for
  n=6,7). They leave max_6 and max_7 OPEN, and remark 2 layers might suffice for all n.
- INTEGER weights: max_n needs ceil(log2 n) layers (Haase-Hertrich-Loho, arXiv:2302.12553), via a normalized-volume
  PARITY argument on lattice Newton polytopes. For n=7 this is 3: so max_7 NOT in integer-2-layer is a THEOREM.
- RATIONAL (N-ary) weights: only ceil(log_p(n+1)) (Averkov-Hojny-Merkert, arXiv:2502.06283). For max_7 this gives
  only >= 2, i.e. it does NOT forbid 2 layers with rational weights. Their Question 18 (real-weight rep => rational-
  weight rep at the same depth? open for Q) is exactly the normal-form gap; their Thm 4 shows it "almost holds".
- REAL weights: no super-constant lower bound is known for any max_n.

## 2. MAIN RESULT: max_6 in two hidden layers
THEOREM. max_6 is exactly computable by a two-hidden-layer ReLU network (over the rationals).
- Construction: an S_6-symmetric signed sum of 6 weight-2 P2 orbit-blocks with rational coefficients (denominators
  90 and 180), no linear term. Built and solved exactly over Q (verify_2layer.py, exact Fraction arithmetic).
- Proof: an exact, floating-point-free CHAMBER certificate (check_max6.py): reduce by S_6 to the chamber
  x1>=...>=x6, enumerate every cell of the resulting hyperplane arrangement in gap coordinates with an exact integer
  interior witness, show grad(construction) = e_1 = grad(max_6) on every cell, and close adjacency with exact
  rational Gordan certificates. Hence construction == max_6 on the chamber, and by continuity + symmetry everywhere.
- Positioning: this BEATS HHL's integer lower bound (which forbids integer-2-layer for n=6) using rational (dyadic-
  admissible) coefficients; it BEATS BBHSY's own general upper bound (ceil(log3(n-2))+1 = 3 for n=6); and it RESOLVES
  the n=6 case they explicitly left open. The coefficients are even-denominator, consistent with Section 4.

## 3. STRUCTURAL THEOREM: forced negativity (n >= 5)
THEOREM. For n >= 5, every two-hidden-layer support-function representation of max_n requires at least one negative
coefficient.
Proof. If h_Delta = sum_t c_t h_{Q_t} + ell with all c_t >= 0, then (ell = support function of a point p)
h_Delta = h_{ sum_t c_t Q_t + p }, so Delta = p + sum_t c_t Q_t (Minkowski). Each nonzero Q_t is a Minkowski summand
of a translate of the simplex. Simplices (dim >= 2) are Minkowski-INDECOMPOSABLE up to homothety/translation
(Shephard; McMullen), so every nonzero Q_t is homothetic to Delta. A homothet of Delta_{n-1} is a join of two
zonotopes only for n <= 4. Contradiction for n >= 5. QED.
Data: max_4 uses 1 block, no negatives; max_5, max_6 require negatives.
CONSEQUENCE. For n >= 5 every representation is a SIGNED/VIRTUAL Minkowski identity  Delta + N = M, with M, N positive
Minkowski sums of P2 blocks. The simplex appears ONLY AFTER cancellation -- neither M nor N individually looks like a
simplex. This is why single-polytope (central-symmetry, volume) obstructions fail: the obstruction is the cancellation.

## 4. THE max_7 FRONTIER (exact computational results)
All exact (two primes, S_7-orbit-sum membership, with random-symmetric controls validating non-vacuity):
- max_7 NOT in COMPLETE weight-2 V_2.
- max_7 NOT in COMPLETE weight-3 V_2.
- max_7 NOT in several COMPLETE LOW-COMPLEXITY weight-4 subfamilies: all <=2-vertex blocks (65 orbits), all
  <=3-vertex blocks (1180 orbits), and all joins of two <=2-generator root zonotopes <=6 vertices (6081 orbits).
- The COMPLETE 4-vertex weight-4 family (29732 orbits) is too large for exact membership at the required sampling
  (rank ~ m); left UNDECIDED. (We do NOT claim "OUT of weight-4".)
- DYADIC NECESSITY: applying AHM's mod-2 valuation, max_7 with odd-denominator/integer weights needs >= 3 layers
  (ceil(log2 7)=3). So ANY 2-layer max_7 must be EVEN-DENOMINATOR (the construction, if it exists, is dyadic; this
  is why all our integer/lattice OUT results -- which are HHL/AHM territory -- cannot settle the real question).
- TYPE-II weight ladder (Type-II = dim of non-braid-clean combinations with nonzero braid effect = the IN mechanism):
  n=7 weight-2: Type-II = 0 (mechanism ABSENT). n=7 weight-3: Type-II = 470 (mechanism PRESENT), and the braid target
  is wall-level reachable -- YET max_7 is exactly OUT of complete weight-3. So at weight-3 the WALL-LEVEL (codim-1)
  necessary conditions PASS while exact membership FAILS: the obstruction is HIGHER-ORDER, finer than the walls.
  (Control: n=6 weight-3 Type-II = 253, validating the machinery.)

## 4.5 SINGLE-CHAMBER INVISIBILITY LEMMA and the global-gluing picture (new; the clean structural statement)
LEMMA. Let D = {x_1 >= ... >= x_n} be the braid chamber; on D, max_n = x_1 is LINEAR. If a linear functional
lambda (a finite signed sum of point evaluations) has all its evaluation points in a single open braid chamber and
annihilates every linear function, then lambda(max_n) = 0.
COROLLARY (certificates are cross-chamber). Any separating certificate -- lambda with lambda(h_Q)=0 for all blocks,
lambda(x_i)=0 for all linear, lambda(max_n) != 0 -- must use data from at least two braid chambers.
So the membership obstruction is a GLOBAL cross-chamber compatibility failure: local linear pieces of max_n exist
chamber-by-chamber but need not glue to a symmetric P2 representation. SCOPE (do not overclaim): this rules out
SINGLE-chamber certificates only; adjacent-chamber wall tests and higher-order finite differences can still detect
finite-family obstructions.
CONSEQUENCE for the dual certificate. At complete weight-2 the exact (floating-point-free) OUT certificate exists
and is verified, but the minimal-support experiment shows its support is FORCED to be rank(M)+1 = 126 points with
~200-DIGIT integer coefficients, spread over 123 distinct braid chambers (~one per chamber). So the raw dual
certificate is a diffuse, cross-chamber basis artifact -- NOT a sparse/named local invariant. This is exactly what
the Lemma predicts, and it is why we do NOT scale the raw certificate to weight-3.

## 4.6 The minimal virtual decomposition (dyadic 4:2:1 template) and construction search (both negative for max_7)
- MINIMAL VIRTUAL DECOMPOSITION (decomposition-polyhedron min-mass vertex, exact + verified): max_5 and max_6 each
  have a 3+3 = 6-block minimal signed identity Delta + N = M with a DYADIC 4:2:1 coefficient pattern,
  M = (4 Q_a + 2 Q_b + Q_c)/D and N = (4 Q_a' + 2 Q_b' + Q_c')/D (D = 240, 360). Consistent with dyadic-necessity;
  suggestive but only two data points and vertex-choice dependent (NOT yet a theorem). max_4 = 2+1 blocks, D=48.
- CONSTRUCTION SEARCH (disciplined: approximate -> propose support -> EXACT test -> chamber verify). Two runs:
  (i) importance-ranked top-K supports over a weight-{3,4,5} pool (3600 blocks): max_7 OUT for all K up to 400,
  LS floor 0.00078. (ii) true OMP over 4000 small (4-8v) blocks with exact test at each K: max_7 OUT for all K up
  to 60; OMP residual descends smoothly (0.99 -> 0.0046 at 60 blocks) with NO collapse to 0 -- so max_7 is
  approximable but NOT sparse in the pool; route A's clean 6-block dyadic sparsity does NOT reappear at n=7.
  Both are pool-limited (random samples, not complete weight-4), so NEITHER is an OUT proof; together they push the
  evidence further toward max_7 OUT.

## 5. The bounded-weight normal form (the open reduction)
CONJECTURE (bounded-weight normal form for max_n). If max_n in V_2 then it is representable with all blocks on the
weight-<= W(n) lattice, W(n) explicit.
REDUCTION. If true, max_n in V_2 is DECIDABLE by exact S_n-orbit-sum membership over the complete weight-<=W(n)
family: IN gives a (dyadic) certificate; OUT gives the first real-weight depth lower bound. This is the max_n-specific
form of AHM Question 18. The PROOF TOOL is open (see Section 6 for what we ruled out and what remains).

## 6. CATALOGUE of proof mechanisms tested (honest: what is ruled out, what remains)
Ruled OUT (each by a rigorous test):
- Floors / closure: uninformative (dense-span artifact; floors -> 0 by universal approximation in the biasless cone).
- Residual / valuation certificates: collapse and rotate with weight; below float resolution (precision-walled).
- Wall-circuit (codim-1) OUT: on COMPLETE families the wall relaxation is FEASIBLE at weight-3 (Type-II>0) even
  though max_7 is exactly OUT -- so walls are TOO COARSE to obstruct. (Earlier "wall-circuit infeasible" was an
  incomplete-family artifact; retracted.)
- Local valuation-move descent: the cancellation in the max_5/max_6 identities is a single FULL-SUPPORT circuit
  (globally coupled, irreducible) -- it does NOT factor into bounded local moves.
- Rigidity / "representations are discrete": FALSE. Infinitesimal-rigidity test gives essential deformation dim 24
  (max_5), 35 (max_6) -- the representations sit on positive-dimensional continuous (semialgebraic) families.
- Braid-spline module regularity: grades by POLYNOMIAL DEGREE on a fixed fan, not by lattice weight (fan refinement)
  -- wrong parameter. Retracted.
- S_n-irrep decomposition of the obstruction: necessarily trivial (max_n symmetric).
- RAW DUAL CERTIFICATE as a named invariant: NO. At complete weight-2 the minimal certificate support is FORCED to
  rank(M)+1 = 126 points (greedy removal over several orders all bottom out there; no structured shortcut), with
  ~200-digit integer coefficients, over 123 braid chambers. Diffuse cross-chamber basis artifact (Section 4.5), not
  a local invariant; not worth scaling to weight-3.
- SINGLE-CHAMBER / local-linear certificates: impossible by the invisibility lemma (Section 4.5).
- SPARSE / dyadic-template construction for max_7: NOT found. OMP + exact test over weight-{3,4,5} pools (routes
  B and (a)) returns OUT at every support size; the max_5/6 6-block sparsity does not reappear. Pool-limited, not a
  proof, but the sparse-construction hope is not realized.
REMAINING (best-motivated), in priority order:
- The finite normal form via the DECOMPOSITION POLYHEDRON (Brandenburg-Grillo-Hertrich, arXiv:2410.04907): the
  correct CONTINUOUS/global object, and the frame the invisibility lemma says is NECESSARY (the obstruction is
  cross-chamber). Membership = nonemptiness of the P2-restricted decomposition polyhedron; OUT = a face/vertex
  certificate of emptiness. The OPEN question (our form of AHM Question 18): does emptiness PERSIST as the P2 family
  grows to all real blocks? This is the primary theory route; no proof tool yet exists.
- MIXED VOLUMES / ALEKSANDROV-FENCHEL on Delta + N = M: degree->=2 invariants that could see the higher-order
  obstruction. Secondary diagnostic. Honest caveat: homogeneity may let cancellation escape volume inequalities.

## 7. Summary of status
- PROVEN: max_4,5,6 in V_2 (max_6 novel, chamber-verified). Forced negativity for n>=5. Single-chamber invisibility
  lemma (Section 4.5). max_7 NOT in integer-2-layer (= HHL). max_7 NOT in complete weight-2, weight-3, and several
  complete low-complexity weight-4 families (exact).
- STRUCTURAL (this session): the OUT obstruction is a GLOBAL cross-chamber gluing failure (invisibility lemma); the
  raw dual certificate is a forced-diffuse basis artifact (min-support = rank+1, ~200-digit coeffs, 123 chambers),
  not a local invariant; the minimal virtual decomposition of max_5/6 is dyadic 4:2:1 (3+3 blocks) but this sparsity
  does NOT extend to max_7 in searched weight-{3,4,5} pools.
- OPEN: max_7 in V_2 over the reals (= the depth-2-vs-3 question for n=7). Any 2-layer max_7 must be even-denominator
  (dyadic). The construction, if it exists, lives at weight >= 4; a lower bound must use a HIGHER-ORDER, cross-chamber
  (non-single-chamber) obstruction, for which no tool currently exists. Primary route forward: the P2-restricted
  DECOMPOSITION POLYHEDRON and its emptiness-persistence question (our form of AHM Question 18).

## Appendix: reproducibility (key files)
verify_2layer.py (max_4,5,6 exact rational), check_max6.py (chamber proof), gpu_w3.py (max_7 OUT weight-3),
construction_search.py (complete weight-4 <=k-vertex), dyadic_search.py (BBHSY-complexity dyadic), wall_poscontrol.py
(validated Type-II on complete weight-2), weight3_typeII.py (Type-II weight-3), virtual_dissect.py /
valuation_factorization_max6.py (cancellation structure), infinitesimal_rigidity_max6.py (rigidity test),
extract_dual_certificate.py + min_cert_weight2.py (exact weight-2 OUT certificate + min-support = rank+1 diffuseness),
min_virtual_decomp.py (dyadic 4:2:1 minimal decomposition), disciplined_construction_max7.py + omp_dyadic_max7.py
(construction search, negative), extract_certificate.py (IN -> certificate). Theory: GLOBAL_GLUE_OBSTRUCTION.md
(single-chamber lemma), MIN_CERT_WEIGHT2.md, NEW_THEORY.md, LITERATURE.md, MAX7_CLASSIFICATION.md, OUT_THEORY.md,
WALL_VALUATION_MAX6.md, MAX7_DYADIC_CONSTRUCTION.md. Log: dev/active/.../RESEARCH_LOG.md (iters 1-56).
