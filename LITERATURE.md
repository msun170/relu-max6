# Literature map for "is max_n in 2 hidden layers (V_2)?" -- and where our problem sits

Consolidated from a 4-front literature sweep (2026-07-01). Our problem is NOT isolated: it is an explicitly stated
open frontier in the Hertrich-Loho circle, and our central obstacle is a NAMED open problem (Averkov-Hojny-Merkert
Question 18). Below: exact statements, our place, corrected records, and the attack plan the literature implies.

## 0. Our problem, in the literature's language
- A 2-hidden-layer ReLU net = a signed sum of support functions of "joins of two zonotopes" + linear (our core.py
  model). In the maxout/tropical language: "polytopes of depth 2 are exactly the ZONOTOPES" (Balakin-Cox-Loho-
  Sturmfels, "Maxout Polytopes", arXiv:2509.21286); a join h_{conv(Z1 u Z2)} = max(h_Z1, h_Z2) is the second layer.
- Duality made precise: P-Q |-> h_P - h_Q is a BIJECTION between virtual polytopes and positively-homogeneous CPWL
  functions (Grillo-Hofmann, "On the expressivity of sparse maxout networks", arXiv:2510.14068, Thm 7). This is
  exactly our support-function dictionary.
- Home paper: Grillo-Hertrich-Loho, "Depth-Bounds for Neural Networks via the Braid Arrangement" (arXiv:2502.09324,
  ICLR 2025) -- CPWL functions compatible with the braid fan; depth bounds for max_n. [VERIFY their precise max_5
  claim vs BBHSY; likely a restricted (braid-fan-compatible) lower bound that BBHSY's finer subdivision escapes.]

## 1. What is PROVEN / OPEN about max_n depth (corrected)
- max_2 in V_1; max_3, max_4 in V_2 (binary tree); **max_5 in V_2** = THEOREM (Bakaev-Brunck-Hertrich-Stade-
  Yehudayoff "Better Neural Network Expressivity: Subdividing the Simplex", arXiv:2505.14338, STOC 2026, Prop 3).
- **max_6 in V_2**: explicitly OPEN in BBHSY -- this is OUR result (rational construction, denominators 90,180).
- **max_7 in V_2**: OPEN. BBHSY's general upper bound is MAX_n in ceil(log_3(n-2))+1 layers => 3 for n=6,7. They
  remark "it could be that two hidden layers suffice for all n." No published claim that max_7 needs 3 layers.
- CORRECTED AUTHOR ATTRIBUTION: arXiv:2505.14338 = Bakaev, Brunck, Hertrich, Stade, Yehudayoff. (Our old notes said
  "Bertschinger/Boroujeni/Servatius" -- WRONG.)

## 2. The integer/rational/real TOWER -- and why our lattice OUTs are not new
- INTEGER weights: ceil(log_2 n) hidden layers NECESSARY for max_n (Haase-Hertrich-Loho, "Lower Bounds on the Depth
  of Integral ReLU NNs via Lattice Polytopes", arXiv:2302.12553, ICLR 2023). For n=7: ceil(log_2 7)=3. So
  **max_7 NOT in integer-2-layer is ALREADY A THEOREM (HHL).** Proof = parity argument on the NORMALIZED VOLUME of
  faces of the (lattice) Newton polytopes -- structurally needs integrality.
  => Our weight-2/weight-3 lattice OUT results are essentially INSTANCES of HHL, not new.
- N-ARY-FRACTION (rational) weights: >= ceil(log_p(n+1)) layers, p a prime not dividing N (Averkov-Hojny-Merkert,
  "On the Expressiveness of Rational ReLU NNs With Bounded Depth", arXiv:2502.06283, ICLR 2025, Thm 2). Decimal =>
  >= ceil(log_3(n+1)). For max_7 this gives only >= 2, i.e. it does NOT forbid max_7 in 2 layers. Method = the
  invariant "d-dim lattice volume mod p" (d a power of p) as a group homomorphism that VANISHES on k-layer-
  representable functions but NOT on the simplex's support function (their Thms 14-15: Vol_{p^k} of faces of
  unimodular-summand polytopes == 0 mod p). THIS IS OUR "valuation/wall functional" IDEA MADE RIGOROUS.
- REAL weights: NO super-constant lower bound is known for any max_n. The only real-weight lower bound on record is
  the HBDS n=4 "H-conforming" computer-assisted case. The HBDS conjecture (max_n needs ceil(log_2(n+1)) layers,
  arXiv:2105.14835) -- the only thing predicting max_7 needs 3 -- is ALREADY FALSE at n=5,6.

CONVERGENCE (3 independent fields locate the difficulty at integer-vs-rational): submodular-cone faces (discrete
convex analysis), Payne-integral vs Brion-rational equivariant toric cohomology, and polyhedral rationality
(Schrijver). All say the obstruction concentrates at integrality.

## 3. THE NORMAL-FORM QUESTION IS A NAMED OPEN PROBLEM
Averkov-Hojny-Merkert **Question 18** (arXiv:2502.06283), verbatim: "Let S be a subfield of R and k in N and let f be
expressible via a ReLU network with weights in S. If f is expressible by a network with k hidden layers and weights
in R, is it also expressible with k hidden layers and weights in S? What is the answer for S=Q?" They note: if YES
for S=Q, the rational-weight conjecture EQUALS the real-weight conjecture. Their Thm 4: the lower bound degrades only
~ln ln N, i.e. rationality "almost" reduces to the integer case -> strong evidence the answer is YES, but UNPROVEN.
=> OUR "real -> rational/lattice normal form" gap IS exactly AHM Question 18. We are at the recognized frontier.
NUANCE (matches our earlier analysis): within a FIXED (rational) polyhedral complex, the decomposition polyhedron of
max_n is rational, so Schrijver (Theory of LP & IP, Ch.10) gives a rational solution of bounded height. The open part
is whether a MINIMAL 2-layer max_7 rep can be taken on a RATIONAL complex (the combinatorial type / breakpoint
arrangement is the free real continuum). Decomposition polyhedron = Brandenburg-Grillo-Hertrich, arXiv:2410.04907,
Thm 3.5 (DC decompositions form a polyhedron = intersection of two shifted convex cones), Thm 3.13 (minimal = vertex),
Cor 5.4 (same for submodular set functions, modulo modular).

## 4. WHY the join is essential (clean structural obstruction)
- ABD basis (Ardila-Benedetti-Doker, arXiv:0810.3947, Prop 2.4): every generalized permutohedron is a UNIQUE signed
  Minkowski sum of simplices Delta_I, coefficients y_I = sum_{J<=I}(-1)^{|I|-|J|} z_J (Mobius inversion). In this
  basis the simplex Delta_{n-1} is its OWN top generator (orthogonal to the zonotope subspace = the degree-2
  segments Delta_{ij}). So the simplex is INDEPENDENT of zonotopes in the linear Minkowski world.
- Central-symmetry obstruction (McMullen/Bolker): a polytope is a virtual ZONOTOPE iff all its 2-faces are centrally
  symmetric; this is invariant under Minkowski sum/difference. The simplex (n>=3) has TRIANGULAR 2-faces => **the
  simplex is NOT a difference of zonotopes.** This is exactly why the JOIN (max, nonlinear) is needed: the join
  leaves the zonotope group, which is what makes the simplex reachable at all. (So "max_n in 2 layers" is genuinely
  a nonlinear/join question; linear Minkowski/valuation theory cannot decide it, only obstruct via test functionals.)
- Generating-set lower bound (Bastidas, arXiv:2009.05876, Prop 6.8/Thm 6.1): any signed-Minkowski generating set for
  type-B generalized permutohedra in R^d needs >= 2^{d-1} full-dim polytopes. Obstruction-flavored complexity bound.

## 5. Construction toolkit (for the IN direction)
- BBHSY MAX_5 EXPLICIT FORMULA (a template for n=7), dyadic coefficient 1/2:
    M(x) = 0.5*( P1+P2+P3+P4 + Q - R13 - R14 - R23 - R24 )
  each P_i,Q,R_ij = max(zonotope, zonotope) = a join h_{conv(Z1 u Z2)}, e.g.
    P1 = max( max(2x5, x1+x2),  max(x1,x3) + max(x1,x4) ).
  This IS our normal form h_Delta = sum_t c_t h_{Q_t} + linear with rational c_t. Concrete evidence the rational
  normal form holds; a worked instance to generalize to n=7.
- Cayley trick / mixed subdivisions (Huber-Rambau-Santos, JEMS 2000; Sturmfels 1994): mixed subdivisions of a
  Minkowski sum <-> subdivisions of the Cayley embedding. This is the GEOMETRIC ENGINE behind BBHSY's "subdivide the
  simplex" construction. A better mixed subdivision of Delta^6 collapsing log_3 to 2 = a max_7 construction.
- Matroid star-mesh / Delta-Y moves (Hertrich-Kober-Loho, "Arithmetic Circuits and NNs for Regular Matroids",
  arXiv:2511.02406, IPCO 2026): signed difference (virtual extended formulation) compresses depth; local moves turn
  high-rank maxout into joins of rank-2 (zonotopal) pieces. A candidate constructive toolkit.

## 6. THE ATTACK PLAN the literature implies
LOWER BOUND (max_7 NOT in V_2 over reals) -- the cleanest target:
  Find an R-LINEAR translation-invariant VALUATION functional phi that VANISHES on the support function of every
  join-of-two-zonotopes (hence on all of V_2) but NOT on h_Delta. This is the REAL-valued analogue of AHM's
  "Vol mod p" homomorphism (which proves the integer/rational bound). AHM's Vol_{p^k} is mod-p; a genuine R-linear
  valuation separating h_Delta from the 2-layer joins would give the first REAL lower bound. Hadwiger/McMullen/
  Alesker classify continuous translation-invariant valuations (Val = sum Val_k, GL-irreducible) -- search there.
  (Our retracted "wall-circuit" attempts were a crude, non-valuation version of exactly this; the principled object
  is the mixed-volume / intrinsic-volume valuation, not ad-hoc second differences.)
CONSTRUCTION (max_7 in V_2) -- the cleanest target:
  Generalize the BBHSY MAX_5 mixed-subdivision formula to n=7 via the Cayley trick, OR search the decomposition
  polyhedron of max_7 (Brandenburg-Grillo-Hertrich) for a vertex whose two convex parts are each zonotope-support
  (single-layer) realizable. Both are rational by Schrijver once the complex is fixed.
NORMAL FORM (AHM Question 18, S=Q): if provable, rational==real and our exact rational/lattice tests become decisive.

## 7. VALUATION ROUTE worked out (2026-07-01): the DYADIC-NECESSITY refinement
Applying AHM's volume-mod-p valuation (arXiv:2502.06283 Thm 2) to max_7 directly:
  MAX_m = max(x_1..x_m) = F_{m-1}, so it needs >= ceil(log_p m) hidden layers for N-ary weights, p prime, p NmidN.
  MAX_7: ceil(log_2 7) = 3. So with p=2 (i.e. ODD-denominator or INTEGER weights), **MAX_7 needs >= 3 layers --
  ALREADY PROVEN by AHM's mod-2 valuation.** The only prime that bites is p=2 (p=3 gives ceil(log_3 7)=2, no force).
  To have MAX_7 in 2 layers one must AVOID p=2, i.e. use EVEN-DENOMINATOR (2|N) weights.

CLEAN CONSEQUENCE (n>=5, since ceil(log_2 n)>=3): **any 2-hidden-layer max_n is necessarily EVEN-DENOMINATOR /
DYADIC; it cannot be integer or odd-denominator.** Confirmed by data: BBHSY's MAX_5 is dyadic (powers of 2); OUR
MAX_6 construction has denominators 90, 180 (both even). MAX_4 (ceil(log_2 4)=2) is the last one realizable with
integer weights (binary tree).

WHY THIS MATTERS (both directions):
- LOWER BOUND: the valuation route is now SHARP -- AHM's mod-2 already kills odd/integer max_7; the OPEN piece is to
  kill EVEN-DENOMINATOR (dyadic) max_7. But mod-2 valuations STRUCTURALLY CANNOT (even denominators are exactly what
  escapes a mod-2 argument -- the lattice refines by 2). So a real lower bound needs a 2-ADIC / new invariant beyond
  AHM. This is precisely why no real lower bound exists, and it is genuinely hard (open).
- WHY OUR LATTICE OUTS CAN'T SETTLE IT: weight-w INTEGER-generator tests live in the integer/odd regime AHM already
  handles. Our weight-2,3 and wt-4 <=2v,<=3v OUT are in HHL/AHM territory. The construction must be dyadic, i.e. on
  a 2-refined lattice -- our weight-4 tests ARE that dyadic refinement of weight-2, and <=4v (running) is the live
  dyadic construction probe.
- CONSTRUCTION: search DYADIC / even-denominator coefficients specifically (BBHSY MAX_5 template is dyadic).

NET: "trying the valuation route" yields not a lower bound but the DYADIC-NECESSITY theorem (a clean, citable
refinement) + the precise open frontier ("kill dyadic max_7", a 2-adic problem) + a sharpened construction target
(dyadic). The lower bound is hard-open; the dyadic construction is the favored concrete direction.

## Key PDFs saved (scratchpad)
bakaev.pdf (2505.14338), averkov.pdf (2502.06283), maxout.pdf (2510.14068), abd.pdf (0810.3947),
polyalg.pdf (2009.05876), and the Zhang-Naitzat-Lim + 2410.04907 fetches.
