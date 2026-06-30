# OUT route (Box B): toward "max_7 is NOT in V_2" -- structural theory

The IN route (template mining) is closed (see MAX7_CLASSIFICATION.md / RESEARCH_LOG iter 45): the best 2-layer
approximant concentrates 83% of its mass on a family we prove exactly insufficient, and at true equivariant-orbit
granularity the coefficients are diffuse (no compression). So the live question is the lower bound: max_7 not in V_2.

This file is the OUT track. It records what is PROVED vs what is the single remaining gap.

## Notation  (CRITICAL distinction: real model vs lattice model)
- Real-V_2 = the ACTUAL 2-layer model for the real-weight lower bound: finite signed sums F = sum_i c_i h_{Q_i} +
  (linear), where each block Q_i is a join of two REAL zonotopes (real generators, real coefficients c_i).
- Lattice-V_2^{<=K} = the SUBmodel where blocks are joins of two LATTICE zonotopes of bounded weight/complexity <=K.
  Our exact mod-p tests live entirely here. Lattice-V_2 is a submodel, NOT automatically the full real model.
- h_Q(x) = max_{v in Q} <v,x>; S_n acts on R^n by permuting coordinates, preserving join/zonotope structure and all
  combinatorial complexity. max_n(x) = max_k x_k = h_{Delta}(x) is S_n-invariant.

What our exact tests actually prove (LATTICE statements only):
    max_7 NOT in Lattice-V_2^{weight 2},  NOT in Lattice-V_2^{complete weight 3},
    NOT in Lattice-V_2^{low-complexity weight 4},  NOT in Lattice-V_2^{point/segment-join weight 4}.
They say NOTHING about arbitrary REAL P2 blocks until a normal-form theorem bridges real -> lattice. (Correction
2026-07-01: do not conflate the two models.)

## LEMMA 1 (Symmetrization).  PROVED.
If max_n in V_2, then max_n admits an S_n-INVARIANT exact 2-layer representation:
    max_n = sum over S_n-orbits O of  a_O * (OrbitSum_O)  +  d * (x_1 + ... + x_n),
where OrbitSum_O(x) = sum_{Q in O} h_Q(x), each a_O in R, d in R, and every block in every orbit O has the SAME
combinatorial complexity as a block of the original representation.

Proof. Suppose max_n = sum_i c_i h_{Q_i} + ell, ell linear. For sigma in S_n, coordinate permutation is
orthogonal, so for any point set Q:
    h_Q(sigma x) = max_{v in Q} <v, sigma x> = max_{v in Q} <sigma^{-1} v, x> = h_{sigma^{-1} Q}(x).
Also max_n(sigma x) = max_n(x) (invariance) and ell(sigma x) = ell_sigma(x) is again linear. Apply sigma to the
identity and then average over the whole group G = S_n (|G| = n!):
    max_n(x) = (1/|G|) sum_{sigma in G} [ sum_i c_i h_{sigma^{-1} Q_i}(x) + ell(sigma x) ]
             = sum_i c_i * [ (1/|G|) sum_{sigma} h_{sigma^{-1} Q_i}(x) ] + (1/|G|) sum_{sigma} ell_sigma(x).
For each i, (1/|G|) sum_{sigma} h_{sigma^{-1} Q_i} = (|Stab(Q_i)|/|G|) * OrbitSum_{O(Q_i)}, an S_n-invariant
function supported on the orbit O(Q_i); collecting equal orbits gives the coefficients a_O. The averaged linear part
(1/|G|) sum_sigma ell_sigma is an S_n-invariant linear functional, and the only such functionals are multiples of
x_1+...+x_n, giving d. Every block sigma^{-1} Q_i is a coordinate-permuted copy of Q_i, hence a join of two lattice
zonotopes of identical vertex/generator counts: complexity is preserved. QED.

CONSEQUENCES (why this matters) -- and the LIMIT of Lemma 1 (correction 2026-07-01):
(a) WLOG every exact rep IN A GIVEN S_n-STABLE BLOCK CLASS is orbit-summed. Within a FIXED finite lattice family,
    this justifies our orbit-sum membership tests: an orbit-sum OUT result is a genuine "no rep in THIS family".
(b) The invariant linear term is forced: d*(x_1+...+x_n), one free scalar, not n.
(c) WHAT LEMMA 1 DOES NOT DO: it only sends  (real arbitrary rep) -> (S_n-invariant real arbitrary orbit-MEASURE
    rep). That symmetrized object is still CONTINUOUS / infinite-dimensional (a signed measure over the real P2
    blocks, real generators). Lemma 1 does NOT discretize to a lattice family. So it does not, by itself, connect
    Real-V_2 to our finite exact lattice tests.

## The REDUCTION (corrected).
For ANY finite S_n-stable family F of lattice P2 blocks, Lemma 1 reduces "max_n in span(F)" to an exact orbit-sum
linear-algebra problem, decided EXACTLY by mod-p membership. We have decided several and gotten OUT (two primes,
control-valid): weight-2 (COMPLETE), complete weight-3 (COMPLETE, 19219 orbits), and SAMPLED weight-4 (low-complexity
and point/segment). CAVEAT: the weight-4 lattice families are high-dimensional and UNDERSAMPLED -- saturation check
shows rank climbs 3298->5386 (m=8000) without plateau, so weight-4 OUT is rigorous for the SAMPLED blocks
(OUT-at-points => OUT-as-function) but is NOT a complete-family result. Only weight-2 and weight-3 are genuinely
complete-family exact OUT.

To upgrade these LATTICE obstructions to the real statement  max_7 NOT in Real-V_2  it remains to prove a FINITE
NORMAL-FORM theorem -- and this is STRICTLY STRONGER than a complexity bound:

    >>> FINITE NORMAL-FORM CONJECTURE:  if max_n in Real-V_2, then max_n admits a representation using blocks from
        an explicitly FINITE, S_n-stable, ENUMERABLE family F_n (e.g. bounded-weight LATTICE P2 blocks). <<<

    (Strong form -- LATTICE NORMAL FORM: if max_n in Real-V_2 then it has a bounded-weight lattice P2 rep.)

WHY a mere complexity bound is NOT enough: "joins of two zonotopes with <= K generators" is still an INFINITE real
family (continuously many real generator directions and lengths). A bound on generator/vertex COUNT does not
discretize the real parameters, so it does not reduce membership to a finite/exact decision. The normal-form theorem
must additionally RATIONALIZE/DISCRETIZE (collapse the real parameters onto a lattice) or otherwise finitize the
decision. That discretization is the actual hard content.

    THEOREM (conditional, corrected).  IF the finite normal-form conjecture holds with family F_n, AND our exact
    orbit-sum enumeration over F_n returns OUT, THEN max_n is NOT in Real-V_2 (needs 3 layers): the first
    real-weight depth separation for max_n.

So the OUT route is cleanly reduced to ONE theorem: real exact 2-layer reps of max_n, if they exist, convert to
bounded lattice/enumerable P2 reps. Everything else we built (symmetrization, exact finite decision) is the
machinery that becomes decisive the moment that theorem exists. This is the clean target for Hertrich's group --
their polyhedral-subdivision / valuation machinery (BBHSY identity (P u Q)+(P n Q)=P+Q) is exactly the tool for
producing such a discretizing normal form.

## Toward the finite normal form (three sharper targets, replacing the vague "complexity bound")

1. REAL-TO-LATTICE NORMAL FORM (strongest useful lemma). Prove: if an S_n-invariant exact rep of max_n exists with
   REAL P2 blocks, then one exists with RATIONAL/LATTICE P2 blocks of bounded weight. This is the bridge that makes
   our finite exact tests decisive. Likely mechanism: the wall-cancellation constraints are LINEAR with integer
   coefficients in the block parameters, so a real solution implies a rational one (then clear denominators ->
   lattice). Pin down the linear system whose solvability = representability and show its defining data is integral.

2. MINIMAL-SUPPORT RIGIDITY / DESCENT (best theoretical route). Assume a counterexample max_7 = sum_O a_O OrbitSum_O
   + d*sum_i x_i; pick one of MINIMAL total complexity (or minimal support). Prove: any non-braid wall introduced by
   a highest-complexity block forces a cancellation identity (a CIRCUIT) that lets you remove or lower that block --
   contradicting minimality. This turns "non-braid cancellation" into a DESCENT argument and would yield the normal
   form directly.

3. KERNEL/CIRCUIT ANALYSIS (the computational handle on (2)). For each finite family with exact matrix A, the
   relations A c = 0 are CIRCUITS (syzygies among orbit-sums). Question: do high-complexity blocks appear ONLY
   through cancellation circuits that re-express them via lower-complexity blocks? If every high block participates
   in such a "lowering circuit", that IS the descent mechanism.

## NEXT EXPERIMENT: circuit mining (not more membership tests).
For the combined finite family (weight-2 + weight-3 + weight-4 point/segment orbit-sums, exact integer matrix A):
  - compute the kernel (sparse dependencies A c = 0);
  - classify each weight-4 / weight-3 block as REDUCIBLE (lies in the lower-weight span -> rank does not increase) or
    IRREDUCIBLE (extends the span);
  - for each circuit, record: highest weight involved; whether it expresses ONE high block as a combo of strictly
    lower blocks ("lowering circuit"); whether the non-braid walls cancel internally;
  - check whether the SAME circuit patterns recur across weight-2 -> weight-3 -> weight-4.
If every high-complexity block participates in a lowering circuit -> evidence FOR a descent/normal-form theorem.
If many high-complexity blocks are irreducible (genuinely extend the span) -> the descent is NOT unconditional and
the normal form must come from the max_n-specific wall constraints, not from generic span collapse.

IMPORTANT (why span-based circuit mining is the WRONG handle here): our exact mod-p tests are NOT vacuous -- the
near-full-dimensionality only ruined the FLOAT FLOORS; exact membership is sound and returns OUT at every weight
(OUT-at-exact-points => OUT-as-function). And the lattice spans grow with weight (121 -> ~19219 -> 3299+ for wt-4
ps), i.e. high-complexity blocks are largely IRREDUCIBLE. So a descent based on "high block lies in low span" is
both vacuous (low lattice spans are already near-full) and false (spans genuinely grow). The descent must be
WALL-based, not span-based.

## WALL TAXONOMY (wall_taxonomy.py) -- the mechanism, made concrete.
A support function h_Q is linear on the normal cones of the vertices of Q and bends across {<v_a - v_b, x> = 0} for
each EDGE (v_a,v_b) of conv(Q); the wall normal is v_a - v_b. max_n = h_Delta bends ONLY across braid walls
(normals ~ e_a - e_b). So an exact rep sum_O a_O OrbitSum_O = max_n must CANCEL every NON-braid wall.

WHERE do non-braid walls come from? Measured on weight-4 segment-joins (4 vertices in convex position, so all
pairwise diffs are genuine edges):
    ROOT-generator joins (BOTH generators braid):  99.4% have >= 1 non-braid wall;  90.1% of BRIDGE edges
        (edges linking the two segments) are non-braid.
    NON-ROOT joins:                                100% have non-braid walls (expected).
So the JOIN OPERATION ITSELF -- the essential 2-layer ingredient -- injects non-braid walls via its BRIDGE edges,
EVEN when both generators are braid roots. Non-braid walls are not a non-root artifact; they are intrinsic to joins.

CONSEQUENCE for the descent / normal-form target. The cancellation constraint is UNIVERSAL: in any exact
S_n-invariant rep of max_n, the bridge walls contributed by every join must pairwise cancel across the
representation. This is the precise generalization of the weight-2 "J feasible, NB feasible, J+NB infeasible"
phenomenon. The concrete descent target is therefore:

    >>> BRIDGE-CANCELLATION CONJECTURE:  in an exact S_n-invariant 2-layer rep of max_n, the requirement that all
        non-braid bridge walls cancel forces the joins to organize into bounded-complexity effective blocks (and,
        with rationality of the linear cancellation equations, onto a bounded-weight lattice) -- i.e. it IS the
        finite normal form. <<<

This connects the OUT route to BBHSY's valuation machinery directly: the identity (P u Q) + (P n Q) = P + Q is the
canonical bridge-cancellation move (it trades a join P u Q for lower-complexity pieces P, Q, P n Q). The conjecture
is that iterating such moves reduces any exact rep to bounded complexity. That is the theorem to prove with the
Hertrich group.
