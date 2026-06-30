# max_7 in V_2: construction search (the ONLY route to an IN proof)

Truthful status:  max_7 in V_3 = PROVEN (max_7 = max(max_6, x_7), max_6 is 2-layer).  max_7 in V_2 = NO PROOF.
The only thing that proves max_7 in V_2 is a CONSTRUCTION CERTIFICATE. Obstruction frameworks, small residuals, dense
floors, OMP approximations, and wall-circuit feasibility are SUPPORTING EVIDENCE, never a proof. This file is the
construction search and nothing else.

## What counts as a proof (4 components)
1. EXPLICIT BLOCK LIST. For each orbit representative t: coefficient c_t, zonotope Z_{t,1}, zonotope Z_{t,2}, and the
   join Q_t = conv(Z_{t,1} u Z_{t,2}). Actual generators/vertices -- no "high-complexity family".
2. EXACT ARITHMETIC. Rational (preferred):  D * max_7 = sum_t a_t * OrbitSum(Q_t) + b * (x_1+...+x_7),  integers D,a_t,b.
3. P2 VALIDITY CHECKER. For every block: Z1, Z2 are zonotopes; Q = conv(Z1 u Z2); h_Q = max(h_Z1, h_Z2). Proves it is
   genuinely TWO hidden layers, not secretly three.
4. GLOBAL EQUALITY CHECKER. Not sampling. A chamber/cell proof: enumerate the common normal fan, check grad = grad(max_7)
   on every chamber + one base value. (Template exists: check_max6.py does exactly this for max_6, S_6 symmetry, exact
   integer gradients + Gordan certificates for cell adjacency. Adapt to n=7.)

THE THEOREM (target):
   Theorem. max_7 is representable by a two-hidden-layer ReLU network.
   Proof. Exhibit rational c_t and P2 blocks Q_t with max_7 = sum_t c_t h_{Q_t} + linear; each Q_t is a join of two
   zonotopes (a valid 2-layer block); the exact verifier checks equality on every cone of the common normal fan. QED.

## Search pipeline
1. Large approximate solution from high-complexity P2 blocks (OMP).  [have it: residual ~0.034 at weight-4]
2. Cluster blocks into META-FAMILIES by signature (vertex counts, generator counts, root/non-root profile, stabilizer
   size, bridge-wall profile, partition pattern).
3. One symbolic coefficient per meta-family.
4. Solve the compressed exact rational system.
5. If exact solve succeeds -> expand to orbit sums.
6. Run the chamber verifier.
Key is step 2: if diffuse at orbit level, maybe structured at a higher level.

## What we have actually done (exact, complete where stated)
- COMPLETE weight-2 P2 family: max_7 OUT (verify_2layer, exact rational).  [max_5,6 IN here]
- COMPLETE weight-3 P2 family: max_7 OUT (gpu_w3, exact mod-p two primes).
- COMPLETE weight-4 <=k-vertex P2 families (construction_search.py, every <=k-subset of the 210 weight-4 points is a
  join of two zonotopes): exact membership, two primes, sym-control.  [results filled in below]
- SAMPLED weight-4 (point/segment, low-complexity): OUT, but sampled (not a complete-family claim).
- OMP approximant: DIFFUSE at orbit level (250 blocks -> 205 orbit types; at equivariant-orbit granularity 80 clusters,
  top-10 = 48% mass). So no small/sparse ansatz, and meta-family compression has weak support.

## RESULTS (construction_search.py) -- COMPLETE weight-4 vertex-capped families
   <= 2 vertices : COMPLETE = 65 orbits,   rank 26,  max_7 OUT (two primes, sym-control OUT, valid).
   <= 3 vertices : COMPLETE = 1180 orbits, rank 458, max_7 OUT (two primes, sym-control OUT, valid).  [NEW complete OUT]
   <= 4 vertices : COMPLETE = ~29024 orbits (= the point/segment-join family). NOT YET RUN. This is the FIRST complete
                   weight-4 family that could CONTAIN a construction (it strictly contains the OUT sampled families, so
                   it could flip IN). Cost estimate: enumeration of 7.6e7 4-subsets ~1 hr (pure-Python canon,
                   optimizable); exact mod-p membership rank ~1.1e4 so m ~1.5e4, matrix ~1.5e4 x 2.9e4 int64 ~3.5 GB
                   (fits 16 GB), elimination ~rank*m*N ~5e15 ops -> ~2-5 hr per prime. Total ~few-to-8 hr. Worth doing:
                   it is the meaningful complete test. Expected OUT (pattern), but genuinely could flip IN.

## The walls (why the construction search is blocked, precisely)
1. ENUMERATION. A construction, if it exists, is weight >= 4 (max_7 is exactly OUT of complete weight-2 and weight-3).
   The complete weight-4 P2 family is ~10^7 orbits / 1.65e5 zonotopes; even the <=4-vertex slice is ~2.9e4 orbits, at
   the edge of exact mod-p. Beyond that, exact complete enumeration is infeasible.
2. META-FAMILY COMPRESSION DOES NOT ESCAPE IT. A meta-family column = sum of orbit-sums over a signature class. Within
   an ENUMERABLE (capped) family, span(meta-families) is a SUBSPACE of span(per-orbit), so if max_7 is OUT per-orbit it
   is OUT after pooling too (pooling only adds constraints). Pooling would only help by reaching HIGHER complexity --
   but the pools there are over unenumerable classes, so the pooled column is not exactly computable. Catch-22.
3. SPARSITY IS OUT. The approximant is diffuse (OMP), so there is no few-block exact identity to find.
4. REAL / NON-LATTICE. Real-V_2 allows real (non-lattice) zonotopes. If max_7 in Real-V_2 only via irrational
   generators (no rational normal form), a LATTICE search never finds it; that needs a semialgebraic search, not
   linear algebra. (Whether 2-layer => rational/lattice is the open normal-form question; true for max_5,6.)

## Honest conclusion
A construction certificate for max_7 in V_2 is BLOCKED: the only place it can live (weight >= 4) is unenumerable; the
approximant is diffuse (no sparse identity); meta-family compression is a subspace of the already-OUT per-orbit span
where computable, and uncomputable where it would matter; and a real/non-lattice construction is outside lattice
search entirely. So we have neither a construction (no IN proof) nor a complete-family OUT past weight-3. The truthful
status stands: max_7 in V_3 proven; max_7 in V_2 open, with all enumerable evidence (exact OUT at weight <=3, diffuse
approximant) leaning toward OUT but PROVING nothing. The verifier (component 4) is ready (check_max6.py) the moment a
candidate construction appears.
