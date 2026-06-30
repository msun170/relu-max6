# Wall-circuit mining: method, a RETRACTED result, and what actually holds

## RETRACTION (2026-07-01)
An earlier version of this file reported a "feasibility table" claiming  J feasible / J+NB INFEASIBLE / Type-II = 0
for weight 2, 3, AND 4, and called it "stable structural evidence for a descent theorem." THAT RESULT IS WITHDRAWN.
It was produced by wall_circuits.py / wall_extract.py using build_blocks(), which generates an INCOMPLETE block
family. On an incomplete family the method gives FALSE infeasibility. Positive control (wall_poscontrol.py) proved
it: on the COMPLETE weight-2 family, max_5 and max_6 are correctly FEASIBLE (they are 2-layer; verify_2layer.py), but
build_blocks reported them INFEASIBLE. So the multi-weight "obstruction" was an artifact of family incompleteness,
not a real phenomenon. Do not cite it.

## The method (sound, but only on a COMPLETE family)
Decompose representability into gradient-jump (wall) constraints. For a probe (x0, d):
    jump_f(x0,d) = f(x0+d) + f(x0-d) - 2 f(x0).
max_n bends only across BRAID hyperplanes {x_i=x_j}. Split probes:  J rows (d ~ e_i-e_j) with target = max_n's braid
jumps;  NB rows (d non-braid, x0 a strict unique max) with target 0. An exact rep in family F exists IFF exists c
with  J c = jt  AND  NB c = 0.  Type-II dim = rank([NB;J]) - rank(NB) = number of independent non-braid-clean
combinations with nonzero braid effect; max_n needs Type-II > 0 (and jt in their image).

SOUNDNESS REQUIRES A COMPLETE FAMILY. If F omits the blocks a construction uses, "J+NB infeasible" only means
"OUT of this incomplete F", NOT OUT of the model. (A genuine construction's coefficient vector satisfies J c=jt and
NB c=0 exactly, so a COMPLETE family containing it is always feasible -- no false negatives only when F is complete.)

## What actually holds (validated, wall_poscontrol.py, EXACT complete weight-2 family = verify_2layer's)
    n=5  weight-2 (N=101):  J+NB FEASIBLE,    Type-II = 1,  controls valid   (max_5 is 2-layer)
    n=6  weight-2 (N=128):  J+NB FEASIBLE,    Type-II = 1,  controls valid   (max_6 is 2-layer)
    n=7  weight-2 (N=136):  J+NB INFEASIBLE,  Type-II = 0,  controls valid   (max_7 has no weight-2 rep)
On the complete weight-2 family the method is CORRECT and tracks the known facts exactly.

THE KEY DIAGNOSTIC is not "infeasible" but WHICH KIND of infeasibility: Type-II = 0 (no non-braid-clean braid
effects at all) vs Type-II > 0 but target outside its image (the circuits exist but the simplex target moved). The
data says the n=7 transition is the SHARP kind:

    n    Type-II dim    target in image?    representable?
    5    1              yes                 yes
    6    1              yes                 yes
    7    0              no (image = {0})    no

So at weight-2 the transition at n=7 is sharp: the non-braid-clean braid-producing circuits VANISH (Type-II 1 -> 0),
they do not merely miss the target. (Confirmed with the fast GPU full-group-sum path AND, for n=5,6, the slow
member-exact path -- identical feasibility.)

## Type-II vector = the construction (wall_typeII.py) -- the nice structural statement
For n=5,6 the Type-II space is 1-DIMENSIONAL, so there is (up to wall-invisible circuits) a UNIQUE non-braid-clean
combination with nonzero braid effect. We recovered the exact construction (core.exact_solve) and confirmed it IS
that unique Type-II generator:
    n=5: 5 signed orbit blocks, coeffs {-1/10, 1/10, 1/15, -1/30, -1/30}, NO linear term.
    n=6: 6 signed orbit blocks, coeffs {1/90, -1/90, 1/90, 1/90, -1/180, -1/180}, NO linear term.
WALL-LEVEL EXPLANATION OF WHY max_6 IS 2-LAYER: among all signed combinations of weight-2 P2 blocks, exactly ONE
(modulo wall-invisible circuits) cancels every non-braid wall while producing the simplex's braid walls -- and that
one is the max_6 construction. At n=7 no such circuit exists (Type-II = 0), so max_7 has no weight-2 representation.

## Hard rule (operational)
    Incomplete family + infeasible  = NO conclusion (false negatives; see retraction).
    Incomplete family + feasible    = useful POSITIVE evidence (a construction exists).
    Complete family   + infeasible  = valid finite-family OUT.
Incomplete wall tests are fine for FINDING constructions, never for lower bounds.

## Honest assessment (does wall-circuit mining help?)
Wall-circuit mining does NOT bypass the completeness wall. It is not a new computational route to max_7 not in V_2.
Its value is CONCEPTUAL: it reformulates the lower bound as a statement about the IMAGE OF THE BRAID-WALL JUMPS
AFTER QUOTIENTING BY NON-BRAID WALL CANCELLATION, i.e. about J(ker NB). On complete finite families this gives a
sound diagnostic (and the clean Type-II invariant); on incomplete families, infeasibility is not meaningful.

## The theorem targets this language gives
    FULL REAL WALL-CANCELLATION THEOREM (the lower bound):
        for the full REAL P2 block family, the braid-jump vector of max_7 is NOT in J(ker NB).

    FINITE NORMAL-FORM VERSION (the bridge from finite tests to the real bound):
        every element of J(ker NB) relevant to max_n is generated by finitely many bounded-complexity valuation
        circuits.
Together these are exactly the bridge from our finite exact tests to the real lower bound: prove the normal form,
enumerate the (now finite) generators of J(ker NB), and check max_7's braid target against them.

## Status
wall_poscontrol.py (complete weight-2, member-exact) is the trustworthy script and the one that caught the bug.
wall_circuits.py and wall_extract.py are RETAINED ONLY as the method skeleton; their build_blocks family is INCOMPLETE
so their multi-weight outputs are NOT valid -- see the header warnings in those files. The OUT route's real content
is unchanged and lives in OUT_THEORY.md: the finite normal-form / bridge-cancellation theorem (pure theory).
