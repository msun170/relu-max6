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
On the complete weight-2 family the method is CORRECT and tracks the known facts exactly. The clean readout is the
Type-II dimension: it is 1 for the representable cases n=5,6 (the weight-2 construction is essentially UNIQUE) and
drops to 0 at n=7 -- a wall-theoretic restatement of the (already known) weight-2 separation. Correct, but not new.
(Confirmed with the fast GPU full-group-sum path AND, for n=5,6, the slow member-exact path -- identical feasibility.)

## Honest assessment (does wall-circuit mining help?)
- As a route to a NEW computational OUT for max_7: NO. Its columns are blocks, so a sound OUT needs a COMPLETE family,
  and complete families past weight-2 are exactly the unenumerable object we already hit. Wall-circuits inherit the
  completeness wall; they do not escape it.
- As a LANGUAGE for the theorem: YES, modestly. "Is Type-II eventually 0 for the complete/real P2 family?" is the
  descent / finite-normal-form question made precise. But that is a theorem to prove, not a computation to run.
- Lesson: the J/NB decomposition and Type-II invariant are meaningful ONLY relative to a complete family. Reported
  results on sampled/generated families are not valid OUT certificates.

## Status
wall_poscontrol.py (complete weight-2, member-exact) is the trustworthy script and the one that caught the bug.
wall_circuits.py and wall_extract.py are RETAINED ONLY as the method skeleton; their build_blocks family is INCOMPLETE
so their multi-weight outputs are NOT valid -- see the header warnings in those files. The OUT route's real content
is unchanged and lives in OUT_THEORY.md: the finite normal-form / bridge-cancellation theorem (pure theory).
