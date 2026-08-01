# M5 E0 final synthesis — meter-specific evidence classification

## Scope

Formal E0 evidence classification for the four meters, integrating the completed
bootstrap (4,000/4,000), the exact leave-one-building analysis (1,196/1,196),
and the segment concentration analysis (4/4). Pinned HEAD
`d8e59da2c40cb5102367d6a73299e807680f6ca6`, execution mode `FORMAL_E0`.

## Decision rule

The rule was fixed in code before any result was read, and applied mechanically.
Permitted vocabulary only:

- stable empirical advantage
- observed advantage but not stable
- no supported advantage
- counterexample
- supporting/contextual pattern

| Parameter | Value |
| --- | --- |
| primary endpoint | `100k_learner_gap` (TabPFN minus matched tree) |
| rank endpoint | `100k_positive_rank_gap`, reported separately, never pooled |
| direction-stability threshold | 0.95 of 1,000 draws |
| segment-concentration limit | top-10 share > 0.25 counts as concentrated |

`counterexample` requires both metrics negative, both intervals excluding zero,
stable direction, and no single-building sign flip. `stable empirical advantage`
requires both metrics positive, both intervals excluding zero, stable direction,
no sign flip, and no segment concentration. Failing any of those while both
point estimates remain positive yields `observed advantage but not stable`;
otherwise `no supported advantage`.

Manuscript roles were fixed in advance and are **not** derived from these
results.

## Classification

### steam — stable empirical advantage

Manuscript role: principal outcome.

| Evidence | Value |
| --- | --- |
| PR-AUC gap | +0.0790, 95% CI [+0.0132, +0.1467], 99.1% of draws positive |
| ROC-AUC gap | +0.0186, 95% CI [+0.0006, +0.0533], 98.9% of draws positive |
| invalid draws | 0 |
| LOO sign flips | 0 of 162 buildings |
| segment top-10 share | +3.3% |

Both metrics clear every criterion. **One caveat, stated once:** omitting a
single building moves ROC-AUC from +0.0186 to +0.0037 and the positive-rank gap
from +0.0179 to +0.0035 — about 80% of each magnitude — while never reversing
sign. The direction is robust; the ROC-AUC and rank *magnitudes* should not be
quoted as precise. PR-AUC is much less sensitive (range +0.0588 to +0.0969).

### chilledwater — observed advantage but not stable

Manuscript role: second empirical outcome; C1 required.

| Evidence | Value |
| --- | --- |
| PR-AUC gap | +0.0486, 95% CI [+0.0017, +0.0965], 98.1% of draws positive |
| ROC-AUC gap | +0.0043, 95% CI [**−0.000027**, +0.0090], 97.3% of draws positive |
| invalid draws | 0 |
| LOO sign flips | 0 of 252 buildings |
| segment top-10 share | +2.2% |

The single disqualifying fact is that the **ROC-AUC interval includes zero**, by
2.7 × 10⁻⁵ at the lower bound. Everything else is favourable: PR-AUC excludes
zero, direction is stable on both metrics, no building flips the sign, and the
effect is diffuse across 2,604 segments. This is a genuinely borderline result
and the PR-AUC/ROC-AUC disagreement is the substance of it, not a rounding
artefact to be argued away.

### electricity — counterexample

Manuscript role: counterexample.

| Evidence | Value |
| --- | --- |
| PR-AUC gap | −0.0065, 95% CI [−0.0108, −0.0035], **0 of 1,000 draws positive** |
| ROC-AUC gap | −0.0011, 95% CI [−0.0017, −0.0007], **0 of 1,000 draws positive** |
| invalid draws | 0 |
| LOO sign flips | 0 of 709 buildings |
| segment top-10 share | −0.7% |

The strongest-evidenced result in E0 in the sense of reliability: TabPFN is
reliably *worse* than the matched tree on electricity. Not one bootstrap draw
out of 1,000 is positive on either metric, and the LOO range spans 0.001 across
709 omissions.

### hotwater — no supported advantage

Manuscript role: supporting/context lever.

| Evidence | Value |
| --- | --- |
| PR-AUC gap | −0.0060, 95% CI [−0.0638, +0.0452], 42.0% of draws positive |
| ROC-AUC gap | −0.0005, 95% CI [−0.0133, +0.0118], 47.9% of draws positive |
| invalid draws | 0 |
| LOO sign flips | 9 of 73 buildings (2 PR-AUC, 7 ROC-AUC); 6 on the rank endpoint |
| segment top-10 share | +5.7% |

Both intervals span zero, direction is close to a coin flip, and single-building
omissions flip the sign on all three endpoints. Its manuscript role as a
supporting/contextual lever is unchanged and was not derived from this result.

## Summary

| Meter | E0 classification | PR-AUC excludes 0 | ROC-AUC excludes 0 | Direction stable | LOO robust | Diffuse |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| steam | stable empirical advantage | yes | yes | yes | yes | yes |
| chilledwater | observed advantage but not stable | yes | **no** | yes | yes | yes |
| electricity | counterexample | yes (neg.) | yes (neg.) | yes | yes | yes |
| hotwater | no supported advantage | no | no | no | **no** | yes |

Score movement and rank movement are kept distinct throughout; no statement
above combines them.

## Does this unlock chilledwater C1?

**E0 does not decide this, and this report does not start C1.** What E0
establishes is that chilledwater is exactly the case C1 exists to resolve: a
positive, diffuse, building-robust PR-AUC advantage whose ROC-AUC interval fails
to exclude zero by a hair. The evidence is neither strong enough to claim a
stable advantage nor weak enough to drop. That is a decision for the human
operator against the C1 entry criteria, not an inference from these numbers.

## Unresolved questions

1. **Chilledwater PR-AUC vs ROC-AUC disagreement.** Whether this reflects a
   prevalence-sensitive advantage concentrated in the high-precision region, or
   simply an underpowered ROC-AUC estimate, is not answerable from E0.
2. **Steam's single-building magnitude sensitivity.** One building carries ~80%
   of the ROC-AUC and rank magnitude. The direction is safe; the magnitude is
   not, and no E0 artifact explains why that building matters.
3. **Onset-phase pattern.** The learner gap is largest at episode onset for all
   four meters. E0 records this descriptively and offers no mechanism; testing
   it belongs to E1/C1.
4. **Hotwater instability source.** With only 73 buildings and 810 segments,
   it is unclear whether hotwater is genuinely null or merely underpowered.
5. **Environment deviation.** The segment phase ran on the main repo venv
   because `pyarrow` is absent from the pinned lock. Identical pandas/numpy/
   sklearn, but the pinned environment cannot currently reproduce that phase
   end-to-end without a lock change.

## Not done

No C1, Path A, Path B, 192-row frozen query, site transfer, 500k run,
full-holdout refit, fit/refit, inference, GPU work, or manuscript edit was
performed or authorized by this synthesis.
