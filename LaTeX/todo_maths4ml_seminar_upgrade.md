# Maths for ML — Seminar Restructuring TODO

## Problem
`Main_Workshop_ML_Maths_{Presentation,Cheatsheet}.tex` -> `workshop_maths4ml_content.tex`
currently `\input`s raw `maths_*.tex` topic files directly, skipping the seminar layer
(this is one of the two known exceptions called out in CLAUDE.md, the other being
`workshop_python_content.tex`).

## Target hierarchy
```
Main_Workshop_ML_Maths_{Presentation,Cheatsheet}.tex
  -> workshop_maths4ml_content.tex
       \input{seminar_maths4ml_<parenttopic>_<subtopic>_content}  (12 seminars, no raw maths_* inputs)
            -> seminar_maths4ml_<parenttopic>_<subtopic>_content.tex
                 \input{maths_<topic>_...}          (existing raw topic files, unchanged)
```

Naming convention (per user direction 2026-07-05): both the content file and the
driver pair carry the parent topic name (Basics / Calculus / LinearAlgebra / Statistics)
followed by the subtopic, so the family relationship is visible in the filename itself
— not just the parent-topic-only name used in the first pass.

Each seminar gets its own standalone driver pair so it can be compiled and
distributed on its own for the webinar series:
`Main_Seminar_MathsML_<ParentTopic>_<Subtopic>_{Presentation,CheatSheet}.tex`
(underscore between parent topic and subtopic, per user direction 2026-07-05)

Audience for the webinar series: students fresh out of school / higher-secondary,
or freshers in college -> when each seminar is later run through `/upgrade-deck`,
bias improvements toward intuition, concrete examples, and gentler pacing over
formal rigor.

Target length per seminar: ~60-80 slides, ~4-5 CheatSheet pages.

**Verified (2026-07-05): none of the 12 content files `\input` any of the commented-out
reference-only raw files** (maths_refs, maths_awsmathforml_intro, maths_linearalgebra_grimmer,
maths_linearalgebra_uky, maths_calculus_oxford, maths_discretemathematics_*, maths_gametheory,
maths_optimization_uky, maths_misc_uky, maths_statistics_probability_duke,
maths_statistics_probability_kale, maths_statistics_python_implementations). The only match
found was `maths_calculus_integration`, which stays commented out (`% TBD`) exactly as in
the original source.

**Compilation status (2026-07-05)**: all 12 `*_Presentation.tex` drivers compiled clean
under their final (underscore-separated) names — see per-seminar page counts below.
The 12 `*_CheatSheet.tex` drivers have not yet been (re)compiled; deferred until requested.

---

## History of how we got to 12 seminars
1st pass created 6 seminars (Basics, LinearAlgebra, Calculus, Probability+Descriptive
combined, Inferential) — see git history / prior conversation for that pass's file names.
All 6 compiled well over the ~50-80 slide target (109-287 pages), so the combined
Probability+Descriptive bucket was split in two (Probability / Descriptive Statistics),
and then, per user direction, **all six were split again** into 12 total, using active
(non-commented) `\begin{frame}` counts per raw file as a sizing proxy (this proxy matched
actual compiled PDF page counts almost exactly in the 1st pass). Two seminars (Calculus,
and to a lesser extent parts of Probability/Inferential) could not be split into two
halves that are both ≥60 slides because their raw-file totals are under 120 frames —
user chose to split anyway, accepting some sub-seminars running under 60.

---

## Seminar 1a — Basics: Numbers, Equations & Exponentials
Content file : `seminar_maths4ml_basics_numbers_equations_content.tex`
Driver       : `Main_Seminar_MathsML_Basics_NumbersEquations_{Presentation,CheatSheet}.tex`
Covers       : maths_basics_intro, maths_basics_numbers, maths_basics_equations,
               maths_basics_linearequations, maths_basics_gauss, maths_basics_exponentials
Est. slides  : ~92

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 99 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 1b — Basics: Sets, Measure & Proofs
Content file : `seminar_maths4ml_basics_sets_proofs_content.tex`
Driver       : `Main_Seminar_MathsML_Basics_SetsProofs_{Presentation,CheatSheet}.tex`
Covers       : maths_basics_sets, maths_basics_measure, maths_basics_proofs
Est. slides  : ~76

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 83 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 2a — Linear Algebra: Vectors & Vector Spaces
Content file : `seminar_maths4ml_linearalgebra_vectors_content.tex`
Driver       : `Main_Seminar_MathsML_LinearAlgebra_Vectors_{Presentation,CheatSheet}.tex`
Covers       : maths_linearalgebra_intro, maths_linearalgebra_vectors_intro,
               maths_linearalgebra_vectorspaces, maths_linearalgebra_vectors_multiplication
Est. slides  : ~75

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 82 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 2b — Linear Algebra: Matrices & Applications
Content file : `seminar_maths4ml_linearalgebra_matrices_content.tex`
Driver       : `Main_Seminar_MathsML_LinearAlgebra_Matrices_{Presentation,CheatSheet}.tex`
Covers       : maths_linearalgebra_matrices, maths_linearalgebra_numpy,
               maths_linearalgebra_python, maths_linearalgebra_summary
Est. slides  : ~66

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 73 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 3a — Calculus: Functions & Limits
Content file : `seminar_maths4ml_calculus_functions_limits_content.tex`
Driver       : `Main_Seminar_MathsML_Calculus_FunctionsLimits_{Presentation,CheatSheet}.tex`
Covers       : maths_calculus_intro, maths_calculus_functions, maths_calculus_limits
Est. slides  : ~49 (under the 60-80 target — Calculus totals only ~102 active frames,
               too small to split into two 60-80 halves; split anyway per user direction)

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 56 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 3b — Calculus: Derivatives & Optimization
Content file : `seminar_maths4ml_calculus_derivatives_optimization_content.tex`
Driver       : `Main_Seminar_MathsML_Calculus_DerivativesOptimization_{Presentation,CheatSheet}.tex`
Covers       : maths_calculus_derivatives, maths_calculus_optimization, maths_calculus_conclusion
               (maths_calculus_integration stays commented-out/TBD as in the source)
Est. slides  : ~53 (under the 60-80 target — see Seminar 3a note)

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 60 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 4a — Statistics: Foundations of Probability
Content file : `seminar_maths4ml_statistics_probability_foundations_content.tex`
Driver       : `Main_Seminar_MathsML_Statistics_ProbabilityFoundations_{Presentation,CheatSheet}.tex`
Covers       : maths_statistics_intro, maths_statistics_probability_intro,
               maths_statistics_probability_multi, maths_statistics_bayes
Est. slides  : ~64

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 71 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 4b — Statistics: Random Variables & Distributions
Content file : `seminar_maths4ml_statistics_random_distributions_content.tex`
Driver       : `Main_Seminar_MathsML_Statistics_RandomDistributions_{Presentation,CheatSheet}.tex`
Covers       : maths_statistics_random, maths_statistics_probability_distributions_intro,
               maths_statistics_probability_distributions_examples
Est. slides  : ~41 (under the 60-80 target — Probability totals ~105 active frames,
               too small to split into two 60-80 halves)

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 48 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 5a — Statistics: Central Tendency & Spread
Content file : `seminar_maths4ml_statistics_centraltendency_spread_content.tex`
Driver       : `Main_Seminar_MathsML_Statistics_CentralTendencySpread_{Presentation,CheatSheet}.tex`
Covers       : maths_statistics_descriptive_intro, maths_statistics_descriptive_central,
               maths_statistics_descriptive_spread, maths_statistics_descriptive_asymmetry
Est. slides  : ~93

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 100 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 5b — Statistics: Distributions & Expected Value
Content file : `seminar_maths4ml_statistics_distributions_expectedvalue_content.tex`
Driver       : `Main_Seminar_MathsML_Statistics_DistributionsExpectedValue_{Presentation,CheatSheet}.tex`
Covers       : maths_statistics_distributions_intro, maths_statistics_distributions_examples,
               maths_statistics_descriptive_expectedvalue
Est. slides  : ~82

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 89 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 6a — Statistics: Hypothesis Testing Fundamentals
Content file : `seminar_maths4ml_statistics_hypothesis_testing_content.tex`
Driver       : `Main_Seminar_MathsML_Statistics_HypothesisTesting_{Presentation,CheatSheet}.tex`
Covers       : maths_statistics_inferential, maths_statistics_tests_hypothesis,
               maths_statistics_tests_pvalue_intro, maths_statistics_tests_pvalue_examples
Est. slides  : ~62

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 69 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Seminar 6b — Statistics: Statistical Tests in Practice
Content file : `seminar_maths4ml_statistics_tests_practice_content.tex`
Driver       : `Main_Seminar_MathsML_Statistics_TestsPractice_{Presentation,CheatSheet}.tex`
Covers       : maths_statistics_tests_ttest, maths_statistics_tests_anova
Est. slides  : ~46 (under the 60-80 target — Inferential totals ~108 active frames,
               too small to split into two 60-80 halves)

- [x] Create content file
- [x] Create driver pair (Presentation + CheatSheet)
- [x] Compile Presentation : builds clean, 53 pages
- [ ] Compile CheatSheet (deferred)
- [ ] Upgrade : `/upgrade-deck` (intuition-first pass)

---

## Compile all 24 drivers
- [x] Compile all 12 `*_Presentation.tex` and record page counts — all 12 build clean:
      Basics_NumbersEquations 99p, Basics_SetsProofs 83p, LinearAlgebra_Vectors 82p,
      LinearAlgebra_Matrices 73p, Calculus_FunctionsLimits 56p,
      Calculus_DerivativesOptimization 60p, Statistics_ProbabilityFoundations 71p,
      Statistics_RandomDistributions 48p, Statistics_CentralTendencySpread 100p,
      Statistics_DistributionsExpectedValue 89p, Statistics_HypothesisTesting 69p,
      Statistics_TestsPractice 53p
- [ ] Compile all 12 `*_CheatSheet.tex` and record page counts (deferred)
- [ ] Confirm all 24 build clean (no errors, cosmetic hbox warnings OK)

## Rewire the workshop
- [ ] Replace the raw `\input{maths_*}` calls in `workshop_maths4ml_content.tex` with
      the 12 `\input{seminar_maths4ml_<parenttopic>_<subtopic>_content}` calls, preserving
      section headers (Basics / LinAlg / Calculus / Stats) and the commented-out
      reference-only files (maths_refs, maths_awsmathforml_intro, maths_linearalgebra_grimmer,
      maths_linearalgebra_uky, maths_calculus_oxford, maths_discretemathematics_*,
      maths_gametheory, maths_optimization_uky, maths_misc_uky,
      maths_statistics_probability_duke, maths_statistics_probability_kale,
      maths_statistics_python_implementations) exactly as-is.
- [ ] Compile full workshop : `texify -cp Main_Workshop_ML_Maths_Presentation.tex`
- [ ] Compile full workshop CheatSheet : `texify -cp Main_Workshop_ML_Maths_Cheatsheet.tex`
- [ ] Update CLAUDE.md "Known issues" line that currently lists
      `workshop_maths4ml_content.tex` as still using raw files — remove it once done.

## After all seminars done
- [ ] Confirm no other driver/course file references the old raw-file inputs directly
- [ ] Spot-check that `course_machinelearning_content.tex` (which pulls in
      `workshop_maths4ml_content.tex`? verify) still compiles end-to-end
