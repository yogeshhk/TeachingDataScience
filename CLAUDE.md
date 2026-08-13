# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

An open-source educational repository containing:
- **LaTeX**: Beamer presentation slides and two-column cheatsheets for Data Science courses (Python, ML, DL, NLP, GenAI, RAG, etc.)
- **Code**: Python scripts and Jupyter notebooks demonstrating the concepts covered in the slides

## Style Conventions

User-visible markdown (`README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `COURSES.md`,
`Code/README.md`, and similar reader-facing docs) must not use em-dashes (`—`); use a colon,
comma, or semicolon instead, whichever reads most naturally. This is checked and enforced as
of Jul 2026. Internal/tooling files (this `CLAUDE.md`, other `CLAUDE.md`s, templates) are
exempt since they're written for Claude Code, not repo visitors.

## Repo popularity/discoverability upgrade (Jul 2026)

This repo's own discoverability upgrade is done (README rewritten as a landing page, `COURSES.md`,
`Code/README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates, `Admin/` cleanup;
see git history and `Code/reports/` for details). That work produced two generic slash commands,
`/upgrade-repo-tech` and `/upgrade-repo-non-tech` (in `~/.claude/commands/`, mirrored in
`Code/claudecode/dot_claude/commands/`), meant to be run next on sibling repos: `/upgrade-repo-non-tech`
on `BharatVidya` (no code, pure content), `/upgrade-repo-tech` on `MidcurveNN` and `Sarvadnya` (both
real codebases). Not yet run on any of them as of this note; run one at a time, plan-then-approve.

## LaTeX Build System

Compile a specific deck from the `LaTeX/` directory using MikTeX's `texify`:
```bat
cd LaTeX
texify -cp Main_Seminar_AI_ClaudeCode_Presentation.tex
```

Compile all decks matching a pattern (Windows):
```bat
cd LaTeX
for /r %i in (Main_Seminar_*Educators*.tex) do texify -cp %i
```

Compile everything:
```bat
cd LaTeX
make_all.bat
```

## LaTeX Architecture

### 4-level content hierarchy

```
Course (40hr)     Main_Course_*_{Presentation,CheatSheet}.tex
                    └─ course_*_content.tex
                         └─ \input{workshop_*_content}  (+ course-specific extras)

Workshop (4-16hr) Main_Workshop_*_{Presentation,CheatSheet}.tex
                    └─ workshop_*_content.tex
                         └─ \input{seminar_*_content}  (seminar layer only)

Seminar (1hr)     Main_Seminar_*_{Presentation,CheatSheet}.tex
                    └─ seminar_*_content.tex
                         └─ \input{<domain>_<topic>}  (raw topic files)
```

Every deliverable has two output forms sharing the same content file:
- `Main_*_Presentation.tex` — Beamer slides (`\documentclass{beamer}`, uses `template_presentation.tex`)
- `Main_*_CheatSheet.tex` — Two-column landscape notes (`\documentclass{article}`, uses `template_cheatsheet.tex`)

CheatSheet column count convention: Seminars use `multicols{3}`; Workshops use `multicols{2}`.

**Never use a float (`\begin{table}[h]`, `\begin{figure}`) in any content file that feeds a
CheatSheet** (Aug 2026). Floats cannot be placed inside `multicols` and are silently
*dropped* — no LaTeX error, no warning, the table just vanishes from the PDF while the
surrounding prose renders normally. Use `\begin{center}` + a bare `tabular` instead. Found
when two newly-added tables in `ml_concepts_short.tex` rendered fine in the Presentation but
were missing entirely from the CheatSheet; only caught by rendering the PDF to an image, not
by the compile log.

**Two-column `adjustbox`+`minipage` frames need `%` line-endings** (Aug 2026). The repo's
established side-by-side convention (explanatory content left ~0.55`\linewidth`, diagram right
~0.4`\linewidth`, see Task 5a in `/upgrade-deck`) sums to ~0.96`\linewidth`, leaving no room
for the inter-line spaces LaTeX inserts at each newline between `}`, `\hfill`, and the next
`\adjustbox`. Those few pt tip it over `\linewidth`, the right minipage wraps *below* the
left one, and the diagram runs off the bottom of the slide. Terminate those lines with `%`
(`\end{minipage}%`, `}%`, `\hfill%`, `\adjustbox{valign=t}{%`). This silently broke the
gradient-descent frame in `ml_concepts{,_short}.tex` from its Aug 2026 authoring until it was
found by rendering slide 15 to an image; the only log signal was a single `Overfull \vbox`.

`\usepackage{beamerarticle}` in `template_cheatsheet.tex` makes Beamer `\begin{frame}` environments compile correctly in article mode — no frame stripping needed.

Both `template_presentation.tex` and `template_cheatsheet.tex` load `\usepackage{upquote}`
(added Jul 2026) right after `\usepackage{listings}`, so straight apostrophes/backticks in
`lstlisting`/verbatim code render as straight quotes instead of the curly OT1-typewriter-font
ligature glyph. Repo-wide fix, not tied to any one deck; found while reviewing ML CoEP Session
4 (`python_intro_pandas.tex`), whose code was rendering both ends of every quoted string curled
the same direction.

### Naming conventions
- Topic files: `<domain>_<topic>.tex` (e.g., `maths_linearalgebra_matrices.tex`)
- Content aggregators: `<type>_<subject>_content.tex`
- Driver files: `Main_[Course|Seminar|Workshop]_<Subject>_[Presentation|CheatSheet].tex`
  - Seminar ≈ 1 hour, Workshop ≈ 1 day, Course ≈ 1 week/semester
- Every Seminar and Workshop **must** have both a `_Presentation.tex` and a `_CheatSheet.tex` driver
  — this pairing rule holds even for the lighter-weight `_Overview` driver category below; only
  the "Seminar ≈ 1 hour" duration expectation doesn't apply to that category
- `_Short` suffix on a driver (e.g. `Main_Seminar_Tech_CareerInDataScience_Short_Presentation.tex`)
  denotes a shorter-duration variant of an existing seminar, sharing topic files with its
  parent via the `X.tex`/`X_short.tex` comment-sync pattern below — see the
  CareerInDataScience note for the first precedent of this at the seminar level
- `_overview` suffix on a topic file (e.g. `dl_intro_overview.tex`) denotes a deep/
  comprehensive standalone treatment that sits *outside* the `X.tex`/`X_short.tex`
  comment-sibling relationship — see the short/full sync audit note below
- `_Overview` suffix on a **driver** (e.g. `Main_Seminar_DL_Foundations_Overview_
  Presentation.tex`) denotes a minimal, single-section standalone seminar that exists
  solely to make an `_overview.tex` topic file independently reachable as its own
  session — deliberately thinner than a normal ~1hr Seminar (no References section,
  by design) — see the short/full sync audit note below for the 4 precedents

### Sibling-file sync rule (standing rule, added Jul 2026)
Whenever a `.tex` file being edited — via `/upgrade-deck` or any other change — has an
`X.tex`/`X_short.tex` comment-sibling (per the naming convention above), check for that
sibling and read it too, even if the driver you were pointed at doesn't `\input` it. Any
frame added, removed, or materially edited in one file must have its comment/uncomment
state mirrored in the sibling, so the two never silently drift apart. This is now built
into the `/upgrade-deck` skill itself (Step 4 and its Guardrails), but applies to any
manual edit of a sibling-paired file too. The CoEP course restructuring (see the ML CoEP
note below) is the first case that leaned on this at scale, discovering along the way
that several `_short.tex`/parent pairs already had far more content commented out by
the original author than a naive `\begin{frame}` grep would suggest — always count *live*
(uncommented) frames, not raw occurrences, when judging a deck's size.

### Known issues
- `seminar_latex4research_conent.tex` — filename typo (`conent` vs `content`); the file and all references would need renaming together
- `Main_Seminar_AI_ClaudeCode_CheatSheet.tex` (only active content: `ai_tools_claudecode_demo_cadcam.tex`) walks through building `stlinspector`, paired with actual code at `Code/claudecode/CadCamWorkshop/` (untracked as of Jul 2026). As of Jul 2026 the deck and the PoC are back in sync: flat `src/` layout with no packaging (no `pyproject.toml`, no console-script entry point), the two-step `load_mesh`/`inspect_mesh` API, JSON-only reports (no Markdown report format), and a `thin_walls` check added alongside the original three. `CadCamWorkshop` now has both `.claude/skills/geometry-validation/` and `.claude/skills/inspection-report-summary/`; `.claude/agents/devops.md` was removed. `Code/claudecode/trial/` (also untracked) was a from-scratch dry run of the same workshop script, used to find and fix these drift points plus several missing/misplaced YAML-frontmatter fences in the tex's subagent/command/skill blocks — it's now redundant and pending manual deletion.
- `workshop_deepnlp_content.tex` line 2 has `\input{nlp_intro_short}` but no `nlp_intro_short.tex` exists in `LaTeX/` (closest matches: `nlp_intro_short_old.tex`, `nlp_intro_short_w_embedding.tex`, `nlp_intro.tex`) — blocks `Main_Workshop_NLP_Deep_*` and (via `course_generativeai_content.tex`) `Main_Course_GenerativeAI_*` from compiling. Found during the short/full sync audit below (Oct 2026); pre-existing and unrelated, not fixed.

### Machine Learning course restructured (June 2026)
Full 4-level hierarchy for the 40-hour ML course "Machine Learning for Graduate Students":
- **Course**: `Main_Course_MachineLearning_{Presentation,CheatSheet}.tex` → `course_machinelearning_content.tex`
- **Workshops** (6 × driver pairs):
  - W1 Python for ML (8h): existing `workshop_python_basic_content.tex` (renamed from `workshop_python_content.tex` in Aug 2026, see Python course note below)
  - W2 Foundations (4h): `workshop_ml_foundations_content.tex`
  - W3 Regression (4h): `workshop_ml_regression_content.tex`
  - W4 Tree-Based & Ensemble (8h): `workshop_ml_treebased_content.tex`
  - W5 Supervised II — KNN/SVM/NB (8h): `workshop_ml_supervisedII_content.tex`
  - W6 Unsupervised & Deployment (8h): `workshop_ml_unsupervised_content.tex`
  - Standalone all-ML workshop (W2–W6, no Python/demos): `Main_Workshop_MachineLearning_{Presentation,CheatSheet}.tex`
- **Seminars** (10 × driver pairs): `seminar_ml_{intro,dataprep,regression,decisiontree,ensemble,knn,svm_nb,clustering,dimreduction,deployment}_content.tex`; drivers are `Main_Seminar_ML_{Intro,DataPrep,Regression,DecisionTree,Ensemble,KNN,SVM_NB,Clustering,DimReduction,Deployment}_{Presentation,CheatSheet}.tex`
- **New demo/assign files**: `ml_course_demo_regression_housing.tex`, `ml_course_demo_svm_digits.tex`, `ml_course_assign_knn_wine.tex`, `ml_course_demo_clustering_customers.tex`, `ml_course_assign_pca_digits.tex`
- **Upgrade status**: All 10 seminars upgraded (compiled clean + `/upgrade-deck` pass: technical
  fixes, "Intuition" callouts, "Quick Check" quizzes) as of Jul 2026 — see git history for
  details, as `LaTeX/todo_ml_seminar_upgrade.md` (the working to-do for this pass) was deleted
  once the work completed, matching the precedent set by the Maths4ML/Python restructuring notes
  below
- **Done**: the 5 new demo/assign files listed above are wired into `course_machinelearning_content.tex` (confirmed on disk, Oct 2026 audit)

### Maths for ML restructured (July 2026), promoted to a course (Aug 2026)
Full 4-level hierarchy for "Zero-to-Hero: Mathematics for Machine Learning", aimed at
fresher/college-level students. 12 seminars × ~2h = 24h ≈ 3 days × 8h, but the 4 topic-workshops
are uneven in size (Basics/LinearAlgebra/Calculus 4h each, Statistics 12h), so the day boundary
cuts across the Calculus/Statistics workshop pair rather than aligning 1:1 with workshops —
annotated as comments in `course_maths4ml_content.tex` and inside
`workshop_maths4ml_statistics_content.tex` (not a structural split):
- **Day 1 (8h)**: Basics + Linear Algebra
- **Day 2 (8h)**: Calculus + Statistics seminars 1–2 (probability_foundations, random_distributions)
- **Day 3 (8h)**: Statistics seminars 3–6 (centraltendency_spread, distributions_expectedvalue, hypothesis_testing, tests_practice)
- **Course**: `Main_Course_MathsML_{Presentation,CheatSheet}.tex` → `course_maths4ml_content.tex`
- **Workshops** (4 × driver pairs), each just chaining its seminars:
  - Basics: `workshop_maths4ml_basics_content.tex` → `seminar_maths4ml_basics_{numbers_equations,sets_proofs}_content.tex`; drivers `Main_Workshop_MathsML_Basics_{Presentation,CheatSheet}.tex`
  - Linear Algebra: `workshop_maths4ml_linearalgebra_content.tex` → `seminar_maths4ml_linearalgebra_{vectors,matrices}_content.tex`; drivers `Main_Workshop_MathsML_LinearAlgebra_{Presentation,CheatSheet}.tex`
  - Calculus: `workshop_maths4ml_calculus_content.tex` → `seminar_maths4ml_calculus_{functions_limits,derivatives_optimization}_content.tex`; drivers `Main_Workshop_MathsML_Calculus_{Presentation,CheatSheet}.tex`
  - Statistics (6 seminars): `workshop_maths4ml_statistics_content.tex` → `seminar_maths4ml_statistics_{probability_foundations,random_distributions,centraltendency_spread,distributions_expectedvalue,hypothesis_testing,tests_practice}_content.tex`; drivers `Main_Workshop_MathsML_Statistics_{Presentation,CheatSheet}.tex`
- **Seminars** (12 × driver pairs, unchanged): each has its own driver pair
  `Main_Seminar_MathsML_<ParentTopic>_<Subtopic>_{Presentation,CheatSheet}.tex`
- All 12 seminars have been through an intuition-first `/upgrade-deck` pass (technical fixes,
  "Intuition" callouts, section-end "Quick Check" quizzes) — see git history for details, as
  `LaTeX/todo_maths4ml_seminar_upgrade.md` (the working to-do for this restructuring) was
  deleted once the work completed.
- Raw `maths_*.tex` topic files are unchanged; only the aggregation layers changed.
- The old single all-in-one `Main_Workshop_ML_Maths_{Presentation,Cheatsheet}.tex` /
  `workshop_maths4ml_content.tex` were removed as redundant once the course/workshop split
  landed (unlike the ML course, no standalone "complete workshop" was kept here).

### Python course added (Aug 2026), seminar layer added
2-day, 16h course combining the two existing standalone Python workshops as Day 1 / Day 2:
- **Course**: `Main_Course_Python_{Presentation,CheatSheet}.tex` → `course_python_content.tex`
- **Day 1 (8h)**: `workshop_python_basic_content.tex` (renamed from `workshop_python_content.tex`;
  also still used standalone via `Main_Workshop_Python_Basic_{Presentation,CheatSheet}.tex`, and
  as W1 "Python for ML" in `course_machinelearning_content.tex`)
- **Day 2 (8h)**: `workshop_python_adv_content.tex` (unchanged; also still used standalone via
  `Main_Workshop_Python_Advanced_{Presentation,CheatSheet}.tex`)
- Both workshops now route through a seminar layer (6 seminars each) between the workshop and
  the raw `python_*.tex` topic files, matching the Maths4ML/ML hierarchy:
  - **Basic** (`workshop_python_basic_content.tex`): B1 Intro, B2 Constructs, B3 Procedures,
    B4 OOP, B5 IOLibraries, B6 Closure — `seminar_python_basic_<name>_content.tex`; drivers
    `Main_Seminar_Python_Basic_<Name>_{Presentation,CheatSheet}.tex`
  - **Advanced** (`workshop_python_adv_content.tex`): A1 OOPIteration, A2 FunctionsOS,
    A3 StringsWeb, A4 DataLibs, A5 Visualization, A6 Problems —
    `seminar_python_adv_<name>_content.tex`; drivers
    `Main_Seminar_Python_Advanced_<Name>_{Presentation,CheatSheet}.tex`
  - `Extra`/`References` sections stay as trailing raw `\input`s in the workshop content files,
    not wrapped in a seminar (same precedent as Maths4ML/ML)
  - `python_oop.tex` is `\input` by both Basic B4 and Advanced A1 — duplication predates this
    restructure, preserved as-is
- Raw `python_*.tex` topic files are unchanged; only the aggregation layers changed.
- All 12 seminars (Basic B1-B6, Advanced A1-A6) have been through an intuition-first
  `/upgrade-deck` pass (technical fixes, "Intuition" callouts, "Quick Check" quizzes) — see git
  history for details, as `LaTeX/todo_python_seminar_restructure.md` (the working to-do for this
  restructuring) was deleted once the work completed. Notable fixes along the way: Advanced A5
  (`python_intro_bokeh.tex`) was rewritten from a Bokeh procedural API removed from the library
  ~8 years ago to current Bokeh 3.x, and its dependent `bokeh.charts`/`bokeh.mpl` demo frames
  were replaced with native-Bokeh equivalents since those modules no longer exist; A5's
  `python_intro_tkinter.tex` had ~1850 lines of dead, never-rendered commented-out Python-2-era
  Tcl/Tk documentation removed. Advanced A6's raw files (`python_dsa.tex`,
  `python_codingproblems_basic.tex`, `python_systemdesign.tex`) were recently AI-authored and
  algorithmically correct throughout, so that pass was lighter — mainly mismatched
  reference-citation cleanup and one `itertools.permutations` simplification.
- No redundant files removed here: unlike Maths4ML, both standalone workshops remain valid
  independent offerings, so nothing was retired.

### CareerInDataScience seminar split into 90-min full + 30-min short (Oct 2026)
First precedent in the repo for a single seminar offered at two durations, sharing
underlying topic files kept in sync by commenting rather than by duplicating content
independently:
- **Full (90 min, unchanged)**: `Main_Seminar_Tech_CareerInDataScience_{Presentation,CheatSheet}.tex`
  → `seminar_careerindatascience_content.tex` (Background, Introduction, Challenges,
  Roles \& Personas, Preparation, Mid-career, References)
- **Short (30 min, new)**: `Main_Seminar_Tech_CareerInDataScience_Short_{Presentation,CheatSheet}.tex`
  → `seminar_careerindatascience_short_content.tex`, dropping Background/Challenges/
  Mid-career entirely (their `\section`+`\input` lines commented out, not deleted) and
  swapping the rest to `_short` topic files: `ai_intro_tech_short.tex` (~11 orientation
  slides), `career_ai_roles_short.tex` (8 of ~16 roles), `career_ai_personas_short.tex`
  (all 3 personas, trimmed detail), `career_ai_prep_short.tex` (5 of ~12 slides);
  `career_refs.tex` kept in full (only 2 frames)
- **Sync convention**: each `_short.tex` sibling is a full copy of its parent with the
  excluded frames commented out (not rewritten/deleted), so a frame added to the parent
  can be manually mirrored into the child as either live or commented — the same
  discipline as the pre-existing repo-wide `X.tex`/`X_short.tex` pattern (e.g.
  `dl_intro.tex`/`dl_intro_short.tex`), now extended to a full seminar-level split
- `ai_intro_tech_short.tex` — shared by 5 other decks (`course_deeplearning_content`,
  `seminar_artificialintelligencemachinelearning_content`,
  `seminar_artificialintelligence_tech_content`, `seminar_machinelearning_content`, and
  commented in `seminar_llm_genai_content`) — was renamed to `ai_intro_tech.tex` first
  (content-preserving; it was never actually short, just misnamed) so a genuine
  ~11-slide `ai_intro_tech_short.tex` could be created without touching those decks
- Both variants went through an `/upgrade-deck` pass, each reviewed as its own
  standalone artifact; working notes (now complete) were in
  `LaTeX/todo_careerindatascience_split.md`

### Short/full topic-file sync audit (Oct 2026)
Repo-wide audit of all `X.tex`/`X_short.tex` pairs against the comment-sibling
convention (see CareerInDataScience note above). Of 24 pairs checked, 13 were
already in sync, 3 near-miss pairs and 3 missing-only pairs got small content
fixes, and 5 pairs had drifted so far apart (independently authored, not a
subset relationship at all) that they were split into a three-file group —
**this introduces a new file-naming pattern**: `<topic>_overview.tex` now
denotes a deep/comprehensive standalone treatment that sits *outside* the
comment-sibling relationship, while `<topic>.tex`/`<topic>_short.tex` were
freshly authored as a genuine (smaller) comment-sibling pair distilled from
it. The five:
- `dnlp_intro_overview.tex` (renamed from the old `dnlp_intro.tex`) —
  Kirill Eremenko technical walkthrough (Seq2Seq/Attention/Decoding), used by
  `workshop_deepnlp_content.tex`. Fresh `dnlp_intro.tex`/`_short.tex` merge in
  the old short's foundational material (Turing Test, NLP tasks, embeddings
  basics), used by `seminar_deepnaturallanguageprocessing_content.tex`.
- `data_intro_overview.tex` (renamed from the old `data_intro.tex`) —
  technical data-types/distance-metrics deep dive (NOIR, Euclidean/Minkowski,
  SMC, Cosine Similarity), used by `seminar_data_tensorflow_content.tex`'s
  "Basic Concepts in Data" section (a good label fit, unlike before). Fresh
  `data_intro.tex`/`_short.tex` are a motivational/historical intro (Man on
  the Moon, "Data is the New Oil", the 4 Vs, Target case study), used by
  `workshop_dataanalytics_content.tex`'s "Introduction" section (kept on the
  plain filename — a deliberate deviation from the mechanical rename, since
  the old full's technical content never fit that section's actual title).
- `dl_intro_overview.tex` (renamed from the old `dl_intro.tex`) — the deeper
  synthesis (still includes backprop math, automatic differentiation,
  optimizer plots), used by `seminar_deeplearning_foundations_content.tex`.
  Fresh `dl_intro.tex`/`_short.tex` condense both old files' strengths into a
  non-mathematical walkthrough, used by `seminar_deeplearning_content.tex`.
- `python_syntax_overview.tex` (renamed from the old `python_syntax.tex`,
  content unchanged) — this one had already been through `/upgrade-deck` as
  part of the Python seminar restructuring's B1 (`seminar_python_basic_intro_
  content.tex`); the rename was verified to preserve that work exactly (same
  118-page compile). Fresh `python_syntax.tex`/`_short.tex` are a broader
  "Python Basics" primer (the old short's actual scope, despite its narrow
  name), used by `course_deeplearning_content.tex`'s W1 recap. **Retired in
  the Jul 2026 Python audit below**: turned out to be a duplicate copy of
  `python_syntax_short.tex`'s content rather than distinct material, so the
  file was deleted and B1 repointed to `python_syntax_short.tex`.
- `nlp_embedding_overview.tex` (renamed from the old `nlp_embedding.tex`) —
  the authoritative, current (2023-2026) material: BERT, RAG systems,
  multimodal/CLIP, bias/ethics, a full Tweet-Sentiment-with-Word2Vec code case
  study. Used by `seminar_wordembeddings_content.tex` (a seminar wholly about
  embeddings, so its "Introduction" section is really the whole seminar's
  substance — repointed here rather than mechanically). Fresh
  `nlp_embedding.tex`/`_short.tex` condense the overview's structure, used by
  `seminar_nlp_advanced_content.tex` (one topic among several in a broader
  workshop) and `workshop_deepnlp_content.tex`.
- Working notes (now complete) were in `LaTeX/todo_short_full_sync_audit.md`,
  deleted once the work finished — same precedent as the other `todo_*`
  restructuring files noted elsewhere in this document.

**Dedicated standalone seminars for 4 of the 5 overview files (Oct 2026,
additive follow-up; one retired Jul 2026, see below)**: each of the 5
`_overview.tex` files is used by an existing consumer deck that specifically
needs its depth (documented above) — those consumers were deliberately left
as-is, not repointed. Instead, 4 of the 5 also got a *new*, minimal,
single-section standalone seminar so the deep-dive content is independently
reachable as its own session (the 5th, `nlp_embedding_overview.tex`, already
had one — `seminar_wordembeddings_content.tex`'s only real section is the
overview, backed by `Main_Seminar_NLP_WordEmbeddings_{Presentation,
CheatSheet}.tex`):
- `dnlp_intro_overview.tex` → `seminar_dnlp_overview_content.tex` →
  `Main_Seminar_NLP_DNLP_Overview_{Presentation,CheatSheet}.tex`
- `data_intro_overview.tex` → `seminar_dataconcepts_overview_content.tex` →
  `Main_Seminar_Data_Concepts_Overview_{Presentation,CheatSheet}.tex`
- `dl_intro_overview.tex` → `seminar_dl_technical_overview_content.tex` →
  `Main_Seminar_DL_Foundations_Overview_{Presentation,CheatSheet}.tex`
- ~~`python_syntax_overview.tex` → `seminar_python_syntax_overview_content.tex`
  → `Main_Seminar_Python_Syntax_Overview_{Presentation,CheatSheet}.tex`~~ —
  **retired Jul 2026**: `python_syntax_overview.tex` turned out to be a
  duplicate copy of `python_syntax_short.tex` rather than distinct material,
  so both the topic file and this standalone seminar (+ its 2 drivers) were
  deleted; see the Python raw-file audit note below. `COURSES.md`'s "Deep
  dives / overviews" list was updated to drop the dead link.

Each content wrapper is deliberately minimal: a single `\section[Overview]
{Overview}` + `\input{<topic>_overview}`, no References section (unlike most
seminars, by design — these are supplementary deep-dive sessions, not full
independent courses). All 8 driver files compiled clean at the time (now 6,
after the Jul 2026 retirement above).

### Python raw-file overlap/redundancy audit + seminar consolidation (Jul 2026)
Cleanup triggered by the CoEP `python_overview.tex` rebuild (see `Code/mlcoep/`)
pulling content from several other `python_*.tex` files, raising the concern that
the ~45+ raw `python_*.tex` files and several `seminar_python_*.tex` wrappers had
accumulated real overlap. Two-pass audit (raw-file consumer map + content diff,
then a seminar-level consolidation pass), executed after explicit sign-off on
each decision point:
- Deleted `python_intro_verbose.tex` — zero live consumers, and structurally
  incompatible anyway: it was book-chapter prose (`\chapter`/`\section`/
  `Verbatim`), not Beamer frames, so it could never have compiled as slide
  content without a full rewrite.
- Wired `python_special.tex` (also zero consumers) into
  `seminar_python_adv_oopiteration_content.tex` (A1) — natural home since that
  seminar already covers OOP/iterators/generators.
- Deleted `python_syntax_overview.tex`, its standalone seminar, and drivers
  (see the correction to the Oct 2026 sync-audit note above) — its live frame
  set was a duplicate copy of `python_syntax_short.tex`, not distinct
  material. `seminar_python_basic_intro_content.tex` (B1) now `\input`s
  `python_syntax_short` instead.
- Trimmed `python_advanced_topics_overview.tex` from 43 → 10 frames (Course
  Overview, Context Managers, Threading, Async/Await, and 6 worked Projects —
  content confirmed to have no home anywhere else in the repo) and renamed it
  `python_advanced_projects.tex`. The other ~32 frames duplicated ground
  already owned by dedicated raw files (`python_datatypes`, `python_oop`,
  `python_fileio`, etc.) or by `python_intro_short.tex`.
- Rewired `seminar_python_content.tex` — the "Python crash-course" seminar
  backing `Main_Seminar_Python_{Presentation,CheatSheet}.tex` and embedded as
  the Python prerequisite section inside `workshop_ai_content.tex` and
  `workshop_ragtoriches_content.tex` — to `\input{python_intro_short}` +
  `\input{python_advanced_projects}`, keeping `python_dsa`/
  `python_systemdesign`/`python_refs_short` unchanged (genuinely shared with
  A6, not duplicated). This is distinct in purpose from
  `seminar_python_overview_content.tex` (`Main_Seminar_Python_Overview_*`,
  just `\input{python_overview}`) — the crash-course assumes basics already
  known and jumps to advanced/practical patterns + DSA/system-design; the
  Overview teaches Python from scratch. No content overlap between the two
  after this cleanup; deliberately not merged.
- Found and fixed a pre-existing, unrelated bug while compile-checking:
  `workshop_ai_content.tex` line 5 did
  `\input{seminar_artificialintelligence_content}`, a file that never existed
  in `LaTeX/`. Repointed to `seminar_artificialintelligence_tech_content.tex`
  (the file's own commented-out alternative, labeled "tech-audience AI
  content" — a fit for this broad technical workshop; the other alternative,
  `seminar_artificialintelligence_tools_content.tex`, is entirely
  commented-out/dead). `Main_Workshop_AI_{Presentation,CheatSheet}.tex` now
  compile clean.

### ML CoEP course session rebalancing (Jul 2026)
`course_mlcoep_content.tex` (driving `Main_Course_ML_CoEP_{Presentation,CheatSheet}.tex`,
"AI-ML for Mechanical Engineers" — a bespoke 19-session course for CoEP, College of
Engineering Pune) was audited and rebalanced against a ~50-55-live-frame-per-1hr-session
target, going from 21 planned sessions (only Session 3 active, everything else a
commented placeholder) to 19 fully-active sessions:
- **Critical methodology finding**: raw `\begin{frame}` grep counts are unreliable across
  this repo — many files already have large chunks commented out by the original author
  (e.g. `ml_concepts.tex` showed 147 by grep but only 89 live frames). Always count *live*
  (uncommented, `^\begin{frame}`-anchored) frames when judging a deck's size; see the
  sibling-file sync rule above, which this finding fed into.
- **Merges**: Pandas + Data Prep (old #3+#4) became a new theory-then-practice pair —
  Session 3 "Understanding Your Data: EDA & Data Prep" (new content, see below) and
  Session 4 "Doing It: Pandas" (`python_intro_pandas` alone). Random Forest merged into
  Ensemble Methods (Session 11) since Random Forest alone was too thin. ML Workflow +
  Data Prep (sklearn) + Model Evaluation merged into one practical session (Session 7),
  absorbing `ml_datapreparation_sklearn` out of Session 4. ME Applications + Project Ideas
  merged (Session 19), moved to after MLOps (Session 18).
- **Split**: the old "ML Concepts & Scikit-Learn Workflow" session split into Sessions 5-7
  (Intro to ML / Core ML Concepts / Sklearn Workflow+DataPrep+Evaluation).
- **New `_short.tex` comment-siblings created** (originals untouched, since several are
  shared with `course_machinelearning_content.tex`'s seminars): `ml_concepts_short.tex`,
  `ml_decisiontree_short.tex`, `ml_naivebayes_short.tex`, `data_preparation_short.tex`
  (the last curated from the previously CoEP-unused `data_preparation.tex`, which turned
  out to be a much better fit for data-prep theory than authoring from scratch).
  `ml_intro_short.tex` already existed (shared with 2 other decks) at exactly the right
  size — reused as-is, nothing created.
  `ml_linearregression.tex`/`ml_logisticregression.tex`/`ml_svm.tex`/`ml_pca.tex` looked
  oversized by raw grep count but were already right-sized once live frames were counted
  correctly — no `_short` needed, used directly.
- **New content authored**: `ml_eda_intro.tex` (why EDA matters, univariate/bivariate,
  6 frames) and `ml_eda_endtoend_churn.tex` (a fresh Load→Assess→Describe→Visualize→
  Engineer→Check-what-matters walkthrough on the telecom churn dataset, following the
  narrative arc of `Code/curiosily_ai_bootcamp/02.exploratory-data-analysis.ipynb`, 14
  frames) — both for Session 3.
- **Renamed** for naming-convention consistency (and fixed a typo), updating references in
  both `course_mlcoep_content.tex` and `course_machinelearning_content.tex`:
  `ml_course_assign4_logisticrgression_alice.tex` → `ml_course_assign_logisticregression_alice.tex`,
  `ml_course_demo3_decisiontree_uciadults.tex` → `ml_course_demo_decisiontree_uciadults.tex`,
  `ml_course_assign3_decisiontree_heartdisease.tex` → `ml_course_assign_decisiontree_heartdisease.tex`.
- **Session 19 exemplars**: 4 of the excluded per-algorithm demo/assign files (housing
  regression, SVM digits, customer clustering, PCA digits) were repurposed as worked
  project exemplars in the new combined ME-Applications session, rather than left unused.
- Each algorithm session (8-16) now includes only its core theory file; the matching
  `_sklearn`/demo/assign companions are deliberately excluded from the active course but
  kept on disk (commented `\input` lines note where each one would go) for a possible
  future practical-companion pass.
- **`python_intro_pandas.tex`** (Session 4) was separately expanded 41→54 live frames in
  the same pass: added Reading Data, Quick Inspection, Value Counts, Rename/Drop Columns,
  Creating New Columns, GroupBy (concept + aggregation + quiz), Concat, Merge, Plotting,
  and a 2-frame Machine-Health-Check capstone — see git history for the exact diff.
- **`upgrade-deck.md`** (`~/.claude/commands/`, mirrored to
  `Code/claudecode/dot_claude/commands/`) was edited during this pass: removed the
  quantum-physics/quantikz sub-tasks (1b/1c, unused anywhere in this repo) and added a
  Step 4 "Sibling-File Check" + a "Sibling sync" guardrail, per the standing rule above.
  Keep both copies mirrored on any future edit.
- Verification is tracked session-by-session in `LaTeX/todo.md` (renamed Jul 2026 from
  `todo_mlcoep_session_verification.md` for a shorter name; delete once complete, per the
  usual `todo_*.md` convention) — as of this note, Sessions 1-4 are compiled and verified
  clean (two PDFs each: `Course_MLCoEP_<N>_..._{Presentation,CheatSheet}.pdf`), and a
  repo-wide `lstlisting`-placement-violation sweep (content after `\end{lstlisting}`,
  violates the Style Preservation Rule) found and fixed 12 instances across Sessions 3, 7,
  and 19 — including a genuine content bug caught in passing: `ml_evaluation_sklearn.tex`'s
  `$R^2$ Metric` frame had text copy-pasted from the MSE frame that was factually wrong for
  R² (not negated by `cross_val_score`, unlike MAE/MSE). Sessions 5-19 still need their
  compile-and-verify pass.
- **`prep-mlcoep-session` command** (`~/.claude/commands/`, mirrored to
  `Code/claudecode/dot_claude/commands/`) was changed (Jul 2026) so its Step 3
  (`/upgrade-deck`) is asked about rather than fired automatically — default is to skip it
  and just produce the renamed PDF from as-is content. A second change (also Jul 2026) added
  a new, non-optional Step 2 that sweeps every live `--` prose dash in the target session's
  topic files into a colon/comma before compiling (numeric ranges and table N/A-placeholder
  dashes excluded), renumbering the steps after it; unlike `/upgrade-deck` this one always
  runs, no asking. Keep both command copies mirrored on any future edit, per the standing
  rule above.
- **Em-dash cleanup** (Jul 2026): a literal em-dash (`—`) sweep across all `.tex` files
  reachable from `course_mlcoep_content.tex` (Sessions 1-19 + appendix) found and fixed 5
  instances — 2 in `ai_intro_tech.tex` (Session 1), 1 each in `ml_intro_short.tex` (Session
  5), `ml_svm.tex` (Session 12, a quote-attribution dash, replaced with a plain hyphen `-`
  rather than comma/colon since attribution is the one case treated as "a must"), and
  `ml_predictive_analytics.tex` (Session 18). Zero remain as of this note.
- `COURSES.md` now lists this course (added Jul 2026, after being flagged as a gap) under
  "ML for Mechanical Engineers (CoEP)" — noted there as structurally different from the
  other 5 courses (no workshop/seminar layer; sessions are chained directly, not
  independently reachable).
- **Sessions 3 and 4 deep-upgrade pass (Jul 2026)**, beyond the compile-verify note above:
  - Session 3 (`ml_eda_intro.tex`, `data_preparation_short.tex`, `ml_eda_endtoend_churn.tex`):
    added a clean-vs-messy rain-prediction tabular example (Pressure/Temperature/Humidity →
    Rain) to ground "why EDA matters"; commented out (not deleted) 4 mechanical-engineering-
    flavored "Intuition" blocks per explicit instruction that ME analogies aren't wanted right
    now; added a `df.head()` "first look" frame in the churn walkthrough (renumbering its
    Step 2-7 titles to 3-8), plus closing/recap frames to `ml_eda_intro.tex` and
    `data_preparation_short.tex` (`ml_eda_endtoend_churn.tex` already had one); split the
    Covariance frame's image into its own frame and rewrapped 3 code lines that had no
    whitespace break point (both were genuine vertical/horizontal overflow, not cosmetic);
    added worked examples/tables/code to 6 previously plain-list frames.
  - Session 4 (`python_intro_pandas.tex`): added 12 missing `\frametitle`s; converted all 17
    `df1.png`-`df17.png` screenshots to `lstlisting` text after visually confirming each was
    plain monospace `In[]/Out[]` output with nothing graphical to lose; while converting,
    found and fixed a real bug where the deck interleaves two different tutorials that both
    reused the variable `df` (a classic synthetic `A/B/C/D`-column walkthrough and a
    `machine_logs.csv` walkthrough added in the June 2026 rebalancing), reassigning it back
    and forth — renamed the synthetic thread to `df_demo` throughout (~15 frames) rather than
    reordering anything; also fixed two narrative gaps the images had been silently
    papering over (a "drop rows" frame whose image never showed the drop, and a downstream
    `A`/`B` = 0 change with no earlier step introducing it). Found via visual PDF-page
    rendering (not just line-counting) that any frame with two separate `lstlisting` blocks
    back-to-back visibly collides (each gets its own bordered box); split the 4 affected
    frames into 8. `/upgrade-deck` was deliberately not run on either session this pass (see
    the command note above) — for Session 4 specifically, declined because the file already
    carries "Intuition"/"Quick Check" scaffolding from a prior pass and running it now would
    risk re-adding ME-flavored intuition right after removing it from Session 3.
- **Session 4 hands-on + REPL-disambiguation + overflow pass (Jul 2026)**: triggered by wanting
  students to run the `machine_logs.csv` walkthrough themselves in class with minimum typing.
  - Created `Code/mlcoep/datasets/session04_pandas/machine_logs.csv` (6 rows), reverse-engineered
    to exactly match every output already printed in the deck (shape, `value_counts`, groupby
    averages, vibration flags, morning/evening split) — the file referenced by
    `pd.read_csv('machine_logs.csv')` never actually existed on disk before this.
  - Applied the standard Python REPL `>>>`/`...` convention across every code block in
    `python_intro_pandas.tex` (~40 blocks): typed lines prefixed `>>> `/`... `, output left
    unprefixed, stale mismatched `Out[N]:` cell-number labels removed, one explanatory sentence
    added on the first code frame. This is now Task 1c in `/upgrade-deck` and Step 3 in
    `/prep-mlcoep-session` (see command notes below) — fires on any REPL/notebook-transcript-style
    block, not just this one.
  - Found and fixed 2 real bugs while verifying every code block actually executes (ran the whole
    deck's code end-to-end against current `pandas 2.2.3`, not just read it): `pd.set_option
    ('max_columns', 50)` → `pd.set_option('display.max_columns', 50)` (shorthand key is now
    ambiguous); stale Python-2-era `Index([u'A', ...])` repr → `Index(['A', ...])`.
  - Found and fixed 2 distinct overflow categories, both silent (neither triggers a LaTeX
    warning — `breaklines=true` swallows the horizontal case, and Beamer doesn't warn on vertical
    frame overflow either): a CheatSheet-only (3-column, narrow) horizontal wrap in 4 frames
    (`Reading Data from Files`, `Missing Data`, `Missing Data: Drop`, `Missing Data: Fill`) where
    a wide DataFrame table split mid-row, fixed via `\begin{lstlisting}[basicstyle=\tiny\ttfamily]`
    on just those 4 blocks; and a Presentation-only vertical overflow in `Quick Inspection`
    (`df.shape`+`df.dtypes`+`df.info()` combined, 23 lines — the only outlier among all blocks,
    confirmed by checking every other long block too), fixed by splitting into `Quick Inspection`
    and a new `Quick Inspection: Full Structure` frame, matching the deck's existing
    split-when-too-long precedent (`Adding Two Series Together`/`Adding Two Series: Output`).
  - `upgrade-deck.md` and `prep-mlcoep-session.md` (`~/.claude/commands/`, mirrored to
    `Code/claudecode/dot_claude/commands/`) both updated with the REPL-disambiguation rule above;
    keep both copies mirrored on any future edit, per the standing rule.
- **19-session restructuring + `prep-mlcoep-session` retirement (Aug 2026)**: the single
  `course_mlcoep_content.tex` (19 sessions chained via comment/uncomment isolation, driving one
  `Main_Course_ML_CoEP_{Presentation,CheatSheet}.tex` pair) was replaced with 19 independent
  content files (`seminar_mlcoep_session_<N>_content.tex`) and 19 independent driver pairs
  (`Main_Seminar_MLCoEP_Session_<N>_<ShortName>_{Presentation,CheatSheet}.tex`), compiled in one
  pass via `make_all_sessions.bat` (all 38 drivers) / `make_all_cheatsheets.bat` (CheatSheets
  only) — replacing the old one-session-at-a-time `prep-mlcoep-session` isolate/compile/restore
  pipeline, which was too slow for a full 19-session run. Renamed `Course` → `Seminar` and
  dropped the `course_mlcoep_*` prefix in favor of `seminar_mlcoep_*` to match this repo's
  duration-based naming convention (~1-2hr sessions are Seminars, not Courses); output PDFs take
  their name directly from the driver filename (`Main_Seminar_MLCoEP_Session_<N>_<ShortName>_*.pdf`),
  no separate rename step. `prep-mlcoep-session.md` is deleted from both `~/.claude/commands/`
  and `Code/claudecode/dot_claude/commands/` — its two mandatory (non-optional) preprocessing
  steps were folded into `/upgrade-deck` so it remains the one generic command for
  reviewing/upgrading any deck, MLCoEP session or otherwise: the REPL disambiguation step was
  already present as Task 1c; the prose-dash cleanup step is now Task 1d (new, same exclusions:
  table N/A placeholders, numeric ranges, commented-out blocks, quote-attribution dash → hyphen).
  `course_mlcoep_content.tex` is kept (not deleted, per explicit instruction) and repurposed: it
  now just `\input`s all 19 `seminar_mlcoep_session_<N>_content.tex` files in order plus the
  References/Datasets Used appendix, so `Main_Course_ML_CoEP_{Presentation,CheatSheet}.tex` still
  works as a single all-in-one compile with zero content duplication — both the all-in-one Course
  driver and the 19 independent Seminar drivers pull from the same 19 source files. Per explicit
  instruction, this combined driver is not itself compiled as a verification step once its 19
  constituent seminar files already compile clean individually (see
  `TeachingQuantumTech/LaTeX/qcnp/workshops/README.md` for the precedent this follows).
  `LaTeX/todo.md` (old session-by-session compile-verification checklist, tied to the retired
  isolate/compile/restore procedure) and `LaTeX/todo_mlcoep_19session_pdfs.md` (old
  `prep-mlcoep-session`-resume tracker, including its untouched dash-cleanup survey for Sessions
  3, 7, 9-19) are both deleted — the goal both tracked (verified PDFs for all 19 sessions) is
  achieved via the new mechanism (confirmed via `texify`, zero errors across all 38 PDFs), and any
  session's outstanding dash cleanup will surface automatically the next time that session goes
  through `/upgrade-deck` Task 1d, same as Sessions 5 and 6 below.
- **Sessions 5 and 6 `/upgrade-deck` passes (Aug 2026)**, run individually via their own
  `Main_Seminar_MLCoEP_Session_<N>_*` drivers per the restructuring above:
  - Session 5 (`ml_intro_short.tex`/`ml_intro.tex`): 2 small Task 1 fixes (a formula/parameters
    mismatch in "Model entities": `c` was missing from the named parameters despite being in the
    formula; a run-on typo in "Decision Tree: Feature Selection"). Tasks 2-6 found nothing to
    change, this deck had already been through an earlier pass (6 Quick Check quizzes, one per
    section, already present) and is intentionally example-driven (Pokemon Go, Pizza Shop, Spam
    Detection, University Admission) rather than formula-heavy, so Task 5 added nothing new.
    Flagged (not fixed, out of scope for a surgical pass) a real content-drift between
    `ml_intro_short.tex` and `ml_intro.tex`: each now has substantial live content the other
    lacks entirely (the short version has "Hardware Revolution"/"ML Renaissance 2023-2025"/"Open
    Source Ecosystem 2025"/all 6 quizzes; the full version has the ~15-frame "Hands-On Example:
    Predicting House Price" walkthrough with 3 live code blocks) — an optional follow-up if a
    full resync is ever wanted.
  - Session 6 (`ml_concepts_short.tex`/`ml_concepts.tex`): this pass added a TikZ bowl-curve
    diagram to "How to Find Best Fit: Gradient Descent" (precedent for the new Task 5a below),
    5 "Intuition" callouts on the driest early math-framework slides, a symbol-by-symbol
    breakdown of the `w = (X^TX)^{-1}X^TY` closed-form formula, and fixed the "Detection" frame's
    vertical overflow (two stacked images → side-by-side). A follow-up full `/upgrade-deck` pass
    then found: a genuine bug in "Bias vs Variations" (the same reference citation line rendered
    twice in one frame, sandwiched around a block of commented-out content); a real redundancy
    ("Learning Method Bias"/"Learning Method Variance" re-defined the same two terms already
    covered by the earlier "Bias"/"Variance" frames AND the StatQuest bv1-bv8 intuitive
    walkthrough — a third redundant pass, removed); a bare formula with no interpretation
    ("Types of Errors" was missing a "Noise Error" bullet despite `noise(X)` appearing in its own
    formula, and had zero plain-language explanation); and 2 sections (Cross Validation,
    Generalization) missing their closing Quick Check quiz (added, matching the other 5
    sections). All fixes mirrored into both files.
- **New `/upgrade-deck` rule -- Task 5a: TikZ Diagram Opportunities (Aug 2026)**, added to
  `upgrade-deck.md` (`~/.claude/commands/`, mirrored to `Code/claudecode/dot_claude/commands/`)
  right after Task 5, plus a TikZ row in the Step 3 package audit table: for slides that describe
  a process/trajectory/relationship in words/equations only (and have no existing image), add a
  simple TikZ diagram using the repo's established two-column `adjustbox`+`minipage` convention
  (explanatory content left, ~0.55-0.56 `\linewidth`; diagram right, ~0.4 `\linewidth`). Cites the
  Session 6 Gradient Descent bowl-curve diagram above as the precedent. Explicitly guards against
  forcing a diagram onto every slide -- most won't qualify.
- **Quick Check quiz format changed: `\pause` overlay -> two separate frames (Aug 2026)**.
  Trigger: presenting MLCoEP Session 5 live, the question/answer reveal (a `\pause` overlay
  splitting one frame into two PDF pages) was easy to click straight past without noticing the
  answer had appeared -- overlay reveals don't read as a distinct "next slide" during a live
  click-through. Fixed by converting every live (non-commented) Quick Check quiz to two
  independent frames -- a question-only frame, then a same-titled answer frame with an
  `\end{frame}` + separator comment + `\begin{frame}` in between instead of `\pause` -- content
  and wording unchanged, purely a frame-structure split. Applied to all 27 live quizzes across
  the 15 topic files any MLCoEP session currently pulls in via `\input`: `ml_intro_short.tex`
  (Session 5, 6 quizzes), `ml_concepts_short.tex` (Session 6, 5 quizzes), `python_overview.tex`
  (Session 2, 2 live + 1 already-commented quiz left untouched), `python_intro_pandas.tex`
  (Session 4, 3), and one quiz each in `ml_eda_endtoend_churn.tex` (Session 3),
  `ml_datapreparation_sklearn.tex`/`ml_evaluation_sklearn.tex` (Session 7),
  `ml_logisticregression.tex` (Session 9), `ml_decisiontree_short.tex` (Session 10), `ml_svm.tex`
  (Session 12), `ml_naivebayes_short.tex` (Session 13), `ml_knn.tex`/`ml_knn_sklearn.tex`
  (Session 14), `ml_kmeans.tex` (Session 15), `ml_predictive_analytics.tex` (Session 18). 12 of
  these 15 files are shared with non-MLCoEP decks (the standalone ML course's 10 seminars,
  `seminar_python_overview_content.tex`, `seminar_artificialintelligencemachinelearning_content.tex`,
  `seminar_deeplearning_foundations_content.tex` via `workshop_deeplearning_content.tex`) --
  per explicit instruction, the change was applied to the shared files directly rather than
  forked into MLCoEP-only sibling copies, so those quizzes now render the same two-frame way
  everywhere they're used, not just in MLCoEP. `upgrade-deck.md`'s own Task 5 Quick Check
  boilerplate (`~/.claude/commands/`, mirrored to `Code/claudecode/dot_claude/commands/`) was
  updated to the two-frame pattern too, so future quiz slides (any deck, not just MLCoEP) are
  generated in the new format by default. **Verification status**: Sessions 5, 6, 7, 9
  recompiled clean with unchanged page counts (confirms nothing was lost/duplicated by the
  split) and spot-checked via extracted PDF text. The other 9 MLCoEP session drivers and 12
  non-MLCoEP driver decks pulling in the 15 edited files were NOT recompiled/verified after this
  change (a batch compile was started, then stopped before completion at the author's request to
  close the session) -- recompiling and visually verifying those is still open, next time any of
  them is touched.
- **Session 6 recall/precision/F1 expansion (Aug 2026)**: triggered by teaching the session live
  and finding recall effectively missing -- it *was* defined, but titled "Sensitivity (Recall or
  True positive rate)" and positioned immediately before Specificity, so it read as half of the
  sensitivity/specificity pair and nothing followed Precision. Checked the sibling first per the
  standing rule: the reusable commented material (Boy-Who-Cried-Wolf Recall/Precision frames, the
  "Precision addresses FP / Recall addresses FN" frame) was commented in *both* files, so only one
  frame could be uncommented and the rest was authored. Net +5 frames, 2 edited, mirrored into
  `ml_concepts_short.tex` and `ml_concepts.tex`: retitled the sensitivity frame to lead with
  "Recall"; added "Precision: Intuition" and "Recall: Intuition" built on one fishing analogy
  (precision = of the fish you kept, what fraction were rainbow trout, denominator is what you
  *claimed*; recall = of all the fish in the lake, what fraction reached your net, denominator is
  what *exists*); a dedicated `REC = TP/(TP+FN)` frame after Precision that says outright it is the
  same quantity as sensitivity; a "One Net, Two Questions" contrast table; and two frames on why F1
  uses the harmonic mean (worked P/R table where arithmetic scores 0.5/0.5, 0.9/0.1 and 1.0/0.0 all
  at 0.50 while harmonic gives 0.50/0.18/0.00, plus the rates-with-different-denominators argument
  and the flag-everything-positive gaming case). The Boy-Who-Cried-Wolf frames were deliberately
  left commented -- two competing analogies for the same pair in one deck. The uncommented
  "Which One Should You Optimize?" frame carried a factual slip, fixed while bringing it live: it
  justified Recall with "the test should not wrongly say that you have cancer", which describes a
  false *positive*.
- **MLCoEP CheatSheets restored to 3 columns (Aug 2026)**: all 19
  `Main_Seminar_MLCoEP_Session_<N>_*_CheatSheet.tex` drivers were on `multicols{2}`, a regression
  introduced by the Aug 2026 19-session restructuring -- the all-in-one
  `Main_Course_ML_CoEP_CheatSheet.tex` was already `multicols{3}`, and Seminars take 3 per the
  convention above. Switching them exposed overflow that 2 columns had been hiding (19 overfull
  boxes across 7 sessions; the other 12 were clean immediately). Root cause was
  `breakatwhitespace=true` in `template_cheatsheet.tex`'s `lstdefinestyle`: long URLs and dotted
  module paths have no whitespace to break at, so they ran off the page rather than wrapping.
  Fixed with a **driver-local** override in each of the 19 drivers, right after
  `\graphicspath` -- `\lstset{basicstyle=\scriptsize\ttfamily, breakatwhitespace=false}` +
  `\sloppy` -- deliberately not put in the shared template, which is used by every CheatSheet in
  the repo including 2-column Workshops. Remaining fixes were content-level: 5 tables wrapped in
  `\adjustbox{max width=\linewidth}` (`ml_concepts{,_short}`, `ml_eda_intro` ×2,
  `data_preparation_short`) and 3 unbreakable identifiers given `\allowbreak` after each dot
  (`ml_course_demo_regression_housing`, `data_preparation{,_short}`, plus
  `ml_linearregression`'s coefficient output, which was also wrongly wrapped in `$...$` math
  mode). End state: 19/19 compile clean, one 5pt overfull left in Session 13 and 5
  `Overfull \vbox` from multicol column balancing at page breaks (absorbed by the 2cm bottom
  margin). Verified downstream too: MLCoEP Presentations 3/6/8/19, `Main_Seminar_ML_Intro_*`,
  `Main_Seminar_ML_Regression_*` and `Main_Workshop_Data_Analytics_*` all recompile clean, since
  the edited topic files are shared with them.
- **Methodology note (Aug 2026)**: when changing CheatSheet column count, baseline the *old*
  column count first. Recompiling the affected drivers at 2 columns and diffing overfull-box
  counts is what separated 3-column-induced overflow (6 of 7 sessions, clean at 2 columns) from
  pre-existing breakage (Session 7's 193pt URL, already broken). Also note the compile log alone
  is not sufficient: floats dropped inside `multicols` and wrapped minipages produce no error,
  so render suspect pages to images (`pdftoppm -png -r 100 -f N -l N`) and look.
- **Session 7 `/upgrade-deck` pass (Aug 2026)**, run via `Main_Seminar_MLCoEP_Session_7_Sklearn_Workflow_*`
  over its 3 topic files (`ml_intro_sklearn`, `ml_datapreparation_sklearn`, `ml_evaluation_sklearn`).
  None has a `_short` sibling, but all 3 are shared: `ml_intro_sklearn` with
  `seminar_ml_intro_content.tex`, the other two with `seminar_ml_dataprep_content.tex`, so edits
  were made in place and those decks recompiled too. 50 -> 54 live frames.
  - **Every code block was executed** against `pandas 2.2.3` / `scikit-learn 1.7.2` before the
    outputs were touched, and this is what the pass turned on. The stored outputs were stale
    relative to the code: the `KFold`s already said `shuffle=True`, but the printed numbers were
    from an unshuffled split. Confirmed exactly: unshuffled Boston gives `R^2: 0.203 (0.595)`
    (the number that was on the slide, worst fold -1.006), shuffled gives `0.718 (0.099)`. All 8
    outputs updated to verified values, and the two $R^2$ frames' prose rewritten: it had claimed
    the large standard deviation was the negative-$R^2$-on-a-bad-fold effect, which is no longer
    true of the shuffled numbers. **Lesson: when a deck shows both code and its output, re-run the
    code, do not trust the pasted result.**
  - Other Task 1 fixes: `delim_whitespace=True` -> `sep='\s+'` (deprecated in pandas 2.2);
    `LogisticRegression()` -> `max_iter=1000` (5 sites, silent `ConvergenceWarning` on unscaled
    Pima); `classification_report` output updated from the pre-0.20 `avg / total` format to
    `accuracy`/`macro avg`/`weighted avg`; a stray Unicode right-quote in `model’s`; "linear
    discriminate analysis" -> "discriminant"; "good prediction and recall" -> "precision";
    Python-3.5-era install slide modernised (its `conda install` line omitted `pandas`/`seaborn`
    that the very next slide imports).
  - Task 2 removed one untitled frame that duplicated the "Estimator" frame; the second "Read Data"
    frame was **kept** (both topic files must stand alone in `seminar_ml_dataprep_content.tex`) but
    retitled "Read Data: the Same Pima Dataset" so the repetition reads as deliberate.
  - Task 4 added a `Pipeline` frame: the existing Quick Check answer told students to fit only on
    training data, but nothing in the session showed the mechanism that enforces it.
  - Task 5a added 2 TikZ diagrams in the two-column `adjustbox`+`minipage` convention: an ROC
    curve with shaded area (AUC frame) and a 2x2 confusion matrix with the diagonal shaded.
    **The ROC frame overflowed 12.18pt and took 3 passes to fix** -- trimming the `Intuition`
    block alone did nothing, because the *left* column was the tall one; only merging bullets
    cleared it. Caught by rendering the page, since Beamer does not error on vertical overflow.
  - Task 6 added 2 quiz pairs (Estimator API; MAE vs MSE outlier sensitivity), matching the
    two-separate-frames format.
  - **Datasets now local**: `Code/mlcoep/datasets/session07_sklearn/` holds
    `pima-indians-diabetes.data.csv` (768 rows) and `housing.data` (506 rows) + a README, mirroring
    the Session 4 precedent. Verified the local copies reproduce the slide numbers exactly. The
    slides still load from the `jbrownlee/Datasets` URLs; these are an offline fallback.
  - End state: Session 7 Presentation 59 pages, CheatSheet 5 pages, both compile clean; the 2
    remaining presentation `Overfull \hbox` are the pre-existing title/footline ones, and the
    CheatSheet has one 16.6pt overfull left. `Main_Seminar_ML_{Intro,DataPrep}_*` also recompile
    clean. **`Main_Seminar_ML_Intro_Presentation` reports ~41k overfull boxes -- pre-existing and
    unrelated to this pass** (a repeating structure in that 161-page deck), not investigated.
- **Session 7 follow-up: `ml_intro_sklearn.tex` individual-algorithm examples revived (Aug 2026)**,
  triggered by teaching the session live and finding the jump from the abstract `Estimator`
  pseudocode frame straight to the `Pipeline` frame too big a leap, with ~200 lines of concrete
  regression/classification/clustering examples sitting commented out in between. Revived and
  modernized (`sklearn.cross_validation` -> `model_selection`, deprecated `plt.cm.get_cmap` ->
  `plt.get_cmap`, dropped the removed `LinearRegression(normalize=True)` kwarg) rather than
  deleted outright: Iris dataset loading (2 frames), a "Sklearn: Algorithms" section divider, a
  synthetic-data Linear Regression example + its 2-feature exercise variant, a K-Nearest-Neighbor
  classifier example, PCA, and K-means clustering (feeding PCA's reduced `X` into the scatter plot)
  -- 8 revived frames total, all on Iris/synthetic data, kept deliberately separate from Data
  Prep/Evaluation's own Pima Indians + Boston Housing case study rather than forced onto one
  dataset. Left commented, not revived: several near-duplicate/dead stubs superseded by the real
  Pima/Boston code in `ml_datapreparation_sklearn.tex`/`ml_evaluation_sklearn.tex` (an SVM stub, a
  generic `ClassifierEstimator()`/`RegressionEstimator()` pseudo-code pair, an old KNN
  `predict_proba` stub, and an SVM classifier frame that depended on an undefined `X_train`), the
  Cheat Sheet image frames, and the Digits/OCR mini-project (a separate tangent, not asked for).
  Added one new frame, "Logistic Regression: Estimator API in Action" (concrete `fit`/`predict` on
  Iris with a real train/test split), placed right before `Pipeline` so the arc reads Estimator API
  -> concrete Logistic Regression example -> Pipeline, per explicit request. Every revived/new code
  snippet was executed against the same `sklearn 1.7.2`/`numpy 2.2.6`/`matplotlib 3.10.8` (`genai`
  env) before being trusted, same discipline as the Session 7 sklearn-workflow pass above. Session
  7 Presentation 59 -> 68 pages, CheatSheet 5 -> 7 pages, both recompile clean (same 2 pre-existing
  footline overfulls in the Presentation; same pre-existing 16.6pt overfull in
  `ml_evaluation_sklearn.tex`'s CheatSheet section, confirmed by log line number this predates the
  change). All new/edited pages rendered to images and visually confirmed no silent overflow.
  `Main_Seminar_ML_Intro_{Presentation,CheatSheet}.tex` (the other consumer of
  `ml_intro_sklearn.tex`) also recompiled without new errors (171-page Presentation still carries
  the pre-existing ~41k-overfull-box issue noted above, unrelated).
- **Session 8 `/upgrade-deck` pass (Aug 2026)**, run via `Main_Seminar_MLCoEP_Session_8_Linear_Regression_*`
  over its single topic file `ml_linearregression.tex`. No `_short` sibling; shared with
  `seminar_ml_regression_content.tex` (`Main_Seminar_ML_Regression_*`). 53 -> 60 live frames,
  65 pages, **zero overfull boxes** in the Presentation.
  - **Count live frames, always**: the file is 1511 lines with 110 raw `\begin{frame}` but only 53
    live: more than half is commented-out author material. Same trap as the Jul 2026 methodology
    note above.
  - **The committed PDF ended in a "Temporary page! LaTeX was unable to guess the total number of
    frames" page** -- it had been built without a converged final pass. Gone after recompiling.
    Worth checking for on any deck whose PDF predates its last edit.
  - Genuine technical errors fixed: the $(X^TX)^{-1}$ slide claimed the matrix is singular "if any
    two **rows** are same" (wrong -- duplicate samples are harmless, linearly dependent **columns**
    are the problem); `x_{p,1}` -> `x_{p,i}` plus a missing intercept; the ISL data URL
    `www-bcf.usc.edu/~gareth/...` is dead (USC retired that host) -> `statlearning.com`;
    `+ -0.0010` -> subtraction; intercept quoted as both `-0.818` and `-0.8188`; "OLS minimizes RMS
    error" -> sum of squared residuals; `Found'm'`, `$y_i$to`, "slopes means", "adverting".
  - **Two defects only visible by rendering, not from the log** (same lesson as Session 7's ROC
    frame): the "Coefficients" frame wrapped prose in math mode, so it rendered as run-together
    italic **"TVModel"/"RadioModel"/"NewspaperModel"**, and its raw `[[0.04753664]][7.03259355]`
    output never said which number was the slope; the "Advertising Dataset" frame used `eqnarray*`
    with a single `&`, putting the relation in the alignment column and leaving a wide gap before
    `≈ β0 + β1 × TV`. Fixed to labelled `\texttt{}` values and `align*`.
  - **Content gap**: the frame *titled* "$R^2$ Statistics" only set up $SST = SSR + SSE$ and stopped;
    a later frame then used $R^2$ as if defined. The definition was sitting commented out in the
    file. Added a frame defining $R^2 = SSR/SST = 1 - SSE/SST$ and evaluating $90/120 = 0.75$ on the
    tip-example numbers already on screen.
  - **Nothing was deleted.** Three sets of repeated frames (`lab2` twice, `corrmat` twice, the
    rotated-line sequence) look redundant but are deliberate question-answer / animation builds.
    The real problem was navigational: **7 consecutive frames titled "Optimization"** and 3 titled
    "Evaluation", all now given distinct titles.
  - Added: a TikZ SST/SSR/SSE decomposition diagram (all three distances measured at the same data
    point -- the first draft drew SSR at an x where it did not touch the regression line, caught by
    rendering), 3 Intuition callouts, and 3 Quick Check pairs (the deck had **zero** quizzes).
    60 frames slightly exceeds the ~50-55/hr target; accepted because quiz frames click through fast.
  - **Open items from this pass** (compiled Session 8 only, at the author's request):
    `Main_Seminar_ML_Regression_*` carries these edits but was **never recompiled**; and the Session 8
    CheatSheet has one unresolved `Overfull \hbox` (9.86pt, log lines 322-323) that was never traced
    to a frame, so it is unknown whether it predates this pass. **Note: this session is now Session 9**,
    see the renumbering note below (Aug 2026) -- all Session-8-specific references above (filenames,
    log line numbers) predate that shift and describe what was then Session 8.
- **Session 8 inserted as a syllabus gap-fill session before the Units 1-3 MCQ test; old Sessions 8-19
  renumbered to 9-20 (Aug 2026).** Triggered by comparing the official BTech Mechanical AIML-ML
  syllabus (4 units, 30 hrs) against Sessions 1-7 as actually taught and finding real, verified gaps
  within Units 1-3: Unit 1 (History of AI's classical era, the Reasoning/Knowledge-Representation/
  Planning/Perception/Motion-\&-Manipulation capability framework, Approaches to AI --
  Cybernetics/Symbolic/Sub-symbolic/Statistical, and "Need of AI in Mechanical Engineering" -- the
  last of these already existed as `ml_mech_short.tex` but sitting in Session 19, not up front);
  Unit 2 (Hyperparameter Tuning was live nowhere -- the only mention was inside an entirely
  commented-out frame in `ml_intro_short.tex`; "Ranking" as a 4th problem-identification type;
  an explicit "Model Selection" step); Unit 3 (Feature extraction's "Statistical features", feature
  selection's "Ranking", and the wrapper search strategies -- Exhaustive, Best-first, Greedy
  forward/backward). Unit 4 needed no new session -- it already maps onto the existing algorithm
  sessions (old Sessions 8-16). Decision-tree entropy/info-gain and PCA, though nominally Unit-3
  topics, were deliberately excluded from this pass since they land in their own sessions later
  (old Sessions 10 and 16) -- confirmed by the author, not assumed.
  - **Scope was narrowed twice by the author during planning, both kept**: (1) Unit 1's gaps were
    dropped entirely from this session -- Session 1 was already delivered in class, and patching it
    now wouldn't reach students before the test, so `ai_intro_tech.tex` was left untouched. (2) Of
    the Unit 2/3 gaps, only the ones with no natural existing home became new content; the rest were
    added as live frames directly into the already-taught reference decks they thematically belong
    to, on the reasoning that keeping those decks complete matters even if not literally re-lectured:
    Ranking + Model Selection into `ml_intro_short.tex` (mirrored into its full sibling `ml_intro.tex`
    per the standing sibling-sync rule), Hyperparameter Tuning into `ml_concepts_short.tex` (mirrored
    into `ml_concepts.tex`). Only the Unit 3 feature-extraction/-selection material, which fit neither
    file, became the new `ml_featureselection.tex` that Session 8 actually delivers.
  - **Content added**: `ml_intro_short.tex`/`ml_intro.tex` gained "Ranking: Overview" (framed as a 4th
    problem type alongside the file's existing Classification/Regression/Clustering/Dimensional-
    Reduction cluster) and "Model Selection". `ml_concepts_short.tex`/`ml_concepts.tex` gained a
    revived (previously commented-out) "Practical Tip" frame plus "Hyperparameter Tuning", "...: Grid
    Search", and "...: Randomized Search" (a verified `GridSearchCV`/`RandomizedSearchCV` run on the
    Pima dataset, KNN's $k$, landing on $k=11$/0.749 and $k=13$/0.755 respectively) and a Quick Check
    pair, inserted right where the revived Practical Tip frame already namedropped "hyper parameters"
    in its pre-existing text. `ml_featureselection.tex` (new, 17 frames) builds one narrative: raw
    signal $\to$ statistical features (a jargon-free sensor-reading example, per explicit instruction
    to ease off Mechanical-Engineering-specific framing since wrapper methods were new to the author
    too) $\to$ filter/ranking selection $\to$ all 4 wrapper strategies (Exhaustive, Greedy Forward,
    Greedy Backward, Best-First), all grounded in one real, executed run on the familiar 8-feature
    Pima dataset (small enough that exhaustive search over all 255 non-empty subsets is genuinely
    runnable, not just asserted) -- exhaustive search proved dropping `skin` costs zero accuracy,
    and greedy backward elimination's first move independently rediscovers exactly that same fact.
    Every code snippet across all four files was executed against the `genai` conda env
    (`sklearn`/`numpy`/`matplotlib` as in the Session 7 pass) before being trusted.
  - **New template gotcha found**: `template_presentation.tex`'s `lstset` carries `belowskip=-15pt`
    (a deliberate negative skip, presumably tuned for the common case of code being the last thing in
    a frame). Any body text placed directly after `\end{lstlisting}` -- regardless of how short the
    code block or the following sentence is -- gets pulled up into the code box, and the resulting
    overlap is **not reliably reported** as `Overfull \vbox` (several instances here produced no log
    warning at all and were only caught by rendering pages to images and reading the extracted text).
    A blank line before the trailing text does not help. `\vspace{15pt}` immediately after
    `\end{lstlisting}` fixes it when the frame has room to spare, but in a fuller frame (the
    Grid/Randomized Search ones) that same 15pt pushed the trailing sentence down far enough to
    interleave character-by-character with the footer/page-number instead -- also invisible in the
    log, only caught by extracting page text and inspecting the last lines before the footer. The
    robust fix used throughout this pass: fold the trailing interpretive sentence into the
    `lstlisting` itself as a final `#` comment line, so nothing sits outside the box at all, rather
    than tuning a `\vspace` value per frame.
  - **Revised after author review (Aug 2026, same day)**: three corrections. (1) Session 8 must not
    reference other session numbers at all -- "all sessions are sort of independent" -- so every
    "(Session 3)"/"(Session 6)"/"(Session 7)" mention and every "same load as Session N" code comment
    was removed; the Ranking worked example now re-imports the Pima CSV from scratch and shows
    `df.head()` before any analysis, rather than assuming an earlier session's load happened. (2) Most
    `\item` bullets had drifted into flowing paragraph sentences, not this repo's established terse
    list style -- rewritten shorter throughout both files. (3) Hyperparameter Tuning, Grid Search, and
    Randomized Search were moved back out of `ml_concepts_short.tex`/`ml_concepts.tex` entirely (the
    revived "Practical Tip" frame stayed) into a new `ml_modeltuning.tex`, also `\input` by Session 8
    -- reasoning: Session 8 needed to be self-contained and closer to a normal session's size (17 live
    frames read as noticeably thin next to other ~50-55-frame sessions), and splitting new content
    across an actively-taught Session 8 deck and a passively-updated Session 6 deck was an awkward
    halfway house once independence was the explicit goal. Ranking (the 4th ML problem type) and
    Model Selection stayed in `ml_intro_short.tex`/`ml_intro.tex` -- unlike the hyperparameter content
    they never carried session references, and they fit that file's existing Classification/
    Regression/Clustering cluster too well to relocate. Also added, per author request: a correlation-
    with-target bar chart (`images/pima_corr_barplot.png`, generated and verified against real Pima
    data, not a mockup) as a visual complement to the F-score ranking table, and an "F-score (a Sharper
    Test)" framing to explicitly connect the two views. End state: `ml_featureselection.tex` 19 frames,
    `ml_modeltuning.tex` 8 frames, 27 total (up from 17) -- both recompiled clean (Presentation:
    just the 2 pre-existing footline overfulls; CheatSheet: zero warnings). Session 6 recompiled after
    the reversion, 89 pages (down from 94, exactly the 5 removed frames), exit clean.
  - **Hands-on revision exercise added (Aug 2026, same day)**, per author request for something
    "before starting in depth understanding of ML algorithms" -- `ml_pipelineexercise.tex` (new, 18
    frames), an 11-step, fully-executed, fully-verified pandas+sklearn pipeline (Load -> Inspect ->
    Split -> Scale -> Select -> Train -> Evaluate -> Compare) on the sklearn-builtin Breast Cancer
    Wisconsin dataset (30 features -- deliberately more than Pima's 8, an authentic stress test for
    the session's feature-selection content). Filter selection (top 10 by F-score) actually *lost*
    accuracy versus all 30 features (0.951 vs 0.986 test, 0.951 vs 0.975 CV) -- a genuine, unstaged
    contrast with Pima's zero-cost `skin` drop, since not every dataset offers a free lunch. Greedy
    forward selection down to just 6 features then beat all 30 on cross validation (0.979 vs 0.975),
    a real result that lands the wrapper-vs-filter argument concretely rather than asserting it.
    Closes with a Quick Check on cross-validation vs. held-out-test-set leakage (the greedy loop
    scores candidates via CV over the full feature matrix `X`, not just `X_train` -- explored whether
    that leaks, concluding it doesn't because CV still holds out a fold per score, and the final
    `X_test` split from Step 3 was never touched by it either), a "Try It Yourself" prompt list, and a
    recap framing the 8-step shape as reusable across every future algorithm session. One CheatSheet-
    only fix needed: the Step 10 comparison table overflowed the narrow 3-column layout, same
    `\adjustbox{max width=\linewidth}` fix as elsewhere in this pass. `seminar_mlcoep_session_8_
    content.tex` now `\input`s all three files in sequence: `ml_featureselection`, `ml_modeltuning`,
    `ml_pipelineexercise`. **End state: 45 content frames** (19+8+18, up from the original 17),
    51 total pages with title/outline/about\_me/thanks boilerplate -- close to the ~50-55 norm without
    padding for its own sake. Both Presentation and CheatSheet recompiled clean.
  - **`ml_pipelineexercise.tex` dataset swapped Breast Cancer -> Steel Plates Faults (Aug 2026,
    same day, author request)**, per explicit "does not need to be very mechanical, but a medical
    dataset isn't ideal either" steer plus a follow-up ask for a genuinely Mechanical-Engineering
    dataset. Now UCI Steel Plates Faults (1941 plates, 27 geometric/luminosity features, target =
    `K_Scatch`, a scratch-type defect, 391 positive/1550 negative) -- fetched via a direct UCI URL
    (`sep=r'\s+'`, no header, matching this course's existing Boston Housing pattern) rather than
    adding a new `ucimlrepo` dependency. **Verification caught a real dead end before it shipped**:
    `Bumps` was tried as the target first and rejected -- greedy forward selection degenerated,
    every step plateauing at exactly the 79.3% majority-class baseline (confusion matrix
    `[[385,0],[101,0]]`, the model just always predicted "no fault"). `K_Scatch` gives a real,
    cleanly-separable problem instead: all 27 features 0.975 CV, filter top-10 0.949 CV, wrapper
    greedy-6 0.963 CV -- same "filter costs accuracy, wrapper recovers most of it" shape as the
    Breast Cancer version, just not quite exceeding the full-feature model this time (unlike Breast
    Cancer's 0.979-beats-0.975) -- reported honestly rather than cherry-picking a target that
    overclaims. `Try It Yourself` now points students at the rejected `Bumps` case as a discussion
    exercise instead of hiding it. Local offline copy saved to
    `Code/mlcoep/datasets/session08_featureselection/Faults.NNA` + README (same pattern as
    `session04_pandas/`, `session07_sklearn/`), for the author to distribute to the class in
    advance -- **Session 8 only, `ml_datapreparation_sklearn.tex`/`ml_evaluation_sklearn.tex`
    (Session 7) still use Pima/Housing, untouched.** One CheatSheet-only regression caught and
    fixed: the Step 10 comparison table lost its `\adjustbox{max width=\linewidth}` wrap during the
    rewrite (copy-paste miss), overflowing the narrow 3-column layout again -- same fix reapplied.
    Both Presentation and CheatSheet recompiled clean afterward; frame count unchanged (45).
  - **Mechanical renumbering**: for K = 19 down to 8 (descending, to avoid overwrite collisions),
    renamed `Main_Seminar_MLCoEP_Session_K_<ShortName>_{Presentation,CheatSheet}.{tex,pdf}` to
    `Session_{K+1}_...`, renamed `seminar_mlcoep_session_K_content.tex` to `_{K+1}_content.tex`
    updating its `% Session K:` comment and `\section[...]{Session K: ...}` label, and updated each
    renamed driver's `\input{seminar_mlcoep_session_K_content}` line -- the CheatSheet driver also
    carries a second, independent hardcoded `Session K:` string in its title block that the
    Presentation driver does not, easy to miss. `course_mlcoep_content.tex`, `make_all_sessions.bat`,
    `make_all_cheatsheets.bat`, `COURSES.md`, and `TODO.md` updated to the new 20-session numbering;
    `TODO.md`'s dated Aug 2026 recompile-backlog narrative was left describing the old numbering it
    actually verified, with a note that it predates this shift, rather than rewritten to a numbering
    that didn't exist yet at the time. Topic files themselves (`ml_linearregression.tex` etc.) were
    untouched -- only the session-number wrapper around them moved.

### Adding a new topic
1. Create `LaTeX/<domain>_<topic>.tex` with Beamer frames
2. `\input{<domain>_<topic>}` inside the relevant `seminar_*_content.tex`
3. Place supporting images in `LaTeX/images/` (5000+ images already there, mostly PDFs)

### Frame boilerplate
```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{frame}[fragile]\frametitle{Slide Title}
\begin{itemize}
\item Point one
\end{itemize}
\end{frame}
```

For section dividers:
```latex
\begin{frame}[fragile]\frametitle{}
\begin{center}
{\Large Section Heading}
\end{center}
\end{frame}
```

## Code Directory

Each subdirectory under `Code/` corresponds to a library or topic. No single build or test command — run scripts individually per subdirectory.

### Environment setup (conda-based)
Every major subdirectory has an `environment.yml`. The standard setup flow is:
```bash
conda env create -f Code/<subdir>/environment.yml
conda activate <env-name>
```
Do not create `venv/` or `.venv/` folders — use conda environments only.

### Key subdirectory map

| Category | Directories |
|----------|-------------|
| GenAI / Agents | `langchain/`, `langgraph/`, `llamaindex/`, `crewai/`, `agents/`, `agno/`, `google-adk/` |
| RAG Applications | `chatbot-faqs/`, `chatbot-multimodal/`, `omni-rag/`, `parsing/`, `graphrag/` |
| LLM Fine-tuning | `fine-tuning/`, `ludwig/`, `gemma/` |
| Document Parsing | `docling/`, `opendataloader/` |
| Deep Learning | `keras/`, `dl_tf2/`, `pytorch/` |
| Classical ML | `ml/`, `math/`, `python/` |
| NLP | `nlp/`, `dnlp/`, `spacy/` |
| GNN | `gnn/pyg/`, `gnn/gnn-project-deepfindr/`, `gnn/molecule-deepfindr/`, `gnn/odsc2021-sujitpal/` |
| Indic Language | `mahamarathi/`, `sarvam/`, `orgpedia/`, `pritamMarathi/` |
| Research Refs | `txt2cad/`, `txt2sql/` (docs only, no runnable code) |
| Other | `amd/` (AMD Academy: agents/fine-tuning/vLLM serving course materials), `chromeext/` (a small Chrome extension side-project, not AI/ML) |

### Code/.gitignore
A repo-wide `Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, model weights (`*.bin`, `*.pt`, `*.safetensors`).

### Notable sub-projects with their own config
- `Code/claudecode/MyWorkshop/` and `Code/claudecode/CadCamWorkshop/` — each has its own `CLAUDE.md` (no `CLAUDE.md` directly under `Code/claudecode/` itself)
- `Code/langgraph/open_deep_research-langcahin-ai/` — has its own `CLAUDE.md` and `README.md`
- `Code/crewai/researcher/` — uses `pyproject.toml` + `uv.lock` (modern uv workflow)

### Security note
`Code/google-adk/my_agent/.env` is gitignored but contains a real `GOOGLE_API_KEY` on disk — rotate it in Google Cloud Console.

## Test Suite

All Python-script directories have a `test_*.py` file runnable with `pytest` in the `genai` conda environment.

### Running tests

Run a single suite:
```bash
conda activate genai
cd Code/<subdir>
python -m pytest test_*.py -v
```

Run all suites together (from repo root):
```bash
conda run -n genai python -m pytest \
  Code/graphrag/test_graphrag.py \
  Code/parsing/test_parsing.py \
  Code/agno/test_agno.py \
  Code/google-adk/test_tools.py \
  Code/chatbot-faqs/test_chatbot_faqs.py \
  Code/chatbot-multimodal/test_models.py \
  Code/omni-rag/test_omnirag.py \
  -v
```

### Test files per directory

| Directory | Test file | Tests | What's covered |
|-----------|-----------|-------|----------------|
| `chatbot-faqs/` | `test_chatbot_faqs.py` | 14 | CSV loading, similarity threshold, cosine similarity logic |
| `chatbot-multimodal/` | `test_models.py` | 19 | Pydantic chunk models, DoclingParser device selection, null-safe heading join |
| `omni-rag/` | `test_omnirag.py` | 9 | Context list-join fix, OmniIngestor structure (mocked), ragas/datasets imports |
| `parsing/` | `test_parsing.py` | 12 | GroqResumeParser: empty-key validation, default model, mock API call |
| `graphrag/` | `test_graphrag.py` | 9 | `distance()` boundary conditions, networkx/pandas integration |
| `google-adk/` | `test_tools.py` | 10 | Tool functions (web_search, get_stock_price, etc.) with mocked yfinance |
| `agno/` | `test_agno.py` | 7 | agno package imports, syntax validation of all .py files |

### Test design notes
- No real API calls — all LLM/embedding clients are mocked with `unittest.mock`.
- No model downloads — `transformers` model-loading calls are patched at the function level.
- The `google-adk` tests mock the `adk` package (not installed on all machines).
- The omni-rag `TestOmniIngestorStructure` tests skip gracefully if a `datasets` circular import occurs when running in a combined pytest session (they pass in isolation).
- `ragas` and `google-adk` packages were added to the `genai` env during the April 2026 upgrade pass.

### Known environment notes
- `ragas 0.4.3` upgraded `openai` from 1.x → 2.x — verify `langchain-openai` compatibility if issues arise.
- A broken system-Python `faiss` install exists at `C:\Users\yoges\AppData\Roaming\Python\Python310\site-packages\faiss\` and conflicts if imported outside the conda env.
- `opendataloader-pdf` and `langchain-opendataloader-pdf` are installed in the `genai` env (added May 2026). The library wraps a Java JAR — **Java 11+ must be on PATH** before any tutorial runs. Install via `conda install -n genai -c conda-forge openjdk=11`. Tutorial 09 (OCR) additionally requires the hybrid backend started in a separate terminal: `opendataloader-pdf-hybrid --port 5002 --force-ocr`.

## Memory
Do not store, write, or update any memory files in the global `~/.claude/projects/` directory unless the user explicitly confirms or allows it in the current conversation.

## Git
Do not run any git commands. The user manages all git operations externally.
