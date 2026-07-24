# TODO — AI-ML for Mechanical Engineers (CoEP)

**Course**: AI-ML for Mechanical Engineers (CoEP Final Year)
**Status**: Course in progress (Jul 16, 2026 - Nov 30, 2026)
**Last updated**: Jul 24, 2026

**Note on staleness**: sections 3-6 below were written before the Aug 2026 restructuring
of `course_mlcoep_content.tex` (Math Foundations sessions removed, Python consolidated
to one session, two new Mechanical-Engineering-specific sessions added). The course is
now 21 sessions, not 24, and old mappings like "A2: Math + Stats" or "Sessions 4-6:
Math & Stats" no longer correspond to real sessions. Re-check assignment-to-session
mapping and the weekly calendar against the current 21-session list (Section 1.1)
before treating dates/topics below as final.

---

## 1. LaTeX content (this repo)

### 1.1 Remaining per-session PDF generation

Process (established, repeat per session): in `course_mlcoep_content.tex`, uncomment
ONLY the target session's `\input{...}` line(s), `texify -cp` both
`Main_Course_ML_CoEP_Presentation.tex` and `Main_Course_ML_CoEP_CheatSheet.tex`, rename
outputs to `Course_ML_CoEP_NN_Presentation.pdf` / `Course_ML_CoEP_NN_CheatSheet.pdf`,
re-comment, move to the next session. Do a few at a time, not all in one go.

| # | Session | Status |
|---|---------|--------|
| 01 | AI Overview | pending - `ai_intro_tech.tex` still commented/unused, no content decision made |
| 02 | Python Overview | **DONE** - complete package, restructured into proper subsections (Setup incl. Anaconda/IDEs/venv, Language Basics & Syntax, Data Types, Control Flow/Functions/Classes, Advanced Constructs & Libraries, Applications, Practice ending on Zen of Python); all `lstlisting` blocks consolidated per the `/upgrade-deck` code-block-placement rule; 70pg/6pg. Also spun off as its own seminar (`seminar_python_overview_content.tex` -> `Main_Seminar_Python_Overview_*`, 72pg/6pg) |
| 03 | Pandas & Data Manipulation | **DONE** - 42pg/4pg |
| 04 | Data Preparation & Feature Engineering | **DONE** - 20pg/2pg |
| 05 | EDA & Feature Engineering Demo (Churn) | **DONE** - 47pg/5pg |
| 06 | ML Concepts & Scikit-Learn Workflow | pending |
| 07 | Linear Regression | pending |
| 08 | Logistic Regression | pending |
| 09 | Decision Trees | pending |
| 10 | Ensemble Methods - Bagging & Boosting | pending |
| 11 | Random Forest | pending |
| 12 | Support Vector Machines | pending |
| 13 | Naive Bayes | pending |
| 14 | K-Nearest Neighbors | pending |
| 15 | K-Means Clustering | pending |
| 16 | PCA | pending |
| 17 | Evaluation Metrics & Cross-Validation | pending |
| 18 | Titanic Capstone Case Study | pending |
| 19 | AI/ML Applications in Mechanical Engineering & Manufacturing | pending |
| 20 | ML Project Ideas for Mechanical Engineers | pending |
| 21 | MLOps & Deployment | pending |

The old plan referenced a `build_session_pdfs.ps1` script for one-shot bulk generation
of all session PDFs — **that script does not exist in `LaTeX/`**. Either write it, or
keep using the manual procedure above (already used successfully for Sessions 2-5).

### 1.2 Python raw-file overlap/redundancy audit (not started)

Rebuilding `python_overview.tex` pulled content from several other `python_*.tex` files
(`python_setup.tex`, `python_advanced_topics_overview.tex`, the old
`python_syntax_overview.tex`/`python_quick_overview.tex`), raising the concern that the
~45+ `python_*.tex` topic files in the repo have accumulated real overlap, and that some
`seminar_python_*.tex` wrapper files may be orphaned ("artificially created", never wired
into a driver). Analysis only — do not execute changes without approval:

1. Enumerate all `python_*.tex` files and summarize what each covers.
2. For each, grep who `\input`s it — build a full consumer map.
3. Cross-check for content overlap (shell walkthroughs, "what is Python", setup/installation,
   OOP basics, etc. now appear in multiple places).
4. Enumerate all `seminar_python_*.tex` / `Main_Seminar_Python_*.tex` files; confirm via grep
   whether anything actually `\input`s each one. Flag zero-consumer files as dismantle
   candidates.
5. Produce a written recommendation (merge / rename / delete) and present it before touching
   any files.

---

## 2. Grading & assessment (rubrics due early Aug)

- [ ] **Assignment rubric** — correctness 40%, clarity/comments 20%, efficiency/best-practices
      20%, documentation 20%. Use for A1-A6.
- [ ] **Project rubric** (T1 & T2) — problem understanding 10%, data prep 15%, model
      selection 25%, evaluation & results 25%, documentation & presentation 25%.
- [ ] **Exam rubric** — MCQs 1 mark each; short answer 3 marks (concept/explanation/application);
      long answer 5 marks (understanding/analysis/clarity, partial credit); coding 5 marks
      (correctness 3 + efficiency 2); design 5 marks (feasibility 2.5 + completeness 2.5).

### Assessment calendar (needs re-check against 21-session structure)

- Assignments: A1 Jul 27 - Python+DataTypes, A2 Aug 5 - Math+Stats, A3 Aug 15 - ML
  Fundamentals, A4 Aug 25 - Regression+Trees, A5 Sep 5 - Classification+Unsupervised,
  A6 Oct 5 - Model Evaluation & Tuning
- Exams: Midterm late Aug/early Sep (after old Session 11); EndSem late Nov (after old
  Session 24)
- T1 project: topic selection Aug, proposal Sep, code+report Oct, presentations mid-Oct
- T2 project: problem definition Sep, EDA+modeling Oct, final submission Nov,
  presentations late Nov

### Grading turnaround

| Assessment | Due | Grade by | Effort |
|---|---|---|---|
| A1 | Jul 27 | Aug 3 | 2h |
| A2 | Aug 5 | Aug 12 | 2h |
| A3 | Aug 15 | Aug 22 | 2h |
| A4 | Aug 25 | Sep 1 | 2h |
| Midterm | late Aug | early Sep | 4h |
| A5 | Sep 5 | Sep 12 | 2h |
| A6 | Oct 5 | Oct 12 | 2h |
| T1 Project | Oct | Oct 19 | 4h |
| EndSem | late Nov | Nov 30 | 6h |
| T2 Project | Nov | Nov 30 | 5h |

---

## 3. Recurring operational workflows (Jul-Nov)

- [ ] Set up a Google Group for the course (attendance posts, sharing session PDFs
      within 24h, announcements) — one-time setup, keep active through the course.
- [ ] Create an attendance tracking sheet (students x sessions, P/A/L), updated weekly;
      target ≥70% attendance.
- [ ] Physical muster sheet per CoEP requirement — sign after every session, submit to
      admin at course end.
- [ ] Proof-of-Performance tracking sheet per session (date, topic, attendance counts,
      issues, notes) — submit to HOD monthly or at course end.

### Per-session cycle (repeat for every session)

- [ ] **Before** (~30 min): confirm slides ready, prep talking points/demo, test
      projector, prep datasets, notify class leader for attendance.
- [ ] **During** (1h): deliver content, coordinate attendance, demo, note questions.
- [ ] **After** (1-2h, within 24h): share PDF/notebook notes to Google Group, update
      PoP tracking sheet, prep next session's materials.

---

## 4. Course timeline (Jul-Nov 2026)

- **Jul**: finalize Python setup guide + `environment.yml`; set up Google Group +
  attendance sheet; generate session PDFs as needed; course starts, A1 distributed.
- **Aug**: grade A1; A2/A3/A4 distributed and graded on rolling weekly cadence; finalize
  rubrics; Midterm exam late Aug.
- **Sep**: grade Midterm; A5 distributed/graded; T1 project topic selection then proposal
  due; finalize A6.
- **Oct**: A6 distributed/graded; T1 project code+report due, then presentations mid-Oct;
  finalize EndSem questions.
- **Nov**: T2 project final submission early Nov; EndSem exam mid/late Nov; T2
  presentations; grade EndSem + T2 (2-week turnaround); consolidate final grades;
  submit attendance + PoP to CoEP admin; archive course materials.

---

## 5. Success metrics (Nov 30, 2026)

- [ ] All sessions completed on schedule; assessments graded within target turnaround;
      PoP sheet complete; muster signed every session.
- [ ] Average class score ≥65/100; ≥80% pass rate (≥40/100); ≥70% attendance.
- [ ] ≥85% assignment submission rate; Google Group active; all materials accessible.

---

## 6. Optional enhancements

- [ ] Collect industry exam papers (Cummins, Bajaj, etc.) for practice problems
      (by Nov 15).
- [ ] Ask HOD for a capstone project list to share as optional T2 project ideas
      (by Sep).

---

## References

- Technical deliverables (notebooks, assignments, exams, datasets):
  `Code/mlcoep/TODO_MLCOEP_TECHNICAL.md`
- Session content source: `LaTeX/course_mlcoep_content.tex`
- Datasets & code: `Code/mlcoep/`
