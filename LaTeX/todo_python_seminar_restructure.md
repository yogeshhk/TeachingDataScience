# Python Seminar Restructure — TODO

Scope: introduce a seminar layer between the two Python workshops and the raw `python_*.tex`
topic files, matching the course → workshop → seminar → raw-file hierarchy used for
Maths4ML and ML. 12 new seminars total (6 per workshop), grouped by line-count weight to
target ~1-2h each while staying thematically coherent.

Status: structural build-out is DONE (all content files, driver pairs, workshop/course
rewrites in place). CheatSheet variants were compiled as a quick sanity check for all 12
seminars, both workshops, the Python course, and the ML course (which depends on
`workshop_python_basic_content.tex`) — all passed with unchanged page counts, so nothing
was lost or duplicated.

Remaining work (deferred to next session, one seminar at a time — Presentation compiles are
slow, so do NOT batch them):
1. Compile the seminar's Presentation driver: `texify -cp <driver>_Presentation.tex`
2. Run `/upgrade-deck <driver>_Presentation.tex`

After all 6 seminars in a workshop have been through this, compile that workshop's
Presentation driver once, then (after both workshops) the course Presentation drivers.

`python_oop.tex` is `\input` by both Basic (as its own seminar) and Advanced (as part of
seminar A1) — this duplication predates this restructure and is preserved as-is, not deduped.

`Extra` (kids_short, automation, engineering_short, ai_short) and `References` (python_refs)
sections in both workshops stay as trailing raw `\input`s directly in the workshop content
file, NOT wrapped in a seminar — same precedent as Maths4ML/ML courses keeping references
out of the seminar layer.

---

## BASIC WORKSHOP (workshop_python_basic_content.tex)

### Seminar B1 — Introduction, Syntax & Data Types
Content file : seminar_python_basic_intro_content.tex
Driver       : Main_Seminar_Python_Basic_Intro_Presentation.tex (+ CheatSheet)
Covers       : python_intro_short, python_setup, python_syntax, python_datatypes, python_operators

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Basic_Intro_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Intro_Presentation.tex`

---

### Seminar B2 — Control Flow & Collections
Content file : seminar_python_basic_constructs_content.tex
Driver       : Main_Seminar_Python_Basic_Constructs_Presentation.tex (+ CheatSheet)
Covers       : python_conditionals, python_loops, python_liststuples, python_dictionariessets

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Basic_Constructs_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Constructs_Presentation.tex`

---

### Seminar B3 — Functions & Exceptions
Content file : seminar_python_basic_procedures_content.tex
Driver       : Main_Seminar_Python_Basic_Procedures_Presentation.tex (+ CheatSheet)
Covers       : python_functions, python_exceptions

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Basic_Procedures_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Procedures_Presentation.tex`

---

### Seminar B4 — Object-Oriented Programming
Content file : seminar_python_basic_oop_content.tex
Driver       : Main_Seminar_Python_Basic_OOP_Presentation.tex (+ CheatSheet)
Covers       : python_oop

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Basic_OOP_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_OOP_Presentation.tex`

---

### Seminar B5 — File IO & Scientific Libraries
Content file : seminar_python_basic_iolibraries_content.tex
Driver       : Main_Seminar_Python_Basic_IOLibraries_Presentation.tex (+ CheatSheet)
Covers       : python_fileio, python_scientific, python_plotting

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Basic_IOLibraries_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_IOLibraries_Presentation.tex`

---

### Seminar B6 — Data Mining, Testing & Wrap-up
Content file : seminar_python_basic_closure_content.tex
Driver       : Main_Seminar_Python_Basic_Closure_Presentation.tex (+ CheatSheet)
Covers       : python_mining, python_testing, python_assignments, python_conclusion

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Basic_Closure_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Closure_Presentation.tex`

---

### Basic workshop wrap-up
- [x] Rewrite `workshop_python_basic_content.tex` to chain seminars B1-B6, keep Extra/References as trailing raw `\input`s
- [x] Sanity-check compile (CheatSheet) — passed, 46 pages (unchanged from before restructure)
- [ ] Compile Presentation : `texify -cp Main_Workshop_Python_Basic_Presentation.tex` (do after all 6 seminars upgraded)

---

## ADVANCED WORKSHOP (workshop_python_adv_content.tex)

### Seminar A1 — OOP Deep-Dive & Iteration
Content file : seminar_python_adv_oopiteration_content.tex
Driver       : Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex (+ CheatSheet)
Covers       : python_oop, python_iterators, python_generators

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex`

---

### Seminar A2 — Advanced Functions & OS Utilities
Content file : seminar_python_adv_functionsos_content.tex
Driver       : Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex (+ CheatSheet)
Covers       : python_decorators, python_lambda, python_directories, python_datetime

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex`

---

### Seminar A3 — Advanced Strings & Web Scraping
Content file : seminar_python_adv_stringsweb_content.tex
Driver       : Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex (+ CheatSheet)
Covers       : python_regularexpressions, python_intro_beautifulsoup

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex`

---

### Seminar A4 — Numerical & Data Libraries
Content file : seminar_python_adv_datalibs_content.tex
Driver       : Main_Seminar_Python_Advanced_DataLibs_Presentation.tex (+ CheatSheet)
Covers       : python_intro_numpy, python_intro_scipy, python_intro_pandas, python_sql

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Advanced_DataLibs_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_DataLibs_Presentation.tex`

---

### Seminar A5 — Visualization
Content file : seminar_python_adv_visualization_content.tex
Driver       : Main_Seminar_Python_Advanced_Visualization_Presentation.tex (+ CheatSheet)
Covers       : python_intro_bokeh, python_intro_tkinter

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Advanced_Visualization_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_Visualization_Presentation.tex`

---

### Seminar A6 — Coding Problems & System Design
Content file : seminar_python_adv_problems_content.tex
Driver       : Main_Seminar_Python_Advanced_Problems_Presentation.tex (+ CheatSheet)
Covers       : python_dsa, python_codingproblems_basic, python_systemdesign

- [x] Create content file
- [x] Create driver pair
- [x] Sanity-check compile (CheatSheet) — passed
- [ ] Compile Presentation : `texify -cp Main_Seminar_Python_Advanced_Problems_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_Problems_Presentation.tex`

---

### Advanced workshop wrap-up
- [x] Rewrite `workshop_python_adv_content.tex` to chain seminars A1-A6, keep References as trailing raw `\input`
- [x] Sanity-check compile (CheatSheet) — passed, 57 pages
- [ ] Compile Presentation : `texify -cp Main_Workshop_Python_Advanced_Presentation.tex` (do after all 6 seminars upgraded)

---

## After both workshops done

- [x] Sanity-check compile Python course (CheatSheet) — passed, 148 pages (unchanged)
- [x] Sanity-check compile ML course (CheatSheet, uses workshop_python_basic_content.tex as W1) — passed, 256 pages (unchanged)
- [ ] Compile Presentation : `texify -cp Main_Course_Python_Presentation.tex`
- [ ] Compile Presentation : `texify -cp Main_Course_MachineLearning_Presentation.tex`
- [ ] Update CLAUDE.md with the new Python seminar-layer structure
- [ ] Delete this file once complete (matches precedent of todo_maths4ml_seminar_upgrade.md)
