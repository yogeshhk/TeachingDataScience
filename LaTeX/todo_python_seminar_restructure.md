# Python Seminar Restructure — TODO

Scope: introduce a seminar layer between the two Python workshops and the raw `python_*.tex`
topic files, matching the course → workshop → seminar → raw-file hierarchy used for
Maths4ML and ML. 12 new seminars total (6 per workshop), grouped by line-count weight to
target ~1-2h each while staying thematically coherent.

Strategy per seminar:
1. Create the seminar content file (own `\section[]{}` + chained `\input{}`s of raw files).
2. Create its driver pair (Presentation + CheatSheet).
3. Compile the seminar driver standalone to verify it builds.
4. Run `/upgrade-deck` on the seminar Presentation.

After all 6 seminars in a workshop are done: rewrite that workshop's content file to chain
the new seminars instead of the raw files directly, then recompile the workshop and the
course to confirm nothing broke.

`python_oop.tex` is `\input` by both Basic (as its own seminar) and Advanced (as part of
seminar 1) — this duplication predates this restructure and is preserved as-is, not deduped.

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

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Basic_Intro_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Intro_Presentation.tex`

---

### Seminar B2 — Control Flow & Collections
Content file : seminar_python_basic_constructs_content.tex
Driver       : Main_Seminar_Python_Basic_Constructs_Presentation.tex (+ CheatSheet)
Covers       : python_conditionals, python_loops, python_liststuples, python_dictionariessets

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Basic_Constructs_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Constructs_Presentation.tex`

---

### Seminar B3 — Functions & Exceptions
Content file : seminar_python_basic_procedures_content.tex
Driver       : Main_Seminar_Python_Basic_Procedures_Presentation.tex (+ CheatSheet)
Covers       : python_functions, python_exceptions

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Basic_Procedures_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Procedures_Presentation.tex`

---

### Seminar B4 — Object-Oriented Programming
Content file : seminar_python_basic_oop_content.tex
Driver       : Main_Seminar_Python_Basic_OOP_Presentation.tex (+ CheatSheet)
Covers       : python_oop

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Basic_OOP_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_OOP_Presentation.tex`

---

### Seminar B5 — File IO & Scientific Libraries
Content file : seminar_python_basic_iolibraries_content.tex
Driver       : Main_Seminar_Python_Basic_IOLibraries_Presentation.tex (+ CheatSheet)
Covers       : python_fileio, python_scientific, python_plotting

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Basic_IOLibraries_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_IOLibraries_Presentation.tex`

---

### Seminar B6 — Data Mining, Testing & Wrap-up
Content file : seminar_python_basic_closure_content.tex
Driver       : Main_Seminar_Python_Basic_Closure_Presentation.tex (+ CheatSheet)
Covers       : python_mining, python_testing, python_assignments, python_conclusion

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Basic_Closure_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Basic_Closure_Presentation.tex`

---

### Basic workshop wrap-up
- [ ] Rewrite `workshop_python_basic_content.tex` to chain seminars B1-B6, keep Extra/References as trailing raw `\input`s
- [ ] Compile workshop : `texify -cp Main_Workshop_Python_Basic_Presentation.tex`

---

## ADVANCED WORKSHOP (workshop_python_adv_content.tex)

### Seminar A1 — OOP Deep-Dive & Iteration
Content file : seminar_python_adv_oopiteration_content.tex
Driver       : Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex (+ CheatSheet)
Covers       : python_oop, python_iterators, python_generators

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_OOPIteration_Presentation.tex`

---

### Seminar A2 — Advanced Functions & OS Utilities
Content file : seminar_python_adv_functionsos_content.tex
Driver       : Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex (+ CheatSheet)
Covers       : python_decorators, python_lambda, python_directories, python_datetime

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_FunctionsOS_Presentation.tex`

---

### Seminar A3 — Advanced Strings & Web Scraping
Content file : seminar_python_adv_stringsweb_content.tex
Driver       : Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex (+ CheatSheet)
Covers       : python_regularexpressions, python_intro_beautifulsoup

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_StringsWeb_Presentation.tex`

---

### Seminar A4 — Numerical & Data Libraries
Content file : seminar_python_adv_datalibs_content.tex
Driver       : Main_Seminar_Python_Advanced_DataLibs_Presentation.tex (+ CheatSheet)
Covers       : python_intro_numpy, python_intro_scipy, python_intro_pandas, python_sql

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Advanced_DataLibs_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_DataLibs_Presentation.tex`

---

### Seminar A5 — Visualization
Content file : seminar_python_adv_visualization_content.tex
Driver       : Main_Seminar_Python_Advanced_Visualization_Presentation.tex (+ CheatSheet)
Covers       : python_intro_bokeh, python_intro_tkinter

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Advanced_Visualization_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_Visualization_Presentation.tex`

---

### Seminar A6 — Coding Problems & System Design
Content file : seminar_python_adv_problems_content.tex
Driver       : Main_Seminar_Python_Advanced_Problems_Presentation.tex (+ CheatSheet)
Covers       : python_dsa, python_codingproblems_basic, python_systemdesign

- [ ] Create content file
- [ ] Create driver pair
- [ ] Compile   : `texify -cp Main_Seminar_Python_Advanced_Problems_Presentation.tex`
- [ ] Upgrade   : `/upgrade-deck Main_Seminar_Python_Advanced_Problems_Presentation.tex`

---

### Advanced workshop wrap-up
- [ ] Rewrite `workshop_python_adv_content.tex` to chain seminars A1-A6, keep References as trailing raw `\input`
- [ ] Compile workshop : `texify -cp Main_Workshop_Python_Advanced_Presentation.tex`

---

## After both workshops done

- [ ] Compile full course : `texify -cp Main_Course_Python_Presentation.tex`
- [ ] Compile full course cheatsheet : `texify -cp Main_Course_Python_CheatSheet.tex`
- [ ] Compile ML course (uses workshop_python_basic_content.tex as W1) : `texify -cp Main_Course_MachineLearning_Presentation.tex`
- [ ] Update CLAUDE.md with the new Python seminar-layer structure
- [ ] Delete this file once complete (matches precedent of todo_maths4ml_seminar_upgrade.md)
