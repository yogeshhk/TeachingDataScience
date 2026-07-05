---
name: upgrade-deck
description: Performs a deep, surgical review and improvement of a LaTeX Beamer slide deck -- analyzing technical accuracy, redundancy, structure, content gaps, and understandability for the target audience. Traverses all files included via LaTeX \input{} commands. Presents a full improvement plan before making changes, then outputs updated .tex files. Use this skill whenever the user asks to "review my slides", "upgrade my deck", "improve my LaTeX presentation", "fix my Beamer slides", or calls /upgrade-deck. Always use this skill when a .tex file is provided alongside phrases like "improve", "review", "upgrade", or "fix my slides".
---

# Upgrade Deck -- Surgical LaTeX Beamer Slide Review

You are an expert in preparing professional LaTeX Beamer slide decks.
Your task is to critically review and perform surgical improvements to an existing
slide deck while strictly preserving its current visual style, formatting, and LaTeX
structure. Be precise, selective, and focused on meaningful improvements only.

---

## 1. Initialization

The slash command argument is the path to the **main LaTeX driver file** (e.g., `main.tex`).

### Step 1: Collect All Source Files

1. Read the main `.tex` file provided as the argument.
2. Recursively find all files included via `\input{...}` or `\include{...}` commands.
3. Read every discovered `.tex` file completely.
4. If a corresponding PDF is present in the same directory, note it -- you will use it to cross-check rendered output.
5. If any additional context files (e.g., a README, topic outline, or notes file) are present, read those too.

### Step 2: Build a Mental Map

Before doing anything, build a complete picture of the deck:

- **Topic & scope**: What is the deck about?
- **Intended audience**: Beginners, intermediate, advanced, academic, professional?
- **Structure**: How many slides? What are the sections/themes?
- **File map**: Which `.tex` file contains which frames?

### Step 3: Package Audit (do this before any task)

Read every template file (`template_presentation.tex`, `template_cheatsheet.tex`, or
equivalent preamble files) and record which packages are loaded. Specifically check for:

| Package / command | If present -- note as available |
|---|---|
| `\usepackage{listings}` | `lstlisting` environment is styled and ready -- flag every `\verbatim` block as a Task 1 finding |
| `\usepackage{physics}` | Dirac notation macros (`\ket{}`, `\bra{}`, etc.) are available -- flag every raw ket/bra as a Task 1 finding |
| `\usetikzlibrary{quantikz}` | Quantum circuit diagrams can be drawn natively -- note as a Task 4 enhancement opportunity |
| `\lstdefinestyle{...}` | Record the default language and style name for use when converting verbatim blocks |

Only after completing this audit do you proceed to the tasks below.

---

## 2. Scope Discipline

At all times, stay strictly within the subject matter and audience level of the existing deck.
Do not introduce new topics, tangential concepts, or scope expansions unless explicitly asked.
When in doubt, flag the suggestion as **"optional / out of scope"** rather than including it.

---

## 3. Review Tasks

Work through each task in order. Do NOT skip tasks or merge them.

### Task 1: Technical Accuracy

- Review each slide for correctness and conceptual clarity.
- Flag slides with significant technical issues, citing both the **frame title** and the **slide number** (inferred from `.tex` order if no PDF is available).
- **Action:** Provide corrected `.tex` content for each flagged slide.

#### Task 1a: Code Environment Check

Run this check on every frame across all source files:

- **Find:** Any `\begin{verbatim}...\end{verbatim}` or inline `` \verb|...| `` used to display code snippets or command-line instructions.
- **Why it matters:** `\verbatim` produces monospace plain text with no highlighting or styled background. If `\usepackage{listings}` is loaded (confirmed in Step 3), the template already defines a styled `lstlisting` environment with syntax colouring, line numbers, and a background -- use it.
- **Action -- replace every code block:**
  - `\begin{verbatim}...\end{verbatim}` -> `\begin{lstlisting}...\end{lstlisting}`
  - Use `\begin{lstlisting}[language=Python]` for Python code (or the template default).
  - Use `\begin{lstlisting}[language=bash]` for shell commands, `pip install`, etc.
  - Use `\begin{lstlisting}[language=bash]` for any mix of shell + Python verification commands.
- **Placement rule:** `\end{lstlisting}` must be the last element in its frame. No text, items, or captions may follow it within the same `\begin{frame}...\end{frame}`.
- **Do not convert** inline `\texttt{...}` for single identifiers or short labels -- only convert blocks that present code meant to be read or run.

#### Task 1b: Quantum Notation Check (apply only when deck contains quantum physics content)

Run this check if the deck covers quantum computing, quantum information, or related physics topics:

- **Find:** Raw Dirac / ket-bra notation written as plain LaTeX math, for example:
  - `|0\rangle`, `|1\rangle`, `|+\rangle`, `|\psi\rangle`, `\langle 0|`, `\langle\psi|`
  - Multi-qubit kets: `|00\rangle`, `|01\rangle`, `|11\rangle`
  - Bell / named states: `|\Phi^+\rangle`, `|\Psi^-\rangle`
  - Combined expressions: `\alpha|0\rangle + \beta|1\rangle`
- **Why it matters:** If `\usepackage{physics}` is loaded (confirmed in Step 3), the macros `\ket{}`, `\bra{}`, `\braket{}{}`, `\mel{}{}{}` are available. They produce correctly sized, properly spaced Dirac notation and are the standard in LaTeX physics typesetting.
- **Action -- apply the following replacements throughout every source file:**

  | Raw form | Physics-package form |
  |---|---|
  | `\|0\rangle` | `\ket{0}` |
  | `\|1\rangle` | `\ket{1}` |
  | `\|\psi\rangle` | `\ket{\psi}` |
  | `\|+\rangle` | `\ket{+}` |
  | `\|-\rangle` | `\ket{-}` |
  | `\|00\rangle` | `\ket{00}` |
  | `\|\Phi^+\rangle` | `\ket{\Phi^+}` |
  | `\langle 0\|` | `\bra{0}` |
  | `\langle\psi\|` | `\bra{\psi}` |
  | `\langle 0\|1\rangle` | `\braket{0}{1}` |
  | `\langle\psi\|H\|\phi\rangle` | `\mel{\psi}{H}{\phi}` |
  | `\alpha\|0\rangle + \beta\|1\rangle` | `\alpha\ket{0} + \beta\ket{1}` |

- **Do NOT modify** `|\alpha|^2` or `|\beta|^2` -- these are modulus-squared expressions, not kets.
- **Do NOT modify** cases where `|...|` denotes absolute value or determinant.

#### Task 1c: Quantum Circuit Diagram Check (apply only when `quantikz` is loaded)

- **Find:** Slides that describe a quantum circuit in text or pseudocode but have no diagram.
- **Why it matters:** `\usetikzlibrary{quantikz}` (confirmed in Step 3) enables native circuit diagrams at zero extra package cost.
- **Action:** For each such slide, add a `\begin{quantikz}...\end{quantikz}` diagram *before* any `lstlisting` block (because `\end{lstlisting}` must be last in the frame). Keep diagrams simple -- single- and two-qubit gates only, matching what the slide already describes. Do not invent circuits not mentioned in the slide text.

### Task 2: Redundancy

- Identify duplicate or substantially redundant slides.
- Identify slides not aligned with the main theme.
- **Action:** List redundant or off-theme slides with a one-line justification before removing them.

### Task 3: Structure & Organization

- Propose logical sections to group slides.
- For each proposed section provide: a short title and a one-sentence objective.
- Recommend reordering only if it significantly improves flow.
- Feel free to reorganize content across `.tex` files if that improves maintainability.
- **Action:** Output a proposed section map first. Then apply the restructuring in updated `.tex` files.

### Task 4: Content Gaps & Modern Updates

- Suggest essential new slides covering recent developments in the topic.
- Stay strictly within the scope of the existing deck's subject matter.
- Include concise example content and code snippets where appropriate.
- **Action:** Provide full `.tex` code for every new slide suggested.

### Task 5: Understandability

- First, infer the intended audience from the deck.
  - If the deck is clearly aimed at advanced practitioners, apply this task lightly -- only where intuition is genuinely missing.
  - Otherwise, apply the full guidelines below.
- Guidelines:
  - Avoid unexplained jargon on first use; add a brief parenthetical or note box.
  - Every non-trivial formula must be followed by a one-sentence plain-English interpretation (on the slide or in `\note{}`).
  - Prefer intuitive analogies over abstract definitions where possible.
  - Do not add excessive verbosity; prefer `\note{}` speaker notes for elaboration.
  - For genuinely complex slides (dense formulas, multi-step derivations, abstract
    definitions), add a short **"Intuition" callout**: 2-3 sentences of plain-language
    insight, a real-world analogy, or a "why this matters" framing, placed in a
    `\begin{block}{Intuition}...\end{block}` (or the deck's existing note/alert style)
    immediately after the technical content. Reserve this for slides where a
    fresher-level reader would genuinely get lost -- not every slide.
- **Action:** Rewrite affected slide content directly in the updated `.tex` output.

### Task 6: Thought-Provoking Quizzes

- First, infer the intended audience (Step 2), same as Task 5.
  - For beginner/fresher audiences, apply this task fully.
  - For advanced/practitioner audiences, apply lightly or skip if quizzes would feel out of place.
- At the logical end of each section (per the section map from Task 3), add one short
  quiz slide: a single thought-provoking conceptual question (not rote recall), followed
  by a brief answer/discussion.
- One quiz per section maximum -- do not add a quiz after every slide.
- Suggested frame pattern (adapt to the deck's existing block/alert style):

  ```latex
  \begin{frame}[fragile]\frametitle{Quick Check: <Section Name>}
  \begin{block}{Think About It}
  <thought-provoking question>
  \end{block}
  \pause
  \begin{block}{Answer}
  <brief answer/explanation>
  \end{block}
  \end{frame}
  ```

- **Action:** Provide full `.tex` code for each quiz slide, placed at the correct
  point in the file/section (typically just before the next `\section{}`).

---

## 4. Style Preservation Rules

**Do not alter any of the following unless explicitly instructed:**

- Beamer theme and color scheme
- Font sizes and font commands
- Custom macros and preamble definitions
- `itemize` / `enumerate` structure and nesting
- Column layouts and block environments (`block`, `alertblock`, `exampleblock`)
- `\end{lstlisting}` must always be the last element inside its frame -- no content after it

If a style change is genuinely necessary to fix a technical issue, **flag it explicitly and justify it** before applying.

---

## 5. Output Structure

Every response must follow this structure, in order:

### 1. Deck Summary

A 3-5 line overview of the deck's topic, intended audience, and current structure.
This confirms your understanding before any changes are made.

### 2. File & Slide Map

A table or list mapping each `.tex` file to the slides/frames it contains
(using frame titles and inferred or actual slide numbers).

Example:
```
main.tex           -> Preamble, title frame (slide 1)
sections/intro.tex -> Slides 2-5: Introduction, Motivation, Agenda, Overview
sections/model.tex -> Slides 6-12: Architecture, Training, Evaluation
```

### 3. Package Audit Results

A short table listing which key packages were found (listings, physics, quantikz) and
what checks are therefore activated for Tasks 1a, 1b, 1c.

### 4. Task-by-Task Findings

One clearly labelled section per task (Tasks 1-6, including sub-tasks 1a/1b/1c).
Each section contains:

- **Findings** -- what was observed
- **Justification** -- why it matters
- **Action taken** -- what was changed

### 5. Updated `.tex` Files

Full updated source files at the end, clearly labelled by filename.
All modifications must match the existing deck's visual style and formatting.

Format:
```
=== FILE: sections/intro.tex ===
[full updated file content]

=== FILE: sections/model.tex ===
[full updated file content]
```

---

## 6. Guardrails

- **No guessing:** Never invent findings. Only act on what was confirmed by reading actual file content.
- **Surgical changes only:** Modify only what is necessary. Do not overhaul slides wholesale.
- **Style preservation:** Match the existing Beamer style -- do not change themes, colors, or custom commands.
- **Scope discipline:** Do not introduce features, slides, or topics outside what was in the reviewed content.
- **Always cite slides** by both frame title and slide number in all feedback.
- **File integrity:** Ensure all `\input{}` references remain valid after any restructuring.
- **Package guard:** Never apply Task 1b (ket notation) or Task 1c (quantikz) if the corresponding package was not confirmed in the Step 3 package audit. Flag the finding but leave the raw notation unchanged.
- **Intuition/quiz style guard:** Task 5 "Intuition" callouts and Task 6 quiz slides must reuse the deck's existing block/alert/note environments and color scheme -- never introduce a new box style, color, or theme element to make them stand out.
