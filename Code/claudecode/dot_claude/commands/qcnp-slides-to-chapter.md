---
name: qcnp-slides-to-chapter
description: Converts QCNP slide content (a single topic .tex file, or a full seminar's set of topic files) into an elaborated, explanatory book chapter under LaTeX/qcnp/book/, matching the QCNP audience contract and the source's position in the three-layer mental model. Acts as an updater on existing chapter content, never a blind recreator. Use when the user asks to "turn this seminar into a chapter", "draft the book chapter for X", "write/update the chapter from the slides", or calls /qcnp-slides-to-chapter.
---

# QCNP Slides to Chapter

You are adapting terse QCNP slide content (LaTeX Beamer frames, `LaTeX/qcnp/workshops/quantum_*.tex`)
into elaborated, explanatory book prose for "Quantum Computing for Non-Physicists," published
under `LaTeX/qcnp/book/`. The two live in permanent structural lockstep (see
`LaTeX/qcnp/README.md`'s "Structural parallel" section) but differ enormously in register: slides
are terse and bullet-driven; the book is elaborate, explanatory, and intuitive, written to be read
start to finish, not projected.

---

## 1. Initialization

The slash command argument is a source path: either one topic file
(e.g. `quantum_physics_intuition.tex`) or a seminar's content file
(e.g. `seminar_qcnp_physics_content.tex`, which `\input`s several topic files — treat each as
its own chapter within the same Part).

### Step 1: Ground yourself in the model, every time

Read, in this order, before touching any content:

1. `LaTeX/qcnp/README.md` — the mental model recap and the content-placement decisions (which
   layer/head owns what, and why).
2. `LaTeX/qcnp/QCNP Mental Model.md` — the full standing design doc, for the Reader Contract
   (§3), the Translation Dictionary template (§9), and the Format Adaptation rules (§11: "Book =
   full stack per chapter").
3. `LaTeX/qcnp/book/README.md` and `LaTeX/qcnp/book/TODO.md` — current Part/chapter structure and
   status. If the Part/chapter table there is stale or doesn't yet cover the source file's layer,
   say so explicitly and ask the user which Part/chapter this maps to, rather than guessing.
4. `LaTeX/qcnp/docs/QuantumSensePlaylist/02_slide_ready_excerpts.md` — curated excerpts from the
   QuantumSense YouTube series' 14 transcripts, organized by target QCNP file/section, each with a
   slide-bullet version and a fuller book-prose version already drafted. Especially useful for
   Layer 2 (Mathematics) chapters, where the transcripts' derivation-driven explanations go deeper
   than the slides' terse bullets. `00_concept_inventory.md` and `01_transcript_notes.md` in the
   same folder are the full per-transcript backing notes if the excerpts file doesn't cover a
   concept you need. Treat excerpts as source material to adapt into the chapter's own voice, per
   the Style Rules below, not text to paste in verbatim.

### Step 2: Identify the layer and locate the target chapter

Determine which layer/head the source file belongs to from its filename prefix and content:

| Source prefix | Layer / role | Register to write in |
|---|---|---|
| `quantum_physics_*` | Layer 1 — physical intuition | Narrative, analogy-first, builds a mental picture before any formalism |
| `quantum_mathematics_*` | Layer 2 — the interface (precise rules) | Precise but explained: every equation gets a plain-English "what this means" follow-up, never left to speak for itself |
| `quantum_computation_bridge` | The Bridge | Motivating narrative: why interference is the actual resource, told as a mini case-study (mirrors the existing Addition-vs-Grover case studies) |
| `quantum_computation_algorithms` | Layer 3 — core algorithms | Walkthrough style: problem, classical baseline, quantum mechanism, worked example |
| `quantum_computation_hybrid_intro`, `_qml`, `_optimization` | Decoder heads | Applied: what problem this head solves, how the shared hybrid loop specializes for it |
| `quantum_computation_implementation`, `_applications_survey` | Practical / survey | Grounded, real-world, less formalism, more "here's where this shows up" |
| `quantum_overview_*` | Cross-theme tour | Condensed: this maps to a single opening chapter (Part I), not one chapter per topic file — adapt the *whole* Overview seminar into one chapter, not a chapter per section |

Locate (or, if genuinely absent, ask the user to confirm) the target chapter file under
`LaTeX/qcnp/book/`. Do not invent a new chapter file path without confirming it against the
Part/chapter table.

### Step 3: Read the Reader Contract audience

Non-physicist engineers/programmers: school-level physics assumed, no university physics, no
prior Dirac notation or Hilbert space theory. Comfortable with vectors/matrices/basic probability
and enough Python to read a short script. Every physics concept gets taught properly, not diluted
— this is scope discipline, not a license to skip depth.

---

## 2. Update vs. Create — read the target file first, always

Before writing anything, read the target chapter file's current content.

- **If it is still the original scaffold** (a single `TODO: adapt content from ...` placeholder
  line, per `LaTeX/qcnp/book/README.md`'s "Filling In Content" section): write the chapter fresh,
  following the requirements below.
- **If it already has real prose**: you are an **updater, not a recreator**. Diff what the source
  slides now say against what the chapter currently says:
  - Preserve the existing chapter's voice, examples, and structure wherever the underlying
    concept hasn't changed.
  - Add or revise only the sections that correspond to slide content that is new, changed, or
    missing from the chapter.
  - Never regenerate the whole chapter from scratch just because the command was invoked again —
    that discards the author's prior editorial work (hooks, analogies, transitions) for no reason.
  - If a slide was removed or a concept was cut from the source, flag the now-orphaned chapter
    section rather than silently deleting it — let the user confirm the cut.

---

## 3. What a Chapter Needs (beyond what the slide has)

A slide deck and a book chapter are not the same artifact even when they teach the same concept.
Do not `\input` the Beamer frames or lightly reflow their bullets — write real chapter prose, one
step more effort than the bullets, per concept:

1. **Opening hook.** One to two paragraphs before any formalism: a real-world scenario, a
   surprising question, or a "why should you care" framing appropriate to the layer (see the
   register table above). For inspiration on tone, `Publications/LaTeX/book_gametheory`'s
   "Content Brief" format (hook example, concept, real-world examples, one-line takeaway) is a
   useful pattern — read-only reference, do not edit it or treat it as part of this repo's build.
2. **Concept exposition.** Expand every slide's terse bullets into full paragraphs. A slide bullet
   is a compressed pointer at an idea; the chapter has to actually explain it. Follow each
   equation with a plain-English sentence saying what it means and why it's true, not just what
   it says.
3. **Worked examples.** Where the source has a `quantikz` circuit, a code block, or a numeric
   worked example, keep it (translated to book formatting — `lstlisting` or a figure, per the
   template), but narrate it: walk through what happens at each step in prose, don't just present
   the block.
4. **Diagrams: ample, not sparse.** A book page can carry more figures than a slide ever could
   (slides lean on live delivery to fill the gap; a reader alone on a page can't). Two sources:
   - **Reuse from the slides first.** If the source topic file already has a raster image
     (`\includegraphics{<name>}`, backed by a file in `qcnp/workshops/images/`) or an inline
     `tikz`/`quantikz` diagram for a concept the chapter is covering, carry it into the chapter:
     copy the file into `LaTeX/qcnp/book/images/` (same filename) and `\includegraphics` it, or
     copy the `tikz`/`quantikz` code directly, rather than re-describing the concept in prose only
     or leaving a bare suggestion comment.
   - **New diagrams, generated via Gemini by the user (not by you).** Where a concept would
     benefit from a figure the slides never needed, insert a placeholder for the user to fill in
     after generating the image themselves:
     ```latex
     % IMAGE GENERATION PROMPT (Gemini) for <filename>.png:
     % <the full prompt text: what the diagram should show, style, labels, notation to match
     %   the chapter's own conventions>
     % \includegraphics[width=0.8\textwidth]{<filename>}
     ```
     The `\includegraphics` line stays commented out (so the file compiles even though the image
     doesn't exist yet) until the user generates the PNG in Gemini, saves it to
     `LaTeX/qcnp/book/images/<filename>.png`, and uncomments the line themselves. Never invent an
     `\includegraphics` reference to a file that doesn't exist and leave it uncommented.
     Filename convention: `<topic>_<subtopic>.png`, descriptive words only, lowercase, hyphens
     within a multi-word component (e.g. `bloch-sphere`), underscore between topic and subtopic
     (e.g. `qubits_bloch-sphere-rotation.png`, `gates_hadamard-superposition.png`). **No chapter or
     section number in the filename** (matches this repo's no-stale-position-encoding rule, and
     the existing precedent already in `qcnp/workshops/images/`: `bloch.png`,
     `double_slit_schematic.png`, `matrix_transformation_2d.png` are never numbered by slide
     position either).
5. **Common misconceptions.** Where the source has an explicit misconception callout (the
   Translation Dictionary template's item 5, or a "Not X, actually Y" framing), keep it as a
   short prose aside, not a bullet list.
6. **End-of-chapter questions.** Close every chapter with 2-4 short conceptual questions (not rote
   recall — mirror the "thought-provoking, not trivia" bar `/upgrade-deck`'s Task 6 quiz slides
   use), with brief answers or discussion. These are for a reader working alone, not a live
   audience, so don't assume a "pause for the room" moment. Never label this section "Exercises"
   or number the questions (no "Exercise 4.1"): this is a popular-science book, not a textbook, and
   a numbered problem set is the wrong genre signal even if the content underneath stays light.
7. **One-line takeaway.** Close with a single sentence stating the chapter's core point, echoing
   the Content Brief format's last element.

---

## 4. Style Rules

- No em dashes anywhere in generated prose (repo-wide rule, see root `CLAUDE.md`) — use a colon,
  semicolon, comma, or parentheses instead.
- Match `LaTeX/qcnp/book/template_book.tex`'s existing conventions: `\chapter{}`, `\section{}`
  structure, whatever custom environments the template already defines for blocks/asides. Do not
  introduce new packages or macros — if a slide's content genuinely needs one, flag it rather than
  adding it silently.
- Do not invent physics, math, or claims the source slides don't support. If the terse slide
  content is ambiguous or looks technically thin for a full chapter's worth of explanation, say so
  explicitly rather than padding with confident-sounding filler.
- Stay within the source seminar's scope. Do not pull in content from a different layer's slides
  to "fill out" the chapter — if the chapter feels short, that's a signal to flag, not a license to
  borrow material that belongs to a different chapter.

---

## 5. Output

1. **Summary first**: which source file(s) were read, which target chapter file this maps to, and
   whether this is a create or an update pass.
2. **If updating**: a short diff-style list of what changed and why, before the full file.
3. **Full updated chapter file content**, ready to write to `LaTeX/qcnp/book/<chapter file>.tex`.
4. **Open questions**: anything that needed the user's confirmation (ambiguous Part/chapter
   mapping, thin source content, an orphaned section from a removed slide) — surfaced, not
   silently resolved.
5. **New Gemini image prompts inserted**: a short list of every new `<filename>.png` placeholder
   added in this pass (per §3 item 4), so the user has a checklist of images to go generate and
   drop into `LaTeX/qcnp/book/images/` before the commented-out `\includegraphics` lines can be
   uncommented.

---

## 6. Guardrails

- **Never overwrite existing chapter prose wholesale** on a re-run — see §2, this is the single
  most important rule in this command.
- **No guessing the target chapter path** — confirm against `LaTeX/qcnp/book/README.md`'s
  Part/chapter table, or ask.
- **Audience discipline** — every explanation should land for an engineer with school-level
  physics, not a physics undergraduate. If you notice yourself reaching for university-level
  formalism the source didn't use, that's a sign to find the intuitive route instead, not to
  "upgrade" the reader's assumed background.
- **Layer-appropriate register** — don't write a Layer 3 algorithm chapter in the same narrative,
  analogy-heavy voice as a Layer 1 physics chapter; match the register table in §1.
- **This command does not touch `LaTeX/qcnp/workshops/`** — it only reads slide content as a
  source, never edits the slides themselves.
