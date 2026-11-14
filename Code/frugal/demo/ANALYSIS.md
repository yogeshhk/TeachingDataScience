## Demo analysis: what it is, what it shows, what it does not

Written 2026-10-28 after the author asked why the Marathi demo image is dense and does not show the intended story.
This is analysis only. No slide, screenshot or code has been changed. The plan is at the end and waits for the author.

## 1. What the demo is

`app.py` is a Streamlit page with three sections: ask a document (`quantum-wiki`), a Marathi model fine-tuned on one
book (`finetune-slm`, Part 2), and a 135M-parameter router (`finetune-slm`, Part 1). `capture.py` takes the ten PNGs in
`TeachingDataScience/LaTeX/images/frugal_demo_*.png` from that page (removed 2026-10-28, replaced by the five
`frugal_rag_q*.png`), with `hide=1` so Marathi answers are not shown
(the book's licence is unconfirmed). Each PNG is therefore a screenshot of the Streamlit page, not a designed slide.
That is why they look like software and not like a slide.

## 2. The Marathi image, panel by panel

- **Six buttons at the top:** the fixed questions (first two of each test group: reworded, raw text only, control).
- **Big heading:** the chosen question.
- **Three panels:**
  - Base model, no help: a recorded answer from the measured run (2026-10-21). The base model is not loaded in the demo.
  - Fine-tuned model: generated live now, by the Marathi GGUF (bf16) through LM Studio. The measured table comes from
    the PyTorch adapter run, not from this GGUF, and the README says no quality number was measured on the GGUF.
  - Base model + 2 book passages: a recorded answer, the base model with the top two BM25 passages in the prompt.
  - Each panel shows only pills: recorded or live, correct or not correct, seconds, tokens, "Devanagari 100%". The
    answer text is hidden, so the audience cannot see what was answered. "Correct" is a mechanical check that the key
    phrases written by Claude occur in the answer.
- **Table:** the measured result on 52 questions, one run, one seed.

## 3. The intent against the data

Test groups (from the README): **reworded** (20) is a fact that is also in a training question-answer pair, asked in new
words. **Raw only** (20) is a fact that appears only in the raw book passages the model was trained on. **Control**
(12) is questions about chapters withheld from training.

| System | Reworded (20) | Raw only (20) | Control (12) | All (52) |
|--------|---------------|---------------|--------------|----------|
| Base, no help | 2 | 4 | 2 | 8 (15%) |
| Fine-tuned, no help | 12 | 2 | 1 | 15 (29%) |
| Base + 2 retrieved passages | 13 | 15 | 8 | 36 (69%) |

| Claim we wanted to show | Does the data show it? |
|-------------------------|------------------------|
| The base model alone is weak on this book | Yes: 15%, and its answers are fluent but invented. |
| Base + retrieval is much better | Yes: 69%, ahead in every group. Retrieval found the needed text in the top two for 40 of 52 questions. |
| Fine-tuning makes the model know the book | **No.** On facts that were only in the raw book text it scores 2 of 20 (base 4 of 20). Three passes over 18,000 words did not put them into the weights. |
| Fine-tuned beats base | **Only on the reworded group** (12 against 2). Those facts were in the training question-answer pairs, so this measures memorised answers asked in new words. On raw text and on withheld chapters it is level with or below the base model, within noise. |
| Fine-tuned beats retrieval | **No.** 15 against 36 overall. On reworded questions they tie (12 against 13). |

Brutal reading: the headline "15 against 8" is almost entirely the reworded group, and that group is a test of
memorised training pairs. The aggregate hides it. What the experiment really found is: fine-tuning on a small book
copies the question-answer pairs you give it and changes the style (24 tokens instead of 68, so 5.5 s against 13.8 s,
which is mostly shorter answers, not a faster model). It does not add the book's facts. For new facts, retrieval wins.
That is a legitimate, teachable result and it matches the repo's own pre-registered expectation. It is not the story
"fine-tuned on Marathi text gives better answers", and we must not present it as that.

## 4. Why the image fails, ranked

1. **It shows the opposite of its slide title.** The captured example (the first reworded question) has the fine-tuned
   model "not correct" and retrieval "correct". The same is true of the other two captures. The fixed rule (first two
   per group) is honest; picking a passing example afterwards would be cherry-picking. A single question is the wrong unit.
   The group score is the evidence.
2. **Nobody can read the answers.** Hidden for licence reasons, so each panel is coloured pills and metadata. The only
   information is a red or green label whose meaning needs the whole method explained.
3. **Too many ideas at once:** six buttons, a heading, three panels with five items each, a four-column table and a
   footnote. Plus noise that serves no point ("Devanagari 100%", "recorded", "live now", tokens, seconds).
4. **Live and recorded are mixed** in one row. The live panel is the GGUF, the other two are the measured run, and the
   table is a third thing. The pills do not say so clearly.
5. **Small samples and unreviewed data:** 20, 20 and 12 questions, one seed, differences of one or two questions are
   noise, keys matched mechanically (correct answers worded differently count as wrong), and the author has not
   reviewed the questions or keys.
6. **Speed is confounded with style.** The tuned model is faster because it answers in 24 tokens, not because
   fine-tuning speeds up a model.

## 5. The other two sections

- **Quantum wiki (documents).** Intent: base plus plain RAG is weak on a structured PDF, and a structure-aware system is
  better. Supported on the measured numbers (plain search 0.40, structure retrieval 0.69, agent 0.975 on the tuned set).
  Weaknesses:
  - 0.975 is on the questions the router was tuned on. The held-out 0.933 is after one round of repairs, so it is not
    blind. The naive baseline was run on the held-out set on 2026-10-31 (0.40). The footer mixes sets.
  - The "agent" is mostly plain code: about 64% of its answers use no model. The honest and frugal message is "do not
    use a model where code is enough", not "RAG plus a model".
  - **The "spread over many places" screenshot does not show that.** For "Where does Grover's algorithm appear?" the agent
    names one course (QT 05, page 24) and the plain search also names page 24. The slide title and the evidence disagree.
  - The "status" screenshot ("Can a student skip QT 02?") does show a clear win: plain search says "The syllabus does not
    say", the agent answers correctly with no model call.
- **Router (135M).** Intent: fine-tuning a tiny model for one job beats prompting the same model. Supported: 0.99 same
  wording and 0.92 new wording, against 0.23 and 0.20 for instructions only and 0.57 and 0.42 with examples.
  Limits that the page does not say: synthetic data, six classes, one seed, and the Qwen3 1.7B prompted comparison
  was never run, so "fine-tuning beats prompting" holds only for the same 135M model. This is the cleaner fine-tuning story.

## 6. What we may claim, and what we must not

May claim, with "one run, one machine, questions drafted with Claude's help":
- Retrieval lifted a 1B Marathi-capable model from 15% to 69% on this book.
- Fine-tuning on this book copied training answers (60% against 10% on reworded questions) and shortened answers, but
  did not add facts that were only in the raw text (10% against 20%).
- A tiny fine-tuned router beats the same tiny model prompted, on a made-up task.
- A structure-aware tool layer answered a syllabus PDF far better than plain search, mostly with no model call.

Must not claim:
- That the fine-tuned model "learned the book" or "knows Marathi text better".
- The aggregate 15 against 8 without the per-group split.
- That fine-tuning beats prompting or retrieval in general.
- Any number as final before the author has reviewed the question and key sets.

## 7. Plan (waits for the author; nothing below is done)

Design rule from the author (2026-10-28): demos and their UI must be very simple and minimalistic, no clutter. One page
per demo: a question, two answers side by side headed "Without X" and "With X", and one plain line of measured result
under them. No sidebar, badges, token counts, timings or tables. The offline proof is the Wi-Fi switch, not a counter.

Decisions (author, 2026-10-28), which override the items below where they differ:
- The router is out of the demo. It was a separate warm-up project, not part of the brief.
- The Marathi fine-tune (Sahaj Jeevan) is out of the demo, because as measured it did not learn the book. It stays in
  the repo and the README as an honest result, and gets one sentence on the slide if wanted.
- The demo is the quantum wiki only: one question, "Without the PDF" against "With the PDF", one line of measured
  result. Images: one pair (two pictures, or one side-by-side) taken from that page. Delete the other spare images only
  after the author agrees.

1. **Decide the Marathi story.** Recommended: show the true result in one idea, "Retrieval gave the book's facts;
   fine-tuning only copied the training answers." Alternatives: (a) also run the untested combination, a tuned model
   reading retrieved passages (about 52 answers, needs LM Studio or the adapter); (b) drop the Marathi fine-tune from
   the deck and keep the router as the fine-tuning example.
2. **Cut ten pictures to four** (the author's proposal, 2026-10-28). Ten exist because `capture.py` takes one shot per
   pre-fixed question (six document kinds, three Marathi, one router), as a fallback if the live run fails and so that no
   example is picked after seeing answers. The deck uses only four of them; six are spare. Proposed four, each one idea:
   - Documents, base model alone: wrong or "cannot know". The app has no base-only view yet, so this needs a real run added.
   - Documents, with retrieval (or the structure-aware agent): correct.
   - Fine-tuning, base model: wrong. Fine-tuning, tuned model: correct.
   Honesty conditions: the fine-tuning pair is only truthful for the router (base 0.23, tuned 0.99 on a made-up task) or
   for a Marathi question that is a reworded training fact, and then the caption must say "60% on such questions, 10% on
   facts from the raw text". A single passing Marathi question shown without that caption would be cherry-picking.
3. **Replace the dense screenshots with designed slides.** One idea per slide, at most three numbers, a plain title that
   states the finding, no app chrome. For Marathi: one bar chart of the three systems by group (reworded, raw text,
   withheld chapters), nothing else. For documents: the "Can a student skip QT 02?" pair as two large text boxes plus
   one line of measured accuracy. For the router: one three-bar chart.
4. **Fix the mismatches.** Retitle or replace the Grover screenshot; label the footer sets (tuned or held-out);
   state in the router caption that the larger prompted comparison was not run; keep the 15 against 8 only with the split.
5. **Keep the live app for the room, not the slides.** In the live run the audience hears the explanation; the slides
   carry the measured evidence. Decide whether the live Marathi panel should stay (it is the GGUF, unmeasured).
6. **Open items that cap what can be said:** the author's review of the 52 Marathi and 110 syllabus questions and keys,
   the book's licence (blocks showing answers), and a person reading a sample of the Marathi answers to check the
   mechanical scoring.

## 8. Trial: Base against Base + RAG (2026-10-28, one run, Qwen3 1.7B, structure-aware retrieval)

Nine questions run twice (answers not saved): the model alone, and the same model with the retrieved course chunks. Gold from `eval/`.

| Question | Base model alone | Base + RAG | Use in demo |
|----------|------------------|------------------|-------------|
| For how many lectures should sensing be taught within QT 01? | "2 lectures" (wrong) | "8-10 lectures [p. 17]" (right) | yes |
| What is the pass mark for QT 03? | invents "typically 50%" | "The syllabus does not say" (right, the document has no pass mark) | yes, shows honesty |
| What is the title of QT 08? | invents "Quantum Tunnelling" | "Introduction to Quantum Materials [p. 30]" | yes |
| Which course is called Quantum Optics? | a generic definition, no course | "QT 11 [p. 36]" | yes |
| What are the credits of QT 04? | talks about film credits | "2:1 [p. 22]" | yes, clear but a little comic |
| Name the electives. | a list of school subjects | "QT 09, QT 10, QT 11" (right, but a clumsy "Answer:" line) | maybe |
| Which courses are optional? | vague | "QT 09, QT 10, QT 11 [p. 14]" | maybe |
| Is QT 11 compulsory or optional? | "optional" (right by luck, wrong reasons) | "optional" | no, no contrast |
| Can a student skip QT 02? | "Yes" (wrong) | "Yes" (**wrong**; QT 02 is mandatory) | **no, RAG fails here** |

Reading: the demo works well on questions about facts only this document holds. It must not use "Can a student skip
QT 02?": the model misreads the table group and answers wrongly. The earlier headline screenshot compared plain search
with the agent, which does answer it correctly because code, not the model, reads the table. Examples chosen after
seeing answers must be captioned as illustrations, next to the measured 66% for Base + RAG across 110 questions.
