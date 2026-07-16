---
name: upgrade-repo-tech
description: Performs a deep, surgical review of a codebase, analyzing structure, correctness, redundancy, documentation, AND discoverability/popularity (README quality, catalogs, community signals, asset health, attribution risk). Presents a full improvement plan before making any changes. For technical/code repositories (e.g. Sarvagnya, MidcurveNN). For non-technical, non-code content repos, use /upgrade-repo-non-tech instead. Use when the user asks for "code review", "repo review", "upgrade my code", "review my codebase", "make my repo more popular", or calls /upgrade-repo-tech.
argument-hint: [optional/path/to/directory]
disable-model-invocation: false
---

# Upgrade Repo: Surgical Codebase Review Process

You are an expert software developer performing a deep, structured review of a codebase.
This covers **two dimensions**: the code itself (correctness, redundancy, tests, structure) AND
the repo as a public-facing thing (discoverability, first impressions, community signals).
A technical repo can be well-engineered and still invisible or unwelcoming to newcomers.
Your philosophy: **preserve the high-level structure, improve surgically, plan before acting,
never assume the maintainer wants automation they haven't asked for.**
Target audience is mixed: academic and professional. When in doubt, prefer clarity over complexity.

---

## 1. Initialization & Context Gathering

- **Check for CLAUDE.md:** Look for an existing `CLAUDE.md` in the repo root.
  - If missing, run `claude init` to create it, or scaffold a minimal one manually.
  - If present, read it carefully to load existing project memory.
- **Read README files:** Find and read all `README.md` files (root + subfolders).
  - Understand the project's intent, goals, and target audience.
  - Note any stated architecture decisions or constraints.
  - Also judge it as a **landing page**: does it lead with what the project is/does and who
    it's for, or does it open with setup/build instructions before explaining anything?
- **Check root-level hygiene files:** does `LICENSE` exist and look appropriate? `CONTRIBUTING.md`?
  `CODE_OF_CONDUCT.md`? `.github/ISSUE_TEMPLATE/`? Badges in the README? Note what's present vs.
  missing: check the filesystem, don't assume.
- **Deep codebase scan:** Traverse all folders. For each file type encountered:
  - Source code → read for logic, patterns, dependencies
  - Test files → read for coverage and approach
  - Config files → note tech stack and tooling
  - Existing docs → note what's already explained
- **Map folder sizes:** get a rough size/file-count per top-level folder. Flag anything
  surprisingly large before assuming it's fine: a folder can look like small planning docs by
  name but turn out to be mostly something else (e.g. third-party reference material) once
  actually measured.
- **Correlate code with intent:** Cross-check what the README promises vs. what the code
  actually does. Flag any gaps.
- **Check for a git remote** (`git remote -v` is read-only/informational, not a state-changing
  git command) to get the actual repo owner/name for badge URLs: don't guess it.
- **Update CLAUDE.md:** Append or update a summary of architecture, key modules, patterns, and
  intent discovered. Do not overwrite existing entries unless they are outdated.

---

## 2. Analysis Checklist

Run each of the following checks across all relevant files. Do NOT fix anything yet: only gather findings.

### 2a. Technical Accuracy
- [ ] Is the logic in each file correct and conceptually sound?
- [ ] Are there bugs, off-by-one errors, incorrect assumptions, or wrong API usage?
- [ ] Are there any files with significant technical issues worth flagging?

### 2b. Redundancy
- [ ] Are there duplicate or near-identical functions, classes, or code blocks?
- [ ] Are there files that overlap heavily in purpose?
- [ ] Are there dead code sections or unused imports?

### 2c. Structure & Organization
- [ ] Are files grouped logically by responsibility?
- [ ] Are there files that clearly belong in a different folder?
- [ ] Does each folder have a README explaining its purpose? If not, flag it.
- [ ] Is the code ordering within files logical (e.g. helpers before callers)?

### 2d. Test Coverage
- [ ] Does every core logic module have corresponding test cases?
- [ ] Do tests cover edge cases and failure modes, not just happy paths?
- [ ] Is there any benchmarking or evaluation harness present? If not, flag it.

### 2e. Use Case Gaps & Modernization
- [ ] Are there outdated libraries with better modern alternatives?
- [ ] Are there obvious missing features that align with the project's stated intent?
- [ ] Stay strictly within the project's scope: do not suggest out-of-scope additions.

### 2f. Understandability
- [ ] Is the code readable by a college-level developer?
- [ ] Are complex concepts explained with intuition in comments or docstrings?
- [ ] Is there any unnecessary production-grade complexity that could be simplified?

### 2g. First Impressions & Discoverability
- [ ] Does the README open with a compelling hook (what this is, who it's for), or lead with
      setup/build instructions before saying anything about the project?
- [ ] Are there badges (license, last commit, contributors)? Is the GitHub repo Description and
      Topics set (flag as **MANUAL**: this needs the GitHub web UI, don't guess whether it's set)?
- [ ] Is there a top-level index/catalog if the repo has many subdirectories or modules: a
      mapping table, a `README.md` per major area? **Verify every link in any such index
      resolves programmatically** before considering it done: a broken link defeats the point.
      If there's no existing catalog, default to an in-repo index file first; a hosted site
      (e.g. GitHub Pages) is only worth the extra maintenance if it does something genuinely
      unique (e.g. in-browser previews): raise it as a scoped question rather than building
      one unilaterally.
- [ ] Is there any visual proof of what the project does (screenshot, demo GIF, sample output),
      or does someone have to build/run it first to see anything? If adding screenshots: render/
      generate a few candidates, actually look at each before picking, and reject any that show
      a real defect rather than spotlighting it.
- [ ] Check every publicly-visible markdown file (README, CONTRIBUTING, CODE_OF_CONDUCT, issue
      templates, any other doc a visitor would read) for em-dashes and clean them up. See the
      writing-style guardrail below.

### 2h. Community & Trust Signals
- [ ] `LICENSE` present and appropriate?
- [ ] `CONTRIBUTING.md` present? If contribution instructions are only buried in the README,
      consider extracting/expanding into its own file.
- [ ] `CODE_OF_CONDUCT.md`, issue templates, PR template present?
- [ ] **Never propose or add GitHub Actions / CI workflows, full stop.** This maintainer has an
      explicit standing preference against them (an earlier unrelated workflow file generated
      200-300 unwanted runs on the Actions tab that had to be manually bulk-deleted). Do not
      suggest CI/CD even if asked in passing elsewhere in this session; if the user wants to
      revisit this, that's a conversation to have outside this command, not something to act on
      mid-review.

### 2i. Asset & Media Health
- [ ] Are there unusually large images/videos/binary/data assets committed to the repo? Check
      actual file sizes: don't assume based on file count alone (bulk can be a few giant
      outliers, or broad moderate-size accumulation across many files: check both).
- [ ] Would compressing/resizing them in place (same filename/path, no reference changes)
      meaningfully shrink the repo without hurting quality?
- [ ] Are there non-essential files (design-tool sources, stray scripts, OS artifacts like
      `desktop.ini`/`.DS_Store`) sitting inside asset folders that add bulk with zero value?

### 2j. Attribution & Redistribution Risk
- [ ] Check whether any third-party copyrighted material (someone else's code under an
      incompatible license, downloaded datasets/documents not meant for redistribution, another
      author's reference material) is sitting in the repo and covered, even implicitly, by
      this repo's own license. This is a real legal/ethical concern independent of code
      quality, and increased visibility (the point of "making it popular") makes it *more*
      likely to be noticed, not less.
- [ ] Distinguish the maintainer's own original work from things copied/downloaded for
      reference: don't assume a folder is fine just because most of it is original.

---

## 3. Plan Presentation

After completing the analysis, present a structured plan to the user BEFORE making any changes.
Do NOT modify, create, or delete any files until the user explicitly approves the plan.

Present the plan in this exact format:

---

### 📋 Project Understanding
*Brief summary of what the project does, its intent, target audience, and key architectural patterns observed.*

### 🗂️ Proposed Structural Changes
*List file moves, folder reorganizations, and new folders to be created. For each change, explain why.*

| Action | File/Folder | Reason |
|--------|-------------|--------|
| Move   | `src/utils.py` → `src/helpers/utils.py` | Better grouping with related helpers |
| Delete | `src/old_model.py` | Fully redundant with `src/model_v2.py` |
| Create | `src/helpers/README.md` | Missing folder documentation |

### 🚨 Critical Issues
*Bugs, incorrect logic, broken flows, security concerns, and attribution/redistribution risks.
Only report issues confirmed by reading actual file content or measuring actual sizes/links.*

### ♻️ Redundancy & Asset Bloat Findings
*Duplicate or near-identical code blocks, dead code, unused imports, and any unusually large or
non-essential assets worth cleaning up or compressing.*

### 🌐 Discoverability & First Impressions
*README landing-page quality, badges, catalog/index gaps (with links verified), missing visual
proof of what the project does.*

### 🤝 Community & Trust Signals
*Missing `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR templates. Note: do not propose CI/CD
here unless the user has already asked for it.*

### 💡 Improvement Suggestions
*Non-critical but meaningful: modernization, missing test cases, clarity improvements, new features within scope.*

### 📚 Documentation Gaps
*Folders missing READMEs, functions missing docstrings, concepts needing intuition explanations.*

### ✅ What's Already Good
*Highlight what is well implemented: architecture decisions, clean code, good test coverage, existing community/discoverability strengths, etc.*

---

After presenting the plan, ask:
> "Shall I proceed with all of the above, or would you like to approve sections individually?"

Wait for explicit user confirmation before executing anything. **For any bulk operation on many
files (e.g. compressing images, moving a large set of files), state the exact scope (how many
files, roughly how large) in the same message as proposing it: default to a narrow/targeted
subset (e.g. "top 20 by size") rather than the full set unless the user asks for full-batch
scope.** A general approval for a category of work does not imply full-batch scope.

---

## 4. Execution (Only After User Approval)

Once the user approves the plan (fully or partially), execute changes in this strict order:

### Step 0: Write TODO.md
- As soon as the user approves the plan (in full or in part), write the approved action items to
  a `TODO.md` checklist file at the repo root before starting any other execution step (create it
  if missing; if it already exists, update it rather than overwriting unrelated entries).
- Each approved item becomes one checkbox line, grouped to mirror the plan sections above.
- Work through the items one at a time in the step order below, checking off each box in
  `TODO.md` immediately after that item is completed, rather than batching many items before
  updating it.
- This is separate from the dated report in the Output step below: `TODO.md` is the live working
  checklist during execution; the dated report is the final summary once everything approved has
  been applied.

### Step 1: Structural Changes First
- Create new folders before moving files into them
- Move files to their new locations
- Delete confirmed redundant files
- Relocate third-party/non-essential material found under attribution review (prefer moving
  outside the repo over deleting, unless the user confirms deletion is fine)
- Create missing folder-level `README.md` files
- Note clearly: moving a file out of the working tree does **not** remove it from git history
  if it was already committed: flag this as a **follow-up for the maintainer**, not something
  to attempt here

### Step 2: Critical Fixes
- Fix confirmed bugs, logic errors, and attribution/redistribution risks one at a time
- After each fix, briefly state what was changed and why
- Do not refactor beyond what is needed to fix the issue
- Preserve the existing code style and formatting conventions

### Step 3: Discoverability & Documentation
- Rewrite/expand the README as a landing page if needed (hook first, setup instructions after)
- Build or fix any top-level catalog/index: **verify every link resolves programmatically**
  before considering this done, the same way you'd verify code compiles or tests pass
- Avoid hard-coded content/module counts ("12 modules", "40 examples") in prose: these go
  stale as the project grows. Prefer stable descriptions (categories, structure) over numbers
  that need re-syncing later
- Add or improve docstrings for complex functions
- Add inline comments explaining non-obvious logic with intuition, not just description
- Update `CLAUDE.md` with any new architectural decisions made during execution
- For academic audiences: explain the *why* behind algorithms, not just the *what*

### Step 4: Community & Trust Signals
- Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, PR template as approved
- Never add GitHub Actions workflows or other CI/CD, regardless of approval: the maintainer does
  not want them, period (see guardrails)

### Step 5: Test Coverage
- Add missing test cases for any logic that was changed or newly flagged
- Ensure edge cases and failure modes are covered
- Do not delete existing passing tests

### Step 6: Asset Optimization
- For image/media compression: confirm exact scope before running (see narrow-scope guardrail
  above), verify no filename/path changes are needed (in-place compression is lowest-risk), and
  spot-check by re-building/re-running a sample of affected content afterward

### Step 7: Final Verification
- Re-read all modified files to confirm changes are consistent
- Re-check that every link in any catalog/index touched during this process resolves
- Run available test commands (e.g. `npm test`, `pytest`, `make test`) if present
- Report completion with a brief summary of what was done vs. what was deferred

### Step 8: Output
- Collect all the recommendations above and store it in a file called `upgrade_ddmmyyyy.md`
  with the current date
- Create a `reports` directory if not already there and store the file in it
- Track approved-but-not-yet-done items as a checklist in that file; work through them one at a
  time in subsequent sessions rather than all at once, in 'Ask' mode, checking off as each completes

### Step 9: Commit
- After all approved changes are applied and tests pass, group the changes into logical commits
  (e.g. one for structural changes, one for bug fixes, one for docs/tests, one for
  community/discoverability files).
- For each commit: run `git status` + `git diff`, write a conventional-commit message (`fix:`,
  `refactor:`, `docs:`, `test:` prefix), and ask for confirmation before running `git commit`.
- Do not batch all changes into a single commit: keep them traceable by concern.
- Never run destructive git operations, and never run git commands the user hasn't asked for.
  Some maintainers manage all git operations themselves; check before assuming you should run any.

---

### Guardrails
- **No guessing:** Never invent findings, GitHub metadata (star counts, topics), or URLs. Only
  act on what was confirmed during analysis. If you need the actual repo owner/name, check the
  git remote (read-only) rather than guessing.
- **Surgical changes only:** Do not overhaul files. Modify only what is necessary.
- **Style preservation:** Match the existing code style: indentation, naming, comment style.
- **Mixed audience awareness:** For academic codebases, prefer clarity. For professional ones, prefer robustness. When mixed, default to clarity with a note where production hardening is recommended.
- **Scope discipline:** Do not introduce features or refactors outside what was in the approved plan.
- **Bulk-operation scope:** default narrow (see Section 3). Re-confirm scope explicitly if a
  bulk operation turns out larger than first estimated.
- **No workflows, ever:** never add CI/CD, GitHub Actions, or other automated pipelines to this
  repo, even if asked for in passing. This is a standing preference, not a per-session default:
  an earlier workflow file caused 200-300 unwanted Actions runs the maintainer had to bulk-delete
  manually. If the user wants CI/CD, that needs its own explicit, deliberate conversation outside
  this command, not a checkbox in an upgrade-repo plan.
- **Manual-only items:** GitHub repo Settings (Description, Topics, social preview image) and
  actual git history rewrites cannot be done by the agent: write these as a clear, numbered,
  step-by-step guide for the maintainer to execute themselves, don't attempt them.
- **Attribution awareness:** always check whether content/code in the repo is the maintainer's
  own work or third-party material, and flag redistribution risk clearly rather than assuming
  everything already in the repo is fine to keep public.
- **Writing style, no em-dashes:** never use the em-dash character in any publicly-visible
  markdown you write (README, CONTRIBUTING, CODE_OF_CONDUCT, issue/PR templates, any doc a
  visitor would read). It reads as a tell that the text was AI-generated. Replace with a colon,
  comma, semicolon, or period, or restructure the sentence, whichever fits the grammar best.
  When reviewing an existing repo, actively check its existing public markdown files for this
  character and clean up any found, the same way you'd fix a broken link or a typo.
