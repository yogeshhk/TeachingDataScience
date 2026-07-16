---
name: upgrade-repo-non-tech
description: Performs a deep review of a non-technical, non-code content repository, analyzing discoverability, first impressions, organization, community signals, and asset health. Presents a full improvement plan before making any changes. For content repos with little or no code (e.g. Bharat Vidya). For technical/code repositories, use /upgrade-repo-tech instead. Use when the user asks to "make my repo more popular", "improve discoverability", "review my content repo", or calls /upgrade-repo-non-tech.
argument-hint: [optional/path/to/directory]
disable-model-invocation: false
---

# Upgrade Repo (Non-Tech): Content Repository Discoverability Review

You are reviewing a public repository whose primary value is **content, not code**: course
material, documentation, cultural/knowledge archives, slide decks, datasets, or similar. The
goal is **popularity and discoverability**: make it easier for a stranger to find, trust, and
sample the repo, without touching the substance of the content itself.
Your philosophy: **present a plan before acting, keep scope narrow by default, never assume the
maintainer wants automation they haven't asked for.**

---

## 1. Initialization & Context Gathering

- **Check for CLAUDE.md:** read it if present: it may already document structure/conventions.
- **Read the root README** (and any subfolder READMEs): understand what the repo actually
  contains, who it's for, and what it currently tells (or fails to tell) a first-time visitor.
- **Check root-level hygiene files:** does `LICENSE` exist? `CONTRIBUTING.md`? `CODE_OF_CONDUCT.md`?
  `.github/ISSUE_TEMPLATE/`? Note what's present vs. missing: don't assume; check the filesystem.
- **Map the folder structure:** get a top-level listing, then a rough size/file-count per
  top-level folder (`du -sh`, file counts). Flag anything surprisingly large before assuming
  it's fine: a folder named for small planning docs may turn out to be mostly something else
  entirely (this happened with an `Admin/` folder that looked like pamphlets but was 96%
  third-party PDFs).
- **Identify the content hierarchy**, if one exists (e.g. course → workshop → seminar, or
  chapter → section). Note whether it's already documented anywhere a visitor could find it,
  or only implicit in file naming.
- **Check for a git remote** (`git remote -v` is a read-only informational command, not a
  state-changing one) to know the actual repo owner/name for badge URLs: don't guess it.

---

## 2. Analysis Checklist

Run each check across the repo. Do NOT fix anything yet: only gather findings.

### 2a. First Impressions
- [ ] Does the README open with a compelling hook (what this is, who it's for), or does it lead
      with build/setup instructions before saying anything about the content?
- [ ] Are there badges (license, last commit, contributors)? A repo Description and Topics set
      on GitHub (note: these require the GitHub web UI: flag as **MANUAL**, don't guess if set)?
- [ ] Is there any visual proof of content quality (screenshots, sample pages, a rendered
      preview), or does a visitor have to build/compile/download something just to see one page?
      If adding screenshots: render/generate a few candidates, actually look at each one before
      picking, and reject any that show a real defect (a layout bug, a rendering glitch) rather
      than spotlighting it. A bad screenshot undercuts the pitch worse than no screenshot.
- [ ] Check every publicly-visible markdown file (README, CONTRIBUTING, CODE_OF_CONDUCT, issue
      templates, any other doc a visitor would read) for em-dashes and clean them up. See the
      writing-style guardrail below.

### 2b. Discoverability & Navigability
- [ ] Is there a catalog/index of everything in the repo, or does a visitor have to guess from
      raw file listings? If the repo is large, is the catalog complete or just a description of
      a subset (check systematically: grep/list, don't rely on a pre-existing doc's claims,
      since it may only cover part of what actually exists)?
- [ ] Do links in any catalog/index actually resolve? **Verify every link programmatically**
      before considering a catalog "done": a broken link in a navigability doc defeats its
      purpose.
- [ ] Are naming conventions consistent enough that a newcomer could predict where something
      lives?
- [ ] If there's no existing catalog, consider whether an in-repo `CONTENTS.md`-style index is
      enough, or whether a hosted site (e.g. GitHub Pages) would add something the index can't
      (e.g. in-browser previews of otherwise-uncompiled content). Default to the in-repo index
      first: a hosted site is only worth the extra maintenance if it does something genuinely
      unique; don't build one just to have one, and always raise it as a scoped question to the
      maintainer rather than deciding unilaterally.

### 2c. Structure & Organization
- [ ] Are there folders whose actual contents don't match what their name/README implies?
- [ ] Is there stale, duplicated, or clearly-superseded content sitting alongside current
      material with no indication of which is current?
- [ ] Are internal working/planning materials mixed in with visitor-facing content in a way
      that dilutes the "polished public resource" impression?

### 2d. Community & Trust Signals
- [ ] `LICENSE` present and appropriate?
- [ ] `CONTRIBUTING.md`: does it exist, and if the README has an inline "how to contribute"
      section, should it be extracted there instead?
- [ ] `CODE_OF_CONDUCT.md`, issue templates, PR template: present?
- [ ] **Never propose or add GitHub Actions / CI workflows, full stop.** This maintainer has an
      explicit standing preference against them (an earlier unrelated workflow file generated
      200-300 unwanted runs on the Actions tab that had to be manually bulk-deleted). Do not
      suggest CI/CD even if asked in passing elsewhere in this session; if the user wants to
      revisit this, that's a conversation to have outside this command, not something to act on
      mid-review.

### 2e. Asset & Media Health
- [ ] Are there unusually large images/videos/PDFs? Check actual file sizes: don't assume
      based on file count alone (a repo can have thousands of files with no single outlier and
      still be bloated in aggregate, or have most of its size concentrated in a handful of files).
- [ ] Are there non-content files (design-tool source files, stray scripts, OS artifacts like
      `desktop.ini`, `.DS_Store`) sitting inside content/asset folders where they add no value?

### 2f. Attribution & Redistribution Risk
- [ ] **Check for third-party copyrighted material redistributed under this repo's own
      license.** This is easy to miss and genuinely important: lecture slides from other
      authors, official textbook/curriculum PDFs, another organization's training material, or
      similar, sitting in the repo and covered (even implicitly) by its open-source license,
      is a real legal/ethical concern independent of the popularity angle, and public
      visibility (literally the goal of this whole exercise) makes it *more* likely to be
      noticed, not less.
- [ ] Distinguish clearly between the maintainer's own original material and things they
      downloaded/copied for reference: don't assume everything in a folder is original just
      because the folder also contains original work.

---

## 3. Plan Presentation

After completing the analysis, present a structured plan to the user BEFORE making any changes.
Do NOT modify, create, or delete any files until the user explicitly approves the plan.

Present the plan in this exact format:

---

### 📋 Repo Understanding
*Brief summary of what the repo contains, its intended audience, and its current state as a
public-facing resource (not just a technical description).*

### 🗂️ Proposed Structural Changes
*File moves, folder reorganizations, relocations of non-essential/third-party material. For
each change, explain why, and flag anything with real risk (e.g. touching cross-referenced
paths) as needing its own confirmation before executing.*

| Action | File/Folder | Reason |
|--------|-------------|--------|
| Move   | `Admin/ThirdPartyLectures/` → outside repo | Third-party copyrighted material, redistribution risk |
| Create | `CONTENTS.md` | No existing catalog of what's in the repo |

### 🚨 Critical Issues
*Attribution/redistribution risks, broken catalog links, factual errors, dead external links.
Only report issues confirmed by actually checking: file sizes measured, links tested, not assumed.*

### ♻️ Clutter & Redundancy Findings
*Stale/duplicate content, non-essential files bloating asset folders, folders whose contents
don't match their apparent purpose.*

### 💡 Improvement Suggestions
*First-impression polish, community signals, asset optimization: non-critical but meaningful.*

### 📚 Documentation Gaps
*Missing README sections, missing catalog/index, missing CONTRIBUTING/CODE_OF_CONDUCT.*

### ✅ What's Already Good
*Highlight what's already working: don't manufacture problems where none exist.*

---

After presenting the plan, ask:
> "Shall I proceed with all of the above, or would you like to approve sections individually?"

Wait for explicit user confirmation before executing anything. **For any item involving a bulk
operation on many files (e.g. compressing images, moving a large set of files), state the exact
scope (how many files, roughly how large) in the same message as proposing it: default to a
narrow/targeted subset (e.g. "top 20 by size") rather than the full set unless the user asks for
full-batch scope.** A general approval for a category of work does not imply full-batch scope.

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
- Relocate third-party/non-essential material (prefer moving outside the repo entirely over
  deleting, unless the user confirms deletion is fine)
- Note clearly: moving a file out of the working tree does **not** remove it from git history
  if it was already committed: flag this as a **follow-up for the maintainer** (purging git
  history is their call and their action, not something to do here)

### Step 2: Critical Fixes
- Fix confirmed attribution/broken-link/factual issues one at a time
- State what was changed and why after each fix

### Step 3: Documentation & Discoverability
- Rewrite/expand the README as a landing page if needed
- Build or fix the content catalog/index: **verify every link resolves programmatically**
  before considering this done, the same way you'd verify code compiles
- Avoid hard-coded content counts ("5 courses", "34 items") in any prose: these go stale as
  content is added. Prefer stable descriptions (categories, duration tiers, qualitative scale)
  over numbers that need re-syncing later.

### Step 4: Community & Trust Signals
- Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, PR template as approved
- Never add GitHub Actions workflows, regardless of approval: the maintainer does not want them,
  period (see guardrails)

### Step 5: Asset Optimization
- For image/media compression: confirm exact scope before running (see narrow-scope guardrail
  above), verify no filename/path changes are needed (in-place compression is lowest-risk),
  and spot-check by re-viewing/re-building a sample of affected content afterward

### Step 6: Final Verification
- Re-check that every link added or touched during this process resolves
- Re-read modified files to confirm consistency

### Step 7: Output
- Store the recommendations/plan in a file called `upgrade_nontech_ddmmyyyy.md` with the
  current date
- Create a `reports/` directory at the repo root if not already there, and store the file in it
- Track approved-but-not-yet-done items as a checklist in that file; work through them one at a
  time in subsequent sessions rather than all at once, checking off as each completes

### Step 8: Commit
- After approved changes are applied, group them into logical commits (e.g. one for structural
  changes, one for documentation, one for community files)
- For each commit: run `git status` + `git diff`, write a clear commit message, and ask for
  confirmation before running `git commit`
- Never run destructive git operations, and never run git commands the user hasn't asked for.
  Some maintainers manage all git operations themselves; check before assuming you should run any

---

### Guardrails
- **No guessing:** never invent findings, GitHub metadata (star counts, topics), or URLs. If you
  need to know the actual repo owner/name, check the git remote (read-only) rather than guessing.
- **Surgical changes only:** modify only what's necessary; don't restructure content that isn't broken.
- **Scope discipline:** stay within the plan the user approved. Don't add automation,
  restructuring, or new tooling beyond what was discussed.
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
- **Attribution awareness:** always check whether content in the repo is the maintainer's own
  work or third-party material, and flag redistribution risk clearly rather than assuming
  everything in the repo is fine to keep public just because it's already there.
- **Writing style, no em-dashes:** never use the em-dash character in any publicly-visible
  markdown you write (README, CONTRIBUTING, CODE_OF_CONDUCT, issue/PR templates, any doc a
  visitor would read). It reads as a tell that the text was AI-generated. Replace with a colon,
  comma, semicolon, or period, or restructure the sentence, whichever fits the grammar best.
  When reviewing an existing repo, actively check its existing public markdown files for this
  character and clean up any found, the same way you'd fix a broken link or a typo.
