# Extensions: demo

The demo is the offline talk page: five question buttons, "Base model" against "Base model + RAG", and one line with
the measured score. Ideas for anyone who forks this repo. Pick one, open an issue so two people do not do the same,
and send a pull request. Rules: never invent a number, keep it offline, and show the measured score, not a claim.
See `CONTRIBUTING.md` in the FrugalAI repo, which holds the projects this page shows.

Size: small is an evening, medium is a weekend, large is a few weekends.

## Your own talk (small)
- Replace the five questions and the notes with your own subject. The five questions are fixed lists in `demo_lib.py` and the retrieval comes from the `quantum-wiki` project, so this takes code changes. A good first contribution is moving the questions and the source into one data file.

## Pick another small model (small)
- The demo runs on Qwen3 1.7B and nomic-embed in LM Studio. Switch the model, rerun the trial, and compare the scores in `ANALYSIS.md`.

## Shared page shell (medium)
- `app.py` and `app_faq.py` repeat the layout. Factor out a small shell so a new page needs only its content. The author has a design decision pending in `docs/upgrade-report.md` of the FrugalAI repo, so open an issue first.

## Marathi and router panels (medium)
- The earlier three-section page (Marathi answers, router) is kept in `_backup/`. Bring it back as a separate page with measured scores.

## Run on a weak laptop (small)
- Time the five questions on an old laptop or a Pi with a stated model and log the CSV. This tells a talk audience what "frugal" really costs.

## Slides from the screenshots (small)
- `capture.py` writes five PNGs of the bare page. Script the step that places them in a deck so the figures stay current when the data changes.
