# Security Policy

This is a teaching content repo (LaTeX slide decks and example code), not a software product,
so there's no versioned release to patch. That said, security reports are still welcome for
things like:

- An example script in `Code/` that encourages an insecure practice (a hardcoded secret,
  an unsafe deserialization pattern, a vulnerable dependency pin).
- A credential or personal data accidentally committed anywhere in the repo.

## Reporting

Please don't open a public issue for a live credential leak or an actively exploitable example.
Instead, open an issue with as much detail as you're comfortable sharing, or reach out via the
contact link in the [README](README.md), and it will be looked at and fixed as soon as possible.

For anything else (a typo, a stale reference, a broken code example with no security angle),
see [`CONTRIBUTING.md`](CONTRIBUTING.md) instead.
