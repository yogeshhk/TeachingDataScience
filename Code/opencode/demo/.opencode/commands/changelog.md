description: >
	Update CHANGELOG from recent git commits
agent: docs
---
Run `git log --oneline --since="$ARGUMENTS"` to get
recent commits. Update CHANGELOG.md with a new
version section grouping commits by type:
feat->Added, fix->Fixed, refactor->Changed, docs->Docs.
Suggest a semantic version bump based on the changes.