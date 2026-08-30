# /project:git — Git Workflow Command

Automates conventional commits, branch management, and MR creation.

## Usage
```
/project:git commit
/project:git branch <name>
/project:git mr
```

## Behavior

### `commit`
1. Run `git diff --staged` to inspect staged changes.
2. If nothing is staged, run `git diff` and ask which files to stage.
3. Draft a Conventional Commit message following this format:
   ```
   <type>(<scope>): <short description>

   [optional body: what changed and why]

   [optional footer: BREAKING CHANGE, closes #issue]
   ```
   Valid types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`
4. Show the drafted message and wait for approval before committing.
5. After approval: `git commit -m "<message>"`

### `branch`
1. Create a branch with the naming convention: `<type>/<short-description>`
   Example: `feat/add-history-export`, `fix/stream-timeout`
2. Switch to the new branch automatically.

### `mr`
1. Show current branch and commits since branching from main.
2. Draft a merge/pull request description with:
   - Summary of changes
   - Testing done
   - Any breaking changes
3. If GitHub MCP is available, create the PR via MCP. Otherwise print the description for manual use.

## Notes
- Never use `--force` or `--no-verify` flags.
- Always verify `git status` is clean before committing.
