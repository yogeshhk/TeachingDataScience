# /project:explain — Codebase Flow Explainer

Maps a request or concept through the codebase with specific file and line references.

## Usage
```
/project:explain <question or flow to trace>
```

Examples:
- `/project:explain How does a user message get from the CLI to the LLM?`
- `/project:explain How is session history managed in the Streamlit app?`
- `/project:explain What happens when the Groq API call times out?`

## Behavior

1. **Identify the entry point** relevant to the question (CLI entrypoint, Streamlit callback, API route).
2. **Trace the call chain** step by step through the codebase, referencing:
   - Exact file paths (relative to project root)
   - Line numbers for key logic
   - Function/method signatures
3. **Explain each step** in plain language: what the code does and why.
4. **Highlight decision points:** conditionals, error handlers, retries, transformations.
5. **End with a summary diagram** (ASCII) of the full flow if the chain is more than 4 hops deep.

## Output Format
```
Entry: src/mychatbot/cli.py:42 — chat_loop()
  → src/mychatbot/chain.py:18 — build_chain()
      Constructs the LangChain pipeline with system prompt and Groq model
  → langchain_groq.ChatGroq.invoke()
      Sends tokenized messages to Groq API endpoint
  → src/mychatbot/cli.py:55 — print_response()
      Renders streamed tokens via Rich console
```

## Notes
- Uses `Read`, `Grep`, `Glob` tools only — no code changes.
- Delegate to the `explore-codebase` subagent for large codebases to keep context lean.
