# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

UIGen is an AI-powered React component generator. Users describe a component in chat; Claude generates JSX/TSX files into an in-memory virtual file system; the UI renders a live preview in an iframe using Babel + ES module import maps — no files ever touch disk.

## Commands

```bash
npm run setup        # install deps + prisma generate + migrate (first-time)
npm run dev          # start dev server with Turbopack (localhost:3000)
npm run build        # production build
npm run lint         # ESLint
npm test             # vitest (watch mode)
npx vitest run       # vitest single run
npx vitest run src/lib/__tests__/file-system.test.ts  # single test file
npm run db:reset     # drop and re-run all migrations (destructive)
npx prisma studio    # GUI for SQLite data
```

Dev server on Windows requires `NODE_OPTIONS=--require ./node-compat.cjs` (already wired into the npm scripts via `node-compat.cjs`).

## Environment

`.env` needs only one key:
```
ANTHROPIC_API_KEY=sk-ant-...
```
Without it the app falls back to `MockLanguageModel` in `src/lib/provider.ts`, which returns static demo components. `JWT_SECRET` is optional (defaults to `"development-secret-key"`).

## Architecture

### Request Flow

1. User sends a chat message from the UI
2. `POST /api/chat` (`src/app/api/chat/route.ts`) — reconstructs `VirtualFileSystem` from serialized JSON sent by the client, calls Vercel AI SDK `streamText` with two tools, streams the response back
3. Claude calls tools to mutate the VFS:
   - `str_replace_editor` (`src/lib/tools/str-replace.ts`) — create/view/edit files via string replacement or insert
   - `file_manager` (`src/lib/tools/file-manager.ts`) — rename/delete/list files
4. After streaming finishes, the updated VFS + messages are saved to Prisma (`Project.data` + `Project.messages` as JSON strings)
5. The client re-renders the preview iframe using the new VFS state

### Live Preview

`src/lib/transform/jsx-transformer.ts` is the client-side preview engine:
- Transforms JSX/TSX with **Babel Standalone** (`@babel/standalone`)
- Builds a browser **ES Module Import Map** that maps file paths to `blob:` URLs
- Third-party packages (anything not starting with `.`, `/`, or `@/`) are auto-resolved via `https://esm.sh/<package>`
- The `@/` import alias in generated code maps to VFS root `/`
- CSS files are collected and injected as a `<style>` block
- Missing local imports get a placeholder module stub
- The entry point is always `/App.jsx`

### Virtual File System

`src/lib/file-system.ts` — `VirtualFileSystem` class:
- In-memory tree of `FileNode` objects (`type: "file" | "directory"`)
- `serialize()` → `Record<string, FileNode>` (plain object, JSON-safe) — sent to client and stored in DB
- `deserializeFromNodes()` — reconstructs from serialized form
- Key operations: `createFile`, `updateFile`, `deleteFile`, `rename`, `replaceInFile`, `insertInFile`, `viewFile`
- A module-level singleton `fileSystem` is exported but the API route always creates a **fresh instance** per request from the client payload

### AI Model

`src/lib/provider.ts` — `getLanguageModel()`:
- Returns `anthropic("claude-haiku-4-5")` when `ANTHROPIC_API_KEY` is set
- Falls back to `MockLanguageModel` (implements `LanguageModelV1`) otherwise
- The mock simulates multi-step tool calls with hardcoded counter/form/card components

### Auth

`src/lib/auth.ts` — JWT sessions via `jose`, stored in `auth-token` httpOnly cookie (7-day expiry). `middleware.ts` protects `/api/projects` and `/api/filesystem` routes. Anonymous users can generate components but cannot save projects.

### Data Model (Prisma / SQLite)

```
User  { id, email, password(bcrypt), projects[] }
Project { id, name, userId?, messages(JSON string), data(JSON string) }
```
`Project.data` is the serialized VFS. `Project.messages` is the full AI conversation history. Both are stored as JSON strings in SQLite text columns.

### System Prompt

`src/lib/prompts/generation.tsx` — key constraints the AI must follow:
- Every project must have `/App.jsx` as the entry point with a default export
- Use Tailwind CSS (not inline styles)
- Import local files with the `@/` alias (e.g. `@/components/Button`)
- No HTML files — the VFS is a pure JS/JSX/CSS environment

### Server Actions

`src/actions/` — Next.js Server Actions for project CRUD (`create-project.ts`, `get-project.ts`, `get-projects.ts`). These are thin wrappers around Prisma that enforce `userId` ownership checks.

## Testing

Tests live in `src/lib/__tests__/`. Uses `vitest` + `jsdom` + `@testing-library/react`. Run all with `npm test`; run a single file by passing its path to `npx vitest run`.
