# CLAUDE.md

This file provides guidance to AI assistants (Claude and others) working in this repository.

## Project Overview

**CC-cluster-analysis** is a cluster analysis project currently in its initial setup phase. The repository was initialized with a single placeholder README and has no source code yet.

- **Repository**: `edloessime/CC-cluster-analysis`
- **Remote**: Configured via a local git proxy at `127.0.0.1:55398`
- **Primary branch**: `master`

## Repository Structure

```
CC-cluster-analysis/
└── README.md       # Project title placeholder
```

The project has no source code, dependencies, tests, or tooling configured yet. All foundational decisions (language, framework, architecture) remain open.

## Git Workflow

### Branch Conventions

- `master` — stable, production-ready code
- `claude/<description>-<session-id>` — AI assistant development branches (e.g., `claude/claude-md-mlvdyp991k3opfih-hZlbJ`)
- Feature branches should be short-lived and merged via pull request

### Commit Messages

Use the imperative mood and be descriptive:

```
Add cluster distance matrix computation
Fix off-by-one error in centroid initialization
Refactor silhouette scoring into separate module
```

Avoid vague messages like `fix`, `update`, or `changes`.

### Push Instructions

Always push with upstream tracking:

```bash
git push -u origin <branch-name>
```

Branch names for AI sessions must start with `claude/` and end with the matching session ID, or push will fail with HTTP 403.

## Development Setup

No tooling is configured yet. When setting up the project, document:

1. **Language & runtime** — add version files (`.python-version`, `.nvmrc`, etc.)
2. **Dependency management** — add `pyproject.toml`, `package.json`, `Cargo.toml`, etc.
3. **Environment setup** — document `venv`, `conda`, or container-based setup
4. **Build commands** — document how to build/compile the project
5. **Run commands** — document how to start/execute the project

Update this file with those details once they are established.

## Testing

No test framework is configured. When tests are added:

- Document the test runner (pytest, jest, cargo test, etc.)
- Document how to run the full test suite
- Document how to run a single test file or test case
- All new features should include corresponding tests
- Tests must pass before merging to `master`

## Code Conventions

Until a language and style guide are chosen, follow these general principles:

- **Readability first** — prefer clear, self-documenting code over clever one-liners
- **Small functions** — each function/method should do one thing
- **No dead code** — remove unused variables, imports, and functions immediately
- **No magic numbers** — use named constants
- **Consistent naming** — pick a convention (snake_case, camelCase, PascalCase) and apply it uniformly within the chosen language's norms

## For AI Assistants

### Before Making Changes

1. Read all relevant files before editing — never modify code you haven't read
2. Check for existing conventions in any code present before adding new code
3. Understand the full scope of a change before starting implementation

### When Adding Code

- Keep changes focused and minimal — only implement what is explicitly requested
- Do not add extra features, abstractions, or refactors beyond the task scope
- Do not add docstrings, comments, or type annotations to code you did not write
- Do not introduce unnecessary abstractions for one-off operations

### When Editing Existing Code

- Prefer editing existing files over creating new ones
- Do not rename or reorganize files unless explicitly asked
- Do not add backwards-compatibility shims for removed code — delete cleanly

### Security

- Never commit secrets, credentials, API keys, or tokens
- Validate input at system boundaries (user input, external APIs)
- Do not add `.env` files containing real secrets to version control

### File Operations

- Never create files unless strictly necessary to accomplish the task
- Prefer modifying existing files over creating new ones
- Never create README, documentation, or markdown files unless explicitly requested

## Updating This File

Keep this file current as the project evolves. Update it when:

- A programming language or framework is chosen
- Build, lint, or test commands are established
- New architectural patterns or conventions are adopted
- New dependencies with non-obvious usage are added
- Development workflows change significantly
