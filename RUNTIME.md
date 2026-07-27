# Runtime release contract

The running MCP application must be a normal Git checkout of the one public
repository, on `main` or an explicitly checked-out release tag. Do not copy
individual Python files into a separate application directory.

## Required layout

- `app/` is the runnable Git checkout and contains `neural_memory.py`,
  `mcp_server.py`, and all `test_*.py` files.
- The memory root is separate from the checkout. Its canonical human records
  remain Markdown; its only live SQLite index is `<memory-root>/memory.sqlite3`.
- `<memory-root>/.neural-memory/index.sqlite` is legacy-only. It is preserved
  and recorded in `legacy-index.sqlite.json`; it is never selected as the live
  database.

## Release procedure

1. Run the complete test suite in the Git checkout.
2. Commit and tag `main`.
3. Update the runtime checkout to that exact tag or `main` commit.
4. Run the same test suite in `app/` and verify `python3 neural_memory.py
   --root <memory-root> doctor` reports `memory.sqlite3` as canonical.

The runtime checkout must stay clean. Local experiments belong in a separate
working directory or a tagged release, not in the live application directory.
