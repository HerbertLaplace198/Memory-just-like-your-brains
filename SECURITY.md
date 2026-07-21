# Security and privacy

Neural Memory stores readable Markdown plus a rebuildable SQLite retrieval index. Treat the entire memory root and every `.nmem` backup as private user data.

- Keep real memory roots, backups, generated Obsidian views and local `encoder.json` files out of Git.
- The bundled `.gitignore` excludes these paths by default; do not weaken it without reviewing the affected files.
- Embedding HTTP endpoints are restricted to loopback hosts. The project does not provide a cloud embedding mode.
- Markdown is not encrypted by this application. Use full-disk encryption or an encrypted volume.
- Before publishing a sample bundle, inspect all Markdown inside it and scan both extracted files and Git history for secrets, usernames and absolute paths.

Security reports should be sent privately to the repository owner rather than opened with sensitive details in a public issue.
