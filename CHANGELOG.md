# Changelog

## 1.0.1 - 2026-07-22

- Canonicalize L3 topic aliases and suppress structural pseudo-topics.
- Scope recall to the strongest L3 route; continuation queries may include its parent route.
- Restrict Obsidian topic narratives to direct active L1 members.
- Add canonical memory/evidence links and an auditable archive page.
- Hide L6 routing nodes from Obsidian while retaining them in SQLite.
- Remove stale generated topic pages and normalize case-only filename changes.
- Preserve upper-node status precedence during indexing.
- Expand regression coverage from 26 to 33 tests.
- Expand semantic A/B evaluation from 12 to 13 cases, gate neural awareness on direct evidence instead of spread activation, and calibrate the English Qwen sample threshold to 0.55.
- Add a complete English installation and operations guide in `USAGE.md`.

## 1.0.0 - 2026-07-21

- Initial standalone release with L0-L6 memory, staged retrieval, human review, MCP, lifecycle hooks, Obsidian views, local embeddings, locking, backup and verified restore.
