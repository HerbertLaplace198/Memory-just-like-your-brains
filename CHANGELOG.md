# Changelog

## 1.0.5 - 2026-07-22

- Follow L3 → L2 episode → L1 routes while compiling topic-page memory links, preventing confirmed episodic memories and evidence from becoming isolated Obsidian nodes.
- Verify automatic Obsidian recompilation after a successful review submission.
- Expand regression coverage from 40 to 41 tests.

## 1.0.4 - 2026-07-22

- Move rejected L1 records and unshared evidence together into hidden, backup-safe `vault/.rejected/` storage instead of leaving active-vault residue.
- Add recoverable `restore-rejected` and `archive-orphan-evidence` commands.
- Detect unreferenced evidence in health checks.
- Expand regression coverage from 38 to 40 tests.

## 1.0.3 - 2026-07-22

- Prune L3 semantic topics, L4 procedures, and their synapses when rejecting an L1 memory leaves them unreachable from every active L1.
- Do not regenerate upper layers for rejected or archived canonical records during index rebuilds.
- Require English-only L3 topic and L4 procedure labels for new records; skip legacy non-English structural labels during rebuilds.
- Expand regression coverage from 35 to 38 tests.

## 1.0.2 - 2026-07-22

- Refresh the generated Obsidian view immediately after MCP or lifecycle-hook proposals so new canonical records do not appear as temporary orphan nodes.
- Link every proposed L1 memory and its evidence from the Obsidian maintenance center so the human-review queue is directly reachable.
- Add explicit Obsidian checkboxes for Confirm, Needs revision, and Incorrect/reject; `sync-obsidian` applies only a single unambiguous human selection.
- Add a desktop Obsidian plugin with a maintenance-page submit button, ribbon action, and command-palette action. Review choices are applied only after the user explicitly submits them.
- Expand regression coverage from 33 to 35 tests.

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
