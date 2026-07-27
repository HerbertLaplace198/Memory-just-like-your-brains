# Changelog

## 1.5.0 - 2026-07-27

- Give larger confirmed L3F families a bounded routing bonus based on member
  L3 count and unique active L1 support; keep the family relevance gate and
  member L3 scoring unchanged.
- Move semantic review, concept identity, duplicate-decision, and L3F family
  records into hidden `<root>/.neural-memory/` backend storage.
- Require English structural labels for new memory writes while preserving the
  original language of memory bodies; normalize known Chinese aliases before
  indexing.
- Improve multilingual L3F triggering with canonical English aliases and add
  the separately calibrated `family_gate_threshold` setting.
- Preserve compatibility with v1.4 encoder configurations: omitted
  `family_size_bonus_cap` defaults to `0.08`, and existing memory stores need
  no migration.
- Expand regression coverage to 62 automated tests.

## 1.4.0 - 2026-07-26

- Promote confirmed L3F families into the first-stage semantic router.
- Search matched family members and independent L3 concepts together.
- Suppress individual competition from unselected family members.
- Add a full-L3 safety fallback when family and independent routes are weak.
- Aggregate shared L4/L5 family relationships and expose L3F routing in MCP.
- Expand regression coverage to 58 automated tests.

## 1.3.0 - 2026-07-26

- Add the non-renumbering L3F concept-family layer.
- Group related L3 concepts without merging or deleting them.
- Preserve stable L3F identity and human decisions through index rebuilds.
- Add Obsidian family pages, maintenance review, collapsed home navigation, and CLI review.
- Expand regression coverage to 56 automated tests.

## 1.2.0 - 2026-07-26

- Add 24-hour startup catch-up consolidation guarded across processes.
- Preserve stable identities for emergent and explicitly named L3 concepts.
- Propose likely duplicate L3 concepts without automatic merging.
- Add human merge-left, merge-right, and distinct decisions in Obsidian and CLI.
- Persist concept identities, aliases, and duplicate decisions as canonical Markdown.
- Expand regression coverage to 52 automated tests.

## 1.1.0 - 2026-07-26

- Add experience-driven L3 emergence from repeated confirmed L1 traces.
- Add trace stability, reactivation counts, retention weighting, reconsolidation, and plastic synapse decay.
- Keep forgetting non-destructive: canonical Markdown memory and evidence are never removed by consolidation.
- Add Obsidian review of emergent concepts and their supporting L1 connections.
- Persist concept confirmation and suppression decisions as canonical Markdown so rebuilds preserve human governance.
- Add the `consolidate` command and optional `recall --learn` / MCP `learn` reconsolidation.
- Expand regression coverage to 45 automated tests.

## 1.0.6 - 2026-07-22

- Compile active English L4 procedures and L5 persona/model records into real Obsidian relation pages, instead of unresolved WikiLink placeholders.
- Link every generated relation page back to its L3 topics, following both direct L3 → L5 and L3 → L4 → L5 routes.
- Exclude legacy non-English structural labels from generated relation links and pages.
- Expand regression coverage from 41 to 42 tests.

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
