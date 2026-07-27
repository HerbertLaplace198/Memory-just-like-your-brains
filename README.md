# Neural Memory 1.5.4

Neural Memory is an auditable, layered, local-first memory system. Canonical memory content is stored in Markdown, while SQLite provides a rebuildable neural retrieval index. It is independent of mdkb, has a zero-dependency hash encoder, and can optionally use a local neural embedding service.

For installation and day-to-day commands, see [`USAGE.md`](USAGE.md).
For the one-source release and runtime contract, see [`RUNTIME.md`](RUNTIME.md).

## Current maintenance updates

- Restrict all retrieval and long-term weighting to confirmed records. Proposed
  memories remain visible in the review workflow but cannot be injected,
  reconsolidated, or used to strengthen concept and family routing.
- Use `<memory-root>/memory.sqlite3` as the sole live SQLite index. A legacy
  `.neural-memory/index.sqlite` is preserved and explicitly marked, never used
  by accident. L3 routing additionally requires live confirmed L1 support.
- Keep Obsidian maintenance actionable: obsolete review issues close
  automatically, and the generated runtime-status page separates current facts
  from historical release records.
- Store canonical governance records under the hidden `<root>/.neural-memory/`
  directory: semantic reviews, concept identities, duplicate decisions, and
  L3F family records. Keep `vault/` for human-auditable memories and evidence;
  do not commit either directory from a real memory root.
- Use readable L3F names in Obsidian while retaining stable internal IDs in
  backend metadata. Confirmed family pages no longer show review links.
- Require English structural labels for new topics, episodes, procedures,
  schemas, and domains; memory bodies may remain in any language. Known
  Chinese aliases are canonicalized to English before indexing.
- Enrich multilingual family queries with known English concept aliases and
  use a separate configurable `family_gate_threshold`; the live Qwen
  configuration is calibrated at `0.42` for L3F routing.
- After the family gate opens, give larger confirmed families a bounded route
  bonus based on member L3 count and unique active L1 support. The default cap
  is `0.08`; L3 member scoring is unchanged.

## What changed in 1.4.0

- Make confirmed L3F families participate in first-stage semantic routing
  instead of serving only as navigation pages.
- Search family representatives and independent L3 concepts together. A
  matched family expands its member L3 concepts while independent L3 remains
  eligible.
- Keep members of unselected families out of first-stage competition because
  their family representative stands in for them.
- Fall back to full L3 search when neither a family nor an independent L3
  provides a safe route.
- Aggregate L4/L5 relationships shared by multiple family members and expose
  the routing decision through MCP and `explain`.

## Unreleased

- Give larger confirmed L3F families slightly more routing authority using a
  saturating member/L1 support bonus capped by `family_size_bonus_cap` (default
  `0.08`). The family relevance gate remains primary, and L3 member scores are
  unchanged.
- Route each new L1 memory through existing L3 topics before accepting new
  labels. Multiple strong existing matches are retained, while genuinely new
  explicit topic hints may still create proposed L3 candidates without
  duplicating the matched topics.
- Keep topic formation experience-driven: confirmed L1 traces without a shared
  topic can still consolidate into a proposed emergent L3, which remains under
  human review.

## What changed in 1.3.0

- Add L3F, a stable concept-family layer that groups three or more related L3
  concepts without merging them or renumbering L4-L6.
- Keep L3F outside the neuron layer numbering and store its identity and review
  state as canonical Markdown under `.neural-memory/concept-families/`.
- Let reviewers confirm or reject proposed families in Obsidian or through the
  `concept-family` CLI.
- Collapse confirmed family members into one family entry on the Obsidian home
  page while retaining every individual L3 topic page.

## What changed in 1.2.0

- Run overdue semantic consolidation when the MCP server starts, using a
  24-hour interval and a cross-process lock.
- Give emergent and named L3 concepts stable identities across consolidation
  and full index rebuilds.
- Detect likely duplicate L3 concepts as review candidates without merging
  them automatically.
- Let reviewers merge into either concept or mark a pair distinct. Aliases and
  decisions are canonical Markdown and survive rebuilds.

## What changed in 1.1.0

- Treat L1 records as memory traces with stability, reactivation count, and time-dependent retention.
- Form proposed emergent L3 concepts when at least three semantically similar confirmed L1 traces have no shared explicit topic.
- Derive L3 stability from the number and episodic diversity of its supporting memories.
- Add retrieval reconsolidation with `recall --learn`, strengthening co-active traces without making ordinary recall mutate state.
- Add `consolidate` for safe plastic-link decay and semantic reorganization. Forgetting changes retrieval strength but never deletes canonical memory or evidence.
- Show emergent L3 concepts and every supporting L1 connection in Obsidian's maintenance page. Human confirmation or rejection is stored as canonical Markdown and survives index rebuilds.

This is a biologically inspired computational model, not a literal simulation of a human brain. See [`BIOLOGICAL_MODEL.md`](BIOLOGICAL_MODEL.md) for the design boundary.

## What changed in 1.0.6

- L4 procedure and L5 persona/model relationships now compile into real, clickable Obsidian pages under `relations/`, rather than unresolved WikiLink placeholders.
- Generated relation pages link back to their related topics, including L5 pages reached through an L4 procedure route.
- Only active English structural labels generate relation pages; legacy non-English labels cannot create ghost graph nodes. Regression coverage expands to 42 automated tests.

## What changed in 1.0.5

- Topic compilation now follows `L3 → L2 episode → L1` routes, so confirmed episodic memories and their evidence receive topic-page links instead of appearing as isolated Obsidian nodes.
- Every successful review submission automatically recompiles the Obsidian view; regression coverage expands to 41 automated tests.

## What changed in 1.0.4

- Rejecting an L1 now moves its canonical record and unshared evidence together into `vault/.rejected/`, a hidden, backup-safe and recoverable archive.
- Add `restore-rejected` and `archive-orphan-evidence` commands; health checks now report unreferenced evidence.
- Regression coverage expands to 40 automated tests.

## What changed in 1.0.3

- Rejecting an L1 memory now prunes L3/L4 nodes and synapses that no remaining active L1 can reach.
- New L3 topic and L4 procedure labels must be English; rebuilds skip legacy non-English structural labels.
- Rejected and archived canonical records no longer regenerate upper-layer nodes during a rebuild.
- Regression coverage expands to 38 automated tests.

## What changed in 1.0.2

- MCP and lifecycle proposals immediately refresh the Obsidian view, eliminating temporary orphan nodes.
- `99 Maintenance.md` links every proposed memory and its evidence from one review queue.
- Each proposed memory has explicit **Confirm**, **Needs revision**, and **Incorrect / reject** actions, synchronized only after an unambiguous human selection.
- The bundled desktop Obsidian plugin adds a maintenance-page submit button, ribbon action, and command-palette action. Checked decisions are applied only after explicit submission.

## What changed in 1.0.1

- L3 topic labels are canonicalized before graph construction, merging aliases, capitalization variants, and duplicates.
- Recall is scoped to the strongest L3 route. Queries containing `continue` or `resume` may also include a configured parent topic.
- Obsidian topic pages include only direct active L1 members, preventing graph diffusion from mixing unrelated topics.
- Topic pages link directly to canonical memory and evidence files. Archived records are listed in `98 Archive.md`.
- L6 routing nodes remain in SQLite but are hidden from the Obsidian graph.
- The compiler removes stale generated topic pages and handles case-only filename changes on macOS.
- Upper-layer status can be upgraded from proposed to confirmed but cannot be downgraded by later archived records.
- Neural awareness now gates on direct semantic evidence rather than spread-inflated activation. The Qwen sample threshold was recalibrated to `0.55` on the expanded 13-case English evaluation set.

## Architecture

```text
L6 meta-memory domain   Awareness and routing
L5 stable model         User profile, goals, stable preferences
L4 procedural memory    Skills, SOPs, workflows
L3 semantic concept     Projects, people, topics, wiki relations
L2 episodic memory      Tasks, events, continuing episodes
L1 atomic memory        Facts, decisions, preferences, constraints
L0 raw evidence         Conversations, files, pages, tool output
```

Core capabilities:

- Pluggable `TextEncoder` interface.
- Hybrid vector, BM25, and lexical retrieval.
- Winner-take-all selection and two rounds of spreading activation.
- Explicit bidirectional synapses and optional Hebbian reinforcement.
- `proposed`, `confirmed`, `rejected`, `stale`, and `archived` states.
- Canonical Markdown records with a fully rebuildable SQLite index.
- Human-readable Obsidian views that are excluded from re-ingestion.
- Portable `.nmem` bundles with SHA-256 integrity verification.
- Maintenance inbox, expiry, conflict candidates, and reviewed replacement relations.
- Staged MCP retrieval and a lifecycle hook with an enforced human review boundary.
- WAL, cross-process locking, atomic backups, verified restore, and encoder migration.

## Quick start

```bash
python3 neural_memory.py --root ./demo-memory seed-demo

# Stage 1: awareness only
python3 neural_memory.py --root ./demo-memory \
  probe "How can I reduce token usage?"

# Stage 2: compact memory cards
python3 neural_memory.py --root ./demo-memory \
  recall "How can I reduce token usage?" --limit 3

# Stage 3: include full evidence only when needed
python3 neural_memory.py --root ./demo-memory \
  recall "How can I reduce token usage?" --limit 1 --detail
```

When awareness returns `UNKNOWN`, recall remains closed and no low-scoring candidate is injected. Use `--force` only for debugging.

To inspect retrieval scoring:

```bash
python3 neural_memory.py --root ./demo-memory \
  explain "How can I reduce token usage?" --limit 10
```

The explanation separates vector, BM25, lexical, direct, and spread components.

## Human review and maintenance

Maintenance scans create review candidates but never mutate memory automatically:

```bash
python3 neural_memory.py --root ./demo-memory maintenance scan
python3 neural_memory.py --root ./demo-memory maintenance inbox
```

Create an expiring fact or declare a replacement:

```bash
python3 neural_memory.py --root ./demo-memory remember \
  "The user now follows the new memory maintenance workflow." \
  --confirmed \
  --expires "2027-01-01" \
  --supersedes "l1_OLD_MEMORY_ID"
```

Replacement and conflict relations begin as `pending`. Only explicit human review can archive an old memory:

```bash
python3 neural_memory.py --root ./demo-memory \
  maintenance confirm-relation "rel_RELATION_ID"
```

Expired memories are not deleted automatically. Verify them first, then mark them stale or write a reviewed replacement.

## Obsidian review loop

`USER-NOTES` blocks are human-maintained output regions. Synchronization creates proposals; it does not ingest them directly:

```bash
python3 neural_memory.py --root ./demo-memory sync-obsidian
python3 neural_memory.py --root ./demo-memory obsidian-review list
python3 neural_memory.py --root ./demo-memory \
  obsidian-review accept "note_PROPOSAL_ID"
```

`compile-obsidian` generates:

```text
obsidian-view/
|-- 00 Home.md
|-- 98 Archive.md
|-- 99 Maintenance.md
|-- topics/*.md
`-- relations/
    |-- procedures/*.md
    `-- personas/*.md
```

Topic pages contain a narrative, canonical L1 and evidence links, links to generated L4/L5 relation pages, sources, and a preserved human notes block. Relation pages link back to their topics. L6 is intentionally excluded from the visual graph. Every generated page includes `generated: true` and `do_not_ingest: true`.

MCP and lifecycle-hook proposals refresh this generated view immediately after a successful write, so a newly created canonical memory and its evidence appear under their topic without a temporary orphan-node window.

`99 Maintenance.md` is the review entry point. Its **Proposed memories** section links directly to every proposed canonical L1 record and its evidence. Select exactly one checkbox—**Confirm**, **Needs revision**, or **Incorrect / reject**—then run `sync-obsidian`. A revision request keeps the candidate proposed and creates a maintenance issue; ambiguous multiple selections are rejected without changing memory.

## MCP integration

`mcp_server.py` is a dependency-free stdio MCP adapter bound to one memory root at startup:

```bash
python3 mcp_server.py --root /ABSOLUTE/PATH/my-neural-memory
```

See `mcp.json.example`. The server exposes five tools:

- `memory_awareness`: low-token first-stage gate.
- `memory_recall`: up to five L1 cards; detailed evidence is opt-in.
- `memory_explain`: score decomposition for debugging.
- `memory_propose`: creates a proposed memory only.
- `memory_inbox`: read-only review inbox.

Recommended flow:

```text
Task may depend on history -> memory_awareness
KNOWN                      -> memory_recall(detail=false)
Summary is insufficient    -> memory_recall(detail=true)
Durable new fact           -> memory_propose
Final confirmation         -> human review in CLI or Obsidian
```

The MCP service cannot confirm or delete memory and cannot read arbitrary files.

## Lifecycle hook

`lifecycle_hook.py` provides platform-neutral JSON stdin/stdout commands:

```bash
printf '%s' '{"task":"How can I reduce token usage?","event_id":"task-123"}' | \
  python3 lifecycle_hook.py start --root /ABSOLUTE/PATH/my-neural-memory
```

At task completion, only explicit `memory_candidates` are written, and every candidate is forced to `proposed`. Full transcripts and message histories are ignored. Requests are idempotent by event ID and content.

See `lifecycle.example.json` for both events.

## Reliability, backup, and restore

The database uses SQLite WAL, a five-second busy timeout, `synchronous=FULL`, and a cooperating cross-process `.write.lock`.

```bash
python3 neural_memory.py --root ./demo-memory doctor
python3 neural_memory.py --root ./demo-memory backup ./backups --keep 10
python3 neural_memory.py verify-bundle ./backups/neural-memory-TIMESTAMP.nmem
python3 neural_memory.py --root ./restored-memory \
  import-bundle ./backups/neural-memory-TIMESTAMP.nmem
```

Backups use the SQLite Backup API, validate the snapshot, generate a SHA-256 manifest, and publish atomically. Restore validates in a staging directory before making the target visible.

## Local neural encoders

The default `feature-hash-v1` encoder requires no model or network access. Local neural encoders may use Ollama or an OpenAI-compatible loopback endpoint.

Allowed hosts:

```text
127.0.0.1
localhost
::1
```

Any non-loopback endpoint is rejected. Templates:

- `encoder.ollama.example.json`
- `encoder.openai-local.example.json`
- `encoder.hash-256.example.json`
- `encoder.qwen3-local.json`

Migrate atomically:

```bash
python3 neural_memory.py --root ./demo-memory \
  reencode ./encoder.qwen3-local.json
```

The migration re-encodes L1-L6 in one transaction, rebuilds vector associations, persists `encoder.json`, and rolls back on any request or dimension error.

`gate_threshold` must be calibrated for each memory corpus. Hash mode gates on final hybrid activation; neural mode gates on the strongest lexically supported L1 vector score. A semantic-only match must exceed the configured threshold by an additional `0.15` margin. The synthetic English Qwen sample uses `0.55` on the included 13-case regression set. A larger, denser, or more specialized corpus may require a different value. Include real paraphrases and explicit UNKNOWN queries when selecting the threshold.

## Writing L0-L6

```bash
python3 neural_memory.py --root ./demo-memory remember \
  "Important memories require human confirmation." \
  --source manual \
  --episode "Memory system design session" \
  --topic "Memory Governance" \
  --procedure "Memory review workflow" \
  --schema "Prefers supervised systems" \
  --domain "AI memory and knowledge management" \
  --confirmed
```

```text
Body        -> L1
--episode   -> L2
--topic     -> L3
--procedure -> L4
--schema    -> L5
--domain    -> L6
```

## Rebuild, evaluate, and benchmark

Canonical records are authoritative:

```bash
python3 neural_memory.py --root ./demo-memory rebuild
python3 neural_memory.py --root ./demo-memory compile-obsidian
python3 neural_memory.py --root ./demo-memory \
  evaluate ./evaluation.json --limit 3
python3 -m unittest -v test_neural_memory.py
```

`benchmark` provides a rough comparison between full-library context and staged recall. It is not an official tokenizer bill.

## Portable bundles

```bash
python3 neural_memory.py --root ./demo-memory \
  export-bundle ./my-memory.nmem
python3 neural_memory.py --root ./restored-memory \
  import-bundle ./my-memory.nmem
```

An `.nmem` file is a ZIP containing the database, canonical records, evidence, Obsidian view, encoder configuration, format manifest, and per-file SHA-256 digests. Import requires an empty target directory.

## Optional legacy mdkb import

mdkb is not a runtime dependency. Shadow mode stores metadata and pointers without copying full content:

```bash
python3 neural_memory.py --root ./mdkb-shadow import-mdkb \
  --workspace "/ABSOLUTE/PATH/legacy-mdkb-workspace"
```

Add `--copy-content` only when a complete offline copy is explicitly required.

## Privacy and publishing

Never commit a real memory root. `vault/`, `memory.sqlite3`, `obsidian-view/`, backups, and local `encoder.json` files may contain private text, absolute paths, or searchable indexes. The bundled `.gitignore` excludes them by default and permits only explicitly named synthetic sample bundles.

Before public release:

1. Scan files and Git history for secrets, usernames, and absolute paths.
2. Extract and manually inspect every sample `.nmem` bundle.
3. Keep application-readable Markdown on full-disk encryption or an encrypted volume.
4. Select an appropriate software license.

## Known limits

- Conflict detection only raises conservative review candidates.
- The review interface is CLI plus Obsidian; there is no standalone GUI.
- The launchd backup file is a template and requires installation-specific paths.
- Markdown is intentionally readable and is not encrypted by the application.
- The bundled evaluation set is a regression aid, not an open-domain benchmark.

See `FINAL_REPORT.md` for acceptance results and `CHANGELOG.md` for release history.
