# Neural Memory Usage Guide

This guide covers installation, daily use, review, Obsidian maintenance, MCP integration, backup, and recovery.

## 1. Requirements

- Python 3.9 or newer.
- No Python packages are required for hash mode.
- Optional: Ollama and `qwen3-embedding:0.6b` for neural semantic retrieval.
- Optional: Obsidian for browsing the generated maintenance view.

Check Python:

```bash
python3 --version
```

## 2. Create a memory store

Choose a permanent private directory outside the Git repository:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  init
```

This creates the SQLite index and the canonical storage directories. Do not commit this directory to Git.

Run a health check:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  doctor
```

## 3. Choose an encoder

### Zero-dependency hash mode

Hash mode is used automatically when the memory root does not contain `encoder.json`.

It is private, deterministic, and requires no model. The English fallback gate defaults to `0.48`.

### Local Qwen embeddings with Ollama

Install and start Ollama, then download the model:

```bash
ollama pull qwen3-embedding:0.6b
```

To create a new Qwen-backed store from the first command, create the directory and copy the supplied configuration before running `init`:

```bash
mkdir -p /ABSOLUTE/PATH/my-neural-memory
cp encoder.qwen3-local.json \
  /ABSOLUTE/PATH/my-neural-memory/encoder.json
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  init
```

If the store was already initialized in hash mode, migrate it instead of copying `encoder.json` manually:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  reencode ./encoder.qwen3-local.json
```

The bundled English sample uses a Qwen gate threshold of `0.55`. Recalibrate the value after the real memory corpus grows.

Only loopback embedding endpoints are accepted:

```text
127.0.0.1
localhost
::1
```

## 4. Write memory

### Write a proposed memory

Use proposed status for inferred, synthesized, or unverified information:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  remember \
  "The project now uses a weekly review cycle." \
  --source manual \
  --topic "Project Operations"
```

### Write a confirmed fact

Use `--confirmed` only when the user has explicitly confirmed the fact:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  remember \
  "The canonical project directory is /ABSOLUTE/PATH/project." \
  --source user-confirmed \
  --topic "Project Operations" \
  --confirmed
```

### Add memory layers

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  remember \
  "Important decisions require human confirmation." \
  --source manual \
  --episode "Memory design session" \
  --topic "Memory Governance" \
  --procedure "Review workflow" \
  --schema "Prefers supervised systems" \
  --domain "Knowledge Management" \
  --confirmed
```

Layer mapping:

| Input | Layer | Purpose |
|---|---:|---|
| Evidence file | L0 | Raw source material |
| Memory body | L1 | Atomic durable fact |
| `--episode` | L2 | Event or continuing task |
| `--topic` | L3 | Semantic routing topic |
| `--procedure` | L4 | Workflow or SOP |

New L3 topic, L2 episode, L4 procedure, L5 schema, and L6 domain labels are English-only. Non-English labels are rejected before any canonical evidence is written; known Chinese aliases are normalized to English. The memory body itself may use any language. Rebuilding an older store skips legacy non-English structural labels. When an L1 memory is rejected, any L3/L4 nodes and synapses that are no longer reachable from another active L1 are pruned automatically.

When a new memory is written, explicit topic labels are treated as hints. The
index first compares them and the memory body with active L3 topics, reuses all
strong matches, and preserves multiple matches when one memory belongs to more
than one topic. A hint that does not match an existing topic may still create
a new topic through the normal proposed/confirmed lifecycle; matched topics
are never recreated as duplicates. A memory without a topic hint remains an
L1 trace until consolidation finds a repeated pattern across other confirmed
L1 traces and proposes an emergent L3 concept.
| `--schema` | L5 | Stable preference or model |
| `--domain` | L6 | High-level awareness route |

### Add expiry and replacement relations

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  remember \
  "The current release target is September 2026." \
  --source user-confirmed \
  --topic "Release Planning" \
  --expires "2026-10-01" \
  --supersedes "l1_OLD_MEMORY_ID" \
  --confirmed
```

The old memory is not archived automatically. The replacement relation enters human review.

## 5. Review memory

List proposed memories:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  review list
```

Confirm, reject, mark stale, or archive a memory:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory review confirm l1_MEMORY_ID
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory review reject l1_MEMORY_ID
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory review stale l1_MEMORY_ID
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory review archive l1_MEMORY_ID
```

Rejecting a memory moves its canonical L1 record and unshared evidence together into `vault/.rejected/`. That hidden folder is included in backups but excluded from active indexing and the Obsidian graph. Restore it as a fresh proposed candidate, or move any legacy orphan evidence into the same archive:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  restore-rejected l1_MEMORY_ID

python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  archive-orphan-evidence
```

Submitting any valid review decision through `sync-obsidian` automatically recompiles the Obsidian view. Confirmed memories that include an episode are linked through their topic pages together with their evidence, so they do not appear as isolated graph nodes.

Scan maintenance issues and pending relations:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory maintenance scan
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory maintenance inbox
```

Review a replacement relation:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  maintenance confirm-relation rel_RELATION_ID
```

Confirming a `supersedes` relation archives the replaced memory while preserving it for audit and rollback.

## 6. Retrieve memory

### Awareness only

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  probe "Continue the release plan"
```

`KNOWN` means the gate found sufficient direct evidence. `UNKNOWN` means no memory should be injected.

### Recall compact cards

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  recall "Continue the release plan" --limit 3
```

### Include full evidence

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  recall "Continue the release plan" --limit 1 --detail
```

### Opt in to reconsolidation

Ordinary recall is read-only. Add `--learn` only when a successful retrieval
should stabilize the recalled L1 traces and strengthen their co-active links:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  recall "Continue the release plan" --limit 3 --learn
```

Run periodic semantic consolidation and plastic-link decay explicitly:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  consolidate
```

Consolidation may create a proposed emergent L3 concept from three or more
similar confirmed L1 traces that have no shared explicit topic. It never
deletes canonical memory or evidence.

The MCP server also runs this cycle on startup when the previous consolidation
is at least 24 hours old. Cooperating clients share the write lock, so only one
client performs an overdue cycle.

### Explain an incorrect result

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  explain "Unexpected query" --limit 10
```

The explanation includes vector, BM25, lexical, stability, retention, direct,
and spread scores. Use it to tune `gate_threshold` and add regression queries.

## 7. Generate and maintain the Obsidian view

Compile the reading view:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  compile-obsidian
```

Open this directory as an Obsidian vault:

```text
/ABSOLUTE/PATH/my-neural-memory/obsidian-view
```

Generated structure:

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

Every active English L4 procedure and L5 persona/model that is connected to a topic gets a real generated page. Topic pages link to those pages, and each relation page links back to its related topics. Legacy non-English structural labels are not emitted as WikiLinks, so they cannot appear as uncreated graph nodes.

`99 Maintenance.md` also lists every proposed emergent L3 concept, its
stability, and the L1 traces that support it. Select exactly one concept action:

- **Confirm concept and connections** preserves the abstraction across index rebuilds.
- **Reject concept and suppress this exact pattern** prevents the same set of
  supporting traces from recreating it.

Submit the decision with the Obsidian plugin button or `sync-obsidian`.
Decisions are stored as canonical Markdown under `.neural-memory/semantic-reviews/`.

Likely duplicate L3 concepts appear in the same maintenance page. They are
never merged automatically. Review them in Obsidian, or use:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  concept-duplicate list
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  concept-duplicate distinct DUPLICATE_REVIEW_ID
```

The other decisions are `merge-left` and `merge-right`. Stable concept
identities are stored under `.neural-memory/concept-identities/`; merge, alias,
and distinct decisions are stored under `.neural-memory/concept-decisions/`.

L3F concept families group three or more related L3 concepts without merging
them or changing the numbering of L4-L6. Confirmed families collapse their
members into one Obsidian home-page entry while retaining every topic page.
Review them in Obsidian or use:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  concept-family list
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  concept-family confirm L3F_FAMILY_ID
```

Use `concept-family reject` to suppress the exact grouping. Family identity
and review state are stored under `.neural-memory/concept-families/`.

After confirmation, L3F participates in retrieval. The coarse pass compares
family representatives with independent L3 concepts. A matched family expands
its own member L3 concepts while independent L3 remains searchable; members of
other families continue to be represented by their family. If no family or
independent L3 supplies a safe route, Neural Memory falls back to full L3
search. Inspect the decision with `explain` or the MCP `l3f_routing` field.

The L3F threshold is separate from the L1 awareness threshold. Configure it as
`family_gate_threshold` in the encoder JSON; recalibrate it with representative
positive family queries and unrelated negative queries when the corpus changes.

Once the family gate opens, route competition uses the base family score plus a
bounded `family_size_bonus_cap` (default `0.08`). The bonus grows with the
number of member L3 concepts and unique active L1 support, with diminishing
returns. Confirmation remains a hard requirement, and this bonus only chooses
the L3F route; member L3 scoring remains `0.45 vector + 0.45 BM25 + 0.10
lexical`.

Existing v1.4 memory stores and encoder configurations remain compatible. If
an older encoder JSON omits `family_size_bonus_cap`, Neural Memory uses `0.08`;
no Markdown or SQLite migration is required.

Write human notes only inside the `USER-NOTES` block. Synchronize them into review candidates:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory sync-obsidian
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory obsidian-review list
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  obsidian-review show note_PROPOSAL_ID
```

Accept or reject the candidate explicitly:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  obsidian-review accept note_PROPOSAL_ID
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory \
  obsidian-review reject note_PROPOSAL_ID
```

Obsidian output is derived and must never be re-ingested automatically.

## 8. Connect an MCP client

Copy `mcp.json.example` and replace both absolute paths:

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python3",
      "args": [
        "/ABSOLUTE/PATH/neural-memory-1.5.1/mcp_server.py",
        "--root",
        "/ABSOLUTE/PATH/my-neural-memory"
      ]
    }
  }
}
```

Restart the MCP client after changing its configuration.

Expected agent workflow:

1. Call `memory_awareness` when prior work may matter.
2. Call `memory_recall` only when awareness returns `known=true`.
3. Request detailed evidence only when summaries are insufficient.
4. Use `memory_propose` for durable new information.
5. Leave confirmation, rejection, archival, and deletion to human review.

## 9. Lifecycle hook

Start a task:

```bash
printf '%s' '{"task":"Continue the release plan","event_id":"task-123"}' | \
  python3 lifecycle_hook.py start \
  --root /ABSOLUTE/PATH/my-neural-memory
```

Finish a task with explicit candidates:

```bash
printf '%s' '{
  "event_id":"task-123",
  "memory_candidates":[
    {
      "text":"Release validation now includes a privacy scan.",
      "topics":["Release Planning"]
    }
  ]
}' | python3 lifecycle_hook.py finish \
  --root /ABSOLUTE/PATH/my-neural-memory
```

The hook ignores full transcripts and forces all candidates to `proposed`.

## 10. Backup and restore

Create a verified backup and keep the ten newest copies:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  backup /ABSOLUTE/PATH/backups --keep 10
```

Verify a bundle without restoring it:

```bash
python3 neural_memory.py verify-bundle \
  /ABSOLUTE/PATH/backups/neural-memory-TIMESTAMP.nmem
```

Restore into an empty directory:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/restored-memory \
  import-bundle /ABSOLUTE/PATH/backups/neural-memory-TIMESTAMP.nmem
```

Run `doctor` after restoration.

The file `com.neural-memory.backup.plist.example` is a macOS launchd template. Replace every placeholder path before installing it.

## 11. Rebuild the index

If the SQLite index is missing or damaged, rebuild it from canonical Markdown:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  rebuild
```

Canonical sources:

```text
vault/memories/*.md
vault/evidence/*.md
```

SQLite and `obsidian-view/` are derived and rebuildable.

## 12. Run tests and evaluation

```bash
python3 -m unittest -v test_neural_memory.py
```

Evaluate a memory store:

```bash
python3 neural_memory.py \
  --root /ABSOLUTE/PATH/my-neural-memory \
  evaluate ./evaluation.semantic.json --limit 3
```

Maintain separate known paraphrases and unrelated UNKNOWN queries. Recalibrate the gate whenever the corpus or encoder changes.

## 13. Privacy rules

- Never commit a real memory root or backup.
- Treat every `.nmem` file as private unless it was generated from synthetic data.
- Keep canonical Markdown on FileVault or another encrypted volume.
- Do not place API keys in encoder configuration files.
- Do not weaken the loopback-only embedding restriction without a security review.
- Inspect sample bundles after extraction before publishing them.

## 14. Troubleshooting

### Encoder mismatch

If the index reports a different encoder or vector dimension, use `reencode` with the correct configuration. Do not bypass the mismatch check during normal operation.

### Ollama connection failure

Confirm Ollama is running and the endpoint is reachable locally:

```bash
ollama list
```

Check that `encoder.json` uses `127.0.0.1`, `localhost`, or `::1`.

### Too many false-positive memories

Add realistic UNKNOWN queries, inspect `explain`, and raise `gate_threshold` carefully.

### Valid paraphrases are rejected

Add the paraphrases to the evaluation set, inspect direct semantic scores, and lower the threshold only if UNKNOWN separation remains acceptable.

### Obsidian pages look stale

Run `compile-obsidian` again. Generated stale topic pages are removed automatically, while valid `USER-NOTES` blocks are preserved.

### Review a proposed memory in Obsidian

Open `99 Maintenance.md`, follow the memory and evidence links, and select exactly one checkbox: **Confirm**, **Needs revision**, or **Incorrect / reject**. Then apply the explicit human decision:

```bash
python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory sync-obsidian
```

Needs revision leaves the memory proposed and adds a maintenance issue. Selecting more than one option changes nothing and returns an error.

For a submit button, install the bundled `obsidian-plugin/neural-memory-review` directory as `.obsidian/plugins/neural-memory-review` inside the memory vault and enable **Neural Memory Review** in Obsidian Community plugins. The plugin adds a button to the maintenance page, a ribbon action, and a command-palette action. It runs the fixed local Python command without a shell only after explicit submission.

### MCP still uses old behavior

Restart the MCP client so it launches the updated server process and reloads the memory root configuration.
