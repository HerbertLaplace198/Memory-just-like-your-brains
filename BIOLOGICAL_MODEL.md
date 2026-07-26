# Biologically Inspired Memory Model

Neural Memory 1.2.0 uses a biologically inspired architecture while preserving
an auditable software boundary. It does not claim to reproduce human
neuroscience literally.

## Memory formation

L0 evidence is the preserved sensory-like source. L1 records are atomic memory
traces. L2 groups traces into episodes. L3 represents semantic concepts formed
from repeated experience. L4-L6 remain procedural, stable-model, and routing
layers.

Explicit topics still create auditable L3 routes. In addition, consolidation
forms an emergent L3 candidate when at least three confirmed L1 traces are
semantically similar and do not already share an explicit L3 topic. The
candidate remains proposed until a human reviews it.

## Consolidation and forgetting

Each neuron has a stability value, reactivation count, and last-reactivated
time. Retrieval can opt into reconsolidation, which stabilizes recalled traces
and strengthens co-active links. The `consolidate` command, or an overdue MCP
startup cycle after 24 hours:

1. rebuilds emergent semantic concepts from active confirmed traces;
2. recalculates L3 stability from supporting L1 and L2 experience;
3. weakens inactive association and co-recall synapses over time.

Structural links and canonical evidence do not decay. Forgetting therefore
means reduced accessibility, not deletion.

## Human governance in Obsidian

`99 Maintenance.md` lists each proposed emergent concept, its stability, and
the supporting L1 connections. A reviewer may confirm or reject the concept.
The decision is stored under `vault/semantic-reviews/` as canonical Markdown.

Confirmation preserves the concept across rebuilds. Rejection suppresses that
exact support pattern. If the supporting memories later change, the system
creates a new candidate that requires a new review.

Stable concept identities prevent a changing support set from creating a new
L3 on every consolidation. Likely duplicate L3 concepts are shown for review;
the system never merges them automatically. A human can merge in either
direction or permanently mark the pair as distinct.

## Safety boundary

- Automatic abstraction never confirms itself.
- Proposed and rejected memories do not support emergent concepts.
- Consolidation never rewrites or deletes L1 evidence.
- SQLite and Obsidian remain rebuildable views.
- Human-reviewed Markdown remains authoritative.
