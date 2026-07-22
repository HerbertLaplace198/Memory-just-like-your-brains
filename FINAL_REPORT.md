# Neural Memory 1.0.1 Acceptance Report

## Conclusion

Neural Memory 1.0.1 is a standalone, reproducible, and auditable local memory system. It implements L0-L6 memory layers, three-stage progressive retrieval, human review, conflict and expiry governance, Obsidian views, MCP, lifecycle hooks, cross-process protection, atomic backup and restore, and real local neural embeddings.

Version 1.0.1 additionally fixes duplicate topics, cross-topic contamination, missing canonical links, L6 graph noise, archive visibility, upper-layer status precedence, stale generated pages, case-only topic filenames, and temporary orphan nodes after automatic proposals.

The reference neural encoder is `qwen3-embedding:0.6b` through Ollama on `127.0.0.1`. The configured vector size is 1024.

## A/B results

The synthetic test library contains five atomic memories and thirteen L1-L6 neurons. The semantic set contains six known paraphrases and seven unrelated queries, including `CNKI Chrome DevTools` as a regression case for archived tool vocabulary.

| Encoder | Gate accuracy | Top-1 | Top-3 |
|---|---:|---:|---:|
| 1024-dimensional feature hash | 61.54% | 50.00% | 50.00% |
| Qwen3-Embedding-0.6B | 100.00% | 83.33% | 100.00% |

The English-only corpus changes tokenization and embedding geometry, so both gates were recalibrated. Hash mode uses `0.48`. Qwen mode uses a direct semantic gate of `0.55`, with an additional `0.15` margin for semantic-only matches that have no lexical support. Neural awareness no longer uses spread-inflated final activation as its gate score.

These results come from a small synthetic regression set. They do not represent an open-domain quality ceiling.

## Security boundaries

- Embedding HTTP accepts only `localhost`, `127.0.0.1`, and `::1`; environment proxies are bypassed.
- MCP cannot confirm or delete memory and cannot switch to arbitrary roots.
- Agent writes are always `proposed`.
- Lifecycle hooks never ingest full transcripts or message histories automatically.
- Obsidian output includes `do_not_ingest: true`; human annotations require explicit review.
- Topic pages link only direct L1 members. L6 remains in SQLite and is excluded from the Obsidian graph.
- Canonical Markdown is readable by design. Use full-disk encryption or an encrypted volume.

## Reliability and recovery

- SQLite WAL, `synchronous=FULL`, and cross-process locking.
- Consistent SQLite Backup API snapshots, SHA-256 manifests, and atomic publication.
- Staged restore with integrity verification before target publication.
- A launchd template for daily backups at 03:00 with ten retained copies.

## Acceptance record

- 34 automated tests passed after the English-only conversion and review-link regression fix.
- Eight independent concurrent writers were verified.
- Tampered bundle detection and restore rejection were verified.
- Export, bundle verification, staged restore, and post-restore health checks were verified.
- Ollama 0.32.1 and `qwen3-embedding:0.6b` were used for the local neural evaluation.

## 1.0.1 regression scope

- Topic alias and capitalization normalization with structural pseudo-topic filtering.
- Upper-layer status upgrade without archival downgrade.
- Direct L1 membership and canonical memory/evidence links on topic pages.
- L6 exclusion from Obsidian and stale generated-page cleanup.
- Auditable archive links.
- Parent-topic inclusion for continuation queries without cross-project noise.
- Case-only filename normalization on macOS.

## Model references

- [Ollama qwen3-embedding](https://ollama.com/library/qwen3-embedding)
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Ollama embedding API example](https://ollama.com/blog/embedding-models)
