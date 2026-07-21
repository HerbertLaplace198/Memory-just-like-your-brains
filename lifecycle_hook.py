#!/usr/bin/env python3
"""Portable lifecycle bridge for neural memory.

This is intentionally platform-neutral. A Codex, Claude, or custom-agent wrapper
can pass JSON on stdin and consume JSON on stdout without granting direct access
to the database internals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from neural_memory import NeuralMemory, resolve_encoder, rough_tokens


class LifecycleHook:
    def __init__(self, root: Path, encoder_config: Path | None = None):
        self.memory = NeuralMemory(root, resolve_encoder(root, encoder_config))

    def close(self) -> None:
        self.memory.close()

    @staticmethod
    def _query(payload: dict[str, Any]) -> str:
        for key in ("query", "task", "prompt", "user_prompt", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = self._query(payload)
        if not query:
            return {
                "event": "start",
                "known": False,
                "context": "",
                "reason": "no task query supplied",
            }
        known, peak, activated = self.memory.probe(query)
        routes = [
            {"layer": item.layer, "label": item.label, "activation": round(item.activation, 4)}
            for item in activated
            if item.layer > 1
        ][:3]
        if not known:
            return {
                "event": "start",
                "known": False,
                "peak_l1_activation": round(peak, 4),
                "context": "",
                "routes": routes,
                "instruction": "Do not inject memory for this task.",
            }
        cards = self.memory.recall(query, 3)
        context_lines = [
            "Relevant reviewed memory (summaries only; verify evidence before high-stakes use):"
        ]
        for card in cards:
            context_lines.append(
                f"- [{card.id}] {card.summary} (status={card.status}, a={card.activation:.3f})"
            )
        context = "\n".join(context_lines)
        return {
            "event": "start",
            "known": True,
            "peak_l1_activation": round(peak, 4),
            "context": context,
            "estimated_context_tokens": rough_tokens(context),
            "routes": routes,
            "card_ids": [card.id for card in cards],
        }

    def finish(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_id = str(payload.get("event_id") or payload.get("task_id") or "anonymous")
        candidates = payload.get("memory_candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("memory_candidates must be an array")
        created: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        for index, raw in enumerate(candidates[:10]):
            candidate = {"text": raw} if isinstance(raw, str) else raw
            if not isinstance(candidate, dict):
                errors.append(f"candidate {index}: expected string or object")
                continue
            text = candidate.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"candidate {index}: text must be non-empty")
                continue
            fingerprint = hashlib.sha256(
                f"{event_id}|{text.strip()}".encode("utf-8")
            ).hexdigest()
            meta_key = f"hook_candidate:{fingerprint}"
            existing = self.memory._get_meta(meta_key)
            if existing:
                skipped.append(existing)
                continue
            neuron_id = self.memory.remember(
                text.strip(),
                f"agent:hook:{event_id}",
                topics=candidate.get("topics", []),
                schemas=candidate.get("schemas", []),
                confirmed=False,
                episode=candidate.get("episode"),
                procedures=candidate.get("procedures", []),
                domain=candidate.get("domain"),
                expires_at=candidate.get("expires"),
            )
            self.memory._set_meta(meta_key, neuron_id)
            self.memory.db.commit()
            created.append(neuron_id)
        return {
            "event": "finish",
            "event_id": event_id,
            "created_proposals": created,
            "duplicate_proposals": skipped,
            "errors": errors,
            "ignored_transcript": "transcript" in payload or "messages" in payload,
            "review_required": bool(created),
        }

    def status(self) -> dict[str, Any]:
        return {
            "event": "status",
            "stats": self.memory.stats(),
            "inbox": self.memory.maintenance_inbox(),
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Portable neural-memory lifecycle hook")
    result.add_argument("event", choices=["start", "finish", "status"])
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--encoder-config", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = json.load(sys.stdin) if args.event != "status" else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid JSON: {exc}"}, ensure_ascii=False))
        return 2
    hook = LifecycleHook(args.root, args.encoder_config)
    try:
        if args.event == "start":
            result = hook.start(payload)
        elif args.event == "finish":
            result = hook.finish(payload)
        else:
            result = hook.status()
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (TypeError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        hook.close()


if __name__ == "__main__":
    raise SystemExit(main())
