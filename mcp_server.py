#!/usr/bin/env python3
"""Dependency-free MCP stdio adapter for the neural memory demo.

The adapter deliberately exposes a narrow capability surface. Agents can probe
and recall memory, and can submit proposed memories, but cannot confirm them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from neural_memory import NeuralMemory, compact, resolve_encoder


PROTOCOL_VERSION = "2025-06-18"


class MCPServer:
    def __init__(self, root: Path, encoder_config: Path | None = None):
        self.memory = NeuralMemory(root, resolve_encoder(root, encoder_config))
        try:
            self.startup_consolidation = self.memory.consolidate_if_due()
            if self.startup_consolidation["performed"]:
                view = self.memory.compile_obsidian()
                self.startup_consolidation["obsidian_pages"] = view["pages"]
        except Exception as exc:
            self.memory.db.rollback()
            self.startup_consolidation = {
                "performed": False,
                "error": str(exc),
            }
            print(f"startup consolidation skipped: {exc}", file=sys.stderr)

    def close(self) -> None:
        self.memory.close()

    @staticmethod
    def tools() -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_awareness",
                "description": (
                    "Low-token first-stage check: decide whether related memory exists. "
                    "Call this before memory_recall."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_recall",
                "description": (
                    "Recall up to five L1 memory cards after memory_awareness returns known=true. "
                    "Detailed evidence is opt-in to save tokens."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                        "detail": {"type": "boolean"},
                        "learn": {
                            "type": "boolean",
                            "description": "Opt in to retrieval reconsolidation and Hebbian strengthening.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_explain",
                "description": (
                    "Explain retrieval scores and neural spreading when a recall looks wrong. "
                    "Use for debugging, not routine recall."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_propose",
                "description": (
                    "Submit a proposed memory after a task. It always enters human review and "
                    "can never create a confirmed memory directly. Structural labels in "
                    "topics, procedures, schemas, episode, and domain must be English; "
                    "the memory text may use any language."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "episode": {"type": "string"},
                        "procedures": {"type": "array", "items": {"type": "string"}},
                        "schemas": {"type": "array", "items": {"type": "string"}},
                        "domain": {"type": "string"},
                        "expires": {"type": "string"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_inbox",
                "description": "Read the human-review inbox without changing any memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        ]

    @staticmethod
    def _required_text(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return value.strip()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "memory_awareness":
            query = self._required_text(arguments, "query")
            known, peak, activated = self.memory.probe(query)
            family_routing = self.memory.concept_family_routes(query)
            return {
                "known": known,
                "peak_l1_activation": round(peak, 4),
                "l3f_routing": {
                    "used": family_routing["used"],
                    "reason": family_routing["reason"],
                    "families": family_routing["families"],
                },
                "active_routes": [
                    {
                        "layer": item.layer,
                        "label": item.label,
                        "activation": round(item.activation, 4),
                    }
                    for item in activated
                    if item.layer > 1
                ][:3],
                "next": "memory_recall" if known else "do_not_inject_memory",
            }
        if name == "memory_recall":
            query = self._required_text(arguments, "query")
            limit = max(1, min(5, int(arguments.get("limit", 3))))
            detail = bool(arguments.get("detail", False))
            learn = bool(arguments.get("learn", False))
            known, peak, _ = self.memory.probe(query)
            if not known:
                return {
                    "known": False,
                    "peak_l1_activation": round(peak, 4),
                    "cards": [],
                    "reason": "recall gate closed",
                }
            cards = self.memory.recall(query, limit, reconsolidate=learn)
            result_cards: list[dict[str, Any]] = []
            for card in cards:
                result: dict[str, Any] = {
                    "id": card.id,
                    "summary": card.summary,
                    "status": card.status,
                    "activation": round(card.activation, 4),
                    "evidence_id": card.evidence_id,
                }
                if detail and card.evidence_id:
                    result["evidence"] = self.memory.evidence_text(card.evidence_id)
                result_cards.append(result)
            return {
                "known": True,
                "cards": result_cards,
                "reconsolidated": learn and bool(result_cards),
            }
        if name == "memory_explain":
            query = self._required_text(arguments, "query")
            limit = max(1, min(10, int(arguments.get("limit", 7))))
            known, peak, activated = self.memory.probe(query)
            family_routing = self.memory.concept_family_routes(query)
            return {
                "known": known,
                "peak_l1_activation": round(peak, 4),
                "l3f_routing": family_routing,
                "formula": "retention × governance × (0.45 vector + 0.45 BM25 + 0.10 lexical) + spread",
                "l3f_formula": (
                    "confirmed family route = 0.75 semantic + 0.25 lexical; "
                    "the gate uses this base score, then adds a bounded size "
                    "bonus from member L3 count and unique active L1 support"
                ),
                "activations": [
                    {
                        "id": item.id,
                        "layer": item.layer,
                        "label": compact(item.label, 90),
                        "activation": round(item.activation, 4),
                        "direct": round(item.direct_activation, 4),
                        "spread": round(item.spread_activation, 4),
                        "vector": round(item.vector_score, 4),
                        "bm25": round(item.bm25_score, 4),
                        "lexical": round(item.lexical_score, 4),
                        "stability": round(item.stability, 4),
                        "retention": round(item.retention, 4),
                    }
                    for item in activated[:limit]
                ],
            }
        if name == "memory_propose":
            text = self._required_text(arguments, "text")
            neuron_id = self.memory.remember(
                text,
                "agent:mcp",
                topics=arguments.get("topics", []),
                schemas=arguments.get("schemas", []),
                confirmed=False,
                episode=arguments.get("episode"),
                procedures=arguments.get("procedures", []),
                domain=arguments.get("domain"),
                expires_at=arguments.get("expires"),
            )
            view = self.memory.compile_obsidian()
            return {
                "id": neuron_id,
                "status": "proposed",
                "next": "human review required",
                "obsidian_view_refreshed": True,
                "obsidian_pages": view["pages"],
            }
        if name == "memory_inbox":
            return self.memory.maintenance_inbox()
        raise ValueError(f"unknown tool: {name}")

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if method == "initialize":
            return self._result(request_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "neural-memory", "version": "1.5.6"},
            })
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": self.tools()})
        if method == "tools/call":
            params = request.get("params") or {}
            try:
                payload = self.call_tool(str(params.get("name", "")), params.get("arguments") or {})
                return self._result(request_id, {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                    "structuredContent": payload,
                    "isError": False,
                })
            except Exception as exc:
                # A write error must remain a tool-level failure.  Letting a
                # database or encoder exception escape closes the stdio stream,
                # which clients report only as a disconnected memory service.
                try:
                    self.memory.db.rollback()
                except Exception:
                    pass
                print(
                    f"tool {params.get('name', '')} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return self._result(request_id, {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                })
        if request_id is None:
            return None
        return self._error(request_id, -32601, f"method not found: {method}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Neural memory MCP stdio server")
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--encoder-config", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    server = MCPServer(args.root, args.encoder_config)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = server.handle(request)
            except (json.JSONDecodeError, TypeError) as exc:
                response = server._error(None, -32700, f"parse error: {exc}")
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
