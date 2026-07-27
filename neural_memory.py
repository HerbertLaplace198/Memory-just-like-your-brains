#!/usr/bin/env python3
"""A dependency-free, auditable neural-style memory demo.

The network stores readable memories outside model weights while using sparse
distributed representations, winner-take-all activation and spreading
activation to decide what to recall.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Protocol
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


VECTOR_DIMS = 1024
TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+|[\u3400-\u9fff]+")
MEMORY_FORMAT = "neural-memory-record/v2"
SEMANTIC_REVIEW_FORMAT = "neural-memory-semantic-review/v1"
CONCEPT_IDENTITY_FORMAT = "neural-memory-concept-identity/v1"
CONCEPT_DECISION_FORMAT = "neural-memory-concept-decision/v1"
CONCEPT_FAMILY_FORMAT = "neural-memory-concept-family/v1"
L3F_SIZE_BONUS_CAP = 0.08
L3F_MEMBER_SIZE_SCALE = 4.0
L3F_SUPPORT_SIZE_SCALE = 12.0
L3_TOPIC_MATCH_MARGIN = 0.12
L3_TOPIC_MATCH_THRESHOLD_HASH = 0.52
L3_TOPIC_MATCH_THRESHOLD_SEMANTIC = 0.68
CONCEPT_ALIASES = {
    "asset allocation": "Asset Allocation",
    "btc": "Bitcoin",
    "cashflow": "Investment Risk and Cash Flow",
    "codex": "Codex",
    "codex配置": "Codex Configuration",
    "communication": "Collaboration Preferences",
    "economics": "Economics",
    "efficiency": "Collaboration Preferences",
    "free-trade-zone": "Free Trade Zone",
    "global": "Global",
    "handoff": "Handoff",
    "institution": "Institutional Verification",
    "investment": "Investment",
    "mcp": "MCP",
    "mdkb": "mdkb",
    "memory": "Memory System",
    "neural memory": "Memory System",
    "neural-memory": "Memory System",
    "portfolio": "Asset Allocation",
    "proactive checking": "Proactive Checking",
    "progress": "Thesis Progress",
    "risk-profile": "Investment Risk and Cash Flow",
    "setup": "Setup",
    "system": "Memory System",
    "thesis": "Thesis",
    "tools": "Tools",
    "tracker": "Investment Tracker",
    "usd": "US Dollar",
    "user-profile": "User Profile",
    "user preference": "User Preference",
    "verification": "Institutional Verification",
    "workflow": "Workflow",
    "writing workflow": "Writing Workflow",
    "writing-guide": "Thesis Writing Guide",
    "投资": "Investment",
    "机构观点": "Institutional Verification",
    "核验": "Institutional Verification",
    "记忆系统": "Memory System",
    "资产配置": "Asset Allocation",
    "主动核对": "Proactive Checking",
    "写作流程": "Writing Workflow",
    "用户偏好": "User Preference",
    "论文": "Thesis",
    "偏好": "User Preference",
    "写作指南": "Thesis Writing Guide",
    "论文写作指南": "Thesis Writing Guide",
    "神经记忆": "Memory System",
    "记忆检索": "Memory Retrieval",
}
NON_TOPIC_CONCEPTS = {"index", "status"}
TOPIC_PARENTS = {
    "Asset Allocation": "Investment",
    "Institutional Verification": "Investment",
    "Investment Risk and Cash Flow": "Investment",
    "Investment Tracker": "Investment",
    "Economics": "Thesis",
    "Thesis Progress": "Thesis",
    "Thesis Writing Guide": "Thesis",
}
USER_NOTES_RE = re.compile(
    r"<!-- USER-NOTES:START -->(.*?)<!-- USER-NOTES:END -->", re.DOTALL
)
MEMORY_REVIEW_RE = re.compile(
    r"^\s*- \[[xX]\].*?<!-- review:(confirm|revise|reject):(l1_[0-9a-f]+) -->\s*$",
    re.MULTILINE,
)
CONCEPT_REVIEW_RE = re.compile(
    r"^\s*- \[[xX]\].*?<!-- concept-review:(confirm|revise|reject):(l3_[0-9a-f]+) -->\s*$",
    re.MULTILINE,
)
CONCEPT_DUPLICATE_REVIEW_RE = re.compile(
    r"^\s*- \[[xX]\].*?<!-- concept-duplicate-review:"
    r"(merge-left|merge-right|distinct):(dup_[0-9a-f]+) -->\s*$",
    re.MULTILINE,
)
CONCEPT_FAMILY_REVIEW_RE = re.compile(
    r"^\s*- \[[xX]\].*?<!-- concept-family-review:"
    r"(confirm|reject):(l3f_[0-9a-f]+) -->\s*$",
    re.MULTILINE,
)
LOCAL_URL_OPENER = build_opener(ProxyHandler({}))


def serialized_write(method):
    """Serialize mutating operations across threads and cooperating processes."""

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.write_guard():
            return method(self, *args, **kwargs)

    return wrapped


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp defensively for biological decay calculations."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def elapsed_days(value: str | None, reference: datetime | None = None) -> float:
    """Return non-negative elapsed days without making malformed dates fatal."""
    parsed = parse_timestamp(value)
    if not parsed:
        return 0.0
    reference = reference or datetime.now(timezone.utc)
    return max(0.0, (reference - parsed).total_seconds() / 86400.0)


def short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def features(text: str) -> list[str]:
    """Return ASCII words and Chinese 2/3-grams, avoiding noisy single chars."""
    raw = TOKEN_RE.findall(text.lower())
    feats: list[str] = []
    ascii_words: list[str] = []
    for item in raw:
        if "\u3400" <= item[0] <= "\u9fff":
            if len(item) == 1:
                feats.append("zh1:" + item)
            else:
                feats.extend("zh2:" + item[index : index + 2] for index in range(len(item) - 1))
                if len(item) >= 3:
                    feats.extend("zh3:" + item[index : index + 3] for index in range(len(item) - 2))
        else:
            feats.append("en:" + item)
            ascii_words.append(item)
    feats.extend("en2:" + a + "::" + b for a, b in zip(ascii_words, ascii_words[1:]))
    return feats


def encode(text: str, dims: int = VECTOR_DIMS) -> list[float]:
    """Feature-hash text into a deterministic sparse distributed vector."""
    vector = [0.0] * dims
    for token in features(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[slot] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class TextEncoder(Protocol):
    """Small plug-in boundary for local semantic encoders."""

    name: str
    dimensions: int
    gate_threshold: float
    family_gate_threshold: float
    family_size_bonus_cap: float

    def encode(self, text: str) -> list[float]: ...

    def encode_many(self, texts: list[str]) -> list[list[float]]: ...


class HashEncoder:
    """Portable fallback: deterministic, private and dependency-free."""

    name = "feature-hash-v1"
    gate_threshold = 0.48
    family_gate_threshold = 0.23
    family_size_bonus_cap = L3F_SIZE_BONUS_CAP

    def __init__(
        self,
        dimensions: int = VECTOR_DIMS,
        gate_threshold: float = 0.48,
        family_size_bonus_cap: float = L3F_SIZE_BONUS_CAP,
    ):
        if not 0.0 <= family_size_bonus_cap <= 0.20:
            raise ValueError("family size bonus cap must be between 0 and 0.20")
        self.dimensions = dimensions
        self.gate_threshold = gate_threshold
        self.family_size_bonus_cap = family_size_bonus_cap

    def encode(self, text: str) -> list[float]:
        return encode(text, self.dimensions)

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]


class LocalHTTPEncoder:
    """Embedding adapter restricted to loopback endpoints."""

    def __init__(
        self,
        provider: str,
        endpoint: str,
        model: str,
        dimensions: int,
        timeout: float = 30.0,
        gate_threshold: float = 0.30,
        family_gate_threshold: float = 0.42,
        family_size_bonus_cap: float = L3F_SIZE_BONUS_CAP,
    ):
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or parsed.hostname not in (
            "127.0.0.1",
            "localhost",
            "::1",
        ):
            raise ValueError("local encoder endpoint must use localhost/loopback")
        if provider not in ("ollama", "openai-compatible"):
            raise ValueError(f"unsupported local encoder provider: {provider}")
        if dimensions <= 0:
            raise ValueError("encoder dimensions must be positive")
        if not 0.0 <= family_size_bonus_cap <= 0.20:
            raise ValueError("family size bonus cap must be between 0 and 0.20")
        self.provider = provider
        self.endpoint = endpoint
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.gate_threshold = gate_threshold
        self.family_gate_threshold = family_gate_threshold
        self.family_size_bonus_cap = family_size_bonus_cap
        self.name = f"{provider}:{model}"

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model, "input": texts}
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with LOCAL_URL_OPENER.open(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        try:
            if self.provider == "ollama":
                vectors = payload["embeddings"]
            else:
                vectors = [item["embedding"] for item in payload["data"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"invalid embedding response from {self.endpoint}") from exc
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("embedding response count does not match input count")
        return [[float(value) for value in vector] for vector in vectors]


def load_encoder_config(path: Path) -> TextEncoder:
    config = json.loads(path.read_text(encoding="utf-8"))
    provider = str(config.get("provider", "hash"))
    if provider == "hash":
        return HashEncoder(
            int(config.get("dimensions", VECTOR_DIMS)),
            float(config.get("gate_threshold", 0.48)),
            float(config.get("family_size_bonus_cap", L3F_SIZE_BONUS_CAP)),
        )
    return LocalHTTPEncoder(
        provider,
        str(config["endpoint"]),
        str(config["model"]),
        int(config["dimensions"]),
        float(config.get("timeout", 30.0)),
        float(config.get("gate_threshold", 0.30)),
        float(config.get("family_gate_threshold", 0.42)),
        float(config.get("family_size_bonus_cap", L3F_SIZE_BONUS_CAP)),
    )


def resolve_encoder(root: Path, explicit_config: Path | None = None) -> TextEncoder:
    config_path = explicit_config or root.resolve() / "encoder.json"
    return load_encoder_config(config_path) if config_path.is_file() else HashEncoder()


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def compact(text: str, width: int = 100) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def rough_tokens(text: str) -> int:
    """A deliberately simple local estimate, suitable only for A/B comparison."""
    chinese = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    other = len(text) - chinese
    return chinese + math.ceil(other / 4)


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "-", text).strip().strip(".")
    return compact(cleaned or "未命名", 80)


def canonical_concept(text: str) -> str:
    """Normalize known bilingual or legacy aliases to one L3 topic label."""
    label = " ".join(text.strip().split())
    return CONCEPT_ALIASES.get(label.casefold(), label)


def normalized_concept_key(label: str) -> str:
    normalized = canonical_concept(label).casefold()
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def canonical_concepts(values: list[str]) -> list[str]:
    """Normalize L3 labels and discard structural tags that are not topics."""
    result: list[str] = []
    for value in values:
        label = canonical_concept(value)
        if not label or label.casefold() in NON_TOPIC_CONCEPTS or label in result:
            continue
        result.append(label)
    if "Memory System" in result and "Codex" in result:
        result.remove("Codex")
    if "Investment Tracker" in result:
        result = [label for label in result if label not in {"Bitcoin", "US Dollar"}]
    if "Thesis Progress" in result and "Free Trade Zone" in result:
        result.remove("Free Trade Zone")
    return result


def is_english_label(text: str) -> bool:
    """Return true for printable ASCII labels containing at least one letter."""
    return text.isascii() and bool(re.search(r"[A-Za-z]", text))


def english_only_labels(values: Iterable[str]) -> list[str]:
    """Normalize label whitespace and retain only English structural labels."""
    return list(
        dict.fromkeys(
            label
            for value in values
            if (label := " ".join(str(value).strip().split())) and is_english_label(label)
        )
    )


def require_english_labels(values: Iterable[str], field: str) -> list[str]:
    """Reject new non-English L3/L4 labels instead of silently indexing them."""
    labels = list(dict.fromkeys(" ".join(str(value).strip().split()) for value in values))
    labels = [label for label in labels if label]
    invalid = [label for label in labels if not is_english_label(label)]
    if invalid:
        raise ValueError(f"{field} must use English-only labels: {invalid!r}")
    return labels


def enrich_query_with_known_concepts(query: str) -> str:
    """Add canonical English aliases so multilingual queries can reach L3F."""
    lowered = query.casefold()
    aliases = [
        canonical
        for alias, canonical in sorted(
            CONCEPT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
        )
        if not alias.isascii() and alias in lowered
    ]
    return " ".join([query, *dict.fromkeys(aliases)]).strip()


def write_record(path: Path, metadata: dict[str, object], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---json\n"
        + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        + "\n---\n\n"
        + body.strip()
        + "\n",
        encoding="utf-8",
    )


def read_record(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---json\n(.+?)\n---\n\n?(.*)\Z", text, re.DOTALL)
    if not match:
        raise ValueError(f"invalid canonical memory record: {path}")
    return json.loads(match.group(1)), match.group(2).rstrip()


@dataclass
class ActivatedNeuron:
    id: str
    layer: int
    label: str
    summary: str
    status: str
    confidence: float
    importance: float
    evidence_id: str | None
    activation: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    lexical_score: float = 0.0
    direct_activation: float = 0.0
    spread_activation: float = 0.0
    stability: float = 0.5
    retention: float = 1.0


class NeuralMemory:
    def __init__(
        self,
        root: Path,
        encoder: TextEncoder | None = None,
        allow_encoder_mismatch: bool = False,
    ):
        self.root = root.resolve()
        self.lock_path = self.root / ".write.lock"
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._lock_handle = None
        self.allow_encoder_mismatch = allow_encoder_mismatch
        self.db_path = self.root / "memory.sqlite3"
        self.vault_dir = self.root / "vault"
        self.backend_dir = self.root / ".neural-memory"
        self.evidence_dir = self.vault_dir / "evidence"
        self.memory_dir = self.vault_dir / "memories"
        self.semantic_review_dir = self.backend_dir / "semantic-reviews"
        self.concept_identity_dir = self.backend_dir / "concept-identities"
        self.concept_decision_dir = self.backend_dir / "concept-decisions"
        self.concept_family_dir = self.backend_dir / "concept-families"
        self.rejected_dir = self.vault_dir / ".rejected"
        self.obsidian_dir = self.root / "obsidian-view"
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_review_dir.mkdir(parents=True, exist_ok=True)
        self.concept_identity_dir.mkdir(parents=True, exist_ok=True)
        self.concept_decision_dir.mkdir(parents=True, exist_ok=True)
        self.concept_family_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA busy_timeout = 5000")
        self.encoder = encoder or HashEncoder()
        with self.write_guard():
            self.db.execute("PRAGMA journal_mode = WAL")
            self.db.execute("PRAGMA synchronous = FULL")
            self._upgrade_layer_schema()
            self._schema()
            self._register_encoder()

    def close(self) -> None:
        self.db.close()

    @contextlib.contextmanager
    def write_guard(self, timeout: float = 5.0):
        """Re-entrant advisory lock used by every public mutation path."""
        with self._thread_lock:
            if self._lock_depth == 0:
                handle = self.lock_path.open("a+")
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            handle.close()
                            raise TimeoutError(f"memory write lock timed out: {self.lock_path}")
                        time.sleep(0.05)
                self._lock_handle = handle
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._lock_handle is not None:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    self._lock_handle.close()
                    self._lock_handle = None

    def _upgrade_layer_schema(self) -> None:
        row = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='neurons'"
        ).fetchone()
        if not row or "BETWEEN 1 AND 3" not in (row["sql"] or ""):
            return
        self.db.execute("PRAGMA foreign_keys = OFF")
        self.db.executescript(
            """
            DROP INDEX IF EXISTS idx_neurons_layer;
            DROP INDEX IF EXISTS idx_neurons_status;
            DROP INDEX IF EXISTS idx_synapses_source;
            ALTER TABLE synapses RENAME TO synapses_legacy;
            ALTER TABLE neurons RENAME TO neurons_legacy;
            CREATE TABLE neurons (
                id TEXT PRIMARY KEY,
                layer INTEGER NOT NULL CHECK(layer BETWEEN 1 AND 6),
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                vector TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('proposed','confirmed','rejected','stale','archived')),
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                evidence_id TEXT REFERENCES evidence(id),
                created_at TEXT NOT NULL,
                last_used TEXT
            );
            INSERT INTO neurons SELECT * FROM neurons_legacy;
            CREATE TABLE synapses (
                source_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                last_fired TEXT,
                PRIMARY KEY(source_id, target_id, relation)
            );
            INSERT INTO synapses SELECT * FROM synapses_legacy;
            DROP TABLE synapses_legacy;
            DROP TABLE neurons_legacy;
            """
        )
        self.db.commit()
        self.db.execute("PRAGMA foreign_keys = ON")

    def _schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS neurons (
                id TEXT PRIMARY KEY,
                layer INTEGER NOT NULL CHECK(layer BETWEEN 1 AND 6),
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                vector TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('proposed','confirmed','rejected','stale','archived')),
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                evidence_id TEXT REFERENCES evidence(id),
                created_at TEXT NOT NULL,
                last_used TEXT,
                expires_at TEXT,
                stability REAL NOT NULL DEFAULT 0.5,
                reactivation_count INTEGER NOT NULL DEFAULT 0,
                last_reactivated TEXT
            );
            CREATE TABLE IF NOT EXISTS synapses (
                source_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                last_fired TEXT,
                fire_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(source_id, target_id, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_neurons_layer ON neurons(layer);
            CREATE INDEX IF NOT EXISTS idx_neurons_status ON neurons(status);
            CREATE INDEX IF NOT EXISTS idx_synapses_source ON synapses(source_id);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL CHECK(relation IN ('supersedes','conflicts_with','duplicates')),
                status TEXT NOT NULL CHECK(status IN ('pending','confirmed','rejected')),
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_relations_status ON memory_relations(status);
            CREATE TABLE IF NOT EXISTS maintenance_issues (
                id TEXT PRIMARY KEY,
                neuron_id TEXT,
                kind TEXT NOT NULL,
                severity TEXT NOT NULL CHECK(severity IN ('info','warning','critical')),
                details TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('open','resolved','ignored')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_issues_status ON maintenance_issues(status);
            CREATE TABLE IF NOT EXISTS annotation_proposals (
                id TEXT PRIMARY KEY,
                page_path TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                notes_hash TEXT NOT NULL,
                notes TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected')),
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                resulting_neuron_id TEXT,
                UNIQUE(page_path, notes_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_annotation_status ON annotation_proposals(status);
            CREATE TABLE IF NOT EXISTS semantic_reviews (
                concept_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('confirmed','rejected')),
                member_ids TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_identities (
                concept_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('proposed','confirmed','rejected','merged')),
                member_ids TEXT NOT NULL,
                merged_into_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_duplicate_reviews (
                id TEXT PRIMARY KEY,
                left_key TEXT NOT NULL,
                right_key TEXT NOT NULL,
                left_label TEXT NOT NULL,
                right_label TEXT NOT NULL,
                vector_score REAL NOT NULL,
                member_overlap REAL NOT NULL,
                label_score REAL NOT NULL,
                combined_score REAL NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','merged','distinct')),
                survivor_key TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                UNIQUE(left_key,right_key)
            );
            CREATE INDEX IF NOT EXISTS idx_concept_duplicates_status
                ON concept_duplicate_reviews(status);
            CREATE TABLE IF NOT EXISTS concept_aliases (
                alias_key TEXT PRIMARY KEY,
                alias_label TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                canonical_label TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_families (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('proposed','confirmed','rejected')),
                member_keys TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(neurons)")}
        if "expires_at" not in columns:
            self.db.execute("ALTER TABLE neurons ADD COLUMN expires_at TEXT")
        if "stability" not in columns:
            self.db.execute(
                "ALTER TABLE neurons ADD COLUMN stability REAL NOT NULL DEFAULT 0.5"
            )
        if "reactivation_count" not in columns:
            self.db.execute(
                "ALTER TABLE neurons ADD COLUMN reactivation_count INTEGER NOT NULL DEFAULT 0"
            )
        if "last_reactivated" not in columns:
            self.db.execute("ALTER TABLE neurons ADD COLUMN last_reactivated TEXT")
        synapse_columns = {
            row[1] for row in self.db.execute("PRAGMA table_info(synapses)")
        }
        if "fire_count" not in synapse_columns:
            self.db.execute(
                "ALTER TABLE synapses ADD COLUMN fire_count INTEGER NOT NULL DEFAULT 0"
            )
        self._load_semantic_reviews()
        self._load_concept_identities()
        self._load_concept_decisions()
        self._load_concept_families()
        self.db.commit()

    def _vector(self, row: sqlite3.Row) -> list[float]:
        return json.loads(row["vector"])

    def _encode(self, text: str) -> list[float]:
        return self._normalize_vector(list(self.encoder.encode(text)))

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.encoder.dimensions:
            raise ValueError(
                f"encoder {self.encoder.name} declared {self.encoder.dimensions} dimensions "
                f"but returned {len(vector)}"
            )
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [float(value) / norm for value in vector]

    def _encode_many(self, texts: list[str]) -> list[list[float]]:
        method = getattr(self.encoder, "encode_many", None)
        vectors = method(texts) if method else [self.encoder.encode(text) for text in texts]
        if len(vectors) != len(texts):
            raise ValueError("encoder returned the wrong number of vectors")
        return [self._normalize_vector(list(vector)) for vector in vectors]

    def _register_encoder(self) -> None:
        stored_name = self._get_meta("encoder_name")
        stored_dims = self._get_meta("encoder_dimensions")
        neuron_count = self.db.execute("SELECT count(*) FROM neurons").fetchone()[0]
        mismatch = neuron_count and stored_name and (
            stored_name != self.encoder.name
            or int(stored_dims or 0) != self.encoder.dimensions
        )
        if mismatch:
            if not self.allow_encoder_mismatch:
                raise ValueError(
                    f"index uses {stored_name}/{stored_dims}; supplied encoder is "
                    f"{self.encoder.name}/{self.encoder.dimensions}. Run reencode with its config."
                )
            return
        self._set_meta("encoder_name", self.encoder.name)
        self._set_meta("encoder_dimensions", str(self.encoder.dimensions))
        self.db.commit()

    @serialized_write
    def reencode_all(self, config_source: Path | None = None) -> dict[str, object]:
        """Atomically migrate every neuron vector and rebuild similarity synapses."""
        rows = self.db.execute("SELECT id,label,summary FROM neurons").fetchall()
        old_name = self._get_meta("encoder_name")
        old_dimensions = self._get_meta("encoder_dimensions")
        try:
            self.db.execute("BEGIN IMMEDIATE")
            representations = self._encode_many(
                [row["label"] + " " + row["summary"] for row in rows]
            )
            for row, representation in zip(rows, representations):
                self.db.execute(
                    "UPDATE neurons SET vector=? WHERE id=?",
                    (json.dumps(representation, separators=(",", ":")), row["id"]),
                )
            self.db.execute("DELETE FROM synapses WHERE relation='association'")
            atoms = self.db.execute(
                "SELECT * FROM neurons WHERE layer=1 AND status!='rejected'"
            ).fetchall()
            for index, left in enumerate(atoms):
                for right in atoms[index + 1 :]:
                    score = max(0.0, cosine(self._vector(left), self._vector(right)))
                    if score >= 0.18:
                        self._connect(
                            left["id"], right["id"], "association", min(0.75, score + 0.2)
                        )
            self._set_meta("encoder_name", self.encoder.name)
            self._set_meta("encoder_dimensions", str(self.encoder.dimensions))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        if config_source:
            config_text = config_source.read_text(encoding="utf-8")
            target = self.root / "encoder.json"
            temporary = self.root / "encoder.json.tmp"
            temporary.write_text(config_text, encoding="utf-8")
            os.replace(temporary, target)
        return {
            "neurons": len(rows),
            "from": {"name": old_name, "dimensions": old_dimensions},
            "to": {"name": self.encoder.name, "dimensions": self.encoder.dimensions},
            "synapses": self.db.execute("SELECT count(*) FROM synapses").fetchone()[0],
        }

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _load_semantic_reviews(self) -> int:
        """Rebuild semantic review decisions from canonical Markdown files."""
        loaded = 0
        for path in sorted(self.semantic_review_dir.glob("*.md")):
            metadata, _ = read_record(path)
            if metadata.get("format") != SEMANTIC_REVIEW_FORMAT:
                continue
            concept_id = str(metadata.get("concept_id", ""))
            status = str(metadata.get("status", ""))
            member_ids = metadata.get("member_ids", [])
            reviewed_at = str(metadata.get("reviewed_at", ""))
            if not concept_id or status not in {"confirmed", "rejected"}:
                continue
            self.db.execute(
                """INSERT INTO semantic_reviews(concept_id,status,member_ids,reviewed_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(concept_id) DO UPDATE SET
                     status=excluded.status,
                     member_ids=excluded.member_ids,
                     reviewed_at=excluded.reviewed_at""",
                (
                    concept_id,
                    status,
                    json.dumps(member_ids, ensure_ascii=False),
                    reviewed_at or now(),
                ),
            )
            loaded += 1
        return loaded

    def _load_concept_identities(self) -> int:
        loaded = 0
        for path in sorted(self.concept_identity_dir.glob("*.md")):
            metadata, _ = read_record(path)
            if metadata.get("format") != CONCEPT_IDENTITY_FORMAT:
                continue
            concept_id = str(metadata.get("concept_id", ""))
            status = str(metadata.get("status", "proposed"))
            if not concept_id or status not in {
                "proposed",
                "confirmed",
                "rejected",
                "merged",
            }:
                continue
            member_ids = sorted(
                str(item) for item in metadata.get("member_ids", []) if str(item)
            )
            created_at = str(metadata.get("created_at", "")) or now()
            updated_at = str(metadata.get("updated_at", "")) or created_at
            self.db.execute(
                """INSERT INTO concept_identities
                   (concept_id,status,member_ids,merged_into_key,created_at,updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(concept_id) DO UPDATE SET
                     status=excluded.status,
                     member_ids=excluded.member_ids,
                     merged_into_key=excluded.merged_into_key,
                     created_at=excluded.created_at,
                     updated_at=excluded.updated_at""",
                (
                    concept_id,
                    status,
                    json.dumps(member_ids, ensure_ascii=False),
                    str(metadata.get("merged_into_key", "")) or None,
                    created_at,
                    updated_at,
                ),
            )
            loaded += 1
        return loaded

    def _load_concept_decisions(self) -> int:
        loaded = 0
        for path in sorted(self.concept_decision_dir.glob("*.md")):
            metadata, _ = read_record(path)
            if metadata.get("format") != CONCEPT_DECISION_FORMAT:
                continue
            decision_id = str(metadata.get("id", ""))
            status = str(metadata.get("status", ""))
            left_key = str(metadata.get("left_key", ""))
            right_key = str(metadata.get("right_key", ""))
            if (
                not decision_id
                or status not in {"merged", "distinct"}
                or not left_key
                or not right_key
            ):
                continue
            left_key, right_key = sorted((left_key, right_key))
            self.db.execute(
                """INSERT INTO concept_duplicate_reviews
                   (id,left_key,right_key,left_label,right_label,vector_score,
                    member_overlap,label_score,combined_score,status,survivor_key,
                    created_at,reviewed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(left_key,right_key) DO UPDATE SET
                     id=excluded.id,
                     left_label=excluded.left_label,
                     right_label=excluded.right_label,
                     status=excluded.status,
                     survivor_key=excluded.survivor_key,
                     reviewed_at=excluded.reviewed_at""",
                (
                    decision_id,
                    left_key,
                    right_key,
                    str(metadata.get("left_label", left_key)),
                    str(metadata.get("right_label", right_key)),
                    float(metadata.get("vector_score", 0.0)),
                    float(metadata.get("member_overlap", 0.0)),
                    float(metadata.get("label_score", 0.0)),
                    float(metadata.get("combined_score", 0.0)),
                    status,
                    str(metadata.get("survivor_key", "")) or None,
                    str(metadata.get("created_at", "")) or now(),
                    str(metadata.get("reviewed_at", "")) or now(),
                ),
            )
            if status == "merged":
                survivor_key = str(metadata.get("survivor_key", ""))
                survivor_label = str(metadata.get("survivor_label", ""))
                loser_label = str(metadata.get("loser_label", ""))
                if survivor_key and survivor_label and loser_label:
                    self.db.execute(
                        """INSERT INTO concept_aliases
                           (alias_key,alias_label,canonical_key,canonical_label,
                            decision_id,created_at)
                           VALUES(?,?,?,?,?,?)
                           ON CONFLICT(alias_key) DO UPDATE SET
                             alias_label=excluded.alias_label,
                             canonical_key=excluded.canonical_key,
                             canonical_label=excluded.canonical_label,
                             decision_id=excluded.decision_id""",
                        (
                            normalized_concept_key(loser_label),
                            loser_label,
                            survivor_key,
                            survivor_label,
                            decision_id,
                            str(metadata.get("reviewed_at", "")) or now(),
                        ),
                    )
            loaded += 1
        return loaded

    def _load_concept_families(self) -> int:
        """Rebuild the L3F grouping layer from canonical Markdown records."""
        loaded = 0
        for path in sorted(self.concept_family_dir.glob("*.md")):
            metadata, body = read_record(path)
            if metadata.get("format") != CONCEPT_FAMILY_FORMAT:
                continue
            family_id = str(metadata.get("id", ""))
            status = str(metadata.get("status", "proposed"))
            if not family_id or status not in {"proposed", "confirmed", "rejected"}:
                continue
            member_keys = sorted(
                str(item) for item in metadata.get("member_keys", []) if str(item)
            )
            created_at = str(metadata.get("created_at", "")) or now()
            updated_at = str(metadata.get("updated_at", "")) or created_at
            label = str(metadata.get("label", "")) or f"概念家族 {family_id[4:]}"
            self.db.execute(
                """INSERT INTO concept_families
                   (id,label,summary,status,member_keys,active,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     label=excluded.label,
                     summary=excluded.summary,
                     status=excluded.status,
                     member_keys=excluded.member_keys,
                     active=excluded.active,
                     created_at=excluded.created_at,
                     updated_at=excluded.updated_at""",
                (
                    family_id,
                    label,
                    body.strip(),
                    status,
                    json.dumps(member_keys, ensure_ascii=False),
                    int(bool(metadata.get("active", True))),
                    created_at,
                    updated_at,
                ),
            )
            loaded += 1
        return loaded

    def _resolve_concept_alias(self, label: str) -> str:
        canonical = canonical_concept(label)
        row = self.db.execute(
            "SELECT canonical_label FROM concept_aliases WHERE alias_key=?",
            (normalized_concept_key(canonical),),
        ).fetchone()
        return row["canonical_label"] if row else canonical

    @staticmethod
    def _stable_named_concept_id(label: str) -> str:
        digest = hashlib.sha256(
            f"named-concept|{normalized_concept_key(label)}".encode("utf-8")
        ).hexdigest()[:10]
        return f"l3_{digest}"

    def _find_named(self, layer: int, label: str) -> sqlite3.Row | None:
        if layer == 3:
            label = self._resolve_concept_alias(label)
        return self.db.execute(
            "SELECT * FROM neurons WHERE layer=? AND lower(label)=lower(?) AND status!='rejected'",
            (layer, label),
        ).fetchone()

    @staticmethod
    def _topic_terms(text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", text.casefold())
            if len(term) > 1
        }

    def _topic_match_threshold(self) -> float:
        configured = getattr(self.encoder, "topic_match_threshold", None)
        if configured is not None:
            return max(0.0, min(1.0, float(configured)))
        if isinstance(self.encoder, HashEncoder):
            return L3_TOPIC_MATCH_THRESHOLD_HASH
        return L3_TOPIC_MATCH_THRESHOLD_SEMANTIC

    def _rank_topic_matches(
        self,
        query: str,
        concepts: list[sqlite3.Row],
    ) -> list[tuple[float, sqlite3.Row]]:
        """Rank active L3 topics against one memory or topic hint.

        A memory may have several strong topic routes.  Keep all close
        matches instead of forcing the memory into a single best topic.
        """
        if not query.strip() or not concepts:
            return []
        query_vector = self._encode(query)
        query_terms = self._topic_terms(query)
        scored: list[tuple[float, sqlite3.Row]] = []
        for concept in concepts:
            prototype, _ = self._concept_prototype(concept)
            content_score = max(0.0, cosine(query_vector, prototype))
            representation_score = max(
                0.0,
                cosine(query_vector, self._vector(concept)),
            )
            label_terms = self._topic_terms(str(concept["label"]))
            lexical_score = (
                len(query_terms & label_terms) / len(label_terms)
                if label_terms
                else 0.0
            )
            score = (
                0.55 * content_score
                + 0.30 * representation_score
                + 0.15 * lexical_score
            )
            scored.append((score, concept))
        scored.sort(key=lambda item: (item[0], str(item[1]["label"])), reverse=True)
        threshold = self._topic_match_threshold()
        eligible = [item for item in scored if item[0] >= threshold]
        if not eligible:
            return []
        best = eligible[0][0]
        margin = getattr(self.encoder, "topic_match_margin", L3_TOPIC_MATCH_MARGIN)
        margin = max(0.0, min(1.0, float(margin)))
        return [item for item in eligible if item[0] >= best - margin]

    def _resolve_memory_topics(
        self,
        text: str,
        requested_topics: Iterable[str] = (),
    ) -> list[str]:
        """Reuse relevant L3 topics before accepting new topic labels.

        Explicit topic labels are hints, not permission to create a duplicate
        L3.  Existing topics are matched both from those hints and from the
        memory text, and every strong match is retained because one memory can
        legitimately belong to multiple topics.  Explicit hints that do not
        match any existing topic remain as new candidates, while matched hints
        are replaced by the existing canonical labels.  A memory without a
        topic hint remains uncategorized until consolidation can form an
        emergent candidate.
        """
        requested = [
            self._resolve_concept_alias(label)
            for label in canonical_concepts(list(requested_topics))
        ]
        requested = list(dict.fromkeys(label for label in requested if label.strip()))
        concepts = self.db.execute(
            """SELECT * FROM neurons
               WHERE layer=3 AND status NOT IN ('rejected','archived','stale')
               ORDER BY label,id"""
        ).fetchall()
        if not concepts:
            return requested

        matched: dict[str, sqlite3.Row] = {}
        unmatched_requested: list[str] = []
        for label in requested:
            exact = self._find_named(3, label)
            if exact:
                matched[str(exact["id"])] = exact
            hint_matches = self._rank_topic_matches(label, concepts)
            for _, concept in hint_matches:
                matched[str(concept["id"])] = concept
            if not exact and not hint_matches:
                unmatched_requested.append(label)

        for _, concept in self._rank_topic_matches(text, concepts):
            matched[str(concept["id"])] = concept

        existing_labels = list(
            dict.fromkeys(str(concept["label"]) for concept in matched.values())
        )
        if not existing_labels:
            return requested

        # Preserve genuinely new explicit aspects, but never carry forward a
        # hint that already matched an existing topic.  This allows one memory
        # to extend its topic set without recreating the topics it already has.
        return list(dict.fromkeys(existing_labels + unmatched_requested))

    def _create_neuron(
        self,
        layer: int,
        label: str,
        summary: str,
        status: str,
        confidence: float,
        importance: float,
        evidence_id: str | None = None,
        neuron_id: str | None = None,
        expires_at: str | None = None,
        stability: float = 0.5,
        created_at: str | None = None,
    ) -> str:
        neuron_id = neuron_id or short_id(f"l{layer}")
        representation = self._encode(label + " " + summary)
        self.db.execute(
            """INSERT INTO neurons
               (id, layer, label, summary, vector, status, confidence, importance,
                evidence_id, created_at, expires_at, stability)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                neuron_id,
                layer,
                label,
                summary,
                json.dumps(representation, separators=(",", ":")),
                status,
                confidence,
                importance,
                evidence_id,
                created_at or now(),
                expires_at,
                max(0.0, min(1.0, stability)),
            ),
        )
        return neuron_id

    @staticmethod
    def _relation_id(source_id: str, target_id: str, relation: str) -> str:
        digest = hashlib.sha256(f"{source_id}|{target_id}|{relation}".encode()).hexdigest()[:12]
        return f"rel_{digest}"

    def _add_relation(
        self, source_id: str, target_id: str, relation: str, reason: str
    ) -> str:
        relation_id = self._relation_id(source_id, target_id, relation)
        self.db.execute(
            """INSERT INTO memory_relations
               (id,source_id,target_id,relation,status,reason,created_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET reason=excluded.reason""",
            (relation_id, source_id, target_id, relation, "pending", reason, now()),
        )
        return relation_id

    def _issue(
        self, neuron_id: str | None, kind: str, severity: str, details: str
    ) -> str:
        fingerprint = hashlib.sha256(
            f"{neuron_id}|{kind}|{details}".encode("utf-8")
        ).hexdigest()[:12]
        issue_id = f"issue_{fingerprint}"
        self.db.execute(
            """INSERT OR IGNORE INTO maintenance_issues
               (id,neuron_id,kind,severity,details,status,created_at)
               VALUES(?,?,?,?,?,'open',?)""",
            (issue_id, neuron_id, kind, severity, details, now()),
        )
        return issue_id

    def _connect(self, left: str, right: str, relation: str, weight: float) -> None:
        for source, target in ((left, right), (right, left)):
            self.db.execute(
                """INSERT INTO synapses(source_id,target_id,relation,weight,last_fired)
                   VALUES(?,?,?,?,NULL)
                   ON CONFLICT(source_id,target_id,relation)
                   DO UPDATE SET weight=min(1.0, synapses.weight + excluded.weight * 0.15)""",
                (source, target, relation, weight),
            )

    def _l3_routes_for_l1(self, neuron_id: str) -> set[str]:
        """Return semantic concepts reachable upward from one atomic trace."""
        return {
            row["id"]
            for row in self.db.execute(
                """WITH RECURSIVE upward(id,layer) AS (
                       SELECT id,layer FROM neurons WHERE id=?
                       UNION
                       SELECT n.id,n.layer
                       FROM upward u
                       JOIN synapses s ON s.source_id=u.id
                       JOIN neurons n ON n.id=s.target_id
                       WHERE n.layer>u.layer
                         AND n.status NOT IN ('rejected','archived','stale')
                   )
                   SELECT id FROM upward WHERE layer=3""",
                (neuron_id,),
            ).fetchall()
        }

    @staticmethod
    def _concept_key(row: sqlite3.Row) -> str:
        if str(row["label"]).startswith("Emergent Concept "):
            return f"emergent:{row['id']}"
        return f"label:{normalized_concept_key(str(row['label']))}"

    def _find_concept_by_key(self, concept_key: str) -> sqlite3.Row | None:
        if concept_key.startswith("emergent:"):
            return self.db.execute(
                """SELECT * FROM neurons
                   WHERE id=? AND layer=3
                     AND status NOT IN ('rejected','archived')""",
                (concept_key.split(":", 1)[1],),
            ).fetchone()
        if concept_key.startswith("label:"):
            normalized = concept_key.split(":", 1)[1]
            return next(
                (
                    row
                    for row in self.db.execute(
                        """SELECT * FROM neurons
                           WHERE layer=3
                             AND status NOT IN ('rejected','archived')"""
                    )
                    if normalized_concept_key(str(row["label"])) == normalized
                ),
                None,
            )
        return None

    def _persist_concept_identity(
        self,
        concept_id: str,
        member_ids: list[str],
        status: str,
        merged_into_key: str | None = None,
    ) -> None:
        existing = self.db.execute(
            "SELECT created_at FROM concept_identities WHERE concept_id=?",
            (concept_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now()
        updated_at = now()
        member_ids = sorted(set(member_ids))
        write_record(
            self.concept_identity_dir / f"{concept_id}.md",
            {
                "format": CONCEPT_IDENTITY_FORMAT,
                "concept_id": concept_id,
                "status": status,
                "member_ids": member_ids,
                "merged_into_key": merged_into_key or "",
                "created_at": created_at,
                "updated_at": updated_at,
            },
            (
                f"Stable identity for emergent concept {concept_id}. "
                f"Current support: {len(member_ids)} atomic memory traces."
            ),
        )
        self.db.execute(
            """INSERT INTO concept_identities
               (concept_id,status,member_ids,merged_into_key,created_at,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(concept_id) DO UPDATE SET
                 status=excluded.status,
                 member_ids=excluded.member_ids,
                 merged_into_key=excluded.merged_into_key,
                 updated_at=excluded.updated_at""",
            (
                concept_id,
                status,
                json.dumps(member_ids, ensure_ascii=False),
                merged_into_key,
                created_at,
                updated_at,
            ),
        )

    def _remove_emergent_concepts(self) -> int:
        """Remove rebuildable L3 abstractions before recomputing them."""
        rows = self.db.execute(
            """SELECT DISTINCT n.id
               FROM neurons n JOIN synapses s
                 ON (s.source_id=n.id OR s.target_id=n.id)
               WHERE n.layer=3 AND s.relation='emergent_member_of'"""
        ).fetchall()
        concept_ids = [row["id"] for row in rows]
        if not concept_ids:
            return 0
        placeholders = ",".join("?" for _ in concept_ids)
        self.db.execute(
            f"DELETE FROM synapses WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            (*concept_ids, *concept_ids),
        )
        self.db.execute(
            f"DELETE FROM neurons WHERE id IN ({placeholders})", concept_ids
        )
        return len(concept_ids)

    def _rebuild_emergent_concepts(
        self,
        min_support: int = 3,
        similarity_threshold: float | None = None,
    ) -> int:
        """Form proposed L3 concepts from repeated, similar confirmed L1 traces."""
        identities = self.db.execute(
            "SELECT * FROM concept_identities ORDER BY created_at,concept_id"
        ).fetchall()
        self._remove_emergent_concepts()
        atoms = self.db.execute(
            """SELECT * FROM neurons
               WHERE layer=1 AND status='confirmed'
               ORDER BY created_at,id"""
        ).fetchall()
        if len(atoms) < min_support:
            return 0

        rows_by_id = {row["id"]: row for row in atoms}
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else (0.34 if isinstance(self.encoder, HashEncoder) else 0.75)
        )
        clusters: list[list[str]] = []
        for atom in atoms:
            choices: list[tuple[float, int]] = []
            for index, cluster in enumerate(clusters):
                similarities = [
                    max(
                        0.0,
                        cosine(
                            self._vector(atom),
                            self._vector(rows_by_id[member_id]),
                        ),
                    )
                    for member_id in cluster
                ]
                if similarities and min(similarities) >= threshold:
                    choices.append((sum(similarities) / len(similarities), index))
            if choices:
                _, best_index = max(choices)
                clusters[best_index].append(atom["id"])
            else:
                clusters.append([atom["id"]])
        components = [
            sorted(cluster) for cluster in clusters if len(cluster) >= min_support
        ]

        created = 0
        used_identity_ids: set[str] = set()
        for member_ids in components:
            shared_routes: set[str] | None = None
            for member_id in member_ids:
                routes = self._l3_routes_for_l1(member_id)
                shared_routes = routes if shared_routes is None else shared_routes & routes
            if shared_routes:
                continue

            member_set = set(member_ids)
            identity_choices: list[tuple[float, sqlite3.Row]] = []
            for candidate in identities:
                if candidate["concept_id"] in used_identity_ids:
                    continue
                old_members = set(json.loads(candidate["member_ids"]))
                overlap = len(member_set & old_members) / max(
                    1, len(member_set | old_members)
                )
                if overlap >= 0.60:
                    identity_choices.append((overlap, candidate))
            identity = (
                max(identity_choices, key=lambda item: item[0])[1]
                if identity_choices
                else None
            )
            if identity:
                concept_id = str(identity["concept_id"])
            else:
                digest = hashlib.sha256(
                    "|".join(member_ids).encode("utf-8")
                ).hexdigest()[:10]
                concept_id = f"l3_{digest}"
            used_identity_ids.add(concept_id)
            digest = concept_id.split("_", 1)[1]
            summaries = [rows_by_id[item]["summary"].strip() for item in member_ids]
            summary = (
                f"Emergent semantic concept supported by {len(member_ids)} confirmed "
                f"memory traces: {compact(' | '.join(summaries), 420)}"
            )
            confidence = min(0.88, 0.52 + 0.08 * len(member_ids))
            importance = sum(
                float(rows_by_id[item]["importance"]) for item in member_ids
            ) / len(member_ids)
            stability = min(0.95, 1.0 - math.exp(-len(member_ids) / 3.0))
            reviewed = self.db.execute(
                "SELECT status FROM semantic_reviews WHERE concept_id=?",
                (concept_id,),
            ).fetchone()
            identity_status = str(identity["status"]) if identity else "proposed"
            merged_into_key = (
                str(identity["merged_into_key"])
                if identity and identity["merged_into_key"]
                else None
            )
            if identity_status == "merged" and merged_into_key:
                survivor = self._find_concept_by_key(merged_into_key)
                if survivor:
                    for member_id in member_ids:
                        self._connect(
                            member_id,
                            survivor["id"],
                            "emergent_member_of",
                            min(0.92, 0.62 + 0.06 * len(member_ids)),
                        )
                self._persist_concept_identity(
                    concept_id,
                    member_ids,
                    "merged",
                    merged_into_key,
                )
                continue
            if (reviewed and reviewed["status"] == "rejected") or identity_status == "rejected":
                self._persist_concept_identity(
                    concept_id,
                    member_ids,
                    "rejected",
                )
                continue
            status = (
                "confirmed"
                if (reviewed and reviewed["status"] == "confirmed")
                or identity_status == "confirmed"
                else "proposed"
            )
            self._create_neuron(
                3,
                f"Emergent Concept {digest}",
                summary,
                status,
                0.95 if status == "confirmed" else confidence,
                importance,
                neuron_id=concept_id,
                stability=stability,
            )
            for member_id in member_ids:
                self._connect(
                    member_id,
                    concept_id,
                    "emergent_member_of",
                    min(0.92, 0.62 + 0.06 * len(member_ids)),
                )
            created += 1
            self._persist_concept_identity(
                concept_id,
                member_ids,
                status,
            )
        return created

    def _refresh_semantic_stability(self) -> None:
        """Derive L3 stability from the amount and diversity of active experience."""
        concepts = self.db.execute(
            """SELECT id FROM neurons
               WHERE layer=3 AND status NOT IN ('rejected','archived','stale')"""
        ).fetchall()
        for concept in concepts:
            descendants = self.db.execute(
                """WITH RECURSIVE downward(id,layer) AS (
                       SELECT id,layer FROM neurons WHERE id=?
                       UNION
                       SELECT n.id,n.layer
                       FROM downward d
                       JOIN synapses s ON s.source_id=d.id
                       JOIN neurons n ON n.id=s.target_id
                       WHERE n.layer<d.layer
                         AND n.status NOT IN ('rejected','archived','stale')
                   )
                   SELECT
                     count(DISTINCT CASE WHEN layer=1 THEN id END) AS atoms,
                     count(DISTINCT CASE WHEN layer=2 THEN id END) AS episodes
                   FROM downward""",
                (concept["id"],),
            ).fetchone()
            support = int(descendants["atoms"] or 0)
            episodes = int(descendants["episodes"] or 0)
            stability = (
                0.05
                if support == 0
                else min(0.98, 1.0 - math.exp(-(support + 0.5 * episodes) / 3.0))
            )
            self.db.execute(
                "UPDATE neurons SET stability=? WHERE id=?",
                (stability, concept["id"]),
            )

    def _decay_plastic_synapses(
        self,
        reference: datetime | None = None,
        half_life_days: float = 120.0,
    ) -> int:
        """Apply forgetting to plastic links while preserving structural evidence."""
        reference = reference or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        reference = reference.astimezone(timezone.utc)
        previous_consolidation = parse_timestamp(
            self._get_meta("last_consolidated_at")
        )
        rows = self.db.execute(
            """SELECT s.source_id,s.target_id,s.relation,s.weight,s.last_fired,
                      n.created_at,n.importance,n.stability
               FROM synapses s JOIN neurons n ON n.id=s.source_id
               WHERE s.relation IN ('association','co_recalled')"""
        ).fetchall()
        changed = 0
        for row in rows:
            anchors = [
                parsed
                for parsed in (
                    parse_timestamp(row["last_fired"]),
                    parse_timestamp(row["created_at"]),
                    previous_consolidation,
                )
                if parsed is not None
            ]
            anchor = max(anchors) if anchors else reference
            age = max(0.0, (reference - anchor).total_seconds() / 86400.0)
            if age < 1.0:
                continue
            protection = 0.55 + 0.45 * (
                0.5 * float(row["importance"]) + 0.5 * float(row["stability"])
            )
            effective_half_life = max(1.0, half_life_days * protection)
            retained = 0.5 ** (age / effective_half_life)
            weight = max(0.04, float(row["weight"]) * retained)
            if abs(weight - float(row["weight"])) < 1e-9:
                continue
            self.db.execute(
                """UPDATE synapses SET weight=?
                   WHERE source_id=? AND target_id=? AND relation=?""",
                (weight, row["source_id"], row["target_id"], row["relation"]),
            )
            changed += 1
        self._set_meta(
            "last_consolidated_at",
            reference.isoformat(timespec="seconds"),
        )
        return changed

    def _consolidate_derived_state(
        self,
        apply_decay: bool = False,
        reference: datetime | None = None,
    ) -> dict[str, int]:
        emergent = self._rebuild_emergent_concepts()
        self._refresh_semantic_stability()
        duplicate_candidates = self._refresh_concept_duplicate_candidates()
        concept_families = self._refresh_concept_families()
        decayed = self._decay_plastic_synapses(reference) if apply_decay else 0
        return {
            "emergent_concepts": emergent,
            "concept_duplicate_candidates": duplicate_candidates,
            "concept_families": concept_families,
            "decayed_synapses": decayed,
        }

    @serialized_write
    def consolidate(
        self,
        reference: datetime | None = None,
    ) -> dict[str, object]:
        """Run experience-driven abstraction, stabilization, and safe forgetting."""
        result = self._consolidate_derived_state(
            apply_decay=True,
            reference=reference,
        )
        self.db.commit()
        return {**result, "stats": self.stats()}

    @serialized_write
    def consolidate_if_due(
        self,
        interval: timedelta = timedelta(hours=24),
        reference: datetime | None = None,
    ) -> dict[str, object]:
        """Consolidate once the configured offline interval has elapsed."""
        if interval.total_seconds() <= 0:
            raise ValueError("consolidation interval must be positive")
        reference = reference or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        reference = reference.astimezone(timezone.utc)
        previous = parse_timestamp(self._get_meta("last_consolidated_at"))
        if previous is not None:
            elapsed = max(0.0, (reference - previous).total_seconds())
            if elapsed < interval.total_seconds():
                return {
                    "performed": False,
                    "last_consolidated_at": previous.isoformat(timespec="seconds"),
                    "due_at": (previous + interval).isoformat(timespec="seconds"),
                }
        try:
            result = self._consolidate_derived_state(
                apply_decay=True,
                reference=reference,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return {
            "performed": True,
            "last_consolidated_at": reference.isoformat(timespec="seconds"),
            "due_at": (reference + interval).isoformat(timespec="seconds"),
            **result,
        }

    @serialized_write
    def review_emergent_concept(self, concept_id: str, decision: str) -> bool:
        """Persist a human decision about one rebuildable semantic abstraction."""
        if decision not in {"confirm", "reject"}:
            raise ValueError("emergent concept decision must be confirm or reject")
        concept = self.db.execute(
            """SELECT id FROM neurons
               WHERE id=? AND layer=3 AND label LIKE 'Emergent Concept %'""",
            (concept_id,),
        ).fetchone()
        if not concept:
            return False
        member_ids = sorted(
            row["source_id"]
            for row in self.db.execute(
                """SELECT s.source_id
                   FROM synapses s JOIN neurons n ON n.id=s.source_id
                   WHERE s.target_id=? AND s.relation='emergent_member_of'
                     AND n.layer=1""",
                (concept_id,),
            ).fetchall()
        )
        reviewed_at = now()
        status = "confirmed" if decision == "confirm" else "rejected"
        write_record(
            self.semantic_review_dir / f"{concept_id}.md",
            {
                "format": SEMANTIC_REVIEW_FORMAT,
                "concept_id": concept_id,
                "status": status,
                "member_ids": member_ids,
                "reviewed_at": reviewed_at,
            },
            (
                f"Human review of {concept_id}: {status}. "
                f"Supported by {len(member_ids)} atomic memory traces."
            ),
        )
        self.db.execute(
            """INSERT INTO semantic_reviews(concept_id,status,member_ids,reviewed_at)
               VALUES(?,?,?,?)
               ON CONFLICT(concept_id) DO UPDATE SET
                 status=excluded.status,
                 member_ids=excluded.member_ids,
                 reviewed_at=excluded.reviewed_at""",
            (concept_id, status, json.dumps(member_ids), reviewed_at),
        )
        self._persist_concept_identity(
            concept_id,
            member_ids,
            status,
        )
        if decision == "confirm":
            self.db.execute(
                """UPDATE neurons
                   SET status='confirmed',confidence=0.95,
                       stability=max(stability,0.72)
                   WHERE id=?""",
                (concept_id,),
            )
        else:
            self.db.execute(
                "DELETE FROM synapses WHERE source_id=? OR target_id=?",
                (concept_id, concept_id),
            )
            self.db.execute("DELETE FROM neurons WHERE id=?", (concept_id,))
        self.db.commit()
        return True

    @serialized_write
    def review_l3_concept(self, concept_id: str, decision: str) -> bool:
        """Apply a human confirmation or rejection to a named L3 concept."""
        if decision not in {"confirm", "reject"}:
            raise ValueError("L3 concept decision must be confirm or reject")
        concept = self.db.execute(
            "SELECT * FROM neurons WHERE id=? AND layer=3",
            (concept_id,),
        ).fetchone()
        if not concept:
            return False
        member_ids = sorted(
            str(row["id"]) for row in self._related_atoms(concept_id)
        )
        reviewed_at = now()
        status = "confirmed" if decision == "confirm" else "rejected"
        write_record(
            self.semantic_review_dir / f"{concept_id}.md",
            {
                "format": SEMANTIC_REVIEW_FORMAT,
                "concept_id": concept_id,
                "status": status,
                "member_ids": member_ids,
                "reviewed_at": reviewed_at,
            },
            (
                f"Human review of L3 {concept_id}: {status}. "
                f"Supported by {len(member_ids)} atomic memory traces."
            ),
        )
        self.db.execute(
            """INSERT INTO semantic_reviews(concept_id,status,member_ids,reviewed_at)
               VALUES(?,?,?,?)
               ON CONFLICT(concept_id) DO UPDATE SET
                 status=excluded.status,
                 member_ids=excluded.member_ids,
                 reviewed_at=excluded.reviewed_at""",
            (concept_id, status, json.dumps(member_ids), reviewed_at),
        )
        if decision == "confirm":
            self.db.execute(
                """UPDATE neurons
                   SET status='confirmed',confidence=max(confidence,0.95),
                       stability=max(stability,0.68)
                   WHERE id=?""",
                (concept_id,),
            )
        else:
            self.db.execute(
                "DELETE FROM synapses WHERE source_id=? OR target_id=?",
                (concept_id, concept_id),
            )
            self.db.execute("DELETE FROM neurons WHERE id=?", (concept_id,))
            self._consolidate_derived_state()
        self.db.commit()
        return True

    def _concept_prototype(
        self, concept: sqlite3.Row
    ) -> tuple[list[float], set[str]]:
        atoms = self._related_atoms(concept["id"])
        member_ids = {str(row["id"]) for row in atoms}
        if not atoms:
            return self._vector(concept), member_ids
        vectors = [self._vector(row) for row in atoms]
        centroid = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        ]
        norm = math.sqrt(sum(value * value for value in centroid)) or 1.0
        return [value / norm for value in centroid], member_ids

    def _refresh_concept_duplicate_candidates(self) -> int:
        """Propose likely duplicate L3 pairs without merging them automatically."""
        self.db.execute(
            "DELETE FROM concept_duplicate_reviews WHERE status='pending'"
        )
        concepts = self.db.execute(
            """SELECT * FROM neurons
               WHERE layer=3 AND status NOT IN ('rejected','archived','stale')
               ORDER BY label,id"""
        ).fetchall()
        profiles = {
            row["id"]: self._concept_prototype(row)
            for row in concepts
        }
        label_terms = {
            row["id"]: {
                (term[:-1] if term.endswith("s") and len(term) > 3 else term)
                for term in re.findall(r"[a-z0-9]+", str(row["label"]).casefold())
            }
            for row in concepts
        }
        created = 0
        for index, left in enumerate(concepts):
            left_key = self._concept_key(left)
            left_vector, left_members = profiles[left["id"]]
            left_terms = label_terms[left["id"]]
            for right in concepts[index + 1 :]:
                right_key = self._concept_key(right)
                if left_key == right_key:
                    continue
                right_vector, right_members = profiles[right["id"]]
                right_terms = label_terms[right["id"]]
                vector_score = max(0.0, cosine(left_vector, right_vector))
                member_overlap = len(left_members & right_members) / max(
                    1, len(left_members | right_members)
                )
                lexical_label_score = len(left_terms & right_terms) / max(
                    1, len(left_terms | right_terms)
                )
                name_vector_score = max(
                    0.0,
                    cosine(self._vector(left), self._vector(right)),
                )
                label_score = (
                    0.60 * name_vector_score
                    + 0.40 * lexical_label_score
                )
                combined = (
                    0.55 * vector_score
                    + 0.30 * member_overlap
                    + 0.15 * label_score
                )
                emergent_pair = (
                    str(left["label"]).startswith("Emergent Concept ")
                    or str(right["label"]).startswith("Emergent Concept ")
                )
                eligible = (
                    (
                        emergent_pair
                        and (
                            (member_overlap >= 0.50 and vector_score >= 0.65)
                            or vector_score >= 0.95
                        )
                    )
                    or (
                        not emergent_pair
                        and label_score >= 0.70
                        and (
                            member_overlap >= 0.25
                            or vector_score >= 0.82
                        )
                    )
                )
                if not eligible:
                    continue
                ordered = sorted(
                    (
                        (left_key, str(left["label"])),
                        (right_key, str(right["label"])),
                    )
                )
                pair_left_key, pair_left_label = ordered[0]
                pair_right_key, pair_right_label = ordered[1]
                digest = hashlib.sha256(
                    f"{pair_left_key}|{pair_right_key}".encode("utf-8")
                ).hexdigest()[:12]
                review_id = f"dup_{digest}"
                cursor = self.db.execute(
                    """INSERT INTO concept_duplicate_reviews
                       (id,left_key,right_key,left_label,right_label,vector_score,
                        member_overlap,label_score,combined_score,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,'pending',?)
                       ON CONFLICT(left_key,right_key) DO UPDATE SET
                         left_label=excluded.left_label,
                         right_label=excluded.right_label,
                         vector_score=excluded.vector_score,
                         member_overlap=excluded.member_overlap,
                         label_score=excluded.label_score,
                         combined_score=excluded.combined_score
                       WHERE concept_duplicate_reviews.status='pending'""",
                    (
                        review_id,
                        pair_left_key,
                        pair_right_key,
                        pair_left_label,
                        pair_right_label,
                        vector_score,
                        member_overlap,
                        label_score,
                        combined,
                        now(),
                    ),
                )
                created += int(cursor.rowcount > 0)
        return created

    def concept_duplicate_candidates(
        self, status: str = "pending"
    ) -> list[dict[str, object]]:
        return [
            dict(row)
            for row in self.db.execute(
                """SELECT * FROM concept_duplicate_reviews
                   WHERE status=? ORDER BY combined_score DESC,id""",
                (status,),
            )
        ]

    def _write_concept_decision(
        self,
        review: sqlite3.Row,
        status: str,
        survivor: sqlite3.Row | None = None,
        loser: sqlite3.Row | None = None,
    ) -> None:
        reviewed_at = now()
        metadata = {
            "format": CONCEPT_DECISION_FORMAT,
            "id": review["id"],
            "left_key": review["left_key"],
            "right_key": review["right_key"],
            "left_label": review["left_label"],
            "right_label": review["right_label"],
            "vector_score": float(review["vector_score"]),
            "member_overlap": float(review["member_overlap"]),
            "label_score": float(review["label_score"]),
            "combined_score": float(review["combined_score"]),
            "status": status,
            "survivor_key": self._concept_key(survivor) if survivor else "",
            "survivor_label": str(survivor["label"]) if survivor else "",
            "loser_label": str(loser["label"]) if loser else "",
            "created_at": review["created_at"],
            "reviewed_at": reviewed_at,
        }
        write_record(
            self.concept_decision_dir / f"{review['id']}.md",
            metadata,
            (
                f"Human concept decision: {status}. "
                f"{review['left_label']} ↔ {review['right_label']}."
            ),
        )

    @serialized_write
    def review_concept_duplicate(self, review_id: str, decision: str) -> bool:
        if decision not in {"merge-left", "merge-right", "distinct"}:
            raise ValueError("concept duplicate decision is invalid")
        review = self.db.execute(
            """SELECT * FROM concept_duplicate_reviews
               WHERE id=? AND status='pending'""",
            (review_id,),
        ).fetchone()
        if not review:
            return False
        left = self._find_concept_by_key(review["left_key"])
        right = self._find_concept_by_key(review["right_key"])
        if not left or not right:
            return False
        reviewed_at = now()
        if decision == "distinct":
            self._write_concept_decision(review, "distinct")
            self.db.execute(
                """UPDATE concept_duplicate_reviews
                   SET status='distinct',reviewed_at=?
                   WHERE id=?""",
                (reviewed_at, review_id),
            )
            self.db.commit()
            return True

        survivor, loser = (
            (left, right) if decision == "merge-left" else (right, left)
        )
        edges = self.db.execute(
            """SELECT target_id,relation,weight
               FROM synapses WHERE source_id=?""",
            (loser["id"],),
        ).fetchall()
        for edge in edges:
            if edge["target_id"] == survivor["id"]:
                continue
            self._connect(
                survivor["id"],
                edge["target_id"],
                edge["relation"],
                float(edge["weight"]),
            )
        self.db.execute(
            "DELETE FROM synapses WHERE source_id=? OR target_id=?",
            (loser["id"], loser["id"]),
        )
        if loser["status"] == "confirmed" and survivor["status"] == "proposed":
            self.db.execute(
                "UPDATE neurons SET status='confirmed',confidence=max(confidence,0.95) WHERE id=?",
                (survivor["id"],),
            )
        self.db.execute(
            "UPDATE neurons SET status='archived' WHERE id=?",
            (loser["id"],),
        )
        survivor_key = self._concept_key(survivor)
        self.db.execute(
            """INSERT INTO concept_aliases
               (alias_key,alias_label,canonical_key,canonical_label,
                decision_id,created_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(alias_key) DO UPDATE SET
                 alias_label=excluded.alias_label,
                 canonical_key=excluded.canonical_key,
                 canonical_label=excluded.canonical_label,
                 decision_id=excluded.decision_id""",
            (
                normalized_concept_key(str(loser["label"])),
                str(loser["label"]),
                survivor_key,
                str(survivor["label"]),
                review_id,
                reviewed_at,
            ),
        )
        if str(loser["label"]).startswith("Emergent Concept "):
            member_ids = [
                str(row["id"]) for row in self._related_atoms(survivor["id"])
            ]
            self._persist_concept_identity(
                loser["id"],
                member_ids,
                "merged",
                survivor_key,
            )
        self._write_concept_decision(review, "merged", survivor, loser)
        self.db.execute(
            """UPDATE concept_duplicate_reviews
               SET status='merged',survivor_key=?,reviewed_at=?
               WHERE id=?""",
            (survivor_key, reviewed_at, review_id),
        )
        self.db.commit()
        return True

    def _persist_concept_family(
        self,
        family_id: str,
        member_keys: list[str],
        status: str,
        active: bool,
        label: str | None = None,
        created_at: str | None = None,
    ) -> None:
        member_keys = sorted(set(member_keys))
        existing = self.db.execute(
            "SELECT * FROM concept_families WHERE id=?",
            (family_id,),
        ).fetchone()
        created_at = (
            created_at
            or (str(existing["created_at"]) if existing else None)
            or now()
        )
        label = (
            label
            or (str(existing["label"]) if existing else None)
            or f"概念家族 {family_id[4:]}"
        )
        members = [
            self._find_concept_by_key(key)
            for key in member_keys
        ]
        member_labels = [
            str(row["label"]) for row in members if row is not None
        ]
        summary = (
            f"L3F 概念家族包含 {len(member_labels)} 个相关 L3 概念："
            f"{'、'.join(member_labels)}。"
        )
        updated_at = now()
        write_record(
            self.concept_family_dir / f"{family_id}.md",
            {
                "format": CONCEPT_FAMILY_FORMAT,
                "id": family_id,
                "label": label,
                "status": status,
                "member_keys": member_keys,
                "active": active,
                "created_at": created_at,
                "updated_at": updated_at,
            },
            summary,
        )
        self.db.execute(
            """INSERT INTO concept_families
               (id,label,summary,status,member_keys,active,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 label=excluded.label,
                 summary=excluded.summary,
                 status=excluded.status,
                 member_keys=excluded.member_keys,
                 active=excluded.active,
                 updated_at=excluded.updated_at""",
            (
                family_id,
                label,
                summary,
                status,
                json.dumps(member_keys, ensure_ascii=False),
                int(active),
                created_at,
                updated_at,
            ),
        )

    def _refresh_concept_families(
        self,
        min_members: int = 3,
        similarity_threshold: float = 0.62,
    ) -> int:
        """Build a stable L3F grouping layer without merging or renumbering L3."""
        concepts = self.db.execute(
            """SELECT * FROM neurons
               WHERE layer=3 AND status NOT IN ('rejected','archived','stale')
               ORDER BY label,id"""
        ).fetchall()
        existing = self.db.execute(
            "SELECT * FROM concept_families ORDER BY created_at,id"
        ).fetchall()
        self.db.execute("UPDATE concept_families SET active=0")
        if len(concepts) < min_members:
            return 0

        profiles = {
            row["id"]: self._concept_prototype(row)
            for row in concepts
        }
        clusters: list[list[sqlite3.Row]] = []
        for concept in concepts:
            vector, members = profiles[concept["id"]]
            choices: list[tuple[float, int]] = []
            for index, cluster in enumerate(clusters):
                similarities: list[float] = []
                for other in cluster:
                    other_vector, other_members = profiles[other["id"]]
                    overlap = len(members & other_members) / max(
                        1, len(members | other_members)
                    )
                    semantic = max(0.0, cosine(vector, other_vector))
                    similarities.append(0.72 * semantic + 0.28 * overlap)
                if similarities and min(similarities) >= similarity_threshold:
                    choices.append((sum(similarities) / len(similarities), index))
            if choices:
                _, best = max(choices)
                clusters[best].append(concept)
            else:
                clusters.append([concept])

        groups = [cluster for cluster in clusters if len(cluster) >= min_members]
        used_ids: set[str] = set()
        active_count = 0
        for group in groups:
            member_keys = sorted(self._concept_key(row) for row in group)
            member_set = set(member_keys)
            identity_choices: list[tuple[float, sqlite3.Row]] = []
            rejected_exact: sqlite3.Row | None = None
            for candidate in existing:
                if candidate["id"] in used_ids:
                    continue
                old_members = set(json.loads(candidate["member_keys"]))
                if candidate["status"] == "rejected":
                    if old_members == member_set:
                        rejected_exact = candidate
                    continue
                overlap = len(member_set & old_members) / max(
                    1, len(member_set | old_members)
                )
                if overlap >= 0.60:
                    identity_choices.append((overlap, candidate))
            identity = (
                rejected_exact
                or (
                    max(identity_choices, key=lambda item: item[0])[1]
                    if identity_choices
                    else None
                )
            )
            if identity is not None:
                family_id = str(identity["id"])
                status = str(identity["status"])
                if (
                    status == "confirmed"
                    and set(json.loads(identity["member_keys"])) != member_set
                ):
                    status = "proposed"
                label = str(identity["label"])
                created_at = str(identity["created_at"])
            else:
                digest = hashlib.sha256(
                    "|".join(member_keys).encode("utf-8")
                ).hexdigest()[:10]
                family_id = f"l3f_{digest}"
                status = "proposed"
                label = f"概念家族 {digest}"
                created_at = now()
            used_ids.add(family_id)
            active = status != "rejected"
            self._persist_concept_family(
                family_id,
                member_keys,
                status,
                active,
                label,
                created_at,
            )
            active_count += int(active)
        return active_count

    def concept_families(
        self,
        status: str | None = None,
        active_only: bool = True,
        include_relations: bool = True,
    ) -> list[dict[str, object]]:
        clauses: list[str] = []
        parameters: list[object] = []
        if status is not None:
            clauses.append("status=?")
            parameters.append(status)
        if active_only:
            clauses.append("active=1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        result: list[dict[str, object]] = []
        for row in self.db.execute(
            f"SELECT * FROM concept_families {where} ORDER BY label,id",
            parameters,
        ):
            item = dict(row)
            item["member_keys"] = json.loads(str(row["member_keys"]))
            item["members"] = [
                {
                    "id": concept["id"],
                    "label": concept["label"],
                }
                for key in item["member_keys"]
                if (concept := self._find_concept_by_key(str(key))) is not None
            ]
            item["shared_relations"] = (
                self._family_shared_relations(
                    [str(member["id"]) for member in item["members"]]
                )
                if include_relations
                else []
            )
            result.append(item)
        return result

    @staticmethod
    def _family_display_name(family: dict[str, object]) -> str:
        """Return a short human-readable name without exposing the internal ID."""
        labels = [
            str(member.get("label", "")).strip()
            for member in family.get("members", [])
            if isinstance(member, dict) and str(member.get("label", "")).strip()
        ]
        normalized = {label.casefold() for label in labels}
        if {"memory system", "memory retrieval"} <= normalized:
            return "L3F · Neural Memory System"
        if {"proactive checking", "user preference", "writing workflow"} <= normalized:
            return "L3F · Personal Writing Workflow"
        if {"economics", "thesis"} <= normalized:
            return "L3F · Academic Thesis Writing"
        if labels:
            return f"L3F · {labels[0]} & related concepts"
        return "L3F · Unnamed Concept Family"

    def _family_shared_relations(
        self,
        concept_ids: list[str],
    ) -> list[dict[str, object]]:
        """Return L4/L5 relations used by at least two members of one L3F."""
        support: dict[str, set[str]] = {}
        rows_by_id: dict[str, sqlite3.Row] = {}
        for concept_id in concept_ids:
            rows = self.db.execute(
                """WITH RECURSIVE upper(id,layer,depth) AS (
                       SELECT id,layer,0 FROM neurons WHERE id=?
                       UNION
                       SELECT n.id,n.layer,u.depth+1
                       FROM upper u
                       JOIN synapses s ON s.source_id=u.id
                       JOIN neurons n ON n.id=s.target_id
                       WHERE n.layer>u.layer AND n.layer<=5 AND u.depth<3
                   )
                   SELECT DISTINCT n.* FROM upper u
                   JOIN neurons n ON n.id=u.id
                   WHERE n.layer BETWEEN 4 AND 5
                     AND n.status NOT IN ('rejected','archived','stale')""",
                (concept_id,),
            ).fetchall()
            for row in rows:
                rows_by_id[row["id"]] = row
                support.setdefault(str(row["id"]), set()).add(concept_id)
        return [
            {
                "id": relation_id,
                "layer": int(rows_by_id[relation_id]["layer"]),
                "label": str(rows_by_id[relation_id]["label"]),
                "member_count": len(member_ids),
            }
            for relation_id, member_ids in sorted(
                support.items(),
                key=lambda item: (
                    int(rows_by_id[item[0]]["layer"]),
                    str(rows_by_id[item[0]]["label"]),
                ),
            )
            if len(member_ids) >= 2
        ]

    def _family_size_profile(self, family: dict[str, object]) -> dict[str, object]:
        """Return a bounded route bonus from family breadth and evidence depth."""
        member_ids = [
            str(member["id"])
            for member in family.get("members", [])
            if isinstance(member, dict) and member.get("id")
        ]
        support_ids: set[str] = set()
        for member_id in member_ids:
            for row in self._related_atoms(member_id):
                if row["status"] not in {"rejected", "archived", "stale"}:
                    support_ids.add(str(row["id"]))

        member_signal = 1.0 - math.exp(
            -len(member_ids) / L3F_MEMBER_SIZE_SCALE
        )
        support_signal = 1.0 - math.exp(
            -len(support_ids) / L3F_SUPPORT_SIZE_SCALE
        )
        size_signal = 0.60 * member_signal + 0.40 * support_signal
        configured_cap = float(
            getattr(self.encoder, "family_size_bonus_cap", L3F_SIZE_BONUS_CAP)
        )
        cap = min(0.20, max(0.0, configured_cap))
        return {
            "member_count": len(member_ids),
            "l1_support_count": len(support_ids),
            "size_bonus": cap * size_signal,
        }

    def concept_family_routes(
        self,
        query: str,
        limit: int = 2,
    ) -> dict[str, object]:
        """Select confirmed L3F routes, or explicitly fall back to global L3."""
        families = self.concept_families(
            status="confirmed",
            include_relations=False,
        )
        if not families:
            return {
                "used": False,
                "has_confirmed_families": False,
                "reason": "no_confirmed_families",
                "families": [],
                "selected_concept_ids": [],
            }
        routing_query = enrich_query_with_known_concepts(query)
        query_vector = self._encode(routing_query)
        query_terms = set(features(routing_query))
        scored: list[dict[str, object]] = []
        for family in families:
            member_labels = " ".join(
                str(member["label"]) for member in family["members"]
            )
            display_name = self._family_display_name(family)
            representation = self._encode(
                f"{display_name} {family['label']} {family['summary']} {member_labels}"
            )
            semantic = max(0.0, cosine(query_vector, representation))
            family_terms = set(
                features(
                    f"{display_name} {family['label']} {family['summary']} {member_labels}"
                )
            )
            lexical = (
                len(query_terms & family_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            base_score = 0.75 * semantic + 0.25 * lexical
            profile = self._family_size_profile(family)
            routing_score = base_score + float(profile["size_bonus"])
            scored.append(
                {
                    "family": family,
                    "base_score": base_score,
                    "routing_score": routing_score,
                    **profile,
                }
            )
        scored.sort(
            key=lambda item: float(item["routing_score"]),
            reverse=True,
        )
        threshold = float(
            getattr(self.encoder, "family_gate_threshold", 0.23)
        )
        base_peak = max(float(item["base_score"]) for item in scored)
        routing_peak = float(scored[0]["routing_score"])
        if base_peak < threshold:
            return {
                "used": False,
                "has_confirmed_families": True,
                "reason": "family_gate_closed",
                "peak": round(base_peak, 6),
                "routing_peak": round(routing_peak, 6),
                "families": [],
                "selected_concept_ids": [],
            }
        eligible = [
            item
            for item in scored
            if float(item["base_score"]) >= threshold
        ]
        eligible.sort(
            key=lambda item: float(item["routing_score"]),
            reverse=True,
        )
        routing_peak = float(eligible[0]["routing_score"])
        selected = [
            item
            for item in eligible[: max(1, limit)]
            if float(item["routing_score"]) >= routing_peak - 0.08
        ]
        selected_concept_ids = list(
            dict.fromkeys(
                str(member["id"])
                for item in selected
                for member in item["family"]["members"]
            )
        )
        return {
            "used": bool(selected_concept_ids),
            "has_confirmed_families": True,
            "reason": "confirmed_family_match",
            "peak": round(base_peak, 6),
            "routing_peak": round(routing_peak, 6),
            "families": [
                {
                    "id": item["family"]["id"],
                    "label": self._family_display_name(item["family"]),
                    "status": "confirmed",
                    "activation": round(float(item["routing_score"]), 6),
                    "base_score": round(float(item["base_score"]), 6),
                    "size_bonus": round(float(item["size_bonus"]), 6),
                    "member_count": int(item["member_count"]),
                    "l1_support_count": int(item["l1_support_count"]),
                    "member_concept_ids": [
                        str(member["id"])
                        for member in item["family"]["members"]
                    ],
                    "shared_relations": self._family_shared_relations(
                        [
                            str(member["id"])
                            for member in item["family"]["members"]
                        ]
                    ),
                }
                for item in selected
            ],
            "selected_concept_ids": selected_concept_ids,
        }

    @serialized_write
    def review_concept_family(self, family_id: str, decision: str) -> bool:
        if decision not in {"confirm", "reject"}:
            raise ValueError("concept family decision must be confirm or reject")
        family = self.db.execute(
            "SELECT * FROM concept_families WHERE id=? AND active=1",
            (family_id,),
        ).fetchone()
        if not family:
            return False
        status = "confirmed" if decision == "confirm" else "rejected"
        self._persist_concept_family(
            family_id,
            list(json.loads(family["member_keys"])),
            status,
            decision == "confirm",
            str(family["label"]),
            str(family["created_at"]),
        )
        self.db.commit()
        return True

    @serialized_write
    def remember(
        self,
        text: str,
        source: str,
        topics: Iterable[str] = (),
        schemas: Iterable[str] = (),
        importance: float = 0.7,
        confirmed: bool = False,
        episode: str | None = None,
        procedures: Iterable[str] = (),
        domain: str | None = None,
        expires_at: str | None = None,
        supersedes: Iterable[str] = (),
        conflicts: Iterable[str] = (),
    ) -> str:
        requested_topics = require_english_labels(
            canonical_concepts(list(topics)), "topics"
        )
        if not requested_topics and any(
            token in source.casefold() for token in ("skill", "tool")
        ):
            requested_topics = ["Tools"]
        concept_labels = self._resolve_memory_topics(text, requested_topics)
        procedure_labels = require_english_labels(procedures, "procedures")
        persona_labels = require_english_labels(schemas, "schemas")
        episode_labels = require_english_labels(
            [episode] if episode else [], "episode"
        )
        domain_labels = require_english_labels(
            [domain] if domain else [], "domain"
        )
        evidence_id = short_id("ev")
        neuron_id = short_id("l1")
        created_at = now()
        evidence_path = self.evidence_dir / f"{evidence_id}.md"
        evidence_path.write_text(
            "---\n"
            f"id: {evidence_id}\n"
            f"source: {json.dumps(source, ensure_ascii=False)}\n"
            f"created_at: {created_at}\n"
            "---\n\n"
            f"{text.strip()}\n",
            encoding="utf-8",
        )
        record = {
            "format": MEMORY_FORMAT,
            "id": neuron_id,
            "evidence_id": evidence_id,
            "evidence_path": str(evidence_path.relative_to(self.root)),
            "source": source,
            "status": "confirmed" if confirmed else "proposed",
            "confidence": 0.95 if confirmed else 0.68,
            "importance": importance,
            "created_at": created_at,
            "episode": episode_labels[0] if episode_labels else "",
            "concepts": concept_labels,
            "procedures": procedure_labels,
            "personas": persona_labels,
            "domain": domain_labels[0] if domain_labels else "",
            "expires_at": (expires_at or "").strip(),
            "supersedes": list(dict.fromkeys(x.strip() for x in supersedes if x.strip())),
            "conflicts": list(dict.fromkeys(x.strip() for x in conflicts if x.strip())),
        }
        write_record(self.memory_dir / f"{neuron_id}.md", record, text)
        self._index_canonical_record(record, text)
        if confirmed:
            self._consolidate_derived_state()
        else:
            self._refresh_semantic_stability()
        self.db.commit()
        return neuron_id

    def _index_canonical_record(self, record: dict[str, object], text: str) -> None:
        if record.get("format") != MEMORY_FORMAT:
            raise ValueError(f"unsupported memory record: {record.get('format')}")
        neuron_id = str(record["id"])
        evidence_id = str(record["evidence_id"])
        source = str(record.get("source", "unknown"))
        status = str(record.get("status", "proposed"))
        confidence = float(record.get("confidence", 0.68))
        importance = float(record.get("importance", 0.7))
        created_at = str(record.get("created_at", now()))
        evidence_path = str(record["evidence_path"])
        self.db.execute(
            "INSERT OR REPLACE INTO evidence(id,path,source,created_at) VALUES(?,?,?,?)",
            (evidence_id, evidence_path, source, created_at),
        )
        label = compact(" ".join(text.strip().split()), 54)
        self._create_neuron(
            1,
            label,
            text.strip(),
            status,
            confidence,
            importance,
            evidence_id,
            neuron_id,
            str(record.get("expires_at", "")) or None,
            0.68 if status == "confirmed" else 0.36,
            created_at,
        )

        for target_id in record.get("supersedes", []):
            self._add_relation(
                neuron_id,
                str(target_id),
                "supersedes",
                "写入时声明为旧记忆的替代版本；需人工确认后归档旧记忆。",
            )
        for target_id in record.get("conflicts", []):
            self._add_relation(
                neuron_id,
                str(target_id),
                "conflicts_with",
                "写入时声明存在冲突；保留双方，等待人工裁决。",
            )

        if status in {"rejected", "archived"}:
            return

        new_vector = self._encode(text)
        peers = self.db.execute(
            "SELECT * FROM neurons WHERE layer=1 AND id!=? AND status!='rejected'",
            (neuron_id,),
        ).fetchall()
        related = sorted(
            ((max(0.0, cosine(new_vector, self._vector(row))), row["id"]) for row in peers),
            reverse=True,
        )[:5]
        for score, peer_id in related:
            if score >= 0.18:
                self._connect(neuron_id, peer_id, "association", min(0.75, score + 0.2))
        self._flag_possible_conflicts(neuron_id, text, peers)

        upper_status = status if status in {"confirmed", "proposed", "archived"} else "proposed"
        current_ids = [neuron_id]
        requested_topics = english_only_labels(
            canonical_concepts([str(x) for x in record.get("concepts", [])])
        )
        if not requested_topics and any(
            token in source.casefold() for token in ("skill", "tool")
        ):
            requested_topics = ["Tools"]
        concept_labels = self._resolve_memory_topics(text, requested_topics)
        level_specs: list[tuple[int, list[str], str, str]] = [
            (2, [str(record.get("episode", ""))], "episode", "情景记忆"),
            (3, concept_labels, "member_of", "语义概念"),
            (4, english_only_labels(record.get("procedures", [])), "used_in", "程序记忆"),
            (5, [str(x) for x in record.get("personas", [])], "supports", "稳定模型"),
            (6, [str(record.get("domain", ""))], "routes_to", "元记忆域"),
        ]
        for layer, labels, relation, kind in level_specs:
            clean = (
                canonical_concepts(labels)
                if layer == 3
                else list(dict.fromkeys(label.strip() for label in labels if label.strip()))
            )
            if not clean:
                continue
            next_ids: list[str] = []
            for upper_label in clean:
                row = self._find_named(layer, upper_label)
                if row:
                    upper_id = row["id"]
                    status_rank = {"archived": 0, "proposed": 1, "confirmed": 2}
                    if status_rank.get(upper_status, 1) > status_rank.get(row["status"], 1):
                        self.db.execute(
                            "UPDATE neurons SET status=? WHERE id=?",
                            (upper_status, upper_id),
                        )
                else:
                    stable_id = (
                        self._stable_named_concept_id(upper_label)
                        if layer == 3
                        else None
                    )
                    upper_id = self._create_neuron(
                        layer,
                        upper_label,
                        f"{kind}：{upper_label}",
                        upper_status,
                        min(0.95, confidence),
                        min(1.0, importance + layer * 0.02),
                        neuron_id=stable_id,
                    )
                next_ids.append(upper_id)
                for lower_id in current_ids:
                    self._connect(lower_id, upper_id, relation, max(0.68, 0.94 - layer * 0.04))
            current_ids = next_ids

    def _flag_possible_conflicts(
        self, neuron_id: str, text: str, peers: Iterable[sqlite3.Row]
    ) -> None:
        """Conservative polarity check; it only opens an issue for human review."""
        opposites = [
            ("likes", "does not like"),
            ("needs", "does not need"),
            ("should", "should not"),
            ("uses", "does not use"),
            ("allows", "does not allow"),
            ("喜欢", "不喜欢"),
            ("需要", "不需要"),
            ("应该", "不应该"),
            ("使用", "不使用"),
            ("允许", "不允许"),
            ("always", "never"),
            ("prefer", "avoid"),
        ]
        new_features = set(features(text))
        for peer in peers:
            old_text = str(peer["summary"])
            shared = len(new_features & set(features(old_text))) / max(1, len(new_features))
            if shared < 0.25:
                continue
            opposite = False
            for positive, negative in opposites:
                new_positive = positive in text.lower() and negative not in text.lower()
                old_positive = positive in old_text.lower() and negative not in old_text.lower()
                if (new_positive and negative in old_text.lower()) or (
                    old_positive and negative in text.lower()
                ):
                    opposite = True
                    break
            if opposite:
                self._issue(
                    neuron_id,
                    "possible_conflict",
                    "warning",
                    f"可能与 {peer['id']} 冲突（词面重叠 {shared:.2f}）；请人工比较原始证据。",
                )

    @staticmethod
    def parse_mdkb_list(output: str) -> list[dict[str, object]]:
        pattern = re.compile(
            r"^\[([^]]+)]\s+(.+?)\s+\(([^)]+)\)\s*(.*?)\s+-\s+\d+\s+access(?:es)?$"
        )
        entries: list[dict[str, object]] = []
        for line in output.splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            entry_id, title, entry_type, tag_text = match.groups()
            entries.append({
                "id": entry_id,
                "title": title,
                "type": entry_type,
                "tags": re.findall(r"#([^\s#]+)", tag_text),
            })
        return entries

    def _import_mdkb_copy(
        self, mdkb_bin: Path, workspace: Path, limit: int = 0
    ) -> dict[str, int]:
        """Copy mdkb records into canonical Markdown while retaining source authority."""
        listing = subprocess.run(
            [str(mdkb_bin), "memory", "list", "--format", "json", "-l", "1000"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        raw_entries = json.loads(listing)
        if not isinstance(raw_entries, list):
            raise ValueError("mdkb JSON listing must be an array")
        entries = raw_entries[:limit] if limit > 0 else raw_entries
        imported = 0
        updated = 0
        copied_bytes = 0
        full_token_estimate = 0
        source_to_neuron: dict[str, str] = {}
        imported_entries: list[dict[str, object]] = []

        for raw in entries:
            if not isinstance(raw, dict):
                continue
            entry_id = str(raw.get("id", "")).strip()
            title = str(raw.get("title", "")).strip()
            content = str(raw.get("content", "")).strip()
            if not entry_id or not title:
                continue
            entry_type = str(raw.get("entry_type", "topic"))
            source_status = str(raw.get("status", "active"))
            tags = [str(tag) for tag in raw.get("tags", []) if str(tag).strip()]
            source = f"mdkb:{entry_id}"
            neural_status = "confirmed" if source_status == "active" else "archived"
            confidence = 0.98
            copied_at = now()
            source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            full_token_estimate += rough_tokens(content)
            copied_bytes += len(content.encode("utf-8"))
            summary = f"mdkb {entry_type}：{title} " + " ".join(
                "#" + tag for tag in tags
            )
            representation = json.dumps(
                self._encode(f"{title} {entry_type} {' '.join(tags)} {content}"),
                separators=(",", ":"),
            )
            existing = self.db.execute(
                """SELECT n.id,n.created_at,e.id AS evidence_id,e.path
                   FROM evidence e JOIN neurons n ON n.evidence_id=e.id
                   WHERE e.source=?""",
                (source,),
            ).fetchone()
            if existing:
                neuron_id = existing["id"]
                evidence_id = existing["evidence_id"]
                evidence_path = self.root / existing["path"]
                created_at = existing["created_at"]
                self.db.execute(
                    """UPDATE neurons
                       SET label=?,summary=?,vector=?,status=?,confidence=?
                       WHERE id=?""",
                    (title, summary.strip(), representation, neural_status, confidence, neuron_id),
                )
                updated += 1
            else:
                evidence_id = short_id("ev")
                evidence_path = self.evidence_dir / f"{evidence_id}.md"
                created_at = copied_at
                self.db.execute(
                    "INSERT INTO evidence(id,path,source,created_at) VALUES(?,?,?,?)",
                    (evidence_id, str(evidence_path.relative_to(self.root)), source, created_at),
                )
                neuron_id = self._create_neuron(
                    1, title, summary.strip(), neural_status, confidence, 0.8, evidence_id
                )
                self.db.execute(
                    "UPDATE neurons SET vector=? WHERE id=?", (representation, neuron_id)
                )
                imported += 1

            evidence_path.write_text(
                "---\n"
                f"id: {evidence_id}\n"
                f"source: {json.dumps(source, ensure_ascii=False)}\n"
                "authority: global-mdkb\n"
                "content_copied: true\n"
                f"source_status: {source_status}\n"
                f"source_updated_at: {raw.get('updated_at', '')}\n"
                f"source_sha256: {source_hash}\n"
                f"copied_at: {copied_at}\n"
                "---\n\n"
                f"{content}\n",
                encoding="utf-8",
            )
            record = {
                "format": MEMORY_FORMAT,
                "id": neuron_id,
                "evidence_id": evidence_id,
                "evidence_path": str(evidence_path.relative_to(self.root)),
                "source": source,
                "source_authority": "global-mdkb",
                "source_status": source_status,
                "source_sha256": source_hash,
                "status": neural_status,
                "confidence": confidence,
                "importance": 0.8,
                "created_at": created_at,
                "episode": "",
                "concepts": tags,
                "procedures": [],
                "personas": [],
                "domain": f"mdkb:{entry_type}",
                "expires_at": "",
                "supersedes": [],
                "conflicts": [],
            }
            write_record(self.memory_dir / f"{neuron_id}.md", record, content or title)
            source_to_neuron[entry_id] = neuron_id
            imported_entries.append(raw)

            grouping_tags = tags or [f"type-{entry_type}"]
            for tag in grouping_tags:
                row = self._find_named(2, tag)
                topic_id = row["id"] if row else self._create_neuron(
                    2, tag, f"mdkb 标签集群 #{tag}", "confirmed", 0.95, 0.75
                )
                self._connect(neuron_id, topic_id, "mdkb_tag", 0.88)

        for raw in imported_entries:
            old_id = str(raw.get("id", ""))
            new_id = str(raw.get("superseded_by") or "")
            if old_id in source_to_neuron and new_id in source_to_neuron:
                self._connect(
                    source_to_neuron[old_id],
                    source_to_neuron[new_id],
                    "mdkb_supersession",
                    0.92,
                )

        self._set_meta("mdkb_bin", str(mdkb_bin))
        self._set_meta("mdkb_workspace", str(workspace))
        self._set_meta("mdkb_full_token_estimate", str(full_token_estimate))
        self._set_meta("mdkb_copy_mode", "full")
        self._set_meta("mdkb_last_copy_at", now())
        self.db.commit()
        return {
            "discovered": len(entries),
            "imported": imported,
            "updated": updated,
            "copied_bytes": copied_bytes,
            "full_token_estimate": full_token_estimate,
        }

    @serialized_write
    def import_mdkb(
        self,
        mdkb_bin: Path,
        workspace: Path,
        limit: int = 0,
        copy_content: bool = False,
    ) -> dict[str, int]:
        """Build a derived neural index while keeping full text only in mdkb."""
        mdkb_bin = mdkb_bin.resolve()
        workspace = workspace.resolve()
        if copy_content:
            return self._import_mdkb_copy(mdkb_bin, workspace, limit)
        listing = subprocess.run(
            [str(mdkb_bin), "memory", "list"],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        entries = self.parse_mdkb_list(listing)
        if limit > 0:
            entries = entries[:limit]
        imported = 0
        updated = 0
        full_token_estimate = 0
        for entry in entries:
            entry_id = str(entry["id"])
            source = f"mdkb:{entry_id}"
            full = subprocess.run(
                [str(mdkb_bin), "get", entry_id],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            full_token_estimate += rough_tokens(full)
            existing = self.db.execute(
                """SELECT n.id FROM evidence e JOIN neurons n ON n.evidence_id=e.id
                   WHERE e.source=?""",
                (source,),
            ).fetchone()
            representation = json.dumps(
                self._encode(f"{entry['title']} {entry['type']} {' '.join(entry['tags'])} {full}"),
                separators=(",", ":"),
            )
            summary = f"mdkb {entry['type']}：{entry['title']} " + " ".join(
                "#" + str(tag) for tag in entry["tags"]
            )
            if existing:
                self.db.execute(
                    "UPDATE neurons SET label=?,summary=?,vector=?,status='confirmed',confidence=0.98 WHERE id=?",
                    (str(entry["title"]), summary.strip(), representation, existing["id"]),
                )
                neuron_id = existing["id"]
                updated += 1
            else:
                evidence_id = short_id("ev")
                pointer = self.evidence_dir / f"{evidence_id}.md"
                pointer.write_text(
                    "---\n"
                    f"source: {source}\n"
                    "authority: global-mdkb\n"
                    "content_copied: false\n"
                    "---\n\n"
                    f"Canonical memory: `{entry_id}`\n\n"
                    f"Read with: `{mdkb_bin} get {entry_id}`\n",
                    encoding="utf-8",
                )
                self.db.execute(
                    "INSERT INTO evidence(id,path,source,created_at) VALUES(?,?,?,?)",
                    (evidence_id, str(pointer.relative_to(self.root)), source, now()),
                )
                neuron_id = self._create_neuron(
                    1,
                    str(entry["title"]),
                    summary.strip(),
                    "confirmed",
                    0.98,
                    0.8,
                    evidence_id,
                )
                self.db.execute("UPDATE neurons SET vector=? WHERE id=?", (representation, neuron_id))
                imported += 1

            grouping_tags = list(entry["tags"]) or [f"type-{entry['type']}"]
            for tag in grouping_tags:
                topic = str(tag)
                row = self._find_named(2, topic)
                topic_id = row["id"] if row else self._create_neuron(
                    2, topic, f"mdkb 标签集群 #{topic}", "confirmed", 0.95, 0.75
                )
                self._connect(neuron_id, topic_id, "mdkb_tag", 0.88)

        self._set_meta("mdkb_bin", str(mdkb_bin))
        self._set_meta("mdkb_workspace", str(workspace))
        self._set_meta("mdkb_full_token_estimate", str(full_token_estimate))
        self.db.commit()
        return {
            "discovered": len(entries),
            "imported": imported,
            "updated": updated,
            "full_token_estimate": full_token_estimate,
        }

    def activate(
        self,
        query: str,
        winners: int = 7,
        rounds: int = 2,
        spread: float = 0.52,
        use_family_routing: bool = True,
    ) -> list[ActivatedNeuron]:
        query_vector = self._encode(query)
        query_features = set(features(query))
        family_routing = (
            self.concept_family_routes(query)
            if use_family_routing
            else {
                "used": False,
                "has_confirmed_families": False,
                "selected_concept_ids": [],
            }
        )
        rows = self.db.execute(
            "SELECT * FROM neurons WHERE status NOT IN ('rejected','stale','archived')"
        ).fetchall()
        if use_family_routing and family_routing["has_confirmed_families"]:
            grouped_concept_ids = {
                str(member["id"])
                for family in self.concept_families(
                    status="confirmed",
                    include_relations=False,
                )
                for member in family["members"]
            }
            selected_concept_ids = set(
                str(item) for item in family_routing["selected_concept_ids"]
            )
            rows = [
                row
                for row in rows
                if row["layer"] != 3
                or row["id"] not in grouped_concept_ids
                or row["id"] in selected_concept_ids
            ]
        direct: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        row_map = {row["id"]: row for row in rows}

        # BM25 is deliberately calculated over every active neuron. The demo is
        # small; a production adapter can replace this with an inverted index.
        query_terms = features(query)
        documents = {
            row["id"]: features(row["label"] + " " + row["summary"])
            for row in rows
        }
        document_frequency: dict[str, int] = {}
        for terms in documents.values():
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        population = max(1, len(documents))
        average_length = (
            sum(len(terms) for terms in documents.values()) / population or 1.0
        )
        bm25_raw: dict[str, float] = {}
        for neuron_id, terms in documents.items():
            frequencies: dict[str, int] = {}
            for term in terms:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                frequency_in_docs = document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1.0 + (population - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1.0 - 0.75 + 0.75 * len(terms) / average_length
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            bm25_raw[neuron_id] = score
        bm25_peak = max(bm25_raw.values(), default=0.0) or 1.0

        for row in rows:
            vector_score = max(0.0, cosine(query_vector, self._vector(row)))
            row_features = set(documents[row["id"]])
            lexical_coverage = (
                len(query_features & row_features) / len(query_features)
                if query_features
                else 0.0
            )
            bm25_score = bm25_raw[row["id"]] / bm25_peak
            fused_score = (
                0.45 * vector_score + 0.45 * bm25_score + 0.10 * lexical_coverage
            )
            stability = float(row["stability"])
            retention = 1.0
            if row["layer"] == 1:
                anchor = row["last_reactivated"] or row["last_used"] or row["created_at"]
                age = elapsed_days(anchor)
                half_life = 30.0 + 335.0 * (
                    0.45 * float(row["importance"]) + 0.55 * stability
                )
                retention = 0.35 + 0.65 * (0.5 ** (age / max(1.0, half_life)))
            governance = row["confidence"] * (
                0.65 + 0.30 * row["importance"] + 0.05 * stability
            )
            direct[row["id"]] = fused_score * governance * retention
            components[row["id"]] = {
                "vector": vector_score,
                "bm25": bm25_score,
                "lexical": lexical_coverage,
                "retention": retention,
            }

        l1_ids = [row["id"] for row in rows if row["layer"] == 1]
        l1_winners = sorted(l1_ids, key=lambda neuron_id: direct[neuron_id], reverse=True)[:winners]
        global_winners = sorted(direct, key=direct.get, reverse=True)[:winners]
        selected = list(dict.fromkeys(l1_winners + global_winners))
        activation = {neuron_id: direct[neuron_id] for neuron_id in selected if direct[neuron_id] > 0}

        for round_index in range(rounds):
            additions: dict[str, float] = {}
            for source_id, source_activation in list(activation.items()):
                edges = self.db.execute(
                    "SELECT target_id,weight FROM synapses WHERE source_id=?",
                    (source_id,),
                ).fetchall()
                for edge in edges:
                    if edge["target_id"] not in row_map:
                        continue
                    propagated = source_activation * edge["weight"] * spread / (round_index + 1)
                    additions[edge["target_id"]] = max(
                        additions.get(edge["target_id"], 0.0), propagated
                    )
            for neuron_id, value in additions.items():
                activation[neuron_id] = min(1.0, activation.get(neuron_id, 0.0) + value)

        result: list[ActivatedNeuron] = []
        for neuron_id, value in sorted(activation.items(), key=lambda item: item[1], reverse=True):
            row = row_map[neuron_id]
            result.append(
                ActivatedNeuron(
                    id=neuron_id,
                    layer=row["layer"],
                    label=row["label"],
                    summary=row["summary"],
                    status=row["status"],
                    confidence=row["confidence"],
                    importance=row["importance"],
                    evidence_id=row["evidence_id"],
                    activation=value,
                    vector_score=components[neuron_id]["vector"],
                    bm25_score=components[neuron_id]["bm25"],
                    lexical_score=components[neuron_id]["lexical"],
                    direct_activation=direct.get(neuron_id, 0.0),
                    spread_activation=max(0.0, value - direct.get(neuron_id, 0.0)),
                    stability=float(row["stability"]),
                    retention=components[neuron_id]["retention"],
                )
            )
        return result

    def _recall_gate_score(
        self,
        activated: list[ActivatedNeuron],
    ) -> float:
        l1 = [item for item in activated if item.layer == 1]
        return l1[0].activation if l1 else 0.0

    def probe(self, query: str) -> tuple[bool, float, list[ActivatedNeuron]]:
        activated = self.activate(query)
        peak = self._recall_gate_score(activated)
        threshold = float(getattr(self.encoder, "gate_threshold", 0.06))
        family_routing = self.concept_family_routes(query)
        l3_route_peak = max(
            (
                item.direct_activation
                for item in activated
                if item.layer == 3
            ),
            default=0.0,
        )
        l3_route_threshold = (
            0.18 if isinstance(self.encoder, HashEncoder) else 0.35
        )
        if (
            family_routing["has_confirmed_families"]
            and not family_routing["used"]
            and (
                peak < threshold
                or l3_route_peak < l3_route_threshold
            )
        ):
            fallback = self.activate(query, use_family_routing=False)
            fallback_score = self._recall_gate_score(fallback)
            if fallback_score > peak:
                activated = fallback
                peak = fallback_score
        return peak >= threshold, peak, activated

    def _topic_memory_ids(
        self, activated: list[ActivatedNeuron], query: str = ""
    ) -> set[str]:
        """Return L1 memories through L3F-selected L3 routes with safe fallback."""
        all_routes = sorted(
            [
                (item.id, item.label, item.direct_activation)
                for item in activated
                if item.layer == 3 and item.direct_activation > 0
            ],
            key=lambda route: route[2],
            reverse=True,
        )
        family_routing = self.concept_family_routes(query)
        selected_family_concepts = set(
            str(item) for item in family_routing["selected_concept_ids"]
        )
        family_routes = [
            route for route in all_routes if route[0] in selected_family_concepts
        ]
        if family_routing["used"] and family_routes:
            family_peak = family_routes[0][2]
            routes = [
                route for route in family_routes[:2]
                if route[2] >= family_peak * 0.80
            ]
            confirmed_member_ids = {
                str(member["id"])
                for family in self.concept_families(
                    status="confirmed",
                    include_relations=False,
                )
                for member in family["members"]
            }
            independent_routes = [
                route for route in all_routes
                if route[0] not in confirmed_member_ids
            ]
            if independent_routes:
                independent = independent_routes[0]
                floor = 0.10 if isinstance(self.encoder, HashEncoder) else 0.20
                if independent[2] >= max(floor, family_peak * 0.60):
                    routes.append(independent)
        else:
            routes = all_routes
        if not routes:
            return set()
        peak = routes[0][2]
        if not family_routing["used"]:
            routes = [
                route for route in routes[:2]
                if route[2] >= peak * 0.80
            ]
        route_ids = [route[0] for route in routes]
        if any(token in query.casefold() for token in ("继续", "continue", "resume")):
            for _, route_label, _ in routes:
                parent_label = TOPIC_PARENTS.get(route_label)
                parent = self._find_named(3, parent_label) if parent_label else None
                if parent and parent["status"] not in {"rejected", "stale", "archived"}:
                    route_ids.append(parent["id"])
        route_ids = list(dict.fromkeys(route_ids))
        placeholders = ",".join("?" for _ in route_ids)
        rows = self.db.execute(
            f"""SELECT s.source_id AS route_id, s.target_id AS memory_id
                FROM synapses s JOIN neurons n ON n.id=s.target_id
                WHERE s.source_id IN ({placeholders})
                  AND s.relation='member_of' AND n.layer=1
                  AND n.status NOT IN ('rejected','stale','archived')""",
            tuple(route_ids),
        ).fetchall()
        return {row["memory_id"] for row in rows}

    def recall(
        self,
        query: str,
        limit: int = 5,
        reconsolidate: bool = False,
    ) -> list[ActivatedNeuron]:
        _, _, activated = self.probe(query)
        cards = [item for item in activated if item.layer == 1]
        topic_memory_ids = self._topic_memory_ids(activated, query)
        if topic_memory_ids:
            scoped = [
                card for card in cards
                if card.id in topic_memory_ids
            ]
            if scoped:
                cards = scoped
        cards = cards[:limit]
        if reconsolidate and cards:
            self.reinforce([card.id for card in cards])
        return cards

    def evidence_text(self, evidence_id: str) -> str:
        row = self.db.execute("SELECT path,source FROM evidence WHERE id=?", (evidence_id,)).fetchone()
        if not row:
            return "[evidence missing]"
        local_path = self.root / row["path"]
        if row["source"].startswith("mdkb:"):
            if self._get_meta("mdkb_copy_mode") == "full":
                return local_path.read_text(encoding="utf-8")
            mdkb_bin = self._get_meta("mdkb_bin")
            workspace = self._get_meta("mdkb_workspace")
            if mdkb_bin and workspace:
                try:
                    return subprocess.run(
                        [mdkb_bin, "get", row["source"].split(":", 1)[1]],
                        cwd=workspace,
                        text=True,
                        capture_output=True,
                        check=True,
                    ).stdout
                except (OSError, subprocess.CalledProcessError) as exc:
                    pointer = local_path.read_text(encoding="utf-8")
                    return f"[live mdkb read failed: {exc}]\n{pointer}"
        return local_path.read_text(encoding="utf-8")

    def benchmark(self, query: str, limit: int = 3) -> dict[str, object]:
        evidence_rows = self.db.execute("SELECT path,source FROM evidence").fetchall()
        mdkb_rows = [row for row in evidence_rows if row["source"].startswith("mdkb:")]
        local_rows = [row for row in evidence_rows if not row["source"].startswith("mdkb:")]
        local_text = "\n".join(
            (self.root / row["path"]).read_text(encoding="utf-8") for row in local_rows
        )
        cards = self.recall(query, limit)
        recall_text = "\n".join(card.summary for card in cards)
        full_tokens = rough_tokens(local_text)
        if mdkb_rows:
            full_tokens += int(self._get_meta("mdkb_full_token_estimate") or 0)
        recall_tokens = rough_tokens(recall_text)
        reduction = 0.0 if not full_tokens else 1.0 - recall_tokens / full_tokens
        return {
            "query": query,
            "corpus_memories": len(evidence_rows),
            "recalled_cards": len(cards),
            "estimated_full_tokens": full_tokens,
            "estimated_recall_tokens": recall_tokens,
            "estimated_reduction": round(reduction, 4),
            "note": "rough local estimate; not a model tokenizer benchmark",
        }

    @serialized_write
    def reinforce(self, neuron_ids: list[str], amount: float = 0.08) -> None:
        """Retrieval reconsolidation: stabilize traces and strengthen co-active links."""
        fired_at = now()
        for index, left in enumerate(neuron_ids):
            for right in neuron_ids[index + 1 :]:
                self._connect(left, right, "co_recalled", amount)
            self.db.execute(
                """UPDATE neurons
                   SET last_used=?,
                       last_reactivated=?,
                       reactivation_count=reactivation_count+1,
                       stability=min(1.0, stability + ? * (1.0-stability))
                   WHERE id=?""",
                (fired_at, fired_at, max(0.0, amount), left),
            )
        self.db.execute(
            """UPDATE synapses
               SET last_fired=?,fire_count=fire_count+1
               WHERE source_id IN ({})""".format(
                ",".join("?" for _ in neuron_ids)
            ),
            (fired_at, *neuron_ids),
        ) if neuron_ids else None
        self._refresh_semantic_stability()
        self.db.commit()

    def _has_active_l1_descendant(self, neuron_id: str) -> bool:
        """Check whether a structural node still routes down to an active L1."""
        row = self.db.execute(
            "SELECT layer FROM neurons WHERE id=?", (neuron_id,)
        ).fetchone()
        if not row:
            return False
        return bool(
            self.db.execute(
                """WITH RECURSIVE downward(id,layer) AS (
                       SELECT ?, ?
                       UNION
                       SELECT n.id,n.layer
                       FROM downward d
                       JOIN synapses s ON s.source_id=d.id
                       JOIN neurons n ON n.id=s.target_id
                       WHERE n.layer<d.layer
                         AND n.status NOT IN ('rejected','archived','stale')
                   )
                   SELECT 1 FROM downward WHERE layer=1 LIMIT 1""",
                (neuron_id, int(row["layer"])),
            ).fetchone()
        )

    def _prune_orphan_semantic_nodes(self) -> dict[int, int]:
        """Delete L3/L4 nodes that no active atomic memory can reach."""
        orphan_ids: list[str] = []
        counts = {3: 0, 4: 0}
        for row in self.db.execute(
            "SELECT id,layer FROM neurons WHERE layer IN (3,4)"
        ).fetchall():
            if self._has_active_l1_descendant(row["id"]):
                continue
            orphan_ids.append(row["id"])
            counts[int(row["layer"])] += 1
        if orphan_ids:
            placeholders = ",".join("?" for _ in orphan_ids)
            self.db.execute(
                f"DELETE FROM synapses WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                (*orphan_ids, *orphan_ids),
            )
            self.db.execute(
                f"DELETE FROM neurons WHERE id IN ({placeholders})", orphan_ids
            )
        return counts

    def _canonical_evidence_ids(self) -> set[str]:
        """Return evidence IDs still referenced by active canonical L1 records."""
        evidence_ids: set[str] = set()
        for path in self.memory_dir.glob("*.md"):
            metadata, _ = read_record(path)
            if metadata.get("format") == MEMORY_FORMAT and metadata.get("evidence_id"):
                evidence_ids.add(str(metadata["evidence_id"]))
        return evidence_ids

    @serialized_write
    def archive_orphan_evidence(self) -> list[str]:
        """Move unreferenced evidence into the hidden, backup-safe rejection archive."""
        referenced = self._canonical_evidence_ids()
        moved: list[str] = []
        for evidence_path in self.evidence_dir.glob("ev_*.md"):
            if evidence_path.stem in referenced:
                continue
            destination = self.rejected_dir / evidence_path.name
            if destination.exists():
                raise FileExistsError(f"rejected evidence already exists: {destination}")
            evidence_path.replace(destination)
            self.db.execute("DELETE FROM evidence WHERE id=?", (evidence_path.stem,))
            moved.append(evidence_path.stem)
        self.db.commit()
        return moved

    def _archive_rejected_record(self, neuron_id: str) -> None:
        """Move a rejected L1 and its unshared evidence out of the active vault."""
        row = self.db.execute(
            "SELECT evidence_id FROM neurons WHERE id=?", (neuron_id,)
        ).fetchone()
        if not row:
            return
        canonical = self.memory_dir / f"{neuron_id}.md"
        evidence_id = str(row["evidence_id"] or "")
        if not canonical.is_file():
            self.db.execute("DELETE FROM neurons WHERE id=?", (neuron_id,))
            return

        metadata, body = read_record(canonical)
        original_metadata = dict(metadata)
        evidence_path = self.root / str(metadata.get("evidence_path", ""))
        evidence_users = self.db.execute(
            "SELECT count(*) FROM neurons WHERE evidence_id=? AND id!=?",
            (evidence_id, neuron_id),
        ).fetchone()[0] if evidence_id else 0
        move_evidence = bool(evidence_id and evidence_path.is_file() and not evidence_users)
        rejected_memory = self.rejected_dir / canonical.name
        rejected_evidence = self.rejected_dir / evidence_path.name
        if rejected_memory.exists() or (move_evidence and rejected_evidence.exists()):
            raise FileExistsError(f"rejected archive already contains {neuron_id}")

        metadata["status"] = "rejected"
        metadata["confidence"] = 0.0
        metadata["rejected_at"] = now()
        metadata["rejected_from_status"] = original_metadata.get("status", "proposed")
        write_record(canonical, metadata, body)
        moved: list[tuple[Path, Path]] = []
        try:
            canonical.replace(rejected_memory)
            moved.append((canonical, rejected_memory))
            if move_evidence:
                evidence_path.replace(rejected_evidence)
                moved.append((evidence_path, rejected_evidence))
        except Exception:
            for source, destination in reversed(moved):
                if destination.exists():
                    destination.replace(source)
            if canonical.exists():
                write_record(canonical, original_metadata, body)
            raise

        self.db.execute("DELETE FROM neurons WHERE id=?", (neuron_id,))
        if move_evidence:
            self.db.execute("DELETE FROM evidence WHERE id=?", (evidence_id,))

    @serialized_write
    def restore_rejected(self, neuron_id: str) -> bool:
        """Restore a rejected L1 as a fresh proposed review candidate."""
        rejected_memory = self.rejected_dir / f"{neuron_id}.md"
        if not rejected_memory.is_file() or (self.memory_dir / rejected_memory.name).exists():
            return False
        metadata, body = read_record(rejected_memory)
        if metadata.get("format") != MEMORY_FORMAT:
            return False
        evidence_id = str(metadata.get("evidence_id", ""))
        evidence_path = self.root / str(metadata.get("evidence_path", ""))
        rejected_evidence = self.rejected_dir / f"{evidence_id}.md"
        if not evidence_path.is_file() and not rejected_evidence.is_file():
            raise FileNotFoundError(f"rejected record evidence is missing: {evidence_id}")
        if not evidence_path.is_file():
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_evidence.replace(evidence_path)
        metadata["status"] = "proposed"
        metadata["confidence"] = 0.68
        metadata.pop("rejected_at", None)
        metadata.pop("rejected_from_status", None)
        restored = self.memory_dir / rejected_memory.name
        write_record(restored, metadata, body)
        rejected_memory.unlink()
        self._index_canonical_record(metadata, body)
        self._consolidate_derived_state()
        self.db.commit()
        return True

    @serialized_write
    def review(self, neuron_id: str, status: str) -> bool:
        row = self.db.execute(
            "SELECT id,confidence FROM neurons WHERE id=?", (neuron_id,)
        ).fetchone()
        if not row:
            return False
        if status == "rejected":
            self._archive_rejected_record(neuron_id)
            self._prune_orphan_semantic_nodes()
            self.archive_orphan_evidence()
            self._consolidate_derived_state()
            self.db.commit()
            return True
        confidence = 0.98 if status == "confirmed" else row["confidence"]
        self.db.execute(
            """UPDATE neurons
               SET status=?, confidence=?,
                   stability=CASE WHEN ?='confirmed'
                                  THEN max(stability,0.68)
                                  ELSE stability END
               WHERE id=?""",
            (status, confidence, status, neuron_id),
        )
        canonical = self.memory_dir / f"{neuron_id}.md"
        if canonical.exists():
            metadata, body = read_record(canonical)
            metadata["status"] = status
            metadata["confidence"] = confidence
            write_record(canonical, metadata, body)
        self._consolidate_derived_state()
        self.db.commit()
        return True

    def proposed(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id,layer,label,confidence,created_at FROM neurons WHERE status='proposed' ORDER BY created_at"
        ).fetchall()

    @serialized_write
    def scan_maintenance(self) -> dict[str, int]:
        """Create idempotent maintenance candidates without mutating memories."""
        for row in self.db.execute(
            "SELECT id FROM neurons WHERE layer=1 AND status='proposed'"
        ):
            self._issue(
                row["id"], "needs_review", "info", "候选原子记忆尚未由用户确认。"
            )
        for row in self.db.execute(
            """SELECT id,expires_at FROM neurons
               WHERE expires_at IS NOT NULL AND expires_at!=''
                 AND expires_at<=? AND status NOT IN ('stale','rejected','archived')""",
            (now(),),
        ):
            self._issue(
                row["id"],
                "expired",
                "warning",
                f"有效期 {row['expires_at']} 已到；应核实后标记 stale 或写入新版。",
            )
        for relation in self.db.execute(
            "SELECT * FROM memory_relations WHERE status='pending'"
        ):
            for role, neuron_id in (("source", relation["source_id"]), ("target", relation["target_id"])):
                exists = self.db.execute(
                    "SELECT 1 FROM neurons WHERE id=?", (neuron_id,)
                ).fetchone()
                if not exists:
                    self._issue(
                        relation["source_id"],
                        "broken_relation",
                        "critical",
                        f"关系 {relation['id']} 的 {role} 神经元 {neuron_id} 不存在。",
                    )
        self.db.commit()
        return {
            "open_issues": self.db.execute(
                "SELECT count(*) FROM maintenance_issues WHERE status='open'"
            ).fetchone()[0],
            "pending_relations": self.db.execute(
                "SELECT count(*) FROM memory_relations WHERE status='pending'"
            ).fetchone()[0],
            "pending_annotations": self.db.execute(
                "SELECT count(*) FROM annotation_proposals WHERE status='pending'"
            ).fetchone()[0],
        }

    def maintenance_inbox(self) -> dict[str, list[dict[str, object]]]:
        self.scan_maintenance()
        issues = [dict(row) for row in self.db.execute(
            """SELECT * FROM maintenance_issues WHERE status='open'
               ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                        created_at"""
        )]
        relations = [dict(row) for row in self.db.execute(
            "SELECT * FROM memory_relations WHERE status='pending' ORDER BY created_at"
        )]
        annotations = self.annotation_proposals()
        return {"issues": issues, "relations": relations, "annotations": annotations}

    @serialized_write
    def resolve_issue(self, issue_id: str, action: str) -> bool:
        status = "resolved" if action == "resolve" else "ignored"
        cursor = self.db.execute(
            "UPDATE maintenance_issues SET status=?,resolved_at=? WHERE id=? AND status='open'",
            (status, now(), issue_id),
        )
        self.db.commit()
        return cursor.rowcount > 0

    @serialized_write
    def review_relation(self, relation_id: str, decision: str) -> bool:
        relation = self.db.execute(
            "SELECT * FROM memory_relations WHERE id=?", (relation_id,)
        ).fetchone()
        if not relation:
            return False
        status = "confirmed" if decision == "confirm" else "rejected"
        self.db.execute(
            "UPDATE memory_relations SET status=?,reviewed_at=? WHERE id=?",
            (status, now(), relation_id),
        )
        if status == "confirmed" and relation["relation"] == "supersedes":
            self.review(relation["target_id"], "archived")
        self.db.commit()
        return True

    def stats(self) -> dict[str, object]:
        layers = {
            row["layer"]: row["count"]
            for row in self.db.execute(
                "SELECT layer,count(*) AS count FROM neurons GROUP BY layer"
            )
        }
        layers[0] = self.db.execute("SELECT count(*) FROM evidence").fetchone()[0]
        statuses = {
            row["status"]: row["count"]
            for row in self.db.execute(
                "SELECT status,count(*) AS count FROM neurons GROUP BY status"
            )
        }
        orphan_l1 = self.db.execute(
            """SELECT count(*) FROM neurons n WHERE n.layer=1 AND NOT EXISTS
               (SELECT 1 FROM synapses s JOIN neurons t ON t.id=s.target_id
                WHERE s.source_id=n.id AND t.layer>1)"""
        ).fetchone()[0]
        return {
            "encoder": {
                "name": self.encoder.name,
                "dimensions": self.encoder.dimensions,
                "gate_threshold": float(getattr(self.encoder, "gate_threshold", 0.06)),
                "family_gate_threshold": float(
                    getattr(self.encoder, "family_gate_threshold", 0.23)
                ),
                "family_size_bonus_cap": float(
                    getattr(
                        self.encoder,
                        "family_size_bonus_cap",
                        L3F_SIZE_BONUS_CAP,
                    )
                ),
            },
            "layers": layers,
            "statuses": statuses,
            "synapses": self.db.execute("SELECT count(*) FROM synapses").fetchone()[0],
            "orphan_l1": orphan_l1,
            "canonical_records": len(list(self.memory_dir.glob("*.md"))),
            "obsidian_pages": len(list(self.obsidian_dir.rglob("*.md")))
            if self.obsidian_dir.exists()
            else 0,
            "open_issues": self.db.execute(
                "SELECT count(*) FROM maintenance_issues WHERE status='open'"
            ).fetchone()[0],
            "pending_relations": self.db.execute(
                "SELECT count(*) FROM memory_relations WHERE status='pending'"
            ).fetchone()[0],
            "pending_annotations": self.db.execute(
                "SELECT count(*) FROM annotation_proposals WHERE status='pending'"
            ).fetchone()[0],
            "biological_memory": {
                "emergent_concepts": self.db.execute(
                    """SELECT count(*) FROM neurons
                       WHERE layer=3 AND label LIKE 'Emergent Concept %'"""
                ).fetchone()[0],
                "reviewed_emergent_concepts": self.db.execute(
                    "SELECT count(*) FROM semantic_reviews"
                ).fetchone()[0],
                "stable_concept_identities": self.db.execute(
                    "SELECT count(*) FROM concept_identities"
                ).fetchone()[0],
                "pending_concept_duplicates": self.db.execute(
                    """SELECT count(*) FROM concept_duplicate_reviews
                       WHERE status='pending'"""
                ).fetchone()[0],
                "concept_aliases": self.db.execute(
                    "SELECT count(*) FROM concept_aliases"
                ).fetchone()[0],
                "active_concept_families": self.db.execute(
                    "SELECT count(*) FROM concept_families WHERE active=1"
                ).fetchone()[0],
                "pending_concept_families": self.db.execute(
                    """SELECT count(*) FROM concept_families
                       WHERE active=1 AND status='proposed'"""
                ).fetchone()[0],
                "reactivations": self.db.execute(
                    "SELECT coalesce(sum(reactivation_count),0) FROM neurons"
                ).fetchone()[0],
            },
        }

    def network(self) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
        nodes = self.db.execute(
            "SELECT id,layer,label,status FROM neurons ORDER BY layer DESC,label"
        ).fetchall()
        edges = self.db.execute(
            """SELECT source_id,target_id,relation,weight FROM synapses
               WHERE source_id < target_id ORDER BY weight DESC"""
        ).fetchall()
        return nodes, edges

    def health_report(self) -> dict[str, object]:
        integrity_rows = [row[0] for row in self.db.execute("PRAGMA integrity_check")]
        missing_evidence = [
            row["id"]
            for row in self.db.execute("SELECT id,path FROM evidence")
            if not (self.root / row["path"]).is_file()
        ]
        unreferenced_evidence = sorted(
            path.stem
            for path in self.evidence_dir.glob("ev_*.md")
            if path.stem not in self._canonical_evidence_ids()
        )
        return {
            "healthy": (
                integrity_rows == ["ok"]
                and not missing_evidence
                and not unreferenced_evidence
            ),
            "sqlite_integrity": integrity_rows,
            "journal_mode": self.db.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": self.db.execute("PRAGMA synchronous").fetchone()[0],
            "missing_evidence_ids": missing_evidence,
            "unreferenced_evidence_ids": unreferenced_evidence,
            "stats": self.stats(),
        }

    @serialized_write
    def rebuild_index(self) -> dict[str, object]:
        """Recreate every derived table from canonical Markdown records."""
        records: list[tuple[dict[str, object], str]] = []
        for path in sorted(self.memory_dir.glob("*.md")):
            metadata, body = read_record(path)
            if metadata.get("format") == MEMORY_FORMAT:
                records.append((metadata, body))
        self.db.execute("DELETE FROM synapses")
        self.db.execute("DELETE FROM neurons")
        self.db.execute("DELETE FROM evidence")
        self.db.execute("DELETE FROM semantic_reviews")
        self.db.execute("DELETE FROM concept_identities")
        self.db.execute("DELETE FROM concept_duplicate_reviews")
        self.db.execute("DELETE FROM concept_aliases")
        self.db.execute("DELETE FROM concept_families")
        self._load_semantic_reviews()
        self._load_concept_identities()
        self._load_concept_decisions()
        self._load_concept_families()
        for metadata, body in records:
            evidence_path = self.root / str(metadata["evidence_path"])
            if not evidence_path.is_file():
                raise FileNotFoundError(f"canonical evidence missing: {evidence_path}")
            self._index_canonical_record(metadata, body)
        self._consolidate_derived_state()
        self.db.commit()
        return {"records": len(records), "stats": self.stats()}

    def _related_atoms(self, neuron_id: str, max_depth: int = 5) -> list[sqlite3.Row]:
        """Return active L1 memories reached directly or through their L2 episode."""
        return self.db.execute(
            """WITH RECURSIVE descendants(id,layer,depth) AS (
                   SELECT id,layer,0 FROM neurons WHERE id=?
                   UNION
                   SELECT n.id,n.layer,d.depth+1
                   FROM descendants d
                   JOIN synapses s ON s.source_id=d.id
                   JOIN neurons n ON n.id=s.target_id
                   WHERE n.layer<d.layer
                     AND d.depth<?
                     AND s.relation IN ('member_of','emergent_member_of','episode')
               )
               SELECT DISTINCT n.*,e.source FROM descendants d
               JOIN neurons n ON n.id=d.id
               LEFT JOIN evidence e ON e.id=n.evidence_id
               WHERE n.layer=1 AND n.status NOT IN ('rejected','archived')
               ORDER BY n.created_at""",
            (neuron_id, max_depth),
        ).fetchall()

    @staticmethod
    def _preserved_user_notes(path: Path) -> str:
        if not path.exists():
            return "\n在这里添加人工批注。批注不会自动进入记忆核心。\n"
        match = USER_NOTES_RE.search(path.read_text(encoding="utf-8"))
        return match.group(1) if match else "\n在这里添加人工批注。\n"

    @staticmethod
    def _narrative(rows: list[sqlite3.Row]) -> str:
        statements = [row["summary"].strip().rstrip("。") for row in rows if row["summary"].strip()]
        if not statements:
            return "当前没有足够的已确认记忆形成连续说明。"
        connectors = ["目前，", "此外，", "在后续记录中，", "同时，", "综合来看，"]
        paragraphs: list[str] = []
        for index in range(0, len(statements), 3):
            group = statements[index : index + 3]
            sentence = "".join(
                (connectors[(index + offset) % len(connectors)] if offset == 0 else "；")
                + statement
                for offset, statement in enumerate(group)
            )
            paragraphs.append(sentence + "。")
        return "\n\n".join(paragraphs)

    @serialized_write
    def sync_obsidian_notes(self) -> dict[str, int]:
        """Turn meaningful USER-NOTES into review proposals, never direct memories."""
        discovered = created = 0
        topic_dir = self.obsidian_dir / "主题"
        if not topic_dir.exists():
            return {"discovered": 0, "created": 0, "pending": 0}
        placeholders = {
            "在这里添加人工批注。",
            "在这里添加人工批注。批注不会自动进入记忆核心。",
        }
        for page in sorted(topic_dir.glob("*.md")):
            page_text = page.read_text(encoding="utf-8")
            match = USER_NOTES_RE.search(page_text)
            concept_match = re.search(r"^concept_id:\s*(\S+)\s*$", page_text, re.MULTILINE)
            if not match or not concept_match:
                continue
            notes = match.group(1).strip()
            if not notes or notes in placeholders:
                continue
            discovered += 1
            relative = str(page.relative_to(self.obsidian_dir))
            notes_hash = hashlib.sha256(notes.encode("utf-8")).hexdigest()
            proposal_id = "note_" + hashlib.sha256(
                f"{relative}|{notes_hash}".encode("utf-8")
            ).hexdigest()[:12]
            cursor = self.db.execute(
                """INSERT OR IGNORE INTO annotation_proposals
                   (id,page_path,concept_id,notes_hash,notes,status,created_at)
                   VALUES(?,?,?,?,?,'pending',?)""",
                (proposal_id, relative, concept_match.group(1), notes_hash, notes, now()),
            )
            created += int(cursor.rowcount > 0)
        self.db.commit()
        pending = self.db.execute(
            "SELECT count(*) FROM annotation_proposals WHERE status='pending'"
        ).fetchone()[0]
        return {"discovered": discovered, "created": created, "pending": pending}

    def annotation_proposals(self, status: str = "pending") -> list[dict[str, object]]:
        return [dict(row) for row in self.db.execute(
            """SELECT p.*,n.label AS concept_label
               FROM annotation_proposals p
               LEFT JOIN neurons n ON n.id=p.concept_id
               WHERE p.status=? ORDER BY p.created_at""",
            (status,),
        )]

    @serialized_write
    def sync_obsidian_reviews(self) -> dict[str, object]:
        """读取维护中心中由用户明确勾选的记忆审核决定。"""
        maintenance_pages = (
            self.obsidian_dir / "99 维护中心.md",
            self.obsidian_dir / "99 Maintenance.md",
        )
        page = next((item for item in maintenance_pages if item.is_file()), None)
        if page is None:
            return {
                "confirmed": 0,
                "needs_revision": 0,
                "rejected": 0,
                "concepts_confirmed": 0,
                "concepts_needs_revision": 0,
                "concepts_rejected": 0,
                "concepts_merged": 0,
                "concepts_kept_distinct": 0,
                "families_confirmed": 0,
                "families_rejected": 0,
                "errors": [],
            }
        page_text = page.read_text(encoding="utf-8")
        decisions: dict[str, list[str]] = {}
        for action, neuron_id in MEMORY_REVIEW_RE.findall(page_text):
            decisions.setdefault(neuron_id, []).append(action)
        concept_decisions: dict[str, list[str]] = {}
        for action, concept_id in CONCEPT_REVIEW_RE.findall(page_text):
            concept_decisions.setdefault(concept_id, []).append(action)
        duplicate_decisions: dict[str, list[str]] = {}
        for action, review_id in CONCEPT_DUPLICATE_REVIEW_RE.findall(page_text):
            duplicate_decisions.setdefault(review_id, []).append(action)
        family_decisions: dict[str, list[str]] = {}
        for action, family_id in CONCEPT_FAMILY_REVIEW_RE.findall(page_text):
            family_decisions.setdefault(family_id, []).append(action)
        result: dict[str, object] = {
            "confirmed": 0,
            "needs_revision": 0,
            "rejected": 0,
            "concepts_confirmed": 0,
            "concepts_needs_revision": 0,
            "concepts_rejected": 0,
            "concepts_merged": 0,
            "concepts_kept_distinct": 0,
            "families_confirmed": 0,
            "families_rejected": 0,
            "errors": [],
        }
        errors = result["errors"]
        assert isinstance(errors, list)
        for neuron_id, actions in decisions.items():
            unique_actions = list(dict.fromkeys(actions))
            if len(unique_actions) != 1:
                errors.append(f"{neuron_id}：必须且只能选择一个审核选项")
                continue
            row = self.db.execute(
                "SELECT status FROM neurons WHERE id=? AND layer=1", (neuron_id,)
            ).fetchone()
            if not row or row["status"] != "proposed":
                continue
            action = unique_actions[0]
            if action == "confirm":
                self.review(neuron_id, "confirmed")
                result["confirmed"] = int(result["confirmed"]) + 1
            elif action == "reject":
                self.review(neuron_id, "rejected")
                result["rejected"] = int(result["rejected"]) + 1
            else:
                self._issue(
                    neuron_id,
                    "needs_revision",
                    "warning",
                    "人工审核在 Obsidian 中将此候选记忆标记为需要修改。",
                )
                result["needs_revision"] = int(result["needs_revision"]) + 1
        for concept_id, actions in concept_decisions.items():
            unique_actions = list(dict.fromkeys(actions))
            if len(unique_actions) != 1:
                errors.append(f"{concept_id}: select exactly one concept review option")
                continue
            action = unique_actions[0]
            concept = self.db.execute(
                "SELECT label FROM neurons WHERE id=? AND layer=3",
                (concept_id,),
            ).fetchone()
            if not concept:
                continue
            if action == "revise":
                self._issue(
                    concept_id,
                    "l3_needs_revision",
                    "warning",
                    "人工审核在 Obsidian 中将此 L3 概念标记为需要修改。",
                )
                result["concepts_needs_revision"] = int(
                    result["concepts_needs_revision"]
                ) + 1
                continue
            if str(concept["label"]).startswith("Emergent Concept "):
                changed = self.review_emergent_concept(concept_id, action)
            else:
                changed = self.review_l3_concept(concept_id, action)
            if not changed:
                continue
            key = "concepts_confirmed" if action == "confirm" else "concepts_rejected"
            result[key] = int(result[key]) + 1
        for review_id, actions in duplicate_decisions.items():
            unique_actions = list(dict.fromkeys(actions))
            if len(unique_actions) != 1:
                errors.append(
                    f"{review_id}：合并左侧、合并右侧、保持独立只能选择一个"
                )
                continue
            action = unique_actions[0]
            if not self.review_concept_duplicate(review_id, action):
                continue
            key = (
                "concepts_kept_distinct"
                if action == "distinct"
                else "concepts_merged"
            )
            result[key] = int(result[key]) + 1
        for family_id, actions in family_decisions.items():
            unique_actions = list(dict.fromkeys(actions))
            if len(unique_actions) != 1:
                errors.append(
                    f"{family_id}: select exactly one concept-family review option"
                )
                continue
            action = unique_actions[0]
            if not self.review_concept_family(family_id, action):
                continue
            key = "families_confirmed" if action == "confirm" else "families_rejected"
            result[key] = int(result[key]) + 1
        self.db.commit()
        return result

    @serialized_write
    def review_annotation(self, proposal_id: str, decision: str) -> str | None:
        proposal = self.db.execute(
            "SELECT * FROM annotation_proposals WHERE id=? AND status='pending'",
            (proposal_id,),
        ).fetchone()
        if not proposal:
            return None
        resulting_neuron_id: str | None = None
        if decision == "accept":
            concept = self.db.execute(
                "SELECT label FROM neurons WHERE id=?", (proposal["concept_id"],)
            ).fetchone()
            topics = [concept["label"]] if concept else []
            resulting_neuron_id = self.remember(
                proposal["notes"],
                f"obsidian-review:{proposal['page_path']}",
                topics=topics,
                confirmed=True,
                domain="Obsidian Manual Maintenance",
            )
        status = "accepted" if decision == "accept" else "rejected"
        self.db.execute(
            """UPDATE annotation_proposals
               SET status=?,reviewed_at=?,resulting_neuron_id=? WHERE id=?""",
            (status, now(), resulting_neuron_id, proposal_id),
        )
        self.db.commit()
        return resulting_neuron_id or status

    @serialized_write
    def compile_obsidian(self) -> dict[str, object]:
        """Compile human-readable views that are explicitly excluded from ingestion."""
        topic_dir = self.obsidian_dir / "主题"
        topic_dir.mkdir(parents=True, exist_ok=True)
        family_dir = self.obsidian_dir / "概念家族"
        family_dir.mkdir(parents=True, exist_ok=True)
        relation_dirs = {
            4: self.obsidian_dir / "relations" / "procedures",
            5: self.obsidian_dir / "relations" / "personas",
        }
        for directory in relation_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        review_sync = self.sync_obsidian_reviews()
        concepts = self.db.execute(
            """SELECT * FROM neurons WHERE layer=3
               AND status NOT IN ('rejected','archived') ORDER BY label"""
        ).fetchall()
        relation_nodes: list[sqlite3.Row] = []
        relation_topics_by_id: dict[str, list[sqlite3.Row]] = {}
        for row in self.db.execute(
            """SELECT * FROM neurons WHERE layer IN (4,5)
               AND status NOT IN ('rejected','archived') ORDER BY layer,label"""
        ).fetchall():
            if not is_english_label(row["label"]):
                continue
            related_topics = self.db.execute(
                """WITH RECURSIVE ancestors(id,layer,depth) AS (
                       SELECT id,layer,0 FROM neurons WHERE id=?
                       UNION
                       SELECT n.id,n.layer,a.depth+1
                       FROM ancestors a
                       JOIN synapses s ON s.target_id=a.id
                       JOIN neurons n ON n.id=s.source_id
                       WHERE n.layer<a.layer AND a.depth<4
                   )
                   SELECT DISTINCT n.id,n.label FROM ancestors a
                   JOIN neurons n ON n.id=a.id
                   WHERE n.layer=3 AND n.status NOT IN ('rejected','archived')
                   ORDER BY n.label""",
                (row["id"],),
            ).fetchall()
            if not related_topics:
                continue
            relation_nodes.append(row)
            relation_topics_by_id[row["id"]] = related_topics
        relation_pages_by_id = {
            row["id"]: relation_dirs[int(row["layer"])] / f"{safe_filename(row['label'])}.md"
            for row in relation_nodes
        }
        generated: list[Path] = []
        topic_pages_by_id: dict[str, Path] = {}
        for concept in concepts:
            atoms = self._related_atoms(concept["id"])
            desired_name = f"{safe_filename(concept['label'])}.md"
            page = topic_dir / desired_name
            for existing in topic_dir.glob("*.md"):
                if existing.name.casefold() != desired_name.casefold() or existing.name == desired_name:
                    continue
                temporary = existing.with_name(
                    f".{existing.name}.{short_id('case-rename')}.tmp"
                )
                existing.rename(temporary)
                temporary.rename(page)
                break
            topic_pages_by_id[concept["id"]] = page
            notes = self._preserved_user_notes(page)
            memory_ids = [row["id"] for row in atoms]
            sources = list(dict.fromkeys(row["source"] or "unknown" for row in atoms))
            related = self.db.execute(
                """WITH RECURSIVE upper(id,layer,depth) AS (
                       SELECT id,layer,0 FROM neurons WHERE id=?
                       UNION
                       SELECT n.id,n.layer,u.depth+1
                       FROM upper u
                       JOIN synapses s ON s.source_id=u.id
                       JOIN neurons n ON n.id=s.target_id
                       WHERE n.layer>u.layer AND n.layer<=5 AND u.depth<3
                   )
                   SELECT DISTINCT n.id,n.layer,n.label FROM upper u
                   JOIN neurons n ON n.id=u.id
                   WHERE n.layer BETWEEN 4 AND 5
                   ORDER BY n.layer,n.label""",
                (concept["id"],),
            ).fetchall()
            related_lines = "\n".join(
                f"- L{row['layer']} [[{relation_pages_by_id[row['id']].relative_to(self.obsidian_dir).with_suffix('').as_posix()}|{row['label']}]]"
                for row in related
                if row["id"] in relation_pages_by_id
            ) or "- 暂无上层关系"
            memory_lines = "\n".join(
                f"- [[vault/memories/{row['id']}|{row['id']}]] — {compact(row['label'], 80)}"
                + (
                    f" · [[vault/evidence/{row['evidence_id']}|evidence]]"
                    if row["evidence_id"]
                    else ""
                )
                for row in atoms
            ) or "- No linked memory cards"
            page.write_text(
                "---\n"
                "view_type: compiled-memory\n"
                "generated: true\n"
                "do_not_ingest: true\n"
                f"compiled_at: {now()}\n"
                f"concept_id: {concept['id']}\n"
                f"source_memory_ids: {json.dumps(memory_ids, ensure_ascii=False)}\n"
                "---\n\n"
                f"# {concept['label']}\n\n"
                "<!-- GENERATED:START -->\n"
                "## 当前理解\n\n"
                f"{self._narrative(atoms)}\n\n"
                "## Linked Memories\n\n"
                f"{memory_lines}\n\n"
                "## 上层关系\n\n"
                f"{related_lines}\n\n"
                "## 来源\n\n"
                + "\n".join(f"- `{source}`" for source in sources)
                + "\n<!-- GENERATED:END -->\n\n"
                "<!-- USER-NOTES:START -->"
                + notes
                + "<!-- USER-NOTES:END -->\n",
                encoding="utf-8",
            )
            generated.append(page)

        family_generated: list[Path] = []
        family_pages_by_id: dict[str, Path] = {}
        grouped_concept_ids: set[str] = set()
        for family in self.concept_families():
            display_label = self._family_display_name(family)
            # Use the readable family name for the generated filename as well as the
            # page title, so Obsidian's file-list title is useful instead of an ID.
            page = family_dir / f"{safe_filename(display_label)}.md"
            member_lines = "\n".join(
                f"- [[主题/{safe_filename(str(member['label']))}|{member['label']}]]"
                for member in family["members"]
            ) or "- No active member concepts"
            shared_relation_lines = "\n".join(
                f"- L{relation['layer']} "
                f"[[{relation_pages_by_id[relation['id']].relative_to(self.obsidian_dir).with_suffix('').as_posix()}|{relation['label']}]] "
                f"（由 {relation['member_count']} 个成员概念共享）"
                for relation in family["shared_relations"]
                if relation["id"] in relation_pages_by_id
            ) or "- 当前没有被多个成员共同使用的 L4/L5 关系"
            review_link = (
                "- [[99 维护中心|前往维护中心审核]]\n\n"
                if family["status"] == "proposed"
                else ""
            )
            page.write_text(
                "---\n"
                "view_type: compiled-concept-family\n"
                "layer: L3F\n"
                "generated: true\n"
                "do_not_ingest: true\n"
                f"family_id: {family['id']}\n"
                f"status: {family['status']}\n"
                "---\n\n"
                f"# {display_label}\n\n"
                "L3F 是分组与注意力层；它不会合并成员 L3，也不会改变 L4–L6 的编号。\n\n"
                "## 当前状态\n\n"
                f"- 状态：`{family['status']}`\n"
                f"{review_link}"
                "## 成员概念\n\n"
                f"{member_lines}\n\n"
                "## 共享的上层关系\n\n"
                f"{shared_relation_lines}\n",
                encoding="utf-8",
            )
            family_generated.append(page)
            family_pages_by_id[str(family["id"])] = page
            if family["status"] == "confirmed":
                grouped_concept_ids.update(
                    str(member["id"]) for member in family["members"]
                )

        relation_generated: list[Path] = []
        for relation in relation_nodes:
            page = relation_pages_by_id[relation["id"]]
            related_topics = relation_topics_by_id[relation["id"]]
            topic_lines = "\n".join(
                f"- [[{topic_pages_by_id[row['id']].relative_to(self.obsidian_dir).with_suffix('').as_posix()}|{row['label']}]]"
                for row in related_topics
                if row["id"] in topic_pages_by_id
            ) or "- 暂无关联主题"
            relation_type = "Procedure" if relation["layer"] == 4 else "Persona / stable model"
            page.write_text(
                "---\n"
                "view_type: compiled-relation\n"
                "generated: true\n"
                "do_not_ingest: true\n"
                f"relation_id: {relation['id']}\n"
                f"layer: {relation['layer']}\n"
                "---\n\n"
                f"# {relation['label']}\n\n"
                "## 关系类型\n\n"
                f"L{relation['layer']} {relation_type}\n\n"
                "## 关联主题\n\n"
                f"{topic_lines}\n",
                encoding="utf-8",
            )
            relation_generated.append(page)

        sync_result = self.sync_obsidian_notes()
        generated_names = {page.name.casefold() for page in generated}
        stale_removed = 0
        for page in sorted(topic_dir.glob("*.md")):
            if page.name.casefold() in generated_names:
                continue
            page_text = page.read_text(encoding="utf-8")
            if "view_type: compiled-memory" not in page_text or "generated: true" not in page_text:
                continue
            page.unlink()
            stale_removed += 1
        stale_relation_removed = 0
        relation_generated_paths = {page.resolve() for page in relation_generated}
        for directory in relation_dirs.values():
            for page in sorted(directory.glob("*.md")):
                if page.resolve() in relation_generated_paths:
                    continue
                page_text = page.read_text(encoding="utf-8")
                if "view_type: compiled-relation" not in page_text or "generated: true" not in page_text:
                    continue
                page.unlink()
                stale_relation_removed += 1
        stale_family_removed = 0
        family_generated_paths = {page.resolve() for page in family_generated}
        for page in sorted(family_dir.glob("*.md")):
            if page.resolve() in family_generated_paths:
                continue
            page_text = page.read_text(encoding="utf-8")
            if (
                "view_type: compiled-concept-family" not in page_text
                or "generated: true" not in page_text
            ):
                continue
            page.unlink()
            stale_family_removed += 1

        inbox = self.maintenance_inbox()
        proposals = self.annotation_proposals()
        proposed_memories = self.db.execute(
            """SELECT n.*,e.source FROM neurons n
               LEFT JOIN evidence e ON e.id=n.evidence_id
               WHERE n.layer=1 AND n.status='proposed'
               ORDER BY n.created_at,n.id"""
        ).fetchall()
        proposed_memory_lines = "\n".join(
            f"- [[vault/memories/{row['id']}|{row['id']}]] — {compact(row['label'], 100)}"
            + (
                f" · [[vault/evidence/{row['evidence_id']}|evidence]]"
                if row["evidence_id"]
                else ""
            )
            + f"\n  - [ ] 确认 <!-- review:confirm:{row['id']} -->"
            + f"\n  - [ ] 需要修改 <!-- review:revise:{row['id']} -->"
            + f"\n  - [ ] 错误／拒绝 <!-- review:reject:{row['id']} -->"
            for row in proposed_memories
        ) or "- 当前没有待审核记忆"
        l3_review_concepts = self.db.execute(
            """SELECT * FROM neurons
               WHERE layer=3 AND status='proposed'
               ORDER BY label"""
        ).fetchall()
        l3_concept_sections: list[str] = []
        for concept in l3_review_concepts:
            topic_page = topic_pages_by_id.get(str(concept["id"]))
            topic_link = (
                f"[[{topic_page.relative_to(self.obsidian_dir).with_suffix('').as_posix()}|{concept['label']}]]"
                if topic_page
                else str(concept["label"])
            )
            support_count = len(self._related_atoms(concept["id"]))
            support_links = "\n".join(
                f"  - [[vault/memories/{atom['id']}|{atom['id']}]] — {compact(atom['label'], 100)}"
                for atom in self._related_atoms(concept["id"])
            ) or "  - 当前没有可用的 L1 支撑"
            l3_concept_sections.append(
                f"- {topic_link}（`{concept['id']}`，当前状态：`{concept['status']}`，L1 支撑：{support_count}）\n"
                f"  - L1 支撑记录：\n{support_links}\n"
                f"  - [ ] 确认 L3 概念 <!-- concept-review:confirm:{concept['id']} -->\n"
                f"  - [ ] 需要修改 <!-- concept-review:revise:{concept['id']} -->\n"
                f"  - [ ] 错误／拒绝 <!-- concept-review:reject:{concept['id']} -->"
            )
        l3_concept_lines = "\n".join(l3_concept_sections) or "- 当前没有可审核的 L3 概念"
        legacy_duplicate_markers = "\n".join(
            f"<!-- legacy concept-duplicate-review:merge-left:{item['id']} | "
            f"concept-duplicate-review:merge-right:{item['id']} | "
            f"concept-duplicate-review:distinct:{item['id']} -->"
            for item in self.concept_duplicate_candidates()
        )
        family_sections: list[str] = []
        for family in self.concept_families(status="proposed"):
            display_label = self._family_display_name(family)
            member_links = "\n".join(
                f"    - [[主题/{safe_filename(str(member['label']))}|{member['label']}]]"
                for member in family["members"]
            )
            family_sections.append(
                f"- `{family['id']}` **{display_label}** "
                f"（成员：{len(family['members'])}）\n"
                f"{member_links}\n"
                f"  - [[{family_pages_by_id[str(family['id'])].relative_to(self.obsidian_dir).with_suffix('').as_posix()}|打开 L3F 页面]]\n"
                f"  - [ ] 确认 L3F 概念家族 "
                f"<!-- concept-family-review:confirm:{family['id']} -->\n"
                f"  - [ ] 拒绝这一组家族关系 "
                f"<!-- concept-family-review:reject:{family['id']} -->"
            )
        family_lines = (
            "\n".join(family_sections)
            if family_sections
            else "- 当前没有待审核的 L3F 概念家族"
        )
        issue_lines = "\n".join(
            f"- `{item['id']}` **{item['severity']}** {item['kind']}：{item['details']}"
            for item in inbox["issues"]
        ) or "- 当前没有待处理问题"
        relation_lines = "\n".join(
            f"- `{item['id']}` {item['source_id']} → {item['relation']} → {item['target_id']}"
            for item in inbox["relations"]
        ) or "- 当前没有待审核关系"
        proposal_lines = "\n".join(
            f"- `{item['id']}` [[{item['page_path']}]]：{compact(str(item['notes']).replace(chr(10), ' '), 100)}"
            for item in proposals
        ) or "- 当前没有待审核人工批注"
        maintenance_page = self.obsidian_dir / "99 维护中心.md"
        maintenance_page.write_text(
            "---\nview_type: maintenance-dashboard\ngenerated: true\ndo_not_ingest: true\n---\n\n"
            "# 记忆维护中心\n\n"
            "> 本页展示待处理候选和 L3F 审核。所有写回核心记忆的动作都必须在命令行明确确认。\n\n"
            "## 待审核记忆\n\n" + proposed_memory_lines + "\n\n"
            "每条记忆只能勾选一个选项，然后点击下方“提交审核决定”按钮。确认或拒绝会更新 canonical 状态；需要修改会保留候选并加入维护问题。命令行备用：`python3 neural_memory.py --root /绝对路径/记忆库 sync-obsidian`。\n\n"
            "```neural-memory-submit\n提交审核决定\n```\n\n"
            "## L3 概念审核\n\n" + l3_concept_lines + "\n\n"
            "这里只显示尚未确认的 proposed L3。确认后会进入正式 L3 路由，并从审核列表中移除；拒绝会移除这一条 L3 连接，但不会删除其 L1 原始记忆。\n\n"
            + legacy_duplicate_markers
            + "\n## L3F 概念家族审核\n\n" + family_lines + "\n\n"
            "L3F 只对相关 L3 进行分组和注意力导航，不会合并概念，也不会改变 L4–L6 的编号。确认后首页会折叠展示其成员；拒绝后不会重复生成完全相同的分组。\n\n"
            "## 人工批注候选\n\n" + proposal_lines + "\n\n"
            "## 系统问题\n\n" + issue_lines + "\n\n"
            "## 待审核关系\n\n" + relation_lines + "\n",
            encoding="utf-8",
        )

        archived = self.db.execute(
            """SELECT n.*,e.source FROM neurons n
               LEFT JOIN evidence e ON e.id=n.evidence_id
               WHERE n.layer=1 AND n.status='archived'
               ORDER BY n.created_at,n.id"""
        ).fetchall()
        archive_lines = "\n".join(
            f"- [[vault/memories/{row['id']}|{row['id']}]] — {compact(row['label'], 80)}"
            + (
                f" · [[vault/evidence/{row['evidence_id']}|evidence]]"
                if row["evidence_id"]
                else ""
            )
            for row in archived
        ) or "- No archived memories"
        archive_page = self.obsidian_dir / "98 Archive.md"
        archive_page.write_text(
            "---\nview_type: compiled-archive\ngenerated: true\ndo_not_ingest: true\n---\n\n"
            "# Archive\n\n"
            "> Archived canonical records remain available for audit and rollback.\n\n"
            + archive_lines
            + "\n",
            encoding="utf-8",
        )

        stats = self.stats()
        home = self.obsidian_dir / "00 首页.md"
        confirmed_family_links = "\n".join(
            f"- [[{family_pages_by_id[str(family['id'])].relative_to(self.obsidian_dir).with_suffix('').as_posix()}]] — {self._family_display_name(family)}"
            for family in self.concept_families(status="confirmed")
            if str(family["id"]) in family_pages_by_id
        )
        proposed_family_links = "\n".join(
            f"- [[{family_pages_by_id[str(family['id'])].relative_to(self.obsidian_dir).with_suffix('').as_posix()}]] — {self._family_display_name(family)} · 待审核"
            for family in self.concept_families(status="proposed")
            if str(family["id"]) in family_pages_by_id
        ) or "- 当前没有待审核的 L3F"
        ungrouped_links = "\n".join(
            f"- [[主题/{page.stem}]]"
            for concept_id, page in topic_pages_by_id.items()
            if concept_id not in grouped_concept_ids
        )
        links = "\n".join(
            item for item in (confirmed_family_links, ungrouped_links) if item
        ) or "- 暂无主题或概念家族页面"
        home.write_text(
            "---\nview_type: compiled-memory\ngenerated: true\ndo_not_ingest: true\n---\n\n"
            "# 记忆系统首页\n\n"
            "## L3F 概念家族与主题导航\n\n"
            f"{links}\n- [[98 Archive]]\n- [[99 维护中心]]\n\n"
            "## 待审核的 L3F\n\n"
            f"{proposed_family_links}\n- [[99 维护中心|在维护中心审计和审核]]\n\n"
            "## 系统状态\n\n"
            f"- L0 原始证据：{stats['layers'].get(0, 0)}\n"
            + "\n".join(
                f"- L{layer} 神经元：{count}"
                for layer, count in sorted(stats["layers"].items())
                if layer != 0
            )
            + f"\n- 突触：{stats['synapses']}\n"
            "\n> `vault/` 保存可审计的记忆与证据；概念身份、家族和审核决定等后台记录位于隐藏的 `.neural-memory/`。平时可从本页和维护中心阅读、审计记忆。\n"
            "> 此目录是可重建的阅读视图，不作为记忆碎片重新摄入。\n",
            encoding="utf-8",
        )
        return {
            "pages": len(generated) + len(family_generated) + len(relation_generated) + 3,
            "root": str(self.obsidian_dir),
            "annotation_sync": sync_result,
            "review_sync": review_sync,
            "stale_topic_pages_removed": stale_removed,
            "stale_relation_pages_removed": stale_relation_removed,
            "stale_family_pages_removed": stale_family_removed,
        }

    def evaluate(self, cases_path: Path, limit: int = 3) -> dict[str, object]:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        details: list[dict[str, object]] = []
        gate_correct = top1_hits = topk_hits = 0
        positive = 0
        for case in cases:
            query = str(case["query"])
            expected_known = bool(case["known"])
            expected = [str(item).lower() for item in case.get("expected", [])]
            known, peak, _ = self.probe(query)
            cards = self.recall(query, limit) if known else []
            texts = [(card.label + " " + card.summary).lower() for card in cards]
            top1 = bool(expected and texts and any(term in texts[0] for term in expected))
            topk = bool(expected and any(any(term in text for term in expected) for text in texts))
            gate_correct += int(known == expected_known)
            if expected_known:
                positive += 1
                top1_hits += int(top1)
                topk_hits += int(topk)
            details.append({
                "query": query,
                "expected_known": expected_known,
                "known": known,
                "peak": round(peak, 4),
                "top1_hit": top1,
                "topk_hit": topk,
                "cards": [card.label for card in cards],
            })
        total = len(cases)
        return {
            "cases": total,
            "gate_accuracy": round(gate_correct / total, 4) if total else 0.0,
            "top1_accuracy": round(top1_hits / positive, 4) if positive else 0.0,
            f"top{limit}_recall": round(topk_hits / positive, 4) if positive else 0.0,
            "details": details,
        }

    @serialized_write
    def export_bundle(self, destination: Path) -> dict[str, object]:
        """Export an atomic bundle from a transactionally consistent DB snapshot."""
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.db.commit()
        with tempfile.TemporaryDirectory(prefix="neural-backup-", dir=destination.parent) as temp:
            temp_dir = Path(temp)
            snapshot_path = temp_dir / "memory.sqlite3"
            snapshot_db = sqlite3.connect(snapshot_path)
            try:
                self.db.backup(snapshot_db)
                check = snapshot_db.execute("PRAGMA integrity_check").fetchone()[0]
                if check != "ok":
                    raise RuntimeError(f"snapshot integrity check failed: {check}")
            finally:
                snapshot_db.close()
            files: list[tuple[Path, str]] = [(snapshot_path, "memory.sqlite3")]
            encoder_config = self.root / "encoder.json"
            if encoder_config.is_file():
                files.append((encoder_config, "encoder.json"))
            files.extend(
                (path, str(path.relative_to(self.root)))
                for path in self.root.joinpath("vault").rglob("*")
                if path.is_file()
            )
            files.extend(
                (path, str(path.relative_to(self.root)))
                for path in self.obsidian_dir.rglob("*")
                if path.is_file()
            )
            checksums = {
                relative: hashlib.sha256(path.read_bytes()).hexdigest()
                for path, relative in files
            }
            manifest = {
                "format": "neural-memory-bundle/v1",
                "created_at": now(),
                "encoder": {
                    "name": self.encoder.name,
                    "dimensions": self.encoder.dimensions,
                },
                "stats": self.stats(),
                "files": checksums,
            }
            temporary_bundle = temp_dir / "bundle.tmp"
            with zipfile.ZipFile(
                temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                )
                for path, relative in files:
                    archive.write(path, relative)
            with temporary_bundle.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_bundle, destination)
        return {
            "bundle": str(destination),
            "files": len(files),
            "bytes": destination.stat().st_size,
            "stats": manifest["stats"],
        }

    @serialized_write
    def create_backup(self, directory: Path, keep: int = 10) -> dict[str, object]:
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        destination = directory / f"neural-memory-{stamp}.nmem"
        result = self.export_bundle(destination)
        backups = sorted(
            directory.glob("neural-memory-*.nmem"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        removed: list[str] = []
        if keep > 0:
            for old in backups[keep:]:
                old.unlink()
                removed.append(str(old))
        result["removed_old_backups"] = removed
        result["retained"] = min(len(backups), keep) if keep > 0 else len(backups)
        return result


def verify_bundle(bundle: Path) -> dict[str, object]:
    """Verify format, safe paths and every checksum without extracting files."""
    bundle = bundle.resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    with zipfile.ZipFile(bundle, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise RuntimeError("bundle has no manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "neural-memory-bundle/v1":
            raise RuntimeError(f"unsupported bundle format: {manifest.get('format')}")
        expected = manifest.get("files", {})
        if set(expected) - names:
            raise RuntimeError("bundle is missing files declared by its manifest")
        for name, digest in expected.items():
            pure = Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"unsafe bundle path: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError(f"checksum mismatch: {name}")
    return {
        "valid": True,
        "bundle": str(bundle),
        "files": len(expected),
        "created_at": manifest.get("created_at"),
        "stats": manifest.get("stats", {}),
    }


def import_bundle(bundle: Path, destination: Path) -> dict[str, object]:
    """Verify and restore through staging, then atomically publish the result."""
    bundle = bundle.resolve()
    destination = destination.resolve()
    verify_bundle(bundle)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"destination must be empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="neural-restore-", dir=destination.parent
    ) as temp:
        staging = Path(temp) / "restored"
        staging.mkdir()
        with zipfile.ZipFile(bundle, "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))
            expected = manifest.get("files", {})
            for name in expected:
                target = staging / Path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        restored_db = sqlite3.connect(staging / "memory.sqlite3")
        try:
            integrity = restored_db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"restored database integrity failed: {integrity}")
        finally:
            restored_db.close()
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    return {
        "restored_to": str(destination),
        "files": len(expected),
        "stats": manifest.get("stats", {}),
    }


def seed_demo(memory: NeuralMemory) -> None:
    samples = [
        (
            "The memory system should use three stages: awareness, low-token recall, and optional access to full evidence.",
            ["AI Memory System"],
            ["Prefers a supervised and token-efficient workflow"],
        ),
        (
            "Detailed memory content should live in Obsidian or Markdown; the database index is a rebuildable derived layer.",
            ["AI Memory System", "Knowledge Management"],
            ["Prefers a supervised and token-efficient workflow"],
        ),
        (
            "Long tasks can offload large tool output to files and keep only a Mermaid task canvas with node IDs in context.",
            ["AI Memory System", "Token Optimization"],
            ["Prefers a supervised and token-efficient workflow"],
        ),
        (
            "Important or uncertain automatic memories should enter the Inbox and require user confirmation before becoming durable facts.",
            ["AI Memory System", "Memory Governance"],
            ["Prefers a supervised and token-efficient workflow"],
        ),
        (
            "Retrieval should use sparse activation and return only three to five memory cards, following evidence pointers only when needed.",
            ["AI Memory System", "Token Optimization"],
            ["Prefers a supervised and token-efficient workflow"],
        ),
    ]
    for text, topics, schemas in samples:
        memory.remember(
            text,
            "demo-conversation",
            topics,
            schemas,
            confirmed=True,
            episode="Neural Memory System Design",
            procedures=["Progressive Memory Maintenance"],
            domain="AI Memory and Knowledge Management",
        )


def print_recall(
    memory: NeuralMemory,
    query: str,
    limit: int,
    detail: bool,
    learn: bool,
    force: bool = False,
) -> int:
    known, peak, activated = memory.probe(query)
    print(f"awareness: {'KNOWN' if known else 'UNKNOWN'} (peak={peak:.3f})")
    clusters = [item for item in activated if item.layer > 1][:4]
    if clusters:
        print("activated clusters:")
        for item in clusters:
            print(f"  L{item.layer} {item.label}  a={item.activation:.3f}")
    if not known and not force:
        print("recall gate: closed; no memory injected (use --force to inspect candidates)")
        return 1
    cards = memory.recall(query, limit)
    if not cards:
        print("no reliable memory cards")
        return 1
    print("memory cards:")
    for index, card in enumerate(cards, 1):
        print(
            f"  {index}. [{card.id}] a={card.activation:.3f} {card.status}\n"
            f"     {compact(card.summary, 150)}"
        )
        if detail and card.evidence_id:
            body = memory.evidence_text(card.evidence_id)
            print(textwrap.indent(body.rstrip(), "       "))
    if learn:
        memory.reinforce([card.id for card in cards])
        print("hebbian learning: co-recalled synapses reinforced")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Auditable neural-style memory demo")
    result.add_argument("--root", type=Path, default=Path("./demo-memory"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("seed-demo")
    importer = commands.add_parser("import-mdkb")
    importer.add_argument("--mdkb-bin", type=Path, default=Path("/opt/homebrew/bin/mdkb"))
    importer.add_argument("--workspace", type=Path, required=True)
    importer.add_argument("--limit", type=int, default=0)
    importer.add_argument(
        "--copy-content",
        action="store_true",
        help="copy full mdkb Markdown while retaining mdkb as source authority",
    )

    remember = commands.add_parser("remember")
    remember.add_argument("text")
    remember.add_argument("--source", default="manual")
    remember.add_argument("--topic", action="append", default=[])
    remember.add_argument("--schema", action="append", default=[])
    remember.add_argument("--episode")
    remember.add_argument("--procedure", action="append", default=[])
    remember.add_argument("--domain")
    remember.add_argument("--expires")
    remember.add_argument("--supersedes", action="append", default=[])
    remember.add_argument("--conflicts", action="append", default=[])
    remember.add_argument("--importance", type=float, default=0.7)
    remember.add_argument("--confirmed", action="store_true")

    probe = commands.add_parser("probe")
    probe.add_argument("query")
    recall = commands.add_parser("recall")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=5)
    recall.add_argument("--detail", action="store_true")
    recall.add_argument("--learn", action="store_true")
    recall.add_argument("--force", action="store_true")
    explain = commands.add_parser("explain")
    explain.add_argument("query")
    explain.add_argument("--limit", type=int, default=10)

    review = commands.add_parser("review")
    review.add_argument("action", choices=["list", "confirm", "reject", "stale", "archive"])
    review.add_argument("neuron_id", nargs="?")
    restore_rejected = commands.add_parser("restore-rejected")
    restore_rejected.add_argument("neuron_id")
    commands.add_parser("archive-orphan-evidence")
    maintenance = commands.add_parser("maintenance")
    maintenance.add_argument(
        "action",
        choices=["scan", "inbox", "resolve", "ignore", "confirm-relation", "reject-relation"],
    )
    maintenance.add_argument("target_id", nargs="?")
    commands.add_parser("doctor")
    commands.add_parser("network")
    commands.add_parser("rebuild")
    commands.add_parser("consolidate")
    reencoder = commands.add_parser("reencode")
    reencoder.add_argument("config", type=Path)
    commands.add_parser("compile-obsidian")
    commands.add_parser("sync-obsidian")
    obsidian_review = commands.add_parser("obsidian-review")
    obsidian_review.add_argument("action", choices=["list", "show", "accept", "reject"])
    obsidian_review.add_argument("proposal_id", nargs="?")
    concept_duplicate = commands.add_parser("concept-duplicate")
    concept_duplicate.add_argument(
        "action",
        choices=["list", "merge-left", "merge-right", "distinct"],
    )
    concept_duplicate.add_argument("review_id", nargs="?")
    concept_family = commands.add_parser("concept-family")
    concept_family.add_argument("action", choices=["list", "confirm", "reject"])
    concept_family.add_argument("family_id", nargs="?")
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("cases", type=Path)
    evaluator.add_argument("--limit", type=int, default=3)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("query")
    benchmark.add_argument("--limit", type=int, default=3)
    exporter = commands.add_parser("export-bundle")
    exporter.add_argument("destination", type=Path)
    backup = commands.add_parser("backup")
    backup.add_argument("directory", type=Path)
    backup.add_argument("--keep", type=int, default=10)
    verifier = commands.add_parser("verify-bundle")
    verifier.add_argument("bundle", type=Path)
    importer_bundle = commands.add_parser("import-bundle")
    importer_bundle.add_argument("bundle", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "verify-bundle":
        print(json.dumps(verify_bundle(args.bundle), ensure_ascii=False, indent=2))
        return 0
    if args.command == "import-bundle":
        print(json.dumps(import_bundle(args.bundle, args.root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "reencode":
        encoder = load_encoder_config(args.config)
        memory = NeuralMemory(args.root, encoder, allow_encoder_mismatch=True)
    else:
        memory = NeuralMemory(args.root, resolve_encoder(args.root))
    try:
        if args.command == "init":
            print(f"initialized: {memory.root}")
        elif args.command == "seed-demo":
            seed_demo(memory)
            print(json.dumps(memory.stats(), ensure_ascii=False))
        elif args.command == "import-mdkb":
            print(json.dumps(
                memory.import_mdkb(
                    args.mdkb_bin,
                    args.workspace,
                    args.limit,
                    args.copy_content,
                ),
                ensure_ascii=False,
                indent=2,
            ))
        elif args.command == "remember":
            neuron_id = memory.remember(
                args.text,
                args.source,
                args.topic,
                args.schema,
                max(0.0, min(1.0, args.importance)),
                args.confirmed,
                args.episode,
                args.procedure,
                args.domain,
                args.expires,
                args.supersedes,
                args.conflicts,
            )
            print(f"created L1 neuron: {neuron_id}")
        elif args.command == "probe":
            known, peak, activated = memory.probe(args.query)
            print(json.dumps({
                "known": known,
                "peak_activation": round(peak, 4),
                "active_clusters": [
                    {"id": item.id, "layer": item.layer, "label": item.label,
                     "activation": round(item.activation, 4)}
                    for item in activated if item.layer > 1
                ][:3],
            }, ensure_ascii=False, indent=2))
            return 0 if known else 1
        elif args.command == "recall":
            return print_recall(
                memory, args.query, args.limit, args.detail, args.learn, args.force
            )
        elif args.command == "explain":
            known, peak, activated = memory.probe(args.query)
            print(json.dumps({
                "query": args.query,
                "known": known,
                "peak_l1_activation": round(peak, 4),
                "encoder": memory.stats()["encoder"],
                "l3f_routing": memory.concept_family_routes(args.query),
                "formula": "retention × governance × (0.45 vector + 0.45 BM25 + 0.10 lexical) + spread",
                "activations": [
                    {
                        "id": item.id,
                        "layer": item.layer,
                        "label": item.label,
                        "activation": round(item.activation, 4),
                        "direct": round(item.direct_activation, 4),
                        "spread": round(item.spread_activation, 4),
                        "vector": round(item.vector_score, 4),
                        "bm25": round(item.bm25_score, 4),
                        "lexical": round(item.lexical_score, 4),
                        "stability": round(item.stability, 4),
                        "retention": round(item.retention, 4),
                    }
                    for item in activated[:args.limit]
                ],
            }, ensure_ascii=False, indent=2))
            return 0 if known else 1
        elif args.command == "review":
            if args.action == "list":
                for row in memory.proposed():
                    print(f"{row['id']} L{row['layer']} c={row['confidence']:.2f} {row['label']}")
            else:
                if not args.neuron_id:
                    print("neuron_id is required", file=sys.stderr)
                    return 2
                status = {
                    "confirm": "confirmed",
                    "reject": "rejected",
                    "stale": "stale",
                    "archive": "archived",
                }[args.action]
                if not memory.review(args.neuron_id, status):
                    print("neuron not found", file=sys.stderr)
                    return 1
                if status == "rejected":
                    memory.compile_obsidian()
                print(f"{args.neuron_id}: {status}")
        elif args.command == "restore-rejected":
            if not memory.restore_rejected(args.neuron_id):
                print("rejected record not found or cannot be restored", file=sys.stderr)
                return 1
            memory.compile_obsidian()
            print(f"{args.neuron_id}: restored as proposed")
        elif args.command == "archive-orphan-evidence":
            print(json.dumps({"moved": memory.archive_orphan_evidence()}, indent=2))
        elif args.command == "maintenance":
            if args.action == "scan":
                print(json.dumps(memory.scan_maintenance(), ensure_ascii=False, indent=2))
            elif args.action == "inbox":
                print(json.dumps(memory.maintenance_inbox(), ensure_ascii=False, indent=2))
            else:
                if not args.target_id:
                    print("target_id is required", file=sys.stderr)
                    return 2
                if args.action in ("resolve", "ignore"):
                    changed = memory.resolve_issue(args.target_id, args.action)
                else:
                    decision = "confirm" if args.action == "confirm-relation" else "reject"
                    changed = memory.review_relation(args.target_id, decision)
                if not changed:
                    print("target not found or already closed", file=sys.stderr)
                    return 1
                print(f"{args.target_id}: {args.action}")
        elif args.command == "doctor":
            print(json.dumps(memory.health_report(), ensure_ascii=False, indent=2))
        elif args.command == "network":
            nodes, edges = memory.network()
            print("nodes:")
            for row in nodes:
                print(f"  {row['id']} L{row['layer']} {row['status']} {row['label']}")
            print("strongest synapses:")
            for row in edges[:30]:
                print(
                    f"  {row['source_id']} <-> {row['target_id']} "
                    f"w={row['weight']:.2f} {row['relation']}"
                )
        elif args.command == "rebuild":
            print(json.dumps(memory.rebuild_index(), ensure_ascii=False, indent=2))
        elif args.command == "consolidate":
            print(json.dumps(memory.consolidate(), ensure_ascii=False, indent=2))
        elif args.command == "reencode":
            print(json.dumps(memory.reencode_all(args.config), ensure_ascii=False, indent=2))
        elif args.command == "compile-obsidian":
            print(json.dumps(memory.compile_obsidian(), ensure_ascii=False, indent=2))
        elif args.command == "sync-obsidian":
            review_sync = memory.sync_obsidian_reviews()
            annotation_sync = memory.sync_obsidian_notes()
            review_changes = sum(
                int(review_sync[key])
                for key in (
                    "confirmed",
                    "needs_revision",
                    "rejected",
                    "concepts_confirmed",
                    "concepts_needs_revision",
                    "concepts_rejected",
                    "concepts_merged",
                    "concepts_kept_distinct",
                    "families_confirmed",
                    "families_rejected",
                )
            )
            should_compile = review_changes > 0 or int(annotation_sync["created"]) > 0
            view = memory.compile_obsidian() if should_compile and not review_sync["errors"] else None
            print(json.dumps({
                "memory_reviews": review_sync,
                "annotations": annotation_sync,
                "view": view,
            }, ensure_ascii=False, indent=2))
        elif args.command == "obsidian-review":
            if args.action == "list":
                print(json.dumps(memory.annotation_proposals(), ensure_ascii=False, indent=2))
            else:
                if not args.proposal_id:
                    print("proposal_id is required", file=sys.stderr)
                    return 2
                if args.action == "show":
                    row = memory.db.execute(
                        "SELECT * FROM annotation_proposals WHERE id=?",
                        (args.proposal_id,),
                    ).fetchone()
                    if not row:
                        print("proposal not found", file=sys.stderr)
                        return 1
                    print(json.dumps(dict(row), ensure_ascii=False, indent=2))
                else:
                    outcome = memory.review_annotation(args.proposal_id, args.action)
                    if outcome is None:
                        print("proposal not found or already reviewed", file=sys.stderr)
                        return 1
                    print(json.dumps({
                        "proposal_id": args.proposal_id,
                        "decision": args.action,
                        "result": outcome,
                    }, ensure_ascii=False, indent=2))
        elif args.command == "concept-duplicate":
            if args.action == "list":
                print(json.dumps(
                    memory.concept_duplicate_candidates(),
                    ensure_ascii=False,
                    indent=2,
                ))
            else:
                if not args.review_id:
                    print("review_id is required", file=sys.stderr)
                    return 2
                if not memory.review_concept_duplicate(
                    args.review_id,
                    args.action,
                ):
                    print(
                        "duplicate review not found or already closed",
                        file=sys.stderr,
                    )
                    return 1
                memory.compile_obsidian()
                print(f"{args.review_id}: {args.action}")
        elif args.command == "concept-family":
            if args.action == "list":
                print(json.dumps(
                    memory.concept_families(),
                    ensure_ascii=False,
                    indent=2,
                ))
            else:
                if not args.family_id:
                    print("family_id is required", file=sys.stderr)
                    return 2
                if not memory.review_concept_family(
                    args.family_id,
                    args.action,
                ):
                    print(
                        "concept family not found or already closed",
                        file=sys.stderr,
                    )
                    return 1
                memory.compile_obsidian()
                print(f"{args.family_id}: {args.action}")
        elif args.command == "evaluate":
            print(json.dumps(memory.evaluate(args.cases, args.limit), ensure_ascii=False, indent=2))
        elif args.command == "benchmark":
            print(json.dumps(memory.benchmark(args.query, args.limit), ensure_ascii=False, indent=2))
        elif args.command == "export-bundle":
            print(json.dumps(memory.export_bundle(args.destination), ensure_ascii=False, indent=2))
        elif args.command == "backup":
            print(json.dumps(
                memory.create_backup(args.directory, max(0, args.keep)),
                ensure_ascii=False,
                indent=2,
            ))
        return 0
    finally:
        memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
