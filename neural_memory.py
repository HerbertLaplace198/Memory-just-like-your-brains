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
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


VECTOR_DIMS = 1024
TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+|[\u3400-\u9fff]+")
MEMORY_FORMAT = "neural-memory-record/v2"
CONCEPT_ALIASES = {
    "asset allocation": "Asset Allocation",
    "btc": "Bitcoin",
    "cashflow": "Investment Risk and Cash Flow",
    "codex": "Codex",
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

    def encode(self, text: str) -> list[float]: ...

    def encode_many(self, texts: list[str]) -> list[list[float]]: ...


class HashEncoder:
    """Portable fallback: deterministic, private and dependency-free."""

    name = "feature-hash-v1"
    gate_threshold = 0.48

    def __init__(self, dimensions: int = VECTOR_DIMS, gate_threshold: float = 0.48):
        self.dimensions = dimensions
        self.gate_threshold = gate_threshold

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
        self.provider = provider
        self.endpoint = endpoint
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.gate_threshold = gate_threshold
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
        )
    return LocalHTTPEncoder(
        provider,
        str(config["endpoint"]),
        str(config["model"]),
        int(config["dimensions"]),
        float(config.get("timeout", 30.0)),
        float(config.get("gate_threshold", 0.30)),
    )


def resolve_encoder(root: Path, explicit_config: Path | None = None) -> TextEncoder:
    config_path = explicit_config or root.resolve() / "encoder.json"
    return load_encoder_config(config_path) if config_path.is_file() else HashEncoder()


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def compact(text: str, width: int = 100) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def rough_tokens(text: str) -> int:
    """A deliberately simple local estimate, suitable only for A/B comparison."""
    chinese = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    other = len(text) - chinese
    return chinese + math.ceil(other / 4)


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "-", text).strip().strip(".")
    return compact(cleaned or "Untitled", 80)


def canonical_concept(text: str) -> str:
    """Normalize known bilingual or legacy aliases to one L3 topic label."""
    label = " ".join(text.strip().split())
    return CONCEPT_ALIASES.get(label.casefold(), label)


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
        self.evidence_dir = self.vault_dir / "evidence"
        self.memory_dir = self.vault_dir / "memories"
        self.rejected_dir = self.vault_dir / ".rejected"
        self.obsidian_dir = self.root / "obsidian-view"
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
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
                expires_at TEXT
            );
            CREATE TABLE IF NOT EXISTS synapses (
                source_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES neurons(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                last_fired TEXT,
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
            """
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(neurons)")}
        if "expires_at" not in columns:
            self.db.execute("ALTER TABLE neurons ADD COLUMN expires_at TEXT")
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

    def _find_named(self, layer: int, label: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM neurons WHERE layer=? AND lower(label)=lower(?) AND status!='rejected'",
            (layer, label),
        ).fetchone()

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
    ) -> str:
        neuron_id = neuron_id or short_id(f"l{layer}")
        representation = self._encode(label + " " + summary)
        self.db.execute(
            """INSERT INTO neurons
               (id, layer, label, summary, vector, status, confidence, importance,
                evidence_id, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                now(),
                expires_at,
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
        concept_labels = require_english_labels(canonical_concepts(list(topics)), "topics")
        procedure_labels = require_english_labels(procedures, "procedures")
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
            "episode": episode or "",
            "concepts": concept_labels,
            "procedures": procedure_labels,
            "personas": list(dict.fromkeys(item.strip() for item in schemas if item.strip())),
            "domain": (domain or "").strip(),
            "expires_at": (expires_at or "").strip(),
            "supersedes": list(dict.fromkeys(x.strip() for x in supersedes if x.strip())),
            "conflicts": list(dict.fromkeys(x.strip() for x in conflicts if x.strip())),
        }
        write_record(self.memory_dir / f"{neuron_id}.md", record, text)
        self._index_canonical_record(record, text)
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
        )

        for target_id in record.get("supersedes", []):
            self._add_relation(
                neuron_id,
                str(target_id),
                "supersedes",
                "Declared as a replacement during ingestion; human review is required before the old memory is archived.",
            )
        for target_id in record.get("conflicts", []):
            self._add_relation(
                neuron_id,
                str(target_id),
                "conflicts_with",
                "Declared as a conflict during ingestion; retain both memories pending human review.",
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
        concept_labels = english_only_labels(
            canonical_concepts([str(x) for x in record.get("concepts", [])])
        )
        if not concept_labels and any(token in source.casefold() for token in ("skill", "tool")):
            concept_labels = ["Tools"]
        level_specs: list[tuple[int, list[str], str, str]] = [
            (2, [str(record.get("episode", ""))], "episode", "episodic memory"),
            (3, concept_labels, "member_of", "semantic concept"),
            (4, english_only_labels(record.get("procedures", [])), "used_in", "procedural memory"),
            (5, [str(x) for x in record.get("personas", [])], "supports", "stable model"),
            (6, [str(record.get("domain", ""))], "routes_to", "meta-memory domain"),
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
                    upper_id = self._create_neuron(
                        layer,
                        upper_label,
                        f"{kind}: {upper_label}",
                        upper_status,
                        min(0.95, confidence),
                        min(1.0, importance + layer * 0.02),
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
                    f"Possible conflict with {peer['id']} (lexical overlap {shared:.2f}); compare the original evidence manually.",
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
            summary = f"mdkb {entry_type}: {title} " + " ".join(
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
                    2, tag, f"mdkb tag cluster #{tag}", "confirmed", 0.95, 0.75
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
            summary = f"mdkb {entry['type']}: {entry['title']} " + " ".join(
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
                    2, topic, f"mdkb tag cluster #{topic}", "confirmed", 0.95, 0.75
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
    ) -> list[ActivatedNeuron]:
        query_vector = self._encode(query)
        query_features = set(features(query))
        rows = self.db.execute(
            "SELECT * FROM neurons WHERE status NOT IN ('rejected','stale','archived')"
        ).fetchall()
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
            governance = row["confidence"] * (0.65 + 0.35 * row["importance"])
            direct[row["id"]] = fused_score * governance
            components[row["id"]] = {
                "vector": vector_score,
                "bm25": bm25_score,
                "lexical": lexical_coverage,
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
                )
            )
        return result

    def probe(self, query: str) -> tuple[bool, float, list[ActivatedNeuron]]:
        activated = self.activate(query)
        l1 = [item for item in activated if item.layer == 1]
        threshold = float(getattr(self.encoder, "gate_threshold", 0.06))
        if isinstance(self.encoder, HashEncoder):
            gate_score = l1[0].activation if l1 else 0.0
        else:
            supported_semantic = max(
                (
                    item.vector_score
                    for item in l1
                    if item.bm25_score > 0.0 or item.lexical_score > 0.0
                ),
                default=0.0,
            )
            semantic_only = max((item.vector_score for item in l1), default=0.0)
            # A semantic-only match needs a margin above the configured gate.
            # This preserves genuine paraphrases while rejecting isolated model
            # similarities that have no lexical support in the active corpus.
            gate_score = max(supported_semantic, semantic_only - 0.15)
        return gate_score >= threshold, gate_score, activated

    def _topic_memory_ids(
        self, activated: list[ActivatedNeuron], query: str = ""
    ) -> set[str]:
        """Return L1 memories attached to the strongest active L3 route."""
        routes = sorted(
            [
                (item.id, item.label, item.direct_activation)
                for item in activated
                if item.layer == 3 and item.direct_activation > 0
            ],
            key=lambda route: route[2],
            reverse=True,
        )[:1]
        if not routes:
            return set()
        route_id, route_label, _ = routes[0]
        route_ids = [route_id]
        if any(token in query.casefold() for token in ("continue", "resume")):
            parent_label = TOPIC_PARENTS.get(route_label)
            parent = self._find_named(3, parent_label) if parent_label else None
            if parent and parent["status"] not in {"rejected", "stale", "archived"}:
                route_ids.append(parent["id"])
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

    def recall(self, query: str, limit: int = 5) -> list[ActivatedNeuron]:
        activated = self.activate(query)
        cards = [item for item in activated if item.layer == 1]
        topic_memory_ids = self._topic_memory_ids(activated, query)
        if topic_memory_ids:
            scoped = [
                card for card in cards
                if card.id in topic_memory_ids
            ]
            if scoped:
                cards = scoped
        return cards[:limit]

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
        """Hebbian learning: neurons recalled together strengthen their links."""
        for index, left in enumerate(neuron_ids):
            for right in neuron_ids[index + 1 :]:
                self._connect(left, right, "co_recalled", amount)
            self.db.execute("UPDATE neurons SET last_used=? WHERE id=?", (now(), left))
        self.db.execute(
            "UPDATE synapses SET last_fired=? WHERE source_id IN ({})".format(
                ",".join("?" for _ in neuron_ids)
            ),
            (now(), *neuron_ids),
        ) if neuron_ids else None
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
            self.db.commit()
            return True
        confidence = 0.98 if status == "confirmed" else row["confidence"]
        self.db.execute(
            "UPDATE neurons SET status=?, confidence=? WHERE id=?",
            (status, confidence, neuron_id),
        )
        canonical = self.memory_dir / f"{neuron_id}.md"
        if canonical.exists():
            metadata, body = read_record(canonical)
            metadata["status"] = status
            metadata["confidence"] = confidence
            write_record(canonical, metadata, body)
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
                row["id"], "needs_review", "info", "Proposed atomic memory has not been confirmed by the user."
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
                f"Expiry date {row['expires_at']} has passed; verify it before marking it stale or writing a replacement.",
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
                        f"The {role} neuron {neuron_id} for relation {relation['id']} does not exist.",
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
        for metadata, body in records:
            evidence_path = self.root / str(metadata["evidence_path"])
            if not evidence_path.is_file():
                raise FileNotFoundError(f"canonical evidence missing: {evidence_path}")
            self._index_canonical_record(metadata, body)
        self.db.commit()
        return {"records": len(records), "stats": self.stats()}

    def _related_atoms(self, neuron_id: str, max_depth: int = 5) -> list[sqlite3.Row]:
        """Return only active L1 memories directly assigned to this topic."""
        return self.db.execute(
            """SELECT n.*,e.source FROM synapses s
               JOIN neurons n ON n.id=s.target_id
               LEFT JOIN evidence e ON e.id=n.evidence_id
               WHERE s.source_id=? AND s.relation='member_of'
                 AND n.layer=1 AND n.status NOT IN ('rejected','archived')
               ORDER BY n.created_at""",
            (neuron_id,),
        ).fetchall()

    @staticmethod
    def _preserved_user_notes(path: Path) -> str:
        if not path.exists():
            return "\nAdd human notes here. Notes are never ingested automatically.\n"
        match = USER_NOTES_RE.search(path.read_text(encoding="utf-8"))
        return match.group(1) if match else "\nAdd human notes here.\n"

    @staticmethod
    def _narrative(rows: list[sqlite3.Row]) -> str:
        statements = [row["summary"].strip().rstrip(".") for row in rows if row["summary"].strip()]
        if not statements:
            return "There are not enough confirmed memories to form a narrative yet."
        connectors = ["Currently, ", "Additionally, ", "In a later record, ", "Meanwhile, ", "Overall, "]
        paragraphs: list[str] = []
        for index in range(0, len(statements), 3):
            group = statements[index : index + 3]
            sentence = "".join(
                (connectors[(index + offset) % len(connectors)] if offset == 0 else "; ")
                + statement
                for offset, statement in enumerate(group)
            )
            paragraphs.append(sentence + ".")
        return "\n\n".join(paragraphs)

    @serialized_write
    def sync_obsidian_notes(self) -> dict[str, int]:
        """Turn meaningful USER-NOTES into review proposals, never direct memories."""
        discovered = created = 0
        topic_dir = self.obsidian_dir / "topics"
        if not topic_dir.exists():
            return {"discovered": 0, "created": 0, "pending": 0}
        placeholders = {
            "Add human notes here.",
            "Add human notes here. Notes are never ingested automatically.",
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
        """Apply explicit review checkboxes from the generated maintenance page."""
        maintenance_pages = (
            self.obsidian_dir / "99 Maintenance.md",
            self.obsidian_dir / "99 维护中心.md",
        )
        page = next((item for item in maintenance_pages if item.is_file()), None)
        if page is None:
            return {"confirmed": 0, "needs_revision": 0, "rejected": 0, "errors": []}
        decisions: dict[str, list[str]] = {}
        for action, neuron_id in MEMORY_REVIEW_RE.findall(
            page.read_text(encoding="utf-8")
        ):
            decisions.setdefault(neuron_id, []).append(action)
        result: dict[str, object] = {
            "confirmed": 0,
            "needs_revision": 0,
            "rejected": 0,
            "errors": [],
        }
        errors = result["errors"]
        assert isinstance(errors, list)
        for neuron_id, actions in decisions.items():
            unique_actions = list(dict.fromkeys(actions))
            if len(unique_actions) != 1:
                errors.append(f"{neuron_id}: select exactly one review option")
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
                    "Human reviewer marked this proposed memory as needing revision in Obsidian.",
                )
                result["needs_revision"] = int(result["needs_revision"]) + 1
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
                domain="Obsidian human maintenance",
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
        topic_dir = self.obsidian_dir / "topics"
        topic_dir.mkdir(parents=True, exist_ok=True)
        review_sync = self.sync_obsidian_reviews()
        concepts = self.db.execute(
            """SELECT * FROM neurons WHERE layer=3
               AND status NOT IN ('rejected','archived') ORDER BY label"""
        ).fetchall()
        generated: list[Path] = []
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
            notes = self._preserved_user_notes(page)
            memory_ids = [row["id"] for row in atoms]
            sources = list(dict.fromkeys(row["source"] or "unknown" for row in atoms))
            related = self.db.execute(
                """SELECT n.layer,n.label FROM synapses s JOIN neurons n ON n.id=s.target_id
                   WHERE s.source_id=? AND n.layer BETWEEN 4 AND 5
                   ORDER BY n.layer,n.label""",
                (concept["id"],),
            ).fetchall()
            related_lines = "\n".join(
                f"- L{row['layer']} [[{row['label']}]]" for row in related
            ) or "- No upper-layer relationships"
            memory_lines = "\n".join(
                f"- [[vault/memories/{row['id']}|{row['id']}]] - {compact(row['label'], 80)}"
                + (
                    f" - [[vault/evidence/{row['evidence_id']}|evidence]]"
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
                "## Current understanding\n\n"
                f"{self._narrative(atoms)}\n\n"
                "## Linked Memories\n\n"
                f"{memory_lines}\n\n"
                "## Upper-layer relationships\n\n"
                f"{related_lines}\n\n"
                "## Sources\n\n"
                + "\n".join(f"- `{source}`" for source in sources)
                + "\n<!-- GENERATED:END -->\n\n"
                "<!-- USER-NOTES:START -->"
                + notes
                + "<!-- USER-NOTES:END -->\n",
                encoding="utf-8",
            )
            generated.append(page)

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
        inbox = self.maintenance_inbox()
        proposals = self.annotation_proposals()
        proposed_memories = self.db.execute(
            """SELECT n.*,e.source FROM neurons n
               LEFT JOIN evidence e ON e.id=n.evidence_id
               WHERE n.layer=1 AND n.status='proposed'
               ORDER BY n.created_at,n.id"""
        ).fetchall()
        proposed_memory_lines = "\n".join(
            f"- [[vault/memories/{row['id']}|{row['id']}]] - {compact(row['label'], 100)}"
            + (
                f" - [[vault/evidence/{row['evidence_id']}|evidence]]"
                if row["evidence_id"]
                else ""
            )
            + f"\n  - [ ] Confirm <!-- review:confirm:{row['id']} -->"
            + f"\n  - [ ] Needs revision <!-- review:revise:{row['id']} -->"
            + f"\n  - [ ] Incorrect / reject <!-- review:reject:{row['id']} -->"
            for row in proposed_memories
        ) or "- No proposed memories"
        issue_lines = "\n".join(
            f"- `{item['id']}` **{item['severity']}** {item['kind']}: {item['details']}"
            for item in inbox["issues"]
        ) or "- No open issues"
        relation_lines = "\n".join(
            f"- `{item['id']}` {item['source_id']} -> {item['relation']} -> {item['target_id']}"
            for item in inbox["relations"]
        ) or "- No pending relationships"
        proposal_lines = "\n".join(
            f"- `{item['id']}` [[{item['page_path']}]]: {compact(str(item['notes']).replace(chr(10), ' '), 100)}"
            for item in proposals
        ) or "- No pending human annotations"
        maintenance_page = self.obsidian_dir / "99 Maintenance.md"
        maintenance_page.write_text(
            "---\nview_type: maintenance-dashboard\ngenerated: true\ndo_not_ingest: true\n---\n\n"
            "# Memory Maintenance Center\n\n"
            "> This page only lists review candidates. Every write to core memory must be explicitly confirmed from the command line.\n\n"
            "## Proposed memories\n\n" + proposed_memory_lines + "\n\n"
            "Select exactly one option for each memory, then use the submit button below. Confirmation and rejection update the canonical status; needs revision keeps the candidate proposed and adds a maintenance issue. CLI fallback: `python3 neural_memory.py --root /ABSOLUTE/PATH/my-neural-memory sync-obsidian`.\n\n"
            "```neural-memory-submit\nSubmit selected review decisions\n```\n\n"
            "## Human annotation candidates\n\n" + proposal_lines + "\n\n"
            "## System issues\n\n" + issue_lines + "\n\n"
            "## Pending relationships\n\n" + relation_lines + "\n",
            encoding="utf-8",
        )

        archived = self.db.execute(
            """SELECT n.*,e.source FROM neurons n
               LEFT JOIN evidence e ON e.id=n.evidence_id
               WHERE n.layer=1 AND n.status='archived'
               ORDER BY n.created_at,n.id"""
        ).fetchall()
        archive_lines = "\n".join(
            f"- [[vault/memories/{row['id']}|{row['id']}]] - {compact(row['label'], 80)}"
            + (
                f" - [[vault/evidence/{row['evidence_id']}|evidence]]"
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
        home = self.obsidian_dir / "00 Home.md"
        links = "\n".join(f"- [[topics/{path.stem}]]" for path in generated) or "- No topic pages"
        home.write_text(
            "---\nview_type: compiled-memory\ngenerated: true\ndo_not_ingest: true\n---\n\n"
            "# Memory System Home\n\n"
            "## Topic navigation\n\n"
            f"{links}\n- [[98 Archive]]\n- [[99 Maintenance]]\n\n"
            "## System status\n\n"
            f"- L0 raw evidence: {stats['layers'].get(0, 0)}\n"
            + "\n".join(
                f"- L{layer} neurons: {count}"
                for layer, count in sorted(stats["layers"].items())
                if layer != 0
            )
            + f"\n- Synapses: {stats['synapses']}\n"
            "\n> This directory is a rebuildable reading view and must never be re-ingested as memory fragments.\n",
            encoding="utf-8",
        )
        return {
            "pages": len(generated) + 3,
            "root": str(self.obsidian_dir),
            "annotation_sync": sync_result,
            "review_sync": review_sync,
            "stale_topic_pages_removed": stale_removed,
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
            episode="Neural memory system design session",
            procedures=["Progressive memory maintenance workflow"],
            domain="AI memory and knowledge management",
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
    reencoder = commands.add_parser("reencode")
    reencoder.add_argument("config", type=Path)
    commands.add_parser("compile-obsidian")
    commands.add_parser("sync-obsidian")
    obsidian_review = commands.add_parser("obsidian-review")
    obsidian_review.add_argument("action", choices=["list", "show", "accept", "reject"])
    obsidian_review.add_argument("proposal_id", nargs="?")
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
                "formula": "governance * (0.45 vector + 0.45 BM25 + 0.10 lexical) + spread",
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
        elif args.command == "reencode":
            print(json.dumps(memory.reencode_all(args.config), ensure_ascii=False, indent=2))
        elif args.command == "compile-obsidian":
            print(json.dumps(memory.compile_obsidian(), ensure_ascii=False, indent=2))
        elif args.command == "sync-obsidian":
            review_sync = memory.sync_obsidian_reviews()
            annotation_sync = memory.sync_obsidian_notes()
            review_changes = sum(
                int(review_sync[key])
                for key in ("confirmed", "needs_revision", "rejected")
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
