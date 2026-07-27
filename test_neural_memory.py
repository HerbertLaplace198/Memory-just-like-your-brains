import tempfile
import unittest
import json
import zipfile
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from neural_memory import (
    HashEncoder,
    LocalHTTPEncoder,
    NeuralMemory,
    import_bundle,
    load_encoder_config,
    main,
    read_record,
    seed_demo,
    verify_bundle,
    write_record,
)
from mcp_server import MCPServer
from lifecycle_hook import LifecycleHook


class TinyEncoder:
    name = "test-tiny-v1"
    dimensions = 8

    def encode(self, text):
        vector = [0.0] * self.dimensions
        for value in text.encode("utf-8"):
            vector[value % self.dimensions] += 1.0
        return vector


class FailingEncoder:
    name = "test-failing-v1"
    dimensions = 8

    def encode(self, text):
        return [1.0] * (7 if "Token" in text else self.dimensions)


class MultiTopicEncoder:
    """Tiny semantic space used to verify multi-topic routing deterministically."""

    name = "test-multi-topic-v1"
    dimensions = 4
    gate_threshold = 0.06
    family_gate_threshold = 0.23
    topic_match_threshold = 0.50

    def encode(self, text):
        lowered = text.casefold()
        vector = [0.0] * self.dimensions
        if any(token in lowered for token in ("research", "method", "study")):
            vector[0] = 1.0
        if any(token in lowered for token in ("writing", "workflow", "paper")):
            vector[1] = 1.0
        if "investment" in lowered or "portfolio" in lowered:
            vector[2] = 1.0
        if not any(vector):
            vector[3] = 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def encode_many(self, texts):
        return [self.encode(text) for text in texts]


class NeuralMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = NeuralMemory(Path(self.temp.name))

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_three_stage_retrieval(self):
        seed_demo(self.memory)
        known, peak, activated = self.memory.probe(
            "How can I reduce token usage and recover raw tool output from a long task?"
        )
        self.assertTrue(known)
        self.assertGreater(peak, 0.06)
        self.assertTrue(any(item.layer == 2 for item in activated))
        cards = self.memory.recall(
            "How can I reduce token usage and recover raw tool output from a long task?", 3
        )
        self.assertTrue(cards)
        self.assertTrue(any("token" in card.summary.lower() or "tool output" in card.summary.lower() for card in cards))
        self.assertIn("---", self.memory.evidence_text(cards[0].evidence_id))

    def test_review_gate(self):
        neuron_id = self.memory.remember("This is an unconfirmed preference.", "test")
        self.assertIn(neuron_id, {row["id"] for row in self.memory.proposed()})
        self.assertTrue(self.memory.review(neuron_id, "confirmed"))
        self.assertNotIn(neuron_id, {row["id"] for row in self.memory.proposed()})

    def test_reject_prunes_orphan_l3_l4_nodes_and_synapses(self):
        rejected = self.memory.remember(
            "A temporary preference.",
            "test",
            topics=["Temporary Topic"],
            procedures=["Use the temporary workflow."],
        )
        retained = self.memory.remember(
            "A retained preference.",
            "test",
            topics=["Retained Topic"],
            procedures=["Use the retained workflow."],
            confirmed=True,
        )
        orphan_ids = {
            row["id"]
            for row in self.memory.db.execute(
                "SELECT id FROM neurons WHERE layer IN (3,4) AND label LIKE 'Temporary%' OR label LIKE 'Use the temporary%'"
            )
        }
        self.assertTrue(orphan_ids)

        self.assertTrue(self.memory.review(rejected, "rejected"))

        for neuron_id in orphan_ids:
            self.assertIsNone(
                self.memory.db.execute(
                    "SELECT id FROM neurons WHERE id=?", (neuron_id,)
                ).fetchone()
            )
            self.assertEqual(
                self.memory.db.execute(
                    "SELECT count(*) FROM synapses WHERE source_id=? OR target_id=?",
                    (neuron_id, neuron_id),
                ).fetchone()[0],
                0,
            )
        self.assertIsNotNone(
            self.memory.db.execute(
                "SELECT id FROM neurons WHERE layer=3 AND label='Retained Topic'"
            ).fetchone()
        )
        self.assertEqual(
            self.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (retained,)
            ).fetchone()[0],
            "confirmed",
        )

    def test_reject_moves_l1_and_evidence_to_hidden_archive_and_restores(self):
        neuron_id = self.memory.remember(
            "A reversible rejected preference.",
            "test",
            topics=["Reversible Review"],
            procedures=["Review a rejected record before restoration."],
        )
        evidence_id = self.memory.db.execute(
            "SELECT evidence_id FROM neurons WHERE id=?", (neuron_id,)
        ).fetchone()[0]

        self.assertTrue(self.memory.review(neuron_id, "rejected"))
        self.assertFalse((self.memory.memory_dir / f"{neuron_id}.md").exists())
        self.assertFalse((self.memory.evidence_dir / f"{evidence_id}.md").exists())
        self.assertTrue((self.memory.rejected_dir / f"{neuron_id}.md").exists())
        self.assertTrue((self.memory.rejected_dir / f"{evidence_id}.md").exists())
        self.assertIsNone(
            self.memory.db.execute(
                "SELECT id FROM neurons WHERE id=?", (neuron_id,)
            ).fetchone()
        )

        bundle = Path(self.temp.name).parent / (Path(self.temp.name).name + ".nmem")
        try:
            self.memory.export_bundle(bundle)
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn(f"vault/.rejected/{neuron_id}.md", archive.namelist())
                self.assertIn(f"vault/.rejected/{evidence_id}.md", archive.namelist())
        finally:
            if bundle.exists():
                bundle.unlink()

        self.assertTrue(self.memory.restore_rejected(neuron_id))
        self.assertTrue((self.memory.memory_dir / f"{neuron_id}.md").exists())
        self.assertTrue((self.memory.evidence_dir / f"{evidence_id}.md").exists())
        self.assertEqual(
            self.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (neuron_id,)
            ).fetchone()[0],
            "proposed",
        )

    def test_orphan_evidence_is_archived_and_reported(self):
        orphan_id = "ev_deadbeef00"
        orphan = self.memory.evidence_dir / f"{orphan_id}.md"
        orphan.write_text("---\nid: ev_deadbeef00\n---\n\nOrphaned evidence.\n", encoding="utf-8")
        self.assertIn(orphan_id, self.memory.health_report()["unreferenced_evidence_ids"])
        self.assertEqual(self.memory.archive_orphan_evidence(), [orphan_id])
        self.assertFalse(orphan.exists())
        self.assertTrue((self.memory.rejected_dir / orphan.name).exists())
        self.assertEqual(self.memory.health_report()["unreferenced_evidence_ids"], [])

    def test_l3_l4_labels_must_be_english(self):
        with self.assertRaisesRegex(ValueError, "topics must use English-only labels"):
            self.memory.remember(
                "A preference with a non-English topic.",
                "test",
                topics=["非英语主题"],
            )
        with self.assertRaisesRegex(ValueError, "procedures must use English-only labels"):
            self.memory.remember(
                "A preference with a non-English procedure.",
                "test",
                topics=["Thesis"],
                procedures=["\u4ec5\u6309\u6307\u5b9a\u8303\u56f4\u4fee\u6539"],
            )

    def test_rebuild_skips_upper_nodes_for_rejected_records(self):
        neuron_id = self.memory.remember(
            "A rejected record should not regenerate topics.",
            "test",
            topics=["Rejected Topic"],
            procedures=["Do not regenerate this procedure."],
        )
        self.memory.review(neuron_id, "rejected")
        self.memory.rebuild_index()
        self.assertIsNone(
            self.memory.db.execute(
                "SELECT id FROM neurons WHERE layer=3 AND label='Rejected Topic'"
            ).fetchone()
        )
        self.assertIsNone(
            self.memory.db.execute(
                "SELECT id FROM neurons WHERE layer=4 AND label='Do not regenerate this procedure.'"
            ).fetchone()
        )

    def test_hebbian_reinforcement(self):
        seed_demo(self.memory)
        cards = self.memory.recall("memory system token", 3)
        self.memory.reinforce([card.id for card in cards])
        count = self.memory.db.execute(
            "SELECT count(*) FROM synapses WHERE relation='co_recalled'"
        ).fetchone()[0]
        self.assertGreater(count, 0)

    def test_retrieval_reconsolidates_atomic_traces(self):
        neuron_id = self.memory.remember(
            "Repeated retrieval should stabilize this memory trace.",
            "test",
            topics=["Memory"],
            confirmed=True,
        )
        before = self.memory.db.execute(
            "SELECT stability,reactivation_count,last_reactivated FROM neurons WHERE id=?",
            (neuron_id,),
        ).fetchone()

        cards = self.memory.recall(
            "stabilize this memory trace",
            1,
            reconsolidate=True,
        )

        after = self.memory.db.execute(
            "SELECT stability,reactivation_count,last_reactivated FROM neurons WHERE id=?",
            (neuron_id,),
        ).fetchone()
        self.assertEqual([card.id for card in cards], [neuron_id])
        self.assertGreater(after["stability"], before["stability"])
        self.assertEqual(after["reactivation_count"], before["reactivation_count"] + 1)
        self.assertIsNotNone(after["last_reactivated"])

    def test_similar_confirmed_fragments_form_proposed_emergent_l3(self):
        fragments = [
            (
                "A risk model connects uncertainty evidence and long term decisions "
                "during thesis research.",
                "Thesis",
            ),
            (
                "A risk model connects uncertainty evidence and long term decisions "
                "during portfolio planning.",
                "Investment",
            ),
            (
                "Risk models connect uncertainty evidence and long term decisions "
                "when evaluating projects.",
                "Project Evaluation",
            ),
        ]
        memory_ids = [
            self.memory.remember(text, "test", topics=[topic], confirmed=True)
            for text, topic in fragments
        ]

        concept = self.memory.db.execute(
            """SELECT id,status,stability FROM neurons
               WHERE layer=3 AND label LIKE 'Emergent Concept %'"""
        ).fetchone()
        self.assertIsNotNone(concept)
        self.assertEqual(concept["status"], "proposed")
        self.assertGreater(concept["stability"], 0.5)
        linked = {
            row["source_id"]
            for row in self.memory.db.execute(
                """SELECT source_id FROM synapses
                   WHERE target_id=? AND relation='emergent_member_of'""",
                (concept["id"],),
            )
        }
        self.assertEqual(linked, set(memory_ids))

        self.memory.compile_obsidian()
        maintenance = self.memory.obsidian_dir / "99 维护中心.md"
        text = maintenance.read_text(encoding="utf-8")
        self.assertIn("## L3 概念审核", text)
        self.assertIn(concept["id"], text)
        for memory_id in memory_ids:
            self.assertIn(memory_id, text)
        text = text.replace(
            f"  - [ ] 确认 L3 概念 <!-- concept-review:confirm:{concept['id']} -->",
            f"  - [x] 确认 L3 概念 <!-- concept-review:confirm:{concept['id']} -->",
        )
        maintenance.write_text(text, encoding="utf-8")

        review = self.memory.sync_obsidian_reviews()
        self.assertEqual(review["concepts_confirmed"], 1)
        self.assertEqual(
            self.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (concept["id"],)
            ).fetchone()[0],
            "confirmed",
        )
        self.assertTrue(
            (self.memory.semantic_review_dir / f"{concept['id']}.md").is_file()
        )

        self.memory.rebuild_index()
        rebuilt = self.memory.db.execute(
            "SELECT status FROM neurons WHERE id=?", (concept["id"],)
        ).fetchone()
        self.assertIsNotNone(rebuilt)
        self.assertEqual(rebuilt["status"], "confirmed")

    def test_consolidation_decays_links_without_deleting_canonical_memory(self):
        left = self.memory.remember(
            "Long-term evidence supports calibrated risk decisions.",
            "test",
            confirmed=True,
        )
        right = self.memory.remember(
            "Calibrated risk decisions depend on long-term evidence.",
            "test",
            confirmed=True,
        )
        before = self.memory.db.execute(
            """SELECT weight FROM synapses
               WHERE source_id=? AND target_id=? AND relation='association'""",
            (left, right),
        ).fetchone()[0]

        future = datetime.now(timezone.utc) + timedelta(days=365)
        result = self.memory.consolidate(reference=future)
        after = self.memory.db.execute(
            """SELECT weight FROM synapses
               WHERE source_id=? AND target_id=? AND relation='association'""",
            (left, right),
        ).fetchone()[0]

        self.assertGreater(result["decayed_synapses"], 0)
        self.assertLess(after, before)
        self.assertTrue((self.memory.memory_dir / f"{left}.md").is_file())
        self.assertTrue((self.memory.memory_dir / f"{right}.md").is_file())
        repeated = self.memory.consolidate(reference=future)
        repeated_weight = self.memory.db.execute(
            """SELECT weight FROM synapses
               WHERE source_id=? AND target_id=? AND relation='association'""",
            (left, right),
        ).fetchone()[0]
        self.assertEqual(repeated["decayed_synapses"], 0)
        self.assertEqual(repeated_weight, after)

    def test_token_budget_benchmark(self):
        seed_demo(self.memory)
        result = self.memory.benchmark("How can I save tokens?", 2)
        self.assertEqual(result["recalled_cards"], 2)
        self.assertLess(result["estimated_recall_tokens"], result["estimated_full_tokens"])

    def test_parse_mdkb_list(self):
        output = (
            "[memory-management] Memory management and architecture rules (topic) "
            "#memory #mdkb #global - 8 accesses\n"
        )
        entries = self.memory.parse_mdkb_list(output)
        self.assertEqual(entries[0]["id"], "memory-management")
        self.assertEqual(entries[0]["tags"], ["memory", "mdkb", "global"])

    def test_archived_neurons_do_not_activate(self):
        active = self.memory.remember(
            "The current plan uses the new rules.", "test", topics=["Plan"], confirmed=True
        )
        archived = self.memory.remember(
            "The current plan uses the old rules.", "test", topics=["Plan"], confirmed=True
        )
        self.memory.review(archived, "archived")
        activated_ids = {item.id for item in self.memory.activate("current plan old rules")}
        self.assertIn(active, activated_ids)
        self.assertNotIn(archived, activated_ids)

    def test_full_mdkb_copy_reads_local_evidence(self):
        neuron_id = self.memory.remember(
            "Complete local copy content.", "mdkb:local-copy-test", confirmed=True
        )
        evidence_id = self.memory.db.execute(
            "SELECT evidence_id FROM neurons WHERE id=?", (neuron_id,)
        ).fetchone()[0]
        self.memory._set_meta("mdkb_copy_mode", "full")
        self.memory.db.commit()
        self.assertIn("Complete local copy content", self.memory.evidence_text(evidence_id))

    def test_portable_bundle_round_trip(self):
        seed_demo(self.memory)
        bundle = Path(self.temp.name).parent / (Path(self.temp.name).name + ".nmem")
        restored_root = Path(self.temp.name).parent / (Path(self.temp.name).name + "-restored")
        try:
            result = self.memory.export_bundle(bundle)
            self.assertGreater(result["bytes"], 0)
            restored = import_bundle(bundle, restored_root)
            self.assertEqual(restored["stats"]["layers"]["1"], 5)
            copy = NeuralMemory(restored_root)
            try:
                known, _, _ = copy.probe(
                    "How can I reduce token usage and recover raw tool output from a long task?"
                )
                self.assertTrue(known)
            finally:
                copy.close()
        finally:
            if bundle.exists():
                bundle.unlink()
            if restored_root.exists():
                for path in sorted(restored_root.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                restored_root.rmdir()

    def test_l0_to_l6_and_markdown_rebuild(self):
        seed_demo(self.memory)
        memory_path = next(self.memory.memory_dir.glob("*.md"))
        metadata, body = read_record(memory_path)
        metadata["created_at"] = "2020-01-02T03:04:05+00:00"
        write_record(memory_path, metadata, body)
        before = self.memory.stats()
        self.assertEqual(set(before["layers"]), {0, 1, 2, 3, 4, 5, 6})
        self.assertEqual(before["canonical_records"], 5)
        self.memory.db.execute("DELETE FROM synapses")
        self.memory.db.execute("DELETE FROM neurons")
        self.memory.db.execute("DELETE FROM evidence")
        self.memory.db.commit()
        result = self.memory.rebuild_index()
        self.assertEqual(result["records"], 5)
        self.assertEqual(result["stats"]["layers"], before["layers"])
        self.assertEqual(result["stats"]["synapses"], before["synapses"])
        self.assertEqual(
            self.memory.db.execute(
                "SELECT created_at FROM neurons WHERE id=?", (metadata["id"],)
            ).fetchone()[0],
            metadata["created_at"],
        )

    def test_obsidian_compiler_preserves_user_notes(self):
        seed_demo(self.memory)
        result = self.memory.compile_obsidian()
        self.assertGreater(result["pages"], 1)
        page = next((self.memory.obsidian_dir / "主题").glob("*.md"))
        text = page.read_text(encoding="utf-8")
        self.assertIn("do_not_ingest: true", text)
        text = text.replace(
            "在这里添加人工批注。批注不会自动进入记忆核心。",
            "This human annotation must be preserved.",
        )
        page.write_text(text, encoding="utf-8")
        self.memory.compile_obsidian()
        self.assertIn("This human annotation must be preserved.", page.read_text(encoding="utf-8"))

    def test_obsidian_compiler_generates_upper_layer_notes(self):
        self.memory.remember(
            "Review the evidence before publishing a memory-system release.",
            "test",
            topics=["Memory"],
            procedures=["Review the evidence before publishing."],
            schemas=["Deliberate maintainer"],
            confirmed=True,
        )
        self.memory.remember(
            "Legacy labels must not create non-English generated relation pages.",
            "test",
            topics=["Memory"],
            schemas=["Legacy Model Label"],
            confirmed=True,
        )

        result = self.memory.compile_obsidian()
        topic = (self.memory.obsidian_dir / "主题" / "Memory System.md").read_text(
            encoding="utf-8"
        )
        procedure = self.memory.obsidian_dir / "relations" / "procedures" / "Review the evidence before publishing.md"
        persona = self.memory.obsidian_dir / "relations" / "personas" / "Deliberate maintainer.md"

        self.assertGreater(result["pages"], 3)
        self.assertIn(
            "[[relations/procedures/Review the evidence before publishing|Review the evidence before publishing.]]",
            topic,
        )
        self.assertIn(
            "[[relations/personas/Deliberate maintainer|Deliberate maintainer]]",
            topic,
        )
        self.assertTrue(procedure.is_file())
        self.assertTrue(persona.is_file())
        self.assertIn("[[主题/Memory System|Memory System]]", procedure.read_text(encoding="utf-8"))
        self.assertFalse(
            (self.memory.obsidian_dir / "relations" / "personas" / "旧的模型标签.md").exists()
        )
        self.assertNotIn("旧的模型标签", topic)

    def test_topic_aliases_are_canonical_and_duplicate_free(self):
        neuron_id = self.memory.remember(
            "Asset allocation follows long-term diversification.",
            "test",
            topics=["Investment", "investment", "index", "portfolio"],
            confirmed=True,
        )
        labels = {
            row["label"]
            for row in self.memory.db.execute(
                """SELECT n.label FROM synapses s JOIN neurons n ON n.id=s.target_id
                   WHERE s.source_id=? AND s.relation='member_of' AND n.layer=3""",
                (neuron_id,),
            )
        }
        self.assertEqual(labels, {"Investment", "Asset Allocation"})
        count = self.memory.db.execute(
            "SELECT count(*) FROM neurons WHERE layer=3 AND lower(label)='investment'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_new_memory_reuses_multiple_relevant_existing_topics(self):
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), MultiTopicEncoder())
            try:
                memory.remember(
                    "Research methods guide study design.",
                    "test",
                    topics=["Research Methods"],
                    confirmed=True,
                )
                memory.remember(
                    "The writing workflow prepares the paper.",
                    "test",
                    topics=["Writing Workflow"],
                    confirmed=True,
                )
                linked = memory.remember(
                    "Research methods should guide the writing workflow for a paper.",
                    "test",
                )

                labels = {
                    row["label"]
                    for row in memory.db.execute(
                        """SELECT n.label FROM synapses s JOIN neurons n ON n.id=s.target_id
                           WHERE s.source_id=? AND s.relation='member_of' AND n.layer=3""",
                        (linked,),
                    )
                }
                self.assertEqual(labels, {"Research Methods", "Writing Workflow"})
                self.assertEqual(
                    memory.db.execute(
                        "SELECT count(*) FROM neurons WHERE layer=3"
                    ).fetchone()[0],
                    2,
                )

                mixed = memory.remember(
                    "Research methods also inform investment portfolio decisions.",
                    "test",
                    topics=["Research Methods", "Investment"],
                )
                mixed_labels = {
                    row["label"]
                    for row in memory.db.execute(
                        """SELECT n.label FROM synapses s JOIN neurons n ON n.id=s.target_id
                           WHERE s.source_id=? AND s.relation='member_of' AND n.layer=3""",
                        (mixed,),
                    )
                }
                self.assertEqual(mixed_labels, {"Research Methods", "Investment"})
                self.assertEqual(
                    memory.db.execute(
                        "SELECT count(*) FROM neurons WHERE layer=3"
                    ).fetchone()[0],
                    3,
                )
            finally:
                memory.close()

    def test_semantically_equivalent_topic_hint_reuses_existing_l3(self):
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), MultiTopicEncoder())
            try:
                memory.remember(
                    "Research methods guide study design.",
                    "test",
                    topics=["Research Methods"],
                    confirmed=True,
                )
                linked = memory.remember(
                    "Study method findings belong to the same research area.",
                    "test",
                    topics=["Study Method"],
                )
                labels = {
                    row["label"]
                    for row in memory.db.execute(
                        """SELECT n.label FROM synapses s JOIN neurons n ON n.id=s.target_id
                           WHERE s.source_id=? AND s.relation='member_of' AND n.layer=3""",
                        (linked,),
                    )
                }
                self.assertEqual(labels, {"Research Methods"})
                self.assertIsNone(
                    memory.db.execute(
                        "SELECT id FROM neurons WHERE layer=3 AND label='Study Method'"
                    ).fetchone()
                )
            finally:
                memory.close()

    def test_confirmed_memory_upgrades_but_never_downgrades_topic_status(self):
        proposed = self.memory.remember(
            "Proposed memory system rule.", "test", topics=["memory"]
        )
        topic = self.memory.db.execute(
            """SELECT n.id,n.status FROM synapses s JOIN neurons n ON n.id=s.target_id
               WHERE s.source_id=? AND n.layer=3""",
            (proposed,),
        ).fetchone()
        self.assertEqual(topic["status"], "proposed")
        confirmed = self.memory.remember(
            "Confirmed memory system rule.",
            "test",
            topics=["Neural Memory"],
            confirmed=True,
        )
        self.assertEqual(
            self.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (topic["id"],)
            ).fetchone()[0],
            "confirmed",
        )
        self.memory.review(confirmed, "archived")
        self.assertEqual(
            self.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (topic["id"],)
            ).fetchone()[0],
            "confirmed",
        )

    def test_topic_page_includes_episode_routed_l1_members_and_real_links(self):
        investment = self.memory.remember(
            "The investment plan uses diversified allocation.",
            "test",
            topics=["investment"],
            confirmed=True,
            episode="Investment review session",
        )
        thesis = self.memory.remember(
            "Thesis progress has entered result verification.", "test", topics=["thesis"], confirmed=True
        )
        self.memory.compile_obsidian()
        page = self.memory.obsidian_dir / "主题" / "Investment.md"
        text = page.read_text(encoding="utf-8")
        self.assertIn(f"[[vault/memories/{investment}|{investment}]]", text)
        self.assertIn("[[vault/evidence/", text)
        self.assertNotIn(thesis, text)

    def test_obsidian_hides_l6_and_removes_stale_generated_pages(self):
        self.memory.remember(
            "The memory system uses human review.",
            "test",
            topics=["memory"],
            domain="private-routing-domain",
            confirmed=True,
        )
        topic_dir = self.memory.obsidian_dir / "主题"
        topic_dir.mkdir(parents=True, exist_ok=True)
        stale = topic_dir / "Old Generated Topic.md"
        stale.write_text(
            "---\nview_type: compiled-memory\ngenerated: true\n---\n",
            encoding="utf-8",
        )
        result = self.memory.compile_obsidian()
        page = topic_dir / "Memory System.md"
        text = page.read_text(encoding="utf-8")
        self.assertNotIn("L6", text)
        self.assertNotIn("private-routing-domain", text)
        self.assertFalse(stale.exists())
        self.assertEqual(result["stale_topic_pages_removed"], 1)

    def test_archive_page_links_archived_canonical_records(self):
        neuron_id = self.memory.remember(
            "This is an old rule used to verify archival behavior.",
            "test",
            topics=["workflow"],
            confirmed=True,
        )
        self.memory.review(neuron_id, "archived")
        self.memory.compile_obsidian()
        archive = (self.memory.obsidian_dir / "98 Archive.md").read_text(encoding="utf-8")
        self.assertIn(f"[[vault/memories/{neuron_id}|{neuron_id}]]", archive)
        self.assertIn("[[vault/evidence/", archive)

    def test_maintenance_page_links_proposed_memories_for_review(self):
        neuron_id = self.memory.remember(
            "This candidate must remain proposed until a human reviews it.",
            "test",
            topics=["Memory Governance"],
        )
        self.memory.compile_obsidian()
        maintenance = (self.memory.obsidian_dir / "99 维护中心.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## 待审核记忆", maintenance)
        self.assertIn(f"[[vault/memories/{neuron_id}|{neuron_id}]]", maintenance)
        self.assertIn("[[vault/evidence/", maintenance)
        self.assertIn(f"review:confirm:{neuron_id}", maintenance)
        self.assertIn(f"review:revise:{neuron_id}", maintenance)
        self.assertIn(f"review:reject:{neuron_id}", maintenance)

    def test_obsidian_review_checkboxes_apply_three_human_decisions(self):
        confirmed_id = self.memory.remember("Confirm this candidate.", "test", topics=["Review"])
        revise_id = self.memory.remember("Revise this candidate.", "test", topics=["Review"])
        rejected_id = self.memory.remember("Reject this candidate.", "test", topics=["Review"])
        rejected_evidence_id = self.memory.db.execute(
            "SELECT evidence_id FROM neurons WHERE id=?", (rejected_id,)
        ).fetchone()[0]
        self.memory.compile_obsidian()
        page = self.memory.obsidian_dir / "99 维护中心.md"
        text = page.read_text(encoding="utf-8")
        for action, neuron_id in (
            ("confirm", confirmed_id),
            ("revise", revise_id),
            ("reject", rejected_id),
        ):
            label = "确认" if action == "confirm" else "需要修改" if action == "revise" else "错误／拒绝"
            text = text.replace(
                f"  - [ ] {label} <!-- review:{action}:{neuron_id} -->",
                f"  - [x] {label} <!-- review:{action}:{neuron_id} -->",
            )
        page.write_text(text, encoding="utf-8")
        result = self.memory.sync_obsidian_reviews()
        self.assertEqual(result["confirmed"], 1)
        self.assertEqual(result["needs_revision"], 1)
        self.assertEqual(result["rejected"], 1)
        statuses = {
            row["id"]: row["status"]
            for row in self.memory.db.execute(
                "SELECT id,status FROM neurons WHERE id IN (?,?,?)",
                (confirmed_id, revise_id, rejected_id),
            )
        }
        self.assertEqual(statuses[confirmed_id], "confirmed")
        self.assertEqual(statuses[revise_id], "proposed")
        self.assertNotIn(rejected_id, statuses)
        self.assertTrue((self.memory.rejected_dir / f"{rejected_id}.md").exists())
        self.assertTrue((self.memory.rejected_dir / f"{rejected_evidence_id}.md").exists())
        self.assertTrue(any(
            issue["neuron_id"] == revise_id and issue["kind"] == "needs_revision"
            for issue in self.memory.maintenance_inbox()["issues"]
        ))

    def test_sync_obsidian_auto_compiles_confirmed_episode_memory(self):
        neuron_id = self.memory.remember(
            "Confirm and link this episodic candidate.",
            "test",
            topics=["Review"],
            episode="Review submission event",
        )
        self.memory.compile_obsidian()
        page = self.memory.obsidian_dir / "99 维护中心.md"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                f"  - [ ] 确认 <!-- review:confirm:{neuron_id} -->",
                f"  - [x] 确认 <!-- review:confirm:{neuron_id} -->",
            ),
            encoding="utf-8",
        )

        with redirect_stdout(StringIO()):
            self.assertEqual(main(["--root", self.temp.name, "sync-obsidian"]), 0)

        topic = (self.memory.obsidian_dir / "主题" / "Review.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"[[vault/memories/{neuron_id}|{neuron_id}]]", topic)
        self.assertIn("[[vault/evidence/", topic)

    def test_continuation_query_adds_parent_topic_without_cross_topic_noise(self):
        thesis = self.memory.remember(
            "The thesis entry stores project file pointers.", "test", topics=["thesis"], confirmed=True
        )
        progress = self.memory.remember(
            "Thesis progress has entered final verification.",
            "test",
            topics=["progress"],
            confirmed=True,
        )
        investment = self.memory.remember(
            "The investment plan remains unchanged this month.", "test", topics=["investment"], confirmed=True
        )
        continued = {card.id for card in self.memory.recall("Continue Thesis Progress", 5)}
        self.assertEqual(continued, {thesis, progress})
        self.assertNotIn(investment, continued)

    def test_case_only_topic_filename_is_normalized(self):
        self.memory.remember(
            "Investment topic filenames should use canonical capitalization.",
            "test",
            topics=["investment"],
            confirmed=True,
        )
        topic_dir = self.memory.obsidian_dir / "主题"
        topic_dir.mkdir(parents=True, exist_ok=True)
        lower = topic_dir / "investment.md"
        lower.write_text(
            "---\nview_type: compiled-memory\ngenerated: true\n---\n",
            encoding="utf-8",
        )
        self.memory.compile_obsidian()
        names = {path.name for path in topic_dir.glob("*.md")}
        self.assertIn("Investment.md", names)
        self.assertNotIn("investment.md", names)

    def test_evaluation_report(self):
        seed_demo(self.memory)
        cases = Path(__file__).with_name("evaluation.json")
        result = self.memory.evaluate(cases)
        self.assertEqual(result["cases"], 8)
        self.assertGreaterEqual(result["gate_accuracy"], 0.875)
        self.assertGreaterEqual(result["top3_recall"], 0.8)

    def test_hybrid_retrieval_has_auditable_components(self):
        self.memory.remember(
            "Saturn has prominent rings and is a gas giant in the Solar System.",
            "test",
            topics=["Astronomy"],
            confirmed=True,
        )
        self.memory.remember(
            "Memory review should inspect sources, confidence, and conflicts.",
            "test",
            topics=["Memory Governance"],
            confirmed=True,
        )
        cards = self.memory.recall("Saturn's rings", 2)
        self.assertIn("Saturn", cards[0].summary)
        self.assertGreater(cards[0].bm25_score, 0.0)
        self.assertGreater(cards[0].direct_activation, 0.0)

    def test_encoder_is_replaceable(self):
        self.assertEqual(self.memory.stats()["encoder"]["name"], HashEncoder.name)

    def test_expiration_enters_maintenance_inbox_without_auto_mutation(self):
        neuron_id = self.memory.remember(
            "This fact has expired but still requires human verification.",
            "test",
            confirmed=True,
            expires_at="2020-01-01",
        )
        inbox = self.memory.maintenance_inbox()
        self.assertTrue(any(
            issue["neuron_id"] == neuron_id and issue["kind"] == "expired"
            for issue in inbox["issues"]
        ))
        status = self.memory.db.execute(
            "SELECT status FROM neurons WHERE id=?", (neuron_id,)
        ).fetchone()[0]
        self.assertEqual(status, "confirmed")

    def test_supersedes_requires_review_then_archives_old_memory(self):
        old_id = self.memory.remember("The user prefers the old workflow.", "test", confirmed=True)
        new_id = self.memory.remember(
            "The user now prefers the new workflow.",
            "test",
            confirmed=True,
            supersedes=[old_id],
        )
        relation = self.memory.db.execute(
            "SELECT * FROM memory_relations WHERE source_id=? AND target_id=?",
            (new_id, old_id),
        ).fetchone()
        self.assertEqual(relation["status"], "pending")
        self.assertEqual(
            self.memory.db.execute("SELECT status FROM neurons WHERE id=?", (old_id,)).fetchone()[0],
            "confirmed",
        )
        self.assertTrue(self.memory.review_relation(relation["id"], "confirm"))
        self.assertEqual(
            self.memory.db.execute("SELECT status FROM neurons WHERE id=?", (old_id,)).fetchone()[0],
            "archived",
        )

    def test_possible_conflict_is_only_a_review_issue(self):
        self.memory.remember(
            "The user likes using Obsidian for durable memory.", "test", confirmed=True
        )
        new_id = self.memory.remember(
            "The user does not like using Obsidian for durable memory.", "test", confirmed=True
        )
        inbox = self.memory.maintenance_inbox()
        self.assertTrue(any(
            issue["neuron_id"] == new_id and issue["kind"] == "possible_conflict"
            for issue in inbox["issues"]
        ))
        self.assertEqual(
            self.memory.db.execute("SELECT status FROM neurons WHERE id=?", (new_id,)).fetchone()[0],
            "confirmed",
        )

    def test_obsidian_notes_require_explicit_acceptance(self):
        seed_demo(self.memory)
        self.memory.compile_obsidian()
        before = self.memory.db.execute(
            "SELECT count(*) FROM neurons WHERE layer=1"
        ).fetchone()[0]
        page = next((self.memory.obsidian_dir / "主题").glob("*.md"))
        text = page.read_text(encoding="utf-8").replace(
            "在这里添加人工批注。批注不会自动进入记忆核心。",
            "Human note: review the maintenance inbox weekly.",
        )
        page.write_text(text, encoding="utf-8")
        sync = self.memory.sync_obsidian_notes()
        self.assertEqual(sync["created"], 1)
        self.assertEqual(
            self.memory.db.execute("SELECT count(*) FROM neurons WHERE layer=1").fetchone()[0],
            before,
        )
        proposal = self.memory.annotation_proposals()[0]
        neuron_id = self.memory.review_annotation(proposal["id"], "accept")
        self.assertTrue(str(neuron_id).startswith("l1_"))
        self.assertEqual(
            self.memory.db.execute("SELECT count(*) FROM neurons WHERE layer=1").fetchone()[0],
            before + 1,
        )
        source = self.memory.db.execute(
            """SELECT e.source FROM neurons n JOIN evidence e ON e.id=n.evidence_id
               WHERE n.id=?""",
            (neuron_id,),
        ).fetchone()[0]
        self.assertTrue(source.startswith("obsidian-review:"))
        self.assertEqual(self.memory.sync_obsidian_notes()["created"], 0)

    def test_mcp_staged_recall_and_proposal_gate(self):
        mcp_root = Path(self.temp.name) / "mcp"
        server = MCPServer(mcp_root)
        try:
            seed_demo(server.memory)
            initialized = server.handle({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            })
            self.assertEqual(initialized["result"]["serverInfo"]["name"], "neural-memory")
            self.assertEqual(initialized["result"]["serverInfo"]["version"], "1.5.1")
            awareness = server.call_tool(
                "memory_awareness", {"query": "Why does the memory system use several layers?"}
            )
            self.assertTrue(awareness["known"])
            self.assertEqual(awareness["next"], "memory_recall")
            learned = server.call_tool(
                "memory_recall",
                {
                    "query": "Why does the memory system use several layers?",
                    "learn": True,
                },
            )
            self.assertTrue(learned["reconsolidated"])
            self.assertGreater(
                server.memory.db.execute(
                    "SELECT reactivation_count FROM neurons WHERE id=?",
                    (learned["cards"][0]["id"],),
                ).fetchone()[0],
                0,
            )
            unknown = server.call_tool(
                "memory_recall", {"query": "How long should French onion soup bake?"}
            )
            self.assertFalse(unknown["known"])
            self.assertEqual(unknown["cards"], [])
            proposed = server.call_tool(
                "memory_propose",
                {"text": "The agent found a new constraint that requires user review.", "topics": ["Memory Governance"]},
            )
            row = server.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (proposed["id"],)
            ).fetchone()
            self.assertEqual(proposed["status"], "proposed")
            self.assertEqual(row["status"], "proposed")
            self.assertTrue(proposed["obsidian_view_refreshed"])
            topic_page = server.memory.obsidian_dir / "主题" / "Memory Governance.md"
            self.assertIn(proposed["id"], topic_page.read_text(encoding="utf-8"))
        finally:
            server.close()

    def test_lifecycle_hook_is_compact_explicit_and_idempotent(self):
        hook_root = Path(self.temp.name) / "hook"
        hook = LifecycleHook(hook_root)
        try:
            seed_demo(hook.memory)
            start = hook.start({"task": "How can I reduce token use and recall prior tasks?"})
            self.assertTrue(start["known"])
            self.assertLessEqual(len(start["card_ids"]), 3)
            self.assertNotIn("---\n", start["context"])
            self.assertNotIn("source:", start["context"])

            empty = hook.finish({
                "event_id": "task-empty",
                "transcript": "This entire transcript must never be ingested automatically.",
            })
            self.assertEqual(empty["created_proposals"], [])
            self.assertTrue(empty["ignored_transcript"])

            payload = {
                "event_id": "task-1",
                "memory_candidates": [{
                    "text": "At task completion, submit only durable facts useful in future work.",
                    "topics": ["Memory Governance"],
                }],
            }
            first = hook.finish(payload)
            second = hook.finish(payload)
            self.assertEqual(len(first["created_proposals"]), 1)
            self.assertEqual(second["created_proposals"], [])
            self.assertEqual(second["duplicate_proposals"], first["created_proposals"])
            status = hook.memory.db.execute(
                "SELECT status FROM neurons WHERE id=?", (first["created_proposals"][0],)
            ).fetchone()[0]
            self.assertEqual(status, "proposed")
            self.assertTrue(first["obsidian_view_refreshed"])
            topic_page = hook.memory.obsidian_dir / "主题" / "Memory Governance.md"
            self.assertIn(
                first["created_proposals"][0],
                topic_page.read_text(encoding="utf-8"),
            )
        finally:
            hook.close()

    def test_health_report_uses_wal_and_integrity_check(self):
        seed_demo(self.memory)
        report = self.memory.health_report()
        self.assertTrue(report["healthy"])
        self.assertEqual(report["sqlite_integrity"], ["ok"])
        self.assertEqual(report["journal_mode"].lower(), "wal")

    def test_atomic_backup_verification_and_rotation(self):
        seed_demo(self.memory)
        backup_dir = Path(self.temp.name) / "backups"
        first = self.memory.create_backup(backup_dir, keep=2)
        self.memory.create_backup(backup_dir, keep=2)
        third = self.memory.create_backup(backup_dir, keep=2)
        self.assertTrue(verify_bundle(Path(first["bundle"]))["valid"] if Path(first["bundle"]).exists() else True)
        retained = list(backup_dir.glob("neural-memory-*.nmem"))
        self.assertEqual(len(retained), 2)
        self.assertEqual(third["retained"], 2)
        self.assertEqual(len(third["removed_old_backups"]), 1)
        self.assertTrue(verify_bundle(retained[0])["valid"])

        corrupt = backup_dir / "corrupt.nmem"
        with zipfile.ZipFile(retained[0], "r") as source:
            manifest = json.loads(source.read("manifest.json"))
            target_name = next(iter(manifest["files"]))
            with zipfile.ZipFile(corrupt, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for name in source.namelist():
                    payload = source.read(name)
                    if name == target_name:
                        payload += b"tampered"
                    output.writestr(name, payload)
        with self.assertRaises(RuntimeError):
            verify_bundle(corrupt)

    def test_cooperating_instances_serialize_writes(self):
        shared_root = Path(self.temp.name) / "concurrent"

        def write_one(index):
            memory = NeuralMemory(shared_root)
            try:
                return memory.remember(
                    f"Concurrent memory write {index}",
                    "concurrency-test",
                    topics=["Concurrency Test"],
                    confirmed=True,
                )
            finally:
                memory.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(write_one, range(8)))
        check = NeuralMemory(shared_root)
        try:
            count = check.db.execute(
                "SELECT count(*) FROM neurons WHERE layer=1"
            ).fetchone()[0]
            self.assertEqual(count, 8)
            self.assertEqual(len(set(ids)), 8)
            self.assertTrue(check.health_report()["healthy"])
        finally:
            check.close()

    def test_encoder_migration_updates_all_vectors_atomically(self):
        migration_root = Path(self.temp.name) / "migration"
        original = NeuralMemory(migration_root)
        seed_demo(original)
        original.close()
        migrated = NeuralMemory(
            migration_root, TinyEncoder(), allow_encoder_mismatch=True
        )
        try:
            result = migrated.reencode_all()
            self.assertEqual(result["to"]["name"], TinyEncoder.name)
            dimensions = {
                len(json.loads(row["vector"]))
                for row in migrated.db.execute("SELECT vector FROM neurons")
            }
            self.assertEqual(dimensions, {TinyEncoder.dimensions})
            self.assertEqual(migrated.stats()["encoder"]["name"], TinyEncoder.name)
        finally:
            migrated.close()

    def test_failed_encoder_migration_rolls_back(self):
        migration_root = Path(self.temp.name) / "failed-migration"
        original = NeuralMemory(migration_root)
        seed_demo(original)
        original.close()
        failing = NeuralMemory(
            migration_root, FailingEncoder(), allow_encoder_mismatch=True
        )
        try:
            with self.assertRaises(ValueError):
                failing.reencode_all()
            stored_name = failing._get_meta("encoder_name")
            dimensions = {
                len(json.loads(row["vector"]))
                for row in failing.db.execute("SELECT vector FROM neurons")
            }
            self.assertEqual(stored_name, HashEncoder.name)
            self.assertEqual(dimensions, {1024})
        finally:
            failing.close()

    def test_local_encoder_rejects_non_loopback_endpoint(self):
        with self.assertRaises(ValueError):
            LocalHTTPEncoder(
                "openai-compatible",
                "https://api.example.com/v1/embeddings",
                "remote-model",
                768,
            )

    def test_v14_encoder_config_defaults_new_family_bonus(self):
        config = Path(self.temp.name) / "legacy-encoder.json"
        config.write_text(
            json.dumps({
                "provider": "hash",
                "dimensions": 1024,
                "gate_threshold": 0.48,
            }),
            encoding="utf-8",
        )
        encoder = load_encoder_config(config)
        self.assertEqual(encoder.family_size_bonus_cap, 0.08)

    def test_local_http_encoder_parses_supported_responses(self):
        class Response:
            def __init__(self, payload):
                self.payload = json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self.payload

        def fake_urlopen(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["model"], "local")
            if request.full_url.endswith("/api/embed"):
                self.assertEqual(body["input"], ["test"])
                return Response({"embeddings": [[1.0, 2.0, 3.0, 4.0]]})
            self.assertEqual(body["input"], ["test"])
            return Response({"data": [{"embedding": [4.0, 3.0, 2.0, 1.0]}]})

        with patch("neural_memory.LOCAL_URL_OPENER.open", side_effect=fake_urlopen):
            ollama = LocalHTTPEncoder(
                "ollama", "http://127.0.0.1:11434/api/embed", "local", 4
            )
            compatible = LocalHTTPEncoder(
                "openai-compatible",
                "http://127.0.0.1:8080/v1/embeddings",
                "local",
                4,
            )
            self.assertEqual(ollama.encode("test"), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(compatible.encode("test"), [4.0, 3.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
