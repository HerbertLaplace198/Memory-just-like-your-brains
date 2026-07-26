from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcp_server import MCPServer
from neural_memory import HashEncoder, NeuralMemory


class StartupConsolidationTests(unittest.TestCase):
    def test_consolidation_runs_only_after_twenty_four_hours(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), HashEncoder())
            first_at = datetime(2026, 7, 26, 12, tzinfo=timezone.utc)
            try:
                first = memory.consolidate_if_due(reference=first_at)
                early = memory.consolidate_if_due(
                    reference=first_at + timedelta(hours=23, minutes=59)
                )
                due = memory.consolidate_if_due(
                    reference=first_at + timedelta(hours=24)
                )
            finally:
                memory.close()

            self.assertTrue(first["performed"])
            self.assertFalse(early["performed"])
            self.assertTrue(due["performed"])

    def test_mcp_startup_catches_up_once_then_stays_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = MCPServer(root)
            try:
                self.assertTrue(first.startup_consolidation["performed"])
                self.assertTrue((root / "obsidian-view").is_dir())
            finally:
                first.close()

            second = MCPServer(root)
            try:
                self.assertFalse(second.startup_consolidation["performed"])
            finally:
                second.close()

    def test_two_clients_share_one_due_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = NeuralMemory(root, HashEncoder())
            first_at = datetime(2026, 7, 26, 12, tzinfo=timezone.utc)
            try:
                baseline.consolidate_if_due(reference=first_at)
            finally:
                baseline.close()

            due_at = first_at + timedelta(hours=24)

            def connect() -> bool:
                memory = NeuralMemory(root, HashEncoder())
                try:
                    return bool(
                        memory.consolidate_if_due(reference=due_at)["performed"]
                    )
                finally:
                    memory.close()

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: connect(), range(2)))

            self.assertEqual(results.count(True), 1)
            self.assertEqual(results.count(False), 1)


class StableConceptIdentityTests(unittest.TestCase):
    def test_emergent_identity_survives_new_supporting_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), HashEncoder())
            try:
                for index in range(3):
                    memory.remember(
                        f"Repeated risk pattern with probability and downside {index}.",
                        "test",
                        confirmed=True,
                    )
                first = memory.db.execute(
                    """SELECT id FROM neurons
                       WHERE layer=3 AND label LIKE 'Emergent Concept %'"""
                ).fetchone()
                self.assertIsNotNone(first)

                memory.remember(
                    "Repeated risk pattern with probability and downside 4.",
                    "test",
                    confirmed=True,
                )
                second = memory.db.execute(
                    """SELECT id FROM neurons
                       WHERE layer=3 AND label LIKE 'Emergent Concept %'"""
                ).fetchone()
                self.assertIsNotNone(second)
                self.assertEqual(first["id"], second["id"])
                self.assertTrue(
                    (
                        memory.concept_identity_dir / f"{second['id']}.md"
                    ).is_file()
                )
            finally:
                memory.close()

    @staticmethod
    def _memory_with_duplicate_topics(root: Path) -> NeuralMemory:
        memory = NeuralMemory(root, HashEncoder())
        for index in range(3):
            memory.remember(
                f"Risk choice evidence {index}.",
                "test",
                topics=["Risk Decision", "Risk Decisions"],
                confirmed=True,
            )
        return memory

    def test_duplicate_candidate_never_merges_without_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_duplicate_topics(Path(temporary))
            try:
                candidates = memory.concept_duplicate_candidates()
                self.assertTrue(candidates)
                active = memory.db.execute(
                    """SELECT count(*) FROM neurons
                       WHERE layer=3 AND status!='archived'
                         AND label IN ('Risk Decision','Risk Decisions')"""
                ).fetchone()[0]
                self.assertEqual(active, 2)
                memory.compile_obsidian()
                maintenance = (
                    memory.obsidian_dir / "99 Maintenance.md"
                ).read_text(encoding="utf-8")
                self.assertIn("concept-duplicate-review:merge-left", maintenance)
                self.assertIn("concept-duplicate-review:distinct", maintenance)
            finally:
                memory.close()

    def test_distinct_decision_survives_rebuild_and_suppresses_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_duplicate_topics(Path(temporary))
            try:
                review_id = memory.concept_duplicate_candidates()[0]["id"]
                self.assertTrue(
                    memory.review_concept_duplicate(review_id, "distinct")
                )
                self.assertFalse(memory.concept_duplicate_candidates())
                memory.rebuild_index()
                self.assertFalse(memory.concept_duplicate_candidates())
                self.assertTrue(
                    (memory.concept_decision_dir / f"{review_id}.md").is_file()
                )
            finally:
                memory.close()

    def test_merge_preserves_alias_and_rebuilds_one_active_concept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_duplicate_topics(Path(temporary))
            try:
                candidate = memory.concept_duplicate_candidates()[0]
                left_label = candidate["left_label"]
                right_label = candidate["right_label"]
                self.assertTrue(
                    memory.review_concept_duplicate(
                        candidate["id"],
                        "merge-left",
                    )
                )
                resolved = memory._find_named(3, right_label)
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved["label"], left_label)
                memory.rebuild_index()
                active_labels = [
                    row["label"]
                    for row in memory.db.execute(
                        """SELECT label FROM neurons
                           WHERE layer=3 AND status NOT IN ('archived','rejected')"""
                    )
                    if row["label"] in {left_label, right_label}
                ]
                self.assertEqual(active_labels, [left_label])
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
