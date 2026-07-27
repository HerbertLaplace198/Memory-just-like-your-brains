from __future__ import annotations

import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


from mcp_server import MCPServer
from neural_memory import HashEncoder, NeuralMemory


class StartupConsolidationTests(unittest.TestCase):
    def test_new_memory_uses_english_structural_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), HashEncoder())
            try:
                memory.remember(
                    "这段正文可以继续使用中文。",
                    "test",
                    topics=["论文"],
                    schemas=["User Preference"],
                    episode="Thesis Review",
                    procedures=["Review Workflow"],
                    domain="Academic Work",
                    confirmed=True,
                )
                labels = {
                    row["label"]
                    for row in memory.db.execute(
                        "SELECT label FROM neurons WHERE layer > 1"
                    )
                }
                self.assertIn("Thesis", labels)
                self.assertIn("Thesis Review", labels)
                self.assertIn("Review Workflow", labels)
                self.assertIn("User Preference", labels)
                self.assertIn("Academic Work", labels)
                with self.assertRaises(ValueError):
                    memory.remember(
                        "正文",
                        "test",
                        topics=["Thesis"],
                        domain="中文领域",
                    )
            finally:
                memory.close()

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
                    memory.obsidian_dir / "99 维护中心.md"
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


class ConceptFamilyLayerTests(unittest.TestCase):
    @staticmethod
    def _memory_with_family(root: Path) -> NeuralMemory:
        memory = NeuralMemory(root, HashEncoder())
        memory.remember(
            "A planning case connects portfolio risk, liquidity, and market exposure.",
            "test",
            topics=["Portfolio Risk", "Liquidity Planning", "Market Exposure"],
            procedures=["Risk Review"],
            confirmed=True,
        )
        return memory

    def test_l3f_identity_survives_growth_without_renumbering_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_family(Path(temporary))
            try:
                first = memory.concept_families()
                self.assertEqual(len(first), 1)
                family_id = first[0]["id"]
                self.assertEqual(len(first[0]["members"]), 3)

                memory.remember(
                    "A second planning case links portfolio risk, liquidity, "
                    "market exposure, and capital preservation.",
                    "test",
                    topics=[
                        "Portfolio Risk",
                        "Liquidity Planning",
                        "Market Exposure",
                        "Capital Preservation",
                    ],
                    confirmed=True,
                )
                second = memory.concept_families()
                self.assertEqual(len(second), 1)
                self.assertEqual(second[0]["id"], family_id)
                self.assertEqual(len(second[0]["members"]), 4)
                self.assertNotIn("3F", memory.stats()["layers"])
                self.assertTrue(
                    (memory.concept_family_dir / f"{family_id}.md").is_file()
                )
            finally:
                memory.close()

    def test_confirmed_l3f_collapses_navigation_and_survives_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_family(Path(temporary))
            try:
                family = memory.concept_families()[0]
                family_id = family["id"]
                self.assertTrue(memory.review_concept_family(family_id, "confirm"))
                memory.compile_obsidian()

                family_page = next(
                    (memory.obsidian_dir / "概念家族").glob("L3F · *.md")
                )
                self.assertTrue(family_page.is_file())
                self.assertIn(
                    "layer: L3F",
                    family_page.read_text(encoding="utf-8"),
                )
                self.assertIn(
                    "Risk Review",
                    family_page.read_text(encoding="utf-8"),
                )
                self.assertNotIn(
                    "[[99 维护中心|前往维护中心审核]]",
                    family_page.read_text(encoding="utf-8"),
                )
                home = (memory.obsidian_dir / "00 首页.md").read_text(
                    encoding="utf-8"
                )
                self.assertIn(f"[[概念家族/{family_page.stem}]]", home)
                self.assertNotIn("[[主题/Portfolio Risk]]", home)

                memory.rebuild_index()
                rebuilt = memory.concept_families(status="confirmed")
                self.assertEqual([item["id"] for item in rebuilt], [family_id])
            finally:
                memory.close()

    @staticmethod
    def _memory_with_two_families(root: Path) -> NeuralMemory:
        memory = NeuralMemory(root, HashEncoder())
        memory.remember(
            "Portfolio liquidity downside planning evidence.",
            "test",
            topics=["Portfolio Risk", "Liquidity Planning", "Market Exposure"],
            confirmed=True,
        )
        memory.remember(
            "Writing collaboration review workflow evidence.",
            "test",
            topics=["Writing Workflow", "Review Practice", "Collaboration Style"],
            confirmed=True,
        )
        memory.remember(
            "Astronomy telescope observation notes.",
            "test",
            topics=["Astronomy"],
            confirmed=True,
        )
        for family in memory.concept_families():
            memory.review_concept_family(str(family["id"]), "confirm")
        return memory

    def test_l3f_hit_expands_family_and_keeps_independent_l3_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_two_families(Path(temporary))
            try:
                routing = memory.concept_family_routes(
                    "portfolio liquidity astronomy"
                )
                self.assertTrue(routing["used"])
                active_l3 = {
                    item.label
                    for item in memory.activate(
                        "portfolio liquidity astronomy",
                        winners=100,
                    )
                    if item.layer == 3
                }
                self.assertIn("Portfolio Risk", active_l3)
                self.assertIn("Liquidity Planning", active_l3)
                self.assertIn("Astronomy", active_l3)
                self.assertNotIn("Writing Workflow", active_l3)
                self.assertNotIn("Review Practice", active_l3)
                recalled = {
                    item.summary
                    for item in memory.recall(
                        "portfolio liquidity astronomy",
                        limit=5,
                    )
                }
                self.assertTrue(any("Portfolio" in item for item in recalled))
                self.assertTrue(any("Astronomy" in item for item in recalled))
            finally:
                memory.close()

    def test_l3f_size_bonus_grows_with_breadth_and_support_but_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), HashEncoder())
            try:
                small = {
                    "members": [
                        {"id": "small-1"},
                        {"id": "small-2"},
                        {"id": "small-3"},
                    ]
                }
                large = {
                    "members": [
                        {"id": f"large-{index}"}
                        for index in range(1, 7)
                    ]
                }

                def support(member_id: str) -> list[dict[str, str]]:
                    count = 1 if member_id.startswith("small") else 10
                    confirmed = [
                        {"id": f"{member_id}-l1-{index}", "status": "confirmed"}
                        for index in range(count)
                    ]
                    return confirmed + [{"id": f"{member_id}-candidate", "status": "proposed"}]

                with patch.object(
                    memory,
                    "_related_atoms",
                    side_effect=lambda member_id: support(member_id),
                ):
                    small_profile = memory._family_size_profile(small)
                    large_profile = memory._family_size_profile(large)

                self.assertEqual(small_profile["member_count"], 3)
                self.assertEqual(small_profile["l1_support_count"], 3)
                self.assertEqual(large_profile["member_count"], 6)
                self.assertEqual(large_profile["l1_support_count"], 60)
                self.assertGreater(
                    float(large_profile["size_bonus"]),
                    float(small_profile["size_bonus"]),
                )
                self.assertLessEqual(float(large_profile["size_bonus"]), 0.08)
            finally:
                memory.close()

    def test_larger_family_can_win_close_route_competition_without_changing_l3_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = NeuralMemory(Path(temporary), HashEncoder())
            try:
                small = {
                    "id": "l3f-small",
                    "label": "Small family",
                    "summary": "Small family",
                    "status": "confirmed",
                    "members": [
                        {"id": "small-1", "label": "Small one"},
                        {"id": "small-2", "label": "Small two"},
                        {"id": "small-3", "label": "Small three"},
                    ],
                }
                large = {
                    "id": "l3f-large",
                    "label": "Large family",
                    "summary": "Large family",
                    "status": "confirmed",
                    "members": [
                        {"id": f"large-{index}", "label": f"Large {index}"}
                        for index in range(1, 7)
                    ],
                }

                def fake_encode(text: str) -> list[float]:
                    if "Large family" in text:
                        semantic = 0.87
                    elif "Small family" in text:
                        semantic = 0.90
                    else:
                        semantic = 1.0
                    return [semantic, math.sqrt(max(0.0, 1.0 - semantic**2))]

                def support(member_id: str) -> list[dict[str, str]]:
                    count = 1 if member_id.startswith("small") else 10
                    return [
                        {"id": f"{member_id}-l1-{index}", "status": "confirmed"}
                        for index in range(count)
                    ]

                with (
                    patch.object(
                        memory,
                        "concept_families",
                        return_value=[large, small],
                    ),
                    patch.object(
                        memory,
                        "_family_display_name",
                        side_effect=lambda family: str(family["label"]),
                    ),
                    patch.object(
                        memory,
                        "_related_atoms",
                        side_effect=support,
                    ),
                    patch.object(memory, "_encode", side_effect=fake_encode),
                ):
                    routing = memory.concept_family_routes("target phrase", limit=2)

                self.assertTrue(routing["used"])
                self.assertEqual(routing["families"][0]["id"], "l3f-large")
                self.assertEqual(routing["families"][1]["id"], "l3f-small")
                self.assertLess(
                    routing["families"][0]["base_score"],
                    routing["families"][1]["base_score"],
                )
                self.assertGreater(
                    routing["families"][0]["activation"],
                    routing["families"][1]["activation"],
                )
                self.assertLessEqual(routing["families"][0]["size_bonus"], 0.08)
            finally:
                memory.close()

    def test_closed_family_gate_can_fall_back_to_full_l3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_two_families(Path(temporary))
            try:
                forced_closed = {
                    "used": False,
                    "has_confirmed_families": True,
                    "reason": "family_gate_closed",
                    "families": [],
                    "selected_concept_ids": [],
                }
                with patch.object(
                    memory,
                    "concept_family_routes",
                    return_value=forced_closed,
                ):
                    known, _, activated = memory.probe("writing workflow")
                self.assertTrue(known)
                self.assertIn(
                    "Writing Workflow",
                    {
                        item.label
                        for item in activated
                        if item.layer == 3
                    },
                )
            finally:
                memory.close()

    def test_rejected_l3f_grouping_is_suppressed_after_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_family(Path(temporary))
            try:
                family_id = memory.concept_families()[0]["id"]
                memory.compile_obsidian()
                family_page = next(
                    (memory.obsidian_dir / "概念家族").glob("L3F · *.md")
                )
                self.assertIn(
                    "[[99 维护中心|前往维护中心审核]]",
                    family_page.read_text(encoding="utf-8"),
                )
                maintenance = memory.obsidian_dir / "99 维护中心.md"
                text = maintenance.read_text(encoding="utf-8")
                marker = (
                    f"- [ ] 拒绝这一组家族关系 "
                    f"<!-- concept-family-review:reject:{family_id} -->"
                )
                self.assertIn(marker, text)
                maintenance.write_text(
                    text.replace(marker, marker.replace("[ ]", "[x]")),
                    encoding="utf-8",
                )

                result = memory.sync_obsidian_reviews()
                self.assertEqual(result["families_rejected"], 1)
                self.assertFalse(memory.concept_families())
                memory.rebuild_index()
                self.assertFalse(memory.concept_families())
            finally:
                memory.close()

    def test_confirmed_family_reopens_review_when_membership_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memory = self._memory_with_family(Path(temporary))
            try:
                family_id = memory.concept_families()[0]["id"]
                self.assertTrue(memory.review_concept_family(family_id, "confirm"))
                memory.remember(
                    "A new case adds capital preservation to the same planning family.",
                    "test",
                    topics=[
                        "Portfolio Risk",
                        "Liquidity Planning",
                        "Market Exposure",
                        "Capital Preservation",
                    ],
                    confirmed=True,
                )
                reopened = memory.concept_families(status="proposed")
                self.assertEqual([item["id"] for item in reopened], [family_id])
                self.assertEqual(len(reopened[0]["members"]), 4)
            finally:
                memory.close()


if __name__ == "__main__":
    unittest.main()
