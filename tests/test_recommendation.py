from __future__ import annotations

import unittest

from ai.enhance import has_concrete_experiment_result
from recommendation.direction import score_direction
from recommendation.quality import assess_quality
from recommendation.selector import enrich_and_select
from to_md.convert import render_card


class DirectionScoringTests(unittest.TestCase):
    def test_uses_all_requested_metadata_and_caps_score(self):
        paper = {
            "title": "Dexterous Hand Grasping with Tactile Sensing",
            "summary": (
                "Robotic manipulation and in-hand manipulation for fruit picking "
                "with multimodal perception and sim-to-real transfer."
            ),
            "categories": ["cs.RO", "cs.CV"],
            "authors": ["Embodied AI Group"],
            "comment": "An underwater robot extension is discussed.",
        }
        result = score_direction(paper)
        self.assertGreater(result["direction_score"], 50)
        self.assertLessEqual(result["direction_score"], 100)
        self.assertIn("dexterous hand", result["direction_matches"]["core"])
        self.assertIn("fruit picking", result["direction_matches"]["application"])

    def test_unrelated_paper_scores_zero(self):
        result = score_direction(
            {
                "title": "A theorem about number fields",
                "summary": "We prove a result in algebra.",
                "categories": ["math.NT"],
                "authors": ["A. Author"],
                "comment": "",
            }
        )
        self.assertEqual(result["direction_score"], 0)


class VenueQualityTests(unittest.TestCase):
    def test_confirms_whitelisted_venue_from_comment(self):
        result = assess_quality({"comment": "Accepted to ICRA 2026", "journal_ref": None})
        self.assertTrue(result["venue_verified"])
        self.assertEqual(result["venue"], "ICRA")
        self.assertEqual(result["quality_level"], "顶级会议")

    def test_unknown_venue_remains_unverified(self):
        result = assess_quality({"comment": "Accepted to ExampleConf", "journal_ref": None})
        self.assertFalse(result["venue_verified"])
        self.assertEqual(result["quality_level"], "待人工核验")

    def test_does_not_treat_topic_word_as_science_venue(self):
        result = assess_quality(
            {"comment": "This benchmark advances science for the robotic cell.", "journal_ref": None}
        )
        self.assertFalse(result["venue_verified"])


class DailySelectionTests(unittest.TestCase):
    def test_quality_precedes_direction_then_date(self):
        papers = [
            {
                "id": "high-direction",
                "title": "Dexterous Hand and Robotic Hand for Grasping",
                "summary": "Tactile sensing for in-hand manipulation.",
                "categories": ["cs.RO"],
                "authors": [],
                "comment": "",
                "published": "2026-07-09T00:00:00+00:00",
            },
            {
                "id": "top-venue",
                "title": "Robotic Manipulation",
                "summary": "A grasping system.",
                "categories": ["cs.RO"],
                "authors": [],
                "comment": "Accepted to RSS 2026",
                "published": "2026-07-08T00:00:00+00:00",
            },
        ]
        selected = enrich_and_select(papers)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "top-venue")

    def test_returns_empty_when_nothing_matches_direction(self):
        selected = enrich_and_select(
            [
                {
                    "id": "unrelated",
                    "title": "Number Theory",
                    "summary": "An algebraic proof.",
                    "categories": ["math.NT"],
                    "authors": [],
                    "comment": "Accepted to NeurIPS 2026",
                }
            ]
        )
        self.assertEqual(selected, [])

    def test_publication_date_breaks_equal_score_ties(self):
        base = {
            "title": "Robotic Manipulation",
            "summary": "A grasping system.",
            "categories": ["cs.RO"],
            "authors": [],
            "comment": "",
        }
        selected = enrich_and_select(
            [
                {**base, "id": "older", "published": "2026-07-08T00:00:00+00:00"},
                {**base, "id": "newer", "published": "2026-07-09T00:00:00+00:00"},
            ]
        )
        self.assertEqual(selected[0]["id"], "newer")


class MarkdownCardTests(unittest.TestCase):
    def test_card_contains_every_required_field(self):
        item = {
            "title": "A Dexterous Hand",
            "authors": ["A. Author"],
            "source": "ICRA",
            "year": "2026",
            "abs": "https://arxiv.org/abs/2601.00001",
            "quality_level": "顶级会议",
            "venue_status": "白名单已确认",
            "direction_score": 88,
            "recommendation_reason": "方向高度匹配。",
            "research_relation": "可支撑灵巧手开题。",
            "AI": {
                "problem": "解决遮挡下抓取问题。",
                "method": "提出触觉融合策略。",
                "experiment": "成功率提高 8%。",
            },
        }
        template = (
            "{title}|{authors}|{source}|{year}|{url}|{quality_level}|"
            "{venue_status}|{direction_score}|{recommendation_reason}|"
            "{problem}|{method}|{experiment}|{research_relation}"
        )
        card = render_card(item, template)
        for expected in (
            "A Dexterous Hand",
            "A. Author",
            "ICRA",
            "2026",
            "88",
            "方向高度匹配",
            "解决遮挡下抓取问题",
            "提出触觉融合策略",
            "成功率提高 8%",
            "可支撑灵巧手开题",
        ):
            self.assertIn(expected, card)


class ExperimentGuardTests(unittest.TestCase):
    def test_generic_goal_is_not_treated_as_result(self):
        self.assertFalse(
            has_concrete_experiment_result(
                "We propose a controller to achieve dexterous manipulation."
            )
        )

    def test_reported_improvement_is_treated_as_result(self):
        self.assertTrue(
            has_concrete_experiment_result(
                "Experiments show a 12% improvement in grasp success."
            )
        )


if __name__ == "__main__":
    unittest.main()
