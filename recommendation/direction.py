"""Deterministic direction matching for embodied intelligence research."""

from __future__ import annotations

import re
from typing import Any


CORE_KEYWORDS = (
    "dexterous hand",
    "robotic hand",
    "robot hand",
    "robotic manipulation",
    "robot manipulation",
    "embodied ai",
    "embodied intelligence",
    "tactile sensing",
    "tactile perception",
    "grasping",
    "in-hand manipulation",
    "in hand manipulation",
)

APPLICATION_KEYWORDS = (
    "agricultural robotics",
    "agricultural robot",
    "fruit picking",
    "fruit harvesting",
    "crop harvesting",
    "crop picking",
    "underwater manipulation",
    "underwater robot",
    "underwater robotics",
    "marine robotics",
    "marine robot",
)

EXTENSION_KEYWORDS = (
    "vision-language-action",
    "vision language action",
    "vla",
    "multimodal perception",
    "multi-modal perception",
    "sim-to-real",
    "sim2real",
    "robot learning",
    "contact-rich manipulation",
    "bimanual manipulation",
    "deformable object manipulation",
    "6d pose",
    "force sensing",
    "haptic perception",
    "visual tactile",
)

FIELD_WEIGHTS = {
    "title": 4.0,
    "summary": 2.0,
    "categories": 1.2,
    "authors": 0.5,
    "comment": 1.0,
}

KEYWORD_WEIGHTS = {
    "core": 3.0,
    "application": 2.5,
    "extension": 1.2,
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(part) for part in value)
    return str(value)


def _normalize(value: Any) -> str:
    text = _as_text(value).lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _contains(text: str, keyword: str) -> bool:
    # Word boundaries avoid matching short aliases such as VLA inside another word.
    pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def score_direction(paper: dict[str, Any]) -> dict[str, Any]:
    """Return a 0-100 score plus transparent keyword evidence.

    Every requested metadata field participates. A keyword is counted at the
    strongest field in which it appears, preventing repeated abstract mentions
    from inflating the score.
    """

    fields = {
        "title": _normalize(paper.get("title")),
        "summary": _normalize(paper.get("summary") or paper.get("abstract")),
        "categories": _normalize(paper.get("categories")),
        "authors": _normalize(paper.get("authors")),
        "comment": _normalize(paper.get("comment") or paper.get("comments")),
    }
    groups = {
        "core": CORE_KEYWORDS,
        "application": APPLICATION_KEYWORDS,
        "extension": EXTENSION_KEYWORDS,
    }

    raw_score = 0.0
    matched: dict[str, list[str]] = {name: [] for name in groups}
    evidence: dict[str, list[str]] = {}

    for group_name, keywords in groups.items():
        for keyword in keywords:
            hit_fields = [name for name, text in fields.items() if _contains(text, keyword)]
            if not hit_fields:
                continue
            matched[group_name].append(keyword)
            evidence[keyword] = hit_fields
            strongest_field = max(hit_fields, key=lambda name: FIELD_WEIGHTS[name])
            raw_score += KEYWORD_WEIGHTS[group_name] * FIELD_WEIGHTS[strongest_field]

    categories = fields["categories"]
    if "cs.ro" in categories:
        raw_score += 5.0
    if matched["core"] and matched["application"]:
        raw_score += 10.0
    elif len(matched["core"]) >= 2:
        raw_score += 5.0

    score = min(100, max(0, round(raw_score)))
    compact_matches = {key: value for key, value in matched.items() if value}
    return {
        "direction_score": score,
        "direction_matches": compact_matches,
        "direction_evidence": evidence,
    }


def build_recommendation_reason(direction: dict[str, Any], quality: dict[str, Any]) -> str:
    matches = direction.get("direction_matches", {})
    keywords = matches.get("core", []) + matches.get("application", []) + matches.get("extension", [])
    keyword_text = "、".join(keywords[:5]) if keywords else "具身智能相关主题"
    venue = quality.get("venue")
    if quality.get("venue_verified") and venue:
        quality_text = f"来源已匹配白名单中的 {venue}"
    else:
        quality_text = "来源尚未得到白名单确认，需人工核验"
    return (
        f"方向匹配得分 {direction.get('direction_score', 0)}/100，"
        f"命中 {keyword_text}；{quality_text}。"
    )


def build_research_relation(direction: dict[str, Any]) -> str:
    matches = direction.get("direction_matches", {})
    core = matches.get("core", [])
    application = matches.get("application", [])
    extension = matches.get("extension", [])
    if application:
        return (
            "可直接支撑智慧农业机器人操作或水下机器人操作的开题论证，"
            f"重点关联：{'、'.join(application[:3])}。"
        )
    if core:
        return (
            "可作为具身智能与灵巧手开题中的方法或技术路线依据，"
            f"重点关联：{'、'.join(core[:3])}。"
        )
    if extension:
        return (
            "属于多模态感知、VLA 或 sim-to-real 等扩展技术，"
            f"可作为开题的能力模块参考：{'、'.join(extension[:3])}。"
        )
    return "与当前开题方向仅有弱关联，建议人工复核后再纳入相关工作。"
