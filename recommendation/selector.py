"""Enrich crawled papers and choose exactly one daily recommendation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .direction import (
    build_recommendation_reason,
    build_research_relation,
    score_direction,
)
from .quality import DEFAULT_WHITELIST, assess_quality


def _date_timestamp(value: Any) -> float:
    if not value:
        return 0.0
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return 0.0


def recommendation_sort_key(paper: dict[str, Any]) -> tuple[int, int, float]:
    """Quality first, then direction score, then publication date."""

    return (
        int(paper.get("quality_rank", 0)),
        int(paper.get("direction_score", 0)),
        _date_timestamp(paper.get("published") or paper.get("updated")),
    )


def enrich_paper(
    paper: dict[str, Any],
    whitelist_path: str | Path = DEFAULT_WHITELIST,
) -> dict[str, Any]:
    enriched = dict(paper)
    direction = score_direction(enriched)
    quality = assess_quality(enriched, whitelist_path)
    enriched.update(direction)
    enriched.update(quality)
    enriched["recommendation_reason"] = build_recommendation_reason(direction, quality)
    enriched["research_relation"] = build_research_relation(direction)
    published = enriched.get("published") or enriched.get("updated")
    enriched["year"] = str(published)[:4] if published else "未知"
    return enriched


def enrich_and_select(
    papers: Iterable[dict[str, Any]],
    whitelist_path: str | Path = DEFAULT_WHITELIST,
) -> list[dict[str, Any]]:
    """Return the single best direction-relevant paper, or an empty list."""

    enriched = [enrich_paper(paper, whitelist_path) for paper in papers]
    relevant = [paper for paper in enriched if paper["direction_score"] > 0]
    if not relevant:
        return []
    return [max(relevant, key=recommendation_sort_key)]
