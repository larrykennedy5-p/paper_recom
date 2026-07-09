"""Conservative venue whitelist matching.

Only explicit venue metadata is considered. arXiv categories, authors, titles,
and paper topics are never used to infer journal quartiles or publication rank.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_WHITELIST = Path(__file__).resolve().parents[1] / "venue_whitelist.json"
AMBIGUOUS_SINGLE_WORD_VENUES = {"nature", "science", "cell"}


@lru_cache(maxsize=4)
def load_whitelist(path: str | Path = DEFAULT_WHITELIST) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata(paper: dict[str, Any]) -> tuple[str, str]:
    journal_ref = str(paper.get("journal_ref") or "")
    comment = str(paper.get("comment") or paper.get("comments") or "")
    return journal_ref, comment


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def _matches(alias: str, journal_ref: str, comment: str) -> bool:
    pattern = _alias_pattern(alias)
    if pattern.search(journal_ref):
        return True

    if alias.lower() not in AMBIGUOUS_SINGLE_WORD_VENUES:
        return pattern.search(comment) is not None

    # Avoid treating phrases such as "robotic cell" or "science benchmark" as
    # venue evidence. One-word journal names require publication context.
    contextual = re.compile(
        rf"(?:accepted|published|appearing|forthcoming|in press)"
        rf".{{0,30}}(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return contextual.search(comment) is not None


def assess_quality(
    paper: dict[str, Any],
    whitelist_path: str | Path = DEFAULT_WHITELIST,
) -> dict[str, Any]:
    config = load_whitelist(whitelist_path)
    journal_ref, comment = _metadata(paper)

    # Longest aliases win, so "Nature Machine Intelligence" is checked before
    # the broader "Nature" entry.
    candidates = []
    for venue in config["venues"]:
        longest_alias = max((len(alias) for alias in venue["aliases"]), default=0)
        candidates.append((longest_alias, venue))

    for _, venue in sorted(candidates, key=lambda pair: pair[0], reverse=True):
        for alias in sorted(venue["aliases"], key=len, reverse=True):
            if _matches(alias, journal_ref, comment):
                level = config["quality_levels"][venue["level"]]
                return {
                    "venue": venue["name"],
                    "source": venue["name"],
                    "quality_level": level["label"],
                    "quality_rank": level["rank"],
                    "venue_verified": True,
                    "venue_status": "白名单已确认",
                    "venue_evidence": journal_ref or comment,
                }

    level = config["quality_levels"]["unverified"]
    return {
        "venue": None,
        "source": journal_ref.strip() or "arXiv（待人工核验）",
        "quality_level": level["label"],
        "quality_rank": level["rank"],
        "venue_verified": False,
        "venue_status": "待人工核验",
        "venue_evidence": journal_ref or comment,
    }
