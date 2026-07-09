"""Convert the single daily recommendation to a Markdown reading card."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def output_path(input_path: Path) -> Path:
    date = input_path.name.split("_AI_enhanced_", 1)[0]
    return input_path.with_name(f"{date}.md")


def render_card(item: dict, template: str) -> str:
    ai_data = item.get("AI") or {}
    published = item.get("published") or item.get("updated") or ""
    year = item.get("year") or (str(published)[:4] if published else "未知")
    return template.format(
        title=item.get("title", "未知题目"),
        authors="、".join(item.get("authors") or ["未知作者"]),
        source=item.get("source") or "arXiv（待人工核验）",
        year=year,
        url=item.get("abs") or item.get("pdf") or "",
        quality_level=item.get("quality_level", "待人工核验"),
        venue_status=item.get("venue_status", "待人工核验"),
        direction_score=item.get("direction_score", 0),
        recommendation_reason=item.get("recommendation_reason", "暂无"),
        problem=ai_data.get("problem", "暂无"),
        method=ai_data.get("method", "暂无"),
        experiment=ai_data.get("experiment", "摘要中未给出充分实验细节"),
        research_relation=item.get("research_relation", "暂无"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to the JSONL file")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    template_path = Path(__file__).with_name("paper_template.md")
    template = template_path.read_text(encoding="utf-8")

    if data:
        # The AI stage already emits at most one item. Slicing here protects the
        # one-card contract if a legacy or manually edited file is supplied.
        markdown = render_card(data[0], template)
    else:
        markdown = "# 今日论文推荐\n\n今日抓取结果中没有达到方向匹配条件的论文。\n"

    output_path(args.data).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
