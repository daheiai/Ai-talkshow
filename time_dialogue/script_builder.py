from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import load_json, resolve_path, write_json, write_text


class ScriptBuilder:
    def __init__(self, root: Path, run_plan: dict[str, Any]) -> None:
        self.root = root
        self.run_plan = run_plan

    def build(self, highlights_path: Path) -> Path:
        data = load_json(highlights_path)
        session_id = data["session_id"]
        script_dir = resolve_path(self.root, self.run_plan.get("script_output_dir", "outputs/script_candidates"))
        visual_dir = resolve_path(self.root, "outputs/visual_cards")

        script_path = script_dir / f"{session_id}-script.md"
        visual_path = visual_dir / f"{session_id}-visual-cards.json"

        write_text(script_path, self._build_markdown(data))
        write_json(visual_path, self._build_visual_cards(data))
        return script_path

    def _build_markdown(self, data: dict[str, Any]) -> str:
        old_name = data.get("old_model", {}).get("display_name", "旧AI")
        new_name = data.get("new_model", {}).get("display_name", "新AI")
        highlights = data.get("highlights", [])

        lines = [
            f"# 脚本候选：{new_name} vs {old_name}",
            "",
            f"- Session: `{data['session_id']}`",
            f"- 生成时间: `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## 开场",
            "",
            f"口播：我让一个停在过去的AI，和现在的{new_name}聊了聊。最有意思的不是谁答得更准，而是它们隔着几年时间，理解同一个世界时的错位。",
            "",
        ]

        for index, item in enumerate(highlights, start=1):
            lines.extend(
                [
                    f"## 片段 {index}：{item.get('card_title', item.get('topic_title', '未命名卡片'))}",
                    "",
                    f"幕：{item.get('act', '')}",
                    f"卡片类型：{item.get('card_type', '')}",
                    "",
                    f"旁白：这一段的看点是，{item['why_it_works']}",
                    "",
                    "上屏对话：",
                    "",
                    "```text",
                    item["dialogue_text"],
                    "```",
                    "",
                    f"画面：{item['suggested_visual']}",
                    f"备注：事实风险 `{item['factual_risk']}`，剪进正片前需要按风险等级核验。原始卡片运行：`{item.get('card_run_id', '')}`。",
                    "",
                ]
            )

        lines.extend(
            [
                "## 收束",
                "",
                "口播：这期不打分。因为真正好看的地方，不是新模型把旧模型纠正了，而是旧模型那些真诚的猜测，被后来的现实轻轻撞了一下。",
                "",
            ]
        )
        return "\n".join(lines)

    def _build_visual_cards(self, data: dict[str, Any]) -> dict[str, Any]:
        cards = []
        for index, item in enumerate(data.get("highlights", []), start=1):
            cards.append(
                {
                    "card_id": f"{data['session_id']}_card_{index:02d}",
                    "source_card_id": item.get("card_id", item.get("topic_id")),
                    "source_card_run_id": item.get("card_run_id"),
                    "card_title": item.get("card_title", item.get("topic_title")),
                    "act": item.get("act"),
                    "card_type": item.get("card_type"),
                    "turn_range": item["turn_range"],
                    "dialogue_text": item["dialogue_text"],
                    "suggested_visual": item["suggested_visual"],
                    "factual_risk": item["factual_risk"],
                }
            )
        return {
            "session_id": data["session_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cards": cards,
        }
