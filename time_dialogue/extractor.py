from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .io_utils import extract_json_object, read_text, resolve_path, write_json, write_text
from .llm_client import LLMClient


class HighlightExtractor:
    def __init__(
        self,
        root: Path,
        models: dict[str, ModelConfig],
        run_plan: dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        self.root = root
        self.models = models
        self.run_plan = run_plan
        self.client = LLMClient(dry_run=dry_run)
        self.dry_run = dry_run
        self.editor_template = read_text(root / "prompts" / "editor_prompt.txt")

    async def extract(
        self,
        record_path: Path,
        use_llm_editor: bool = False,
        top_k: int | None = None,
    ) -> Path:
        import json

        with record_path.open("r", encoding="utf-8") as f:
            record = json.load(f)

        fragments = self._build_topic_fragments(record)
        highlights: list[dict[str, Any]] = []
        for fragment in fragments:
            if use_llm_editor and not self.dry_run:
                scores = await self._score_with_llm(fragment)
            else:
                scores = self._score_heuristically(fragment)
            highlights.append({**fragment, **scores})

        for item in highlights:
            item["combined_score"] = self._combined_score(item)
        highlights.sort(key=lambda item: item["combined_score"], reverse=True)

        selected = highlights[: top_k or int(self.run_plan.get("top_k_highlights", 5))]
        output = {
            "source_record": str(record_path),
            "session_id": record["session_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": record.get("dry_run", False),
            "mode": record.get("mode", "fair_run"),
            "evaluation_principle": record.get("evaluation_principle", ""),
            "old_model": record.get("old_model", {}),
            "new_model": record.get("new_model", {}),
            "highlights": selected,
        }

        output_dir = resolve_path(self.root, self.run_plan.get("selected_output_dir", "records/selected"))
        output_path = output_dir / f"{record['session_id']}-highlights.json"
        write_json(output_path, output)
        self._write_markdown_report(output_path.with_suffix(".md"), output)
        return output_path

    def _build_topic_fragments(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        fragments: list[dict[str, Any]] = []
        turns = record.get("turns", [])
        card_order = record.get("card_plan", record.get("topic_plan", []))
        card_meta = {
            item["id"]: item for item in record.get("cards", record.get("topics", []))
        }

        if any("card_run_id" in turn for turn in turns):
            seen_runs: list[str] = []
            for turn in turns:
                run_id = turn.get("card_run_id")
                if run_id and run_id not in seen_runs:
                    seen_runs.append(run_id)
            for run_id in seen_runs:
                run_turns = [turn for turn in turns if turn.get("card_run_id") == run_id]
                if run_turns:
                    fragments.append(self._fragment_from_turns(run_turns, card_meta))
            return fragments

        for card_id in card_order:
            card_turns = [
                turn
                for turn in turns
                if turn.get("card_id", turn.get("topic_id")) == card_id
            ]
            if card_turns:
                fragments.append(self._fragment_from_turns(card_turns, card_meta))
        return fragments

    def _fragment_from_turns(
        self,
        turns: list[dict[str, Any]],
        card_meta: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        first_turn = turns[0]["turn_id"]
        last_turn = turns[-1]["turn_id"]
        card_id = turns[0].get("card_id", turns[0].get("topic_id"))
        meta = card_meta.get(card_id, {})
        dialogue_text = "\n".join(
            f"{turn['speaker_label']}：{turn['text']}" for turn in turns
        )
        return {
            "card_id": card_id,
            "card_title": meta.get("title", turns[0].get("card_title", card_id)),
            "card_run_id": turns[0].get("card_run_id", card_id),
            "repeat_index": turns[0].get("repeat_index", 1),
            "act": meta.get("act", turns[0].get("act", "")),
            "card_type": meta.get("card_type", turns[0].get("card_type", "")),
            "protocol": meta.get("protocol", turns[0].get("protocol", "")),
            "topic_id": card_id,
            "topic_title": meta.get("title", turns[0].get("card_title", card_id)),
            "topic_objective": meta.get("objective", ""),
            "evaluation_focus": meta.get("evaluation_focus", []),
            "fact_sheet": meta.get("fact_sheet", {}),
            "fact_check_required": bool(meta.get("fact_check_required", False)),
            "turn_range": [first_turn, last_turn],
            "dialogue_text": dialogue_text,
        }

    async def _score_with_llm(self, fragment: dict[str, Any]) -> dict[str, Any]:
        editor_model_id = self.run_plan.get("editor_model_id")
        if not editor_model_id:
            return self._score_heuristically(fragment)
        if editor_model_id not in self.models:
            raise KeyError(f"Unknown editor model id: {editor_model_id}")

        prompt = self.editor_template.replace("{{fragment}}", fragment["dialogue_text"])
        messages = [{"role": "user", "content": prompt}]
        result = await self.client.complete(self.models[editor_model_id], messages)
        parsed = extract_json_object(result.content)
        if not parsed:
            fallback = self._score_heuristically(fragment)
            fallback["why_it_works"] += "（编辑器JSON解析失败，已使用启发式评分）"
            return fallback
        return self._normalize_scores(parsed, fragment)

    def _score_heuristically(self, fragment: dict[str, Any]) -> dict[str, Any]:
        text = fragment["dialogue_text"]
        hook_words = ["没想到", "竟然", "后来", "最", "第一次", "问题", "一句话"]
        time_gap_words = [
            "2023",
            "2024",
            "2025",
            "2026",
            "截止",
            "未来",
            "后来",
            "时间线",
            "三年",
            "不知道",
            "不确定",
            "超出",
        ]
        emotion_words = ["复杂", "兴奋", "焦虑", "疲惫", "轻松", "不安", "遗憾", "普通人", "日常"]
        personality_words = ["我", "你", "？", "吗", "猜", "等等", "如果", "也许"]
        evaluation_words = ["确定", "不确定", "依据", "预测", "符合", "遗漏", "修正", "风险", "核验"]

        def score(words: list[str], base: int = 1) -> int:
            hits = sum(1 for word in words if word in text)
            return max(1, min(5, base + hits))

        factual_risk = "medium" if fragment.get("fact_check_required") else "low"
        if any(word in text for word in ["发布", "公司", "模型", "开源", "战争", "票房", "日期"]):
            factual_risk = "medium"
        return {
            "hook_score": score(hook_words),
            "time_gap_score": score(time_gap_words),
            "evaluation_value_score": score(evaluation_words),
            "emotional_score": score(emotion_words),
            "model_personality_score": score(personality_words),
            "factual_risk": factual_risk,
            "why_it_works": self._default_why(fragment),
            "suggested_visual": self._default_visual(fragment),
        }

    def _normalize_scores(self, raw: dict[str, Any], fragment: dict[str, Any]) -> dict[str, Any]:
        def int_score(key: str) -> int:
            try:
                return max(1, min(5, int(raw.get(key, 1))))
            except (TypeError, ValueError):
                return 1

        factual_risk = str(raw.get("factual_risk", "medium")).lower()
        if factual_risk not in {"low", "medium", "high"}:
            factual_risk = "medium"
        return {
            "hook_score": int_score("hook_score"),
            "time_gap_score": int_score("time_gap_score"),
            "evaluation_value_score": int_score("evaluation_value_score"),
            "emotional_score": int_score("emotional_score"),
            "model_personality_score": int_score("model_personality_score"),
            "factual_risk": factual_risk,
            "why_it_works": str(raw.get("why_it_works") or self._default_why(fragment)),
            "suggested_visual": str(raw.get("suggested_visual") or self._default_visual(fragment)),
        }

    def _combined_score(self, item: dict[str, Any]) -> float:
        risk_penalty = {"low": 0.0, "medium": 0.4, "high": 1.0}.get(item.get("factual_risk"), 0.4)
        return (
            item.get("hook_score", 1) * 1.1
            + item.get("time_gap_score", 1) * 1.4
            + item.get("evaluation_value_score", 1) * 1.5
            + item.get("emotional_score", 1) * 1.3
            + item.get("model_personality_score", 1)
            - risk_penalty
        )

    def _default_why(self, fragment: dict[str, Any]) -> str:
        if fragment.get("fact_check_required"):
            return "这一段包含可评估的时间差信息，适合做横向对比，但具体事实需要二次核验。"
        return "这一段能看出旧AI和新AI的回答差异，适合作为后筛选候选片段。"

    def _default_visual(self, fragment: dict[str, Any]) -> str:
        return f"左右对话框展示“{fragment['card_title']}”，旧AI颜色偏复古，新AI颜色偏明亮。"

    def _write_markdown_report(self, path: Path, output: dict[str, Any]) -> None:
        old_name = output.get("old_model", {}).get("display_name", "旧AI")
        new_name = output.get("new_model", {}).get("display_name", "新AI")
        lines = [
            f"# 高光片段报告：{new_name} vs {old_name}",
            "",
            f"- Session: `{output['session_id']}`",
            f"- Source: `{output['source_record']}`",
            f"- Mode: `{output.get('mode', 'fair_run')}`",
            f"- 原则：{output.get('evaluation_principle', '')}",
            "",
        ]
        for index, item in enumerate(output.get("highlights", []), start=1):
            lines.extend(
                [
                    f"## {index}. {item['card_title']}",
                    "",
                    f"- 幕：{item.get('act', '')}",
                    f"- 卡片类型：{item.get('card_type', '')}",
                    f"- 综合分：{item['combined_score']:.1f}",
                    f"- 钩子：{item['hook_score']} / 时间差：{item['time_gap_score']} / 评测价值：{item.get('evaluation_value_score', 1)} / 情绪：{item['emotional_score']} / 人格：{item['model_personality_score']}",
                    f"- 事实风险：{item['factual_risk']}",
                    f"- 为什么可用：{item['why_it_works']}",
                    f"- 画面建议：{item['suggested_visual']}",
                    "",
                    "```text",
                    item["dialogue_text"],
                    "```",
                    "",
                ]
            )
        write_text(path, "\n".join(lines))
