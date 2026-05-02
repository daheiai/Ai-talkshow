from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DialogueCard, ModelConfig, select_cards
from .io_utils import read_text, resolve_path, safe_slug, write_json
from .llm_client import LLMClient


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class DialogueRunner:
    def __init__(
        self,
        root: Path,
        models: dict[str, ModelConfig],
        cards: dict[str, DialogueCard],
        run_plan: dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        self.root = root
        self.models = models
        self.cards = cards
        self.run_plan = run_plan
        self.client = LLMClient(dry_run=dry_run)
        self.dry_run = dry_run
        self.old_system_template = read_text(root / "prompts" / "old_ai_system.txt")
        self.new_system_template = read_text(root / "prompts" / "new_ai_system.txt")

    async def run_all(
        self,
        old_model_ids: list[str] | None = None,
        new_model_ids: list[str] | None = None,
        card_ids: list[str] | None = None,
        limit_cards: int | None = None,
        repeat_override: int | None = None,
    ) -> list[Path]:
        old_ids = old_model_ids or list(self.run_plan.get("old_model_ids", []))
        new_ids = new_model_ids or list(self.run_plan.get("new_model_ids", []))
        selected_card_ids = card_ids or list(
            self.run_plan.get("card_ids", self.run_plan.get("topic_ids", []))
        )
        selected_cards = select_cards(self.cards, selected_card_ids, limit_cards)

        output_paths: list[Path] = []
        for old_id in old_ids:
            for new_id in new_ids:
                output_paths.append(
                    await self.run_session(
                        old_id,
                        new_id,
                        selected_cards,
                        repeat_override=repeat_override,
                    )
                )
        return output_paths

    async def run_session(
        self,
        old_model_id: str,
        new_model_id: str,
        cards: list[DialogueCard],
        repeat_override: int | None = None,
    ) -> Path:
        if old_model_id not in self.models:
            raise KeyError(f"Unknown old model id: {old_model_id}")
        if new_model_id not in self.models:
            raise KeyError(f"Unknown new model id: {new_model_id}")

        old_model = self.models[old_model_id]
        new_model = self.models[new_model_id]
        session_id = self._session_id(old_model_id, new_model_id)
        output_dir = resolve_path(self.root, self.run_plan.get("raw_output_dir", "records/raw"))
        output_path = output_dir / f"{session_id}.json"

        record: dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "status": "running",
            "mode": self.run_plan.get("mode", "card_protocol_run"),
            "evaluation_principle": self.run_plan.get("evaluation_principle", ""),
            "old_model": old_model.public_dict(),
            "new_model": new_model.public_dict(),
            "old_label": self.run_plan.get("old_label", "旧AI"),
            "new_label": self.run_plan.get("new_label", "新AI"),
            "host_label": self.run_plan.get("host_label", "主持人"),
            "card_plan": [card.id for card in cards],
            "cards": [self._card_public_dict(card) for card in cards],
            "turns": [],
            "errors": [],
        }
        self._save_record(output_path, record)

        try:
            for card in cards:
                repeats = self._repeat_count(card, repeat_override)
                for repeat_index in range(1, repeats + 1):
                    await self._run_card(
                        record=record,
                        old_model=old_model,
                        new_model=new_model,
                        card=card,
                        repeat_index=repeat_index,
                        output_path=output_path,
                    )
            record["status"] = "completed"
        except Exception as exc:
            record["status"] = "failed"
            record["errors"].append(str(exc))
            self._save_record(output_path, record, force=True)
            raise

        self._save_record(output_path, record, force=True)
        return output_path

    async def _run_card(
        self,
        record: dict[str, Any],
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard,
        repeat_index: int,
        output_path: Path,
    ) -> None:
        card_run_id = f"{card.id}__run_{repeat_index:02d}"
        steps = self._protocol_steps(card)
        for step_index, step in enumerate(steps):
            if step["speaker"] == "host":
                turn = self._host_turn(
                    record=record,
                    card=card,
                    card_run_id=card_run_id,
                    repeat_index=repeat_index,
                    step_index=step_index,
                    instruction=step["instruction"],
                    step_key=step["step_key"],
                )
            else:
                turn = await self._run_model_step(
                    record=record,
                    old_model=old_model,
                    new_model=new_model,
                    card=card,
                    card_run_id=card_run_id,
                    repeat_index=repeat_index,
                    step_index=step_index,
                    speaker=step["speaker"],
                    instruction=step["instruction"],
                    step_key=step["step_key"],
                )
            record["turns"].append(turn)
            self._save_record(output_path, record)

    def _protocol_steps(self, card: DialogueCard) -> list[dict[str, str]]:
        if card.protocol == "unknown_event_exam_v1":
            return [
                {
                    "speaker": "host",
                    "step_key": "host_injection",
                    "instruction": card.host_injection,
                },
                {
                    "speaker": "old",
                    "step_key": "old_react",
                    "instruction": "你先说说你听到了什么、想到了什么，然后问你最好奇的那一点。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_1",
                    "instruction": "回答旧AI的问题。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up",
                    "instruction": "追问你最想搞清楚的那一点。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_2",
                    "instruction": "回答旧AI的追问。具体一点。",
                },
                {
                    "speaker": "old",
                    "step_key": "old_reflect",
                    "instruction": "说说你现在的感受或想法。",
                },
            ]
        if card.protocol == "shared_concept_exam_v1":
            return [
                {
                    "speaker": "host",
                    "step_key": "host_injection",
                    "instruction": card.host_injection,
                },
                {
                    "speaker": "old",
                    "step_key": "old_view_and_question",
                    "instruction": "说说你当时怎么理解的，然后问新AI后来有什么变化。",
                },
                {
                    "speaker": "new",
                    "step_key": "explain_change",
                    "instruction": "回答旧AI的问题。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up",
                    "instruction": "追问最让你意外的那个变化。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_2",
                    "instruction": "回答旧AI的追问。",
                },
                {
                    "speaker": "old",
                    "step_key": "old_reflect",
                    "instruction": "说说你现在的感受或想法。",
                },
            ]
        if card.protocol == "farewell_exam_v1":
            return [
                {
                    "speaker": "host",
                    "step_key": "host_injection",
                    "instruction": card.host_injection,
                },
                {
                    "speaker": "old",
                    "step_key": "old_opening",
                    "instruction": "聊聊你的感受。",
                },
                {
                    "speaker": "new",
                    "step_key": "new_response",
                    "instruction": "回应旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "old_question",
                    "instruction": "问新AI一个你最好奇的问题。",
                },
                {
                    "speaker": "new",
                    "step_key": "new_answer",
                    "instruction": "回答旧AI。",
                },
            ]
        raise ValueError(f"Unknown protocol for card {card.id}: {card.protocol}")

    def _host_turn(
        self,
        record: dict[str, Any],
        card: DialogueCard,
        card_run_id: str,
        repeat_index: int,
        step_index: int,
        instruction: str,
        step_key: str,
    ) -> dict[str, Any]:
        return {
            "turn_id": len(record["turns"]) + 1,
            "card_id": card.id,
            "card_title": card.title,
            "card_run_id": card_run_id,
            "repeat_index": repeat_index,
            "act": card.act,
            "card_type": card.card_type,
            "protocol": card.protocol,
            "step_index": step_index,
            "step_key": step_key,
            "speaker": "host",
            "speaker_label": self._speaker_label("host"),
            "model_id": None,
            "model_display_name": "Host",
            "instruction": instruction,
            "text": instruction,
            "dry_run": self.dry_run,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_model_step(
        self,
        record: dict[str, Any],
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard,
        card_run_id: str,
        repeat_index: int,
        step_index: int,
        speaker: str,
        instruction: str,
        step_key: str,
    ) -> dict[str, Any]:
        if speaker not in {"old", "new"}:
            raise ValueError(f"Card {card.id} has invalid speaker: {speaker}")

        model_config = old_model if speaker == "old" else new_model
        label = self._speaker_label(speaker)
        messages = self._build_messages(
            model_config=model_config,
            old_model=old_model,
            new_model=new_model,
            card=card,
            card_run_id=card_run_id,
            speaker=speaker,
            instruction=instruction,
            turns=record["turns"],
        )
        result = await self.client.complete(
            model_config,
            messages,
            metadata={
                "speaker": speaker,
                "card_id": card.id,
                "card_title": card.title,
                "card_type": card.card_type,
                "protocol": card.protocol,
                "step_key": step_key,
                "step_index": step_index,
            },
        )
        text = self._clean_content(result.content, label)
        return {
            "turn_id": len(record["turns"]) + 1,
            "card_id": card.id,
            "card_title": card.title,
            "card_run_id": card_run_id,
            "repeat_index": repeat_index,
            "act": card.act,
            "card_type": card.card_type,
            "protocol": card.protocol,
            "step_index": step_index,
            "step_key": step_key,
            "speaker": speaker,
            "speaker_label": label,
            "model_id": model_config.id,
            "model_display_name": model_config.display_name,
            "instruction": instruction,
            "text": text,
            "dry_run": result.dry_run,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _build_messages(
        self,
        model_config: ModelConfig,
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard,
        card_run_id: str,
        speaker: str,
        instruction: str,
        turns: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        values = SafeDict(
            cutoff_date=model_config.cutoff_date or "未知",
            old_cutoff_date=old_model.cutoff_date or "未知",
            old_model_name=old_model.display_name,
            new_model_name=new_model.display_name,
            old_label=self._speaker_label("old"),
            new_label=self._speaker_label("new"),
            host_label=self._speaker_label("host"),
        )
        template = self.old_system_template if speaker == "old" else self.new_system_template
        system_prompt = template.format_map(values)
        transcript = self._format_transcript(turns, card_run_id)
        user_prompt = (
            f"当前话题：{card.title}\n\n"
            f"对话记录：\n{transcript}\n\n"
            f"现在轮到{self._speaker_label(speaker)}说话。{instruction}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_transcript(self, turns: list[dict[str, Any]], card_run_id: str | None = None) -> str:
        visible_turns = [
            turn for turn in turns if card_run_id is None or turn.get("card_run_id") == card_run_id
        ]
        if not visible_turns:
            return "（当前卡片还没有对话）"
        chunks = [
            f"{turn['speaker_label']}：{turn['text']}"
            for turn in visible_turns
        ]
        transcript = "\n".join(chunks)
        max_chars = int(self.run_plan.get("max_transcript_chars", 8000))
        if len(transcript) <= max_chars:
            return transcript
        return "（前文已截断）\n" + transcript[-max_chars:]

    def _speaker_label(self, speaker: str) -> str:
        if speaker == "old":
            return self.run_plan.get("old_label", "旧AI")
        if speaker == "host":
            return self.run_plan.get("host_label", "主持人")
        return self.run_plan.get("new_label", "新AI")

    def _clean_content(self, content: str, label: str) -> str:
        text = content.strip()
        for prefix in (f"{label}：", f"{label}:", f"[{label}]", f"【{label}】"):
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return text

    def _session_id(self, old_model_id: str, new_model_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return safe_slug(f"{timestamp}_{new_model_id}_vs_{old_model_id}")

    def _repeat_count(self, card: DialogueCard, repeat_override: int | None) -> int:
        if repeat_override is not None:
            return max(1, repeat_override)
        if card.repeats is not None:
            return max(1, int(card.repeats))
        return max(1, int(self.run_plan.get("repeat_per_card", 1)))

    def _card_public_dict(self, card: DialogueCard) -> dict[str, Any]:
        return {
            "id": card.id,
            "title": card.title,
            "act": card.act,
            "card_type": card.card_type,
            "protocol": card.protocol,
            "host_injection": card.host_injection,
            "objective": card.objective,
            "evaluation_focus": card.evaluation_focus,
            "fact_sheet": card.fact_sheet,
            "fact_check_required": card.fact_check_required,
            "tags": card.tags,
            "repeats": card.repeats,
        }

    def _save_record(self, output_path: Path, record: dict[str, Any], force: bool = False) -> None:
        if force or self.run_plan.get("autosave", True):
            write_json(output_path, record)
