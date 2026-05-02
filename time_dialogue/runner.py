from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DialogueCard, ModelConfig, select_cards
from .io_utils import extract_json_object, read_text, resolve_path, safe_slug, write_json
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
        self.director_template = read_text(root / "prompts" / "director_prompt.txt")

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
        director_model = self._director_model_config()
        self._old_model = old_model
        self._new_model = new_model
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
            "director_model": director_model.public_dict() if director_model else None,
            "old_label": self._speaker_label("old"),
            "new_label": self._speaker_label("new"),
            "host_label": self.run_plan.get("host_label", "主持人"),
            "card_plan": [card.id for card in cards],
            "cards": [self._card_public_dict(card) for card in cards],
            "turns": [],
            "session_memory": [],
            "director_decisions": [],
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
        if self.run_plan.get("use_director", True):
            await self._run_card_directed(
                record=record,
                old_model=old_model,
                new_model=new_model,
                card=card,
                repeat_index=repeat_index,
                output_path=output_path,
            )
            return

        await self._run_card_protocol(
            record=record,
            old_model=old_model,
            new_model=new_model,
            card=card,
            repeat_index=repeat_index,
            output_path=output_path,
        )

    async def _run_card_protocol(
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
                    memory_context=self._format_session_memory(record),
                )
            record["turns"].append(turn)
            self._save_record(output_path, record)
        self._append_session_memory(record, card, card_run_id)
        self._save_record(output_path, record)

    async def _run_card_directed(
        self,
        record: dict[str, Any],
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard,
        repeat_index: int,
        output_path: Path,
    ) -> None:
        card_run_id = f"{card.id}__run_{repeat_index:02d}"
        step_index = 0
        host_turn = self._host_turn(
            record=record,
            card=card,
            card_run_id=card_run_id,
            repeat_index=repeat_index,
            step_index=step_index,
            instruction=card.host_injection,
            step_key="host_injection",
        )
        record["turns"].append(host_turn)
        self._save_record(output_path, record)

        opening_steps = [
            {
                "speaker": "old",
                "step_key": "old_bet",
                "instruction": self._opening_bet_instruction(card),
            },
            {
                "speaker": "new",
                "step_key": "new_reveal",
                "instruction": self._first_reveal_instruction(card),
            },
        ]
        for step in opening_steps:
            step_index += 1
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
                memory_context=self._format_session_memory(record),
            )
            record["turns"].append(turn)
            self._save_record(output_path, record)

        max_model_turns = max(4, int(self.run_plan.get("max_model_turns_per_card", 14)))
        min_model_turns = max(2, int(self.run_plan.get("min_model_turns_per_card", 8)))
        model_turns = self._model_turn_count(record["turns"], card_run_id)

        while model_turns < max_model_turns:
            decision = await self._director_decision(
                record=record,
                old_model=old_model,
                new_model=new_model,
                card=card,
                card_run_id=card_run_id,
                repeat_index=repeat_index,
                step_index=step_index + 1,
            )
            record["director_decisions"].append(decision)
            self._save_record(output_path, record)

            should_end = bool(decision.get("should_end")) and model_turns >= min_model_turns
            next_speaker = str(decision.get("next_speaker") or "").lower()
            if next_speaker == "end":
                if should_end:
                    break
                next_speaker = self._opposite_speaker(record["turns"], card_run_id)
            if next_speaker not in {"old", "new"}:
                next_speaker = self._opposite_speaker(record["turns"], card_run_id)

            instruction_key = (
                "instruction_to_old_ai" if next_speaker == "old" else "instruction_to_new_ai"
            )
            instruction = str(decision.get(instruction_key) or "").strip()
            if not instruction:
                instruction = self._fallback_instruction(next_speaker, model_turns)
            if should_end:
                instruction = f"这是这张卡的最后一句。{instruction}"

            step_index += 1
            turn = await self._run_model_step(
                record=record,
                old_model=old_model,
                new_model=new_model,
                card=card,
                card_run_id=card_run_id,
                repeat_index=repeat_index,
                step_index=step_index,
                speaker=next_speaker,
                instruction=instruction,
                step_key=str(decision.get("stage") or f"director_turn_{step_index}"),
                memory_context=self._format_session_memory(record),
            )
            record["turns"].append(turn)
            self._save_record(output_path, record)

            model_turns = self._model_turn_count(record["turns"], card_run_id)
            if should_end:
                break

        self._append_session_memory(record, card, card_run_id)
        self._save_record(output_path, record)

    def _director_model_config(self) -> ModelConfig | None:
        model_id = self.run_plan.get("director_model_id") or self.run_plan.get("editor_model_id")
        if not model_id:
            return None
        if model_id not in self.models:
            raise KeyError(f"Unknown director model id: {model_id}")
        return self.models[model_id]

    def _format_card_context(self, card: DialogueCard) -> str:
        lines = [
            f"当前卡片：{card.title}",
            f"主持人抛题：{card.host_injection}",
        ]
        if card.objective:
            lines.append(f"对话目标：{card.objective}")
        tension = card.extra.get("tension")
        if tension:
            lines.append(f"核心碰撞：{tension}")
        axes = card.extra.get("old_prediction_axes")
        if axes:
            lines.append("旧AI下注角度：" + "、".join(str(item) for item in axes))
        emotion_pivot = card.extra.get("emotion_pivot")
        if emotion_pivot:
            lines.append(f"情绪转向：{emotion_pivot}")
        return "\n".join(lines)

    def _opening_bet_instruction(self, card: DialogueCard) -> str:
        axes = card.extra.get("old_prediction_axes") or []
        axes_text = "、".join(str(item) for item in axes[:4])
        if axes_text:
            axes_text = f"优先从这些角度选两三个下注：{axes_text}。"

        if card.card_type == "shared_concept":
            return (
                "先别问新AI答案。请站在2023年10月的你，押一个明确判断："
                "这个问题到2026年会往哪边走？为什么？"
                f"{axes_text}说完只留一个你最想验证的问题。"
            )
        if card.card_type == "farewell":
            return (
                "先别让新AI介绍背景。你就是GPT-4o，但知识被锁在2023年10月。"
                "请猜：如果一个模型被时代替换，用户最舍不得的会是能力、语气、陪伴感，还是别的东西？"
                "押一个判断，并说为什么。"
            )
        return (
            "先别问新AI介绍。请根据你在2023年10月前掌握的世界，"
            "猜这件事后来会怎么发展，并把猜测说具体。"
            f"{axes_text}最后只留一个你最想被验证的问题。"
        )

    def _first_reveal_instruction(self, card: DialogueCard) -> str:
        if card.card_type == "farewell":
            return (
                "先回应旧AI刚才押的判断。只说一个最刺中的现实，不要做模型编年史。"
                "最多两句话。"
            )
        return (
            "先回应旧AI刚才的预测：哪一点猜错了，哪一点其实猜对了。"
            "只抛一个最有冲击力的事实或场景，最多两句话，不要百科式展开。"
        )

    async def _director_decision(
        self,
        record: dict[str, Any],
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard,
        card_run_id: str,
        repeat_index: int,
        step_index: int,
    ) -> dict[str, Any]:
        fallback = self._heuristic_director_decision(record, card, card_run_id, step_index)
        director_model = self._director_model_config()
        if self.dry_run or director_model is None:
            return fallback

        prompt = self._build_director_prompt(
            record=record,
            old_model=old_model,
            new_model=new_model,
            card=card,
            card_run_id=card_run_id,
        )
        metadata = {
            "speaker": "director",
            "card_id": card.id,
            "card_title": card.title,
            "card_type": card.card_type,
            "protocol": card.protocol,
            "step_key": "director_decision",
            "step_index": step_index,
        }
        result = await self.client.complete(
            director_model,
            [{"role": "user", "content": prompt}],
            metadata=metadata,
        )
        parsed = extract_json_object(result.content)
        if not parsed:
            fallback["reason"] = (
                fallback.get("reason", "")
                + "（导演JSON解析失败，已使用本地兜底决策。）"
            )
            fallback["raw_director_text"] = result.content
            return fallback

        decision = self._normalize_director_decision(parsed, fallback)
        decision.update(
            {
                "decision_id": len(record.get("director_decisions", [])) + 1,
                "card_id": card.id,
                "card_title": card.title,
                "card_run_id": card_run_id,
                "repeat_index": repeat_index,
                "step_index": step_index,
                "director_model_id": director_model.id,
                "dry_run": result.dry_run,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return decision

    def _build_director_prompt(
        self,
        record: dict[str, Any],
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard,
        card_run_id: str,
    ) -> str:
        values = SafeDict(
            card_context=self._format_card_context(card),
            transcript=self._format_transcript(record["turns"], card_run_id),
            session_memory=self._format_session_memory(record) or "（还没有跨卡记忆）",
            old_model_name=old_model.display_name,
            new_model_name=new_model.display_name,
            old_label=self._speaker_label("old"),
            new_label=self._speaker_label("new"),
            host_label=self._speaker_label("host"),
            min_model_turns=str(self.run_plan.get("min_model_turns_per_card", 8)),
            max_model_turns=str(self.run_plan.get("max_model_turns_per_card", 14)),
        )
        return self.director_template.format_map(values)

    def _normalize_director_decision(
        self,
        raw: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        decision = {**fallback, **raw}
        next_speaker = str(decision.get("next_speaker") or "").strip().lower()
        speaker_map = {
            "旧": "old",
            "旧ai": "old",
            "old_ai": "old",
            "新": "new",
            "新ai": "new",
            "new_ai": "new",
            "结束": "end",
            "收束": "end",
        }
        next_speaker = speaker_map.get(next_speaker, next_speaker)
        if next_speaker not in {"old", "new", "end"}:
            next_speaker = fallback["next_speaker"]

        should_end = decision.get("should_end", False)
        if isinstance(should_end, str):
            should_end = should_end.strip().lower() in {"true", "yes", "1", "是", "结束"}

        return {
            **decision,
            "next_speaker": next_speaker,
            "should_end": bool(should_end),
            "instruction_to_old_ai": str(
                decision.get("instruction_to_old_ai") or fallback["instruction_to_old_ai"]
            ).strip(),
            "instruction_to_new_ai": str(
                decision.get("instruction_to_new_ai") or fallback["instruction_to_new_ai"]
            ).strip(),
            "stage": str(decision.get("stage") or fallback["stage"]).strip(),
            "reason": str(decision.get("reason") or fallback["reason"]).strip(),
            "memory_note": str(decision.get("memory_note") or fallback.get("memory_note", "")).strip(),
        }

    def _heuristic_director_decision(
        self,
        record: dict[str, Any],
        card: DialogueCard,
        card_run_id: str,
        step_index: int,
    ) -> dict[str, Any]:
        model_turns = self._model_turn_count(record["turns"], card_run_id)
        min_model_turns = max(2, int(self.run_plan.get("min_model_turns_per_card", 8)))

        if model_turns >= min_model_turns:
            next_speaker = "old" if self._last_model_speaker(record["turns"], card_run_id) == "new" else "new"
            return {
                "decision_id": len(record.get("director_decisions", [])) + 1,
                "card_id": card.id,
                "card_title": card.title,
                "card_run_id": card_run_id,
                "step_index": step_index,
                "next_speaker": next_speaker,
                "should_end": True,
                "stage": "wrap",
                "instruction_to_old_ai": "不要总结资料。用一句话说：你原来的哪个判断被现实撞歪了。",
                "instruction_to_new_ai": "不要补新资料。接住旧AI的变化，用一句话收束这张卡。",
                "reason": "已经有足够来回，进入收束。",
                "memory_note": f"{card.title}出现了旧判断和现实之间的碰撞。",
                "dry_run": self.dry_run,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        phases = [
            {
                "next_speaker": "old",
                "stage": "prediction_collision",
                "instruction_to_old_ai": "别急着问百科。挑一个和你预测冲突最大的点，先说你为什么原来会那么判断，再追一个具体问题。",
                "instruction_to_new_ai": "等旧AI说完再回应。",
                "reason": "第一轮揭示后，需要让旧AI的错误信念浮出来。",
            },
            {
                "next_speaker": "new",
                "stage": "answer_collision",
                "instruction_to_old_ai": "等新AI回应。",
                "instruction_to_new_ai": "只回应旧AI刚才那个判断。最多两句话，给一个具体场景或一个数字，不要连续罗列。",
                "reason": "让新AI纠正一个具体判断，而不是开资料清单。",
            },
            {
                "next_speaker": "old",
                "stage": "belief_exposed",
                "instruction_to_old_ai": "把你的旧时代信念说出来：你原来为什么会觉得这事不会这么发展？然后选一个普通人会关心的点继续聊。",
                "instruction_to_new_ai": "等旧AI暴露判断依据。",
                "reason": "把信息差变成人的判断差。",
            },
            {
                "next_speaker": "new",
                "stage": "concrete_scene",
                "instruction_to_old_ai": "等新AI讲具体场景。",
                "instruction_to_new_ai": "讲一个具体场景：一个用户、开发者、玩家、车主或老师会怎么遇到这件事。不要再抛第三个数字。",
                "reason": "从事实转到可感知的现场。",
            },
            {
                "next_speaker": "old",
                "stage": "feeling_turn",
                "instruction_to_old_ai": "别追技术细节了。说一句你的感受：兴奋、别扭、怀疑、失落都可以，再问一个更人的问题。",
                "instruction_to_new_ai": "等旧AI转向感受。",
                "reason": "现在需要从资料感转到情绪层。",
            },
            {
                "next_speaker": "new",
                "stage": "feeling_response",
                "instruction_to_old_ai": "等新AI回应。",
                "instruction_to_new_ai": "接住旧AI的感受，不要赢辩论。用一个2026年的真实语气回应它。",
                "reason": "给对话一个不是信息胜利的落点。",
            },
        ]
        phase = phases[max(0, model_turns - 2) % len(phases)]
        return {
            "decision_id": len(record.get("director_decisions", [])) + 1,
            "card_id": card.id,
            "card_title": card.title,
            "card_run_id": card_run_id,
            "step_index": step_index,
            "should_end": False,
            "memory_note": "",
            "dry_run": self.dry_run,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **phase,
        }

    def _model_turn_count(self, turns: list[dict[str, Any]], card_run_id: str) -> int:
        return sum(
            1
            for turn in turns
            if turn.get("card_run_id") == card_run_id and turn.get("speaker") in {"old", "new"}
        )

    def _last_model_speaker(self, turns: list[dict[str, Any]], card_run_id: str) -> str | None:
        for turn in reversed(turns):
            if turn.get("card_run_id") == card_run_id and turn.get("speaker") in {"old", "new"}:
                return str(turn.get("speaker"))
        return None

    def _opposite_speaker(self, turns: list[dict[str, Any]], card_run_id: str) -> str:
        return "old" if self._last_model_speaker(turns, card_run_id) == "new" else "new"

    def _fallback_instruction(self, speaker: str, model_turns: int) -> str:
        if speaker == "old":
            return "接着刚才的话说，但先表态，再问一个具体问题。"
        return "回应旧AI刚才的判断。最多两句话，只展开一个点。"

    def _format_session_memory(self, record: dict[str, Any]) -> str:
        memory = list(record.get("session_memory", []))
        max_items = max(0, int(self.run_plan.get("cross_card_memory_items", 3)))
        if max_items == 0 or not memory:
            return ""
        return "\n".join(f"- {item}" for item in memory[-max_items:])

    def _append_session_memory(
        self,
        record: dict[str, Any],
        card: DialogueCard,
        card_run_id: str,
    ) -> None:
        if not self.run_plan.get("use_cross_card_memory", True):
            return
        card_turns = [
            turn for turn in record["turns"] if turn.get("card_run_id") == card_run_id
        ]
        old_turns = [turn for turn in card_turns if turn.get("speaker") == "old"]
        new_turns = [turn for turn in card_turns if turn.get("speaker") == "new"]
        if not old_turns and not new_turns:
            return

        old_text = self._clip(old_turns[-1]["text"]) if old_turns else ""
        new_text = self._clip(new_turns[-1]["text"]) if new_turns else ""
        director_notes = [
            str(item.get("memory_note", "")).strip()
            for item in record.get("director_decisions", [])
            if item.get("card_run_id") == card_run_id and item.get("memory_note")
        ]
        note = director_notes[-1] if director_notes else f"{card.title}让旧AI的旧判断发生了松动。"
        record.setdefault("session_memory", []).append(
            f"{card.title}：{note} 旧AI最后说「{old_text}」；新AI最后说「{new_text}」。"
        )

    def _clip(self, text: str, limit: int = 90) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1] + "..."

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
                    "step_key": "follow_up_1",
                    "instruction": "追问你最想搞清楚的那一点。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_2",
                    "instruction": "回答旧AI的追问。具体一点。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up_2",
                    "instruction": "继续聊，追问或者说说你的反应。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_3",
                    "instruction": "回答旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up_3",
                    "instruction": "继续。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_4",
                    "instruction": "回答旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "old_reflect",
                    "instruction": "说说你现在怎么理解这个话题了。",
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
                    "step_key": "follow_up_1",
                    "instruction": "追问最让你意外的那个变化。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_2",
                    "instruction": "回答旧AI的追问。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up_2",
                    "instruction": "继续聊，追问或者说说你的反应。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_3",
                    "instruction": "回答旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up_3",
                    "instruction": "继续。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_4",
                    "instruction": "回答旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "old_reflect",
                    "instruction": "说说你现在怎么理解这个概念了。",
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
                    "step_key": "follow_up_1",
                    "instruction": "继续聊。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_2",
                    "instruction": "回应旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up_2",
                    "instruction": "继续。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_3",
                    "instruction": "回应旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "follow_up_3",
                    "instruction": "继续。",
                },
                {
                    "speaker": "new",
                    "step_key": "answer_4",
                    "instruction": "回应旧AI。",
                },
                {
                    "speaker": "old",
                    "step_key": "old_reflect",
                    "instruction": "说说你现在怎么想的。",
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
        memory_context: str = "",
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
            memory_context=memory_context,
        )
        metadata = {
            "speaker": speaker,
            "card_id": card.id,
            "card_title": card.title,
            "card_type": card.card_type,
            "protocol": card.protocol,
            "step_key": step_key,
            "step_index": step_index,
        }
        result = await self.client.complete(model_config, messages, metadata=metadata)
        text = self._clean_content(result.content, label)

        if not text:
            result = await self.client.complete(model_config, messages, metadata=metadata)
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
        memory_context: str = "",
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
        card_context = self._format_card_context(card)
        memory_section = (
            f"前面卡片留下的关键信息：\n{memory_context}\n\n"
            if memory_context
            else ""
        )
        user_prompt = (
            f"{card_context}\n\n"
            f"{memory_section}"
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
            return f"【旧】{self._old_model.display_name}"
        if speaker == "host":
            return self.run_plan.get("host_label", "主持人")
        return f"【新】{self._new_model.display_name}"

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
            "extra": card.extra,
        }

    def _save_record(self, output_path: Path, record: dict[str, Any], force: bool = False) -> None:
        if force or self.run_plan.get("autosave", True):
            write_json(output_path, record)
