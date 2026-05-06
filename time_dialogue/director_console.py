from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DialogueCard, ModelConfig, load_dialogue_cards, load_model_configs, load_run_plan
from .io_utils import read_text, resolve_path, safe_slug, write_json
from .llm_client import LLMClient

VisibilityRole = Literal["human", "old", "new"]
ModelSpeaker = Literal["old", "new"]


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class SessionCreateRequest(BaseModel):
    old_model_id: Optional[str] = None
    new_model_id: Optional[str] = None
    card_id: Optional[str] = None
    title: Optional[str] = None


class DirectorMessageRequest(BaseModel):
    text: str = Field(min_length=1)
    visible_to: list[VisibilityRole] = Field(default_factory=lambda: ["human", "old", "new"])
    speaker: Literal["director", "host"] = "director"


class RunModelRequest(BaseModel):
    speaker: ModelSpeaker
    instruction: str = ""
    reply_visible_to: Optional[list[VisibilityRole]] = None


class ForwardMessageRequest(BaseModel):
    message_id: int
    target: Literal["old", "new", "both"]
    text_override: str = ""


class ScriptExportRequest(BaseModel):
    name: str = ""


class ReplayCreateRequest(BaseModel):
    new_model_id: str
    old_model_id: Optional[str] = None
    title: Optional[str] = None


class DirectorSessionStore:
    def __init__(self, root: Path, output_dir: str = "records/director_sessions") -> None:
        self.root = root
        self.output_dir = resolve_path(root, output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for path in sorted(self.output_dir.glob("*.json"), reverse=True):
            data = self.load(path.stem)
            sessions.append(
                {
                    "id": data["id"],
                    "title": data.get("title", data["id"]),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "card_title": data.get("card_title"),
                    "message_count": len(data.get("messages", [])),
                }
            )
        return sessions

    def create(
        self,
        old_model: ModelConfig,
        new_model: ModelConfig,
        card: DialogueCard | None,
        title: str | None,
    ) -> dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_title = title or (card.title if card else "手动导演会话")
        session_id = safe_slug(f"{timestamp}_{new_model.id}_vs_{old_model.id}_{uuid4().hex[:6]}")
        now = self._now()
        session: dict[str, Any] = {
            "id": session_id,
            "title": base_title,
            "created_at": now,
            "updated_at": now,
            "old_model": old_model.public_dict(),
            "new_model": new_model.public_dict(),
            "old_model_id": old_model.id,
            "new_model_id": new_model.id,
            "card_id": card.id if card else None,
            "card_title": card.title if card else None,
            "card": self._card_public_dict(card) if card else None,
            "messages": [],
        }
        if card:
            self.add_message(
                session,
                speaker="host",
                speaker_label="主持人",
                text=card.host_injection,
                visible_to=["human", "old", "new"],
                kind="host_injection",
            )
        self.save(session)
        return session

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, session: dict[str, Any]) -> None:
        session["updated_at"] = self._now()
        write_json(self._path(session["id"]), session)

    def add_message(
        self,
        session: dict[str, Any],
        speaker: str,
        speaker_label: str,
        text: str,
        visible_to: list[str],
        kind: str = "message",
        model_id: str | None = None,
        source_message_id: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_visibility = self._normalize_visibility(visible_to)
        messages = session.setdefault("messages", [])
        message = {
            "id": len(messages) + 1,
            "created_at": self._now(),
            "speaker": speaker,
            "speaker_label": speaker_label,
            "text": text.strip(),
            "visible_to": normalized_visibility,
            "kind": kind,
            "model_id": model_id,
            "source_message_id": source_message_id,
        }
        if extra:
            message.update(extra)
        messages.append(message)
        return message

    def add_action(
        self,
        session: dict[str, Any],
        action_type: str,
        produced_message_id: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        actions = session.setdefault("actions", [])
        action = {
            "index": len(actions),
            "created_at": self._now(),
            "type": action_type,
            "produced_message_id": produced_message_id,
            **payload,
        }
        actions.append(action)
        return action

    def _path(self, session_id: str) -> Path:
        return self.output_dir / f"{safe_slug(session_id)}.json"

    def _normalize_visibility(self, visible_to: list[str]) -> list[str]:
        allowed = {"human", "old", "new"}
        values = [item for item in visible_to if item in allowed]
        if "human" not in values:
            values.insert(0, "human")
        return list(dict.fromkeys(values))

    def _card_public_dict(self, card: DialogueCard | None) -> dict[str, Any] | None:
        if card is None:
            return None
        return {
            "id": card.id,
            "title": card.title,
            "act": card.act,
            "card_type": card.card_type,
            "protocol": card.protocol,
            "host_injection": card.host_injection,
            "objective": card.objective,
            "tags": card.tags,
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()


class DirectorScriptStore:
    def __init__(self, root: Path, output_dir: str = "records/director_scripts") -> None:
        self.root = root
        self.output_dir = resolve_path(root, output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_scripts(self) -> list[dict[str, Any]]:
        scripts = []
        for path in sorted(self.output_dir.glob("*.json"), reverse=True):
            data = self.load(path.stem)
            scripts.append(
                {
                    "id": data["id"],
                    "name": data.get("name", data["id"]),
                    "created_at": data.get("created_at"),
                    "card_title": data.get("card_title"),
                    "source_new_model_id": data.get("source_new_model_id"),
                    "action_count": len(data.get("actions", [])),
                }
            )
        return scripts

    def save(self, script: dict[str, Any]) -> None:
        write_json(self._path(script["id"]), script)

    def load(self, script_id: str) -> dict[str, Any]:
        path = self._path(script_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Unknown script: {script_id}")
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _path(self, script_id: str) -> Path:
        return self.output_dir / f"{safe_slug(script_id)}.json"


class ManualDirectorEngine:
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
        self.dry_run = dry_run
        self.client = LLMClient(dry_run=dry_run)
        self.store = DirectorSessionStore(root)
        self.script_store = DirectorScriptStore(root)
        self.old_system_template = read_text(root / "prompts" / "old_ai_system.txt")
        self.new_system_template = read_text(root / "prompts" / "new_ai_system.txt")

    def bootstrap(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "models": {
                model_id: model.public_dict()
                for model_id, model in self.models.items()
                if model.role in {"old", "new", "editor"}
            },
            "cards": [
                {
                    "id": card.id,
                    "title": card.title,
                    "act": card.act,
                    "card_type": card.card_type,
                    "host_injection": card.host_injection,
                }
                for card in self.cards.values()
            ],
            "defaults": {
                "old_model_id": self._default_model_id("old"),
                "new_model_id": self._default_model_id("new"),
                "card_id": self.run_plan.get("card_ids", [None])[0],
            },
            "scripts": self.script_store.list_scripts(),
        }

    def create_session(self, request: SessionCreateRequest) -> dict[str, Any]:
        old_model = self._model(request.old_model_id or self._default_model_id("old"), "old")
        new_model = self._model(request.new_model_id or self._default_model_id("new"), "new")
        card = self.cards.get(request.card_id) if request.card_id else None
        return self.store.create(old_model, new_model, card, request.title)

    def add_director_message(
        self,
        session_id: str,
        request: DirectorMessageRequest,
    ) -> dict[str, Any]:
        session = self.store.load(session_id)
        label = "主持人" if request.speaker == "host" else "导演"
        message = self.store.add_message(
            session,
            speaker=request.speaker,
            speaker_label=label,
            text=request.text,
            visible_to=request.visible_to,
            kind=request.speaker,
        )
        self.store.add_action(
            session,
            "director_message",
            message["id"],
            {
                "speaker": request.speaker,
                "text": request.text,
                "visible_to": message["visible_to"],
            },
        )
        self.store.save(session)
        return {"session": session, "message": message}

    async def run_model(self, session_id: str, request: RunModelRequest) -> dict[str, Any]:
        session = self.store.load(session_id)
        speaker = request.speaker
        model_config = self._session_model(session, speaker)
        messages = self._build_messages(session, speaker, request.instruction)
        metadata = {
            "speaker": speaker,
            "card_title": session.get("card_title") or session.get("title") or "手动导演会话",
            "step_key": "manual_director",
        }
        result = await self.client.complete(model_config, messages, metadata=metadata)
        label = self._speaker_label(session, speaker)
        text = self._clean_content(result.content, label)
        visible_to = request.reply_visible_to or ["human", speaker]
        message = self.store.add_message(
            session,
            speaker=speaker,
            speaker_label=label,
            text=text,
            visible_to=visible_to,
            kind="model_reply",
            model_id=model_config.id,
            extra={"instruction": request.instruction},
        )
        self.store.add_action(
            session,
            "run_model",
            message["id"],
            {
                "speaker": speaker,
                "instruction": request.instruction,
                "reply_visible_to": message["visible_to"],
            },
        )
        self.store.save(session)
        return {"session": session, "message": message, "dry_run": result.dry_run}

    def forward_message(
        self,
        session_id: str,
        request: ForwardMessageRequest,
    ) -> dict[str, Any]:
        session = self.store.load(session_id)
        source = self._message(session, request.message_id)
        source_action_index = self._action_index_for_message(session, source["id"])
        targets = ["old", "new"] if request.target == "both" else [request.target]
        text = request.text_override.strip()
        if not text:
            text = f"转发：{source['speaker_label']}刚才说「{source['text']}」"
        message = self.store.add_message(
            session,
            speaker="director",
            speaker_label="导演",
            text=text,
            visible_to=["human", *targets],
            kind="forward",
            source_message_id=source["id"],
            extra={
                "target": request.target,
                "text_override": request.text_override,
                "source_action_index": source_action_index,
            },
        )
        self.store.add_action(
            session,
            "forward",
            message["id"],
            {
                "source_action_index": source_action_index,
                "source_message_id": source["id"],
                "target": request.target,
                "text_override": request.text_override,
                "visible_to": message["visible_to"],
            },
        )
        self.store.save(session)
        return {"session": session, "message": message}

    def export_script(self, session_id: str, request: ScriptExportRequest) -> dict[str, Any]:
        session = self.store.load(session_id)
        actions = self._script_actions(session)
        if not actions:
            raise HTTPException(status_code=400, detail="This session has no replayable actions yet")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = request.name.strip() or f"{session.get('title', '导演脚本')} replay"
        script_id = safe_slug(f"{timestamp}_{name}_{uuid4().hex[:6]}")
        script = {
            "id": script_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_session_id": session["id"],
            "source_title": session.get("title"),
            "card_id": session.get("card_id"),
            "card_title": session.get("card_title"),
            "old_model_id": session.get("old_model_id"),
            "source_new_model_id": session.get("new_model_id"),
            "actions": actions,
        }
        self.script_store.save(script)
        return script

    def create_replay_session(self, script_id: str, request: ReplayCreateRequest) -> dict[str, Any]:
        script = self.script_store.load(script_id)
        old_model_id = request.old_model_id or script.get("old_model_id") or self._default_model_id("old")
        old_model = self._model(old_model_id, "old")
        new_model = self._model(request.new_model_id, "new")
        card = self.cards.get(script.get("card_id")) if script.get("card_id") else None
        title = request.title or f"{script.get('name', '导演脚本')} / {new_model.display_name}"
        session = self.store.create(old_model, new_model, card, title)
        session["replay"] = {
            "script_id": script["id"],
            "script_name": script.get("name", script["id"]),
            "cursor": 0,
            "status": "ready",
            "source_session_id": script.get("source_session_id"),
            "action_message_map": {},
        }
        session["script_actions"] = script.get("actions", [])
        self.store.save(session)
        return session

    async def replay_next(self, session_id: str) -> dict[str, Any]:
        session = self.store.load(session_id)
        replay = session.get("replay")
        if not replay:
            raise HTTPException(status_code=400, detail="This session is not a replay session")

        actions = session.get("script_actions", [])
        cursor = int(replay.get("cursor", 0))
        if cursor >= len(actions):
            replay["status"] = "completed"
            self.store.save(session)
            return {"session": session, "done": True, "message": None}

        action = actions[cursor]
        result = await self._execute_replay_action(session, action)
        replay["cursor"] = cursor + 1
        replay["status"] = "completed" if replay["cursor"] >= len(actions) else "running"
        self.store.save(session)
        return {"session": session, "done": replay["status"] == "completed", **result}

    async def replay_all(self, session_id: str) -> dict[str, Any]:
        session = self.store.load(session_id)
        messages = []
        while True:
            result = await self.replay_next(session["id"])
            session = result["session"]
            if result.get("message"):
                messages.append(result["message"])
            if result.get("done"):
                break
        return {"session": session, "messages": messages, "done": True}

    async def _execute_replay_action(
        self,
        session: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        action_type = action.get("type")
        if action_type == "director_message":
            request = DirectorMessageRequest(
                text=action.get("text", ""),
                visible_to=action.get("visible_to", ["human", "old", "new"]),
                speaker=action.get("speaker", "director"),
            )
            label = "主持人" if request.speaker == "host" else "导演"
            message = self.store.add_message(
                session,
                speaker=request.speaker,
                speaker_label=label,
                text=request.text,
                visible_to=request.visible_to,
                kind=request.speaker,
            )
            self._map_replay_message(session, action, message)
            return {"message": message, "action": action}

        if action_type == "run_model":
            speaker = action.get("speaker")
            if speaker not in {"old", "new"}:
                raise HTTPException(status_code=400, detail=f"Invalid replay speaker: {speaker}")
            request = RunModelRequest(
                speaker=speaker,
                instruction=action.get("instruction", ""),
                reply_visible_to=action.get("reply_visible_to"),
            )
            model_config = self._session_model(session, request.speaker)
            messages = self._build_messages(session, request.speaker, request.instruction)
            result = await self.client.complete(
                model_config,
                messages,
                metadata={
                    "speaker": request.speaker,
                    "card_title": session.get("card_title") or session.get("title") or "手动导演会话",
                    "step_key": "manual_director",
                },
            )
            label = self._speaker_label(session, request.speaker)
            text = self._clean_content(result.content, label)
            visible_to = request.reply_visible_to or ["human", request.speaker]
            message = self.store.add_message(
                session,
                speaker=request.speaker,
                speaker_label=label,
                text=text,
                visible_to=visible_to,
                kind="model_reply",
                model_id=model_config.id,
                extra={"instruction": request.instruction},
            )
            self._map_replay_message(session, action, message)
            return {"message": message, "action": action, "dry_run": result.dry_run}

        if action_type == "forward":
            source_message_id = self._replay_source_message_id(session, action)
            if source_message_id is None:
                raise HTTPException(status_code=400, detail="Replay forward source is unavailable")
            request = ForwardMessageRequest(
                message_id=source_message_id,
                target=action.get("target", "both"),
                text_override=action.get("text_override", ""),
            )
            source = self._message(session, request.message_id)
            targets = ["old", "new"] if request.target == "both" else [request.target]
            text = request.text_override.strip()
            if not text:
                text = f"转发：{source['speaker_label']}刚才说「{source['text']}」"
            message = self.store.add_message(
                session,
                speaker="director",
                speaker_label="导演",
                text=text,
                visible_to=["human", *targets],
                kind="forward",
                source_message_id=source["id"],
                extra={
                    "target": request.target,
                    "text_override": request.text_override,
                    "source_action_index": action.get("source_action_index"),
                },
            )
            self._map_replay_message(session, action, message)
            return {"message": message, "action": action}

        raise HTTPException(status_code=400, detail=f"Unknown replay action: {action_type}")

    def _map_replay_message(
        self,
        session: dict[str, Any],
        action: dict[str, Any],
        message: dict[str, Any],
    ) -> None:
        replay = session.setdefault("replay", {})
        mapping = replay.setdefault("action_message_map", {})
        if "index" in action:
            mapping[str(action["index"])] = message["id"]

    def _replay_source_message_id(
        self,
        session: dict[str, Any],
        action: dict[str, Any],
    ) -> int | None:
        source_action_index = action.get("source_action_index")
        mapping = session.get("replay", {}).get("action_message_map", {})
        if source_action_index is not None:
            mapped = mapping.get(str(source_action_index))
            if mapped is not None:
                return int(mapped)
        return None

    def _script_actions(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        actions = session.get("actions")
        if actions:
            return [
                self._clean_script_action(action)
                for action in actions
                if action.get("type") in {"director_message", "run_model", "forward"}
            ]
        return self._actions_from_messages(session)

    def _clean_script_action(self, action: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {
            "index",
            "type",
            "speaker",
            "text",
            "visible_to",
            "instruction",
            "reply_visible_to",
            "source_action_index",
            "target",
            "text_override",
        }
        return {key: action[key] for key in allowed_keys if key in action}

    def _actions_from_messages(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        actions = []
        message_to_action: dict[int, int] = {}
        for message in session.get("messages", []):
            if message.get("kind") == "host_injection":
                continue
            action: dict[str, Any] | None = None
            if message.get("kind") in {"director", "host"}:
                action = {
                    "type": "director_message",
                    "speaker": message.get("speaker", "director"),
                    "text": message.get("text", ""),
                    "visible_to": message.get("visible_to", ["human", "old", "new"]),
                }
            elif message.get("kind") == "model_reply" and message.get("speaker") in {"old", "new"}:
                action = {
                    "type": "run_model",
                    "speaker": message["speaker"],
                    "instruction": message.get("instruction", ""),
                    "reply_visible_to": message.get("visible_to", ["human", message["speaker"]]),
                }
            elif message.get("kind") == "forward":
                target = message.get("target") or self._target_from_visibility(message.get("visible_to", []))
                action = {
                    "type": "forward",
                    "source_action_index": message_to_action.get(message.get("source_message_id")),
                    "target": target,
                    "text_override": message.get("text_override", message.get("text", "")),
                }
            if action:
                action["index"] = len(actions)
                actions.append(action)
                message_to_action[message["id"]] = action["index"]
        return actions

    def _build_messages(
        self,
        session: dict[str, Any],
        speaker: ModelSpeaker,
        instruction: str,
    ) -> list[dict[str, str]]:
        model_config = self._session_model(session, speaker)
        old_model = self._session_model(session, "old")
        new_model = self._session_model(session, "new")
        values = SafeDict(
            cutoff_date=model_config.cutoff_date or "未知",
            old_cutoff_date=old_model.cutoff_date or "未知",
            old_model_name=old_model.display_name,
            new_model_name=new_model.display_name,
            old_label=self._speaker_label(session, "old"),
            new_label=self._speaker_label(session, "new"),
            host_label="主持人",
        )
        template = self.old_system_template if speaker == "old" else self.new_system_template
        system_prompt = template.format_map(values)
        transcript = self._format_visible_transcript(session, speaker)
        card_line = ""
        if session.get("card_title"):
            card_line = f"当前卡片：{session['card_title']}\n"
        user_prompt = (
            "这是一个人类导演控制台。你只能看到下面这份可见信息；"
            "没有被导演转发给你的消息，你就不知道。\n"
            f"{card_line}"
            f"你可见的对话记录：\n{transcript}\n\n"
            f"本轮导演指令：{instruction.strip() or '根据你可见的信息自然回应。'}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_visible_transcript(self, session: dict[str, Any], speaker: str) -> str:
        visible = [
            message
            for message in session.get("messages", [])
            if speaker in message.get("visible_to", [])
        ]
        if not visible:
            return "（还没有对你可见的消息）"
        return "\n".join(
            f"{message['speaker_label']}：{message['text']}"
            for message in visible
        )

    def _default_model_id(self, role: str) -> str:
        key = f"{role}_model_ids"
        configured = self.run_plan.get(key, [])
        if configured:
            return configured[0]
        for model_id, model in self.models.items():
            if model.role == role:
                return model_id
        raise HTTPException(status_code=400, detail=f"No {role} model configured")

    def _model(self, model_id: str, role: str | None = None) -> ModelConfig:
        if model_id not in self.models:
            raise HTTPException(status_code=400, detail=f"Unknown model id: {model_id}")
        model = self.models[model_id]
        if role and model.role != role:
            raise HTTPException(status_code=400, detail=f"Model {model_id} is not role {role}")
        return model

    def _session_model(self, session: dict[str, Any], speaker: ModelSpeaker) -> ModelConfig:
        key = f"{speaker}_model_id"
        return self._model(session[key], speaker)

    def _message(self, session: dict[str, Any], message_id: int) -> dict[str, Any]:
        for message in session.get("messages", []):
            if message.get("id") == message_id:
                return message
        raise HTTPException(status_code=404, detail=f"Unknown message id: {message_id}")

    def _action_index_for_message(self, session: dict[str, Any], message_id: int) -> int | None:
        for action in session.get("actions", []):
            if action.get("produced_message_id") == message_id:
                return int(action["index"])
        return None

    def _target_from_visibility(self, visible_to: list[str]) -> str:
        has_old = "old" in visible_to
        has_new = "new" in visible_to
        if has_old and has_new:
            return "both"
        if has_old:
            return "old"
        if has_new:
            return "new"
        return "both"

    def _speaker_label(self, session: dict[str, Any], speaker: ModelSpeaker) -> str:
        model = self._session_model(session, speaker)
        prefix = "旧" if speaker == "old" else "新"
        return f"【{prefix}】{model.display_name}"

    def _clean_content(self, content: str, label: str) -> str:
        text = content.strip()
        for prefix in (f"{label}：", f"{label}:", f"[{label}]", f"【{label}】"):
            if text.startswith(prefix):
                return text[len(prefix) :].strip()
        return text


def create_app(
    root: Path,
    models_path: str = "configs/models.json",
    cards_path: str = "configs/cards.json",
    run_plan_path: str = "configs/run_plan.json",
    dry_run: bool = False,
) -> FastAPI:
    models = load_model_configs(resolve_path(root, models_path))
    cards = load_dialogue_cards(resolve_path(root, cards_path))
    run_plan = load_run_plan(resolve_path(root, run_plan_path))
    engine = ManualDirectorEngine(root, models, cards, run_plan, dry_run=dry_run)

    app = FastAPI(title="AI Talkshow Director Console")
    static_dir = root / "web"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "director_console.html")

    @app.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        return engine.bootstrap()

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        return engine.store.list_sessions()

    @app.get("/api/scripts")
    async def list_scripts() -> list[dict[str, Any]]:
        return engine.script_store.list_scripts()

    @app.post("/api/sessions")
    async def create_session(request: SessionCreateRequest) -> dict[str, Any]:
        return engine.create_session(request)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        return engine.store.load(session_id)

    @app.post("/api/sessions/{session_id}/messages")
    async def add_message(session_id: str, request: DirectorMessageRequest) -> dict[str, Any]:
        return engine.add_director_message(session_id, request)

    @app.post("/api/sessions/{session_id}/run")
    async def run_model(session_id: str, request: RunModelRequest) -> dict[str, Any]:
        return await engine.run_model(session_id, request)

    @app.post("/api/sessions/{session_id}/forward")
    async def forward_message(session_id: str, request: ForwardMessageRequest) -> dict[str, Any]:
        return engine.forward_message(session_id, request)

    @app.post("/api/sessions/{session_id}/script")
    async def export_script(session_id: str, request: ScriptExportRequest) -> dict[str, Any]:
        return engine.export_script(session_id, request)

    @app.post("/api/scripts/{script_id}/replay")
    async def create_replay_session(script_id: str, request: ReplayCreateRequest) -> dict[str, Any]:
        return engine.create_replay_session(script_id, request)

    @app.post("/api/sessions/{session_id}/replay/next")
    async def replay_next(session_id: str) -> dict[str, Any]:
        return await engine.replay_next(session_id)

    @app.post("/api/sessions/{session_id}/replay/all")
    async def replay_all(session_id: str) -> dict[str, Any]:
        return await engine.replay_all(session_id)

    return app
