from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import load_json


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    model: str
    role: str
    cutoff_date: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "LLM_URL"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_id: str, data: dict[str, Any]) -> "ModelConfig":
        known_keys = {
            "display_name",
            "model",
            "role",
            "cutoff_date",
            "api_key_env",
            "base_url_env",
            "api_key",
            "base_url",
            "temperature",
            "max_tokens",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            id=model_id,
            display_name=data.get("display_name", model_id),
            model=data["model"],
            role=data.get("role", "new"),
            cutoff_date=data.get("cutoff_date"),
            api_key_env=data.get("api_key_env", "OPENAI_API_KEY"),
            base_url_env=data.get("base_url_env", "LLM_URL"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            extra=extra,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "model": self.model,
            "role": self.role,
            "cutoff_date": self.cutoff_date,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class DialogueCard:
    id: str
    title: str
    act: str
    card_type: str
    protocol: str
    host_injection: str
    objective: str = ""
    evaluation_focus: list[str] = field(default_factory=list)
    fact_sheet: dict[str, Any] = field(default_factory=dict)
    fact_check_required: bool = True
    tags: list[str] = field(default_factory=list)
    repeats: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DialogueCard":
        extra = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "id",
                "title",
                "act",
                "card_type",
                "protocol",
                "host_injection",
                "objective",
                "evaluation_focus",
                "fact_sheet",
                "fact_check_required",
                "tags",
                "repeats",
            }
        }
        return cls(
            id=data["id"],
            title=data["title"],
            act=data.get("act", ""),
            card_type=data.get("card_type", "unknown_event"),
            protocol=data.get("protocol", "unknown_event_exam_v1"),
            host_injection=data["host_injection"],
            objective=data.get("objective", ""),
            evaluation_focus=list(data.get("evaluation_focus", [])),
            fact_sheet=dict(data.get("fact_sheet", {})),
            fact_check_required=bool(data.get("fact_check_required", False)),
            tags=list(data.get("tags", [])),
            repeats=data.get("repeats"),
            extra=extra,
        )


def load_model_configs(path: Path) -> dict[str, ModelConfig]:
    raw = load_json(path)
    return {
        model_id: ModelConfig.from_dict(model_id, data)
        for model_id, data in raw.get("models", {}).items()
    }


def load_dialogue_cards(path: Path) -> dict[str, DialogueCard]:
    raw = load_json(path)
    return {
        item["id"]: DialogueCard.from_dict(item)
        for item in raw.get("cards", raw.get("topics", []))
    }


def load_topic_cards(path: Path) -> dict[str, DialogueCard]:
    return load_dialogue_cards(path)


def load_run_plan(path: Path) -> dict[str, Any]:
    return load_json(path)


def select_cards(
    cards: dict[str, DialogueCard],
    card_ids: list[str],
    limit: int | None = None,
) -> list[DialogueCard]:
    selected: list[DialogueCard] = []
    for card_id in card_ids:
        if card_id not in cards:
            raise KeyError(f"Unknown card id: {card_id}")
        selected.append(cards[card_id])
    if limit is not None:
        selected = selected[:limit]
    return selected


def select_topics(
    topics: dict[str, DialogueCard],
    topic_ids: list[str],
    limit: int | None = None,
) -> list[DialogueCard]:
    return select_cards(topics, topic_ids, limit)
