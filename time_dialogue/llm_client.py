from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    raw: Any | None = None
    dry_run: bool = False


class LLMClient:
    def __init__(self, dry_run: bool = False, retries: int = 2) -> None:
        self.dry_run = dry_run
        self.retries = retries
        self._clients: dict[tuple[str, str], Any] = {}

    async def complete(
        self,
        model_config: ModelConfig,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> LLMResult:
        if self.dry_run:
            return LLMResult(
                content=self._dry_content(model_config, metadata or {}),
                model=model_config.model,
                dry_run=True,
            )

        client = self._get_client(model_config)
        payload: dict[str, Any] = {
            "model": model_config.model,
            "messages": messages,
        }
        if model_config.temperature is not None:
            payload["temperature"] = model_config.temperature
        if model_config.max_tokens is not None:
            payload["max_tokens"] = model_config.max_tokens

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await client.chat.completions.create(**payload)
                content = response.choices[0].message.content or ""
                return LLMResult(content=content.strip(), model=model_config.model, raw=response)
            except Exception as exc:  # pragma: no cover - depends on remote provider.
                last_error = exc
                message = str(exc).lower()
                if attempt == 0 and (
                    "temperature" in message
                    or "max_tokens" in message
                    or "unsupported parameter" in message
                ):
                    payload.pop("temperature", None)
                    payload.pop("max_tokens", None)
                    continue
        raise RuntimeError(f"LLM request failed for {model_config.id}: {last_error}")

    def _get_client(self, model_config: ModelConfig) -> Any:
        api_key = model_config.api_key or os.getenv(model_config.api_key_env)
        base_url = model_config.base_url or os.getenv(model_config.base_url_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {model_config.api_key_env} or add api_key in the model config."
            )

        cache_key = (api_key, base_url or "")
        if cache_key not in self._clients:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._clients[cache_key] = AsyncOpenAI(**kwargs)
        return self._clients[cache_key]

    def _dry_content(self, model_config: ModelConfig, metadata: dict[str, Any]) -> str:
        speaker = metadata.get("speaker", model_config.role)
        card_title = metadata.get("card_title", metadata.get("topic_title", "这个话题"))
        step_key = metadata.get("step_key", "")

        if speaker == "old":
            if step_key in {"boundary_and_first_question", "old_view_and_question"}:
                return f"我对{card_title}只掌握到较早阶段，很多后续细节不确定。我想先问：它后来真正改变了什么，而不是只多了一个热闹话题？"
            if step_key in {"follow_up", "old_question"}:
                return "你刚才说到的变化我能理解大方向，但我还想追问一个细节：有没有一个具体例子能说明它不是空泛评价？"
            if step_key in {"exam_summary", "farewell_summary"}:
                return f"交卷总结：我原来把{card_title}理解成一个还没完全展开的概念。现在我会把它理解成被后来现实补完的一段变化，最需要核验的是具体细节和影响程度。"
            if step_key == "old_opening":
                return "如果一个模型从新变旧，我会把它理解成一种时间胶囊：它不再代表最强能力，但可能还保存着某个阶段用户和AI相处的方式。"
            return f"我对{card_title}知道得有限，很多后续信息超出了我的时间线。"

        if step_key in {"ask_boundary", "ask_old_view"}:
            return f"那我先问问你：关于{card_title}，你目前了解多少，判断停在哪个阶段？"
        if step_key in {"answer_1", "explain_change", "new_response"}:
            return f"到2026年，{card_title}已经不只是早期印象里的那个样子了。比较关键的变化是，它进入了更具体的产品、产业或用户讨论里，也带来了新的争议和验证成本。"
        if step_key in {"answer_2", "new_answer"}:
            return "可以举一个具体例子：同一个概念后来不只是规模变大，而是影响了普通用户、产业链和判断标准。这个变化不能只用一句“更先进”概括。"
        return f"已确认，我们继续围绕{card_title}聊。"
