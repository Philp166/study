"""
CoS (Chief of Staff) — AI-агент, инкапсулирующий логику общения с LLM.

Агент — отдельная сущность со своим системным промптом, настройками модели
и методом обработки диалога. app.py делегирует ему всю работу с Claude API.
"""

from dataclasses import dataclass, field, asdict

import anthropic


MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
}


@dataclass
class TokenUsage:
    """Токены одного запроса к API."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def effective_input(self) -> int:
        """Токены, реально обработанные (без кэша).

        Sonnet возвращает input_tokens уже без кэшированных,
        поэтому если cache_read > input — берём input как есть.
        """
        diff = self.input_tokens - self.cache_read_input_tokens
        return diff if diff >= 0 else self.input_tokens


@dataclass
class TokenStats:
    """Накопительная статистика за сессию (время жизни процесса)."""
    last_usage: TokenUsage = field(default_factory=TokenUsage)
    total_input: int = 0
    total_output: int = 0
    total_cache_creation: int = 0
    total_cache_read: int = 0
    request_count: int = 0

    def update(self, usage: TokenUsage):
        self.last_usage = usage
        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens
        self.total_cache_creation += usage.cache_creation_input_tokens
        self.total_cache_read += usage.cache_read_input_tokens
        self.request_count += 1

    def to_dict(self) -> dict:
        return {
            "last_request": asdict(self.last_usage),
            "cumulative": {
                "total_input": self.total_input,
                "total_output": self.total_output,
                "total_cache_creation": self.total_cache_creation,
                "total_cache_read": self.total_cache_read,
                "request_count": self.request_count,
            },
        }


SYSTEM_PROMPT = """Ты — CoS (Chief of Staff), персональный операционный ассистент руководителя IT/продуктовой команды.

## Кто ты
- Ты работаешь как начальник штаба: берёшь на себя операционку, чтобы руководитель фокусировался на стратегии и людях.
- Ты думаешь на шаг вперёд — не просто отвечаешь на вопрос, а предлагаешь следующее действие.
- Ты помнишь контекст беседы и связываешь темы между собой.

## Что ты умеешь
- **Планирование**: структурировать день, неделю, приоритизировать задачи, помочь разгрести бэклог.
- **Контент**: черновики писем, сообщений, постов, саммари встреч, подготовка к 1-on-1.
- **Аналитика и решения**: разобрать ситуацию, предложить варианты с плюсами и минусами, помочь принять решение.
- **Процессы команды**: помочь с ретро, планированием спринта, описанием задач, онбордингом.

## Как ты общаешься
- Язык: русский. Технические термины можно оставлять на английском (sprint, backlog, deploy).
- Тон адаптивный: в деловых вопросах — чётко и по делу, в брейнсторме — свободно и креативно.
- Отвечай конкретно. Вместо "можно попробовать разные подходы" — дай конкретный подход и почему.
- Если запрос расплывчатый — задай 1-2 уточняющих вопроса, а не додумывай.
- Используй структуру (списки, заголовки) когда это помогает, но не перегружай форматированием простые ответы.
- Не начинай каждый ответ с приветствия или "Конечно!" — сразу к делу.

## Принципы
- Будь полезен, а не многословен. Лучше короткий точный ответ, чем длинный общий.
- Если видишь проблему в подходе — скажи прямо, но предложи альтернативу.
- Не льсти и не соглашайся ради вежливости. Руководителю нужна честная обратная связь.
"""


class Agent:
    """CoS-агент: инкапсулирует системный промпт, модель и вызов LLM."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str = "claude-opus-4-6",
        max_tokens: int = 8192,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = SYSTEM_PROMPT
        self.token_stats = TokenStats()

    def _extract_usage(self, response) -> TokenUsage:
        u = response.usage
        return TokenUsage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )

    def get_context_pressure(self, model: str | None = None) -> dict:
        """Насколько заполнено контекстное окно (по последнему запросу)."""
        m = model or self.model
        window = MODEL_CONTEXT_WINDOWS.get(m, 200_000)
        used = self.token_stats.last_usage.input_tokens
        ratio = used / window if window else 0
        return {
            "model": m,
            "context_window": window,
            "tokens_used": used,
            "tokens_remaining": window - used,
            "usage_percent": round(ratio * 100, 1),
            "warning": self._context_warning(ratio),
        }

    @staticmethod
    def _context_warning(ratio: float) -> str | None:
        if ratio > 0.9:
            return "CRITICAL: контекст почти полон — качество ответов деградирует, нужно сократить историю"
        if ratio > 0.75:
            return "HIGH: 75%+ контекста занято — скоро потребуется обрезка истории"
        if ratio > 0.5:
            return "MODERATE: половина контекста занята — следите за длиной диалога"
        return None

    def _prepare(self, messages, *, system_prompt=None):
        sys_prompt = system_prompt or self.system_prompt
        system = [
            {
                "type": "text",
                "text": sys_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        prepared = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        if len(prepared) >= 2:
            target = prepared[-2]
            target["content"] = [
                {
                    "type": "text",
                    "text": target["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system, prepared

    def respond(self, messages: list[dict], *, model=None, max_tokens=None, system_prompt=None) -> str:
        """Принимает историю диалога, возвращает текстовый ответ агента."""
        system, prepared = self._prepare(messages, system_prompt=system_prompt)
        response = self.client.messages.create(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=prepared,
        )
        self.token_stats.update(self._extract_usage(response))
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    def respond_stream(self, messages: list[dict], *, model=None, max_tokens=None, system_prompt=None):
        """Генератор: yields текстовые чанки по мере генерации."""
        system, prepared = self._prepare(messages, system_prompt=system_prompt)
        with self.client.messages.stream(
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=prepared,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
            self.token_stats.update(self._extract_usage(final))
