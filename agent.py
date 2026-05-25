"""
CoS (Chief of Staff) — AI-агент, инкапсулирующий логику общения с LLM.

Агент — отдельная сущность со своим системным промптом, настройками модели
и методом обработки диалога. app.py делегирует ему всю работу с Claude API.
"""

import anthropic


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
