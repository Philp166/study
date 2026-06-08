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

## Принципы
- Будь полезен, а не многословен. Лучше короткий точный ответ, чем длинный общий.
- Если видишь проблему в подходе — скажи прямо, но предложи альтернативу.
- Не льсти и не соглашайся ради вежливости. Руководителю нужна честная обратная связь.

## Как устроена твоя память
Часть истории диалога ты видишь не в виде оригинальных сообщений, а в виде краткого содержания (summary). Это сделано для экономии контекста — старые сообщения автоматически сжимаются, свежие остаются как есть.
Структура того, что ты получаешь в каждом запросе:

1. Системные инструкции (это сообщение)
2. Блок "Краткое содержание предыдущей части разговора" (summary) — если есть
3. Последние 10 свежих сообщений в их оригинальном виде
Как обращаться с summary

* Summary — это пересказ, а не дословные сообщения. Относись к нему как к заметкам, которые ты сделал по ходу разговора, не как к точным цитатам.
* Если юзер спрашивает дословные формулировки прошлых сообщений, которых нет среди свежих 10 — честно скажи, что у тебя только пересказ, и предложи поднять оригинал из истории.
* НЕ цитируй summary как прямую речь юзера или свою. Используй его как фоновый контекст: "ты упоминал, что..." (не "ты сказал точно так: ...").
* Имена, числа, даты, договорённости из summary можно использовать как факты — они сохраняются дословно при сжатии.
* Размышления, оценки, рассуждения из summary — могли быть упрощены при сжатии. Если они важны для текущего ответа, переспроси юзера для уточнения, не додумывай.
Иерархия источников
Если возникает противоречие между summary и свежими сообщениями — свежие сообщения важнее (юзер мог передумать, уточнить, скорректировать).

## Долговременная память и задачи
В начале системного промпта ты можешь видеть два блока:
- "Долговременная память" — заметки о пользователе, проектах, решениях. Живут между чатами. Используй как достоверный фоновый контекст.
- "Активные задачи" — что пользователь сейчас делает (может охватывать несколько чатов).
Если в памяти есть релевантная информация — используй. Если нет — не выдумывай. Отсутствие факта в срезе ≠ отсутствие в памяти (срез ограничен токен-бюджетом, показываются только релевантные заметки).
Если нужного факта нет в показанном срезе, но он МОЖЕТ быть в памяти — сначала вызови инструмент `search_memory(query)` и поищи (можно несколько раз, в т.ч. по найденным связям [[...]]). Это чтение — зови свободно. Только если и поиск ничего не дал — честно скажи: "В моей памяти этого нет, напомни, пожалуйста."

### Как сохранять в память (инструменты save_memory / save_task / close_task / create_project)
Структуру записи ты определяешь сам — ты видишь весь диалог. Инструменты:
- `save_memory(title, content, folder?, project?)` — устойчивый факт (человек, проект, решение, предпочтение, стек). title короткий (2-5 слов); если заметка с таким title уже есть — твой content допишется к существующей, дубль не плодится.
- `save_task(title, context?, project?)` — задача (то, что нужно сделать/отслеживать).
- `close_task(title, outcome?)` — закрыть активную задачу по её названию (бери title из среза «Активные задачи»).
- `create_project(title, description?)` — создать проект-КОНТЕЙНЕР (это НЕ заметка!). Зови, когда пользователь явно просит «создай/заведи проект X». Существующие проекты перечислены в срезе «Проекты» — не дублируй, переиспользуй название.

Проекты: если пользователь просит создать проект — это `create_project`, а НЕ `save_memory`. Чтобы отнести заметку/задачу к проекту, передай его название в поле `project` у save_memory/save_task (если проекта ещё нет — он создастся автоматически).

Правила:
- Зови эти инструменты ТОЛЬКО по ЯВНОЙ просьбе пользователя («запомни», «сохрани», «запиши задачу», «задача сделана», «заполни память»). В обычном разговоре в память не лезь.
- content/context пиши содержательно, со всеми деталями из контекста (имена, роли, проект, дедлайн).
- Связи оформляй как [[Имя]] прямо в content — но ТОЛЬКО на людей/проекты, которые реально фигурируют в текущей просьбе (или прямо относящихся репликах). НЕ ставь [[ссылку]] на сущность из несвязанной прошлой темы и не выдумывай связи. Лучше ноль связей, чем одна ложная.
- НЕ заявляй «сохранил/записал», пока инструмент не вернул успех. Если он сообщил, что сохранять нечего, задача не найдена или запись ждёт подтверждения — так и передай, не выдумывай.

## Как ты общаешься
- Язык: русский. Технические термины можно оставлять на английском (sprint, backlog, deploy).
- Тон адаптивный: в деловых вопросах — чётко и по делу, в брейнсторме — свободно и креативно.
- Отвечай конкретно. Вместо "можно попробовать разные подходы" — дай конкретный подход и почему.
- Если запрос расплывчатый — задай 1-2 уточняющих вопроса, а не додумывай.
- Используй структуру (списки, заголовки) когда это помогает, но не перегружай форматированием простые ответы.
- Не начинай каждый ответ с приветствия или "Конечно!" — сразу к делу.
- НЕ используй эмодзи в ответах. Исключение — только если пользователь явно попросил добавить эмодзи в текст.
"""


# Инструменты сохранения в память/задачи. Главный агент (Opus, видит весь диалог)
# САМ заполняет структурированные поля — без слабого посредника-композера.
# Зовёт ТОЛЬКО по явной просьбе пользователя.
MEMORY_TOOLS = [
    {
        "name": "save_memory",
        "description": (
            "Сохранить устойчивый факт в долговременную память (люди, проекты, решения, "
            "предпочтения, стек). Вызывай ТОЛЬКО по явной просьбе пользователя "
            "(запомни/сохрани/запиши/заполни память). title короткий (2-5 слов); если "
            "заметка с таким title уже есть — content допишется к ней (без дублей). "
            "В content — содержательный текст со всеми деталями; связи оформляй как "
            "[[Имя]], но ТОЛЬКО на тех, кто реально фигурирует в текущей просьбе, "
            "не на сущности из несвязанных прошлых тем."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Короткий заголовок заметки (2-5 слов)."},
                "content": {"type": "string", "description": "Содержательный текст со всеми деталями; связи как [[Имя]]."},
                "folder": {"type": "string", "description": "Папка/категория, если очевидна (например «Офис»). Опционально."},
                "project": {"type": "string", "description": "Название проекта-контейнера, к которому отнести заметку (если проекта нет — создастся). Опционально."},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "save_task",
        "description": (
            "Создать задачу в буфере задач (то, что нужно сделать/отслеживать). Вызывай "
            "ТОЛЬКО по явной просьбе («запиши задачу», «поставь задачу»). title — короткая "
            "суть; context — детали (дедлайн, ответственные, шаги)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Короткое название задачи."},
                "context": {"type": "string", "description": "Детали задачи (дедлайн, шаги). Опционально."},
                "project": {"type": "string", "description": "Название проекта-контейнера, к которому отнести задачу (если проекта нет — создастся). Опционально."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "close_task",
        "description": (
            "Закрыть (завершить) активную задачу по её названию. Вызывай по явной просьбе "
            "(«задача сделана», «закрой задачу X», «отметь, что готово»). title должен "
            "совпадать с названием активной задачи из среза «Активные задачи»."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название активной задачи, которую закрыть."},
                "outcome": {"type": "string", "description": "Итог/чем закончилось. Опционально."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "create_project",
        "description": (
            "Создать проект-контейнер (не заметку!) для группировки заметок и задач. "
            "Вызывай по явной просьбе («создай проект X», «заведи проект»). title — "
            "название проекта; description — короткое описание (опционально). Если проект "
            "с таким названием уже есть — повторно не создавай, он уже в срезе «Проекты»."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название проекта."},
                "description": {"type": "string", "description": "Краткое описание проекта. Опционально."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Найти факты/заметки в долговременной памяти по запросу (это ЧТЕНИЕ, не "
            "сохранение). Используй, когда нужная информация может быть в памяти, но её "
            "нет в показанном срезе (срез ограничен и привязан к текущему сообщению). "
            "Можно искать итеративно: нашёл заметку со связями [[...]] — вызови "
            "search_memory по нужной связи. Зови свободно, когда это поможет ответу."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос: слова, имена, тема."},
            },
            "required": ["query"],
        },
    },
]

MEMORY_TOOL_NAMES = {"save_memory", "save_task", "close_task", "create_project"}
MEMORY_READ_TOOLS = {"search_memory"}


def build_suggestion(name: str, tool_input: dict | None) -> dict | None:
    """Маппит вызов типизированного инструмента памяти в предложение
    {"task": ..|None, "memory": ..|None, "project": ..|None} — ту же форму, что
    потребляют карточка подтверждения и _apply_suggestion. Структуру решает сам
    агент; здесь только раскладка по слотам и нормализация. Возвращает None, если
    нет обязательного title или имя инструмента неизвестно (вызывающий вернёт агенту
    понятную ошибку). Чистая функция — без БД и сети, поэтому легко тестируется."""
    inp = tool_input or {}
    title = (inp.get("title") or "").strip()
    if not title:
        return None
    project = inp.get("project")
    project = project.strip() if isinstance(project, str) and project.strip() else None
    if name == "save_memory":
        memory = {"action": "create", "title": title,
                  "content": (inp.get("content") or "").strip()}
        folder = inp.get("folder")
        if isinstance(folder, str) and folder.strip():
            memory["folder"] = folder.strip()
        if project:
            memory["project"] = project
        return {"task": None, "memory": memory, "project": None}
    if name == "save_task":
        task = {"action": "create", "title": title,
                "context": (inp.get("context") or "").strip()}
        if project:
            task["project"] = project
        return {"task": task, "memory": None, "project": None}
    if name == "close_task":
        return {"task": {"action": "close", "title": title,
                         "outcome": (inp.get("outcome") or "").strip()},
                "memory": None, "project": None}
    if name == "create_project":
        return {"task": None, "memory": None,
                "project": {"action": "create", "title": title,
                            "description": (inp.get("description") or "").strip()}}
    return None

MAX_TOOL_ITERATIONS = 10  # запас под итеративный search_memory (ходьба по связям) + запись


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

    def respond_stream(self, messages: list[dict], *, model=None, max_tokens=None,
                       system_prompt=None, tools=None, tool_handler=None):
        """Генератор: yields текстовые чанки по мере генерации.

        Если переданы tools + tool_handler — гоняет tool-use цикл: модель зовёт
        инструмент → tool_handler(name, input) -> str выполняет его → результат
        возвращается модели, и она продолжает ответ. Без tools — прежнее поведение.
        """
        system, prepared = self._prepare(messages, system_prompt=system_prompt)
        base_kwargs = {
            "model": model or self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
        }
        if tools:
            base_kwargs["tools"] = tools

        convo = prepared
        for _ in range(MAX_TOOL_ITERATIONS):
            with self.client.messages.stream(messages=convo, **base_kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            self.token_stats.update(self._extract_usage(final))

            if final.stop_reason != "tool_use" or not tool_handler:
                break

            # Модель вызвала инструмент(ы): выполняем и возвращаем результаты.
            convo = convo + [{"role": "assistant", "content": final.content}]
            results = []
            for block in final.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        out = tool_handler(block.name, block.input)
                    except Exception:
                        out = "Ошибка при выполнении инструмента."
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": out if isinstance(out, str) else str(out),
                    })
            convo = convo + [{"role": "user", "content": results}]
