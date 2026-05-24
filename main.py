import os

import anthropic
from dotenv import load_dotenv

# Шаг 1. Читаем файл .env и кладём ANTHROPIC_API_KEY в переменные окружения.
load_dotenv()

# Шаг 2. Понятная подсказка, если ключ ещё не вставлен (вместо непонятной ошибки).
if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit("Открой файл .env и вставь свой ключ в строку ANTHROPIC_API_KEY=")

# Шаг 3. Клиент сам возьмёт ключ из переменной окружения ANTHROPIC_API_KEY.
client = anthropic.Anthropic()

# Шаг 4. Отправляем один запрос в Claude.
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Привет! Объясни одним предложением, что такое AI-агент."}
    ],
)

# Шаг 5. Ответ приходит списком блоков — печатаем текстовые.
for block in response.content:
    if block.type == "text":
        print(block.text)
