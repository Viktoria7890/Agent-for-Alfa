# AIDA — AI Document Agent

RAG-агент для автоматического анализа юридических и финансовых документов.

## Стек

| Слой | Технология |
|---|---|
| LLM | `qwen2.5:14b` via Ollama (локально) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector DB | ChromaDB (in-memory) |
| Chunking | RecursiveCharacterTextSplitter |
| UI | Gradio |

## Возможности

- **Извлечение параметров** — ИНН, ОГРН, КПП, стороны, даты, подписанты
- **Суммаризация** — структурированное резюме документа
- **Категоризация** — тип, юридическая сила, риски, рекомендации
- **Q&A (RAG)** — свободные вопросы по содержимому документа

Поддерживаемые форматы: PDF, DOCX, TXT

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Загрузить модели в Ollama
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# 3. Запустить
python app.py
```

Открыть в браузере: http://localhost:7860

## Структура проекта

```
├── app.py            # Gradio UI
├── pipeline.py       # DocumentPipeline (загрузка, чанкинг, RAG, LLM-инструменты)
├── requirements.txt
└── data/
    └── samples/      # Тестовые документы (устав, доверенность, выписка ЕГРЮЛ)
```

## Демо-документы

В `data/samples/` три синтетических примера:
- `ustav_ooo_tekhprom.txt` — Устав ООО
- `doverennost.txt` — Доверенность
- `vypiska_egryl.txt` — Выписка из ЕГРЮЛ

Реальные документы можно скачать с [egrul.nalog.ru](https://egrul.nalog.ru) (выписки ЕГРЮЛ в PDF).
