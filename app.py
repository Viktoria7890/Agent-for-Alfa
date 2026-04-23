from pathlib import Path

import gradio as gr

from pipeline import DocumentPipeline

pipeline = DocumentPipeline(model_name="qwen2.5:14b")

SAMPLES_DIR = Path("data/samples")


def _sample_choices() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(str(f) for f in SAMPLES_DIR.iterdir() if f.suffix in (".txt", ".pdf", ".docx"))


def load_document(file, sample_path: str) -> str:
    path = file.name if file else sample_path
    if not path:
        return "⚠️ Выберите или загрузите документ"
    try:
        meta = pipeline.load_and_index(path)
        return (
            f"✅ Загружено: **{meta['doc_name']}**\n\n"
            f"📄 Символов: {meta['char_count']:,}  |  🔢 Чанков: {meta['chunk_count']}"
        )
    except Exception as e:
        return f"❌ Ошибка загрузки: {e}"


def run_extract():
    if not pipeline.doc_text:
        yield "⚠️ Сначала загрузите документ"
        return
    yield "⏳ Извлекаю параметры, подождите..."
    yield pipeline.extract_parameters()


def run_summarize():
    if not pipeline.doc_text:
        yield "⚠️ Сначала загрузите документ"
        return
    yield "⏳ Суммаризирую документ, подождите..."
    yield pipeline.summarize()


def run_categorize():
    if not pipeline.doc_text:
        yield "⚠️ Сначала загрузите документ"
        return
    yield "⏳ Категоризирую документ, подождите..."
    yield pipeline.categorize()


def send_message(msg: str, history: list) -> tuple[list, str]:
    if not msg.strip():
        return history, ""
    if not pipeline.doc_text:
        new_history = history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": "⚠️ Сначала загрузите документ"},
        ]
        return new_history, ""
    answer = pipeline.answer_question(msg)
    new_history = history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": answer},
    ]
    return new_history, ""


# ── UI ───────────────────────────────────────────────────────────────────────

CSS = """
.panel-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; }
footer { display: none !important; }
"""

with gr.Blocks(title="AIDA — Document Analysis Agent") as demo:
    gr.Markdown(
        "# AIDA — AI Document Agent\n"
        "#### Анализ юридических и финансовых документов  ·  Ollama `qwen2.5:14b`  ·  ChromaDB RAG"
    )

    with gr.Row():
        # ── Left panel: document loading ─────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Документ", elem_classes="panel-header")
            file_input = gr.File(
                label="Загрузить PDF / DOCX / TXT",
                file_types=[".pdf", ".docx", ".txt"],
            )
            sample_dd = gr.Dropdown(
                choices=_sample_choices(),
                label="Или выбрать пример из /data/samples",
                value=None,
            )
            refresh_btn = gr.Button("🔄 Обновить список файлов", size="sm")
            load_btn = gr.Button("Загрузить и индексировать", variant="primary", size="lg")
            status_md = gr.Markdown("_Документ не загружен_")

            gr.Markdown("---")
            gr.Markdown(
                "**Стек:**\n"
                "- LLM: `qwen2.5:14b` via Ollama\n"
                "- Embeddings: `nomic-embed-text`\n"
                "- Vector DB: ChromaDB (in-memory)\n"
                "- Chunking: RecursiveCharacterTextSplitter\n"
                "- UI: Gradio"
            )

        # ── Right panel: analysis tabs ────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📋 Параметры"):
                    extract_btn = gr.Button("Извлечь параметры", variant="primary")
                    params_out = gr.Markdown("_Нажмите кнопку после загрузки документа_")

                with gr.Tab("📝 Суммаризация"):
                    summarize_btn = gr.Button("Суммаризировать", variant="primary")
                    summary_out = gr.Markdown("_Нажмите кнопку после загрузки документа_")

                with gr.Tab("🏷️ Категоризация"):
                    categorize_btn = gr.Button("Категоризировать", variant="primary")
                    category_out = gr.Markdown("_Нажмите кнопку после загрузки документа_")

                with gr.Tab("💬 Q&A по документу"):
                    history_state = gr.State([])
                    chatbot = gr.Chatbot(
                        height=420,
                        label="RAG-чат",
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Задайте вопрос по документу...",
                            label="",
                            scale=5,
                            show_label=False,
                        )
                        chat_btn = gr.Button("Отправить", variant="primary", scale=1)
                    gr.Examples(
                        examples=[
                            "Кто является учредителями?",
                            "Каков размер уставного капитала?",
                            "На какой срок выдана доверенность?",
                            "Какие виды деятельности разрешены?",
                            "Перечисли основные риски документа",
                        ],
                        inputs=msg_input,
                    )

    # ── Event wiring ─────────────────────────────────────────────────────────

    refresh_btn.click(lambda: gr.Dropdown(choices=_sample_choices()), outputs=sample_dd)
    load_btn.click(load_document, inputs=[file_input, sample_dd], outputs=status_md)
    extract_btn.click(run_extract, outputs=params_out)
    summarize_btn.click(run_summarize, outputs=summary_out)
    categorize_btn.click(run_categorize, outputs=category_out)

    chat_btn.click(
        send_message,
        inputs=[msg_input, history_state],
        outputs=[chatbot, msg_input],
    ).then(lambda h: h, inputs=chatbot, outputs=history_state)

    msg_input.submit(
        send_message,
        inputs=[msg_input, history_state],
        outputs=[chatbot, msg_input],
    ).then(lambda h: h, inputs=chatbot, outputs=history_state)


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(), css=CSS)
