from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """Ты — специалист по анализу юридических и финансовых документов.
Работаешь с российскими правовыми документами: уставами, доверенностями, выписками ЕГРЮЛ, договорами, отчётностью.
Отвечай точно, структурированно, на русском языке.
Если информация не найдена в документе — говори об этом явно."""


class DocumentPipeline:
    def __init__(self, model_name: str = "qwen2.5:14b", embed_model: str = "nomic-embed-text"):
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.vectorstore: Optional[Chroma] = None
        self.doc_text: str = ""
        self.doc_name: str = ""

    def load_and_index(self, file_path: str) -> dict:
        path = Path(file_path)
        self.doc_name = path.name

        if path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif path.suffix.lower() in (".docx", ".doc"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        raw_docs = loader.load()
        self.doc_text = "\n\n".join(d.page_content for d in raw_docs)

        chunks = self.splitter.create_documents(
            [self.doc_text],
            metadatas=[{"source": path.name}],
        )

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        return {
            "doc_name": self.doc_name,
            "char_count": len(self.doc_text),
            "chunk_count": len(chunks),
        }

    def _context(self, query: str, k: int = 5) -> str:
        if not self.vectorstore:
            return self.doc_text[:4000]
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def _ask(self, user_prompt: str) -> str:
        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return response.content

    # ── Tool 1: Извлечение параметров ────────────────────────────────────────

    def extract_parameters(self) -> str:
        context = self.doc_text[:5000]
        return self._ask(f"""Извлеки все ключевые параметры из документа.

ДОКУМЕНТ:
{context}

Верни структурированный список:
**Тип документа:** ...
**Наименование организации:** ...
**ИНН:** ...
**ОГРН:** ...
**КПП:** ...
**Юридический адрес:** ...
**Дата документа:** ...
**Стороны документа:** ...
**Предмет / цель:** ...
**Срок действия:** ...
**Подписанты:** ...
**Дополнительные реквизиты:** ...

Если параметр отсутствует — укажи «не указано».""")

    # ── Tool 2: Суммаризация ─────────────────────────────────────────────────

    def summarize(self) -> str:
        context = self.doc_text[:8000]
        return self._ask(f"""Создай структурированное резюме юридического документа.

ДОКУМЕНТ:
{context}

Резюме:
1. **Тип и назначение документа**
2. **Основные стороны**
3. **Ключевые положения** (3–5 пунктов)
4. **Права и обязанности сторон**
5. **Сроки и условия**
6. **Важные оговорки или ограничения**

Будь конкретным, ссылайся на факты из документа.""")

    # ── Tool 3: Категоризация ────────────────────────────────────────────────

    def categorize(self) -> str:
        context = self.doc_text[:4000]
        return self._ask(f"""Проанализируй и категоризируй документ.

ДОКУМЕНТ:
{context}

Предоставь:
1. **Тип документа:** Устав / Доверенность / Выписка ЕГРЮЛ / Договор / Отчётность / Иное
2. **Подтип / специализация**
3. **Юридическая сила:** действующий / истёкший / требует проверки
4. **Полнота документа:** полный / неполный (что отсутствует)
5. **Риски и замечания:** потенциальные проблемы
6. **Рекомендации:** что следует проверить или уточнить""")

    # ── Tool 4: Q&A (RAG) ────────────────────────────────────────────────────

    def answer_question(self, question: str) -> str:
        context = self._context(question)
        return self._ask(f"""Ответь на вопрос, используя только информацию из документа.

РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ:
{context}

ВОПРОС: {question}

Если ответ не содержится в документе — скажи об этом явно.
Ссылайся на конкретные части документа.""")
