import os
import re
import shutil
from pathlib import Path
from typing import List, Literal, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv(override=True)

# Defaults & env overrides
DEFAULT_PDF_PATH = os.getenv("FAQ_PDF_PATH", "docs/FAQ Dexa Medica.pdf")
DEFAULT_CHROMA_DIR = os.getenv("FAQ_CHROMA_DIR", "chroma_faq")
COLLECTION_NAME = "faq_dexa_medica"
# Google GenAI embedding model (v1beta) – use newer default
EMBED_MODEL = os.getenv("FAQ_EMBED_MODEL", "intfloat/multilingual-e5-large-instruct")
TOP_K = int(os.getenv("FAQ_TOP_K", "4"))
USE_CHECKPOINTER = os.getenv("FAQ_USE_CHECKPOINTER", "0").lower() in {"1", "true", "yes"}

# Globals initialised lazily so we don't rebuild unnecessarily
_EMBEDDINGS = None
_VECTORSTORE = None


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBEDDINGS


def _format_chunk(faq_no: str, question: str, answer: str) -> str:
    question = question.strip()
    answer = answer.strip()
    return f"FAQ No: {faq_no}\nQ: {question}\nA: {answer}"


def _split_question_answer(text: str) -> Tuple[str, str]:
    """Attempt to split a FAQ item body into question and answer."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", ""

    q_lines, a_lines = [], []
    mode = "q"
    for line in lines:
        low = line.lower()
        if re.match(r"^(jawaban|answer|a[:.]?)\\s*", low):
            mode = "a"
            stripped = re.sub(r"^(jawaban|answer|a[:.]?)\\s*", "", line, flags=re.I).strip()
            if stripped:
                a_lines.append(stripped)
            continue
        if mode == "q":
            q_lines.append(line)
        else:
            a_lines.append(line)

    if not a_lines and len(lines) > 1:
        a_lines = lines[1:]

    question = " ".join(q_lines).strip() or lines[0]
    answer = " ".join(a_lines).strip()
    return question, answer


def _chunk_faq_documents(pages: List[Document], source: str) -> List[Document]:
    """Split PDF into FAQ chunks; tolerate numbering without leading newlines."""
    combined_text = ""
    page_offsets: List[int] = []
    for idx, doc in enumerate(pages):
        page_offsets.append(len(combined_text))
        combined_text += f"\n[[PAGE:{idx+1}]]\n{doc.page_content}"

    # Match any occurrence of "<number>. " (not necessarily at line start)
    pattern = re.compile(r"(\d+)\.\s")
    spans = list(pattern.finditer(combined_text))
    chunks: List[Document] = []

    for i, match in enumerate(spans):
        faq_no = match.group(1)
        start = match.end()
        end = spans[i + 1].start() if i + 1 < len(spans) else len(combined_text)
        body = combined_text[start:end].strip()
        if not body:
            continue

        # Map start index to page number (best effort)
        start_idx = match.start()
        page = 1
        for j, offset in enumerate(page_offsets):
            if start_idx >= offset:
                page = j + 1

        question, answer = _split_question_answer(body)
        if not answer:
            answer = "Informasi jawaban tidak tersedia di dokumen."

        formatted = _format_chunk(faq_no, question, answer)
        chunks.append(
            Document(
                page_content=formatted,
                metadata={
                    "faq_no": faq_no,
                    "question": question,
                    "page": page,
                    "source": source,
                },
            )
        )

    return chunks


def parse_faq_pairs(pdf_path: str = DEFAULT_PDF_PATH) -> List[Tuple[str, str, str]]:
    """Return (faq_no, question, answer) tuples parsed from the FAQ PDF."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    docs = _chunk_faq_documents(pages, pdf_path)
    pairs = []
    for doc in docs:
        faq_no = str(doc.metadata.get("faq_no", ""))
        content = doc.page_content
        q_match = re.search(r"Q:\\s*(.*)", content)
        a_match = re.search(r"A:\\s*(.*)", content, re.S)
        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        pairs.append((faq_no, question, answer))
    return pairs


def build_vectorstore(
    pdf_path: str = DEFAULT_PDF_PATH,
    persist_directory: str = DEFAULT_CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
) -> Chroma:
    """Create or load the persistent Chroma store for FAQs."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"FAQ PDF not found at '{pdf_path}'. Set FAQ_PDF_PATH env var or place the file correctly."
        )

    persist_path = Path(persist_directory)
    if force_rebuild and persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)

    embeddings = _get_embeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    try:
        existing = store._collection.count()  # type: ignore[attr-defined]
    except Exception:
        existing = 0

    if existing and not force_rebuild:
        return store

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    faq_docs = _chunk_faq_documents(pages, pdf_path)
    if not faq_docs:
        raise ValueError("No FAQ items were parsed from the PDF; check the document format.")

    store.add_documents(faq_docs)
    store.persist()
    return store


def get_vectorstore() -> Chroma:
    """Load or create the Chroma store (lazy, singleton)."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = build_vectorstore()
    return _VECTORSTORE


@tool("faq_similarity_search", parse_docstring=False)
def faq_similarity_search(query: str, k: int = TOP_K):
    """
    Perform semantic similarity search over the FAQ knowledge base.
    Args:
        query: user question
        k: number of top chunks to return (default from env FAQ_TOP_K or 4)
    Returns:
        List of retrieved FAQ chunks with metadata for grounding.
    """
    store = get_vectorstore()
    docs = store.similarity_search(query, k=k)
    return [
        {
            "faq_no": d.metadata.get("faq_no"),
            "question": d.metadata.get("question"),
            "page": d.metadata.get("page"),
            "source": d.metadata.get("source"),
            "content": d.page_content,
        }
        for d in docs
    ]


class JudgeResult(BaseModel):
    answerable: bool = Field(..., description="True if retrieved context is sufficient to answer.")
    reason: str


class FAQState(MessagesState):
    user_question: str
    rewritten_question: str
    attempt: int
    retrieved_docs: List[Document] = Field(default_factory=list)
    judge_decision: dict | None = None


def _prepare_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        prefix = f"FAQ {meta.get('faq_no')} (hal {meta.get('page')}):"
        parts.append(f"{prefix}\n{doc.page_content}")
    return "\n\n".join(parts)


llm = ChatOpenAI(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)



def retrieve(state: FAQState):
    query = state.get("rewritten_question") or state["user_question"]
    docs = get_vectorstore().similarity_search(query, k=TOP_K)
    return {
        "retrieved_docs": docs,
        # Keep messages untouched; they live in state already.
    }


def judge(state: FAQState):
    docs = state.get("retrieved_docs", [])
    # Skip LLM call if tidak ada dokumen yang ditemukan
    if not docs:
        return {"judge_decision": {"answerable": False, "reason": "No context retrieved"}}
    context = _prepare_context(docs)
    question = state.get("user_question", "")
    prompt = [
        SystemMessage(
            content=(
                "Tentukan apakah konteks berikut sudah cukup untuk menjawab pertanyaan pengguna. "
                "Jawab dengan field answerable=True jika informasi kunci ada; jika konteks tipis/umum/tidak terkait, gunakan answerable=False. "
                "Berikan alasan singkat."
            )
        ),
        HumanMessage(content=f"Pertanyaan: {question}\n\nKonteks:\n{context or '(kosong)'}"),
    ]
    structured_llm = llm.with_structured_output(JudgeResult)
    decision = structured_llm.invoke(prompt)
    return {"judge_decision": decision.model_dump()}


def rewrite(state: FAQState):
    question = state.get("user_question", "")
    prior = state.get("retrieved_docs", [])
    snippets = "\n".join(doc.page_content for doc in prior[:2])
    prompt = [
        SystemMessage(
            content=(
                "Tulis ulang pertanyaan supaya lebih cocok dengan pencarian di basis FAQ Dexa Medica. "
                "Gunakan kata kunci khusus (layanan, kontak, jam operasional, pemesanan, pengaduan) jika relevan."
            )
        ),
        HumanMessage(content=f"Pertanyaan awal: {question}\nCuplikan yang ditemukan:\n{snippets}"),
    ]
    rewritten = llm.invoke(prompt)
    return {
        "rewritten_question": rewritten.content.strip(),
        "attempt": state.get("attempt", 0) + 1,
    }


def answer(state: FAQState):
    docs = state.get("retrieved_docs", [])
    context = _prepare_context(docs)
    question = state.get("user_question", "")
    not_enough = False
    if state.get("judge_decision"):
        not_enough = not state["judge_decision"].get("answerable", False)

    system_prompt = (
        "Kamu adalah agen layanan pelanggan Dexa Medica. Jawab dalam bahasa Indonesia dengan sopan dan ringkas. "
        "Gunakan HANYA informasi pada konteks. Jika informasi tidak ada di konteks, katakan tidak ada informasi dan sarankan hubungi CS Dexa Medica."
    )
    if not_enough and state.get("attempt", 0) >= 2 and not docs:
        content = (
            "Maaf, aku tidak menemukan informasi yang kamu butuhkan di FAQ. "
            "Silakan hubungi layanan pelanggan Dexa Medica untuk bantuan lebih lanjut."
        )
        return {"messages": [AIMessage(content=content)]}

    prompt: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Pertanyaan: {question}\n\nKonteks untuk dijadikan dasar jawaban:\n{context or '(tidak ada konteks)'}"
        ),
    ]
    response = llm.invoke(prompt)
    return {"messages": [response]}


def route_from_judge(state: FAQState) -> Literal["answer", "rewrite"]:
    decision = state.get("judge_decision") or {}
    answerable = decision.get("answerable", False)
    # Batasi hanya 1 rewrite (attempt 0 -> 1). Setelah itu jawab apa adanya.
    if answerable or state.get("attempt", 0) >= 1:
        return "answer"
    return "rewrite"


graph_builder = (
    StateGraph(FAQState)
    .add_node("retrieve", retrieve)
    .add_node("judge", judge)
    .add_node("rewrite", rewrite)
    .add_node("answer", answer)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "judge")
    .add_conditional_edges("judge", route_from_judge, {"answer": "answer", "rewrite": "rewrite"})
    .add_edge("rewrite", "retrieve")
    .add_edge("answer", END)
)

graph = (
    graph_builder.compile(checkpointer=MemorySaver(), name="FAQ")
    if USE_CHECKPOINTER
    else graph_builder.compile(name="FAQ")
)


__all__ = [
    "graph",
    "build_vectorstore",
    "get_vectorstore",
    "faq_similarity_search",
    "parse_faq_pairs",
]
