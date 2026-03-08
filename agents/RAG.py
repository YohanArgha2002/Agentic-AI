import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv(override=True)

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-large-instruct")
RAG_PDF_PATH = os.getenv("RAG_PDF_PATH", "./docs/Tentang Dexa Medica.pdf")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

_EMBEDDINGS = None
_VECTOR_STORE = None


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _EMBEDDINGS


def _build_vector_store():
    if not os.path.exists(RAG_PDF_PATH):
        raise FileNotFoundError(
            f"RAG PDF not found at '{RAG_PDF_PATH}'. Set RAG_PDF_PATH or place the file correctly."
        )

    vector_store = InMemoryVectorStore(_get_embeddings())
    loader = PyPDFLoader(RAG_PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(all_splits)
    return vector_store


def get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = _build_vector_store()
    return _VECTOR_STORE


def reset_vector_store():
    global _EMBEDDINGS, _VECTOR_STORE
    _EMBEDDINGS = None
    _VECTOR_STORE = None


llm = ChatOpenAI(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    temperature=0,
)


class RAGState(MessagesState):
    retrieved_docs: List[Document]


def _latest_user_question(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def retrieve_context(state: RAGState):
    question = _latest_user_question(state["messages"])
    retrieved_docs = get_vector_store().similarity_search(question, k=RAG_TOP_K)
    return {"retrieved_docs": retrieved_docs}


def generate(state: RAGState):
    question = _latest_user_question(state["messages"])
    docs = state.get("retrieved_docs", [])
    docs_content = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
    )
    prompt = [
        SystemMessage(
            content=(
                "You are an assistant for question-answering tasks. "
                "Use the following retrieved context to answer the question. "
                "If the context is insufficient, say you do not know. Keep the answer concise."
            )
        ),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{docs_content or '(no context)'}"),
    ]
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph = (
    StateGraph(RAGState)
    .add_node("retrieve_context", retrieve_context)
    .add_node("generate", generate)
    .add_edge(START, "retrieve_context")
    .add_edge("retrieve_context", "generate")
    .add_edge("generate", END)
    .compile(name="RAG")
)


__all__ = ["graph", "get_vector_store", "reset_vector_store"]
