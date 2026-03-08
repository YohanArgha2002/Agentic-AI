# revou-gen-ai-tutorial

Customer-service lab with LangGraph, Streamlit, and multiple agents (DBQNA, RAG, FAQ).

## Setup
- Install deps: `pip install -r requirements.txt`
- For `supervisor_demo.ipynb` only, prefer `pip install -r requirements-supervisor.txt`
- Copy/confirm FAQ PDF: `docs/FAQ Dexa Medica.pdf` (override via `FAQ_PDF_PATH`)
- Env vars (.env): `GOOGLE_API_KEY` (Gemini), `DB_PATH=sqlite/chinook.db`; optional `FAQ_PDF_PATH`, `FAQ_CHROMA_DIR=chroma_faq`, `FAQ_TOP_K`, `FAQ_EMBED_MODEL`, `RAG_EMBED_MODEL`, `FAQ_USE_CHECKPOINTER=false`, `HF_HOME`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`

## Build FAQ index (once)
`python -c "from agents.FAQ import build_vectorstore; build_vectorstore()"`

## Low-memory notebook run
- Install dependencies once from terminal, not from repeated notebook cells.
- Build the FAQ index once, then skip re-running the index build cell unless the PDF changes.
- For lower RAM usage, set `FAQ_EMBED_MODEL=intfloat/multilingual-e5-small` and `RAG_EMBED_MODEL=intfloat/multilingual-e5-small` in `.env`.
- If you use Ollama locally, prefer a smaller model such as `OLLAMA_MODEL=llama3.2:1b` or `OLLAMA_MODEL=qwen2.5:1.5b`.
- Leave `FAQ_USE_CHECKPOINTER=false` unless you explicitly need persistent FAQ thread memory.
- Set `HF_HOME` to a folder inside the repo if you want Hugging Face model downloads to stay on `D:` instead of the default cache under your user profile on `C:`.
- In `supervisor_demo.ipynb`, avoid repeatedly running the module reload cell because it recreates agent objects.

## Run app
`streamlit run pages/Lab 8.py`

## Evaluate FAQ agent
`python scripts/eval_faq_rouge.py --output results/faq_rouge.json`

## Manual routing examples
- FAQ → “Bagaimana cara menghubungi customer service Dexa Medica?”  
- RAG → “Ceritakan profil perusahaan Dexa Medica.”  
- DBQNA → “Tampilkan 5 produk terlaris tahun lalu dari database.”  
Expect FAQ answers grounded in the PDF, RAG for company profile, DBQNA for SQL-able questions.
