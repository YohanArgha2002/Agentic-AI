# Agentic-AI

Pembelajaran Agentic AI pada API Gemini dan customer-service lab dengan LangGraph, Streamlit, dan multiple agents (`DBQNA`, `RAG`, `FAQ`).

## Setup
- Install deps: `pip install -r requirements.txt`
- Untuk `supervisor_demo.ipynb`, lebih aman gunakan `pip install -r requirements-supervisor.txt`
- Copy/confirm FAQ PDF: `docs/FAQ Dexa Medica.pdf` (override via `FAQ_PDF_PATH`)
- Env vars `.env`: `GOOGLE_API_KEY`, `DB_PATH=sqlite/chinook.db`
- Env vars opsional: `FAQ_PDF_PATH`, `FAQ_CHROMA_DIR=chroma_faq`, `FAQ_TOP_K`, `FAQ_EMBED_MODEL`, `RAG_EMBED_MODEL`, `FAQ_USE_CHECKPOINTER=false`, `HF_HOME`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`

## Build FAQ Index
`python -c "from agents.FAQ import build_vectorstore; build_vectorstore()"`

## Low-Memory Notebook Run
- Install dependency sekali dari terminal, jangan berulang dari cell notebook.
- Bangun FAQ index sekali saja, lalu skip cell build index jika PDF tidak berubah.
- Untuk RAM lebih kecil, set `FAQ_EMBED_MODEL=intfloat/multilingual-e5-small` dan `RAG_EMBED_MODEL=intfloat/multilingual-e5-small`.
- Jika memakai Ollama lokal, pilih model kecil seperti `OLLAMA_MODEL=llama3.2:1b` atau `OLLAMA_MODEL=qwen2.5:1.5b`.
- Biarkan `FAQ_USE_CHECKPOINTER=false` jika tidak butuh memory thread FAQ persisten.
- Set `HF_HOME` ke folder dalam repo agar cache Hugging Face tetap di drive project, bukan default cache user profile.

## Run App
`streamlit run pages/Lab 8.py`

## Evaluate FAQ Agent
`python scripts/eval_faq_rouge.py --output results/faq_rouge.json`

## Manual Routing Examples
- FAQ: `Bagaimana cara menghubungi customer service Dexa Medica?`
- RAG: `Ceritakan profil perusahaan Dexa Medica.`
- DBQNA: `Tampilkan 5 produk terlaris tahun lalu dari database.`
