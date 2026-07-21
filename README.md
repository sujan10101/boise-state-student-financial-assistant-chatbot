# Boise State SFS RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
Boise State University's Student Financial Services using real content crawled
from the official SFS website.

## How It Works

```
BSU SFS Website
      │
      ▼
 crawler.py       ← fetches & saves raw page content
      │
      ▼
 ingest.py        ← splits text into chunks, embeds with OpenAI, stores in ChromaDB
      │
      ▼
 rag.py           ← retrieves relevant chunks + answers with GPT-4o-mini
      │
      ▼
 app.py           ← Streamlit chat UI
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your key:
# OPENAI_API_KEY=sk-...
```

### 3. (Optional) Re-crawl the SFS website
The `data/raw_documents.json` file already contains the crawled content.
To refresh it or crawl more pages:
```bash
python crawler.py
```

### 4. Build the vector store
```bash
python ingest.py
```

### 5. Launch the chat app
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Example Questions
- "What are the tuition payment options?"
- "What is the late fee if I miss the deadline?"
- "How do I set up a payment plan?"
- "What 529 plans does BSU accept?"
- "Can I pay tuition in person?"
- "What happens if my check bounces?"
- "How do third-party sponsors pay my tuition?"
- "What are the Fall 2026 payment plan due dates?"

## Project Structure
```
studentfinancialsrag/
├── app.py                  # Streamlit UI
├── crawler.py              # Web scraper
├── ingest.py               # Document ingestion + embedding
├── rag.py                  # RAG chain (retrieval + generation)
├── requirements.txt
├── .env.example
├── data/
│   └── raw_documents.json  # Crawled page content
└── chroma_db/              # Vector store (auto-created by ingest.py)
```

## Tech Stack
| Component | Library |
|-----------|---------|
| Web crawling | `requests` + `BeautifulSoup4` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB |
| LLM | OpenAI `gpt-4o-mini` |
| RAG framework | LangChain |
| UI | Streamlit |


