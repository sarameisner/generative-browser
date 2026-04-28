# Generative Browser

A Flask-based web application that simulates a fully functional browser — except every page is generated in real time by a local LLM via Ollama. Enter any URL (real or imaginary) and watch the AI write the HTML live.

Built as an exam project for the **LLM for Developers** course at Erhvervsakademi København.

---

## Features

- **Live HTML streaming** — pages are generated token by token and rendered progressively in a sandboxed iframe via Server-Sent Events (SSE)
- **RAG pipeline** — upload documents to a local ChromaDB knowledge base; relevant chunks are retrieved and injected into the prompt before generation
- **User style profiles** — choose reading style, tone, and experience level; the prompt changes visibly and produces different pages for the same URL
- **Domain memory** — the browser remembers brand identity (colors, fonts, tone) across pages on the same domain
- **4T's prompt structure** — all prompts are explicitly structured around Traits, Task, Tone, and Target
- **Back / Forward navigation** — full history stack with cached HTML for instant back navigation
- **Prompt debug panel** — collapsible panel showing exactly what was sent to the LLM, including retrieved RAG context

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- The following Ollama models pulled:

```bash
ollama pull qwen3:1.7b
ollama pull embeddinggemma
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/generative-browser.git
cd generative-browser

# 2. Install Python dependencies
pip3 install flask ollama chromadb pypdf requests

# 3. Make sure Ollama is running
ollama serve
```

---

## Running the app

```bash
python3 app.py
```

Open your browser and go to: **http://localhost:5000**

---

## How to use

### Browsing
1. Set up your profile on the welcome screen (reading style, tone, experience level)
2. Type any URL in the address bar and press **Enter** — real or imaginary
3. Watch the page generate live

### RAG — Knowledge Base
1. Click the **KB** button in the toolbar
2. Upload a `.txt` or `.pdf` file, or paste text directly
3. Visit a URL related to the content — the retrieved text will appear in the debug panel and influence the generated page

**Example:** Upload a text file describing a fictional flower shop. Visit `luna-blomster.dk`. The generated page will include your specific products, prices, and details instead of generic AI content.

### Debug panel
Click **Prompt Debug** at the bottom of the screen to see:
- What was retrieved from the knowledge base (RAG context)
- The full 4T-structured system prompt sent to the LLM
- The user task prompt

---

## Architecture

```
User enters URL
      │
      ▼
Flask /generate
  ├── Reads user profile from session
  ├── Retrieves relevant chunks from ChromaDB (RAG)
  ├── Builds 4T-structured prompt
  └── Returns stream_id
      │
      ▼
Flask /stream/<id>  (SSE)
  ├── Calls Ollama (qwen3:1.7b) with streaming
  ├── Filters <think> blocks and markdown fences
  ├── Yields HTML chunks as SSE events
  └── Stores domain context after completion
      │
      ▼
Browser (JavaScript)
  ├── Opens EventSource to /stream
  ├── Writes chunks to iframe via document.write()
  ├── Hides loading overlay on first chunk
  └── Saves completed HTML to history stack
```

---

## Project structure

```
generative-browser/
├── app.py              # Flask backend — routes, RAG, prompt builder
├── requirements.txt    # Python dependencies
├── chroma_db/          # Persistent ChromaDB vector store (auto-created)
└── templates/
    └── index.html      # Single-page frontend — browser UI, profile modal, RAG panel
```

---

## 4T's prompt structure

Every prompt is built around the 4T's framework:

| T | Description | Source |
|---|-------------|--------|
| **Traits** | Expert web designer, front-end developer, and copywriter | Hardcoded in system prompt |
| **Task** | Generate a complete HTML page for the given URL, grounded in RAG context | URL + retrieved chunks + domain memory |
| **Tone** | How content should read — reading style and voice | User profile selection |
| **Target** | Who the content is written for — experience level | User profile selection |

---

## RAG pipeline

1. User uploads a document (`.txt` or `.pdf`) via the KB panel
2. Text is split into chunks (400 characters, 60-character overlap)
3. Each chunk is embedded using `embeddinggemma` via Ollama
4. Embeddings are stored in a persistent **ChromaDB** collection
5. On every page request, the URL path is used as a query
6. The top 3 most relevant chunks are retrieved and injected into the prompt

---

## Known limitations

- Small local models (qwen3:1.7b) occasionally ignore formatting instructions or produce inconsistent output
- Face detection in FaceMe extension requires Chrome's experimental web platform features flag
- Domain memory and RAG context are not persisted across server restarts (in-memory dict)
- No authentication — intended for local single-user use only

---

## Models used

| Model | Purpose |
|-------|---------|
| `qwen3:1.7b` | HTML page generation |
| `embeddinggemma` | Document embeddings for RAG |
