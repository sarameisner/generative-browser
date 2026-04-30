# Generative Browser

A Flask-based web application that simulates a fully functional browser — except every page is generated in real time by a local LLM via Ollama. Enter any URL (real or imaginary) and watch the AI write the HTML live.

Built as an exam project for the **LLM for Developers** course at Erhvervsakademi København.

---

## Features

- **Live HTML streaming** — pages are generated token by token and rendered progressively in a sandboxed iframe via Server-Sent Events (SSE)
- **Automatic design trend selection** — a RAG pipeline matches the user's style profile against a curated design trends PDF and injects the best-fitting aesthetic into every prompt
- **User style profiles** — choose reading style, tone, and experience level; the prompt changes visibly and produces different pages for the same URL
- **Domain memory** — the browser remembers brand identity (colors, fonts, tone) across pages on the same domain
- **4T's prompt structure** — all prompts are explicitly structured around Traits, Task, Tone, and Target
- **Back / Forward navigation** — full history stack with cached HTML for instant back navigation
- **Prompt debug panel** — collapsible panel showing exactly what was sent to the LLM, including retrieved RAG context
- **Admin panel** — developer-only panel at `/admin` for managing the knowledge base via Google Drive sync

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- The following Ollama models pulled:

```bash
ollama pull qwen3:1.7b
ollama pull embeddinggemma:latest
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/generative-browser.git
cd generative-browser

# 2. (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip3 install -r requirements.txt

# 4. Make sure Ollama is running
ollama serve
```

> **Note:** Google Drive sync requires a `credentials.json` file in the project root. This file is not included in the repository — get it from a group member and place it in the project folder. Once you have it, go to `/admin` and click **Sync fra Google Drive** to load the knowledge base content into your local ChromaDB.

---

## Running the app

```bash
python3 app.py
```

Open your browser and go to: **http://localhost:5000**

The admin panel is available at: **http://localhost:5000/admin**

---

## How to use

### Browsing
1. Set up your profile on the welcome screen (reading style, tone, experience level)
2. Type any URL in the address bar and press **Enter** — real or imaginary
3. Watch the page generate live — the design aesthetic is automatically chosen based on your profile

### Debug panel
Click **Prompt Debug** at the bottom of the screen to see:
- The design trend retrieved from the knowledge base
- The full 4T-structured system prompt sent to the LLM
- The user task prompt

### Admin panel (developers only)
Go to `/admin` to manage the knowledge base:
1. Click **Sync fra Google Drive** to download all files from the shared Drive folder and ingest them into ChromaDB
2. The status panel shows how many chunks and sources are currently in the knowledge base
3. Use **Ryd vidensbase** to clear all content

---

## Architecture

```
User enters URL
      │
      ▼
Flask /generate
  ├── Reads user profile from session
  ├── Retrieves design trend from ChromaDB (RAG — profile as query)
  ├── Builds 4T-structured prompt with design instructions
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

Developer (admin panel)
  ├── Triggers /rag/sync-drive
  ├── Flask downloads files from Google Drive (service account)
  ├── Extracts text, chunks, embeds and stores in ChromaDB
  └── Design trend PDF is now queryable at generation time
```

---

## Project structure

```
generative-browser/
├── app.py              # Flask backend — routes, RAG, prompt builder, Drive sync
├── requirements.txt    # Python dependencies
├── credentials.json    # Google service account key (not in repo)
├── chroma_db/          # Persistent ChromaDB vector store (auto-created, not in repo)
└── templates/
    ├── index.html      # Single-page frontend — browser UI, profile modal, debug panel
    └── admin.html      # Developer admin panel — Drive sync and knowledge base management
```

---

## 4T's prompt structure

Every prompt is built around the 4T's framework:

| T | Description | Source |
|---|-------------|--------|
| **Traits** | Expert web designer, front-end developer, and copywriter | Hardcoded in system prompt |
| **Task** | Generate a complete HTML page for the given URL | URL + domain memory |
| **Tone** | How content should read — reading style and voice | User profile selection |
| **Target** | Who the content is written for — experience level | User profile selection |
| **Design** | Visual aesthetic and layout instructions | RAG — matched from design trends PDF |

---

## RAG pipeline

We use a design trends PDF stored in a shared Google Drive folder as the sole knowledge source. Only we as developers have access to the Drive folder and the admin panel.

1. We place a design trends PDF in the shared Google Drive folder
2. From the admin panel, we trigger a sync — Flask downloads the file via the Google Drive API using a service account
3. The text is extracted, split into chunks (400 characters, 60-character overlap) and embedded using `embeddinggemma:latest` via Ollama
4. Embeddings are stored in a persistent ChromaDB collection (`site_knowledge`)
5. On every page request, the user's profile (tone, reading style, experience level) is used as a query
6. The 2 most relevant chunks are retrieved and injected into the prompt as design instructions

A user with the profile "Playful, Story, Beginner" will typically match chunks describing Cyberpunk or Retro Y2K aesthetics, while "Dry, Short, Expert" matches Minimalism or Brutalism. The user never chooses a design trend directly — it is selected automatically and invisibly based on their preferences.

---

## Known limitations

- Small local models (qwen3:1.7b) occasionally ignore formatting instructions or produce inconsistent output
- Domain memory is not persisted across server restarts (in-memory dict)
- `chroma_db/` is local — team members each need to run a Drive sync to populate their own knowledge base
- No authentication on the admin panel — intended for local development use only

---

## Models used

| Model | Purpose |
|-------|---------|
| `qwen3:1.7b` | HTML page generation |
| `embeddinggemma:latest` | Document embeddings for RAG |
