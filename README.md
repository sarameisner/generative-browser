# Generative Browser

A Flask-based web application that simulates a fully functional browser — except every page is generated in real time by a local LLM via Ollama. Enter any URL, real or imaginary, and watch the AI write the HTML live. Each generated page also receives domain-aware images from an external text-to-image model (Flux via Pollinations.ai).

Built as an exam project for the **LLM for Developers** course at Erhvervsakademi København.

---

## Features

- **Live HTML streaming** — pages are generated token by token and rendered progressively in a sandboxed iframe via Server-Sent Events (SSE)
- **RAG pipeline** — documents are stored in a shared Google Drive folder and synced into a local ChromaDB knowledge base via the admin panel. Syncing requires a valid `credentials.json` file - without it, the existing local ChromaDB can still be queried, but cannot be updated. On every page request, relevant chunks are retrieved from the local ChromaDB and injected into the prompt before generation.
- **User style profiles** — choose reading style, tone preference, and experience level; the prompt adapts and produces different pages for the same URL
- **Domain memory** — the browser remembers brand identity (colors, fonts, tone) across pages on the same domain within a session
- **AI image generation** — each generated page receives domain-aware image URL from Flux (text-to-image model) via Pollinations.ai, used in hero and content sections
- **4T's prompt structure** — all prompts are explicitly structured around Traits, Task, Tone, and Target
- **Back / Forward navigation** — full history stack with cached HTML for instant back navigation
- **Prompt debug panel** — collapsible panel showing the fill 4T-structured promt, retrieved RAG chunks, and design trend sent to the LLM

---

## Prerequisites

- Python 3.9-3.12
- [Ollama](https://ollama.com) installed and running
- The following Ollama models pulled:

```bash
ollama pull qwen3:1.7b
ollama pull embeddinggemma
```

- *(Optional)* `credentials.json` in the project root - a Google Drive service account credentials file, required only if you want to sync or update the RAG knowledge base from Google Drive. Without it, the existing local ChromaDB still works.

---

## Google Drive credentials *(optional)*

To sync the RAG knowledge base from Google Drive, you need a `credentials.json` file in the project root. This is a Google service account credentials file tied to the shared Drive folder.

Place it here:

```
generative-browser/
└── credentials.json
```

The file should contain the following - replace the placeholder values with the actual credentials you received:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

```

This file is listed in `.gitignore` and will never be committed to the repository. The actual `credentials.json` is shared separately with project members.

Without this file the app runs normally, but the **Sync fra Google Drive** button in the admin panel will not work. The existing local ChromaDB knowledge base can still be queried.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/generative-browser.git
cd generative-browser

# 2. (Optional) Add Google Drive credentials
# See "Google Drive credentials" section above for details
# Place credentials.json in the project root if you want Drive sync

# 3. Install Python dependencies
# Installs: flask, ollama, chromadb, pypdf, requests, google-api-python-client, google-auth
pip install -r requirements.txt

# 4. Make sure Ollama is running
ollama serve

```

---

## Running the app

```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

> **Note:** Use `python3` instead of `python` if your system requires it.

---

## How to use

### Browsing
1. Set up your profile on the welcome screen — choose your reading style, tone preference, and experience level
2. Type any URL in the address bar and press **Enter** — real or imaginary (e.g. `flowers.com`, `mars-colony.gov`, `luna-blomster.dk`)
3. Watch the page generate live
4. Click any link on a generated page — it will generate a new page for that URL

### RAG — Knowledge Base & Admin panel
The admin panel is available at **http://localhost:5000/admin**. From here you can manage the RAG knowledge base.

- **Sync fra Google Drive** - downloads all files from the shared Google Drive folder and ingests them into ChromaDB. Requires `credentials.json`
- **Ryd vidensbase** — clears the entire local knowledge base. Useful if files have been removed from Google Drive
- **Vidensbase** — shows all currently loaded sources and chunk count

> **Important:** Always sync from Google Drive after files in the Drive folder have been added, edited, or removed.

The knowledge base is stored locally in `chroma_db/` and persists across restarts. You do **not** need `credentials.json` to query the existing knowledge base — only to sync or update it from Google Drive.

The shared Google Drive folder (`RAG_generative-browser`) currently contains two files:
- `user_profiles.txt`
- `web_design_trends_rag.md`

To add more knowledge: add a `.txt`, `.md`, or `.pdf` file to the Drive folder, then sync from the admin panel.

**Example:** Add a text file describing a fictional flower shop to the Drive folder. Sync from the admin panel. Visit `luna-blomster.dk` — the generated page will include your specific products, prices, and details instead of generic AI content.

### Debug panel
Click **Prompt Debug** at the bottom of the screen to see:
- The design trend retrieved from the knowledge base
- The top RAG chunks retrieved for the current URL
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
  ├── Retrieves relevant RAG chunks via ChromaDB
  ├── Builds 4T-structured prompt
  └── Returns stream-ID
      │
      ▼
Flask /stream/<id>  (SSE)
  ├── Calls Ollama with streaming (qwen3:1.7b)
  ├── Filters <think> blocks and markdown
  ├── Send HTML chunks as SSE events
  └── Stores domain context after completion
      │
      ▼
Browser (JavaScript)
  ├── Opens EventSource to /stream
  ├── Writes HTML chunks progressively to the iframe as they arrive
  ├── Hides loading overlay on first chunk
  └── Saves completed HTML to history stack

External services
  ├── Pollinations.ai - image generation (Flux)
  └── Google Drive API - syncing documents to ChromaDB
```

---

## Project structure

```
generative-browser/
├── chroma_db/          # Persistent ChromaDB vector store (auto-created)
├── templates/
    ├── admin.html      # Admin panel - knowledge base management and Google Drive sync
    └── index.html      # Single-page frontend — browser UI, profile modal, debug panel
├── .gitignore          # Excludes credentials.json and other sensitive files
├── app.py              # Flask backend — routes, RAG pipeline, prompt builder, SSE streaming
├── credentials.json    # Google Drive service account credentials (not committed, see prerequisites)
└── requirements.txt    # Python dependencies - Flask, Ollama, ChromaDB, Google Drive API
```

---

## 4T's prompt structure

Every prompt is built around the 4T's framework:

| T | Description | Source |
|---|-------------|--------|
| **Traits** | Expert web designer, front-end developer, and copywriter - outputs only raw HTML, no markdown or commentary | Hardcoded in system prompt |
| **Task** | Generate a complete, domain-appropriate HTML page for the given URL - includes structure rules, image URL, internal link limits, and brand identity from domain memory | URL + RAG chunks + domain memory + image proxy URL |
| **Tone** | How content should read — maps to a named writing voice (casual, professional, playful, dry) | User profile selection (tone field) |
| **Target** | Who the content is written for — experience level and reading style | User profile selection (via profile_to_4ts()) |

A fifth block, `[DESIGN]`, is injected alongside the 4T's. It is not a content instruction — it passes the RAG-retrieved design trend as visual and aesthetic direction only (CSS, layout, mood), explicitly instructing the model not to write *about* design concepts.

---

## RAG pipeline

1. Documents are added to the shared Google Drive folder (`RAG_generative-browser`) and synced into ChromaDB via the admin panel (requires `credentials.json`)
2. Text is split into chunks of 400 characters with a 60-character overlap
3. Each chunk is embedded using `embeddinggemma` via Ollama
4. Embeddings are stored in a persistent ChromaDB collection (`site_knowledge`)
5. On every page request, the domain and URL path are combined into a search query
6. The top 3 most relevant chunks are retrieved and injected into the prompt

---

## Known limitations

- Small local models (`qwen3:1.7b`) occasionally ignore formatting instructions or produce inconsistent output
- Domain memory is not persisted across server restarts (in-memory dict)
- No authentication — intended for local single-user use only
- Image generation depends on external Pollinations.ai API — if the service is down, images will not load
- Google Drive sync requires a valid `credentials.json` — without it, the knowledge base cannot be updated from Drive
- PDF text extraction is basic — `pypdf` struggles with scanned PDFs or complex layouts, so RAG quality depends heavily on the source document format

---

## Models used

| Model | Purpose | Hosted |
|-------|---------|-------|
| `qwen3:1.7b` | HTML page generation (core LLM) | Local via Ollama |
| `embeddinggemma` | Document embeddings for RAG | Local via Ollama |
| `flux` | Text-to-image URLs injected into generated pages | External via Pollinations.ai API |
