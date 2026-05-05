import re
import json
import uuid
import random
import os
import io
import requests as http_requests
from typing import Optional
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode
from flask import Flask, request, jsonify, Response, render_template, stream_with_context, session
import ollama
import chromadb
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

app = Flask(__name__)
app.secret_key = "generative-browser-secret"

# ── Google Drive config ────────────────────────────────
DRIVE_FOLDER_ID   = "1qcvx55UkGe3y92f2tLN3S0UTWOm3KHbz"
CREDENTIALS_FILE  = os.path.join(os.path.dirname(__file__), "credentials.json")
DRIVE_SCOPES      = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds)

def sync_from_drive() -> dict:
    """Download all .txt and .pdf files from the shared Drive folder and ingest into RAG."""
    service = get_drive_service()
    col     = get_collection()

    # List files in folder
    results = service.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get("files", [])

    added_total = 0
    synced = []
    for f in files:
        name     = f["name"]
        mime     = f["mimeType"]
        file_id  = f["id"]

        # Download file content
        try:
            if mime == "application/pdf":
                request_dl = service.files().get_media(fileId=file_id)
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request_dl)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                buf.seek(0)
                import pypdf
                reader  = pypdf.PdfReader(buf)
                content = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
            else:
                request_dl = service.files().get_media(fileId=file_id)
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request_dl)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                content = buf.getvalue().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[drive] skipping {name}: {e}")
            continue

        if not content.strip():
            continue

        # Remove old chunks for this source, then re-add
        existing = col.get(where={"source": name})
        if existing["ids"]:
            col.delete(ids=existing["ids"])

        chunks = chunk_text(content)
        added  = 0
        for i, chunk in enumerate(chunks):
            emb = embed(chunk)
            if not emb:
                continue
            col.add(
                ids=[f"{name}_{i}_{uuid.uuid4().hex[:6]}"],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": name}],
            )
            added += 1
        added_total += added
        synced.append({"file": name, "chunks": added})
        print(f"[drive] synced '{name}' → {added} chunks")

    return {"ok": True, "files_synced": len(synced), "chunks_added": added_total, "details": synced}

# ── Model config ───────────────────────────────────────
MODEL       = "Ravishka/Miku"
EMBED_MODEL = "embeddinggemma:latest"
IMAGE_MODEL = "gptimage"

print(f"[config] Ollama — model: {MODEL}")
print(f"[config] Image model: {IMAGE_MODEL} (via Pollinations)")

# ── ChromaDB (persistent) ──────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    return chroma_client.get_or_create_collection(
        name="site_knowledge",
        metadata={"hnsw:space": "cosine"}
    )


# ── Embedding via Ollama ───────────────────────────────
def embed(text: str) -> list:
    try:
        resp = http_requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:2000]},
            timeout=30
        )
        return resp.json().get("embedding", [])
    except Exception:
        return []

# ── Text chunking ──────────────────────────────────────
def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

# ── RAG retrieval ──────────────────────────────────────
def retrieve(query: str, n: int = 3) -> str:
    col = get_collection()
    count = col.count()
    if count == 0:
        return ""
    emb = embed(query)
    if not emb:
        return ""
    results = col.query(query_embeddings=[emb], n_results=min(n, count))
    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    if not docs:
        return ""
    parts = [f"[Kilde: {m.get('source','?')}]\n{d}" for d, m in zip(docs, metas)]
    return "\n\n---\n\n".join(parts)

# ── Domain & session state ─────────────────────────────
domain_contexts = {}
pending_streams = {}
generated_images = {}
generated_image_cache = {}

# ── Profile helpers ────────────────────────────────────
READING_STYLES = {
    "short":    "Keep all content concise and to the point — no padding, no filler.",
    "detailed": "Be thorough and detailed with rich, in-depth explanations and examples.",
    "bullets":  "Present ALL content using bullet points, numbered lists, and scannable structure.",
    "story":    "Present content as a flowing narrative story — engaging, with a beginning, middle, and end.",
}
TONES = {
    "casual":       "Use casual, warm, conversational language — like talking to a friend.",
    "professional": "Use formal, polished, professional language throughout.",
    "playful":      "Be fun, playful, and inject humor and personality into every section.",
    "dry":          "Be dry, factual, and matter-of-fact — no filler, no emotion.",
}
EXPERIENCE_LEVELS = {
    "beginner": "The reader is a complete beginner — define all terms, explain from scratch.",
    "basics":   "The reader knows the basics — skip introductions, explain advanced concepts.",
    "expert":   "The reader is an expert — use technical terminology, skip all basics.",
}
def retrieve_design_trend(profile: dict) -> str:
    """Pick the best design trend from the Drive knowledge base based on user profile."""
    query = (
        f"design trend for {profile.get('tone', 'casual')} "
        f"{profile.get('reading_style', 'detailed')} "
        f"{profile.get('experience', 'basics')} user"
    )
    col = get_collection()
    if col.count() == 0:
        return ""
    emb = embed(query)
    if not emb:
        return ""
    results = col.query(query_embeddings=[emb], n_results=2)
    docs = results["documents"][0]
    return "\n".join(docs) if docs else ""

def default_profile():
    return {"reading_style": "detailed", "tone": "casual", "experience": "basics"}

def profile_to_4ts(profile: dict) -> dict:
    """Map the user profile to explicit 4T's values."""
    return {
        "tone":   (
            READING_STYLES.get(profile.get("reading_style", "detailed"), "") + " "
            + TONES.get(profile.get("tone", "casual"), "")
        ).strip(),
        "target": EXPERIENCE_LEVELS.get(profile.get("experience", "basics"), ""),
    }

def parse_url(raw: str):
    raw = raw.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    p = urlparse(raw)
    return p.netloc, p.path or "/", raw

def build_image_urls(domain: str, path: str, count: int = 3) -> list:
    """
    Build deterministic image URLs using a text-to-image model endpoint.
    These URLs are injected into the page prompt so generated HTML can use
    domain-specific visuals instead of generic placeholders.
    """
    slug = f"{domain} {path.replace('/', ' ').replace('-', ' ')}".strip()
    base_prompt = f"high quality website hero photo for {slug}, professional lighting, modern style"
    urls = []
    for i in range(1, count + 1):
        seed = 1
        prompt = http_requests.utils.quote(f"{base_prompt}, variation {i}")
        remote_url = (
            "https://image.pollinations.ai/prompt/"
            f"{prompt}?model={IMAGE_MODEL}&width=1280&height=720&nologo=true&seed={seed}"
        )
        token = uuid.uuid4().hex
        generated_images[token] = {"url": remote_url}
        urls.append(f"/api/image/{token}")
    return urls

def resolve_image_token(token: str):
    cached = generated_image_cache.get(token)
    if cached:
        return cached

    entry = generated_images.get(token)
    if not entry:
        return None
    if isinstance(entry, dict):
        remote_url = entry.get("url", "")
    else:
        # Backward compatibility for any old in-memory values.
        remote_url = entry
    if not remote_url:
        return None

    parsed = urlsplit(remote_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    candidate_urls = [remote_url]
    seed_raw = query.get("seed", "1")
    try:
        seed = int(seed_raw)
    except ValueError:
        seed = 1
    for bump in (31, 97, 173):
        q_variant = dict(query)
        q_variant["seed"] = str(seed + bump)
        candidate_urls.append(
            urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(q_variant), parsed.fragment))
        )
    # Same prompt, provider default model fallback.
    q_no_model = dict(query)
    q_no_model.pop("model", None)
    if q_no_model != query:
        candidate_urls.append(
            urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(q_no_model), parsed.fragment))
        )
        for bump in (31, 97):
            q_variant = dict(q_no_model)
            q_variant["seed"] = str(seed + bump)
            candidate_urls.append(
                urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(q_variant), parsed.fragment))
            )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/*,*/*;q=0.8",
        "Referer": "",
    }
    for candidate in candidate_urls:
        for _ in range(2):
            try:
                resp = http_requests.get(candidate, timeout=35, headers=headers)
                if resp.status_code != 200 or not resp.content:
                    continue
                content_type = resp.headers.get("Content-Type", "image/jpeg")
                generated_image_cache[token] = (resp.content, content_type)
                return generated_image_cache[token]
            except Exception:
                continue
    return None

# ── Prompt builder (4T's structure) ───────────────────
def build_messages(url: str, domain: str, path: str,
                   context: Optional[str], profile: dict, image_urls: list,
                   rag_context: str = "", design_trend: str = "") -> list:
    """
    Prompt is structured around the 4T's framework:

    TRAITS  — who the LLM is and what it can do
    TASK    — what it must produce, incl. RAG context and domain memory
    TONE    — how content should read (from user reading_style + tone profile)
    TARGET  — who the content is written for (from user experience profile)
    """

    ts = profile_to_4ts(profile)

    # ── TRAITS ────────────────────────────────────────
    traits = (
        "You are an expert web designer, front-end developer, and copywriter. "
        "You specialise in building realistic, visually impressive websites for any domain. "
        "You write clean, semantic HTML5 with inline CSS and never output anything "
        "other than raw HTML — no markdown, no code fences, no commentary. "
        "Always start output directly with <!DOCTYPE html>."
    )

    # ── TASK ──────────────────────────────────────────
    context_block = ""
    if context:
        context_block = (
            f"\nPrevious pages on {domain} established this brand identity — maintain it exactly:\n"
            f"{context}\n"
        )

    task = (
        f"Generate a complete HTML page for: {url}\n"
        f"{context_block}"
        f"The page must be ABOUT {domain} — invent realistic, creative content that fits that domain. "
        f"Do NOT write about web design, design trends, or anything meta. "
        f"Write as if this is a real website for {domain}.\n\n"
        "STRUCTURE RULES — follow these strictly:\n"
        "- Navigation bar with max 4 links to subpages\n"
        "- Hero/header section\n"
        "- Rich main content relevant to the domain\n"
        "- Footer with generic copyright only — do NOT include any real person's name, email, or personal details\n"
        + (
            f"- Exactly 1 image. Use ONLY this URL: {image_urls[0]}\n"
            if image_urls else
            "- Exactly 1 image. Use: https://picsum.photos/800/400?random=1\n"
        )
        +
        "- Max 4 internal <a href='/path'> links\n"
        "- Reuse layout components and color palette for consistency\n"
        "- Prioritise quality and depth over quantity — fewer, better sections\n"
        "- Use inline <style> — no external CSS or JS libraries\n"
    )

    # ── TONE ──────────────────────────────────────────
    tone = ts["tone"]

    # ── TARGET ────────────────────────────────────────
    target = ts["target"]

    # ── PERSONALITY — map tone to a writing mode ──────
    tone_val = profile.get("tone", "casual")
    personality_map = {
        "casual":   (
            "Write in a warm, charming voice — a little cheeky, always engaging. "
            "The text should feel like it's gently flirting with the reader. "
            "Clever word choices, light wit, never cold."
        ),
        "professional": (
            "Write with commanding confidence — sharp, authoritative, like it runs the place. "
            "Every sentence earns its spot. No filler."
        ),
        "playful":  (
            "Go unhinged (in a good way). Be chaotic, funny, wildly creative. "
            "Subvert what the reader expects. Make them laugh or raise an eyebrow. "
            "The content should feel alive and slightly unpredictable."
        ),
        "dry":      (
            "Be deadpan and economical. Say exactly what needs to be said, nothing more. "
            "Dry humor is welcome — the kind that lands without announcing itself. "
            "Cold, efficient, oddly compelling."
        ),
    }
    personality = personality_map.get(tone_val, "")

    # ── DESIGN ────────────────────────────────────────
    # Design trend is used as VISUAL STYLE ONLY — do not reproduce its text as page content
    if design_trend:
        design_block = (
            "Apply the following as your visual and aesthetic direction ONLY. "
            "Do not write about these design concepts — use them to shape CSS, layout, and mood:\n"
            + design_trend
        )
    else:
        design_block = "Clean modern design with ample white space and a professional color scheme."

    # ── Assemble system prompt with explicit 4T labels ─
    system = (
        f"[TRAITS]\n{traits}\n\n"
        f"[PERSONALITY]\n{personality}\n\n"
        f"[TONE]\n{tone}\n\n"
        f"[TARGET]\n{target}\n\n"
        f"[DESIGN]\n{design_block}"
    )

    user = f"/no_think\n[TASK]\n{task}\n\nStart with <!DOCTYPE html> now:"

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

# ══════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

# ── Profile ────────────────────────────────────────────
@app.route("/api/profile", methods=["GET"])
def get_profile():
    return jsonify(session.get("profile", None))

@app.route("/api/profile", methods=["POST"])
def set_profile():
    data = request.get_json(force=True)
    profile = {
        "reading_style": data.get("reading_style") if data.get("reading_style") in READING_STYLES    else "detailed",
        "tone":          data.get("tone")           if data.get("tone")           in TONES            else "casual",
        "experience":    data.get("experience")     if data.get("experience")     in EXPERIENCE_LEVELS else "basics",
    }
    session["profile"] = profile
    return jsonify({"ok": True, "profile": profile})

@app.route("/api/image/<token>")
def proxy_generated_image(token: str):
    resolved = resolve_image_token(token)
    if not resolved:
        if token not in generated_images:
            return jsonify({"error": "Unknown image token"}), 404
        return jsonify({"error": "Image upstream unavailable"}), 502
    body, content_type = resolved
    return Response(
        body,
        mimetype=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )

# ── RAG management ─────────────────────────────────────
@app.route("/rag/status")
def rag_status():
    col = get_collection()
    return jsonify({"count": col.count(), "embed_model": EMBED_MODEL})

@app.route("/rag/list")
def rag_list():
    col = get_collection()
    if col.count() == 0:
        return jsonify({"sources": [], "count": 0})
    all_items = col.get(include=["metadatas"])
    sources   = sorted(set(m.get("source", "?") for m in all_items["metadatas"]))
    return jsonify({"sources": sources, "count": col.count()})

@app.route("/rag/upload", methods=["POST"])
def rag_upload():
    col = get_collection()

    # File upload
    if "file" in request.files:
        f        = request.files["file"]
        filename = f.filename or "upload"
        if filename.lower().endswith(".pdf"):
            import pypdf
            reader  = pypdf.PdfReader(f)
            content = "\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )
        else:
            content = f.read().decode("utf-8", errors="ignore")
        source = filename

    # JSON text paste
    else:
        data    = request.get_json(force=True)
        content = data.get("text", "").strip()
        source  = data.get("source", "manual")

    if not content:
        return jsonify({"error": "No content provided"}), 400

    chunks = chunk_text(content)
    added  = 0
    for i, chunk in enumerate(chunks):
        emb = embed(chunk)
        if not emb:
            continue
        col.add(
            ids=[f"{source}_{i}_{uuid.uuid4().hex[:6]}"],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"source": source}],
        )
        added += 1

    return jsonify({"ok": True, "chunks_added": added, "source": source})

@app.route("/rag/clear", methods=["POST"])
def rag_clear():
    chroma_client.delete_collection("site_knowledge")
    return jsonify({"ok": True})

@app.route("/rag/sync-drive", methods=["POST"])
def rag_sync_drive():
    try:
        result = sync_from_drive()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Page generation ────────────────────────────────────
@app.route("/generate", methods=["POST"])
def generate():
    data    = request.get_json(force=True)
    raw_url = (data.get("url") or "").strip()
    if not raw_url:
        return jsonify({"error": "URL required"}), 400

    domain, path, full_url = parse_url(raw_url)
    profile     = session.get("profile", default_profile())
    context     = domain_contexts.get(domain)
    image_urls  = build_image_urls(domain, path, count=1)

    # RAG retrieval — query combines domain + path keywords
    rag_query   = f"{domain} {path.replace('/', ' ').replace('-', ' ')}"
    rag_context = retrieve(rag_query)

    # Design trend — chosen automatically via RAG based on user profile
    # (only the design trends PDF is in the knowledge base — no content RAG)
    design_trend = retrieve_design_trend(profile)

    messages = build_messages(full_url, domain, path, context, profile, image_urls, rag_context, design_trend)

    debug_prompt = (
        f"── RAG: DESIGN TREND (auto-valgt) ─────────\n"
        f"{design_trend or '(ingen trend fundet)'}\n\n"
        f"── IMAGE MODEL URLS ({IMAGE_MODEL}) ─────────\n"
        f"{chr(10).join(image_urls)}\n\n"
        f"── 4T SYSTEM PROMPT ────────────────────────\n"
        f"{messages[0]['content']}\n\n"
        f"── 4T USER PROMPT (TASK) ───────────────────\n"
        f"{messages[1]['content']}"
    )

    sid = str(uuid.uuid4())
    pending_streams[sid] = {"url": full_url, "domain": domain, "path": path, "messages": messages, "image_urls": image_urls}
    return jsonify({"stream_id": sid, "url": full_url, "debug_prompt": debug_prompt})

@app.route("/stream/<sid>")
def stream_page(sid: str):
    if sid not in pending_streams:
        return jsonify({"error": "Invalid stream ID"}), 404

    info       = pending_streams.pop(sid)
    domain     = info["domain"]
    path       = info.get("path", "/")
    messages   = info["messages"]
    image_urls = info.get("image_urls", [])

    def generate_sse():
        accumulated    = []
        fence_stripped = False
        buffer         = ""
        in_think       = False

        try:
            stream = ollama.chat(model=MODEL, messages=messages, stream=True)
            for chunk in stream:
                raw = chunk["message"]["content"]
                if not raw:
                    continue

                buffer += raw

                if in_think:
                    end = buffer.find("</think>")
                    if end == -1:
                        buffer = ""
                        continue
                    buffer   = buffer[end + 8:]
                    in_think = False

                if "<think>" in buffer:
                    start    = buffer.find("<think>")
                    pre      = buffer[:start]
                    buffer   = buffer[start + 7:]
                    in_think = True
                    raw      = pre
                    if not raw.strip():
                        buffer = ""
                        continue
                else:
                    raw    = buffer
                    buffer = ""

                if not fence_stripped:
                    if len(raw) >= 20 or "\n" in raw:
                        raw            = re.sub(r"^```\w*\n?", "", raw)
                        fence_stripped = True
                    else:
                        buffer = raw
                        continue

                raw = raw.replace("```", "")
                if not raw:
                    continue

                accumulated.append(raw)
                yield f"data: {json.dumps({'chunk': raw})}\n\n"

            if buffer:
                buffer = re.sub(r"^```\w*\n?", "", buffer).replace("```", "")
                if buffer:
                    accumulated.append(buffer)
                    yield f"data: {json.dumps({'chunk': buffer})}\n\n"

            full_html = "".join(accumulated)

            # Guarantee at least 1 image — inject one before </body> if model forgot
            if image_urls and not re.search(r"<img\b", full_html, re.I):
                inject = (
                    f'<div style="max-width:900px;margin:40px auto;padding:0 24px">'
                    f'<img src="{image_urls[0]}" style="width:100%;border-radius:8px" alt=""></div>'
                )
                full_html = full_html.replace("</body>", inject + "</body>")
                accumulated = [full_html]
                yield f"data: {json.dumps({'chunk': inject})}\n\n"

            # Pre-fetch the pre-built image tokens so they're cached when the browser requests them
            if image_urls:
                for image_url in image_urls:
                    token = image_url.rsplit("/", 1)[-1]
                    resolve_image_token(token)

            # Store domain context
            title_m   = re.search(r"<title[^>]*>(.*?)</title>", full_html, re.I | re.S)
            style_m   = re.search(r"<style[^>]*>(.*?)</style>",  full_html, re.I | re.S)
            title         = title_m.group(1).strip() if title_m else domain
            style_excerpt = style_m.group(1).strip()[:800] if style_m else ""
            colors        = list(dict.fromkeys(
                re.findall(r"#[0-9a-fA-F]{3,6}|rgba?\([^)]+\)", style_excerpt)
            ))[:10]
            domain_contexts[domain] = (
                f"Site title: {title}\n"
                f"Color palette: {', '.join(colors)}\n"
                f"CSS (excerpt):\n{style_excerpt[:600]}"
            )

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as exc:
            print(f"[stream error] {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
