import re
import json
import uuid
import os
import requests as http_requests
from typing import Optional
from urllib.parse import urlparse
from flask import Flask, request, jsonify, Response, render_template, stream_with_context, session
import ollama
import chromadb

app = Flask(__name__)
app.secret_key = "generative-browser-secret"

# ── Model config ───────────────────────────────────────
MODEL       = "qwen3:1.7b"
EMBED_MODEL = "embeddinggemma:latest"

print(f"[config] Ollama — model: {MODEL}")

# ── ChromaDB (persistent) ──────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    return chroma_client.get_or_create_collection(
        name="site_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

def get_trend_collection():
    return chroma_client.get_or_create_collection(
        name="design_trends",
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
# ── Design trends (developer-controlled RAG knowledge) ─
DESIGN_TREND_DOCS = [
    ("trend_modern",
     "Modern design trend: clean professional aesthetic with subtle shadows and rounded corners. "
     "Best for professional, formal, structured users who prefer detailed or bullet-point content at intermediate or expert level. "
     "CSS style: ample white space, subtle box-shadows, rounded corners, sans-serif font, polished two-tone color scheme."),
    ("trend_glassmorphism",
     "Glassmorphism design trend: frosted glass effect with blur and transparency. "
     "Best for casual or playful users who enjoy detailed or story-driven content. Creative and visual audiences. "
     "CSS style: backdrop-filter blur, semi-transparent rgba backgrounds, soft glowing borders, layered depth on gradient background."),
    ("trend_brutalism",
     "Brutalism design trend: bold raw aesthetic with thick borders and high contrast. "
     "Best for expert, dry, factual users who prefer short and direct content. Tech-savvy or artistic audiences. "
     "CSS style: thick black borders, raw bold typography, high-contrast colors, asymmetric layouts, no rounded corners."),
    ("trend_neumorphism",
     "Neumorphism design trend: soft tactile UI with extruded elements. "
     "Best for professional users who prefer detailed structured content at intermediate level. "
     "CSS style: dual inset/outset box-shadows on monochromatic background, tactile raised buttons."),
    ("trend_cyberpunk",
     "Cyberpunk design trend: dark neon aesthetic with glowing effects. "
     "Best for playful, fun, casual users who enjoy storytelling and beginner-friendly content. Gaming and creative audiences. "
     "CSS style: pure black background, neon accent colors hot pink #ff2d78 and cyan #00fff7, glowing text-shadow, monospace font."),
    ("trend_minimalism",
     "Minimalism design trend: extreme simplicity with maximum whitespace. "
     "Best for expert users who prefer dry, short, factual and no-nonsense content. Professional or academic audiences. "
     "CSS style: white background, single accent color, generous whitespace, thin typography, zero decorative elements."),
    ("trend_retro",
     "Retro Y2K design trend: vibrant nostalgic aesthetic with bright colors and bubbly shapes. "
     "Best for beginner or casual users who enjoy playful and story-driven content. Fun and nostalgic audiences. "
     "CSS style: bright bubbly gradients, pixel-style fonts, vivid colors, decorative dividers, 2000s web nostalgia."),
    ("trend_bento",
     "Bento Grid design trend: modular card-based layout inspired by Japanese bento boxes. "
     "Best for users who prefer bullet points, structured scannable content, casual or professional tone at basics level. "
     "CSS style: CSS grid with mixed card sizes, clean section blocks, subtle color fills, organized modular structure."),
]

def seed_design_trends():
    """Seed design trend documents into ChromaDB at startup."""
    col = get_trend_collection()
    existing = set(col.get()["ids"]) if col.count() > 0 else set()
    added = 0
    for doc_id, text in DESIGN_TREND_DOCS:
        if doc_id in existing:
            continue
        emb = embed(text)
        if not emb:
            continue
        col.add(ids=[doc_id], embeddings=[emb], documents=[text], metadatas=[{"source": "design-trends"}])
        added += 1
    if added:
        print(f"[rag] seeded {added} design trend(s)")
    else:
        print("[rag] design trends already seeded")

def retrieve_design_trend(profile: dict) -> str:
    """Pick the best design trend for this user profile via RAG."""
    query = (
        f"{profile.get('tone', 'casual')} "
        f"{profile.get('reading_style', 'detailed')} "
        f"{profile.get('experience', 'basics')}"
    )
    col = get_trend_collection()
    if col.count() == 0:
        return ""
    emb = embed(query)
    if not emb:
        return ""
    results = col.query(query_embeddings=[emb], n_results=1)
    docs = results["documents"][0]
    return docs[0] if docs else ""

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

# ── Prompt builder (4T's structure) ───────────────────
def build_messages(url: str, domain: str, path: str,
                   context: Optional[str], profile: dict,
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
    rag_block = ""
    if rag_context:
        rag_block = (
            "\nThe following information was retrieved from the knowledge base. "
            "Ground the page content in these specific facts — do not invent details "
            "that contradict this material:\n"
            f"{rag_context}\n"
        )

    context_block = ""
    if context:
        context_block = (
            f"\nPrevious pages on {domain} established this brand identity — maintain it exactly:\n"
            f"{context}\n"
        )

    task = (
        f"Generate a complete HTML page for: {url}\n"
        f"{context_block}"
        f"{rag_block}"
        "The page must include: navigation bar, hero/header section, "
        "rich main content relevant to the domain, and a footer. "
        "Use inline <style> with a cohesive modern color scheme. "
        "Add at least 5 internal <a href='/path'> links. "
        "For ALL images use Pollinations.ai — format exactly like this: "
        "<img src='https://image.pollinations.ai/prompt/fresh+red+roses+in+a+flower+shop?width=800&height=400' alt='...'> "
        "Replace spaces with + in the prompt. Make each description specific and relevant to the page. "
        "Never use picsum.photos or any other image source."
    )

    # ── TONE ──────────────────────────────────────────
    tone = ts["tone"]

    # ── TARGET ────────────────────────────────────────
    target = ts["target"]

    # ── DESIGN ────────────────────────────────────────
    design_block = design_trend if design_trend else "Clean modern design with ample white space and a professional color scheme."

    # ── Assemble system prompt with explicit 4T labels ─
    system = (
        f"[TRAITS]\n{traits}\n\n"
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

    # RAG retrieval — query combines domain + path keywords
    rag_query   = f"{domain} {path.replace('/', ' ').replace('-', ' ')}"
    rag_context = retrieve(rag_query)

    # Design trend — chosen automatically via RAG based on user profile
    design_trend = retrieve_design_trend(profile)

    messages = build_messages(full_url, domain, path, context, profile, rag_context, design_trend)

    debug_prompt = (
        f"── RAG: RETRIEVED CONTENT ──────────────────\n"
        f"{rag_context or '(ingen dokumenter i vidensbasen)'}\n\n"
        f"── RAG: DESIGN TREND (auto-valgt) ─────────\n"
        f"{design_trend or '(ingen trend fundet)'}\n\n"
        f"── 4T SYSTEM PROMPT ────────────────────────\n"
        f"{messages[0]['content']}\n\n"
        f"── 4T USER PROMPT (TASK) ───────────────────\n"
        f"{messages[1]['content']}"
    )

    sid = str(uuid.uuid4())
    pending_streams[sid] = {"url": full_url, "domain": domain, "messages": messages}
    return jsonify({"stream_id": sid, "url": full_url, "debug_prompt": debug_prompt})

@app.route("/stream/<sid>")
def stream_page(sid: str):
    if sid not in pending_streams:
        return jsonify({"error": "Invalid stream ID"}), 404

    info     = pending_streams.pop(sid)
    domain   = info["domain"]
    messages = info["messages"]

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

            # Store domain context
            full_html = "".join(accumulated)
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

            # Inject image-repair script — fixes any broken img src using alt text via Pollinations
            img_fix = (
                "<script>"
                "document.querySelectorAll('img').forEach(function(img){"
                "  function fix(){"
                "    var desc=(img.alt||'website image').trim().replace(/\\s+/g,'+');"
                "    img.src='https://image.pollinations.ai/prompt/'+desc+'?width=800&height=400';"
                "    img.onerror=null;"
                "  }"
                "  img.onerror=fix;"
                "  if(img.complete&&img.naturalWidth===0) fix();"
                "});"
                "</script>"
            )
            yield f"data: {json.dumps({'chunk': img_fix})}\n\n"

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
    seed_design_trends()
    app.run(debug=True, threaded=True, port=5000)
