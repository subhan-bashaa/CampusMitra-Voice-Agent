import json
import os
import re
import sys
import time
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Cache to prevent duplicate processing for identical requests arriving within TTL
_REQUEST_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 4.0

def normalize_stt_transcript(raw_text: str) -> str:
    """Normalize common acoustic STT misrecognitions for MVGR College domain terms."""
    if not raw_text:
        return ""
    text = raw_text

    # 1. Remove rogue acoustic artifacts like 'indonesia'
    text = re.sub(r'\b(in\s+)?indonesia\b', '', text, flags=re.IGNORECASE).strip()

    # 2. Normalize 'free structure' -> 'fee structure'
    text = re.sub(r'\bfree\s+structure\b', 'fee structure', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwhat\s+is\s+the\s+free\b', 'what is the fee', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+free\s+of\b', 'the fee of', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfree\s+(details|schedule|tuition|hostel|transport)\b', r'fee \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(the\s+)?free\s+of\s+mvgr\b', 'fee of MVGR', text, flags=re.IGNORECASE)

    # 3. Normalize MVGR variants
    def _replace_engager(m):
        return 'MVGR College' if m.group(3) else 'MVGR'

    text = re.sub(r'\b(the\s+)?(engager|en\s*gager)\s*(college)?\b', _replace_engager, text, flags=re.IGNORECASE)
    text = re.sub(r'\b(manager|passenger|messenger)\s+college\b', 'MVGR College', text, flags=re.IGNORECASE)

    def _replace_mvgr(m):
        return 'MVGR College' if m.group(1) else 'MVGR'
    text = re.sub(r'\bm[\s\.]*v[\s\.]*[gj][\s\.]*r[\s\.]*(college)?\b', _replace_mvgr, text, flags=re.IGNORECASE)
    text = re.sub(r'\bmbgr\b', 'MVGR', text, flags=re.IGNORECASE)

    # 4. Normalize department acronyms
    text = re.sub(r'\b(c[\s\.]*s[\s\.]*e)\.?(?=\s|$|[,\?!])', 'CSE', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(e[\s\.]*c[\s\.]*e)\.?(?=\s|$|[,\?!])', 'ECE', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(e[\s\.]*e[\s\.]*e)\.?(?=\s|$|[,\?!])', 'EEE', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(i[\s\.]*o[\s\.]*t)\.?(?=\s|$|[,\?!])', 'IoT', text, flags=re.IGNORECASE)
    text = re.sub(r'\bai[\s\.]*(&|and)?[\s\.]*ml\b', 'AI & ML', text, flags=re.IGNORECASE)

    # 5. Clean spacing and proper noun capitalization
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'\bmvgr\s+college\b', 'MVGR College', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmvgr\b', 'MVGR', text, flags=re.IGNORECASE)

    return text

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import settings
from langchain_agent.rag import (
    rag,
    detect_intent,
    check_clarification_needed,
    OFFICE_DIRECTORY,
    COLLEGE_INTENTS,
)

SAMPLE_QUESTIONS = [
    {"category": "Admissions", "query": "How can I apply for admission?", "label": "Admission Process"},
    {"category": "Admissions", "query": "What is the CSE intake?", "label": "CSE Intake"},
    {"category": "Courses", "query": "What courses does MVGR offer?", "label": "Offered Courses"},
    {"category": "Courses", "query": "Does MVGR offer AI and ML?", "label": "AI & ML Branch"},
    {"category": "Fees", "query": "What is the B.Tech fee?", "label": "B.Tech Tuition Fee"},
    {"category": "Hostel", "query": "Does MVGR have hostel?", "label": "Hostel Facilities"},
    {"category": "Transport", "query": "Does MVGR provide transport?", "label": "Transport Routes"},
    {"category": "Placements", "query": "What are the latest placement figures?", "label": "Placement Stats"},
    {"category": "Placements", "query": "How does placement work?", "label": "Placement Policy"},
    {"category": "Academics", "query": "Is lateral entry available?", "label": "Lateral Entry"},
    {"category": "Library", "query": "Does MVGR have a library?", "label": "Library & Delnet"},
    {"category": "Contact", "query": "Who should I contact for admission?", "label": "Admission Contact"},
]


class CampusMitraHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler for CampusMitra Web & Voice Interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR / "static"), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json({
                "status": "online",
                "app": "CampusMitra",
                "full_title": "CampusMitra — AI-Powered Voice Assistant for College Queries",
                "tagline": "Your AI Companion for Everything Campus.",
                "institution": "MVGR College of Engineering (Autonomous)",
                "collection": settings.collection_name,
                "docs_dir": settings.docs_dir,
                "intents_count": len(COLLEGE_INTENTS),
            })
            return

        elif path == "/api/offices":
            self.send_json(OFFICE_DIRECTORY)
            return

        elif path == "/api/samples":
            self.send_json(SAMPLE_QUESTIONS)
            return

        elif path == "/api/intents":
            self.send_json(COLLEGE_INTENTS)
            return

        # Serve static files or index.html fallback
        if path == "/" or path == "":
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}

        if path == "/api/chat" or path == "/api/voice-query":
            raw_query = data.get("query", "").strip()
            request_id = data.get("request_id", "").strip()
            query = normalize_stt_transcript(raw_query)
            if not query:
                self.send_json({"error": "Empty query"}, status=HTTPStatus.BAD_REQUEST)
                return

            now = time.time()
            cache_key = request_id if request_id else f"q_{query.lower()}"

            # Purge expired cache entries
            expired = [k for k, (t, _) in _REQUEST_CACHE.items() if now - t > _CACHE_TTL_SECONDS]
            for k in expired:
                _REQUEST_CACHE.pop(k, None)

            # Deduplication: return cached response if request arrived within TTL
            if cache_key in _REQUEST_CACHE:
                _, cached_resp = _REQUEST_CACHE[cache_key]
                self.send_json(cached_resp)
                return

            # Check ambiguity
            is_ambiguous, clarification_prompt = check_clarification_needed(query)
            if is_ambiguous:
                resp_payload = {
                    "answer": clarification_prompt,
                    "is_clarification": True,
                    "intent": detect_intent(query),
                    "sources": [],
                    "context": "",
                }
                _REQUEST_CACHE[cache_key] = (now, resp_payload)
                self.send_json(resp_payload)
                return

            # Run RAG
            try:
                answer, context, intent = rag(query)
                sources = []
                if context:
                    for part in context.split("---"):
                        part = part.strip()
                        if part:
                            sources.append(part[:300])

                resp_payload = {
                    "answer": answer,
                    "is_clarification": False,
                    "intent": intent,
                    "sources": sources,
                    "context": context,
                }
                _REQUEST_CACHE[cache_key] = (now, resp_payload)
                self.send_json(resp_payload)
            except Exception as e:
                self.send_json({
                    "answer": f"I encountered an error retrieving that information: {str(e)}",
                    "error": str(e),
                    "intent": "ERROR",
                }, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        elif path == "/api/escalate":
            name = data.get("name", "Student")
            phone = data.get("phone", "")
            topic = data.get("topic", "General")
            query = data.get("query", "")

            ticket_id = "CM-" + str(abs(hash(name + query)) % 100000).zfill(5)
            self.send_json({
                "ticket_id": ticket_id,
                "status": "registered",
                "message": f"Enquiry ticket {ticket_id} registered successfully. The college office will contact you.",
            })
            return

        self.send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def send_json(self, data: dict | list, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run(host: str = "0.0.0.0", port: int = 8000):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, CampusMitraHandler)
    print(f"\n==================================================")
    print(f"  CampusMitra — MVGR College Voice & Web Assistant")
    print(f"  Running at: http://localhost:{port}")
    print(f"==================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down CampusMitra server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run(port=port)
