import json
import os
import re
import sys
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

# Attempt to load heavy packages conditionally so tests and offline runs work seamlessly
try:
    from qdrant_client import QdrantClient
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from langchain_groq import ChatGroq
    HAS_LANGCHAIN_GROQ = True
except ImportError:
    HAS_LANGCHAIN_GROQ = False


# College Intent Taxonomy
COLLEGE_INTENTS = [
    "ADMISSIONS",
    "COURSES",
    "DEPARTMENTS",
    "FEES",
    "ACADEMICS",
    "EXAMINATIONS",
    "ACADEMIC_CALENDAR",
    "HOSTEL",
    "LIBRARY",
    "TRANSPORT",
    "CAMPUS_FACILITIES",
    "MEDICAL",
    "PLACEMENTS",
    "INTERNSHIPS",
    "SCHOLARSHIPS",
    "CERTIFICATES",
    "STUDENT_SERVICES",
    "CONTACT",
    "GENERAL_COLLEGE_QUERY",
    "UNKNOWN",
]

# Office escalation mapping
OFFICE_DIRECTORY = {
    "ADMISSIONS": {
        "office": "Admission Office",
        "phone": "08922-241749",
        "helplines": ["9704950954", "9441663201"],
        "email": "helpdesk@mvgrce.edu.in",
        "timings": "10:00 AM to 4:00 PM",
    },
    "FEES": {
        "office": "Accounts / Finance Section",
        "phone": "08922-241732 / 241039",
        "timings": "10:00 AM to 4:00 PM",
    },
    "EXAMINATIONS": {
        "office": "Examination Cell",
        "phone": "08922-241732",
        "timings": "9:30 AM to 4:30 PM",
    },
    "ACADEMICS": {
        "office": "Academic Section & Respective Department HOD Office",
        "phone": "08922-241732",
        "timings": "9:00 AM to 4:30 PM",
    },
    "HOSTEL": {
        "office": "Hostel Office / Chief Warden",
        "phone": "08922-241732",
        "facilities": "Boys Hostels & Maharani Vidyavathi Devi Hostel for Girls",
    },
    "TRANSPORT": {
        "office": "Transport Office",
        "phone": "08922-241732",
        "routes": "Vizag, Vizianagaram, Srikakulam, Bobbili, Saluru, L.Kota, Cheepurupalli",
    },
    "PLACEMENTS": {
        "office": "Training & Placement (T&P) Cell",
        "phone": "08922-241732",
        "email": "helpdesk@mvgrce.edu.in",
    },
    "CERTIFICATES": {
        "office": "Administration & Student Affairs Office",
        "phone": "08922-241732",
    },
}


def detect_intent(query: str) -> str:
    """Classify the user query into college-specific intents."""
    q = query.lower()

    if any(k in q for k in ["isn't in", "not in", "outside", "salary", "who is the president"]):
        return "UNKNOWN"
    if any(k in q for k in ["who should i contact", "helpline", "phone number", "contact number", "call", "whom to contact", "contact admissions", "reach"]):
        return "CONTACT"
    if any(k in q for k in ["fee", "tuition", "cost", "afrc", "dues", "payment"]):
        return "FEES"
    if any(k in q for k in ["admit", "admission", "apply", "application", "eapcet", "eamcet", "ecet", "convener", "management quota", "lateral entry", "join mvgr"]):
        return "ADMISSIONS"
    if any(k in q for k in ["course", "branch", "program", "intake", "seats", "b.tech", "m.tech", "mba", "cse", "ece", "eee", "mechanical", "civil", "chemical", "ai & ml", "ai and ml", "data science"]):
        return "COURSES"
    if any(k in q for k in ["placement", "job offer", "recruitment", "package", "companies", "t&p", "campus drive", "highest package", "average package"]):
        return "PLACEMENTS"
    if any(k in q for k in ["internship", "intern", "industrial training"]):
        return "INTERNSHIPS"
    if any(k in q for k in ["exam", "mid exam", "end exam", "viva", "result", "grade", "backlog", "hall ticket"]):
        return "EXAMINATIONS"
    if any(k in q for k in ["academic calendar", "semester start", "instruction period", "holiday list"]):
        return "ACADEMIC_CALENDAR"
    if any(k in q for k in ["hostel", "mess", "boarding", "room", "maharani vidyavathi", "sharing"]):
        return "HOSTEL"
    if any(k in q for k in ["transport", "bus", "route", "commute", "pickup", "drop"]):
        return "TRANSPORT"
    if any(k in q for k in ["library", "koha", "rfid", "delnet", "book", "reading hall", "journal"]):
        return "LIBRARY"
    if any(k in q for k in ["doctor", "medical", "first aid", "health", "hospital", "dispensary", "anm"]):
        return "MEDICAL"
    if any(k in q for k in ["scholarship", "jvd", "jagananna", "fee reimbursement", "financial aid"]):
        return "SCHOLARSHIPS"
    if any(k in q for k in ["certificate", "transcript", "bonafide", "custody", "tc", "transfer certificate", "migration"]):
        return "CERTIFICATES"
    if any(k in q for k in ["department", "faculty", "hod", "labs", "laboratories"]):
        return "DEPARTMENTS"
    if any(k in q for k in ["curriculum", "regulation", "cbcs", "syllabus", "credits", "autonomous"]):
        return "ACADEMICS"
    if any(k in q for k in ["canteen", "cafeteria", "rasamyai", "infrastructure", "campus size", "acres", "sports", "ground", "gym"]):
        return "CAMPUS_FACILITIES"
    if any(k in q for k in ["phone", "email", "address", "location", "contact", "office"]):
        return "CONTACT"
    if any(k in q for k in ["mvgr", "college", "autonomous", "accreditation", "naac", "nba", "jntugv", "aicte"]):
        return "GENERAL_COLLEGE_QUERY"

    return "UNKNOWN"


def check_clarification_needed(query: str) -> tuple[bool, str]:
    """
    Check if a query is ambiguous and requires immediate clarification before answering.
    Examples from requirements:
      - 'What is the fee?' -> asks tuition, hostel, or transport.
      - 'What about CSE?' -> asks admission, fees, eligibility, or course itself.
      - 'When are exams?' -> asks batch and semester.
    """
    q = query.strip().lower()
    clean_q = re.sub(r"[^\w\s]", "", q)

    if clean_q in ["what is the fee", "fee", "fees", "fee details", "how much is the fee"]:
        return (
            True,
            "Sure. Do you mean tuition fee, hostel fee, or transport fee?",
        )

    if clean_q in ["what about cse", "cse", "tell me about cse", "what about ece", "what about mechanical"]:
        branch = clean_q.split()[-1].upper()
        return (
            True,
            f"Sure. Are you asking about {branch} admission, fees, eligibility, intake, or the course details?",
        )

    if clean_q in ["when are the exams", "when are exams", "exam dates", "when is my exam"]:
        return (
            True,
            "MVGR publishes exam dates by batch and semester. Could you please specify your admitted batch year and current semester?",
        )

    return (False, "")


def retrieve_from_cache(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant chunks from cached JSON file using keyword and term overlap."""
    cache_file = Path(settings.qdrant_path) / "mvgr_chunks.json"
    if not cache_file.exists():
        # Look in docs if cache not yet created
        return []

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception:
        return []

    words = set(re.findall(r"\w+", query.lower()))
    scored_chunks = []

    for chunk in chunks:
        text = chunk.get("chunk_text", "").lower()
        chunk_words = set(re.findall(r"\w+", text))
        overlap = len(words.intersection(chunk_words))

        # Boost exact phrase matches
        score = float(overlap)
        if query.lower() in text:
            score += 5.0
        for w in words:
            if len(w) > 3 and w in text:
                score += 1.5

        if score > 0:
            scored_chunks.append({
                "source": chunk.get("source", "MVGR KB"),
                "headings": chunk.get("headings", []),
                "content": chunk.get("content", ""),
                "chunk_text": chunk.get("chunk_text", ""),
                "page": chunk.get("page", 1),
                "score": round(score, 4),
            })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


def retrieve(
    client,
    embedder,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Embed the query and return the top-k most similar chunks from Qdrant, falling back to cache."""
    if client is not None and embedder is not None:
        try:
            query_vector = embedder.encode(query).tolist()
            hits = client.query_points(
                collection_name=settings.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            if hits and hasattr(hits, "points") and hits.points:
                return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]
        except Exception as e:
            print(f"[RAG] Qdrant retrieval fallback triggered: {e}")

    return retrieve_from_cache(query, top_k=top_k)


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build a context string from retrieved chunks with source citations."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "MVGR Document")
        page = chunk.get("page", "")
        page_info = f" (Page {page})" if page else ""
        header = f"[Source {i}: {source}{page_info}]"
        parts.append(f"{header}\n{chunk.get('chunk_text', chunk.get('content', ''))}")
    return "\n\n---\n\n".join(parts)


# Verified Knowledge Base Responses for Direct Spoken Accuracy
# (Adheres strictly to the 3 MVGR PDFs without hallucination)
DIRECT_KB_FACTS = {
    "courses": (
        "MVGR offers B.Tech programs in CSE (300 intake), CSE AI & ML (240), Data Science (60), "
        "IoT & Cyber Security including Blockchain (60), CS & IT (120), ECE (180), Mechanical (120), "
        "EEE (60), Chemical (60), and Civil (60), plus M.Tech Structural Engineering (18) and MBA (60)."
    ),
    "cse_intake": "The current admissions information lists 300 seats for B.Tech Computer Science and Engineering (CSE).",
    "ai_ml": "Yes, MVGR offers B.Tech CSE with AI and Machine Learning, with an intake of 240 seats.",
    "data_science": "Yes, MVGR offers B.Tech CSE in Data Science with an intake of 60 seats.",
    "iot": "Yes, MVGR offers B.Tech CSE in IoT & Cyber Security including Blockchain Technology, with an intake of 60 seats.",
    "btech_fee": (
        "The official fee schedule lists ₹76,900 for B.Tech tuition under the 2025–26 AFRC structure, "
        "as well as ₹76,900 for II B.Tech lateral entry. For current 2026–27 B-category fees, please verify the latest official notice."
    ),
    "mtech_mba_fee": "The 2025–26 AFRC fee structure lists ₹65,000 for M.Tech and ₹60,000 for MBA.",
    "admission_apply": (
        "Regular B.Tech convener admissions (70% quota) are through AP EAPCET. Lateral entry to second year is through ECET. "
        "Management quota accounts for 30%. You can contact the admission office at 08922-241749 or 9704950954, open 10 AM to 4 PM."
    ),
    "lateral_entry": "Yes, MVGR offers B.Tech lateral entry through ECET or the relevant state-level test with direct admission into second year.",
    "hostel": (
        "Yes, MVGR provides boys' hostels and the Maharani Vidyavathi Devi Hostel for girls, featuring 3-sharing and 4-sharing rooms, "
        "dining facilities, playgrounds, gym, Rasamyai canteen, and a medical room with an on-duty medical officer and ANM."
    ),
    "hostel_fee": "I don't have the current hostel fee information in my knowledge base. Please check with the Hostel Office or Chief Warden for the latest fee details.",
    "transport": (
        "Yes, MVGR provides college transport connecting Visakhapatnam, Vizianagaram, Srikakulam, Bobbili, Saluru, L.Kota, S.Kota, and Cheepurupalli. "
        "Current route numbers and schedules should be verified with the Transport Office."
    ),
    "bus_timing": "Bus timings vary by route and date. For today's specific route schedule, please contact the Transport Office directly.",
    "library": (
        "Yes, MVGR has a comprehensive library facility with 48,000 square feet (and 12,000 sq ft mentioned in About records), 400 seating capacity, "
        "12 departmental libraries, RFID-enabled services, KOHA software, and DELNET inter-library loan access."
    ),
    "placements_latest": (
        "For 2026, the MVGR homepage highlights 764 job offers across 122 companies (60 IT and 62 Core). "
        "For the 2025 reporting period, there were 810 job offers for 975 eligible students across 77 companies, with an average package of ₹5.2 lakh."
    ),
    "placement_process": (
        "MVGR's Training & Placement Policy governs campus recruitment through a placement committee, student registration, pre-placement talks, "
        "and employability training for on-campus and off-campus drives."
    ),
    "admission_contact": (
        "You can reach the MVGR Admission Office at 08922-241749, or helplines 9704950954 and 9441663201, "
        "or email helpdesk@mvgrce.edu.in. In-person visits are welcomed between 10:00 AM and 4:00 PM."
    ),
    "general_contact": (
        "MVGR College of Engineering is located at Vijayaram Nagar Campus, Chintalavasa, Vizianagaram, Andhra Pradesh 535005. "
        "General college contact numbers are 08922-241732 and 08922-241039."
    ),
    "medical": "Yes, MVGR has medical facilities with a qualified medical officer, an ANM, a dedicated medical room with two beds, stretcher, wheelchair, and first-aid kits.",
    "exam_batch_clarification": "MVGR publishes batch-specific academic calendars and examination notifications. To provide your exact exam schedule, please let me know your batch and current semester.",
}


def direct_mvgr_answer(query: str, intent: str) -> str | None:
    """Provides precise, zero-hallucination answers directly from verified MVGR knowledge."""
    q = query.lower()

    if any(k in q for k in ["isn't in", "not in", "outside", "salary", "who is the president", "tell me something that isn't"]):
        return (
            "I do not have that information available in my current college knowledge base. "
            "I am currently focused on MVGR college-related queries such as admissions, courses, fees, academics, hostels, transport, and placements."
        )

    if "course" in q or "program" in q or "branch" in q or "offer" in q:
        if "ai" in q or "ml" in q:
            return DIRECT_KB_FACTS["ai_ml"]
        if "data science" in q:
            return DIRECT_KB_FACTS["data_science"]
        if "iot" in q or "cyber" in q:
            return DIRECT_KB_FACTS["iot"]
        if "what courses" in q or "courses does mvgr" in q or "programs offered" in q:
            return DIRECT_KB_FACTS["courses"]

    if "cse intake" in q or "cse seats" in q or ("how many" in q and "cse" in q):
        return DIRECT_KB_FACTS["cse_intake"]

    if "ai and ml" in q or "ai & ml" in q or ("offer" in q and "ai" in q):
        return DIRECT_KB_FACTS["ai_ml"]

    if "lateral entry" in q:
        return DIRECT_KB_FACTS["lateral_entry"]

    if "apply for admission" in q or "how do i get" in q or "admission process" in q or "how to join" in q:
        return DIRECT_KB_FACTS["admission_apply"]

    if "b.tech fee" in q or "btech fee" in q or "tuition fee" in q:
        return DIRECT_KB_FACTS["btech_fee"]

    if "hostel fee" in q:
        return DIRECT_KB_FACTS["hostel_fee"]

    if "hostel" in q or "hostels" in q:
        return DIRECT_KB_FACTS["hostel"]

    if "transport" in q or "bus" in q or "commute" in q:
        if "timing" in q or "today" in q:
            return DIRECT_KB_FACTS["bus_timing"]
        return DIRECT_KB_FACTS["transport"]

    if "library" in q:
        return DIRECT_KB_FACTS["library"]

    if "medical" in q or "doctor" in q or "first aid" in q:
        return DIRECT_KB_FACTS["medical"]

    if "placement" in q or "job offer" in q or "package" in q:
        if "process" in q or "how does placement work" in q or "policy" in q:
            return DIRECT_KB_FACTS["placement_process"]
        return DIRECT_KB_FACTS["placements_latest"]

    if "contact" in q or "reach" in q or "helpline" in q:
        if "admission" in q:
            return DIRECT_KB_FACTS["admission_contact"]
        return DIRECT_KB_FACTS["general_contact"]

    if "exam" in q:
        return DIRECT_KB_FACTS["exam_batch_clarification"]

    return None


def rag(query: str, top_k: int = 5, client=None, embedder=None) -> tuple[str, str, str]:
    """
    CampusMitra End-to-End Voice RAG Pipeline:
      1. Ambiguity detection & clarification check
      2. Intent classification
      3. Retrieve relevant chunks from MVGR Knowledge Base
      4. Answer generation via Groq LLM (or direct verified MVGR knowledge)
      5. Safe fallback and office escalation for unsupported/unavailable queries

    Returns:
        tuple: (answer_text, context_used, intent)
    """
    intent = detect_intent(query)

    # 1. Ambiguity check
    needs_clarification, clarification_prompt = check_clarification_needed(query)
    if needs_clarification:
        return clarification_prompt, "", intent

    # 2. Retrieve chunks
    chunks = retrieve(client, embedder, query, top_k=top_k)
    context = build_context(chunks) if chunks else ""

    # Check for direct verified answer
    direct_ans = direct_mvgr_answer(query, intent)

    # If Groq API Key is available and langchain_groq is present, use LLM
    has_groq_key = bool(settings.groq_api_key or os.getenv("GROQ_API_KEY"))

    if has_groq_key and HAS_LANGCHAIN_GROQ and context:
        try:
            system_prompt = (
                "You are CampusMitra, an AI-powered voice assistant for MVGR College of Engineering.\n"
                "Your job is to help students, parents, applicants and visitors with college-related questions.\n"
                "Use the provided MVGR knowledge base as the primary source of truth.\n"
                "Answer clearly, accurately and conversationally.\n"
                "Never invent college information.\n"
                "When information is unavailable, say so politely and direct the user to the appropriate college office.\n"
                "For time-sensitive information (fees, exams, placements), always respect the year/source in the context.\n"
                "Never combine placement statistics from different years.\n"
                "Keep responses SHORT and natural for voice calls (1 to 2 sentences preferred).\n"
                "Ask clarification questions when the user's intent is unclear.\n"
                "You are a college information assistant, not a general-purpose unrestricted assistant."
            )

            user_message = f"MVGR Knowledge Base Context:\n{context}\n\nStudent Query: {query}"

            llm = ChatGroq(
                model=settings.groq_model,
                temperature=0.1,
            )
            messages = [
                ("system", system_prompt),
                ("user", user_message),
            ]

            response = llm.invoke(messages)
            answer = response.content.strip()
            return answer, context, intent
        except Exception as e:
            print(f"[RAG LLM Error] {e} — using verified direct answer.")

    # 3. Direct verified answer fallback
    if direct_ans:
        return direct_ans, context, intent

    # 4. If we retrieved relevant chunks from docs, summarize them concisely
    if chunks:
        top_chunk = chunks[0]
        summary = top_chunk.get("content", "").replace("\n", " ").strip()
        if len(summary) > 280:
            summary = summary[:277] + "..."
        return summary, context, intent

    # 5. Unsupported / Fallback handling
    office_info = OFFICE_DIRECTORY.get(intent)
    if office_info:
        office_name = office_info["office"]
        phone = office_info.get("phone", "08922-241732")
        fallback_msg = (
            f"I don't have that specific information in my current college knowledge base. "
            f"Please reach out to the {office_name} at {phone} for official assistance."
        )
    else:
        fallback_msg = (
            "I'm currently focused on MVGR College queries such as admissions, courses, fees, academics, "
            "hostels, transport, and placements. I don't have that information in my current knowledge base."
        )

    return fallback_msg, context, intent


def main():
    """CLI interface for testing CampusMitra voice queries."""
    # Ensure Windows console handles unicode cleanly
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What courses are available in MVGR?"

    print(f"CampusMitra Query: {query}\n")
    answer, context, intent = rag(query)

    print(f"INTENT: {intent}")
    print(f"ANSWER:\n{answer}\n")
    if context:
        clean_context = context.replace("\u25a0", "₹")
        try:
            print(f"{'='*60}\nSOURCES:\n{clean_context}")
        except Exception:
            print(clean_context.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
