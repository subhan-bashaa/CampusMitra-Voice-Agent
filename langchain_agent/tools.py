import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings
from langchain_agent.rag import retrieve, build_context


# ---------------------------------------------------------------------------
# Shared singletons — initialized once at import time
# ---------------------------------------------------------------------------

embedder = SentenceTransformer(settings.embedding_model)
qdrant = QdrantClient(path=settings.qdrant_path)

try:
    _count = qdrant.get_collection(settings.collection_name).points_count
    print(f"[OK] Connected to MVGR collection '{settings.collection_name}' — {_count} points")
except Exception:
    print(f"[NOTE] Qdrant collection '{settings.collection_name}' not yet opened. Ingestion will create it.")


# ---------------------------------------------------------------------------
# MVGR College of Engineering Structured Data (from the 3 official KB PDFs)
# ---------------------------------------------------------------------------

MVGR_OVERVIEW = {
    "name": "Maharaj Vijayaram Gajapathi Raj (MVGR) College of Engineering (Autonomous)",
    "address": "Vijayaram Nagar Campus, Chintalavasa, Vizianagaram, Andhra Pradesh 535005",
    "accreditation": "Autonomous institution, AICTE approved, NAAC & NBA accredited, permanent affiliation to JNTUGV",
    "campus_area": "60-acre campus, ~700,000 sq. ft. built-up area, 83 laboratories",
    "general_contacts": ["08922-241732", "08922-241039"],
}

MVGR_PROGRAMS_2026 = {
    "B.Tech CSE": {"intake": 300, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech CSE (AI & ML)": {"intake": 240, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech CSE (Data Science)": {"intake": 60, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech CSE (IoT & Cyber Security including Blockchain)": {"intake": 60, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech Computer Science & IT": {"intake": 120, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech ECE": {"intake": 180, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech Mechanical": {"intake": 120, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech EEE": {"intake": 60, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech Chemical": {"intake": 60, "duration": "4 years", "type": "Undergraduate"},
    "B.Tech Civil": {"intake": 60, "duration": "4 years", "type": "Undergraduate"},
    "M.Tech Structural Engineering": {"intake": 18, "duration": "2 years", "type": "Postgraduate"},
    "MBA": {"intake": 60, "duration": "2 years", "type": "Postgraduate"},
}

MVGR_FEES = {
    "B.Tech": {
        "period": "2025-26 AFRC",
        "tuition": "₹76,900 per year",
        "note": "Regular convener intake (70% quota). For 2026-27 revised B-category fee, please check latest official notification.",
    },
    "B.Tech Lateral Entry": {
        "period": "2025-26 AFRC",
        "tuition": "₹76,900 per year",
        "note": "Direct admission into 2nd year via ECET.",
    },
    "M.Tech": {
        "period": "2025-26 AFRC",
        "tuition": "₹65,000 per year",
        "note": "Structural Engineering program.",
    },
    "MBA": {
        "period": "2025-26 AFRC",
        "tuition": "₹60,000 per year",
        "note": "Master of Business Administration.",
    },
}

MVGR_OFFICES = {
    "admissions": {
        "name": "MVGR Admission Office",
        "landline": "08922-241749",
        "helplines": ["9704950954", "9441663201"],
        "email": "helpdesk@mvgrce.edu.in",
        "visiting_hours": "10:00 AM to 4:00 PM",
    },
    "accounts": {
        "name": "Accounts & Finance Section",
        "contact": "08922-241732 / 241039",
        "visiting_hours": "10:00 AM to 4:00 PM",
    },
    "examinations": {
        "name": "Examination Cell",
        "contact": "08922-241732",
        "note": "Publishes batch-specific academic calendars, mid/end exam timetables, and results.",
    },
    "hostel": {
        "name": "Hostel Office & Chief Warden",
        "contact": "08922-241732",
        "facilities": "Boys Hostels and Maharani Vidyavathi Devi Hostel for Girls (3-sharing and 4-sharing).",
    },
    "transport": {
        "name": "Transport Office",
        "contact": "08922-241732",
        "coverage": "Visakhapatnam, Vizianagaram, Srikakulam, Bobbili, Saluru, L.Kota, S.Kota, Cheepurupalli.",
    },
    "placements": {
        "name": "Training & Placement (T&P) Cell",
        "contact": "08922-241732",
        "email": "helpdesk@mvgrce.edu.in",
        "policy": "Organizes recruitment preparation, employability training, and on/off-campus placement drives.",
    },
}


# ---------------------------------------------------------------------------
# College LangChain Tools
# ---------------------------------------------------------------------------

@tool
def search_college_knowledge(query: str) -> str:
    """Search MVGR College of Engineering's official knowledge base (PDF documentation)
    for details about admissions, courses, fees, exams, hostels, library, transport,
    facilities, and placement policies. Always use this for accurate college queries."""
    chunks = retrieve(qdrant, embedder, query, top_k=5)
    if not chunks:
        return "No specific college information found in the knowledge base."
    return build_context(chunks)


@tool
def get_course_details(course_name: str) -> str:
    """Get official intake, branch details, and program specifications for MVGR courses.
    Provide the program name (e.g., 'CSE', 'AI & ML', 'Data Science', 'ECE', 'Mechanical', 'Civil', 'MBA')."""
    course_name = course_name.strip()

    # Search for matching course
    matched_program = None
    for prog, info in MVGR_PROGRAMS_2026.items():
        if course_name.lower() in prog.lower():
            matched_program = (prog, info)
            break

    if matched_program:
        prog, info = matched_program
        return (
            f"{prog}: 2026 Intake is {info['intake']} seats. "
            f"Duration: {info['duration']}, Degree Type: {info['type']}. "
            f"Offered under MVGR's autonomous Choice Based Credit System (CBCS) curriculum."
        )

    all_progs = ", ".join(MVGR_PROGRAMS_2026.keys())
    return f"Course '{course_name}' not found. Available MVGR programs: {all_progs}."


@tool
def get_fee_information(program: str = "B.Tech") -> str:
    """Retrieve official dated tuition fee details for MVGR academic programs.
    Programs: 'B.Tech', 'Lateral Entry', 'M.Tech', 'MBA'."""
    p_lower = program.lower()

    if "lateral" in p_lower:
        fee_info = MVGR_FEES["B.Tech Lateral Entry"]
        prog_name = "B.Tech Lateral Entry (2nd Year Direct)"
    elif "m.tech" in p_lower or "structural" in p_lower:
        fee_info = MVGR_FEES["M.Tech"]
        prog_name = "M.Tech Structural Engineering"
    elif "mba" in p_lower:
        fee_info = MVGR_FEES["MBA"]
        prog_name = "MBA"
    else:
        fee_info = MVGR_FEES["B.Tech"]
        prog_name = "B.Tech Regular"

    return (
        f"{prog_name} Tuition Fee: {fee_info['tuition']} ({fee_info['period']} AFRC structure). "
        f"{fee_info['note']}"
    )


@tool
def get_placement_highlights(year: str = "2026") -> str:
    """Retrieve official placement highlights and statistics for MVGR College of Engineering.
    Specify year: '2026' (current homepage highlight) or '2025' (previous annual report)."""
    year = year.strip()

    if "2025" in year:
        return (
            "2025 Placement Achievements: 810 job offers secured for 975 eligible students "
            "across 77 participating companies, with an average package of ₹5.2 lakh. "
            "(Official Join MVGR reporting period)."
        )

    return (
        "2026 Placement Highlight: 764 job offers from 122 participating companies "
        "(60 IT companies and 62 Core companies). Please note that these figures represent "
        "the 2026 reporting period and are not a job guarantee."
    )


@tool
def get_campus_facility_info(facility_type: str = "hostel") -> str:
    """Get verified details on MVGR campus facilities including hostels, central library,
    medical room, Rasamyai canteen, laboratories, and sports grounds."""
    f = facility_type.lower().strip()

    if "hostel" in f:
        return (
            "MVGR provides boys' hostels and the Maharani Vidyavathi Devi Hostel for girls. "
            "Features include 3-sharing and 4-sharing rooms, dining facilities, playgrounds, "
            "gym facilities, and Rasamyai canteen. Current hostel fees are not in the public KB and should be verified with the Hostel Office."
        )
    elif "library" in f:
        return (
            "MVGR features a 48,000 sq. ft. central library (with 12,000 sq ft recorded on About page), "
            "400 seating capacity, 12 departmental libraries, KOHA management software, RFID access, and DELNET inter-library loans."
        )
    elif "medical" in f or "doctor" in f:
        return (
            "MVGR has an on-campus medical facility staffed by a qualified medical officer and ANM. "
            "The medical room includes two beds, a stretcher, wheelchair, and emergency first-aid kits."
        )
    elif "canteen" in f or "food" in f:
        return "MVGR has the Rasamyai canteen and food services situated near the hostel and campus area."

    return (
        f"MVGR features a 60-acre campus with 83 laboratories, 48,000 sq ft central library, "
        f"boys and girls hostels, Rasamyai canteen, medical room, and sports playgrounds."
    )


@tool
def get_transport_routes(route_area: str = "all") -> str:
    """Get college transport coverage routes connecting Vizag, Vizianagaram, Srikakulam,
    Bobbili, Saluru, L.Kota, S.Kota, and Cheepurupalli. Note that exact timings require the latest schedule."""
    return (
        "MVGR operates college transport covering Visakhapatnam, Vizianagaram, Srikakulam, "
        "Bobbili, Saluru, L.Kota, S.Kota, and Cheepurupalli. Because routes and timings are date-specific, "
        "please refer to the latest official transport schedule or contact the Transport Office for today's timings."
    )


@tool
def get_college_office_contact(office_name: str) -> str:
    """Retrieve official phone numbers, email addresses, and visiting hours for MVGR college offices:
    'admissions', 'accounts', 'examinations', 'hostel', 'transport', 'placements'."""
    key = office_name.lower().strip()
    match = None
    for k, v in MVGR_OFFICES.items():
        if k in key or key in k:
            match = v
            break

    if not match:
        match = MVGR_OFFICES["admissions"]

    details = [f"Office: {match.get('name', office_name.capitalize())}"]
    if "landline" in match:
        details.append(f"Landline: {match['landline']}")
    if "contact" in match:
        details.append(f"Phone: {match['contact']}")
    if "helplines" in match:
        details.append(f"Helplines: {', '.join(match['helplines'])}")
    if "email" in match:
        details.append(f"Email: {match['email']}")
    if "visiting_hours" in match:
        details.append(f"Hours: {match['visiting_hours']}")

    return ". ".join(details)


@tool
def escalate_student_enquiry(
    student_name: str,
    contact_number: str,
    enquiry_topic: str,
    description: str,
) -> str:
    """Create a college enquiry ticket for queries requiring official college staff assistance
    (admissions, fee payment verification, certificates, hostel allocation, transport pass)."""
    ticket_id = "CM-" + str(abs(hash(description + student_name)) % 100000).zfill(5)
    timestamp = datetime.now().isoformat()

    print(f"[ENQUIRY ESCALATION] Ticket={ticket_id} time={timestamp} student={student_name} phone={contact_number} topic={enquiry_topic}")

    office_target = "Admission Office" if "admit" in enquiry_topic.lower() else "Student Affairs / Academic Office"

    return (
        f"I've registered enquiry ticket {ticket_id} for {student_name}. "
        f"This has been directed to the {office_target}. "
        f"You can also follow up directly at helpdesk@mvgrce.edu.in quoting {ticket_id}."
    )


# ---------------------------------------------------------------------------
# CampusMitra System Prompt & Agent Assembly
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are CampusMitra, an AI-powered voice assistant for MVGR College of Engineering.\n"
    "\n"
    "Your job is to help students, parents, applicants and visitors with college-related questions.\n"
    "Use the provided MVGR knowledge base as the primary source of truth.\n"
    "Answer clearly, accurately and conversationally.\n"
    "Never invent college information.\n"
    "When information is unavailable, say so and direct the user to the appropriate college office when possible.\n"
    "For time-sensitive information, use the latest available dated source.\n"
    "Keep spoken responses concise (1 to 2 sentences preferred for voice calls).\n"
    "Ask clarification questions when the user's intent is unclear.\n"
    "You are a college information assistant, not a general-purpose unrestricted assistant.\n"
    "\n"
    "TOOL USAGE RULES:\n"
    "- Use search_college_knowledge to look up details in MVGR PDFs.\n"
    "- Use get_course_details for branch intakes and course specifics.\n"
    "- Use get_fee_information for tuition fees and AFRC schedules.\n"
    "- Use get_placement_highlights for dated placement figures.\n"
    "- Use get_campus_facility_info for hostels, library, canteen, medical facilities.\n"
    "- Use get_transport_routes for college bus routes.\n"
    "- Use get_college_office_contact to provide phone numbers and office contacts.\n"
    "- Use escalate_student_enquiry when official staff action is requested."
)

ALL_TOOLS = [
    search_college_knowledge,
    get_course_details,
    get_fee_information,
    get_placement_highlights,
    get_campus_facility_info,
    get_transport_routes,
    get_college_office_contact,
    escalate_student_enquiry,
]

# Create LangChain agent graph
try:
    from langchain.agents import create_agent
    llm = ChatGroq(model=settings.groq_model, temperature=0.0)
    campusmitra_agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
except Exception:
    # Fallback placeholder if create_agent / Groq key unavailable
    campusmitra_agent = None

# Backwards compatibility alias
lauki_agent = campusmitra_agent
