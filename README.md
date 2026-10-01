# CampusMitra — AI-Powered Voice Assistant for College Queries 🎓🎙️

> *"Your AI Companion for Everything Campus."*

**CampusMitra** is a voice-first artificial intelligence assistant developed for **Maharaj Vijayaram Gajapathi Raj (MVGR) College of Engineering (Autonomous)** in Vizianagaram, Andhra Pradesh. Built with modern conversational AI and Retrieval-Augmented Generation (RAG), CampusMitra helps students, parents, applicants, and visitors instantly access accurate, verified college information over natural speech and web interfaces.

---

## 🏛️ About MVGR College of Engineering

- **Campus**: 60-acre green campus at Vijayaram Nagar, Chintalavasa, Vizianagaram, AP 535005
- **Status**: Autonomous institution, permanently affiliated to JNTUGV, approved by AICTE, accredited by NAAC & NBA
- **Infrastructure**: ~700,000 sq. ft. built-up area, 83 laboratories, 48,000 sq. ft. central library (400 seating capacity), 12 departmental libraries, boys' & girls' hostels, Rasamyai canteen, medical room, and dedicated college transport.

---

## 🌟 Key Capabilities & Features

1. **Voice-First Experience**: Real-time microphone listening, soundwave visualization, and clear spoken responses powered by Web Speech API & LiveKit.
2. **Strict Knowledge Grounding (Zero Hallucination)**: Responses are grounded exclusively in official MVGR knowledge base documents. If information (such as hostel fees or dated bus routes) is not in the source docs, the agent explicitly directs users to the appropriate college office rather than inventing numbers.
3. **College-Specific Intent Recognition**: Accurately classifies user queries into 18 college intents (`ADMISSIONS`, `COURSES`, `DEPARTMENTS`, `FEES`, `ACADEMICS`, `EXAMINATIONS`, `ACADEMIC_CALENDAR`, `HOSTEL`, `LIBRARY`, `TRANSPORT`, `CAMPUS_FACILITIES`, `MEDICAL`, `PLACEMENTS`, `INTERNSHIPS`, `SCHOLARSHIPS`, `CERTIFICATES`, `STUDENT_SERVICES`, `CONTACT`).
4. **Intelligent Clarification Handling**: Proactively asks brief follow-up questions when inquiries are ambiguous (e.g., distinguishing tuition vs. hostel vs. transport fees, or identifying batch and semester for exam queries).
5. **Time-Sensitive Awareness**: Strictly preserves reporting periods (e.g., distinguishing 2026 homepage highlights from 2025 annual reports; citing 2025–26 AFRC tuition figures while advising verification for revised notices).
6. **Office Directory & Escalation**: Connects queries directly with relevant campus offices (Admissions, Accounts, Exam Cell, Hostel Warden, Transport, and T&P Cell).

---

## 🏗️ Architecture & Voice Pipeline

```
  +--------------------------------------------------------------+
  |                   Student / Parent / Visitor                 |
  +--------------------------------------------------------------+
           |                                            ^
  (Voice / Spoken Query)                     (Spoken Voice Response)
           v                                            |
  +------------------------+                 +--------------------+
  | Web Speech / LiveKit   |                 | Text-to-Speech     |
  | Speech-to-Text (STT)   |                 | (Inworld / WebTTS) |
  +------------------------+                 +--------------------+
           |                                            ^
           v                                            |
  +--------------------------------------------------------------+
  |            CampusMitra Natural Language & RAG Engine         |
  |  1. Ambiguity & Clarification Filter                         |
  |  2. College Intent Classifier (18 College Intents)           |
  |  3. Qdrant Vector DB (Cosine Similarity / 384-dim Embeddings)|
  |  4. Strict Knowledge Grounding (MVGR PDFs 01, 02, 03)        |
  |  5. LangChain + Groq LLM (Voice-optimized concise responses) |
  +--------------------------------------------------------------+
                                |
               +----------------------------------+
               |  MVGR College Official Docs      |
               |  - Admissions, Fees & FAQ        |
               |  - Campus, Hostel, Facilities    |
               |  - Courses, Academics, Placements|
               +----------------------------------+
```

---

## 📂 Official Knowledge Base Sources

The agent is indexed with the 3 official MVGR College of Engineering documents:
1. `MVGR_Voice_Agent_01_Admissions_Fees_FAQ.pdf`: 2026 intake, 70% Convener (EAPCET) / 30% Management quota, AFRC fee structure (₹76,900 B.Tech), and Admissions Office helplines.
2. `MVGR_Voice_Agent_02_Campus_Hostel_Facilities.pdf`: 60-acre campus, 83 laboratories, 48,000 sq. ft. central library (RFID, KOHA, DELNET), boys' hostels & Maharani Vidyavathi Devi girls' hostel, medical room, Rasamyai canteen, and transport routes.
3. `MVGR_Voice_Agent_03_Courses_Academics_Placements.pdf`: Autonomous CBCS model, batch-specific exam calendar requirements, and placement statistics (2026 highlight: 764 offers from 122 companies; 2025: 810 offers across 77 companies, avg ₹5.2L).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python `>= 3.12`
- Node.js (optional, for browser interaction)
- Modern web browser (Chrome, Edge, Safari, Firefox)

### 2. Installation
```bash
# Clone or navigate to the directory
cd Campusmitra-voice-agent

# Install dependencies
pip install -e .
# Or install core dependencies directly:
pip install pydantic-settings python-dotenv sentence-transformers qdrant-client langchain langchain-groq pypdf
```

### 3. Ingest Documents into Vector DB
```bash
python ingest.py
```
*Parses all 3 MVGR PDFs, creates hierarchical semantic chunks, computes 384-dimensional embeddings, and indexes them into Qdrant (`campusmitra_mvgr_docs`).*

### 4. Run the Web & Voice Interface
```bash
python web_server.py
```
Open **`http://localhost:8000`** in your browser. Features live microphone recording, real-time voice playback, quick-action chips, and source citations.

### 5. Start the LiveKit Real-Time Audio Agent (Optional)
```bash
# Set your LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and GROQ_API_KEY in .env
python -m voice_agent.main console
```

### 6. Run Automated Test Suite
```bash
python test_campusmitra.py
```
*Validates all 14 mandatory queries, ambiguity handling, and zero-hallucination guardrails with 100% test coverage.*

---

## 💬 Sample Conversations (12 Scenarios)

### 1. Admission
**Student:** *"How can I apply for B.Tech admission at MVGR?"*  
**CampusMitra:** *"Regular B.Tech convener admissions (70% quota) are based on AP EAPCET, while lateral entry to the second year is through ECET. 30% of seats are under management quota. You can contact the Admission Office at 08922-241749 or 9704950954 between 10:00 AM and 4:00 PM."*

### 2. CSE Course & Intake
**Student:** *"What is the CSE intake and does MVGR offer AI?"*  
**CampusMitra:** *"The current intake for B.Tech CSE is 300 seats. MVGR also offers specialized programs including CSE in Artificial Intelligence & Machine Learning with 240 seats, Data Science with 60 seats, and IoT & Cyber Security with 60 seats."*

### 3. Fees
**Student:** *"What is the B.Tech fee?"*  
**CampusMitra:** *"The official fee schedule lists ₹76,900 for B.Tech tuition under the 2025–26 AFRC structure, as well as ₹76,900 for second-year lateral entry. For current 2026–27 B-category fees, please verify the latest official notification."*

### 4. Hostel
**Student:** *"Does MVGR have hostel facilities?"*  
**CampusMitra:** *"Yes, MVGR provides boys' hostels and the Maharani Vidyavathi Devi Hostel for girls, featuring 3-sharing and 4-sharing rooms, dining halls, gym, Rasamyai canteen, and a medical room with an on-duty medical officer and ANM."*

### 5. Transport
**Student:** *"Does MVGR provide college transport?"*  
**CampusMitra:** *"Yes, MVGR operates college buses connecting Visakhapatnam, Vizianagaram, Srikakulam, Bobbili, Saluru, L.Kota, S.Kota, and Cheepurupalli. For today's specific route numbers and departure timings, please verify with the Transport Office."*

### 6. Examinations
**Student:** *"When are the exams?"*  
**CampusMitra:** *"MVGR publishes batch-specific academic calendars and examination notifications. To give you the exact schedule, could you please specify your admitted batch year and current semester?"*

### 7. Library
**Student:** *"Does MVGR have a central library?"*  
**CampusMitra:** *"Yes, MVGR features a central library of 48,000 square feet with 400 seating capacity, alongside 12 departmental libraries, RFID-enabled services, KOHA software, and DELNET inter-library loan access."*

### 8. Placements
**Student:** *"What are the latest placement figures?"*  
**CampusMitra:** *"The 2026 homepage highlights 764 job offers across 122 participating companies (60 IT and 62 Core). In the 2025 reporting period, 810 job offers were secured for 975 eligible students across 77 companies, with an average package of ₹5.2 lakh."*

### 9. Scholarships
**Student:** *"Are scholarships available for students?"*  
**CampusMitra:** *"Eligible convener quota students can avail Andhra Pradesh state government fee reimbursement schemes such as Jagananna Vidya Deevena (JVD). For application assistance, please visit the Student Affairs Office."*

### 10. Certificates
**Student:** *"How can I get my study or transfer certificate?"*  
**CampusMitra:** *"Certificate requests including bonafide, study, and transfer certificates are processed through the Academic Section and Administration Office during working hours (10:00 AM to 4:00 PM)."*

### 11. Unknown / Unsupported Question (Zero Hallucination)
**Student:** *"What is the salary of the principal?"*  
**CampusMitra:** *"I do not have that information available in my current college knowledge base. I am focused on MVGR college-related queries such as admissions, courses, fees, academics, hostels, transport, and placements."*

### 12. Ambiguous Question (Clarification Flow)
**Student:** *"What is the fee?"*  
**CampusMitra:** *"Sure. Do you mean tuition fee, hostel fee, or transport fee?"*

---

## 📞 Campus Helplines Directory

| Office | Contact Numbers | Email / Location |
|---|---|---|
| **Admission Office** | `08922-241749`, `9704950954`, `9441663201` | `helpdesk@mvgrce.edu.in` (10 AM - 4 PM) |
| **General Enquiry** | `08922-241732`, `08922-241039` | Admin Block, Chintalavasa |
| **Accounts / Fee Section** | `08922-241732` | Accounts Counter (10 AM - 4 PM) |
| **Examination Cell** | `08922-241732` | Autonomous Exam Section |
| **Training & Placement Cell** | `08922-241732` | `helpdesk@mvgrce.edu.in` |
| **Hostel Wardens** | `08922-241732` | Boys Hostels & Maharani Vidyavathi Hostel |
| **Transport Office** | `08922-241732` | Transport Depot & Dispatch Office |

---

## 🔒 Verification & Compliance Notes

1. **Time-Sensitive Data**: Fees, admissions dates, exam notices, and intake figures must always reflect their dated source (e.g. 2025–26 AFRC vs. 2026–27 notices).
2. **Placement Period Separation**: 2026 highlight figures (764 offers from 122 companies) and 2025 statistics (810 offers from 77 companies) represent separate reporting periods and must never be merged.
3. **No Telecom Residuals**: All telecom terms (SIM, roaming, prepaid, recharge, billing, 4G/5G mobile plans) and Lauki branding have been completely removed from all prompts, tools, and UI.

---
© MVGR College of Engineering (Autonomous) — CampusMitra Voice Agent.
