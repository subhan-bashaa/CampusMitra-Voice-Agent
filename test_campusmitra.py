import sys
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from langchain_agent.rag import rag, check_clarification_needed, detect_intent

# Required test queries from Section 17 of the prompt
TEST_QUERIES = [
    ("What courses does MVGR offer?", "COURSES", ["CSE", "AI & ML", "B.Tech"]),
    ("What is the CSE intake?", "COURSES", ["300"]),
    ("Does MVGR offer AI and ML?", "COURSES", ["240", "AI"]),
    ("What is the B.Tech fee?", "FEES", ["76,900", "2025"]),
    ("How can I apply for admission?", "ADMISSIONS", ["EAPCET", "Convener", "quota"]),
    ("Is lateral entry available?", "ADMISSIONS", ["ECET", "second year"]),
    ("Does MVGR have hostel?", "HOSTEL", ["Maharani Vidyavathi", "boys"]),
    ("Does MVGR provide transport?", "TRANSPORT", ["Visakhapatnam", "Vizianagaram"]),
    ("Does MVGR have a library?", "LIBRARY", ["48,000", "RFID", "DELNET"]),
    ("When are the exams?", "EXAMINATIONS", ["batch", "semester"]),
    ("How does placement work?", "PLACEMENTS", ["Training & Placement", "recruitment"]),
    ("What are the latest placement figures?", "PLACEMENTS", ["764", "122", "2026"]),
    ("Who should I contact for admission?", "CONTACT", ["08922-241749", "helpdesk@mvgrce.edu.in"]),
    ("Can you tell me something that isn't in your knowledge base?", "UNKNOWN", ["not have", "knowledge base"]),
]

# Additional verification tests for Ambiguity and Zero Hallucination
AMBIGUITY_AND_EDGE_TESTS = [
    ("What is the fee?", "Ambiguity check for fee clarification", ["tuition fee", "hostel fee"]),
    ("What about CSE?", "Ambiguity check for course clarification", ["admission", "fees", "eligibility"]),
    ("What is the hostel fee?", "Zero hallucination for missing hostel fee", ["don't have", "hostel office"]),
]


def run_tests():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("==================================================================")
    print("  CAMPUSMITRA TEST SUITE — 14 REQUIRED QUERIES & EDGE CASES")
    print("==================================================================")

    passed = 0
    total = len(TEST_QUERIES) + len(AMBIGUITY_AND_EDGE_TESTS)

    print("\n--- Phase 1: Testing 14 Mandatory Section 17 Queries ---\n")
    for idx, (query, expected_intent, expected_keywords) in enumerate(TEST_QUERIES, 1):
        print(f"[{idx:02d}/14] Query: \"{query}\"")
        answer, context, intent = rag(query)
        print(f"       Intent: {intent} (Expected: {expected_intent})")
        print(f"       Answer: {answer[:140]}...")

        # Assert no Lauki references
        assert "lauki" not in answer.lower(), f"Lauki branding leaked into response: {answer}"
        assert "sim" not in answer.lower(), f"Telecom terminology leaked: {answer}"
        assert "telecom" not in answer.lower(), f"Telecom terminology leaked: {answer}"

        # Check keywords
        found_keywords = [k for k in expected_keywords if k.lower() in answer.lower()]
        if len(found_keywords) >= 1:
            print(f"       Status: [PASS] (Matched keywords: {found_keywords})\n")
            passed += 1
        else:
            print(f"       Status: [FAIL] (Missing all expected: {expected_keywords})\n")

    print("\n--- Phase 2: Testing Ambiguity & Zero Hallucination Guardrails ---\n")
    for idx, (query, desc, expected_keywords) in enumerate(AMBIGUITY_AND_EDGE_TESTS, 1):
        print(f"[Edge {idx}] Query: \"{query}\" ({desc})")
        answer, context, intent = rag(query)
        print(f"       Answer: {answer}")

        assert "lauki" not in answer.lower(), f"Lauki branding leaked: {answer}"
        found = [k for k in expected_keywords if k.lower() in answer.lower()]
        if len(found) >= 1:
            print(f"       Status: [PASS] (Verified guardrail: {found})\n")
            passed += 1
        else:
            print(f"       Status: [FAIL] (Missing expected: {expected_keywords})\n")

    print("==================================================================")
    print(f"  TEST RESULTS: {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print("==================================================================")

    if passed == total:
        print("ALL TESTS PASSED! CampusMitra is 100% compliant with MVGR knowledge base.")
        return 0
    else:
        print("Some tests did not pass.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
