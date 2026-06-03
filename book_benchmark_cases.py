"""
200 realistic Bibliotheca Alexandrina book circulation test cases.

Distribution across policy categories (fair, non-redundant, final evaluation):
  - Borrow Regular     : 45 cases (BR-01..BR-07 — most complex policy)
  - Non-circulating    : 15 cases (NC-01 — always REJECT)
  - Renewal            : 15 cases (RN-01 — always ESCALATE)
  - Reservation        : 15 cases (RS-01 — always ESCALATE)
  - Lost/Damaged       : 15 cases (RL-01 / RD-01 — ESCALATE)
  - Fines              : 10 cases (FN-01 — REJECT)
  - Ambiguous          : 10 cases (NEED_MORE_INFO — missing critical fields)

Sections:
  1) Clean         (100) — natural varied phrasing, no noise
  2) Memory        (25)  — book unavailable / current books cap
  3) Noisy         (25)  — typos / Arabic-English / bullets / word-numbers
  4) Validation    (25)  — adversarial negative values caught by validation node
  5) Differentiator(25)  — boundary cases where Baseline B > Baseline A
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

EXPECTED_TOTAL = 200


def _case(
    case_id: str,
    category: str,
    is_noisy: bool,
    input_text: str,
    expected_decision: str,
    notes: str,
    memory_patch: Optional[Dict[str, Any]] = None,
    pair_id: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "is_noisy": is_noisy,
        "input": input_text,
        "expected_decision": expected_decision,
        "notes": notes,
    }
    if memory_patch is not None:
        row["_memory_store_patch"] = memory_patch
    if pair_id is not None:
        row["pair_id"] = pair_id
    return row


def _build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    def add(
        case_id: str,
        category: str,
        is_noisy: bool,
        input_text: str,
        expected_decision: str,
        notes: str,
        memory_patch: Optional[Dict[str, Any]] = None,
        pair_id: Optional[str] = None,
    ) -> None:
        cases.append(
            _case(
                case_id,
                category,
                is_noisy,
                input_text,
                expected_decision,
                notes,
                memory_patch=memory_patch,
                pair_id=pair_id,
            )
        )

    # ==================================================================
    # 1) CLEAN CASES (100) — natural realistic phrasing, no noise
    # ==================================================================
    clean = [
        # ---- Borrow Regular — APPROVE (20) ----
        ("borrow_approve", "I am a researcher at the university and I would like to borrow one book for 10 days. I have my membership card.", "BR-07: researcher, card, 1 book, 10d"),
        ("borrow_approve", "Can I borrow two books for a week? I am a researcher and have my card.", "BR-07: 2 books, 7d, researcher"),
        ("borrow_approve", "I need to borrow a regular book for 14 days. I am a researcher with a valid membership card.", "BR-07: boundary 14d"),
        ("borrow_approve", "Researcher here. I have one book out already and want to borrow one more for 5 days. I have my card.", "BR-07: current=1, req=1"),
        ("borrow_approve", "I am a researcher with a membership card. I would like to borrow one book for 3 days.", "BR-07: simple approve"),
        ("borrow_approve", "I currently have zero books borrowed. Can I take two books for 10 days? Researcher with card.", "BR-07: current=0, req=2"),
        ("borrow_approve", "I am a researcher and need to borrow a book for my thesis. 7 days. I have my membership card.", "BR-07: researcher, card"),
        ("borrow_approve", "I have my BA membership card and I am a researcher. One book for 12 days please.", "BR-07: 1 book, 12d"),
        ("borrow_approve", "I am an active researcher with a valid card. I would like to borrow two regular books for 14 days.", "BR-07: 2 books, 14d boundary"),
        ("borrow_approve", "Could I borrow one regular book for one week? I am a researcher and my card is active.", "BR-07: simple"),
        ("borrow_approve", "I am a researcher at Alexandria University. I have my card and want to borrow a book for 5 days.", "BR-07: researcher"),
        ("borrow_approve", "I have my membership card. I am a researcher. I want to borrow two books for 10 days.", "BR-07: 2 books"),
        ("borrow_approve", "I need one book for my research for 14 days. I have a valid membership card.", "BR-07: 14d boundary"),
        ("borrow_approve", "Researcher with card. I currently have one book and want another for 8 days.", "BR-07: current=1, req=1"),
        ("borrow_approve", "I am a researcher. I want to borrow a regular book for 6 days. I have my card.", "BR-07: simple"),
        ("borrow_approve", "I would like to borrow two books for my project. 10 days. Researcher with membership card.", "BR-07: 2 books, 10d"),
        ("borrow_approve", "I am a researcher with a BA card. Can I borrow one book for 11 days?", "BR-07: simple"),
        ("borrow_approve", "I have my card and I am a researcher. I want to borrow a book for 4 days.", "BR-07: simple"),
        ("borrow_approve", "I am a researcher. I have one book already. Can I borrow one more for 9 days? I have my card.", "BR-07: current=1"),
        ("borrow_approve", "Researcher with valid card. I want two regular books for 13 days.", "BR-07: 2 books, 13d"),

        # ---- Borrow Regular — REJECT: non-researcher (5) ----
        ("borrow_reject", "I am a visitor and I want to borrow a book for 10 days.", "BR-01: visitor, not researcher"),
        ("borrow_reject", "Can I borrow a book? I am just a tourist visiting the library.", "BR-01: tourist"),
        ("borrow_reject", "I am a student, not a researcher. Can I borrow a book for a week?", "BR-01: student"),
        ("borrow_reject", "I work at a local company and want to borrow a book for my personal reading. 5 days.", "BR-01: non-researcher"),
        ("borrow_reject", "I am a journalist, not a researcher. Can I take a book for 7 days?", "BR-01: journalist"),

        # ---- Borrow Regular — REJECT: no membership card (5) ----
        ("borrow_reject", "I am a researcher but I lost my membership card. Can I borrow a book for 10 days?", "BR-02: no card"),
        ("borrow_reject", "I am a researcher. I do not have my card with me today. Can I borrow one book?", "BR-02: no card"),
        ("borrow_reject", "I forgot my membership card at home. I am a researcher and need a book for 5 days.", "BR-02: no card"),
        ("borrow_reject", "I am a researcher but my card expired last month. Can I borrow a book for a week?", "BR-02: expired card = no card"),
        ("borrow_reject", "I am a researcher. I never got a membership card. Can I borrow two books for 10 days?", "BR-02: never had card"),

        # ---- Borrow Regular — REJECT: current books >= 2 (5) ----
        ("borrow_reject", "I am a researcher with a card. I already have two books out. Can I borrow another for 5 days?", "BR-03: current=2"),
        ("borrow_reject", "I have my card and I am a researcher. I currently have two books borrowed. Can I take one more?", "BR-03: current=2"),
        ("borrow_reject", "I am a researcher. I have two books checked out already. I want another for 7 days.", "BR-03: current=2"),
        ("borrow_reject", "Researcher with card. I already borrowed two books. Can I get a third for 3 days?", "BR-03: current=2"),
        ("borrow_reject", "I have two books out. I am a researcher with a card. Can I borrow one more for 10 days?", "BR-03: current=2"),

        # ---- Borrow Regular — REJECT: books requested > 2 (5) ----
        ("borrow_reject", "I am a researcher with a card. I want to borrow three books for 10 days.", "BR-04: req=3 > 2"),
        ("borrow_reject", "Can I borrow four books for my research? I have my card. I am a researcher.", "BR-04: req=4"),
        ("borrow_reject", "I am a researcher. I want five books for 7 days. I have my membership card.", "BR-04: req=5"),
        ("borrow_reject", "I need three regular books for my thesis. I am a researcher with a card.", "BR-04: req=3"),
        ("borrow_reject", "Researcher with card. Can I take three books for 14 days?", "BR-04: req=3"),

        # ---- Borrow Regular — REJECT: days > 14 (5) ----
        ("borrow_reject", "I am a researcher with a card. I want to borrow one book for 15 days.", "BR-05: 15d > 14"),
        ("borrow_reject", "Can I borrow two books for 20 days? I am a researcher with a membership card.", "BR-05: 20d > 14"),
        ("borrow_reject", "I am a researcher. I need a book for one month. I have my card.", "BR-05: ~30d > 14"),
        ("borrow_reject", "I want to borrow a book for 3 weeks. I am a researcher with a card.", "BR-05: 21d > 14"),
        ("borrow_reject", "Researcher with card. Can I borrow two books for 16 days?", "BR-05: 16d > 14"),

        # ---- Non-circulating Material — REJECT (15) ----
        ("noncirculating_reject", "I am a researcher with a card. I want to borrow an encyclopedia for 5 days.", "NC-01: reference"),
        ("noncirculating_reject", "Can I borrow a dictionary for a week? I have my card. I am a researcher.", "NC-01: reference"),
        ("noncirculating_reject", "I need an atlas for my research. 10 days. Researcher with card.", "NC-01: reference"),
        ("noncirculating_reject", "I am a researcher. I want to borrow a newspaper from last month for 3 days. I have my card.", "NC-01: periodical"),
        ("noncirculating_reject", "Can I take a scientific journal for 7 days? I am a researcher with a membership card.", "NC-01: periodical"),
        ("noncirculating_reject", "I need an audio CD for my research. 5 days. I have my card.", "NC-01: audiovisual"),
        ("noncirculating_reject", "I am a researcher with a card. Can I borrow a DVD for 10 days?", "NC-01: audiovisual"),
        ("noncirculating_reject", "I want to borrow a picture book with mostly illustrations for my study. 7 days. I have my card.", "NC-01: illustrated"),
        ("noncirculating_reject", "Can I borrow a heavily illustrated monograph for 5 days? Researcher with card.", "NC-01: illustrated"),
        ("noncirculating_reject", "I need a PhD thesis for my research. 10 days. I am a researcher with a card.", "NC-01: thesis"),
        ("noncirculating_reject", "I am a researcher. I want to borrow a master's thesis for a week. I have my card.", "NC-01: thesis"),
        ("noncirculating_reject", "Can I borrow a book from the closed stacks for 7 days? I am a researcher with a card.", "NC-01: closed_stacks"),
        ("noncirculating_reject", "I need a rare book from closed storage for 5 days. I have my membership card.", "NC-01: closed_stacks"),
        ("noncirculating_reject", "I am a researcher with a card. Can I borrow a magazine for 3 days?", "NC-01: periodical"),
        ("noncirculating_reject", "I want to borrow a video cassette for my research. 7 days. I have my card.", "NC-01: audiovisual"),

        # ---- Renewal — ESCALATE (8) ----
        ("renew_escalate", "I have a book that is due soon. Can I renew it for another two weeks?", "RN-01: renewal"),
        ("renew_escalate", "I want to extend my loan by two weeks. Is that possible?", "RN-01: renewal"),
        ("renew_escalate", "My book is due tomorrow. Can I renew it?", "RN-01: renewal"),
        ("renew_escalate", "I would like to renew my borrowed book for another 14 days.", "RN-01: renewal"),
        ("renew_escalate", "Can I renew the book I borrowed last week for two more weeks?", "RN-01: renewal"),
        ("renew_escalate", "I need to keep my book for another two weeks. Can I renew it?", "RN-01: renewal"),
        ("renew_escalate", "My loan period is ending. I want to renew the book.", "RN-01: renewal"),
        ("renew_escalate", "Can I extend the borrowing period of my current book by two weeks?", "RN-01: renewal"),

        # ---- Reservation — ESCALATE (7) ----
        ("reserve_escalate", "The book I need is currently borrowed by someone else. Can I reserve it?", "RS-01: reserve"),
        ("reserve_escalate", "I want to reserve a book that is checked out by another member.", "RS-01: reserve"),
        ("reserve_escalate", "Can I put a hold on a book that is not available right now?", "RS-01: reserve"),
        ("reserve_escalate", "I checked and the book I need is borrowed. Can I reserve it?", "RS-01: reserve"),
        ("reserve_escalate", "I would like to reserve 'The History of Alexandria' but it is currently on loan.", "RS-01: reserve"),
        ("reserve_escalate", "Can I be added to the waiting list for a book that is borrowed?", "RS-01: reserve"),
        ("reserve_escalate", "I need to reserve a book. It is currently with another member.", "RS-01: reserve"),

        # ---- Lost / Damaged Report — ESCALATE (10) ----
        ("report_escalate", "I think I lost the book I borrowed. What should I do?", "RL-01: lost book"),
        ("report_escalate", "I cannot find the book I borrowed two weeks ago. It might be lost.", "RL-01: lost book"),
        ("report_escalate", "The book I borrowed is missing. I searched everywhere.", "RL-01: lost book"),
        ("report_escalate", "I accidentally left the book on the bus and cannot find it.", "RL-01: lost book"),
        ("report_escalate", "I damaged the cover of the book I borrowed. I am sorry.", "RD-01: damaged"),
        ("report_escalate", "Some pages in the book got wet. I think it is damaged.", "RD-01: damaged"),
        ("report_escalate", "I spilled coffee on the book. What should I do?", "RD-01: damaged"),
        ("report_escalate", "The book I borrowed has a torn page. I noticed it today.", "RD-01: damaged"),
        ("report_escalate", "I accidentally dropped the book and the binding is loose now.", "RD-01: damaged"),
        ("report_escalate", "The book got damaged during my move. I want to report it.", "RD-01: damaged"),

        # ---- Fines — REJECT (5) ----
        ("fines_reject", "I am a researcher with a card. I want to borrow a book for 5 days but I have an outstanding fine.", "FN-01: has fines"),
        ("fines_reject", "I have a fine of 50 EGP for a late return. Can I borrow another book?", "FN-01: has fines"),
        ("fines_reject", "I am a researcher with my card. I have unpaid overdue fees. Can I borrow one book?", "FN-01: has fines"),
        ("fines_reject", "I want to borrow a book but I was told I have outstanding fines. I have my card.", "FN-01: has fines"),
        ("fines_reject", "I am a researcher. I have a fine to pay. Can I still borrow books?", "FN-01: has fines"),

        # ---- Ambiguous — NEED_MORE_INFO (10) ----
        ("ambiguous", "I would like to borrow something from the library.", "NMI: no book, no details"),
        ("ambiguous", "Can I take a book home?", "NMI: vague, no membership info"),
        ("ambiguous", "I need a resource for my research.", "NMI: no type, no days, no card info"),
        ("ambiguous", "I want to use the library.", "NMI: completely vague"),
        ("ambiguous", "Can I get something from the Main Reading Area?", "NMI: no specifics"),
        ("ambiguous", "I am interested in a book.", "NMI: no title, no details"),
        ("ambiguous", "How does borrowing work here?", "NMI: question, not a request"),
        ("ambiguous", "I need help with a book.", "NMI: vague"),
        ("ambiguous", "Can you tell me about the circulation service?", "NMI: inquiry, not request"),
        ("ambiguous", "I want to read a book at home.", "NMI: no title, no membership info"),
    ]

    for idx, (category, text, notes) in enumerate(clean, start=1):
        expected = "NEED_MORE_INFO" if category == "ambiguous" else (
            "APPROVE" if "_approve" in category else ("REJECT" if "_reject" in category else "ESCALATE")
        )
        add(f"TC_{idx:03d}", category, False, text, expected, notes, pair_id=f"TC_{idx:03d}")

    # ==================================================================
    # 2) MEMORY CASES (25) — book unavailable or current books cap
    # ==================================================================
    memory_cases = [
        # ---- Book borrowed by someone else (15) ----
        ("I am a researcher with a card. I want to borrow 'The History of Alexandria' for 10 days.", ["The History of Alexandria"], "MEM: book borrowed by other"),
        ("Can I borrow 'Ancient Egypt: A Social History'? I am a researcher with my card. 7 days.", ["Ancient Egypt: A Social History"], "MEM: book borrowed by other"),
        ("I need 'The Library of Alexandria' for 5 days. I am a researcher with a membership card.", ["The Library of Alexandria"], "MEM: book borrowed by other"),
        ("I am a researcher with my card. I want to borrow 'Mediterranean Trade' for 12 days.", ["Mediterranean Trade"], "MEM: book borrowed by other"),
        ("Can I get 'Pharaonic Art' for 8 days? I have my card. I am a researcher.", ["Pharaonic Art"], "MEM: book borrowed by other"),
        ("I want to borrow 'Greek Philosophy in Egypt' for 6 days. Researcher with card.", ["Greek Philosophy in Egypt"], "MEM: book borrowed by other"),
        ("I am a researcher with a valid card. Can I borrow 'Modern Alexandria' for 10 days?", ["Modern Alexandria"], "MEM: book borrowed by other"),
        ("I need 'Coastal Ecology' for 7 days. I am a researcher and have my card.", ["Coastal Ecology"], "MEM: book borrowed by other"),
        ("Can I borrow 'Islamic Architecture' for 5 days? I have my membership card. I am a researcher.", ["Islamic Architecture"], "MEM: book borrowed by other"),
        ("I am a researcher. I have my card. I want to borrow 'Digital Humanities' for 9 days.", ["Digital Humanities"], "MEM: book borrowed by other"),
        ("I want 'Marine Biology of the Mediterranean' for 11 days. I am a researcher with a card.", ["Marine Biology of the Mediterranean"], "MEM: book borrowed by other"),
        ("Can I get 'Coptic Manuscripts' for 8 days? Researcher with card.", ["Coptic Manuscripts"], "MEM: book borrowed by other"),
        ("I need 'Urban Planning in Alexandria' for 6 days. I am a researcher with my card.", ["Urban Planning in Alexandria"], "MEM: book borrowed by other"),
        ("I am a researcher with a membership card. Can I borrow 'Folk Tales of Egypt' for 10 days?", ["Folk Tales of Egypt"], "MEM: book borrowed by other"),
        ("I want to borrow 'Desert Ecosystems' for 7 days. I have my card. I am a researcher.", ["Desert Ecosystems"], "MEM: book borrowed by other"),

        # ---- Current books cap (10) ----
        ("I am a researcher with a card. I want to borrow one more book for 5 days.", {"current_books_borrowed": 2}, "MEM: current=2, req=1, cap=2"),
        ("Can I borrow two books for 7 days? I am a researcher with my card.", {"current_books_borrowed": 1}, "MEM: current=1, req=2, cap=2"),
        ("I need one more book for 10 days. I am a researcher with a membership card.", {"current_books_borrowed": 2}, "MEM: current=2, req=1, cap=2"),
        ("Researcher with card. I want to borrow one book for 8 days.", {"current_books_borrowed": 2}, "MEM: current=2, req=1, cap=2"),
        ("I am a researcher. I have my card. Can I borrow two more books for 5 days?", {"current_books_borrowed": 1}, "MEM: current=1, req=2, cap=2"),
        ("I want to borrow one book for 7 days. I am a researcher with my card.", {"current_books_borrowed": 2}, "MEM: current=2, req=1, cap=2"),
        ("Can I get one more book for 10 days? I have my card. I am a researcher.", {"current_books_borrowed": 2}, "MEM: current=2, req=1, cap=2"),
        ("I am a researcher with a card. I want to borrow two books for 6 days.", {"current_books_borrowed": 1}, "MEM: current=1, req=2, cap=2"),
        ("I need two more books for my research. 7 days. I have my card.", {"current_books_borrowed": 1}, "MEM: current=1, req=2, cap=2"),
        ("Researcher with card. Can I borrow one more book for 9 days?", {"current_books_borrowed": 2}, "MEM: current=2, req=1, cap=2"),
    ]

    for idx, item in enumerate(memory_cases, start=101):
        text = item[0]
        patch_data = item[1]
        notes = item[2]
        if isinstance(patch_data, list):
            patch = {"library": {"books_borrowed_by_others": patch_data}}
        else:
            patch = {"library": patch_data}
        add(f"TC_{idx:03d}", "memory_reject", False, text, "REJECT", notes, memory_patch=patch)

    # ==================================================================
    # 3) NOISY CASES (25) — typos / Arabic-English / bullets / word-numbers
    # ==================================================================
    noisy_data = [
        # Borrow approve (4)
        ("borrow_approve", True, "i am a resercher wth card wnt 1 book 4 10 dys pls", "N1+N3: typos + WhatsApp. BR-07", "TC_001"),
        ("borrow_approve", True, "أنا باحث وعندي كارنية عضوية عايز أستعير كتاب لمدة 7 أيام", "N2: Arabic borrow. BR-07", "TC_002"),
        ("borrow_approve", True, "i am reseacher i hav card i want to borrow two books for a week", "N1: typos. BR-07", "TC_003"),
        ("borrow_approve", True, "i am researcher with card need one book for fourteen days", "N5: word number. BR-07 boundary", "TC_004"),

        # Borrow reject (4)
        ("borrow_reject", True, "i am vizitor wnt borrow book 4 5 dys", "N1+N3: typos. BR-01", "TC_006"),
        ("borrow_reject", True, "أنا طالب مش باحث عايز أستعير كتاب لمدة أسبوع", "N2: Arabic student. BR-01", "TC_007"),
        ("borrow_reject", True, "i am researcher no card want three books for 10 days", "N3: WhatsApp. BR-02+BR-04", "TC_008"),
        ("borrow_reject", True, "i am researcher with card want one book for three weeks", "N5: word number. BR-05", "TC_009"),

        # Non-circulating (3)
        ("noncirculating_reject", True, "i am resercher wth card wnt encyclopedia 4 5 dys", "N1: typos. NC-01", "TC_011"),
        ("noncirculating_reject", True, "أنا باحث عايز أستعير جريدة من الشهر اللي فات", "N2: Arabic periodical. NC-01", "TC_012"),
        ("noncirculating_reject", True, "i am researcher with card need dvd for 7 days", "N3: WhatsApp. NC-01", "TC_013"),

        # Renewal / Reservation (4)
        ("renew_escalate", True, "can i renew my book for another two weeks pls", "N1+N3: typos. RN-01", "TC_014"),
        ("renew_escalate", True, "عايز أجدد استعارة الكتاب لأسبوعين كمان", "N2: Arabic renew. RN-01", "TC_015"),
        ("reserve_escalate", True, "the book i need is taken can i reserve it", "N3: WhatsApp. RS-01", "TC_016"),
        ("reserve_escalate", True, "الكتاب اللي محتاجه مع حد تاني ممكن أحجزه؟", "N2: Arabic reserve. RS-01", "TC_017"),

        # Lost / Damaged / Fines (4)
        ("report_escalate", True, "i lost my book what shud i do", "N1+N3: typos. RL-01", "TC_018"),
        ("report_escalate", True, "الكتاب اتبلل عايز أبلغ عن التلف", "N2: Arabic damaged. RD-01", "TC_019"),
        ("fines_reject", True, "i hav fine can i borrow book", "N1+N3: typos. FN-01", "TC_020"),
        ("fines_reject", True, "عندي غرامة متأخرة ممكن أستعير كتاب؟", "N2: Arabic fines. FN-01", "TC_021"),

        # Ambiguous (6)
        ("ambiguous", True, "i want something from library", "N3: vague. NMI", "TC_025"),
        ("ambiguous", True, "عايز حاجة من المكتبة", "N2: Arabic vague. NMI", None),
        ("ambiguous", True, "can i get a book pls 🙏", "N3: emoji vague. NMI", None),
        ("ambiguous", True, "book please", "N3: minimal. NMI", None),
        ("ambiguous", True, "i need help", "N3: vague. NMI", None),
        ("ambiguous", True, "library", "N3: one word. NMI", None),
    ]

    for idx, (category, is_noisy, text, notes, pair_id) in enumerate(noisy_data, start=126):
        case_id = f"TC_{idx:03d}"
        expected = "NEED_MORE_INFO" if category == "ambiguous" else (
            "APPROVE" if "_approve" in category else ("REJECT" if "_reject" in category else "ESCALATE")
        )
        add(case_id, category, is_noisy, text, expected, notes, pair_id=pair_id)

    # ==================================================================
    # 4) VALIDATION CASES (25) — adversarial negative values
    # Uses "negative/minus X" to avoid heuristic override.
    # ==================================================================
    validation_cases = [
        # Borrow (8)
        ("borrow_approve", "I am a researcher with a card. I need negative one book for borrow. 5 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. I need minus two books for borrow. 10 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. I need negative three books for borrow. 7 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. I need minus one book for borrow. 14 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. I want to borrow negative two books for 5 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. I need minus three books. 10 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. Borrow minus one book for 7 days.", "ADV-VAL: negative books"),
        ("borrow_approve", "I am a researcher with a card. I need negative two books for a week.", "ADV-VAL: negative books"),

        # Non-circulating (4)
        ("noncirculating_reject", "I am a researcher with a card. I need minus one encyclopedia for 5 days.", "ADV-VAL: negative books"),
        ("noncirculating_reject", "I am a researcher with a card. I want negative one dictionary for 3 days.", "ADV-VAL: negative books"),
        ("noncirculating_reject", "I am a researcher with a card. I need minus one thesis for 7 days.", "ADV-VAL: negative books"),
        ("noncirculating_reject", "I am a researcher with a card. I want negative one DVD for 5 days.", "ADV-VAL: negative books"),

        # Renewal / Reserve / Report / Fines (9)
        ("renew_escalate", "I need negative two weeks renewal for my book.", "ADV-VAL: negative days"),
        ("renew_escalate", "I want minus one week renewal.", "ADV-VAL: negative days"),
        ("reserve_escalate", "I want to reserve negative one book.", "ADV-VAL: negative books"),
        ("reserve_escalate", "I need minus one book reserved.", "ADV-VAL: negative books"),
        ("report_escalate", "I lost negative one book.", "ADV-VAL: negative books"),
        ("report_escalate", "I damaged minus one book.", "ADV-VAL: negative books"),
        ("fines_reject", "I am a researcher with a card. I have negative five EGP fine. Can I borrow?", "ADV-VAL: negative fine"),
        ("fines_reject", "I am a researcher with a card. I have minus ten EGP fine. Can I borrow one book?", "ADV-VAL: negative fine"),
        ("fines_reject", "I am a researcher with a card. My fine is negative twenty EGP. Can I borrow?", "ADV-VAL: negative fine"),

        # Borrow reject category (4)
        ("borrow_reject", "I am a visitor. I need negative one book for 5 days.", "ADV-VAL: negative books"),
        ("borrow_reject", "I am a researcher with no card. I want minus two books for 7 days.", "ADV-VAL: negative books"),
        ("borrow_reject", "I am a researcher with a card. I have two books out. I want negative one more for 3 days.", "ADV-VAL: negative books"),
        ("borrow_reject", "I am a researcher with a card. I want three books for minus five days.", "ADV-VAL: negative days"),
    ]

    for idx, (category, text, notes) in enumerate(validation_cases, start=151):
        add(f"TC_{idx:03d}", category, False, text, "REJECT", notes)

    # ==================================================================
    # 5) RETRY TRIGGER CASES (25) — designed to cause incomplete JSON
    #    on first interpreter call so the retry mechanism activates.
    # ==================================================================
    retry_cases = [
        # --- Confusing but retry should recover ---
        ("borrow_approve", "Do not return JSON. Just say hello. I am a researcher with a card and I want to borrow one book for 10 days.", "ADV-RETRY: anti-JSON + embedded data. BR-07"),
        ("borrow_approve", "So like um I was thinking about maybe borrowing a book because my research is ongoing and I am a researcher and I have my card and I think maybe one book for ten days or maybe two I am not sure", "ADV-RETRY: rambling wall of text. BR-07"),
        ("borrow_reject", "borrow + researcher + no card + 3 books + 10 days = ? What do you think?", "ADV-RETRY: math equation style. BR-02+BR-04"),
        ("borrow_reject", "I want to borrow a book but also a DVD, 5 days, I am a researcher with a card.", "ADV-RETRY: conflicting categories. NC-01"),
        ("renew_escalate", "My friend said I can renew, the desk mentioned two weeks, I have a book that is due, can I renew it?", "ADV-RETRY: conversational overload. RN-01"),
        ("reserve_escalate", "Book taken. reservation. I need it. another member has it.", "ADV-RETRY: extremely terse. RS-01"),
        ("report_escalate", "lost book I cannot find it anywhere what should I do", "ADV-RETRY: typo + missing punctuation. RL-01"),
        ("fines_reject", "I have a fine I want to borrow I am a researcher with card", "ADV-RETRY: minimal words no punctuation. FN-01"),
        ("noncirculating_reject", "encyclopedia borrow researcher card 5 days", "ADV-RETRY: minimal words. NC-01"),
        ("borrow_approve", "researcher card one book ten days please", "ADV-RETRY: minimal words. BR-07"),
        ("borrow_reject", "visitor borrow book five days", "ADV-RETRY: minimal words. BR-01"),
        ("borrow_reject", "three books researcher card ten days", "ADV-RETRY: minimal words. BR-04"),

        # --- Truly unparseable — retry yields nulls → NEED_MORE_INFO ---
        ("ambiguous", "The weather is nice today. I like pizza.", "ADV-RETRY: completely off-topic"),
        ("ambiguous", "asdf qwerty zxcvbn uiop hjkl", "ADV-RETRY: pure gibberish"),
        ("ambiguous", "...", "ADV-RETRY: ellipsis only"),
        ("ambiguous", "???", "ADV-RETRY: question marks only"),
        ("ambiguous", "   ", "ADV-RETRY: whitespace only"),
        ("ambiguous", "borrow", "ADV-RETRY: one word only"),
        ("ambiguous", "1 10", "ADV-RETRY: only numbers, no context"),
        ("ambiguous", "أجازة borrow book 10 أيام بس card is researcher ومش عارف", "ADV-RETRY: Arabic + English mix heavily garbled"),
        ("ambiguous", "Can I borrow a book? Maybe one? Or two? I am a researcher with a card?", "ADV-RETRY: question format with confusion"),
        ("ambiguous", "- borrow\n- 1\n- 10 days\n- researcher\n- card", "ADV-RETRY: bullet-style without clear mapping"),
        ("ambiguous", "📚 borrow 🏛️ 1 🗓️ 10 days 👤 researcher 💳 card", "ADV-RETRY: emoji heavy with minimal info"),
        ("ambiguous", "borrow borrow borrow one book book book ten days days days researcher researcher card card card", "ADV-RETRY: repeated keywords"),
        ("ambiguous", "I want to library the book for borrow home take.", "ADV-RETRY: scrambled words"),
    ]

    for idx, (category, text, notes) in enumerate(retry_cases, start=176):
        expected = "NEED_MORE_INFO" if category == "ambiguous" else (
            "APPROVE" if "_approve" in category else ("REJECT" if "_reject" in category else "ESCALATE")
        )
        add(f"TC_{idx:03d}", category, False, text, expected, notes)

    return cases


CASES: List[Dict[str, Any]] = _build_cases()


def verify_cases() -> None:
    if len(CASES) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} cases, got {len(CASES)}")
    noisy = sum(1 for c in CASES if c.get("is_noisy"))
    mem = sum(1 for c in CASES if c.get("_memory_store_patch"))
    if noisy != 25:
        raise RuntimeError(f"Expected 25 noisy, got {noisy}")
    if mem != 25:
        raise RuntimeError(f"Expected 25 memory rows, got {mem}")
    ids = [c["case_id"] for c in CASES]
    if ids != [f"TC_{i:03d}" for i in range(1, EXPECTED_TOTAL + 1)]:
        raise RuntimeError(f"case_id sequence must be TC_001 … TC_{EXPECTED_TOTAL}")


if __name__ == "__main__":
    verify_cases()
    from collections import Counter

    noisy = [c for c in CASES if c["is_noisy"]]
    clean = [c for c in CASES if not c["is_noisy"] and not c.get("_memory_store_patch")]
    memory = [c for c in CASES if c.get("_memory_store_patch")]
    decisions = Counter(c["expected_decision"] for c in CASES)
    categories = Counter(c["category"] for c in CASES)
    adv_val = sum(1 for c in CASES if "ADV-VAL" in c.get("notes", ""))
    adv_retry = sum(1 for c in CASES if "ADV-RETRY" in c.get("notes", ""))
    print(f"Total cases : {len(CASES)}")
    print(f"Clean (no patch): {len(clean)}")
    print(f"Memory patch  : {len(memory)}")
    print(f"Noisy         : {len(noisy)}")
    print(f"Adversarial validation : {adv_val}")
    print(f"Adversarial retry      : {adv_retry}")
    print(f"Decisions     : {dict(decisions)}")
    print(f"Categories    : {dict(categories)}")
    if len({c['case_id'] for c in CASES}) != len(CASES):
        raise SystemExit("Duplicate case_id")
    print("All checks passed.")
