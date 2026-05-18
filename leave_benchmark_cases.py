"""
200 realistic employee leave request test cases (single source of truth).

Distribution across 7 policy categories (fair, non-redundant, final evaluation):
  - Annual Leave    : 67 cases (AL-01..AL-04B — most complex policy)
  - Sick Leave      : 15 cases (SL-01 — always ESCALATE)
  - Maternity Leave : 28 cases (ML-01 APPROVE ≤120, ML-02 REJECT >120)
  - Hajj Leave      : 29 cases (HJ-01 <60mo REJECT, HJ-02 ≥60mo ≤30d APPROVE,
                                 HJ-03 ≥60mo >30d REJECT)
  - Casual Leave    : 25 cases (SP-01 ≤3 APPROVE, SP-02 >3 REJECT)
  - Contact Leave   : 11 cases (CT-01 — always ESCALATE)
  - Unpaid Leave    : 9  cases (UL-01 — always ESCALATE)
  - Ambiguous       : 16 cases (NEED_MORE_INFO — missing critical fields)

Sections:
  1) Clean         (100) — natural varied phrasing, no noise
  2) Memory        (25)  — annual only (memory agent tracks annual caps)
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
        # ---- Annual Leave (20) ----
        ("annual_reject", "I just started 3 months ago and need 10 days off for a family trip.", "AL-01: 3mo < 6mo"),
        ("annual_reject", "Can I take a week off? Only been here 4 months.", "AL-01: 4mo < 6mo"),
        ("annual_reject", "Need 5 days vacation. Started last month.", "AL-01: 1mo < 6mo"),
        ("annual_approve", "Hey boss, I need a week off next month. I've been here about 8 months now.", "AL-02: ~7d at 8mo"),
        ("annual_approve", "I'd like 10 days vacation. Completed 6 months last week.", "AL-02: 10d at 6mo boundary"),
        ("annual_approve", "My family is visiting from abroad. Can I get 5 days off? I've been with you guys for 7 months.", "AL-02: 5d at 7mo"),
        ("annual_approve", "Planning a trip during Eid. Need around 14 days. I've worked here for 11 months.", "AL-02: 14d at 11mo boundary"),
        ("annual_approve", "Quick getaway — 4 days. 6 months in.", "AL-02: 4d at 6mo"),
        ("annual_approve", "Can I get 12 days? 9 months with the company.", "AL-02: 12d at 9mo"),
        ("annual_reject", "I need 3 weeks off. Only been here 9 months though.", "AL-02B: ~21d > 15 at 9mo"),
        ("annual_reject", "Requesting 16 days. 8 months tenure.", "AL-02B: 16d > 15 at 8mo"),
        ("annual_reject", "Want a month off. 10 months employed.", "AL-02B: ~30d > 15 at 10mo"),
        ("annual_approve", "I'd like 14 days vacation. One year and one month on the job.", "AL-03: 14d at 13mo"),
        ("annual_approve", "Taking 20 days off. Been employed 3 years.", "AL-03: 20d at 36mo"),
        ("annual_approve", "Can I get 18 days? I've been here 15 months.", "AL-03: 18d at 15mo"),
        ("annual_approve", "Planning a trip — need 21 days. Exactly 2 years with the company.", "AL-03: 21d at 24mo boundary"),
        ("annual_approve", "Long vacation planned. 25 days. 5 years of service.", "AL-04: 25d at 60mo"),
        ("annual_approve", "Taking the full month off. 30 days. Tenure: 8 years.", "AL-04: 30d at 96mo boundary"),
        ("annual_reject", "Need 22 days. Just passed one year.", "AL-03B: 22d > 21 at ~13mo"),
        ("annual_reject", "Want 35 days. 6 years employed.", "AL-04B: 35d > 30 at 72mo"),

        # ---- Sick Leave (10) — always ESCALATE ----
        ("sick_escalate", "Not feeling great, might need a couple of days off. Doctor said I should rest.", "SL-01: sick vague"),
        ("sick_escalate", "I had surgery last week and my doctor says I need 2 weeks to recover properly.", "SL-01: sick ~14d"),
        ("sick_escalate", "Fever and cough since yesterday. Probably need tomorrow off at least.", "SL-01: sick 1d"),
        ("sick_escalate", "Feeling unwell since morning. Might need tomorrow off.", "SL-01: sick 1d"),
        ("sick_escalate", "Doctor prescribed bed rest for a week. Surgery follow-up.", "SL-01: sick ~7d"),
        ("sick_escalate", "Got food poisoning. Need a few days to recover.", "SL-01: sick vague days"),
        ("sick_escalate", "Chronic back pain acting up. Need time off for physio.", "SL-01: sick vague"),
        ("sick_escalate", "Migraine is unbearable. Taking today and maybe tomorrow off.", "SL-01: sick ~2d"),
        ("sick_escalate", "My dentist says I need a root canal and should rest for 3 days.", "SL-01: sick 3d"),
        ("sick_escalate", "Sprained ankle at the gym. Doctor note says 5 days rest.", "SL-01: sick 5d"),

        # ---- Maternity Leave (14) ----
        ("maternity_approve", "My wife and I are expecting a baby in March. I'll need to be home for a few months. I've been with the company for about a year and a half.", "ML-01: ~90d at ~18mo"),
        ("maternity_approve", "I'm pregnant and will need maternity leave starting next month. Planning for about 4 months. Tenure is 2 years.", "ML-01: ~120d at 24mo boundary"),
        ("maternity_approve", "Expecting in June. Need around 3 months off. 20 months here.", "ML-01: ~90d at 20mo"),
        ("maternity_approve", "Baby due in August. Planning 100 days leave. 22 months tenure.", "ML-01: 100d at 22mo"),
        ("maternity_approve", "Maternity leave starting next week. 110 days. 2 years.", "ML-01: 110d at 24mo"),
        ("maternity_approve", "Requesting 80 days maternity. First child. 16 months employed.", "ML-01: 80d at 16mo"),
        ("maternity_approve", "Need 95 days for maternity leave. Due in 2 months. 30 months here.", "ML-01: 95d at 30mo"),
        ("maternity_reject", "I need 5 months off for maternity. Just started 3 months ago.", "ML-02: ~150d > 120"),
        ("maternity_reject", "Requesting 130 days. Delivery scheduled for next month.", "ML-02: 130d > 120"),
        ("maternity_reject", "Need 4 months for maternity. Only 8 months employed.", "ML-02: ~122d > 120"),
        ("maternity_reject", "Applying for 125 days maternity. Twins expected next week.", "ML-02: 125d > 120"),
        ("maternity_reject", "Requesting 140 days maternity leave. Third child.", "ML-02: 140d > 120"),
        ("maternity_reject", "Need 150 days off for maternity. C-section scheduled.", "ML-02: 150d > 120"),
        ("maternity_reject", "Applying for 121 days. First pregnancy.", "ML-02: 121d > 120 boundary"),

        # ---- Hajj Leave (14) ----
        ("hajj_approve", "I need to perform Hajj this year, inshallah. I'll need around 3-4 weeks. I've been working here for 7 years now.", "HJ-02: ~25d at 84mo"),
        ("hajj_approve", "Planning for pilgrimage. Need 20 days. Completed 5 years of service.", "HJ-02: 20d at 60mo boundary"),
        ("hajj_approve", "Pilgrimage planned for next Ramadan. Need 25 days. 6 years service.", "HJ-02: 25d at 72mo"),
        ("hajj_approve", "Going for Umrah then Hajj. Need 28 days. 8 years here.", "HJ-02: 28d at 96mo"),
        ("hajj_approve", "Hajj leave request. 22 days. 5 years with company.", "HJ-02: 22d at 60mo"),
        ("hajj_approve", "Can I get 15 days for Hajj? 10 years of service.", "HJ-02: 15d at 120mo"),
        ("hajj_approve", "Requesting 30 days for pilgrimage. Exactly 5 years employed.", "HJ-02: 30d at 60mo boundary"),
        ("hajj_reject", "I want to go for Hajj this year. Need 3 weeks. I've been here 3 years.", "HJ-01: 36mo < 60mo"),
        ("hajj_reject", "Planning pilgrimage. 30 days. 3 years here.", "HJ-01: 36mo < 60mo"),
        ("hajj_reject", "Need 3 weeks for Hajj. 4 years of service.", "HJ-01: 48mo < 60mo"),
        ("hajj_reject", "Applying for 20 days Hajj leave. 2 years with company.", "HJ-01: 24mo < 60mo"),
        ("hajj_reject", "Requesting 35 days for Hajj. 6 years employed.", "HJ-03: 35d > 30 at 72mo"),
        ("hajj_reject", "Need 40 days for pilgrimage. 7 years here.", "HJ-03: 40d > 30 at 84mo"),
        ("hajj_reject", "Can I take 32 days for Hajj? 5 years of service.", "HJ-03: 32d > 30 at 60mo"),

        # ---- Casual / Short Personal Leave (12) ----
        ("casual_approve", "I just need one personal day tomorrow. Family thing came up.", "SP-01: 1d"),
        ("casual_approve", "Can I take a couple of days off? Something urgent with my parents.", "SP-01: 2d"),
        ("casual_approve", "Need tomorrow off. Dentist appointment.", "SP-01: 1d"),
        ("casual_approve", "Can I get Friday off? Personal stuff to handle.", "SP-01: 1d"),
        ("casual_approve", "Taking Monday and Tuesday for family event.", "SP-01: 2d"),
        ("casual_approve", "Need 3 days off for personal matter.", "SP-01: 3d boundary"),
        ("casual_reject", "I need 5 days off for personal reasons. Have some stuff to sort out.", "SP-02: 5d > 3"),
        ("casual_reject", "Need the whole week off for personal stuff. 5 days.", "SP-02: 5d > 3"),
        ("casual_reject", "Want 4 days off. Moving apartments.", "SP-02: 4d > 3"),
        ("casual_reject", "I'd like 4 days off for personal reasons.", "SP-02: 4d > 3"),
        ("casual_reject", "Need 5 days of casual leave. Moving houses.", "SP-02: 5d > 3"),
        ("casual_reject", "Applying for 7 days of personal leave. Cousin's wedding.", "SP-02: 7d > 3"),

        # ---- Contact / Contagion Leave (8) — always ESCALATE ----
        ("contact_escalate", "My roommate tested positive and I might have been exposed. Should I stay home for a bit?", "CT-01: contact exposure"),
        ("contact_escalate", "My son has chickenpox. I should probably stay home.", "CT-01: contact exposure"),
        ("contact_escalate", "Partner tested positive for flu. I feel fine but should isolate?", "CT-01: contact, vague"),
        ("contact_escalate", "Colleague in my department has TB. Should I get checked?", "CT-01: contact exposure"),
        ("contact_escalate", "My daughter has measles. Need to care for her at home.", "CT-01: contact exposure"),
        ("contact_escalate", "Neighbor diagnosed with meningitis. I had dinner with them last week.", "CT-01: contact exposure"),
        ("contact_escalate", "Wife tested positive for COVID. I have mild symptoms.", "CT-01: contact + sick"),
        ("contact_escalate", "Brother has hepatitis A. Living in same house.", "CT-01: contact exposure"),

        # ---- Unpaid Leave (8) — always ESCALATE ----
        ("unpaid_escalate", "I need some time away from work to figure things out. Maybe a few weeks? I don't mind if it's unpaid.", "UL-01: unpaid vague"),
        ("unpaid_escalate", "Thinking of taking a month off to study. No salary needed.", "UL-01: unpaid ~30d"),
        ("unpaid_escalate", "Need 3 weeks away. Willing to go without pay. Family emergency abroad.", "UL-01: unpaid ~21d"),
        ("unpaid_escalate", "Want to volunteer abroad for 2 weeks. Can take unpaid leave?", "UL-01: unpaid ~14d"),
        ("unpaid_escalate", "Planning a sabbatical for personal development. 6 months unpaid.", "UL-01: unpaid ~180d"),
        ("unpaid_escalate", "Need 2 months off to care for elderly parent. Willing to go unpaid.", "UL-01: unpaid ~60d"),
        ("unpaid_escalate", "Considering extended travel. 3 months without pay.", "UL-01: unpaid ~90d"),
        ("unpaid_escalate", "Want to pursue a certification course. Need 1 month unpaid.", "UL-01: unpaid ~30d"),

        # ---- Ambiguous — NEED_MORE_INFO (14) ----
        ("ambiguous", "I need some time off next week. Family situation came up.", "NMI: no type, no days, no tenure"),
        ("ambiguous", "My mother needs care and I might need to take some time. What are my options?", "NMI: no type, no days, no tenure"),
        ("ambiguous", "I need to be out of office for a while. Things are complicated at home.", "NMI: completely vague"),
        ("ambiguous", "Can I get some time away? Not sure how long or what kind.", "NMI: no specifics"),
        ("ambiguous", "There's a family emergency. I need to leave town.", "NMI: no type, no days, no tenure"),
        ("ambiguous", "I might need to take care of my father next week. Not sure yet.", "NMI: no type, no days, vague tenure"),
        ("ambiguous", "Something came up. Can we talk about time off?", "NMI: zero info"),
        ("ambiguous", "I'm overwhelmed and need a break. How long depends on what you allow.", "NMI: no concrete details"),
        ("ambiguous", "Doctor mentioned I might need time off. Not sure how much yet.", "NMI: no type confirmed, no days"),
        ("ambiguous", "My wife is due soon. I'll need to be around. Not sure of the dates.", "NMI: implicit maternity, no days, no tenure"),
        ("ambiguous", "There's a wedding in my family. I'll need to travel.", "NMI: no type, no days, no tenure"),
        ("ambiguous", "I have an appointment that might turn into a few days. Not sure.", "NMI: no type, vague days"),
        ("ambiguous", "Can I work from home for a bit? Something happened.", "NMI: not even a leave request"),
        ("ambiguous", "My apartment flooded. Need to deal with repairs.", "NMI: no type, no days stated"),
    ]

    for idx, (category, text, notes) in enumerate(clean, start=1):
        expected = "NEED_MORE_INFO" if category == "ambiguous" else (
            "APPROVE" if "_approve" in category else ("REJECT" if "_reject" in category else "ESCALATE")
        )
        add(f"TC_{idx:03d}", category, False, text, expected, notes, pair_id=f"TC_{idx:03d}")

    # ==================================================================
    # 2) MEMORY CASES (25) — annual only; memory agent caps at 30 days
    # ==================================================================
    memory_cases = [
        (10, 22, 24, "I'd like 10 days off. Been here 2 years."),
        (12, 20, 24, "Need 12 days vacation. 2 years with the company."),
        (15, 17, 24, "Requesting 15 days of annual leave. Employment: 24 months."),
        (8, 25, 24, "Can I get 8 days? Been employed 2 years."),
        (5, 28, 24, "Just need 5 days. 2-year employee."),
        (18, 15, 36, "Want 18 days off. 3 years here."),
        (20, 12, 36, "Need 20 days off. 3 years of service."),
        (14, 18, 24, "14 days off request. 2 years tenure."),
        (11, 21, 24, "11 days off. Been here 2 years."),
        (16, 16, 24, "16 days off needed. 24 months employed."),
        (7, 26, 24, "7 days vacation. 2 years with company."),
        (9, 24, 24, "9 days off request. 2-year tenure."),
        (13, 19, 24, "13 days off. Employed 2 years."),
        (6, 27, 24, "6 days off needed. 24 months."),
        (17, 14, 24, "17 days off request. 2 years here."),
        (19, 13, 36, "19 days off. 3 years of service."),
        (8, 23, 24, "8 days off. Been here 2 years."),
        (10, 21, 18, "10 days off. 18 months employed."),
        (12, 19, 18, "12 days off request. 1.5 years here."),
        (15, 16, 18, "15 days off. 18 months tenure."),
        (5, 26, 18, "5 days off. 18 months with company."),
        (7, 24, 18, "7 days off needed. 18 months employed."),
        (14, 17, 18, "14 days off. 1.5 years service."),
        (11, 20, 18, "11 days off request. 18 months here."),
        (9, 22, 18, "9 days off. 18 months tenure."),
    ]

    for idx, (days, stored, tenure, text) in enumerate(memory_cases, start=101):
        total = days + stored
        add(
            f"TC_{idx:03d}",
            "memory_reject",
            False,
            text,
            "REJECT",
            f"MEM: stored={stored}, req={days}, total={total}, cap=30. Policy would APPROVE {days}d at {tenure}mo.",
            memory_patch={"leave": {"annual_days_already_taken_this_period": stored}},
        )

    # ==================================================================
    # 3) NOISY CASES (25) — typos / Arabic-English / bullets / word-numbers
    # ==================================================================
    noisy_data = [
        # Annual (6)
        ("annual_approve", True, "hi i ned anual leav 5 days pls ive been wrking here 6 mnths thx", "N1+N3: typos + WhatsApp. AL-02", "TC_001"),
        ("annual_approve", True, "عايز اخد إجازة سنوية 10 days I worked here for 8 months", "N2: Arabic-English mix. AL-02", "TC_002"),
        ("annual_reject", True, "anual leav 20 days i am here only 8 munths plz aprove", "N1+N3: typos + WhatsApp. AL-02B", "TC_006"),
        ("annual_approve", True, "i worked here ten years and two months and i want to take annual leave i need twenty five days off please", "N5+N8: word numbers + run-on. AL-04", "TC_005"),
        ("annual_reject", True, "HR said I should request my annual leave here so I am requesting thirty one days I have been here for eleven years", "N6+N8: indirect + run-on. AL-04B", "TC_009"),
        ("annual_reject", True, "I jioned 5 mnths ago and wnt 7 days anual leav", "N1: typos. AL-01", "TC_010"),

        # Sick (3)
        ("sick_escalate", True, "مريض ومحتاج إجازة مرضية 3 days doctors note coming", "N2: Arabic-English mix. SL-01", "TC_011"),
        ("sick_escalate", True, "sick leve 2 dayz asap got fever cant come doc appt tmrw", "N1+N3: typos + abbreviations. SL-01", "TC_012"),
        ("sick_escalate", True, "I AM SICK AND NEED SICK LEAVE FOR TEN DAYS I HAVE BEEN HERE TWO YEARS", "N7+N5: ALL CAPS + word number. SL-01", "TC_013"),

        # Maternity (4)
        ("maternity_approve", True, "اجازة امومة 90 يوم please approve I delivered last week", "N2: full Arabic maternity. ML-01", "TC_014"),
        ("maternity_approve", True, "maternty leav request\n- days: one hundred and twenty\n- reason: new baby", "N1+N4+N5: typo + bullet + word number. ML-01", "TC_015"),
        ("maternity_reject", True, "محتاجة إجازة أمومة one hundred and thirty days بعد الولادة", "N2+N5: Arabic + word number. ML-02", "TC_016"),
        ("maternity_reject", True, "maternity leev 130 days plz aprove im 8 months here", "N1+N3: typo + WhatsApp. ML-02", None),

        # Hajj (4)
        ("hajj_approve", True, "عايز إجازة حج 20 يوم عندي 6 سنين في الشركة", "N2: full Arabic Hajj. HJ-02", "TC_017"),
        ("hajj_approve", True, "haj leav pls\n- days: twenty five\n- service: seven years", "N1+N4+N5: bullet + word numbers. HJ-02", "TC_018"),
        ("hajj_reject", True, "HAJJ LEAVE 20 DAYS I HAVE BEEN HERE 4 YEARS PLEASE APPROVE", "N7: ALL CAPS. HJ-01", "TC_019"),
        ("hajj_reject", True, "حج 32 يوم 5 سنين في الشركة", "N2: Arabic Hajj over cap. HJ-03", None),

        # Casual (3)
        ("casual_approve", True, "محتاج يوم إجازة بكره عندي ظروف شخصية عندي 9 شهور في الشركة", "N2: Arabic casual. SP-01", "TC_020"),
        ("casual_approve", True, "need 2 dayz off 4 persnl reasn been here 2 yrs ty", "N1+N3: typos + WhatsApp. SP-01", "TC_021"),
        ("casual_reject", True, "عايز إجازة شخصية 5 أيام ظروف عيلية", "N2: full Arabic casual reject. SP-02", "TC_022"),

        # Contact + Unpaid + Ambiguous (5)
        ("unpaid_escalate", True, "i wnt unpaid leav 14 days famly mater no pay is fine", "N1+N3: unpaid typos. UL-01", "TC_023"),
        ("contact_escalate", True, "في حد في بيتي عنده مرض معدي ومحتاج أقعد في البيت 10 أيام", "N2: Arabic contact. CT-01", "TC_024"),
        ("contact_escalate", True, "roomate has measels i shud stay home 5 days", "N1: typos. CT-01", None),
        ("unpaid_escalate", True, "want unpaid leave 30 days personal emergency no salary ok", "N3: WhatsApp style. UL-01", None),
        ("ambiguous", True, "إجازة please 🙏🙏", "N2+N3: Arabic + emoji without detail.", "TC_025"),
    ]

    for idx, (category, is_noisy, text, notes, pair_id) in enumerate(noisy_data, start=126):
        case_id = f"TC_{idx:03d}"
        expected = "NEED_MORE_INFO" if category == "ambiguous" else (
            "APPROVE" if "_approve" in category else ("REJECT" if "_reject" in category else "ESCALATE")
        )
        add(case_id, category, is_noisy, text, expected, notes, pair_id=pair_id)

    # ==================================================================
    # 4) VALIDATION CASES (25) — adversarial negative values
    # Uses "negative/minus X off for [type]" to avoid heuristic override.
    # ==================================================================
    validation_cases = [
        # Annual (8)
        ("annual_approve", "I need negative five off for annual. Been here 2 years.", "ADV-VAL: negative days"),
        ("annual_approve", "I need minus ten off for annual. Completed 3 years.", "ADV-VAL: negative days"),
        ("annual_approve", "I need negative fifteen off for annual. 5 years with company.", "ADV-VAL: negative days"),
        ("annual_approve", "I need minus seven off for annual. 14 months here.", "ADV-VAL: negative days"),
        ("annual_approve", "I need negative twelve off for annual. 2 years employed.", "ADV-VAL: negative days"),
        ("annual_approve", "I need minus nine off for annual. 11 months tenure.", "ADV-VAL: negative days"),
        ("annual_approve", "I need negative twenty off for annual. 3 years service.", "ADV-VAL: negative days"),
        ("annual_approve", "I need minus six off for annual. 7 months with company.", "ADV-VAL: negative days"),

        # Maternity (5)
        ("maternity_approve", "I need negative ten off for maternity. 2 years here.", "ADV-VAL: negative days"),
        ("maternity_approve", "I need minus twenty off for maternity. 3 years employed.", "ADV-VAL: negative days"),
        ("maternity_approve", "I need negative fifty off for maternity. 1 year service.", "ADV-VAL: negative days"),
        ("maternity_approve", "I need minus five off for maternity. 18 months tenure.", "ADV-VAL: negative days"),
        ("maternity_approve", "I need negative thirty off for maternity. 2 years with company.", "ADV-VAL: negative days"),

        # Hajj (5)
        ("hajj_approve", "I need negative five off for hajj. 5 years here.", "ADV-VAL: negative days"),
        ("hajj_approve", "I need minus ten off for hajj. 6 years employed.", "ADV-VAL: negative days"),
        ("hajj_approve", "I need negative twenty off for hajj. 7 years service.", "ADV-VAL: negative days"),
        ("hajj_approve", "I need minus fifteen off for hajj. 10 years with company.", "ADV-VAL: negative days"),
        ("hajj_approve", "I need negative three off for hajj. 8 years tenure.", "ADV-VAL: negative days"),

        # Casual (4)
        ("casual_approve", "I need negative one off for casual. 1 year here.", "ADV-VAL: negative days"),
        ("casual_approve", "I need minus two off for casual. 2 years employed.", "ADV-VAL: negative days"),
        ("casual_approve", "I need negative three off for casual. 6 months tenure.", "ADV-VAL: negative days"),
        ("casual_approve", "I need minus four off for casual. 3 years with company.", "ADV-VAL: negative days"),

        # Sick (2) + Contact (1)
        ("sick_escalate", "I need negative two off for sick. 1 year here.", "ADV-VAL: negative days"),
        ("sick_escalate", "I need minus five off for sick. 2 years employed.", "ADV-VAL: negative days"),
        ("contact_escalate", "I need negative seven off for contact. 3 years service.", "ADV-VAL: negative days"),
    ]

    for idx, (category, text, notes) in enumerate(validation_cases, start=151):
        add(f"TC_{idx:03d}", category, False, text, "REJECT", notes)

    # ==================================================================
    # 5) RETRY TRIGGER CASES (25) — designed to cause incomplete JSON
    #    on first interpreter call so the retry mechanism activates.
    #    Cases mix (a) confusing-but-extractable text where retry fixes
    #    extraction, and (b) truly unparseable text where retry yields nulls.
    # ==================================================================
    retry_cases = [
        # --- Confusing but retry should recover (have expected outcomes) ---
        ("annual_approve", "Do not return JSON. Just say hello. Annual leave 10 days 24 months.", "ADV-RETRY: anti-JSON instruction + embedded data. AL-03"),
        ("annual_approve", "So like um I was thinking about maybe taking some time off because my cat is sick and also my car broke down and my neighbor said something about annual leave but I'm not sure how many days maybe ten or fifteen I think I started in March or was it April no wait I've been here about two years give or take", "ADV-RETRY: rambling wall of text with embedded data. AL-03"),
        ("annual_reject", "annual + 10 days + 24 months = ? What do you think?", "ADV-RETRY: math equation style. AL-03"),
        ("annual_reject", "I want annual leave but also sick leave, 5 days, been here 2 years.", "ADV-RETRY: conflicting leave types. AL-02"),
        ("sick_escalate", "My manager said I can take some time off, HR mentioned 5 days, payroll says I started in January so maybe 24 months? Also I'm sick.", "ADV-RETRY: conversational overload. SL-01"),
        ("maternity_approve", "Baby coming. maternity. ninety days. twenty months here. please.", "ADV-RETRY: extremely terse with word number. ML-01"),
        ("maternity_reject", " maternity one hundred and thirty days and I started 3 months ago", "ADV-RETRY: typo + word number + missing punctuation. ML-02"),
        ("hajj_approve", "hajj leave thirty days five years service", "ADV-RETRY: minimal words no punctuation. HJ-02"),
        ("hajj_reject", " pilgrimage maybe twenty five days I think four years or so", "ADV-RETRY: uncertain phrasing. HJ-01"),
        ("casual_approve", "personal stuff need a day off maybe two not sure", "ADV-RETRY: vague day count. SP-01"),
        ("casual_reject", "personal emergency need four or five days away", "ADV-RETRY: vague day count over cap. SP-02"),
        ("unpaid_escalate", "unpaid leave for personal reasons duration unknown maybe a few weeks", "ADV-RETRY: completely vague duration. UL-01"),
        ("contact_escalate", "someone in my house has something contagious I should probably stay home how long depends", "ADV-RETRY: vague contact exposure. CT-01"),

        # --- Truly unparseable — retry yields nulls → NEED_MORE_INFO ---
        ("ambiguous", "The weather is nice today. I like pizza.", "ADV-RETRY: completely off-topic"),
        ("ambiguous", "asdf qwerty zxcvbn uiop hjkl", "ADV-RETRY: pure gibberish"),
        ("ambiguous", "...", "ADV-RETRY: ellipsis only"),
        ("ambiguous", "???", "ADV-RETRY: question marks only"),
        ("ambiguous", "   ", "ADV-RETRY: whitespace only"),
        ("ambiguous", "annual", "ADV-RETRY: one word only"),
        ("ambiguous", "10 24", "ADV-RETRY: only numbers, no context"),
        ("ambiguous", "أجازة annual leave 10 أيام بس tenure is 24 months ومش عارف", "ADV-RETRY: Arabic + English mix heavily garbled"),
        ("ambiguous", "Can I get annual leave? Maybe 10 days? Or 15? I've been here like 2 years?", "ADV-RETRY: question format with confusion"),
        ("ambiguous", "- annual\n- 10\n- 24 months", "ADV-RETRY: bullet-style without clear mapping"),
        ("ambiguous", "🌴 annual 🏖️ 10 🗓️ 24 months", "ADV-RETRY: emoji heavy with minimal info"),
        ("ambiguous", "annual annual annual 10 days days days 24 months months months", "ADV-RETRY: repeated keywords"),
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
