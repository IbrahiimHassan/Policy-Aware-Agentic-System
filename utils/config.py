import os

# Canonical policy bundle path (relative to cwd or absolute). Override with env POLICY_JSON_PATH.
POLICY_JSON_PATH = os.environ.get("POLICY_JSON_PATH", "policies/policies.json")

# Human-facing wording for prompts and baselines — update when switching business domain.
# Schema field names live in SCHEMA below (domain-specific structured fields).
DOMAIN_PROMPTS = {
    "assistant_role": "Bibliotheca Alexandrina circulation desk assistant",
    "policies_intro": "You are given the BA circulation policy and a library member request.",
    # Used as "Decide … for this {request_subject}." and section headers derived via .title() / upper()
    "request_subject": "circulation request",
    # Fallback text when flattening policies JSON to prose (baseline B)
    "policy_category_fallback": "Policy",
    "policy_rule_fallback": "Apply the Library's stated rules for this policy category.",
}

# Optional bullets appended to the LLM interpreter (after SCHEMA). Empty = SCHEMA only.
INTERPRETER_EXTRACTION_HINTS = [
    "- borrow / check out / take home → request_type 'borrow'",
    "- renew / extend loan → request_type 'renew'",
    "- reserve / hold / waiting list → request_type 'reserve'",
    "- lost book / cannot find → request_type 'report_lost'",
    "- damaged / torn / wet / broken → request_type 'report_damaged'",
    "- researcher / faculty / scholar → user_type 'researcher'",
    "- visitor / tourist / student / journalist → user_type 'non_researcher'",
    "- membership card / library card → has_membership_card (boolean)",
    "- number of books requested → books_requested (integer)",
    "- loan period / for how many days → days_requested (integer)",
    "- regular book / novel / monograph → book_category 'regular'",
    "- encyclopedia / dictionary / atlas → book_category 'reference'",
    "- newspaper / magazine / journal → book_category 'periodical'",
    "- DVD / CD / audio / video → book_category 'audiovisual'",
    "- illustrated / picture book → book_category 'illustrated'",
    "- thesis / dissertation → book_category 'thesis'",
    "- closed stacks / rare storage → book_category 'closed_stacks'",
    "- book title / name of the book → book_title (string)",
    "- outstanding fine / unpaid fee → has_fines (boolean, true if mentioned)",
]

# Numeric thresholds aligned with policies/policies.json (for tests & docs).
# Runtime policy evaluation uses policies.json only; these values mirror it.
POLICY_LIMITS = {
    "max_books_per_borrow": 2,
    "max_loan_days": 14,
    "current_books_cap": 2,
}

SCHEMA = {
    "request_type": {
        "type": "string",
        "default": "unknown",
        "question": "What type of circulation request are you making?",
        "examples": ["borrow", "renew", "reserve", "report_lost", "report_damaged"],
        "description": "The type of circulation request."
    },
    "user_type": {
        "type": "string",
        "default": "unknown",
        "question": "What is your user type?",
        "examples": ["researcher", "non_researcher"],
        "description": "Whether the user is a researcher or not."
    },
    "books_requested": {
        "type": "number",
        "default": 0,
        "question": "How many books are you requesting?",
        "description": "Number of books requested."
    },
    "book_category": {
        "type": "string",
        "default": "unknown",
        "question": "What category is the material?",
        "examples": ["regular", "reference", "periodical", "audiovisual", "illustrated", "thesis", "closed_stacks"],
        "description": "The category of the requested material."
    },
    "has_membership_card": {
        "type": "boolean",
        "default": False,
        "question": "Do you have an individual membership card?",
        "description": "Whether the user has a valid individual membership card."
    },
    "current_books_borrowed": {
        "type": "number",
        "default": 0,
        "question": "How many books do you currently have borrowed?",
        "description": "Number of books currently on loan."
    },
    "days_requested": {
        "type": "number",
        "default": 0,
        "question": "For how many days do you want to borrow the material?",
        "description": "Loan period requested in days."
    },
    "book_title": {
        "type": "string",
        "default": "",
        "question": "What is the title of the book?",
        "description": "Title of the requested book (for memory lookups)."
    },
    "has_fines": {
        "type": "boolean",
        "default": False,
        "question": "Do you have any outstanding fines?",
        "description": "Whether the user has unpaid fines."
    }
}

# Memory agent: only "what we remember" about the user (facts). Not policy text.
# `store` = nested dict of facts. `reject_rules` = how those facts block a request (edit per domain).
# Disable for policy-only benchmarks: env MEMORY_AGENT_ENABLED=0
_MEMORY_ENV = os.environ.get("MEMORY_AGENT_ENABLED", "1").strip().lower()
MEMORY_AGENT_ENABLED_DEFAULT = _MEMORY_ENV not in ("0", "false", "no", "off")

MEMORY_AGENT = {
    "enabled": MEMORY_AGENT_ENABLED_DEFAULT,
    "policy_id": "MEM-RECORD",
    "category": "What we remember about this user",
    # Facts only — e.g. "which books are currently borrowed by others" (no caps/entitlements here).
    "store": {
        "library": {
            # Global defaults (evaluation rows can override via gold `_memory_store_patch`).
            "books_borrowed_by_others": [],
            "current_books_borrowed": 0,
        },
    },
    # Each rule: if it fires, a REJECT is appended after normal policy matching (policies still run first).
    "reject_rules": [
        {
            "type": "case_member_of_fact_list",
            "when": {"request_type": "borrow"},
            "case_field": "book_title",
            "fact_path": ["library", "books_borrowed_by_others"],
            "message": "Our records show that this book is currently borrowed by another member and is not available for borrowing right now.",
        },
        {
            "type": "sum_exceeds_cap",
            "when": {"request_type": "borrow"},
            "request_field": "books_requested",
            "fact_path": ["library", "current_books_borrowed"],
            "cap": 2,
            "kind_label": "book",
            "message": (
                "Our records show you already borrowed {stored} book(s). "
                "You are asking for {requested} more now, which exceeds the total borrowing limit of {cap} books at a time."
            ),
        },
    ],
    # Documentation-only — not read by engine.memory_agent. Use as `_memory_store_patch` shapes in tests.
    "illustrative_store_patches_for_docs": [
        {"library": {"books_borrowed_by_others": []}},  # same as default; no effect
        {"library": {"books_borrowed_by_others": ["The History of Alexandria"]}},  # one unavailable
        {"library": {"books_borrowed_by_others": ["Book A", "Book B"]}},  # multiple unavailable
        {"library": {"current_books_borrowed": 1}},  # room under cap 2
        {"library": {"current_books_borrowed": 2}},  # at cap 2
        {"library": {"current_books_borrowed": 1, "books_borrowed_by_others": ["Ancient Egypt"]}},
    ],
}

# Interpreter/validator constants used across the pipeline.
ALLOWED_FIELDS = list(SCHEMA.keys())
DEFAULT_CASE = {key: meta["default"] for key, meta in SCHEMA.items()}

# Optional alias mapping from extracted keys to schema keys.
FIELD_MAPPING = {
    "type_of_request": "request_type",
    "request": "request_type",
    "number_of_books": "books_requested",
    "book_count": "books_requested",
    "category": "book_category",
    "material_type": "book_category",
    "membership_card": "has_membership_card",
    "has_card": "has_membership_card",
    "card": "has_membership_card",
    "current_books": "current_books_borrowed",
    "books_out": "current_books_borrowed",
    "loan_days": "days_requested",
    "borrowing_days": "days_requested",
    "title": "book_title",
    "book_name": "book_title",
    "outstanding_fines": "has_fines",
    "fines": "has_fines",
}

# Optional lightweight inference rules for obvious values in natural language.
FIELD_INFERENCE = {
    "request_type": {
        "keywords": {
            "borrow": "borrow",
            "check out": "borrow",
            "take home": "borrow",
            "renew": "renew",
            "extend": "renew",
            "reserve": "reserve",
            "hold": "reserve",
            "waiting list": "reserve",
            "lost": "report_lost",
            "cannot find": "report_lost",
            "missing": "report_lost",
            "damaged": "report_damaged",
            "torn": "report_damaged",
            "wet": "report_damaged",
            "broken": "report_damaged",
        },
        "phrase_values": [
            {"pattern": r"\blost\s+book\b", "value": "report_lost"},
            {"pattern": r"\bdamaged\s+book\b", "value": "report_damaged"},
            {"pattern": r"\brenew\s+loan\b", "value": "renew"},
            {"pattern": r"\brenew\s+book\b", "value": "renew"},
            {"pattern": r"\breserve\s+book\b", "value": "reserve"},
        ],
    },
    "user_type": {
        "keywords": {
            "researcher": "researcher",
            "faculty": "researcher",
            "scholar": "researcher",
            "professor": "researcher",
            "visitor": "non_researcher",
            "tourist": "non_researcher",
            "student": "non_researcher",
            "journalist": "non_researcher",
            "general public": "non_researcher",
        },
    },
    "books_requested": {
        "quantity_regex": [
            r"(-?\d+)\s+books?",
            r"borrow\s+(-?\d+)\s+books?",
            r"request(?:ing)?\s+(-?\d+)\s+books?",
            r"(?:need|want|take)\s+(-?\d+)\s+books?",
        ],
        "phrase_values": [
            {"pattern": r"\bone\s+book\b", "value": 1},
            {"pattern": r"\bone\s+more\s+book\b", "value": 1},
            {"pattern": r"\ba\s+book\b", "value": 1},
            {"pattern": r"\btwo\s+books\b", "value": 2},
            {"pattern": r"\btwo\s+more\s+books\b", "value": 2},
            {"pattern": r"\bthree\s+books\b", "value": 3},
            {"pattern": r"\bfour\s+books\b", "value": 4},
            {"pattern": r"\bfive\s+books\b", "value": 5},
        ],
    },
    "book_category": {
        "keywords": {
            "regular": "regular",
            "novel": "regular",
            "monograph": "regular",
            "encyclopedia": "reference",
            "dictionary": "reference",
            "atlas": "reference",
            "reference": "reference",
            "newspaper": "periodical",
            "magazine": "periodical",
            "journal": "periodical",
            "periodical": "periodical",
            "dvd": "audiovisual",
            "cd": "audiovisual",
            "audio": "audiovisual",
            "video": "audiovisual",
            "illustrated": "illustrated",
            "picture": "illustrated",
            "thesis": "thesis",
            "dissertation": "thesis",
            "closed stacks": "closed_stacks",
            "rare": "closed_stacks",
        },
        "phrase_values": [
            {"pattern": r"\bscientific\s+journal\b", "value": "periodical"},
            {"pattern": r"\bphd\s+thesis\b", "value": "thesis"},
            {"pattern": r"\bmaster'?s\s+thesis\b", "value": "thesis"},
        ],
    },
    "has_membership_card": {
        "true_phrases": ["have my card", "have a card", "have my membership card", "have a membership card", "valid card", "active card"],
        "false_phrases": ["no card", "lost my card", "forgot my card", "do not have a card", "never got a card", "expired card", "without a card"],
    },
    "book_title": {
        "regex": [
            r"['\"]([^'\"]+)['\"]",
        ],
    },
    "current_books_borrowed": {
        "quantity_regex": [
            r"already\s+(?:have|borrowed)\s+(\d+)\s+books?",
            r"currently\s+(?:have|borrowed)\s+(\d+)\s+books?",
            r"(\d+)\s+books?\s+(?:out|borrowed|checked\s+out)",
            r"(\d+)\s+books?\s+on\s+loan",
        ],
        "phrase_values": [
            {"pattern": r"\bno\s+books\s+out\b", "value": 0},
            {"pattern": r"\bzero\s+books\b", "value": 0},
            {"pattern": r"\bone\s+book\s+out\b", "value": 1},
            {"pattern": r"\btwo\s+books\s+out\b", "value": 2},
        ],
    },
    "days_requested": {
        "quantity_regex": [
            r"(-?\d+)\s+days?",
            r"for\s+(-?\d+)\s+days?",
            r"(?:period|loan)\s+of\s+(-?\d+)\s+days?",
        ],
        "phrase_values": [
            {"pattern": r"\ba\s+week\b", "value": 7},
            {"pattern": r"\btwo\s+weeks\b", "value": 14},
            {"pattern": r"\bthree\s+weeks\b", "value": 21},
            {"pattern": r"\bone\s+week\b", "value": 7},
            {"pattern": r"\ba\s+ fortnight\b", "value": 14},
        ],
    },
    "has_fines": {
        "true_phrases": ["have a fine", "have fines", "outstanding fine", "unpaid fine", "fine to pay", "overdue fee"],
        "false_phrases": ["no fines", "no outstanding", "paid my fine", "no overdue"],
    },
}