import os

# Canonical policy bundle path (relative to cwd or absolute). Override with env POLICY_JSON_PATH.
POLICY_JSON_PATH = os.environ.get("POLICY_JSON_PATH", "policies/policies.json")

# Human-facing wording for prompts and baselines — update when switching business domain.
# Schema field names live in SCHEMA below (domain-specific structured fields).
DOMAIN_PROMPTS = {
    "assistant_role": "Egypt private-sector leave decision assistant (Law 14 of 2025 summaries)",
    "policies_intro": "You are given company leave policies (Egypt) and an employee leave request.",
    # Used as "Decide … for this {request_subject}." and section headers derived via .title() / upper()
    "request_subject": "leave request",
    # Fallback text when flattening policies JSON to prose (baseline B)
    "policy_category_fallback": "Policy",
    "policy_rule_fallback": "Apply the company's stated rules for this policy category.",
}

# Optional bullets appended to the LLM interpreter (after SCHEMA). Empty = SCHEMA only.
INTERPRETER_EXTRACTION_HINTS = [
    "- annual / annual leave / vacation → leave_type \"annual\"",
    "- casual / personal day(s) / special short leave → leave_type \"casual\"",
    "- sick / medical leave → leave_type \"sick\"",
    "- maternity / parental (mother) → leave_type \"maternity\"",
    "- Hajj / pilgrimage leave → leave_type \"hajj\"",
    "- unpaid leave / leave without pay → leave_type \"unpaid\"",
    "- contact with contagious disease / quarantine-style leave → leave_type \"contact\"",
    "- months employed / tenure → employment_duration_months (integer)",
    "- number of days requested → days_requested (integer)",
]

# Numeric thresholds aligned with policies/policies.json (for tests & docs).
# Runtime policy evaluation uses policies.json only; these values mirror it.
POLICY_LIMITS = {
    "months_below_which_annual_rejected": 6,
    "months_pro_rata_band_upper_exclusive": 12,
    "months_long_tenure_minimum": 12,
    "months_ten_year_tier": 120,
    "annual_days_pro_rata_cap": 15,
    "annual_days_after_one_year": 21,
    "annual_days_above_which_rejected_when_long_tenure": 30,
    "casual_max_approved_days": 3,
    "casual_days_above_which_rejected": 4,
}

SCHEMA = {
    "leave_type": {
        "type": "string",
        "default": "unknown",
        "question": "What type of leave are you requesting?",
        "examples": ["annual", "casual", "sick", "maternity", "hajj", "unpaid", "contact"],
        "description": "The category of leave being requested."
    },
    "employment_duration_months": {
        "type": "number",
        "default": 0,
        "question": "How many months have you worked with your current employer?",
        "description": "Duration of employment with the current employer in months."
    },
    "days_requested": {
        "type": "number",
        "default": 0,
        "question": "How many leave days are you requesting?",
        "description": "Number of leave days requested."
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
    # Facts only — e.g. "they already took N casual days this period" (no caps/entitlements here).
    "store": {
        "leave": {
            # Global defaults (evaluation rows can override via gold `_memory_store_patch`).
            "casual_days_already_taken_this_period": 0,
            "sick_days_already_taken_this_period": 0,
            "annual_days_already_taken_this_period": 0,
        },
    },
    # Each rule: if it fires, a REJECT is appended after normal policy matching (policies still run first).
    "reject_rules": [
        {
            "type": "sum_exceeds_cap",
            "when": {"leave_type": "casual"},
            "request_field": "days_requested",
            "fact_path": ["leave", "casual_days_already_taken_this_period"],
            "cap": 3,
            "kind_label": "casual",
            "message": (
                "Our records show you already took {stored} short personal / casual day(s) this period. "
                "You are asking for {requested} more now, which we cannot add under the informal cap at this time."
            ),
        },
        {
            "type": "sum_exceeds_cap",
            "when": {"leave_type": "sick"},
            "request_field": "days_requested",
            "fact_path": ["leave", "sick_days_already_taken_this_period"],
            "cap": 7,
            "kind_label": "sick",
            "message": (
                "Our records show you already took {stored} sick day(s) in the period we track. "
                "You are asking for {requested} more now, which we cannot add at this time."
            ),
        },
        {
            "type": "sum_exceeds_cap",
            "when": {"leave_type": "annual"},
            "request_field": "days_requested",
            "fact_path": ["leave", "annual_days_already_taken_this_period"],
            "cap": 30,
            "kind_label": "annual",
            "message": (
                "Our records show you already took {stored} annual day(s) in the period we track. "
                "You are asking for {requested} more now, which we cannot add at this time."
            ),
        },
        # Example: block when a case field value appears in a remembered list (needs SCHEMA field).
        # {
        #     "type": "case_member_of_fact_list",
        #     "when": {},
        #     "case_field": "book_id",
        #     "fact_path": ["library", "book_ids_not_available_now"],
        #     "message": "Our records show that copy is not available to borrow right now.",
        # },
    ],
    # Documentation-only — not read by engine.memory_agent. Use as `_memory_store_patch` shapes in tests.
    "illustrative_store_patches_for_docs": [
        {"leave": {"casual_days_already_taken_this_period": 0}},  # same as default; no effect
        {"leave": {"casual_days_already_taken_this_period": 1}},  # e.g. room under cap 3
        {"leave": {"casual_days_already_taken_this_period": 2}},  # near cap 3
        {"leave": {"sick_days_already_taken_this_period": 6}},  # illustrative sick tally
        {"leave": {"annual_days_already_taken_this_period": 20}},  # room under cap 30
        {"leave": {"casual_days_already_taken_this_period": 1, "sick_days_already_taken_this_period": 3}},
    ],
}

# Interpreter/validator constants used across the pipeline.
ALLOWED_FIELDS = list(SCHEMA.keys())
DEFAULT_CASE = {key: meta["default"] for key, meta in SCHEMA.items()}

# Optional alias mapping from extracted keys to schema keys.
FIELD_MAPPING = {
    "type_of_leave": "leave_type",
    "leave_days": "days_requested",
    "requested_days": "days_requested",
    "employment_months": "employment_duration_months",
}

# Optional lightweight inference rules for obvious values in natural language.
FIELD_INFERENCE = {
    "days_requested": {
        "quantity_regex": [
            r"(\d+)\s+days?",
            r"\b(\d+)\s+(?:[\w'-]+\s+){0,8}days?\b",
            r"for\s+(\d+)\s+days?",
            r"request(?:ing)?\s+(\d+)\s+days?",
            r"(?:need|want|take)\s+(\d+)\s+(?:days?\b|day\b)",
        ],
        "phrase_values": [
            {"pattern": r"\ba\s+week\b", "value": 7},
            {"pattern": r"\btwo\s+weeks\b", "value": 14},
            {"pattern": r"\bthree\s+weeks\b", "value": 21},
            {"pattern": r"\bfour\s+weeks\b", "value": 28},
            {"pattern": r"\ba\s+month\b", "value": 30},
            {"pattern": r"\bfull\s+month\b", "value": 30},
            {"pattern": r"\bone\s+month\b", "value": 30},
        ],
    },
    "employment_duration_months": {
        "regex": [
            r"(?:employment|tenure)\s+(\d+)\s+months?",
            r"(?:employment|tenure)\s*[:=]\s*(\d+)\s+months?",
            r"(\d+)\s+months?\s*(?:,\s*)?(?:employment|tenure|service)",
            r"(\d+)\s+months?\s+(?:of\s+)?(?:service|employment|work)",
            r"(?:worked|working)\s+for\s+(\d+)\s+months?",
            r"(?:been here|here for)\s+(\d+)\s+months?",
            r"(?:for|only)\s+(\d+)\s+months?",
        ],
        "multiplier_regex": [
            {"pattern": r"(?:been|worked|completed|here|joined)\s+(?:for\s+)?(\d+)\s*years?", "factor": 12},
            {"pattern": r"(\d+)\s*years?\s*(?:with|in|at|of)?\s+(?:the\s+)?(?:company|firm|organization|service|employment|work|tenure|job)?", "factor": 12},
            {"pattern": r"(\d+)\s*years?\s+(?:of\s+)?(?:service|employment|work|tenure)", "factor": 12},
            {"pattern": r"(\d+)-year\s+(?:employee|tenure|service)", "factor": 12},
        ],
    },
    "leave_type": {
        "keywords": {
            "annual": "annual",
            "vacation": "annual",
            "sick": "sick",
            "casual": "casual",
            "personal day": "casual",
            "maternity": "maternity",
            "hajj": "hajj",
            "unpaid": "unpaid",
            "contact": "contact",
        },
        "phrase_values": [
            {"pattern": r"\bsick\s+leave\b", "value": "sick"},
            {"pattern": r"\bmedical\s+leave\b", "value": "sick"},
            {"pattern": r"\bmaternity\b", "value": "maternity"},
            {"pattern": r"\bhajj\b", "value": "hajj"},
            {"pattern": r"\bunpaid\s+leave\b", "value": "unpaid"},
            {"pattern": r"\bleave\s+balance\b", "value": "annual"},
            {"pattern": r"\balready\s+taken\b", "value": "annual"},
            {"pattern": r"\bthis\s+year\b", "value": "annual"},
            {"pattern": r"\bthis\s+period\b", "value": "annual"},
            {"pattern": r"\btime\s+off\b", "value": "annual"},
            {"pattern": r"\bdays\s+off\b", "value": "annual"},
            {"pattern": r"\boff\s+work\b", "value": "annual"},
            {"pattern": r"\bdays\s+requested\b", "value": "annual"},
            {"pattern": r"\bdays\s+needed\b", "value": "annual"},
            {"pattern": r"\bused\s+\d+", "value": "annual"},
            {"pattern": r"\btaken\s+\d+", "value": "annual"},
            {"pattern": r"\d+\s+days\s+used\b", "value": "annual"},
            {"pattern": r"\d+\s+days\s+taken\b", "value": "annual"},
            {"pattern": r"\balready\s+used\b", "value": "annual"},
        ],
    },
}