import json
from utils.config import (
    ALLOWED_FIELDS,
    DEFAULT_CASE,
    FIELD_INFERENCE,
    FIELD_MAPPING,
    INTERPRETER_EXTRACTION_HINTS,
    SCHEMA,
)
from utils.openai_client import get_llm, get_json_mode_llm, message_content
from utils.validator import validate_case

llm = get_llm()
# Dedicated client with response_format=json_object (more reliable than bind() on some LC versions).
_json_llm = get_json_mode_llm()


def _format_schema_description():
    lines = []
    for field, meta in SCHEMA.items():
        example_text = ""
        if meta.get("examples"):
            example_text = f" ({', '.join(meta['examples'])})"
        desc = meta.get("description", "")
        tail = f" — {desc}" if desc else ""
        lines.append(f"- {field}: {meta['type']}{example_text}{tail}")
    return "\n".join(lines)


def _format_type_rules_line() -> str:
    parts = []
    for field, meta in SCHEMA.items():
        t = meta.get("type")
        if t == "string" and meta.get("examples"):
            ex = ", ".join(f'"{e}"' for e in meta["examples"])
            parts.append(f'for {field} use a string (allowed examples: {ex})')
        elif t == "number":
            parts.append(f"for {field} use an integer")
        elif t == "string":
            parts.append(f"for {field} use a string")
    return "; ".join(parts) if parts else "Follow the type for each field above."


def _format_optional_extraction_hints() -> str:
    if not INTERPRETER_EXTRACTION_HINTS:
        return ""
    return "\n\nEXTRACTION (only when clearly stated in the user text):\n" + "\n".join(
        INTERPRETER_EXTRACTION_HINTS
    )


def _strip_markdown_json_fence(content: str) -> str:
    """Remove ```json ... ``` wrappers so JSON parsing is reliable."""
    text = content.strip()
    if "```" not in text:
        return text
    lines = text.split("\n")
    out = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            inside = not inside
            continue
        if inside:
            out.append(line)
    cleaned = "\n".join(out).strip()
    return cleaned if cleaned else text


def _coerce_empty_json_keys(data):
    """Ensure every allowed field exists so downstream logic sees explicit nulls."""
    if not isinstance(data, dict):
        return {}
    out = {}
    for key in ALLOWED_FIELDS:
        out[key] = data.get(key)
    return out


def _normalize_null_like_strings(data):
    """Treat JSON/null-ish strings as None so validation does not reject them as wrong types."""
    if not isinstance(data, dict):
        return data
    nullish = {"", "null", "none", "n/a", "na", "unknown"}
    fixed = {}
    for key, val in data.items():
        if isinstance(val, str) and val.strip().lower() in nullish:
            fixed[key] = None
        else:
            fixed[key] = val
    return fixed


def _extract_json_from_text(content):
    content = _strip_markdown_json_fence(content)
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(content[start:end])
    except Exception:
        return None


def _normalize_field_names(data):
    """
    Normalize field names to match the strict schema.
    Maps wrong field names to correct ones.
    """
    normalized = {}
    
    for key, value in data.items():
        # Map to correct field name
        correct_key = FIELD_MAPPING.get(key, key)
        
        # Only include allowed fields
        if correct_key in ALLOWED_FIELDS:
            normalized[correct_key] = value
        # If field not in allowed list but looks like a known field, map it
        elif key in FIELD_MAPPING:
            normalized[correct_key] = value
        # Otherwise skip - don't include unknown fields
    
    return normalized


def _infer_obvious_values(text, data):
    """
    Infer obvious values from natural language that can be reasonably extracted.
    """
    text_lower = text.lower()
    inferred = data.copy()
    import re

    explicit_user_type = None
    user_type_rules = FIELD_INFERENCE.get("user_type", {})
    for keyword, normalized in user_type_rules.get("keywords", {}).items():
        if keyword in text_lower:
            explicit_user_type = normalized
            break

    for field, rules in FIELD_INFERENCE.items():
        if inferred.get(field) is not None:
            continue

        if rules.get("quantity_regex"):
            required_keywords = rules.get("required_keywords", [])
            if all(keyword in text_lower for keyword in required_keywords):
                for regex in rules["quantity_regex"]:
                    match = re.search(regex, text_lower)
                    if match:
                        inferred[field] = int(match.group(1))
                        break
                if inferred.get(field) is None:
                    for phrase in rules.get("single_phrases", []):
                        if phrase in text_lower:
                            inferred[field] = 1
                            break

        if rules.get("true_phrases") or rules.get("false_phrases"):
            if any(phrase in text_lower for phrase in rules.get("false_phrases", [])):
                inferred[field] = False
            elif any(phrase in text_lower for phrase in rules.get("true_phrases", [])):
                inferred[field] = True

        if rules.get("regex") and inferred.get(field) is None:
            for regex in rules["regex"]:
                match = re.search(regex, text_lower)
                if match:
                    inferred[field] = int(match.group(1))
                    break

        if rules.get("phrase_values") and inferred.get(field) is None:
            for pv in rules["phrase_values"]:
                pat = pv.get("pattern")
                val = pv.get("value")
                if pat is None or val is None:
                    continue
                if re.search(pat, text_lower):
                    inferred[field] = val
                    break

        if rules.get("keywords") and inferred.get(field) is None:
            for keyword, normalized in rules["keywords"].items():
                if keyword in text_lower:
                    inferred[field] = normalized
                    break

    # If an explicit role appears in text, trust it over LLM output.
    if explicit_user_type is not None:
        inferred["user_type"] = explicit_user_type
    # Otherwise, if user_type was invented with no explicit role keyword, remove it.
    elif "user_type" in inferred and inferred["user_type"] is not None:
        explicit_role_keywords = user_type_rules.get("keywords", {}).keys()
        if not any(keyword in text_lower for keyword in explicit_role_keywords):
            inferred["user_type"] = None

    return inferred


def heuristic_extract(text):
    """Config-driven deterministic fallback extraction.
    Uses ONLY patterns defined in FIELD_INFERENCE (config.py).
    No domain-specific hardcoded regexes — keeps the pipeline generic.
    """
    out = {}
    t = (text or "").lower()
    import re

    for field, rules in FIELD_INFERENCE.items():
        if field not in ALLOWED_FIELDS:
            continue

        # Quantity regex (e.g. days_requested)
        if rules.get("quantity_regex"):
            required_keywords = rules.get("required_keywords", [])
            if not required_keywords or all(kw in t for kw in required_keywords):
                for rx in rules["quantity_regex"]:
                    m = re.search(rx, t)
                    if m:
                        try:
                            out[field] = int(m.group(1))
                        except Exception:
                            pass
                        break

        # Boolean phrase inference
        if rules.get("true_phrases") or rules.get("false_phrases"):
            if any(phrase in t for phrase in rules.get("false_phrases", [])):
                out[field] = False
            elif any(phrase in t for phrase in rules.get("true_phrases", [])):
                out[field] = True

        # Generic regex fallback
        if rules.get("regex") and field not in out:
            for rx in rules["regex"]:
                m = re.search(rx, t)
                if m:
                    try:
                        out[field] = int(m.group(1))
                    except Exception:
                        pass
                    break

        # Multiplier regex (e.g. years -> months)
        if rules.get("multiplier_regex") and field not in out:
            for item in rules["multiplier_regex"]:
                m = re.search(item["pattern"], t)
                if m:
                    try:
                        out[field] = int(m.group(1)) * item["factor"]
                    except Exception:
                        pass
                    break

        # Phrase-value mappings
        if rules.get("phrase_values") and field not in out:
            for pv in rules.get("phrase_values", []):
                pat = pv.get("pattern")
                val = pv.get("value")
                if pat is None or val is None:
                    continue
                if re.search(pat, t):
                    out[field] = val
                    break

        # Keyword mappings
        if rules.get("keywords") and field not in out:
            for keyword, normalized in rules["keywords"].items():
                if keyword in t:
                    out[field] = normalized
                    break

    return out


def _generate_followup_question(missing_fields, original_text):
    fields_list = "\n".join([f"- {field}" for field in missing_fields])
    prompt = f"""
You are an assistant that generates clear, natural questions to gather missing information.

The user request is:
{original_text}

The system is missing the following fields:
{fields_list}

Generate one clear, simple question in plain English for each missing field.
Make the questions conversational and easy to understand.
Write each question on a new line.
Do not return any JSON or additional commentary.
"""
    response = llm.invoke(prompt)
    return message_content(response)


def _extract_followup_answers(missing_fields, user_answer):
    schema_sample = {key: DEFAULT_CASE[key] for key in missing_fields}
    prompt = f"""
You are a strict JSON extractor.
Extract only the requested fields from the user's answer.
Return ONLY valid JSON and nothing else.
Use this schema as a reference for value types:
{json.dumps(schema_sample, indent=2)}

If a boolean field is present, use true or false.
If a number is present, use an integer.
If the value is ambiguous, omit that field.

User answer:
{user_answer}
"""

    response = _json_llm.invoke(prompt)
    extracted = _extract_json_from_text(message_content(response))
    return extracted or {}


def llm_interpreter(text):
    """
    Converts natural language input into structured data using OpenAI (ChatOpenAI).
    
    STRICT SCHEMA:
    - Only use exact field names from ALLOWED_FIELDS
    - Map wrong field names to correct ones
    - Infer obvious values from natural language
    - Preserve null for truly unknown fields
    """

    keys_literal = json.dumps(ALLOWED_FIELDS)
    prompt = f"""
You extract structured fields from the user's message into JSON.

OUTPUT RULES (critical):
1. Output exactly ONE JSON object.
2. It MUST contain these keys and NO others: {keys_literal}
3. Every key MUST appear. If the user did not state a value for a key, set it to JSON null.
4. Never output {{}}. Never omit keys.
5. {_format_type_rules_line()}; use JSON null when unknown.

INFERENCE RULES:
- If a field is not explicitly stated, infer it from context clues in the message.
- For leave_type: choose the most likely type based on what the user is asking for (vacation/time-off usually means annual, doctor/illness means sick, baby/pregnancy means maternity, etc.). Only use null if the message gives absolutely no clue about the leave type.
- For employment_duration_months: convert years to months (1 year = 12 months) and extract the total. DO NOT calculate remaining service.
- For days_requested: extract ONLY the number of days the user is explicitly asking for right now. DO NOT calculate, subtract, or infer remaining entitlement. Convert common phrases like "a week" to 7 days, "two weeks" to 14 days, "a month" to 30 days.

ALLOWED FIELDS AND TYPES:
{_format_schema_description()}
{_format_optional_extraction_hints()}

User message:
{text}
"""

    retry_prompt = f"""
Return ONE JSON object with exactly these keys: {keys_literal}.
Fill values from the text; use null only if that field is not stated at all.
Do not return {{}}.

Text:
{text}
"""

    def run_once(user_prompt: str):
        response = _json_llm.invoke(user_prompt)
        raw = message_content(response)
        print("\n[RAW LLM OUTPUT]")
        print(raw if raw else repr(response))
        parsed = _extract_json_from_text(raw)
        # Merge pre-extracted deterministic heuristics into parsed JSON when
        # LLM leaves numeric fields empty or non-integer strings. This helps
        # when the LLM returns human-friendly strings like '18 months'.
        # For numeric fields, if heuristic and LLM disagree, prefer heuristic
        # because it is regex-grounded in the text and less prone to hallucination.
        if isinstance(parsed, dict):
            pre = heuristic_extract(text)
            def _to_int_like(val):
                try:
                    return int(val)
                except Exception:
                    return None

            for k, v in pre.items():
                cur = parsed.get(k)
                if cur is None:
                    parsed[k] = v
                else:
                    if isinstance(DEFAULT_CASE.get(k), int):
                        cur_int = _to_int_like(cur)
                        v_int = _to_int_like(v)
                        if v_int is not None and (cur_int is None or cur_int != v_int):
                            parsed[k] = v
        return parsed if isinstance(parsed, dict) else {}

    data = run_once(prompt)
    # Retry if model returned empty object or missing keys
    if not data or set(data.keys()) != set(ALLOWED_FIELDS):
        print("[WARN] Retrying interpreter once (incomplete JSON keys).")
        data = run_once(retry_prompt)

    data = _coerce_empty_json_keys(data)
    data = _normalize_null_like_strings(data)

    # -----------------------------
    # NORMALIZE FIELD NAMES
    # Map wrong names to correct ones
    # -----------------------------
    data = _normalize_field_names(data)
    data = _normalize_null_like_strings(data)

    # -----------------------------
    # INFER OBVIOUS VALUES
    # Extract values that can be reasonably inferred
    # -----------------------------
    data = _infer_obvious_values(text, data)

    # -----------------------------
    # FALLBACK HEURISTIC EXTRACTION
    # If critical numeric or categorical fields remain None after LLM inference,
    # attempt a deterministic regex/keyword-based extraction and merge results.
    # This prevents the pipeline returning NEED_MORE_INFO for many straightforward cases.
    # -----------------------------
    fallback = heuristic_extract(text)

    # if LLM produced non-integer strings for numeric fields (e.g. '13 months'),
    # overwrite them with deterministic numeric fallbacks when available.
    def _to_int_like(val):
        try:
            return int(val)
        except Exception:
            return None

    for k, v in fallback.items():
        cur = data.get(k)
        if cur is None:
            data[k] = v
            continue
        # override non-int-like values for numeric fields
        if isinstance(DEFAULT_CASE.get(k), int):
            if _to_int_like(cur) is None and isinstance(v, int):
                data[k] = v
        # for strings, keep LLM output unless it's null-ish; fallback can win if LLM left empty
        else:
            if cur in (None, "", "null"):
                data[k] = v

    # -----------------------------
    # VALIDATION STEP
    # -----------------------------
    validated_data, validation_errors = validate_case(data, DEFAULT_CASE)

    return {
        "case_data": validated_data,
        "validation_errors": validation_errors,
        "needs_clarification": False,
        "questions": "",
        "missing_fields": []
    }
