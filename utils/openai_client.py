import os
from typing import Optional

from langchain_openai import ChatOpenAI


def _apply_env_file(path: str) -> None:
    """Load KEY=value pairs into os.environ without overriding existing vars."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def load_project_env(project_root: Optional[str] = None) -> None:
    """
    Load `.env` from (in order): explicit project_root, package root, current working directory.
    Later files only fill keys not already set (setdefault).
    """
    paths = []
    if project_root:
        paths.append(os.path.join(project_root, ".env"))
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths.append(os.path.join(pkg_root, ".env"))
    cwd_env = os.path.join(os.getcwd(), ".env")
    if cwd_env not in paths:
        paths.append(cwd_env)

    for path in paths:
        if os.path.isfile(path):
            _apply_env_file(path)


# Load .env as soon as this module is imported (before get_llm() runs elsewhere).
load_project_env()


def message_content(response) -> str:
    """Normalize ChatOpenAI response content (string or content blocks)."""
    raw = getattr(response, "content", response)
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(raw or "").strip()


def get_llm():
    """
    Returns a ChatOpenAI client. Requires OPENAI_API_KEY in the environment
    or in a project-root .env file.
    Optional: OPENAI_MODEL (default gpt-4o-mini), OPENAI_BASE_URL.
    """
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to a `.env` file next to the project "
            "(see `.env.example`), or export OPENAI_API_KEY before running."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


def get_json_mode_llm() -> ChatOpenAI:
    """
    ChatOpenAI with OpenAI JSON-object response format.
    Use only for structured extraction (interpreter); prose nodes should use get_llm().
    """
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to `.env` or export it before running."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)
