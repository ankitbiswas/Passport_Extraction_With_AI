"""Send OCR text + page images to GPT-4o and extract structured data."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

_client: OpenAI | None = None
USER_PROMPT_PATH = Path(__file__).parent / "user_prompt.txt"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env file")
        _client = OpenAI(api_key=api_key)
    return _client


def _load_user_prompt() -> str:
    if not USER_PROMPT_PATH.exists():
        raise FileNotFoundError(f"User prompt file not found: {USER_PROMPT_PATH}")
    return USER_PROMPT_PATH.read_text(encoding="utf-8")


def _build_messages(
    ocr_texts: list[str],
    page_images_b64: list[str],
) -> list[dict[str, Any]]:
    user_prompt = _load_user_prompt()

    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_prompt},
    ]

    for i, (text, img_b64) in enumerate(zip(ocr_texts, page_images_b64)):
        content.append({
            "type": "text",
            "text": f"--- Page {i + 1} OCR text ---\n{text}",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    return content


def extract(
    ocr_texts: list[str],
    page_images_b64: list[str],
    schema: type[BaseModel],
) -> BaseModel:
    """Send OCR text + page images to GPT-4o, return a populated Pydantic model.

    Args:
        ocr_texts: OCR text for each page.
        page_images_b64: Base64-encoded PNG for each page (same order as ocr_texts).
        schema: Pydantic model class defining the fields to extract.

    Returns:
        An instance of *schema* populated with extracted values.
    """
    client = _get_client()
    messages = _build_messages(ocr_texts, page_images_b64)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise document data extraction assistant. Extract only the requested fields. Return valid JSON matching the schema exactly. If a field is not found, use an empty string."},
            {"role": "user", "content": messages},
        ],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        },
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise RuntimeError("GPT-4o returned empty response")

    data = json.loads(raw)
    return schema.model_validate(data)
