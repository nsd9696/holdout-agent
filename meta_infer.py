# meta_infer.py
# Functions that call OpenAI to infer topic and difficulty and to create embeddings.

import json
from openai import OpenAI
from config import OPENAI_TAG_MODEL, OPENAI_EMBED_MODEL

client = OpenAI()

TAG_SYS_PROMPT = (
    "You are a concise annotation assistant. Given a short benchmark example, "
    "produce JSON exactly in this form:\n"
    '{"topic":"<one_or_two_word_keyword>","difficulty":<integer_1_to_5>}'
    "\nChoose a single short topic and an integer from 1 (very easy) to 5 (very hard)."
)

def infer_topic_and_difficulty(text: str, max_chars: int = 2500) -> dict:
    """
    Infer topic and difficulty using an LLM. Returns {'topic':str, 'difficulty':int}.
    Use deterministic settings (temperature=0).
    """
    snippet = text[:max_chars]
    resp = client.responses.create(
        model=OPENAI_TAG_MODEL,
        input=[
            {"role": "system", "content": TAG_SYS_PROMPT},
            {"role": "user", "content": f"Example:\n{snippet}"}
        ],
        temperature=0.0
    )
    out = resp.output_text.strip()
    try:
        parsed = json.loads(out)
        topic = parsed.get("topic", "unknown").lower().strip().replace(" ", "_")
        difficulty = int(parsed.get("difficulty", 3))
        difficulty = max(1, min(5, difficulty))
        return {"topic": topic, "difficulty": difficulty}
    except Exception:
        # fallback
        return {"topic": "unknown", "difficulty": 3}

def embed_text(text: str):
    """Return embedding vector using the configured embeddings model."""
    emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return emb.data[0].embedding
