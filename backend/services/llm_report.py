import os
from openai import OpenAI

_client = None

# Default to Ollama (local, free Llama). Set LLM_BASE_URL for OpenAI or other compatible APIs.
OLLAMA_BASE = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"

def _get_client():
    global _client
    if _client is None:
        base_url = os.environ.get("LLM_BASE_URL", OLLAMA_BASE)
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client

def generate_report(body_part, fracture, bone_age):
    prompt = f"""
    Generate a radiology-style summary:
    Body part: {body_part}
    Fracture: {fracture}
    Bone age: {bone_age}
    """.strip()

    try:
        client = _get_client()
        model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"{prompt}\n\nError: {e}"
