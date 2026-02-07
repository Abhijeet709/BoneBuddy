import os
from openai import OpenAI

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"{prompt}\n\nError: {e}"
