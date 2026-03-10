import re
import os
import requests


SYSTEM_PROMPT = (
    "You are a PromQL expert. Given a monitoring request and context, "
    "return only the PromQL query with no explanation."
)


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:promql)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


class OllamaBackend:
    def __init__(self, model: str = "promql-model", url: str = "http://localhost:11434"):
        self.model = model
        self.url   = url.rstrip("/")

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        return clean_response(resp.json()["message"]["content"])

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model   = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def generate(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        return clean_response(resp.choices[0].message.content)

    def is_available(self) -> bool:
        return bool(self.api_key)


def get_backend(backend: str, model: str = None, api_key: str = None):
    if backend == "ollama":
        return OllamaBackend(model=model or "promql-model")
    elif backend == "openai":
        return OpenAIBackend(model=model or "gpt-4o-mini", api_key=api_key)
    else:
        raise ValueError(f"unknown backend: {backend}. use 'ollama' or 'openai'")