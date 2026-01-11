import json
import urllib.request
import urllib.error
from typing import Any, Dict
from .base import BaseProvider, ProviderError
from ..config_manager import get_api_key


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_api_key("gemini")

    def available(self) -> bool:
        return bool(self.api_key)
    def generate(self, prompt: str, timeout: int = 30) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError("missing_key", "Gemini API key not found")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                try:
                    return json.loads(body)
                except Exception:
                    return {"text": body}
        except urllib.error.HTTPError as e:
            if e.code == 401 or e.code == 403:
                raise ProviderError("invalid_key", f"Authentication failed: {e.code}", transient=False)
            if e.code >= 500:
                raise ProviderError("server_error", f"Provider error: {e.code}", transient=True)
            try:
                body = e.read().decode("utf-8")
                return {"text": body}
            except Exception:
                raise ProviderError("network_error", str(e), transient=False)
        except urllib.error.URLError as e:
            raise ProviderError("network_error", str(e.reason), transient=True)
            return {"text": resp.text}


    def get_ai_response(prompt: str, timeout: int = 30) -> Dict[str, Any]:
        """Minimal reusable wrapper: call GeminiProvider.generate and return a
        small, normalized dictionary suitable for callers.

        Returns a dict with keys: text, tokens_used (optional), provider_type,
        provider_name, raw, and error (optional).
        """
        prov = GeminiProvider()
        try:
            raw = prov.generate(prompt, timeout=timeout)
        except ProviderError as e:
            return {
                "text": "",
                "tokens_used": None,
                "provider_type": "cloud",
                "provider_name": "gemini",
                "raw": None,
                "error": {"code": e.code, "message": e.message, "transient": bool(getattr(e, "transient", False))},
            }

        # lightweight normalization (keep minimal and consistent with manager)
        text = None
        tokens = None
        if isinstance(raw, dict):
            text = raw.get("text") or raw.get("content") or raw.get("answer")
            if not text:
                candidates = raw.get("candidates") or raw.get("choices")
                if candidates and isinstance(candidates, list) and len(candidates) > 0:
                    first = candidates[0]
                    if isinstance(first, dict):
                        text = first.get("content") or first.get("text") or first.get("message")
            usage = raw.get("usage") or raw.get("token_usage")
            if isinstance(usage, dict):
                tokens = usage.get("total_tokens") or usage.get("prompt_tokens")
        else:
            text = str(raw)

        return {
            "text": text or "",
            "tokens_used": tokens,
            "provider_type": "cloud",
            "provider_name": "gemini",
            "raw": raw,
        }
