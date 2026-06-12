"""
ai_config.py

Unified AI provider configuration.
Supports: OpenAI (cloud), Ollama (local), Mock (testing).

Usage:
    from ai_config import get_ai_client
    client = get_ai_client()
    result = client.classify("What type of call is this?")
    result = client.generate("Write a summary.")

How to set up each provider:

1. OPENAI (cloud, recommended for assignment):
   - Get API key from https://platform.openai.com
   - Set OPENAI_API_KEY in .env file
   - Cost: ~$0.50 for 70 calls with gpt-4o-mini

2. OLLAMA (local, free):
   - Download from https://ollama.com
   - Run: ollama pull llama3.2
   - Run: ollama serve (starts on localhost:11434)
   - Set AI_PROVIDER=ollama in .env
   - No API key needed, works offline
   - Slower but zero cost

3. MOCK (testing, no AI):
   - Set AI_PROVIDER=mock in .env
   - Returns placeholder responses
   - Use when you have no internet or no API key
   - Good for testing script logic
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Base Provider Interface
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def classify(self, prompt: str) -> str:
        """Classify text. Return a single word or short phrase."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text. Return a paragraph or two."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
        pass


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

class OpenAIProvider(AIProvider):
    """OpenAI cloud API provider."""

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("ERROR: openai package not installed. Run: pip install openai")
                return None
        return self._client

    def _retry_call(self, fn, retries: int = 3, backoff: float = 2.0):
        """Call fn with exponential backoff on transient errors."""
        import time
        last_err = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Retry on rate limits, timeouts, and server errors
                if any(k in msg for k in ("rate limit", "timeout", "503", "502", "connection", "500")):
                    sleep = backoff * (2 ** attempt)
                    print(f"  Retryable error (attempt {attempt+1}/{retries}): {e}. Sleeping {sleep}s...")
                    time.sleep(sleep)
                else:
                    break
        raise last_err

    def classify(self, prompt: str) -> str:
        client = self._get_client()
        if not client:
            return "unknown"
        try:
            response = self._retry_call(lambda: client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful classifier. Respond with exactly one word or phrase."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            ))
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"OpenAI classify error (final): {e}")
            return "unknown"

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        client = self._get_client()
        if not client:
            return ""
        try:
            response = self._retry_call(lambda: client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful analyst. Be concise and specific."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            ))
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI generate error (final): {e}")
            return ""


# ---------------------------------------------------------------------------
# Ollama Provider (Local)
# ---------------------------------------------------------------------------

class OllamaProvider(AIProvider):
    """
    Ollama local LLM provider.

    Ollama runs as a local HTTP server. You interact with it via HTTP POST,
    just like OpenAI. No Jupyter needed. No notebooks needed.

    Installation:
        1. Download from https://ollama.com
        2. Run: ollama pull llama3.2
        3. Run: ollama serve (keeps running in background)
        4. Test: curl http://localhost:11434/api/tags

    How it works:
        - Ollama listens on localhost:11434
        - Your Python script sends HTTP POST requests
        - Ollama runs the model and returns text
        - Everything stays on your machine
    """

    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _chat(self, messages: list[dict], max_tokens: int = 500) -> str:
        """Send chat request to Ollama."""
        import json
        import urllib.request

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    def classify(self, prompt: str) -> str:
        result = self._chat([
            {"role": "system", "content": "You are a helpful classifier. Respond with exactly one word or phrase."},
            {"role": "user", "content": prompt},
        ], max_tokens=50)
        return result.lower()

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return self._chat([
            {"role": "system", "content": "You are a helpful analyst. Be concise and specific."},
            {"role": "user", "content": prompt},
        ], max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Mock Provider (Testing)
# ---------------------------------------------------------------------------

class MockProvider(AIProvider):
    """
    Mock AI provider for testing without any API.
    Returns sensible placeholder responses.
    """

    def is_available(self) -> bool:
        return True

    def classify(self, prompt: str) -> str:
        # Simple keyword matching for call types
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["support", "ticket", "incident", "outage", "bug", "error"]):
            return "support"
        if any(w in prompt_lower for w in ["renewal", "contract", "pricing", "demo", "sales", "competitive"]):
            return "external"
        if any(w in prompt_lower for w in ["sprint", "planning", "roadmap", "retro", "team", "standup"]):
            return "internal"
        return "external"  # Default for ambiguous

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        # Return a generic response based on prompt keywords
        prompt_lower = prompt.lower()
        if "name" in prompt_lower and "cluster" in prompt_lower:
            if "sso" in prompt_lower or "mfa" in prompt_lower or "identity" in prompt_lower:
                return "Identity and Access Management"
            if "outage" in prompt_lower or "incident" in prompt_lower:
                return "Incident Response and Reliability"
            if "compliance" in prompt_lower or "audit" in prompt_lower or "hipaa" in prompt_lower:
                return "Compliance and Security Audits"
            if "renewal" in prompt_lower or "contract" in prompt_lower or "pricing" in prompt_lower:
                return "Sales and Renewal Discussions"
            return "Product Development and Engineering"
        if "findings" in prompt_lower or "ceo" in prompt_lower:
            return (
                "1. Reliability issues are the top driver of negative sentiment, "
                "affecting support calls most severely.\n"
                "2. Multiple enterprise accounts show churn risk signals including "
                "competitor mentions and escalation requests.\n"
                "3. Customers repeatedly request reporting capabilities and alerting "
                "improvements, indicating product gaps."
            )
        return "Analysis complete. See detailed findings in the report."


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_ai_client() -> AIProvider:
    """
    Get the configured AI provider.

    Priority:
        1. Use AI_PROVIDER env var if set
        2. Fall back to OpenAI if OPENAI_API_KEY exists
        3. Fall back to Ollama if server is running
        4. Use Mock as last resort
    """
    provider_name = os.environ.get("AI_PROVIDER", "auto").lower()

    if provider_name == "openai":
        provider = OpenAIProvider()
        if provider.is_available():
            print("AI: Using OpenAI (cloud)")
            return provider
        print("WARNING: AI_PROVIDER=openai but OPENAI_API_KEY not set")

    elif provider_name == "ollama":
        provider = OllamaProvider()
        if provider.is_available():
            print("AI: Using Ollama (local)")
            return provider
        print("WARNING: AI_PROVIDER=ollama but server not running on " + provider.base_url)
        print("  Run: ollama serve")

    elif provider_name == "mock":
        print("AI: Using Mock provider (no real AI calls)")
        return MockProvider()

    # Auto-detect
    openai_provider = OpenAIProvider()
    if openai_provider.is_available():
        print("AI: Auto-detected OpenAI (cloud)")
        return openai_provider

    ollama_provider = OllamaProvider()
    if ollama_provider.is_available():
        print("AI: Auto-detected Ollama (local)")
        return ollama_provider

    print("AI: No provider available. Using Mock (placeholder responses)")
    print("  To use real AI, set one of:")
    print("    - OPENAI_API_KEY for OpenAI")
    print("    - Run 'ollama serve' for local models")
    return MockProvider()


# ---------------------------------------------------------------------------
# Convenience functions (backward compatible)
# ---------------------------------------------------------------------------

_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = get_ai_client()
    return _CLIENT


def llm_classify(prompt: str) -> str:
    """Classify text using configured AI provider."""
    return _get_client().classify(prompt)


def llm_generate(prompt: str, max_tokens: int = 500) -> str:
    """Generate text using configured AI provider."""
    return _get_client().generate(prompt, max_tokens=max_tokens)
