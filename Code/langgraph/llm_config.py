"""One place to choose the chat model for the LangGraph seminar examples.

Change MODEL_NAME below (or set the LLM_MODEL environment variable) and every example that calls
get_llm() uses the new model. The default is a small model running on your own computer through LM Studio,
so no API key and no per-token charge:

    1. In LM Studio, download "gemma-2-2b-it" (the Q4_K_M file, about 1.7 GB) and start the local server
       (Developer tab, default address http://localhost:1234).
    2. Run `python llm_config.py`. It prints the model ids the server offers.
    3. Put that id in MODEL_NAME (or set LLM_MODEL) and run any example.

Environment variables (all optional):
    LLM_PROVIDER     lmstudio (default) | ollama | groq | gemini
    LLM_MODEL        model id; overrides MODEL_NAME for the chosen provider
    LLM_BASE_URL     server address for lmstudio and ollama
    LLM_TEMPERATURE  number; overrides the default of 0 (1.0 for gemini)
    LLM_API_KEY      only if your local server needs one (LM Studio does not)
    GROQ_API_KEY, GEMINI_API_KEY   only for the groq and gemini providers (never put keys in code)
"""
import json
import os
import urllib.request

PROVIDERS = ("lmstudio", "ollama", "groq", "gemini")
PROVIDER = os.getenv("LLM_PROVIDER", "lmstudio").lower()

# Default model id per provider. For lmstudio and ollama the id must match what the server lists.
DEFAULT_MODELS = {
    "lmstudio": "gemma-2-2b-it",       # run `python llm_config.py` to see the exact id on your machine
    "ollama": "gemma2:2b",
    "groq": "openai/gpt-oss-20b",      # open-weight model served by Groq
    "gemini": "gemini-3.8-flash",      # name taken from the Gemini API docs
}
DEFAULT_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
}

# The single variable the examples share.
MODEL_NAME = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(PROVIDER, "")
BASE_URL = os.getenv("LLM_BASE_URL") or DEFAULT_URLS.get(PROVIDER, "")


def _temperature(provider, override):
    if override is not None:
        return override
    env = os.getenv("LLM_TEMPERATURE")
    if env:
        return float(env)
    return 1.0 if provider == "gemini" else 0.0


def model_for(provider):
    """The model id used for a provider: LLM_MODEL if set, else MODEL_NAME for the active provider, else its default."""
    provider = provider.lower()
    if os.getenv("LLM_MODEL"):
        return os.getenv("LLM_MODEL")
    if provider == PROVIDER:
        return MODEL_NAME
    return DEFAULT_MODELS[provider]


def get_llm(provider=None, temperature=None, model=None):
    """Build the chat model. Arguments are optional; the defaults come from this file and the environment."""
    provider = (provider or PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r} (use one of {', '.join(PROVIDERS)})")
    model = model or model_for(provider)
    temperature = _temperature(provider, temperature)

    if provider in ("lmstudio", "ollama"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=BASE_URL if provider == PROVIDER and BASE_URL else DEFAULT_URLS[provider],
            api_key=os.getenv("LLM_API_KEY", provider),   # the local servers ignore the key
            temperature=temperature,
            max_retries=2,
            timeout=300,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, max_retries=2)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_retries=2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def list_local_models(base_url=None, timeout=3):
    """Model ids the local server offers, or [] if it is not running."""
    url = (base_url or BASE_URL or DEFAULT_URLS["lmstudio"]).rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return [m["id"] for m in json.load(resp).get("data", [])]
    except Exception:
        return []


def fold_system_prompt(messages):
    """Merge a leading system message into the first user message.

    Some small models (for example Gemma 2 in LM Studio's Gemma preset) have no system role. Call this on a
    list of (role, text) tuples before invoking such a model. Other messages pass through unchanged.
    """
    messages = list(messages)
    if messages and messages[0][0] == "system":
        system, rest = messages[0][1], messages[1:]
        for i, (role, text) in enumerate(rest):
            if role == "user":
                rest[i] = ("user", f"{system}\n\n{text}")
                return rest
        return [("user", system)] + rest
    return messages


if __name__ == "__main__":
    print(f"provider   : {PROVIDER}")
    print(f"model name : {MODEL_NAME}")
    if PROVIDER in ("lmstudio", "ollama"):
        print(f"server     : {BASE_URL}")
        found = list_local_models()
        if not found:
            print("server not reachable: start the LM Studio local server (Developer tab) and load a model")
        else:
            print("models the server offers:")
            for name in found:
                print("   ", name, "  <-- selected" if name == MODEL_NAME else "")
            if MODEL_NAME not in found:
                print(f"note: {MODEL_NAME!r} is not in the list; set MODEL_NAME or LLM_MODEL to one of the ids above")
